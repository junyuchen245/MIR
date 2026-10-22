"""Train an LDDMM (time-varying velocity field) registration model on LUMIR25.

The backbone (TransMorph, VFA, VoxelMorph, LessNet, Fourier-Net+, SITReg, ...) and the LDDMM flow
are trained jointly: the image-similarity loss is evaluated on the moving image warped by the
integrated flow, and the regularizer is the LDDMM kinetic energy returned by the model
(``<m_0, K m_0>`` for geodesic shooting, the path energy ``sum_t dt <p_t, K p_t>`` for the
relaxation / tvf parameterization).  See ``backbones.py`` for how each backbone is built.

Examples
--------
Dedicated TransMorph-LDDMM, geodesic shooting (default), encoder/decoder initialized from the
LUMIR24 TransMorph-TVF weights::

    python train_TM_LDDMM.py --backbone transmorph_lddmm --param shooting --time_steps 7 --reg_weight 0.01

Any other MIR backbone through the model-agnostic wrapper::

    python train_TM_LDDMM.py --backbone vfa --param shooting
    python train_TM_LDDMM.py --backbone sitreg --param shooting --init_from ''
    python train_TM_LDDMM.py --backbone voxelmorph --param tvf --time_steps 5

Quick calibration run (300 iterations, 8 validation pairs)::

    python train_TM_LDDMM.py --iters_per_epoch 300 --val_subset 8 --max_epoch 1 --tag calib
"""
from torch.utils.tensorboard import SummaryWriter
import os, glob, sys, json, random, argparse, time
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from natsort import natsorted
from MIR.models import SpatialTransformer
from MIR.models.lddmm import jacobian_fd, det3x3
from MIR.image_similarity import NCC_vxm
from MIR.deformation_regularizer import Grad3d, logBeta
from MIR.utils import Logger, AverageMeter, mk_grid_img
from MIR.accuracy_measures import dice_val_VOI
from MIR import RandomMultiContrastRemap
from backbones import build_model, init_backbone, BACKBONES

label_lookup = [0.,  4., 11., 23., 30., 31., 32., 35., 36., 37., 38., 39., 40., 41.,
            44.,  45.,  47.,  48.,  49.,  50.,  51.,  52.,  55.,  56.,  57.,  58.,  59., 60.,
            61.,  62.,  71.,  72.,  73.,  75.,  76., 100., 101., 102., 103., 104., 105., 106.,
            107., 108., 109., 112., 113., 114., 115., 116., 117., 118., 119., 120., 121., 122.,
            123., 124., 125., 128., 129., 132., 133., 134., 135., 136., 137., 138., 139., 140.,
            141., 142., 143., 144., 145., 146., 147., 148., 149., 150., 151., 152., 153., 154.,
            155., 156., 157., 160., 161., 162., 163., 164., 165., 166., 167., 168., 169., 170.,
            171., 172., 173., 174., 175., 176., 177., 178., 179., 180., 181., 182., 183., 184.,
            185., 186., 187., 190., 191., 192., 193., 194., 195., 196., 197., 198., 199., 200.,
            201., 202., 203., 204., 205., 206., 207.,]


class L2RLUMIRJSONDataset(Dataset):
    def __init__(self, base_dir, json_path, stage='train', subset=None):
        with open(json_path) as f:
            d = json.load(f)
        self.imgs = d['training'] if stage.lower() == 'train' else d['validation']
        if subset is not None:
            self.imgs = self.imgs[:subset]
        self.base_dir = base_dir
        self.stage = stage

    def label_reassign(self, label_map):
        label = label_map.copy()
        ref = label_map.copy()
        for i in range(len(label_lookup)):
            label[ref == label_lookup[i]] = i
        return label

    def to_torch(self, x):
        return torch.from_numpy(np.ascontiguousarray(x[None, ...]))

    def __getitem__(self, index):
        if self.stage == 'train':
            mov_dict = self.imgs[index]
            fix_dict = random.choice([d for d in self.imgs if d is not mov_dict])
            x = nib.load(self.base_dir + mov_dict['image'])
            y = nib.load(self.base_dir + fix_dict['image'])
        else:
            img_dict = self.imgs[index]
            mov_path, fix_path = img_dict['moving'], img_dict['fixed']
            x = nib.load(self.base_dir + mov_path)
            y = nib.load(self.base_dir + fix_path)
        x = self.to_torch(x.get_fdata() / 255.)
        y = self.to_torch(y.get_fdata() / 255.)
        if self.stage == 'validation':
            x_lbl = nib.load(self.base_dir + 'labelsVal/' + mov_path.split('/')[-1])
            y_lbl = nib.load(self.base_dir + 'labelsVal/' + fix_path.split('/')[-1])
            x_lbl = self.to_torch(self.label_reassign(x_lbl.get_fdata()))
            y_lbl = self.to_torch(self.label_reassign(y_lbl.get_fdata()))
            return x.float(), y.float(), x_lbl.float(), y_lbl.float()
        return x.float(), y.float()

    def __len__(self):
        return len(self.imgs)


def parse_args(defaults=None):
    """Command-line arguments; ``defaults`` (dict) overrides the built-in defaults (used by the sibling tutorials)."""
    p = argparse.ArgumentParser(description='Train an LDDMM registration model on LUMIR25')
    p.add_argument('--train_dir', default='/scratch2/jchen/DATA/LUMIR_L2R24_Internal/')
    p.add_argument('--val_dir', default='/scratch2/jchen/DATA/LUMIR/LUMIR25/')
    p.add_argument('--json', default='LUMIR25_dataset_wo_landmarks.json', help='dataset json inside val_dir')
    # model
    p.add_argument('--backbone', default='transmorph_lddmm', choices=BACKBONES,
                   help="'transmorph_lddmm' = dedicated TransMorphLDDMM; others go through LDDMMWrapper")
    p.add_argument('--scale_factor', type=int, default=0,
                   help='image down-sampling before the network (0 = default of the backbone tutorial)')
    p.add_argument('--param', default='shooting', choices=['shooting', 'tvf'])
    p.add_argument('--time_steps', type=int, default=7)
    p.add_argument('--kernel', default='helmholtz', choices=['helmholtz', 'gaussian'])
    p.add_argument('--alpha', type=float, default=2.0, help='Helmholtz: L=(-alpha*Lap+gamma)^power')
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--power', type=float, default=2.0)
    p.add_argument('--sigma', type=float, nargs='+', default=[2.0], help='Gaussian kernel std(s) in voxels')
    p.add_argument('--int_scale', type=int, default=1, help='integrate the flow on a int_scale x finer grid (shooting only)')
    # spatially varying (region-specific) kernel
    p.add_argument('--sigmas', type=float, nargs='*', default=[],
                   help='Gaussian stds (voxels) of a spatially varying multi-Gaussian kernel, e.g. 1 2 4 8 '
                        '(empty = stationary kernel defined by --kernel)')
    p.add_argument('--weight_smooth_sigma', type=float, default=1.0, help='smoothing of the kernel-weight maps (voxels)')
    p.add_argument('--omt_weight', type=float, default=0.1, help='weight of the optimal-mass-transport penalty on the kernel weights')
    # SPR (spatially varying regularization, TransMorph-SPR style) integrated into LDDMM
    p.add_argument('--spr', default='none', choices=['none', 'kernel', 'energy'],
                   help="'kernel': the SPR weight map sets the per-voxel kernel scale (needs --sigmas); "
                        "'energy': the SPR map weights the kinetic-energy integrand (stationary kernel)")
    p.add_argument('--spr_source', default='auto', choices=['auto', 'head', 'backbone'],
                   help="where the SPR map comes from: a model-agnostic SPRHead or the backbone itself "
                        "(transmorph_spr / vfa_spr); 'auto' = backbone for SPR backbones, head otherwise")
    p.add_argument('--beta_weight', type=float, default=0.1,
                   help='Beta-prior weight on the SPR map, i.e. logBeta(lam, 1 + beta_weight) as in the SPR tutorials')
    p.add_argument('--init_from', default='lumir24',
                   help="'lumir24' = LUMIR24 weights of the backbone (downloaded; TransMorph/VFA only), "
                        "a checkpoint path, or '' for random init")
    # loss
    p.add_argument('--sim_weight', type=float, default=1.0)
    p.add_argument('--reg_weight', type=float, default=0.01, help='weight of the LDDMM kinetic energy')
    p.add_argument('--grad_weight', type=float, default=0.0, help='optional extra diffusion penalty on the final flow')
    # optimisation
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--max_epoch', type=int, default=500)
    p.add_argument('--iters_per_epoch', type=int, default=0, help='cap training iterations per epoch (0 = full epoch)')
    p.add_argument('--val_subset', type=int, default=0, help='use only the first N validation pairs (0 = all)')
    p.add_argument('--cont_training', action='store_true')
    p.add_argument('--no_aug', action='store_true', help='disable random multi-contrast augmentation')
    p.add_argument('--tag', default='', help='extra tag appended to the experiment name')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pretrained_dir', default='pretrained_wts/', help='where downloaded LUMIR24 weights are cached')
    if defaults:
        p.set_defaults(**defaults)
    return p.parse_args()


def downsample(img, scale_factor):
    return img if scale_factor == 1 else F.avg_pool3d(img, scale_factor)


def flow_to_full_res(flow, size):
    """Resample a displacement field (any grid) to the full-resolution grid, rescaling its units."""
    if tuple(flow.shape[2:]) == tuple(size):
        return flow
    factor = size[0] / flow.shape[2]
    return F.interpolate(flow, size=size, mode='trilinear', align_corners=False) * factor


_transformers = {}


def transformer_for(shape, device):
    """Cached bilinear SpatialTransformer for an arbitrary grid."""
    key = tuple(shape)
    if key not in _transformers:
        _transformers[key] = SpatialTransformer(key).to(device)
    return _transformers[key]


def main(defaults=None):
    args = parse_args(defaults)
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    batch_size = 1
    H, W, D = 160, 224, 192
    if args.spr != 'none':
        kernel_desc = (f'spr-{args.spr}_beta_{args.beta_weight:g}'
                       + (f'_s{"-".join(f"{s:g}" for s in args.sigmas)}' if args.sigmas else f'_{args.kernel}'))
    elif args.sigmas:
        kernel_desc = f'svk_s{"-".join(f"{s:g}" for s in args.sigmas)}_omt_{args.omt_weight:g}'
    elif args.kernel == 'helmholtz':
        kernel_desc = f'{args.kernel}_a{args.alpha:g}_g{args.gamma:g}_p{args.power:g}'
    else:
        kernel_desc = f'{args.kernel}_s{"-".join(f"{s:g}" for s in args.sigma)}'
    save_dir = (f'LDDMM_{args.backbone}_{args.param}_T{args.time_steps}_{kernel_desc}_int{args.int_scale}'
                f'_ncc_{args.sim_weight:g}_reg_{args.reg_weight:g}' + (f'_grad_{args.grad_weight:g}' if args.grad_weight else '')
                + (f'_{args.tag}' if args.tag else '') + '/')
    os.makedirs('experiments/' + save_dir, exist_ok=True)
    os.makedirs('logs/' + save_dir, exist_ok=True)
    sys.stdout = Logger('logs/' + save_dir)
    print(vars(args))

    '''
    Initialize model
    '''
    model, scale_factor = build_model(args, H, W, D)
    print(f'Backbone: {args.backbone} at 1/{scale_factor} resolution, param={args.param}, T={args.time_steps}')
    if args.spr != 'none':
        print(f'SPR integrated into LDDMM: mode={args.spr}, source={getattr(model, "spr_source", "decoder head")}, Beta prior weight {args.beta_weight}')
    if args.sigmas:
        print(f'Spatially varying kernel, sigmas={args.sigmas}; FWHM (smoothest, sharpest) = {model.kernel.fwhm()} voxels')
    else:
        print(f'Kernel impulse response FWHM: {model.kernel.fwhm():.1f} voxels (integration grid)')
    epoch_start = 0
    model_dir = 'experiments/' + save_dir
    if args.cont_training and len(os.listdir(model_dir)) > 0:
        ckpt_name = natsorted(os.listdir(model_dir))[-1]
        ckpt = torch.load(model_dir + ckpt_name, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        epoch_start = ckpt['epoch']
        print(f'Model: {ckpt_name} loaded, resuming from epoch {epoch_start}')
    else:
        init_backbone(model, args, pretrained_dir=args.pretrained_dir)
    model.cuda()
    print(f'Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M')

    '''
    Spatial transformation functions (full resolution)
    '''
    reg_model = SpatialTransformer((H, W, D), 'nearest').cuda()
    spatial_trans = SpatialTransformer((H, W, D)).cuda()

    random_mri = RandomMultiContrastRemap(p=0.0 if args.no_aug else 0.85)

    '''
    Data
    '''
    train_set = L2RLUMIRJSONDataset(base_dir=args.train_dir, json_path=args.val_dir + args.json, stage='train')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_set = L2RLUMIRJSONDataset(base_dir=args.val_dir, json_path=args.val_dir + args.json, stage='validation',
                                  subset=args.val_subset or None)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)
    if args.cont_training and epoch_start > 0:
        optimizer.load_state_dict(ckpt['optimizer'])
        print('Optimizer loaded!')

    best_dsc = 0
    criterion_ncc = NCC_vxm()
    criterion_grad = Grad3d(penalty='l2')
    criterion_beta = logBeta()
    writer = SummaryWriter(log_dir='logs/' + save_dir)

    def register(mov_net, fix_net):
        """Run the model and return the full-resolution flow plus the model output dict."""
        out = model((mov_net, fix_net), return_dict=True)
        return flow_to_full_res(out['flow'], (H, W, D)), out

    def train_step(x, y, x_net, y_net):
        flow, out = register(x_net, y_net)
        output = spatial_trans(x, flow)
        loss_sim = criterion_ncc(output, y) * args.sim_weight
        loss_reg = out['energy'] * args.reg_weight
        loss = loss_sim + loss_reg
        if args.grad_weight > 0:
            loss = loss + criterion_grad(flow, y) * args.grad_weight
        loss_omt = out['omt'] * args.omt_weight if 'omt' in out else torch.zeros(())
        if 'spr_weight' in out:  # Beta prior on the SPR map, same form as the TransMorph-SPR tutorials
            loss_omt = criterion_beta(out['spr_weight'], 1.0 + args.beta_weight)
        loss = loss + loss_omt
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        energies = torch.stack(out['energies'])
        drift = (energies[-1] / energies[0].clamp_min(1e-12) - 1).item()
        return loss.item(), loss_sim.item(), loss_reg.item(), drift, flow, loss_omt.item()

    for epoch in range(epoch_start, args.max_epoch):
        print('Training Starts')
        loss_all = AverageMeter()
        sim_all = AverageMeter()
        reg_all = AverageMeter()
        t_start = time.time()
        n_iters = len(train_loader) if args.iters_per_epoch <= 0 else min(args.iters_per_epoch, len(train_loader))
        for idx, data in enumerate(train_loader, 1):
            if idx > n_iters:
                break
            model.train()
            adjust_learning_rate(optimizer, epoch, args.max_epoch, args.lr)
            with torch.no_grad():
                x = data[0].cuda()
                y = data[1].cuda()
                x_aug = random_mri(x, x > 0)
                y_aug = random_mri(y, y > 0)
                x_net = downsample(x_aug, scale_factor)
                y_net = downsample(y_aug, scale_factor)
            if x_net.max() < 0.2 or y_net.max() < 0.2:
                continue
            loss, loss_sim, loss_reg, drift, flow, loss_omt = train_step(x, y, x_net, y_net)
            loss_all.update(loss, y.numel()); sim_all.update(loss_sim, y.numel()); reg_all.update(loss_reg, y.numel())
            loss, loss_sim, loss_reg, drift2, flow, loss_omt = train_step(y, x, y_net, x_net)
            loss_all.update(loss, y.numel()); sim_all.update(loss_sim, y.numel()); reg_all.update(loss_reg, y.numel())
            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}, Prior: {:.4f}, |u|max: {:.2f}, E drift: {}, {:.2f}s/it'.format(
                idx, n_iters, loss, loss_sim, loss_reg, loss_omt, flow.abs().max().item(), drift_str(drift2, args),
                (time.time() - t_start) / idx))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        writer.add_scalar('Loss/train_sim', sim_all.avg, epoch)
        writer.add_scalar('Loss/train_reg', reg_all.avg, epoch)
        print('Epoch {} loss {:.4f} (sim {:.4f}, reg {:.4f})'.format(epoch, loss_all.avg, sim_all.avg, reg_all.avg))

        '''
        Validation
        '''
        eval_dsc = AverageMeter()
        eval_negj = AverageMeter()
        eval_inv = AverageMeter()
        eval_drift = AverageMeter()
        eval_sigma = AverageMeter()
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                data = [t.cuda() for t in data]
                x, y, x_seg, y_seg = data
                x_net = downsample(x, scale_factor)
                y_net = downsample(y, scale_factor)
                flow, out = register(x_net, y_net)
                def_out = reg_model(x_seg.float(), flow)
                dsc = dice_val_VOI(def_out.long(), y_seg.long(), num_clus=len(label_lookup))
                eval_dsc.update(dsc.item(), x.size(0))
                # diagnostics on the integration grid: folding, inverse consistency, energy conservation
                u, w = out['flow'], out['inv_flow']
                st = transformer_for(u.shape[2:], u.device)
                det = det3x3(jacobian_fd(u))
                eval_negj.update((det <= 0).float().mean().item() * 100, x.size(0))
                inv_res = (u + st(w, u)).norm(dim=1)
                eval_inv.update(inv_res.mean().item(), x.size(0))
                energies = torch.stack(out['energies'])
                eval_drift.update((energies[-1] / energies[0].clamp_min(1e-12) - 1).item() * 100, x.size(0))
                sigma_map = None
                if 'kernel_weights' in out:
                    sigma_map = model.kernel.effective_sigma(out['kernel_weights'])
                    eval_sigma.update(sigma_map.mean().item(), x.size(0))
                elif 'spr_weight' in out:
                    sigma_map = out['spr_weight']
                    eval_sigma.update(sigma_map.mean().item(), x.size(0))
                print('DSC {:.4f} | negJ {:.3f}% | inv-res {:.3f} vox | E drift {}{}'.format(
                    eval_dsc.avg, eval_negj.avg, eval_inv.avg, drift_str(eval_drift.avg / 100, args),
                    spr_str(eval_sigma.avg, out)))
            grid_img = mk_grid_img(8, 1, tuple(u.shape[2:]), dim=0).cuda()
            def_grid = st(grid_img.float(), u)
        print('Epoch {} validation: DSC {:.4f} | negJ {:.3f}% | inv-res {:.3f} vox | E drift {}{}'.format(
            epoch, eval_dsc.avg, eval_negj.avg, eval_inv.avg, drift_str(eval_drift.avg / 100, args),
            spr_str(eval_sigma.avg, out)))
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
            'scale_factor': scale_factor,
        }, save_dir='experiments/' + save_dir, filename='dsc{:.4f}.pth.tar'.format(eval_dsc.avg))
        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        writer.add_scalar('NegJ_percent/validate', eval_negj.avg, epoch)
        writer.add_scalar('InvResidual/validate', eval_inv.avg, epoch)
        writer.add_scalar('EnergyDrift_percent/validate', eval_drift.avg, epoch)
        figs = [('Grid', def_grid), ('moving', x), ('fixed', y), ('deformed', spatial_trans(x, flow))]
        if sigma_map is not None:
            writer.add_scalar('EffectiveSigma/validate', eval_sigma.avg, epoch)
            figs.append(('kernel_sigma', sigma_map))
        for name, img in figs:
            fig = comput_fig(img)
            writer.add_figure(name, fig, epoch)
            plt.close(fig)
        loss_all.reset()
    writer.close()


def spr_str(value, out):
    """Validation summary of the spatially varying regularization, if any."""
    if 'kernel_weights' in out:
        return f' | eff. sigma {value:.2f} vox' + (f' | mean lam {out["spr_weight"].mean().item():.3f}' if 'spr_weight' in out else '')
    if 'spr_weight' in out:
        return f' | mean lam {value:.3f}'
    return ''


def drift_str(drift, args):
    """Kinetic-energy drift along the geodesic; only meaningful for geodesic shooting."""
    return '{:+.1%}'.format(drift) if args.param == 'shooting' else 'n/a (tvf)'


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0]
    img = img[img.shape[0] * 3 // 10: img.shape[0] * 3 // 10 + 16]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=2):
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


if __name__ == '__main__':
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        print('     GPU #' + str(GPU_idx) + ': ' + torch.cuda.get_device_name(GPU_idx))
    main()
