#!/usr/bin/env python3.8
"""Train the small 2D hypernetwork VFA models behind the browser registration playground (https://junyuchen.me/playground/).

  --model spr    HyperVFASPR: spatially varying regularization, 2 hyperparameters in [0,1]
                 (h0 -> Beta-prior strength alpha, h1 -> gradient-regularizer weight)
  --model plain  HyperVFA: spatially invariant regularization, 1 hyperparameter in [0,1]
                 (h0 -> gradient-regularizer weight)

Data: data/{train,val}.npz from extract_slices.py (IXI axial slices, 192x160). Pairs are inter-subject
slices at the same axial index. Built on MIR (https://github.com/junyuchen245/MIR).
"""
import argparse, os, time, json, math, numpy as np, torch
from MIR.models import HyperVFASPR, HyperVFA, SpatialTransformer
import MIR.models.configs_VFA as cfgs
from MIR.image_similarity import NCC_vxm
from MIR.deformation_regularizer import logBeta, LocalGrad2d

# Hyperparameter mapping (slider codes in [0,1] -> loss weights), stored in each checkpoint as `hyper_map`.
#   reg   : weight of the (lambda-weighted) L2 gradient penalty, log-spaced in REG_RANGE (MSE units)
#   kappa : Beta-prior strength relative to reg, log-spaced in KAPPA_RANGE; alpha = 1 + kappa * reg.
#           Per pixel the optimum of  reg*lam*g - (alpha-1)*log(lam)  is lam* = kappa / g with g = |grad u|^2,
#           so kappa directly sets how much local deformation energy it takes before lambda drops below 1.
REG_RANGE   = (0.0005, 0.2)
KAPPA_RANGE = (0.03, 3.0)
HYPER_MAP = {'reg_range': REG_RANGE, 'kappa_range': KAPPA_RANGE, 'scale': 'log', 'alpha': '1 + kappa * reg',
             'codes': ['prior strength kappa (SPR only)', 'regularization weight']}

def _logmap(c, lo, hi):
    return lo * (hi / lo) ** c

def map_hyper(code):
    """code: [B, k] in [0,1] -> (alpha [B,1] or None, reg_wt [B,1])."""
    rcode = code[:, 1:2] if code.shape[1] == 2 else code[:, :1]
    reg = _logmap(rcode, *HYPER_MAP['reg_range'])
    alpha = None
    if code.shape[1] == 2:
        kappa = _logmap(code[:, :1], *HYPER_MAP['kappa_range'])
        alpha = 1.0 + kappa * reg
    return alpha, reg

def build(model_kind, img_size=(192, 160), start=8, maxch=32, hyp=16, beta0=0.1, device='cuda'):
    c = cfgs.get_VFA_default_config()
    c.img_size = tuple(img_size); c.start_channels = start; c.matching_channels = start
    c.max_channels = maxch; c.nb_hyp_units = hyp; c.initialize = beta0
    M = HyperVFASPR if model_kind == 'spr' else HyperVFA
    model = M(c, device=device).to(device)
    init_hypernetwork(model)
    return model, c

def init_hypernetwork(model):
    """Variance-preserving init for the hyperparameter MLP.

    PyTorch's default Linear init shrinks activations by ~1/sqrt(3) per ReLU layer, so a narrow 6-layer
    embedding collapses towards zero (dead units, kernels ~0, no gradient). Kaiming init keeps the embedding
    O(1) and sensitive to the hyperparameter input. Weights loaded from a checkpoint override this.
    """
    for fc in model.hyper_model.fcs:
        torch.nn.init.kaiming_normal_(fc.weight, nonlinearity='relu')
        torch.nn.init.constant_(fc.bias, 0.1)   # non-zero so an all-zero hyperparameter code still yields a live embedding
    # Rescale every kernel generator so that, at the reference code 0.5, generated conv kernels have He-normal
    # scale (std = sqrt(2 / fan_in)). Without this the kernels come out ~5x too large and VFA's attention softmax
    # saturates, which stalls learning. The bias generators start at zero.
    from MIR.models.HyperVFA import HyperConv
    nh = model.hyper_model.fcs[0].in_features
    with torch.no_grad():
        hf = model.hyper_model(torch.full((1, nh), 0.5, device=next(model.parameters()).device))
        for mod in model.modules():
            if isinstance(mod, HyperConv):
                target = math.sqrt(2.0 / (mod.in_channels * mod.ks ** mod.ndims))
                std = mod.linear_conv(hf).std().item()
                if std > 0:
                    mod.linear_conv.weight.mul_(target / std)
                mod.linear_bias.weight.zero_()

def load_split(path, device):
    d = np.load(path)
    img = torch.from_numpy(d['img']).float()
    img = img / img.flatten(1).max(1)[0].clamp_min(1e-6)[:, None, None]   # per-slice max -> [0,1]
    return img.unsqueeze(1).to(device), torch.from_numpy(d['seg']).long().unsqueeze(1).to(device), d['slice']

def jac_det_2d(flow):
    """flow [B,2,H,W] (voxel displacement, channel 0 along H). Returns |J| [B,H-1,W-1]."""
    dy = flow[:, :, 1:, :-1] - flow[:, :, :-1, :-1]
    dx = flow[:, :, :-1, 1:] - flow[:, :, :-1, :-1]
    j11 = dy[:, 0] + 1; j12 = dx[:, 0]
    j21 = dy[:, 1];     j22 = dx[:, 1] + 1
    return j11 * j22 - j12 * j21

def dice(seg_a, seg_b, labels):
    out = []
    for l in labels:
        a = seg_a == l; b = seg_b == l
        den = a.sum() + b.sum()
        if den > 0:
            out.append((2.0 * (a & b).sum().float() / den.float()).item())
    return float(np.mean(out)) if out else float('nan')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['spr', 'plain'], required=True)
    ap.add_argument('--iters', type=int, default=40000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--val_every', type=int, default=1000)
    ap.add_argument('--out', default=None)
    ap.add_argument('--data_dir', default='data', help='folder with train.npz / val.npz from extract_slices.py')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--hyp_units', type=int, default=16)
    ap.add_argument('--beta0', type=float, default=1.0, help='VFA initial displacement scale (config.initialize)')
    ap.add_argument('--ncc_win', type=int, default=15, help='local NCC window (only with --sim ncc)')
    ap.add_argument('--sim', choices=['mse', 'ncc'], default='mse', help='image similarity; MSE trains reliably on these intensity-normalised 2D slices, local NCC does not')
    ap.add_argument('--reg_min', type=float, default=None); ap.add_argument('--reg_max', type=float, default=None)
    ap.add_argument('--kappa_min', type=float, default=None); ap.add_argument('--kappa_max', type=float, default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = 'cuda'
    out = args.out or f'experiments/{args.model}'
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, 'train_log.txt'), 'a')
    def P(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

    tr_img, tr_seg, tr_slice = load_split(os.path.join(args.data_dir, 'train.npz'), dev)
    va_img, va_seg, va_slice = load_split(os.path.join(args.data_dir, 'val.npz'), dev)
    H, W = tr_img.shape[2:]
    groups = {s: np.where(tr_slice == s)[0] for s in np.unique(tr_slice)}
    model, cfg = build(args.model, (H, W), hyp=args.hyp_units, beta0=args.beta0, device=dev)
    nparams = sum(p.numel() for p in model.parameters())
    P(f'model={args.model} params={nparams/1e6:.2f}M img={H}x{W} cfg={dict(cfg)}')
    stn = SpatialTransformer((H, W)).to(dev)
    stn_nn = SpatialTransformer((H, W), mode='nearest').to(dev)
    if args.reg_min is not None: HYPER_MAP['reg_range'] = (args.reg_min, HYPER_MAP['reg_range'][1])
    if args.reg_max is not None: HYPER_MAP['reg_range'] = (HYPER_MAP['reg_range'][0], args.reg_max)
    if args.kappa_min is not None: HYPER_MAP['kappa_range'] = (args.kappa_min, HYPER_MAP['kappa_range'][1])
    if args.kappa_max is not None: HYPER_MAP['kappa_range'] = (HYPER_MAP['kappa_range'][0], args.kappa_max)
    P(f'similarity={args.sim} hyper_map={HYPER_MAP}')
    if args.sim == 'ncc':
        ncc = NCC_vxm(win=[args.ncc_win, args.ncc_win], ndims=2)
    else:
        ncc = lambda x, y: torch.nn.functional.mse_loss(x, y)
    prior = logBeta(); lgrad = LocalGrad2d(penalty='l2')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    nh = 2 if args.model == 'spr' else 1

    # fixed validation pairs (same slice index, different subjects) and hyper settings
    rng = np.random.RandomState(1)
    vgroups = {s: np.where(va_slice == s)[0] for s in np.unique(va_slice)}
    vpairs = []
    for s in sorted(vgroups):
        idx = vgroups[s]
        for _ in range(6):
            a, b = rng.choice(idx, 2, replace=False); vpairs.append((a, b))
    vcodes = [0.0, 0.5, 1.0] if nh == 1 else [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)]
    best = -1; t0 = time.time()

    def step(mov, fix, code):
        alpha, reg = map_hyper(code)
        if args.model == 'spr':
            flow, wts = model((mov, fix), code)
        else:
            flow = model((mov, fix), code); wts = torch.ones_like(flow[:, :1])
        warped = stn(mov, flow)
        l_sim = ncc(warped, fix)
        l_grad = (lgrad(flow, wts) * reg.view(-1)).mean() if reg.shape[0] > 1 else lgrad(flow, wts) * reg.view(())
        l_prior = prior(wts, alpha.mean()) if alpha is not None else torch.zeros((), device=dev)
        return l_sim + l_grad + l_prior, (l_sim.item(), l_grad.item(), l_prior.item())

    for it in range(1, args.iters + 1):
        model.train()
        lr = args.lr * (1 - (it - 1) / args.iters) ** 0.9
        for g in opt.param_groups: g['lr'] = lr
        s = np.random.choice(list(groups)); idx = np.random.choice(groups[s], 2 * args.batch, replace=False)
        a = tr_img[idx[:args.batch]]; b = tr_img[idx[args.batch:]]
        # hypernetwork: one random code per batch (shared across the batch, like HyperMorph)
        code = torch.rand(1, nh, device=dev)
        loss, parts = step(a, b, code)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        loss2, parts2 = step(b, a, code)
        opt.zero_grad(set_to_none=True); loss2.backward(); opt.step()
        if it % 100 == 0:
            P(f'it {it} lr {lr:.2e} loss {loss.item():.4f} sim {parts[0]:.4f} grad {parts[1]:.4f} prior {parts[2]:.4f} code {code.cpu().numpy().round(2).tolist()} {time.time()-t0:.0f}s')
        if it % args.val_every == 0 or it == args.iters:
            model.eval(); res = {}
            with torch.no_grad():
                for cval in vcodes:
                    code = torch.tensor([list(cval) if isinstance(cval, tuple) else [cval]], device=dev, dtype=torch.float32)
                    ds, fold, lam, mag = [], [], [], []
                    for (ia, ib) in vpairs:
                        mov, fix = va_img[ia:ia+1], va_img[ib:ib+1]
                        if args.model == 'spr':
                            flow, wts = model((mov, fix), code); lam.append(wts.mean().item())
                        else:
                            flow = model((mov, fix), code)
                        wseg = stn_nn(va_seg[ia:ia+1].float(), flow).long()
                        labels = torch.unique(va_seg[ib:ib+1]); labels = labels[labels > 0]
                        ds.append(dice(wseg, va_seg[ib:ib+1], labels.tolist()))
                        fold.append((jac_det_2d(flow) <= 0).float().mean().item() * 100); mag.append(flow.abs().mean().item())
                    res[cval] = (float(np.mean(ds)), float(np.mean(fold)), float(np.mean(lam)) if lam else None, float(np.mean(mag)))
            msg = ' | '.join(f'h={c}: dice {d:.4f} fold {f:.3f}% |u| {u:.2f}' + (f' lam {l:.3f}' if l is not None else '') for c, (d, f, l, u) in res.items())
            P(f'VAL it {it}: {msg}')
            score = res[(0.5, 0.5) if nh == 2 else 0.5][0]
            torch.save({'state_dict': model.state_dict(), 'cfg': dict(cfg), 'args': vars(args), 'it': it, 'val': res,
                        'hyper_map': dict(HYPER_MAP)}, os.path.join(out, 'last.pth'))
            if score > best:
                best = score
                torch.save({'state_dict': model.state_dict(), 'cfg': dict(cfg), 'args': vars(args), 'it': it, 'val': res,
                            'hyper_map': dict(HYPER_MAP)}, os.path.join(out, 'best.pth'))
                P(f'  saved best (dice {best:.4f})')
    P('done')

if __name__ == '__main__':
    main()
