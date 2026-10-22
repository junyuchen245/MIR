import argparse
import glob
import json
import os
import random
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
from natsort import natsorted
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from MIR import RandomMultiContrastRemap
from MIR.accuracy_measures import dice_val_VOI
from MIR.deformation_regularizer import Grad3d
from MIR.image_similarity import NCC_vxm
from MIR.models import FourierNetPlus, SpatialTransformer
import MIR.models.configs_FourierNet as configs_FourierNet
from MIR.utils import AverageMeter, Logger, mk_grid_img


TRAIN_DIR = '/scratch2/jchen/DATA/LUMIR_L2R24_Internal/'
VAL_DIR = '/scratch2/jchen/DATA/LUMIR/LUMIR25/'
VAL_JSON = VAL_DIR + 'LUMIR25_dataset_wo_landmarks.json'

LABEL_LOOKUP = [
    0., 4., 11., 23., 30., 31., 32., 35., 36., 37., 38., 39., 40., 41.,
    44., 45., 47., 48., 49., 50., 51., 52., 55., 56., 57., 58., 59., 60.,
    61., 62., 71., 72., 73., 75., 76., 100., 101., 102., 103., 104., 105., 106.,
    107., 108., 109., 112., 113., 114., 115., 116., 117., 118., 119., 120., 121., 122.,
    123., 124., 125., 128., 129., 132., 133., 134., 135., 136., 137., 138., 139., 140.,
    141., 142., 143., 144., 145., 146., 147., 148., 149., 150., 151., 152., 153., 154.,
    155., 156., 157., 160., 161., 162., 163., 164., 165., 166., 167., 168., 169., 170.,
    171., 172., 173., 174., 175., 176., 177., 178., 179., 180., 181., 182., 183., 184.,
    185., 186., 187., 190., 191., 192., 193., 194., 195., 196., 197., 198., 199., 200.,
    201., 202., 203., 204., 205., 206., 207.,
]


class L2RLUMIRJSONDataset(Dataset):
    def __init__(self, base_dir, json_path, stage='train'):
        with open(json_path) as f:
            data = json.load(f)
        self.imgs = data['training'] if stage == 'train' else data['validation']
        self.base_dir = base_dir
        self.stage = stage

    def label_reassign(self, label_map):
        label = label_map.copy()
        ref = label_map.copy()
        for idx, value in enumerate(LABEL_LOOKUP):
            label[ref == value] = idx
        return label

    @staticmethod
    def to_torch(array):
        array = np.ascontiguousarray(array[None])
        return torch.from_numpy(array).float()

    def __getitem__(self, index):
        if self.stage == 'train':
            mov_dict = self.imgs[index]
            fix_dicts = [item for item in self.imgs if item != mov_dict]
            random.shuffle(fix_dicts)
            fix_dict = fix_dicts[0]
            x = nib.load(self.base_dir + mov_dict['image']).get_fdata() / 255.
            y = nib.load(self.base_dir + fix_dict['image']).get_fdata() / 255.
            return self.to_torch(x), self.to_torch(y)

        img_dict = self.imgs[index]
        mov_path = img_dict['moving']
        fix_path = img_dict['fixed']
        x = nib.load(self.base_dir + mov_path).get_fdata() / 255.
        y = nib.load(self.base_dir + fix_path).get_fdata() / 255.
        x_lbl = self.label_reassign(
            nib.load(self.base_dir + 'labelsVal/' + mov_path.split('/')[-1]).get_fdata()
        )
        y_lbl = self.label_reassign(
            nib.load(self.base_dir + 'labelsVal/' + fix_path.split('/')[-1]).get_fdata()
        )
        return self.to_torch(x), self.to_torch(y), self.to_torch(x_lbl), self.to_torch(y_lbl)

    def __len__(self):
        return len(self.imgs)


def parse_args():
    parser = argparse.ArgumentParser(description='Train FourierNet+ on the LUMIR25 tutorial split.')
    parser.add_argument('--aug', action='store_true', help='Use RandomMultiContrastRemap augmentation.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epoch-start', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--ncc-weight', type=float, default=1.0)
    parser.add_argument('--diffusion-weight', type=float, default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def adjust_learning_rate(optimizer, epoch, max_epochs, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - epoch / max_epochs, power), 8)


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=2):
    torch.save(state, os.path.join(save_dir, filename))
    model_lists = natsorted(glob.glob(os.path.join(save_dir, '*')))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(os.path.join(save_dir, '*')))


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def main():
    args = parse_args()
    set_seed(args.seed)

    h, w, d = 160, 224, 192
    config = configs_FourierNet.get_FourierNetPlus_default_config()
    config.img_size = (h, w, d)
    config.low_img_size = (h // 2, w // 2, d // 2)
    lr = args.lr if args.lr is not None else float(config.lr)
    diffusion_weight = args.diffusion_weight
    if diffusion_weight is None:
        diffusion_weight = float(getattr(config, 'smth_labda', 5.0))

    aug_tag = 'w_aug' if args.aug else 'wo_aug'
    save_name = 'FourierNetPlus_{}_LUMIR25_ncc_{}_diffusion_{}'.format(
        aug_tag, args.ncc_weight, diffusion_weight
    )
    save_dir = os.path.join('experiments', save_name)
    log_dir = os.path.join('logs', save_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Logger(log_dir + '/')

    print('Currently using:', torch.cuda.get_device_name(0))
    print(config)
    print(args)

    model = FourierNetPlus(config).cuda()
    spatial_trans = SpatialTransformer((h, w, d)).cuda()
    spatial_trans_nn = SpatialTransformer((h, w, d), 'nearest').cuda()
    random_mri = RandomMultiContrastRemap(p=0.85)

    updated_lr = lr
    if args.resume:
        checkpoints = natsorted(glob.glob(os.path.join(save_dir, '*.pth.tar')))
        if checkpoints:
            checkpoint = torch.load(checkpoints[-1])
            model.load_state_dict(checkpoint['state_dict'])
            args.epoch_start = checkpoint.get('epoch', args.epoch_start)
            updated_lr = round(lr * np.power(1 - args.epoch_start / args.epochs, 0.9), 8)
            print('Model loaded:', checkpoints[-1])

    train_set = L2RLUMIRJSONDataset(TRAIN_DIR, VAL_JSON, stage='train')
    val_set = L2RLUMIRJSONDataset(VAL_DIR, VAL_JSON, stage='validation')
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    if args.resume:
        checkpoints = natsorted(glob.glob(os.path.join(save_dir, '*.pth.tar')))
        if checkpoints:
            optimizer.load_state_dict(torch.load(checkpoints[-1])['optimizer'])
            print('Optimizer loaded.')

    criterion_ncc = NCC_vxm()
    criterion_reg = Grad3d(penalty='l2')
    writer = SummaryWriter(log_dir=log_dir)
    best_dsc = 0

    for epoch in range(args.epoch_start, args.epochs):
        print('Training Starts')
        loss_all = AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, args.epochs, lr)
            with torch.no_grad():
                x, y = data[0].cuda(), data[1].cuda()
                if args.aug:
                    x_model = random_mri(x, x > 0)
                    y_model = random_mri(y, y > 0)
                else:
                    x_model, y_model = x, y
            if x.max() < 0.2 or y.max() < 0.2:
                continue

            flow = model((x_model, y_model))
            output = spatial_trans(x, flow)
            loss_ncc = criterion_ncc(output, y) * args.ncc_weight
            loss_reg = criterion_reg(flow, y) * diffusion_weight
            loss = loss_ncc + loss_reg
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            flow = model((y_model, x_model))
            output = spatial_trans(y, flow)
            loss_ncc = criterion_ncc(output, x) * args.ncc_weight
            loss_reg = criterion_reg(flow, x) * diffusion_weight
            loss = loss_ncc + loss_reg
            loss_all.update(loss.item(), x.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(
                idx, len(train_loader), loss.item(), loss_ncc.item(), loss_reg.item()
            ))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))

        eval_dsc = AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                x, y, x_seg, y_seg = [item.cuda() for item in data]
                grid_img = mk_grid_img(8, 1, config.img_size, dim=0).cuda()
                flow = model((x.float(), y.float()))
                def_out = spatial_trans_nn(x_seg.float(), flow)
                def_grid = spatial_trans(grid_img.float(), flow)
                dsc = dice_val_VOI(def_out.long(), y_seg.long(), num_clus=133)
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)

        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_dsc': best_dsc,
                'optimizer': optimizer.state_dict(),
            },
            save_dir=save_dir,
            filename='dsc{:.4f}.pth.tar'.format(eval_dsc.avg),
        )

        for name, img in [('Grid', def_grid), ('moving', x), ('fixed', y), ('deformed', flow)]:
            fig = comput_fig(img)
            writer.add_figure(name, fig, epoch)
            plt.close(fig)
        loss_all.reset()

    writer.close()


if __name__ == '__main__':
    main()
