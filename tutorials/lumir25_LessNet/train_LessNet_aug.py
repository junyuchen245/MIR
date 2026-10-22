import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.tensorboard import SummaryWriter
import glob, sys, json, random
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt
from natsort import natsorted
import nibabel as nib
import matplotlib
matplotlib.use('Agg')

from MIR.models import SpatialTransformer, LessNet
import MIR.models.configs_LessNet as configs_LessNet
from MIR.image_similarity import NCC_vxm
from MIR.deformation_regularizer import Grad3d
from MIR.accuracy_measures import dice_val_VOI
from MIR.utils import Logger, AverageMeter, mk_grid_img
from MIR import RandomMultiContrastRemap

TRAIN_JSON = '/scratch2/jchen/DATA/LUMIR_L2R24_Internal/'
VAL_DIR    = '/scratch2/jchen/DATA/LUMIR/LUMIR25/'
VAL_JSON   = VAL_DIR + 'LUMIR25_dataset_wo_landmarks.json'

label_lookup = [
    0.,   4.,  11.,  23.,  30.,  31.,  32.,  35.,  36.,  37.,  38.,  39.,  40.,  41.,
   44.,  45.,  47.,  48.,  49.,  50.,  51.,  52.,  55.,  56.,  57.,  58.,  59.,  60.,
   61.,  62.,  71.,  72.,  73.,  75.,  76., 100., 101., 102., 103., 104., 105., 106.,
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
            d = json.load(f)
        self.imgs = d['training'] if stage == 'train' else d['validation']
        self.base_dir = base_dir
        self.stage = stage

    def label_reassign(self, label_map):
        label = label_map.copy()
        ref = label_map.copy()
        for i, v in enumerate(label_lookup):
            label[ref == v] = i
        return label

    def __getitem__(self, index):
        if self.stage == 'train':
            mov_dict = self.imgs[index]
            fix_dicts = [d for d in self.imgs if d != mov_dict]
            random.shuffle(fix_dicts)
            fix_dict = fix_dicts[0]
            x = nib.load(self.base_dir + mov_dict['image']).get_fdata() / 255.
            y = nib.load(self.base_dir + fix_dict['image']).get_fdata() / 255.
            x = torch.from_numpy(np.ascontiguousarray(x[None])).float()
            y = torch.from_numpy(np.ascontiguousarray(y[None])).float()
            return x, y
        else:
            img_dict = self.imgs[index]
            x = nib.load(self.base_dir + img_dict['moving']).get_fdata() / 255.
            y = nib.load(self.base_dir + img_dict['fixed']).get_fdata() / 255.
            x_lbl = self.label_reassign(
                nib.load(self.base_dir + 'labelsVal/' + img_dict['moving'].split('/')[-1]).get_fdata()
            )
            y_lbl = self.label_reassign(
                nib.load(self.base_dir + 'labelsVal/' + img_dict['fixed'].split('/')[-1]).get_fdata()
            )
            x = torch.from_numpy(np.ascontiguousarray(x[None])).float()
            y = torch.from_numpy(np.ascontiguousarray(y[None])).float()
            x_lbl = torch.from_numpy(np.ascontiguousarray(x_lbl[None])).float()
            y_lbl = torch.from_numpy(np.ascontiguousarray(y_lbl[None])).float()
            return x, y, x_lbl, y_lbl

    def __len__(self):
        return len(self.imgs)


def main():
    batch_size = 1
    weights = [1, 1]
    save_dir = 'LessNet_w_aug_LUMIR25_ncc_{}_diffusion_{}/'.format(weights[0], weights[1])
    os.makedirs('experiments/' + save_dir, exist_ok=True)
    os.makedirs('logs/' + save_dir, exist_ok=True)
    sys.stdout = Logger('logs/' + save_dir)

    lr = 0.0001
    epoch_start = 0
    max_epoch = 500
    cont_training = False

    H, W, D = 160, 224, 192
    config = configs_LessNet.get_LessNet_default_config()
    config.img_size = (H, W, D)
    model = LessNet(config)
    model.cuda()

    spatial_trans    = SpatialTransformer((H, W, D)).cuda()
    spatial_trans_nn = SpatialTransformer((H, W, D), 'nearest').cuda()
    random_mri = RandomMultiContrastRemap(p=0.85)

    if cont_training:
        model_dir = 'experiments/' + save_dir
        updated_lr = round(lr * np.power(1 - epoch_start / max_epoch, 0.9), 8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        model.load_state_dict(best_model)
        print('Model loaded:', natsorted(os.listdir(model_dir))[-1])
    else:
        updated_lr = lr

    train_set  = L2RLUMIRJSONDataset(TRAIN_JSON, VAL_JSON, stage='train')
    val_set    = L2RLUMIRJSONDataset(VAL_DIR,    VAL_JSON, stage='validation')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=1,          shuffle=False, num_workers=1, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    if cont_training:
        model_dir = 'experiments/' + save_dir
        opt_state = torch.load(model_dir + natsorted(os.listdir(model_dir))[0])['optimizer']
        optimizer.load_state_dict(opt_state)

    criterion_ncc = NCC_vxm()
    criterion_reg = Grad3d(penalty='l2')
    writer = SummaryWriter(log_dir='logs/' + save_dir)
    best_dsc = 0

    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        loss_all = AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            with torch.no_grad():
                x, y = data[0].cuda(), data[1].cuda()
                x_aug = random_mri(x, x > 0)
                y_aug = random_mri(y, y > 0)
            if x.max() < 0.2 or y.max() < 0.2:
                continue

            flow = model((x_aug, y_aug))
            output = spatial_trans(x, flow)
            loss = criterion_ncc(output, y) * weights[0] + criterion_reg(flow, y) * weights[1]
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            flow = model((y_aug, x_aug))
            output = spatial_trans(y, flow)
            loss = criterion_ncc(output, x) * weights[0] + criterion_reg(flow, x) * weights[1]
            loss_all.update(loss.item(), x.numel())
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            print('Iter {} of {} loss {:.4f}'.format(idx, len(train_loader), loss.item()))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))

        eval_dsc = AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                x, y, x_seg, y_seg = [t.cuda() for t in data]
                grid_img = mk_grid_img(8, 1, (H, W, D), dim=0).cuda()
                flow     = model((x.float(), y.float()))
                def_out  = spatial_trans_nn(x_seg.float(), flow)
                def_grid = spatial_trans(grid_img.float(), flow)
                dsc = dice_val_VOI(def_out.long(), y_seg.long(), num_clus=133)
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)

        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint(
            {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'best_dsc': best_dsc, 'optimizer': optimizer.state_dict()},
            save_dir='experiments/' + save_dir,
            filename='dsc{:.4f}.pth.tar'.format(eval_dsc.avg),
        )

        plt.switch_backend('agg')
        for name, img in [('Grid', def_grid), ('moving', x), ('fixed', y), ('deformed', flow)]:
            fig = comput_fig(img)
            writer.add_figure(name, fig, epoch)
            plt.close(fig)
        loss_all.reset()

    writer.close()


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - epoch / MAX_EPOCHES, power), 8)


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=2):
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


if __name__ == '__main__':
    torch.manual_seed(0)
    print('Currently using:', torch.cuda.get_device_name(0))
    main()
