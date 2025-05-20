from torch.utils.tensorboard import SummaryWriter
import os, glob
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from MIR.models import TransMorphTVF, SpatialTransformer, SSLHeadNLvl
from MIR.image_similarity import NCC_vxm
from MIR.deformation_regularizer import Grad3d
from MIR.utils import Logger, AverageMeter
import MIR.models.configs_TransMorph as CONFIGS_TM
from MIR.accuracy_measures import dice_val_VOI
from MIR.utils import mk_grid_img, pkload
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')
import torch.nn.functional as F

def main():
    batch_size = 1
    atlas_dir = '/scratch/jchen/DATA/IXI/atlas.pkl'
    train_dir = '/scratch/jchen/DATA/IXI/Train/'
    val_dir = '/scratch/jchen/DATA/IXI/Val/'
    weights = [1, 1] # loss weights
    save_dir = 'TransMorphTVFDWin_ncc_{}_diffusion_{}/'.format(weights[0], weights[1])
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    sys.stdout = Logger('logs/'+save_dir)
    lr = 0.0001 # learning rate
    epoch_start = 0
    max_epoch = 250 #max traning epoch
    cont_training = False #if continue training

    '''
    Initialize model
    '''
    scale_factor = 2
    win_factor = 64
    H, W, D = 160, 192, 224
    config = CONFIGS_TM.get_3DTransMorphDWin3Lvl_config()
    config.img_size = (H//scale_factor, W//scale_factor, D//scale_factor)
    config.out_size = (H//scale_factor, W//scale_factor, D//scale_factor)
    config.window_size = (H // win_factor, W // win_factor, D // win_factor)
    config.out_chan = 3
    model = TransMorphTVF(config, SVF=True, composition='composition', swin_type='swin')
    
    pretrained_dir = '/scratch/jchen/python_projects/Registration_SSL/experiments/TransMorphSSL_HWD_160_192_224_Scale_2_Wsize_64_PreTrain_RS_dice_1_diffusion_1_sKL_1e-07/'
    pretrained = torch.load(pretrained_dir + natsorted(os.listdir(pretrained_dir))[-1])['state_dict']
    sslencoder = SSLHeadNLvl(model.transformer, img_size=(H//2, W//2, D//2), channels=(config.embed_dim * 4, config.embed_dim * 2, config.embed_dim), if_upsamp=True)
    sslencoder.load_state_dict(pretrained)
    model.transformer.load_state_dict(sslencoder.encoder.state_dict())
    print('model: pretrained.pth.tar loaded!')
    del sslencoder
    model.cuda()
    
    '''
    Initialize spatial transformation function
    '''
    reg_model = SpatialTransformer((H, W, D), 'nearest')
    reg_model.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 201
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

    val_composed = transforms.Compose([trans.Seg_norm(), #rearrange segmentation label to 1 to 46
                                       trans.NumpyType((np.float32, np.int16))])
    train_set = datasets.IXIBrainDataset(glob.glob(train_dir + '*.pkl'), atlas_dir, transforms=train_composed, stage='train')
    val_set = datasets.IXIBrainDataset(glob.glob(val_dir + '*.pkl'), atlas_dir, transforms=val_composed, stage='validation')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion_ncc = NCC_vxm()
    criterion_reg = Grad3d(penalty='l2')
    best_dsc = 0
    writer = SummaryWriter(log_dir='logs/'+save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            with torch.no_grad():
                x = data[0].cuda().float()
                y = data[1].cuda().float()
                x_half = F.avg_pool3d(x, 2).cuda()
                y_half = F.avg_pool3d(y, 2).cuda()
            flow = model((x_half, y_half))
            flow = F.interpolate(flow.cuda(), scale_factor=2, mode='trilinear', align_corners=False) * 2
            output = model.spatial_trans(x, flow)
            loss_ncc = criterion_ncc(output, y) * weights[0]
            loss_reg = criterion_reg(flow, flow) * weights[1]
            loss = loss_ncc + loss_reg
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_ncc.item(), loss_reg.item()))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_dsc = AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0].cuda().float()
                y = data[1].cuda().float()
                x_half = F.avg_pool3d(x, 2).cuda()
                y_half = F.avg_pool3d(y, 2).cuda()
                x_seg = data[2]
                y_seg = data[3]
                flow = model((x_half, y_half))
                flow = F.interpolate(flow.cuda(), scale_factor=2, mode='trilinear', align_corners=False) * 2
                grid_img = mk_grid_img(8, 1, config.img_size, dim=0).cuda()
                def_seg = reg_model(x_seg.cuda().float(), flow.cuda())
                def_grid = model.spatial_trans(grid_img.float(), flow.cuda())
                dsc = dice_val_VOI(def_seg.long(), y_seg.long(), num_clus=36)
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/'+save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        plt.switch_backend('agg')
        pred_fig = comput_fig(def_seg)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x_seg)
        tar_fig = comput_fig(y_seg)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('input', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
        loss_all.reset()
    writer.close()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=4):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

class IXIBrainDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms, amount=1, stage='train'):
        num_paths = len(data_path)
        num_to_load = max(1, int(num_paths * amount))  # Ensure at least 1 path is loaded
        self.paths = data_path[:num_to_load]  # Slice to first 10%
        self.atlas_path = atlas_path
        self.transforms = transforms
        self.stage = stage

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        if self.stage == 'train':
            x, y = x[None, ...], y[None, ...]
            x,y = self.transforms([x, y])
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            return x, y
        else:
            x, y = x[None, ...], y[None, ...]
            x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
            x, x_seg = self.transforms([x, x_seg])
            y, y_seg = self.transforms([y, y_seg])
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)
            x_seg = np.ascontiguousarray(x_seg)
            y_seg = np.ascontiguousarray(y_seg)
            x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
            return x, y, x_seg, y_seg
    def __len__(self):
        return len(self.paths)

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()