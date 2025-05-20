from torch.utils.tensorboard import SummaryWriter
import os, glob
import sys
from torch.utils.data import DataLoader
from data import datasets
import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt
from natsort import natsorted
from MIR.models import HyperVxmDense, SpatialTransformer
from MIR.image_similarity import NCC_vxm
from MIR.deformation_regularizer import Grad3d
from MIR.utils import Logger, AverageMeter
import MIR.models.configs_VoxelMorph as CONFIGS_VXM
from MIR.accuracy_measures import dice_val_VOI
from MIR.utils import mk_grid_img
import matplotlib
matplotlib.use('Agg')
import torch.nn.functional as F
from data import datasets, trans
from torchvision import transforms
VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]

def main():
    batch_size = 1
    val_hyper = 0.3
    atlas_dir = '/scratch/jchen/DATA/IXI/atlas.pkl'
    train_dir = '/scratch/jchen/DATA/IXI/Train/'
    val_dir = '/scratch/jchen/DATA/IXI/Val/'
    weights = [1, 1] # loss weights
    save_dir = 'HyperMorph2_0.3_ncc_{}_diffusion_{}/'.format(weights[0], weights[1])
    if not os.path.exists('/scratch2/jchen/IXI/experiments/'+save_dir):
        os.makedirs('/scratch2/jchen/IXI/experiments/'+save_dir)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    sys.stdout = Logger('logs/'+save_dir)
    lr = 0.0001 # learning rate
    epoch_start = 0
    max_epoch = 500 #max traning epoch
    cont_training = False #if continue training

    '''
    Initialize model
    '''
    H, W, D = 160, 192, 224
    scale_factor=1
    config = CONFIGS_VXM.get_VXM_default_config()
    config.img_size = (H//scale_factor, W//scale_factor, D//scale_factor)
    print(config)
    model = HyperVxmDense(config)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    spatial_trans = SpatialTransformer((H, W, D))
    spatial_trans.cuda()
    spatial_trans_nn = SpatialTransformer((H, W, D), 'nearest')
    spatial_trans_nn.cuda()

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
    train_set = datasets.IXIBrainDataset(glob.glob(train_dir + '*.pkl'), atlas_dir, transforms=train_composed)
    val_set = datasets.IXIBrainInferDataset(glob.glob(val_dir + '*.pkl'), atlas_dir, transforms=val_composed)
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
            reg_code = torch.rand(1, dtype=x.dtype, device=x.device).unsqueeze(dim=0)
            x_in = torch.cat((x_half, y_half), dim=1)
            flow = model(x_in, reg_code)
            flow = F.interpolate(flow.cuda(), scale_factor=2, mode='trilinear', align_corners=False) * 2
            output = spatial_trans(x, flow)
            loss_ncc = criterion_ncc(output, y) * (1-reg_code)
            loss_reg = criterion_reg(flow, flow) * reg_code#weights[1]
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
                reg_code = torch.tensor([val_hyper], dtype=x.dtype, device=x.device).unsqueeze(dim=0)
                x_in = torch.cat((x_half, y_half), dim=1)
                flow = model(x_in, reg_code)
                flow = F.interpolate(flow.cuda(), scale_factor=2, mode='trilinear', align_corners=False) * 2
                grid_img = mk_grid_img(8, 1, (H, W, D), dim=0).cuda()
                def_seg = spatial_trans_nn(x_seg.cuda().float(), flow.cuda())
                def_grid = spatial_trans(grid_img.float(), flow.cuda())
                dsc = dice_val_VOI(def_seg.long(), y_seg.long(), eval_labels=VOI_lbls)
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='/scratch2/jchen/IXI/experiments/'+save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
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