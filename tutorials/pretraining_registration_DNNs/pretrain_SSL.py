from torch.utils.tensorboard import SummaryWriter
import os, glob
import sys
from torch.utils.data import DataLoader
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from MIR.models import TransMorphTVF, SpatialTransformer, SSLHeadNLvl
from MIR.image_similarity import NCC_vxm, NCC_gauss
from MIR.deformation_regularizer import KL_divergence, Grad3d
from MIR.utils import Logger, AverageMeter
import MIR.models.configs_TransMorph as CONFIGS_TM
from MIR.accuracy_measures import dice_val_VOI
import torch.nn.functional as F
import MIR.random_image_generation as rs
from MIR.utils import mk_grid_img

def main():
    iter_max = 3000
    val_step = 50
    weights = [1, 0.5, 1e-7]#[1, 1, 1e-7]
    lr = 0.0004
    epoch_start = 0
    max_epoch = 50
    scale_factor = 2
    win_factor = 64
    
    '''
    Initialize model
    '''
    H, W, D = 224, 224, 224
    config = CONFIGS_TM.get_3DTransMorphDWin3Lvl_config()
    config.img_size = (H//scale_factor, W//scale_factor, D//scale_factor)
    config.window_size = (H // win_factor, W // win_factor, D // win_factor)
    config.out_chan = 3
    tm = TransMorphTVF(config, time_steps=7, SVF=True, composition='addition', swin_type='swin')
    encoder = tm.transformer.cuda()
    model = SSLHeadNLvl(encoder, img_size=(H//scale_factor, W//scale_factor, D//scale_factor), channels=(config.embed_dim*4, config.embed_dim*2, config.embed_dim), if_upsamp=True, encoder_output_type='single')
    model.cuda()
    del tm
    
    save_dir = 'TransMorphTVFSSL_HWD_{}_{}_{}_Scale_{}_Wsize_{}_PreTrain_RS_SSIM_{}_diffusion_{}_sKL_{}/'.format(H, W, D, scale_factor, win_factor, weights[0], weights[1], weights[2])
    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    sys.stdout = Logger('logs/' + save_dir)
    
    
    reg_model = SpatialTransformer((H, W, D), 'nearest')
    reg_model.cuda()
    reg_model_bilin = SpatialTransformer((H, W, D), 'bilinear')
    reg_model_bilin.cuda()
    updated_lr = lr
    optimizer = optim.AdamW(model.parameters(), lr=updated_lr)
    criterion = NCC_gauss()
    criterion_KL = KL_divergence()
    criterion_reg = Grad3d(penalty='l2')
    best_dsc = 0
    writer = SummaryWriter(log_dir='logs/' + save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = AverageMeter()
        for idx in range(iter_max):
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            model.train()
            with torch.no_grad():
                data = rs.gen_shapes((H, W, D), res=(H // 32, W // 32, D // 32),)
                x = data[0].cuda()
                y = data[1].cuda()
                x_half = F.avg_pool3d(x, 2).cuda()
                y_half = F.avg_pool3d(y, 2).cuda()
            _, flow, stats = model((x_half, y_half))
            flow = F.interpolate(flow.cuda(), scale_factor=2, mode='trilinear', align_corners=False) * 2
            output = reg_model_bilin(x.float(), flow.float())
            loss_ncc = criterion(output, y)
            loss_ncc = loss_ncc * weights[0]
            loss_kl = 0
            if weights[2] > 0:
                ref_flow, ref_log_sigma = stats[-1]
                for stat in stats[:-1]:
                    mean_flow, log_sigma = stat
                    loss_kl += criterion_KL((mean_flow, log_sigma), (ref_flow, ref_log_sigma))/(len(stats)-1)
                loss_kl = loss_kl * weights[2]
            else:
                loss_kl = loss_ncc * weights[2]
            loss_reg = criterion_reg(flow, y)
            loss_reg = loss_reg * weights[1]
            loss = loss_ncc + loss_kl + loss_reg
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch {}: Iter {} of {} loss {:.4f}, Img NCC: {:.6f}, Reg: {:.6f}, KL: {:.6f}'.format(epoch, idx, iter_max, loss.item(), loss_ncc.item(), loss_reg.item(), loss_kl.item()))
            del output, flow, stats
        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        '''
        Validation
        '''
        eval_dsc = AverageMeter()
        with torch.no_grad():
            for val_i in range(val_step):
                model.eval()
                data = rs.gen_shapes((H, W, D), res=(H // 32, W // 32, D // 32),)
                x = data[0].cuda()
                y = data[1].cuda()
                x_seg = data[2].float().cuda()
                y_seg = data[3].float().cuda()
                x_half = F.avg_pool3d(x, 2).cuda()
                y_half = F.avg_pool3d(y, 2).cuda()
                grid_img = mk_grid_img(8, 1, (H, W, D))
                outputs, flow, _  = model((x_half, y_half))
                flow = F.interpolate(flow.cuda(), scale_factor=2, mode='trilinear', align_corners=False) * 2
                flow_avg = flow
                output = reg_model_bilin(x.float(), flow_avg.cuda())
                def_out = reg_model(x_seg.cuda().float(), flow_avg.cuda())
                def_grid = reg_model_bilin(grid_img.float(), flow_avg.cuda())
                dsc = rs.dice_Shape_VOI(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mse': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/' + save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        plt.switch_backend('agg')
        pred_fig = comput_fig(output)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x)
        tar_fig = comput_fig(y)
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

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=3):
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