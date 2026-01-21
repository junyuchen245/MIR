from torch.utils.tensorboard import SummaryWriter
import os, utils, glob
import sys
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt
from natsort import natsorted
from MIR.models import SpatialTransformer, SSLHeadNLvl, VFA, VFASPR, HyperVFASPR, TemplateCreation
from MIR.image_similarity import NCC_vxm, LocalCorrRatio, MIND_loss, NCC_gauss, NCC_mok, CorrRatio, NCC_vfa, FastNCC, NCC_fp16, NCC, SSIM3D, DiceLoss, sparse_dice_from_int_labels
from MIR.deformation_regularizer import Grad3d, GradICON3d, GradICONExact3d, logBeta, LocalGrad3d
from MIR.utils import Logger, AverageMeter, load_partial_weights
import MIR.models.configs_VFA as CONFIGS_VFA
from MIR.accuracy_measures import dice_val_VOI
from MIR.utils import mk_grid_img
from MIR import RandomMultiContrastRemap
import matplotlib, random
matplotlib.use('Agg')
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import nibabel as nib
from torch.utils.data import Dataset
import torch.nn as nn

class JHUPSMADataset(Dataset):
    def __init__(self, data_path, data_names, stage='train'):
        self.path = data_path
        self.data_names = data_names
        self.stage = stage

    def norm_img(self, img):
        img[img < -300] = -300
        img[img > 300] = 300
        norm = (img - img.min()) / (img.max() - img.min())
        return norm

    def norm_suv(self, img):
        img_max = 15.#np.percentile(img, 95)
        img_min = 0.#np.percentile(img, 5)
        norm = (img - img_min)/(img_max - img_min)
        norm[norm < 0] = 0
        norm[norm > 1] = 1
        return norm

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        mov_name = self.data_names[index]
        x = nib.load(self.path + '{}_CT.nii.gz'.format(mov_name))
        x = self.norm_img(x.get_fdata())
        x_suv = nib.load(self.path + '{}_SUV.nii.gz'.format(mov_name))
        x_suv = self.norm_suv(x_suv.get_fdata())
        x_seg = nib.load(self.path + '{}_CT_seg.nii.gz'.format(mov_name))
        x_seg = x_seg.get_fdata()
        if self.stage != 'train':
            x_seg = utils.remap_totalseg_lbl(x_seg)
        x_suv_seg = nib.load(self.path + '{}_SUV_seg.nii.gz'.format(mov_name))
        x_suv_seg = utils.remap_suv_lbl(x_suv_seg.get_fdata())
        x = x[None, ...]
        x_suv = x_suv[None, ...]
        x_seg = x_seg[None, ...]
        x_suv_seg = x_suv_seg[None, ...]
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        x_suv = np.ascontiguousarray(x_suv)  # [Bsize,channelsHeight,,Width,Depth]
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        x_suv_seg = np.ascontiguousarray(x_suv_seg)  # [Bsize,channelsHeight,,Width,Depth]
        x = torch.from_numpy(x)
        x_suv = torch.from_numpy(x_suv)
        x_seg = torch.from_numpy(x_seg)
        x_suv_seg = torch.from_numpy(x_suv_seg)
        return {'CT': x.float(), 'SUV': x_suv.float(), 'CT_seg': x_seg.long(), 'SUV_seg': x_suv_seg.long()}

    def __len__(self):
        return len(self.data_names)

def main():
    batch_size = 1
    wt_ncc = 1.0
    wt_logBeta = 0.005
    wt_localGrad = 1
    wt_dsc = 0.5
    train_dir = '/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/JHU/'
    val_dir = '/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/JHU/'
    save_dir = f'VFASPR_wt_logBeta_{wt_logBeta}_ssim_{wt_ncc}_localGrad_{wt_localGrad}_dsc_{wt_dsc}/'
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    sys.stdout = Logger('logs/'+save_dir)
    lr_model = 0.0001 #learning rate
    epoch_start = 0
    max_epoch = 500 #max traning epoch

    '''
    Initialize model
    '''
    H, W, D = 192, 192, 256
    scale_factor=1
    config = CONFIGS_VFA.get_VFA_default_config()
    config.img_size = (H//scale_factor, W//scale_factor, D//scale_factor)
    print(config)
    model = VFASPR(config, device='cuda:0', SVF=True, return_full=True)
    best_model = torch.load('experiments/VFASPR_wt_logBeta_0.005_ncc_1.0_localGrad_2.298/dsc0.6127.pth.tar')['state_dict']
    model.load_state_dict(best_model)
    model.cuda()
    
    ct_atlas_dir = 'atlas/ct/VFAAtlas_SSIM_1_MS_1_diffusion_1/'
    x = nib.load(ct_atlas_dir + natsorted(os.listdir(ct_atlas_dir))[-1])
    x = x.get_fdata()[None, None, ...]
    x = torch.from_numpy(x).cuda().float()
    pt_atlas_dir = 'atlas/suv/VFAAtlas_SSIM_1_MS_1_diffusion_1/'
    x_suv = nib.load(pt_atlas_dir + natsorted(os.listdir(pt_atlas_dir))[-1])
    x_suv = x_suv.get_fdata()[None, None, ...]
    x_suv = torch.from_numpy(x_suv).cuda().float()
    
    ''' Initialize segmentation atlas'''
    ct_seg_118 = nib.load('atlas/seg/ct_seg_atlas_w_reg_118lbls.nii.gz')
    ct_seg_118 = ct_seg_118.get_fdata()[None, None, ...]
    ct_seg_118 = torch.from_numpy(ct_seg_118).cuda().float()
    ct_seg_40 = nib.load('atlas/seg/ct_seg_atlas_w_reg_40lbls.nii.gz')
    ct_seg_40 = ct_seg_40.get_fdata()[None, None, ...]
    ct_seg_40 = torch.from_numpy(ct_seg_40).cuda().float()
    ct_seg_oh = nn.functional.one_hot(ct_seg_40.long(), num_classes=40)
    ct_seg_oh = torch.squeeze(ct_seg_oh, 1)
    ct_seg_oh = ct_seg_oh.permute(0, 4, 1, 2, 3).contiguous().cuda()

    suv_seg = nib.load('atlas/seg/suv_seg_atlas_w_reg_14lbls.nii.gz')
    suv_seg = suv_seg.get_fdata()[None, None, ...]
    suv_seg = torch.from_numpy(suv_seg).cuda().float()
    suv_seg_oh = nn.functional.one_hot(suv_seg.long(), num_classes=14)
    suv_seg_oh = torch.squeeze(suv_seg_oh, 1)
    suv_seg_oh = suv_seg_oh.permute(0, 4, 1, 2, 3).contiguous().cuda()
    
    
    '''
    Initialize training
    '''
    full_names = glob.glob(train_dir + '*_CT_seg*')
    full_names = [name.split('/')[-1].split('_CT_seg')[0] for name in full_names]
    train_names = full_names[0:-8]
    val_names = full_names[-8:]
    print('Training on {} patients, validating on {} patients'.format(len(train_names), len(val_names)))
    train_set = JHUPSMADataset(train_dir, train_names, stage='train')
    val_set = JHUPSMADataset(val_dir, val_names, stage='validation')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    params = [{'params': model.parameters(), 'lr': lr_model}]
    optimizer = optim.AdamW(params, lr=lr_model, weight_decay=0, amsgrad=True)
    criterion_ncc = SSIM3D()
    criterion_reg = logBeta()
    criterion_reg2 = LocalGrad3d(penalty='l2')
    criterion_dsc_CT = DiceLoss(num_class=36, one_hot=False)
    criterion_dsc_SUV = DiceLoss(num_class=14, one_hot=True)
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
            adjust_learning_rate(optimizer, epoch, max_epoch, lr_model)
            with torch.no_grad():
                y = data['CT'].cuda()
                y_suv = data['SUV'].cuda()
                y_seg = data['CT_seg'].cuda()
                y_suv_seg = data['SUV_seg'].cuda()
                
                #y_seg_oh = nn.functional.one_hot(y_seg.long(), num_classes=40)
                #y_seg_oh = torch.squeeze(y_seg_oh, 1)
                #y_seg_oh = y_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
                
                y_suv_seg_oh = nn.functional.one_hot(y_suv_seg.long(), num_classes=14)
                y_suv_seg_oh = torch.squeeze(y_suv_seg_oh, 1)
                y_suv_seg_oh = y_suv_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
                
            # CT registration
            _, _, pos_flow, neg_flow, wts = model((x, y))
            
            #x_seg_def = model.spatial_trans(ct_seg_oh.float(), pos_flow.float())
            #y_seg_def = model.spatial_trans(y_seg_oh.float(), neg_flow.float())
            
            x_def = model.spatial_trans(x.float(), pos_flow.float())
            y_def = model.spatial_trans(y.float(), neg_flow.float())
            
            smo_weight  = 1. + wt_logBeta
            loss_img = criterion_ncc(x_def, y) * wt_ncc / 2 + criterion_ncc(y_def, x) * wt_ncc / 2
            #loss_dsc_CT = criterion_dsc_CT(x_seg_def, y_seg.long()) * wt_dsc / 2 + criterion_dsc_CT(y_seg_def, ct_seg.long()) * wt_dsc / 2
            loss_dsc_CT_forward = sparse_dice_from_int_labels(src_lbl_int=ct_seg_118, tgt_lbl_int=y_seg, flow_src2tgt=pos_flow, warp_fn=model.spatial_trans, num_classes=118, K=36, include_bg=True, present_only=True, dice_loss=criterion_dsc_CT) * 0.5
            loss_dsc_CT_backward = sparse_dice_from_int_labels(src_lbl_int=y_seg, tgt_lbl_int=ct_seg_118, flow_src2tgt=neg_flow, warp_fn=model.spatial_trans, num_classes=118, K=36, include_bg=True, present_only=True, dice_loss=criterion_dsc_CT) * 0.5
            loss_dsc_CT = (loss_dsc_CT_forward + loss_dsc_CT_backward) * wt_dsc
            loss_reg = criterion_reg(wts, smo_weight)
            loss_reg2 = criterion_reg2(pos_flow, wts) * wt_localGrad / 2 + criterion_reg2(neg_flow, wts) * wt_localGrad / 2
            loss = loss_img + loss_reg + loss_reg2 + loss_dsc_CT
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del x_def, y_def, pos_flow, neg_flow, wts
            torch.cuda.empty_cache()
            print('CT: Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Wts Reg: {:.6f}, Flow Reg: {:.6f}, Seg: {:.6f}'.format(idx, len(train_loader),
                                                                                                        loss.item(),
                                                                                                        loss_img.item(),
                                                                                                        loss_reg.item(),
                                                                                                        loss_reg2.item(),
                                                                                                        loss_dsc_CT.item()))
            
            # PET registration
            _, _, pos_flow, neg_flow, wts = model((x_suv, y_suv))

            x_suv_def = model.spatial_trans(x_suv.float(), pos_flow.float())
            y_suv_def = model.spatial_trans(y_suv.float(), neg_flow.float())
            
            x_suv_seg_def = model.spatial_trans(suv_seg_oh.float(), pos_flow.float())
            y_suv_seg_def = model.spatial_trans(y_suv_seg_oh.float(), neg_flow.float())

            loss_img = criterion_ncc(x_suv_def, y_suv) * wt_ncc / 2 + criterion_ncc(y_suv_def, x_suv) * wt_ncc / 2
            loss_dsc_SUV = criterion_dsc_SUV(x_suv_seg_def, y_suv_seg.long()) * wt_dsc / 2 + criterion_dsc_SUV(y_suv_seg_def, suv_seg.long()) * wt_dsc / 2
            loss_reg = criterion_reg(wts, smo_weight)
            loss_reg2 = criterion_reg2(pos_flow, wts) * wt_localGrad / 2 + criterion_reg2(neg_flow, wts) * wt_localGrad / 2
            loss = loss_img + loss_reg + loss_reg2 + loss_dsc_SUV
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del x_suv_def, y_suv_def, x_suv_seg_def, y_suv_seg_def, pos_flow, neg_flow, wts
            torch.cuda.empty_cache()
            print('PET: Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Wts Reg: {:.6f}, Flow Reg: {:.6f}, Seg: {:.6f}'.format(idx, len(train_loader),
                                                                                                        loss.item(),
                                                                                                        loss_img.item(),
                                                                                                        loss_reg.item(),
                                                                                                        loss_reg2.item(),
                                                                                                        loss_dsc_SUV.item()))


        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_dsc = AverageMeter()
        dsc_raw = []
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                y = data['CT'].cuda().float()
                y_suv = data['SUV'].cuda().float()
                y_ct_seg = data['CT_seg'].cuda().float()
                y_suv_seg = data['SUV_seg'].cuda().float()

                grid_img = mk_grid_img(8, 1, grid_sz=(H, W, D))

                _, _, pos_flow_ct, _, _ = model((x, y))
                _, _, pos_flow_suv, neg_flow_suv, _ = model((x_suv, y_suv))

                x_suv_def = model.spatial_trans(x_suv.cuda().float(), pos_flow_suv.cuda())
                y_suv_def = model.spatial_trans(y_suv.cuda().float(), neg_flow_suv.cuda())
                def_ct_seg = model.spatial_trans(ct_seg_oh.cuda().float(), pos_flow_ct.cuda())
                def_ct_seg = torch.argmax(def_ct_seg, dim=1, keepdim=True)
                def_suv_seg = model.spatial_trans(suv_seg_oh.cuda().float(), pos_flow_suv.cuda())
                def_suv_seg = torch.argmax(def_suv_seg, dim=1, keepdim=True)
                def_grid = model.spatial_trans(grid_img.float().cuda(), pos_flow_ct.cuda())
                dsc = 0.5*dice_val_VOI((def_ct_seg).long(), y_ct_seg.long(), 40) + 0.5*dice_val_VOI((def_suv_seg).long(), y_suv_seg.long(), 14)
                if epoch == 0:
                    dsc_raw.append(0.5*dice_val_VOI(suv_seg_oh.long(), y_suv_seg.long()).item()+0.5*dice_val_VOI(ct_seg_oh.long(), y_ct_seg.long()).item())
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
            if epoch == 0:
                print('raw dice: {}'.format(np.mean(dsc_raw)))
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/' + save_dir, filename='dsc{:.4f}.pth.tar'.format(eval_dsc.avg))
        plt.switch_backend('agg')
        pred_fig = comput_fig(pos_flow_ct)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x)
        tar_fig = comput_fig(y)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('moving', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('fixed', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('deformed', pred_fig, epoch)
        plt.close(pred_fig)
        loss_all.reset()
        del x_suv_def, y_suv_def, def_ct_seg, def_suv_seg, pos_flow_ct, pos_flow_suv, neg_flow_suv, def_grid, grid_img, y_ct_seg, y_suv_seg, y, y_suv
        torch.cuda.empty_cache()
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

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=2):
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
    torch.manual_seed(0)
    main()