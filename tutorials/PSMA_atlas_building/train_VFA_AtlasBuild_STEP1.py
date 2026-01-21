from torch.utils.tensorboard import SummaryWriter
import os, glob
import sys, random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from MIR.models import VFA, TemplateCreation, SpatialTransformer
import MIR.models.configs_VFA as CONFIGS_VFA
from MIR.image_similarity import SSIM3D
from MIR.deformation_regularizer import Grad3d
from MIR.utils import Logger, AverageMeter
import nibabel as nib

class JHUPSMADataset(Dataset):
    def __init__(self, data_path, data_names):
        self.path = data_path
        self.data_names = data_names

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
        x = x[None, ...]
        x_suv = x_suv[None, ...]
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        x_suv = np.ascontiguousarray(x_suv)  # [Bsize,channelsHeight,,Width,Depth]
        x = torch.from_numpy(x)
        x_suv = torch.from_numpy(x_suv)
        return x, x_suv

    def __len__(self):
        return len(self.data_names)

def estimate_initial_template(train_loader):
    print('Generating initial template...')
    with torch.no_grad():
        idx = 0
        x_mean = 0
        x_suv_mean = 0
        for data in train_loader:
            x = data[0].cuda().float()
            x_suv = data[1].cuda().float()
            x_mean += x
            x_suv_mean += x_suv
            idx += 1
            print('Image {} of {}'.format(idx, len(train_loader)))
        x_mean = x_mean/idx
        x_suv_mean = x_suv_mean / idx
    return x_mean, x_suv_mean

def main():
    batch_size = 1
    weights = [1, 1, 1] # loss weights
    train_dir = '/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/JHU/'
    val_dir = '/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/JHU/'
    save_dir = 'VFAAtlas_JHU_SSIM_{}_MS_{}_diffusion_{}/'.format(weights[0], weights[1], weights[2])
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('atlas/ct/'+save_dir):
        os.makedirs('atlas/ct/'+save_dir)
    if not os.path.exists('atlas/suv/'+save_dir):
        os.makedirs('atlas/suv/'+save_dir)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    sys.stdout = Logger('logs/'+save_dir)
    lr_model = 0.0001  # learning rate
    lr_atlas = 0.001
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
    vfa_model = VFA(config, device='cuda:0', SVF=True, return_full=True).cuda()
    model = TemplateCreation(vfa_model, (H, W, D))
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = SpatialTransformer((H, W, D), 'nearest')
    reg_model.cuda()
    spatial_trans = SpatialTransformer((H, W, D)).cuda()

    '''
    Initialize training
    '''
    full_names = glob.glob(train_dir + '*_CT_seg*')
    full_names = [name.split('/')[-1].split('_CT_seg')[0] for name in full_names]
    train_names = full_names[0:-8]
    val_names = full_names[-8:]
    print('Training on {} patients, validating on {} patients'.format(len(train_names), len(val_names)))
    train_set = JHUPSMADataset(train_dir, train_names)
    val_set = JHUPSMADataset(val_dir, val_names)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    x, x_suv = estimate_initial_template(train_loader)
    x.requires_grad_(); x_suv.requires_grad_()
    params = [{'params': model.parameters(), 'lr': lr_model}] + [{'params': x, 'lr': lr_atlas}] + \
             [{'params': x_suv, 'lr': lr_atlas}]
    optimizer = optim.AdamW(params, lr=lr_model, weight_decay=0, amsgrad=True)
    criterion_img = SSIM3D()##nn.L1Loss()
    criterion_mse = nn.MSELoss()
    criterion_reg = Grad3d(penalty='l2')
    SSIM = SSIM3D()
    best_ssim = 0
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
            x.requires_grad_(); x_suv.requires_grad_()
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr_model, lr_atlas)
            with torch.no_grad():
                y = data[0].cuda().float()
                y_suv = data[1].cuda().float()

            def_atlas, def_image, pos_flow, neg_flow, mean_stream = model((x, y))
            x_suv_def = model.reg_model.spatial_trans(x_suv.float(), pos_flow.float())
            y_suv_def = model.reg_model.spatial_trans(y_suv.float(), neg_flow.float())

            loss_suv_1 = criterion_img(x_suv_def, y_suv) * weights[0] / 2
            loss_suv_2 = criterion_img(y_suv_def, x_suv) * weights[0] / 2

            loss_img_1 = criterion_img(def_atlas, y) * weights[0] / 2
            loss_img_2 = criterion_img(def_image, x) * weights[0] / 2
            loss_mean_stream = criterion_mse(mean_stream, torch.zeros_like(mean_stream).cuda()) * weights[1]
            loss_reg = criterion_reg(pos_flow, y) * weights[2]
            loss = (loss_img_1 + loss_img_2 + loss_suv_1 + loss_suv_2)/2 + loss_mean_stream + loss_reg
            loss_img = loss_img_1 + loss_img_2
            loss_suv = loss_suv_1 + loss_suv_2
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Suv Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader),
                                                                                            loss.item(),
                                                                                            loss_img.item(),
                                                                                            loss_suv.item(),
                                                                                            loss_reg.item()))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_ssim = AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                y = data[0].cuda().float()
                y_suv = data[1].cuda().float()

                grid_img = mk_grid_img(8, 1, grid_sz=(H, W, D))

                def_atlas, def_image, pos_flow, neg_flow, mean_stream = model((x, y))
                
                def_out = model.reg_model.spatial_trans(x.cuda().float(), pos_flow.cuda())
                def_grid = model.reg_model.spatial_trans(grid_img.float(), pos_flow.cuda())
                ssim = 1 - SSIM(def_out, y)
                eval_ssim.update(ssim.item(), x.size(0))
                print(eval_ssim.avg)
        best_ssim = max(eval_ssim.avg, best_ssim)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_ssim': best_ssim,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/'+save_dir, filename='ssim{:.4f}.pth.tar'.format(eval_ssim.avg))
        save_atlas(x, 'atlas/ct/' + save_dir, filename='ssim{:.4f}.nii.gz'.format(eval_ssim.avg))
        save_atlas(x_suv, 'atlas/suv/' + save_dir, filename='ssim{:.4f}.nii.gz'.format(eval_ssim.avg))
        writer.add_scalar('SSIM/validate', eval_ssim.avg, epoch)
        plt.switch_backend('agg')
        suv_fig = comput_fig(x_suv)
        pred_fig = comput_fig(def_out)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x)
        tar_fig = comput_fig(y)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('template', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
        writer.add_figure('suv_template', suv_fig, epoch)
        plt.close(suv_fig)
        loss_all.reset()
    writer.close()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, :, 88:96, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[1]):
        plt.subplot(2, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[:, i, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR_model, INIT_LR_atlas, power=0.9):
    idx = 0
    for param_group in optimizer.param_groups:
        if idx == 0:
            param_group['lr'] = round(INIT_LR_model * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)
        else:
            param_group['lr'] = round(INIT_LR_atlas * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)
        idx += 1

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[0], grid_step):
        grid_img[j+line_thickness-1, :, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=4):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

def make_affine_from_pixdim(pixdim):
    # Create a 4x4 affine with spacing along the diagonal
    affine = np.eye(4)
    affine[0, 0] = pixdim[0]
    affine[1, 1] = pixdim[1]
    affine[2, 2] = pixdim[2]
    return affine

def save_atlas(atlas, save_dir='atlas', filename='checkpoint.pth.tar', max_model_num=4):
    target_pixdim = [2.8, 2.8, 3.8]
    affine = make_affine_from_pixdim(target_pixdim)
    atlas = nib.Nifti1Image(atlas.float().detach().cpu().numpy()[0, 0], affine=affine)
    nib.save(atlas, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

def seedBasic(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def seedTorch(seed=2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    DEFAULT_RANDOM_SEED = 42

    seedBasic(DEFAULT_RANDOM_SEED)
    seedTorch(DEFAULT_RANDOM_SEED)
    main()