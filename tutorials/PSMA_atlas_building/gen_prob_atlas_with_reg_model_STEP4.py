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
from MIR.models import VFASPR, SpatialTransformer
import MIR.models.configs_VFA as CONFIGS_VFA
from MIR.image_similarity import SSIM3D
from MIR.deformation_regularizer import Grad3d
from MIR.utils import Logger, AverageMeter
import nibabel as nib
import utils

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
        x_seg = nib.load(self.path + '{}_CT_seg.nii.gz'.format(mov_name))
        x_seg = utils.remap_totalseg_lbl(x_seg.get_fdata())
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
    _scale = 0.777047
    train_dir = '/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/JHU/'
    val_dir = '/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/JHU/'
    save_dir = 'VFASPR_wt_logBeta_0.005_ncc_1.0_localGrad_2.298/'
    model_dir = 'experiments/' + save_dir
    '''
    Initialize model
    '''
    H, W, D = 192, 192, 256
    scale_factor=1
    config = CONFIGS_VFA.get_VFA_default_config()
    config.img_size = (H//scale_factor, W//scale_factor, D//scale_factor)
    print(config)
    model = VFASPR(config, device='cuda:0', SVF=True, return_full=True).cuda()
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
    model.load_state_dict(best_model)
    model.cuda()
    raw_scale = torch.tensor([_scale], dtype=torch.float32, device='cuda:0').unsqueeze(dim=0)
    
    
    ct_atlas_dir = 'atlas/ct/VFAAtlas_SSIM_1_MS_1_diffusion_1/'
    pt_atlas_dir = 'atlas/suv/VFAAtlas_SSIM_1_MS_1_diffusion_1/'
    x_ct = nib.load(ct_atlas_dir + natsorted(os.listdir(ct_atlas_dir))[-1])
    x_ct = x_ct.get_fdata()[None, None, ...]
    x_ct = torch.from_numpy(x_ct).cuda().float()
    x_pt = nib.load(pt_atlas_dir + natsorted(os.listdir(pt_atlas_dir))[-1])
    x_pt = x_pt.get_fdata()[None, None, ...]
    x_pt = torch.from_numpy(x_pt).cuda().float()
    
    '''
    Initialize training
    '''
    full_names = glob.glob(train_dir + '*_CT_seg*')
    full_names = [name.split('/')[-1].split('_CT_seg')[0] for name in full_names]

    val_names = full_names
    val_set = JHUPSMADataset(val_dir, val_names)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    '''
    Validation
    '''
    ct_prob_atlas = torch.zeros((1, 40, H, W, D)).cuda()
    suv_prob_atlas = torch.zeros((1, 14, H, W, D)).cuda()
    with torch.no_grad():
        for data in val_loader:
            print('Processing patient: {}'.format(data['CT'].shape))
            model.eval()
            scale = torch.sigmoid(raw_scale)
            y = data['CT'].cuda().float()
            y_suv = data['SUV'].cuda().float()
            y_seg = data['CT_seg'].cuda().long()
            y_suv_seg = data['SUV_seg'].cuda().long()

            y_seg_oh = nn.functional.one_hot(y_seg.long(), num_classes=40)
            y_seg_oh = torch.squeeze(y_seg_oh, 1)
            y_seg_oh = y_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            
            y_suv_seg_oh = nn.functional.one_hot(y_suv_seg.long(), num_classes=14)
            y_suv_seg_oh = torch.squeeze(y_suv_seg_oh, 1)
            y_suv_seg_oh = y_suv_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            
            x_in = x_ct * scale + x_pt * (1 - scale)
            y_in = y * scale + y_suv * (1 - scale)
            def_atlas, def_image, pos_flow, neg_flow, wts = model((x_in, y_in))
            
            def_ct_seg = model.spatial_trans(y_seg_oh.cuda().float(), neg_flow.cuda())
            ct_prob_atlas = ct_prob_atlas + def_ct_seg

            def_suv_seg = model.spatial_trans(y_suv_seg_oh.cuda().float(), neg_flow.cuda())
            suv_prob_atlas = suv_prob_atlas + def_suv_seg

        ct_prob_atlas = ct_prob_atlas / len(val_loader)
        suv_prob_atlas = suv_prob_atlas / len(val_loader)
        
        ct_prob_atlas[:, 27, ...] = ct_prob_atlas[:, 27, ...]*6 #enhance ribs
        ct_seg = torch.argmax(ct_prob_atlas, dim=1, keepdim=True)
        suv_seg = torch.argmax(suv_prob_atlas, dim=1, keepdim=True)
        target_pixdim = [2.8, 2.8, 3.8]
        affine = make_affine_from_pixdim(target_pixdim)
        ct_seg_atlas = nib.Nifti1Image(ct_seg.float().detach().cpu().numpy()[0, 0], affine=affine)
        nib.save(ct_seg_atlas, 'atlas/seg/ct_seg_atlas_w_reg_40lbls.nii.gz')
        suv_seg_atlas = nib.Nifti1Image(suv_seg.float().detach().cpu().numpy()[0, 0], affine=affine)
        nib.save(suv_seg_atlas, 'atlas/seg/suv_seg_atlas_w_reg_14lbls.nii.gz')



def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[0], grid_step):
        grid_img[j+line_thickness-1, :, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def make_affine_from_pixdim(pixdim):
    # Create a 4x4 affine with spacing along the diagonal
    affine = np.eye(4)
    affine[0, 0] = pixdim[0]
    affine[1, 1] = pixdim[1]
    affine[2, 2] = pixdim[2]
    return affine

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