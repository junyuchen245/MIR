import ants
import os, gdown
import json, glob
from torch.utils.data import DataLoader
import numpy as np
import torch
from natsort import natsorted
from MIR.models import VFA, SpatialTransformer
import MIR.models.configs_VFA as CONFIGS_VFA
from MIR import ModelWeights, DatasetJSONs
import matplotlib
matplotlib.use('Agg')
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
import random
from MIR.models import fit_warp_to_svf
from torch.utils.data import Dataset
from MIR.intensity_normalization.normalize.kde import KDENormalize
from MIR.intensity_normalization.typing import Modality, TissueType
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def reorient_image_to_match(reference_nii, target_nii):
    reference_ornt = nib.aff2axcodes(reference_nii.affine)
    target_reoriented = nib.as_closest_canonical(target_nii, enforce_diag=False)
    target_ornt = nib.aff2axcodes(target_reoriented.affine)
    
    # If orientations don't match, perform reorientation
    if target_ornt != reference_ornt:
        # Calculate the transformation matrix to match the reference orientation
        ornt_trans = nib.orientations.ornt_transform(nib.io_orientation(target_reoriented.affine),
                                                     nib.io_orientation(reference_nii.affine))
        target_reoriented = target_reoriented.as_reoriented(ornt_trans)
    return target_reoriented


def resampling(img_npy, img_pixdim, tar_pixdim, order, mode='constant'):
    if order == 0:
        img_npy = img_npy.astype(np.uint16)
    img_npy = zoom(img_npy, ((img_pixdim[0] / tar_pixdim[0]), (img_pixdim[1] / tar_pixdim[1]), (img_pixdim[2] / tar_pixdim[2])), order=order, prefilter=False, mode=mode)
    return img_npy

def intensity_norm(img_npy: np.ndarray, mod: Modality) -> np.ndarray:
    kde_norm = KDENormalize(norm_value=110)
    out = kde_norm(img_npy.astype(np.float32), modality=mod)
    out[out < 0] = 0
    vmax = np.percentile(out[out > 0], 99.9) if np.any(out > 0) else 1.0
    out = np.clip(out / max(vmax, 1e-6), 0, 1).astype(np.float32)
    return out

def ct_norm(img_npy: np.ndarray) -> np.ndarray:
    img_max = np.percentile(img_npy, 98)
    img_min = np.percentile(img_npy, 0)
    out = (img_npy - img_min) / (img_max - img_min)
    out[out < 0] = 0
    out[out > 1] = 1
    return out

def make_affine_from_pixdim(pixdim):
    # Create a 4x4 affine with spacing along the diagonal
    affine = np.eye(4)
    affine[0, 0] = pixdim[0]
    affine[1, 1] = pixdim[1]
    affine[2, 2] = pixdim[2]
    return affine

def save_nii(img, file_name, pix_dim=[1., 1., 1.]):
    x_nib = nib.Nifti1Image(img, np.eye(4))
    x_nib.header.get_xyzt_units()
    x_nib.header['pixdim'][1:4] = pix_dim
    x_nib.to_filename('{}.nii.gz'.format(file_name))

def main():
    batch_size = 1
    val_dir = '/scratch2/jchen/DATA/data_dsa_xa_ct_new/stripped/'
    scale_factor = 1
    output_dir = 'LUMIR_VFAlumir25_ValPhase/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    '''
    Initialize model
    '''
    H, W, D = 160, 224, 192
    config = CONFIGS_VFA.get_VFA_default_config()
    config.img_size = (H//scale_factor, W//scale_factor, D//scale_factor)
    print(config)
    model = VFA(config, device='cuda:0')
    pretrained_dir = 'pretrained_wts/'
    pretrained_wts_multi = 'VFA_LUMIR25.pth'
    if not os.path.isdir("pretrained_wts/"):
        os.makedirs("pretrained_wts/")
    if not os.path.isfile(pretrained_dir+pretrained_wts_multi):
        # download model
        file_id = ModelWeights['VFA-LUMIR25-MultiModal']['wts']
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, pretrained_dir+pretrained_wts_multi, quiet=False)

    pretrained_multi = torch.load(pretrained_dir+pretrained_wts_multi)[ModelWeights['VFA-LUMIR25-MultiModal']['wts_key']]
    model.load_state_dict(pretrained_multi)
    model.cuda()
    
    reg_model = SpatialTransformer((H, W, D))
    reg_model.cuda()
    
    mv_nib = nib.load('mni_icbm152_2009c_t1_1mm_masked_img.nii.gz')
    tmp_nib = nib.load('sub-01_T1w.nii.gz')
    mv_nib = reorient_image_to_match(tmp_nib, mv_nib)
    mv_img = mv_nib.get_fdata()
    mv_img = intensity_norm(mv_img, Modality.T1)
    mv_ants = ants.from_numpy(mv_img)
    
    tmp_img = tmp_nib.get_fdata()
    tmp_img = intensity_norm(tmp_img, Modality.T1)
    tmp_ants = ants.from_numpy(tmp_img)

    for img_i in glob.glob(val_dir + '*_stripped.nii.gz'):
        model.eval()
        
        img_nib = nib.load(img_i)
        img_nib = reorient_image_to_match(tmp_nib, img_nib)
        fx_npy = img_nib.get_fdata()
        fx_npy = ct_norm(fx_npy)
        fx_ants = ants.from_numpy(fx_npy)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(fx_npy[:, :, D//2], cmap='gray')
        plt.axis('off')
        plt.savefig('fx.png')
        
        affine_type = 'Affine'
        affine_metric = 'mattes'
        
        regMovTmp = ants.registration(fixed=tmp_ants, moving=mv_ants, type_of_transform=affine_type, aff_metric=affine_metric)
        mv_ants = ants.apply_transforms(fixed=tmp_ants, moving=mv_ants, transformlist=regMovTmp['fwdtransforms'],)
        
        regFixTmp = ants.registration(fixed=tmp_ants, moving=fx_ants, type_of_transform=affine_type, aff_metric=affine_metric)
        fx_ants = ants.apply_transforms(fixed=tmp_ants, moving=fx_ants, transformlist=regFixTmp['fwdtransforms'],)
        
        mv_img = torch.from_numpy(mv_ants.numpy()[None, None, ...]).cuda().float()
        fx_img = torch.from_numpy(fx_ants.numpy()[None, None, ...]).cuda().float()
        print(mv_img.shape, fx_img.shape)
        with torch.no_grad():
            flow = model((mv_img, fx_img))
        flow = fit_warp_to_svf(flow, nb_steps=7, iters=500, lr=0.1, output_type='svf')
        
        dmv_img = reg_model(mv_img, flow)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(mv_img[0, 0, :, :, D//2].cpu().numpy(), cmap='gray')
        plt.title('Moving Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(fx_img[0, 0, :, :, D//2].cpu().numpy(), cmap='gray')
        plt.title('Fixed Image')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(dmv_img[0, 0, :, :, D//2].cpu().numpy(), cmap='gray')
        plt.title('Deformed Moving Image')
        plt.axis('off')

        plt.savefig('tmp.png')

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