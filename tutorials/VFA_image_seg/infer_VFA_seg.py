import os, gdown
import numpy as np
import torch
from natsort import natsorted
from MIR.models import VFA, AdvancedDecoder3D
import MIR.models.configs_VFA as CONFIGS_VFA
import matplotlib
matplotlib.use('Agg')
import torch.nn.functional as F
import nibabel as nib
from MIR.utils import CenterCropPad3D, normalize_01, SLANT_label_reassign, sliding_window_inference_multires
import matplotlib.pyplot as plt
from MIR import fill_holes_torch, ModelWeights

def save_nii(img, file_name, pix_dim=[1., 1., 1.]):
    x_nib = nib.Nifti1Image(img, np.eye(4))
    x_nib.header.get_xyzt_units()
    x_nib.header['pixdim'][1:4] = pix_dim
    x_nib.to_filename('{}.nii.gz'.format(file_name))

def VFA_synth(exp, model, vfa):
    exp = normalize_01(exp)
    mask = fill_holes_torch(exp>0)
    x_feats = vfa.encoder(exp)
    x_feat = x_feats[0]
    output = model(x_feat)*mask.float()
    return output

def main():
    pretrained_dir = 'pretrained_wts/'
    pretrained_wts = 'VFA_LUMIR25.pth'
    pretrained_seg_head_wts = 'VFA_LUMIR25_seg_head.pth'
    scale_factor = 1
    '''
    Initialize model
    '''
    H, W, D = 160, 224, 192
    scale_factor=1
    config = CONFIGS_VFA.get_VFA_default_config()
    config.img_size = (H//scale_factor, W//scale_factor, D//scale_factor)
    print(config)
    vfa = VFA(config, device='cuda:0')
    if not os.path.isdir(pretrained_dir):
        os.makedirs(pretrained_dir)
    if not os.path.isfile(pretrained_dir + pretrained_wts):
        # download model
        file_id = ModelWeights['VFA-LUMIR25-MultiModal']['wts']
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, pretrained_dir + pretrained_wts, quiet=False)
    pretrained = torch.load(pretrained_dir + pretrained_wts)[ModelWeights['VFA-LUMIR25-MultiModal']['wts_key']]
    vfa.load_state_dict(pretrained)
    print('model: pretrained_wts loaded!')
    vfa.cuda()
    encoder_channels = [min(config.start_channels * 2**(i+1), 64)for i in range(config.downsamples+1)]
    model = AdvancedDecoder3D(
        encoder_channels=encoder_channels,
        aspp_out=128,
        num_classes=133
    )
    if not os.path.isfile(pretrained_dir + pretrained_seg_head_wts):
        # download model
        file_id = ModelWeights['VFA-SegHead']['wts']
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, pretrained_dir + pretrained_seg_head_wts, quiet=False)
    pretrained = torch.load(pretrained_dir + pretrained_seg_head_wts)[ModelWeights['VFA-SegHead']['wts_key']]
    model.load_state_dict(pretrained)
    model.cuda()
    
    '''
    Validation
    '''
    patch_size = (128, 128, 128)
    with torch.no_grad():
        exp = torch.from_numpy(nib.load(f'/scratch/jchen/DATA/Myelin/t2w_affine/sub-720320_t2w.nii.gz').get_fdata()[None, None, ...]).cuda().float()
        exp = normalize_01(exp)
        seg = torch.from_numpy(SLANT_label_reassign(nib.load(f'/scratch/jchen/DATA/Myelin/t1w_affine_seg/sub-720320_t1w.nii.gz').get_fdata()[None, None, ...])).cuda().float()
        print(exp.shape)
        vfa.eval()
        model.eval()
        x_feats = vfa.encoder(exp)
        seg_map = sliding_window_inference_multires(
            x_feats, model, patch_size, overlap=0.5/8, num_classes=133, mode='argmax'
        )
        #seg_map = model(x_feats)
        #seg_map = torch.argmax(seg_map, dim=1, keepdim=True)
        plt.figure(figsize=(6,6), dpi=150)
        plt.subplot(3, 3, 1)
        plt.imshow(seg_map[0, 0, :, :, 96].detach().cpu().numpy(), cmap='gray')
        plt.subplot(3, 3, 2)
        plt.imshow(seg[0, 0, :, :, 96].detach().cpu().numpy(), cmap='gray')
        plt.subplot(3, 3, 3)
        plt.imshow(exp[0, 0, :, :, 96].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=0.9)
        plt.subplot(3, 3, 4)
        plt.imshow(seg_map[0, 0, :, 112, :].detach().cpu().numpy(), cmap='gray')
        plt.subplot(3, 3, 5)
        plt.imshow(seg[0, 0, :, 112, :].detach().cpu().numpy(), cmap='gray')
        plt.subplot(3, 3, 6)
        plt.imshow(exp[0, 0, :, 112, :].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=0.9)
        plt.subplot(3, 3, 7)
        plt.imshow(seg_map[0, 0, 80, :, :].detach().cpu().numpy(), cmap='gray')
        plt.subplot(3, 3, 8)
        plt.imshow(seg[0, 0, 80, :, :].detach().cpu().numpy(), cmap='gray')
        plt.subplot(3, 3, 9)
        plt.imshow(exp[0, 0, 80, :, :].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=0.9)
        plt.savefig('VFA_seg.png')
        plt.close()
        
            

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