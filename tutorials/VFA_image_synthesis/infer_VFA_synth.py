import os, gdown
import numpy as np
import torch
from natsort import natsorted
from MIR.models import VFA, SynthesisHead3DAdvanced
import MIR.models.configs_VFA as CONFIGS_VFA
import matplotlib
matplotlib.use('Agg')
import nibabel as nib
from MIR.utils import CenterCropPad3D, normalize_01
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
    pretrained_synth_head_wts = 'VFA_LUMIR25_synth_head.pth'
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
    model = SynthesisHead3DAdvanced(in_channels=config.start_channels*2)
    if not os.path.isfile(pretrained_dir + pretrained_synth_head_wts):
        # download model
        file_id = ModelWeights['VFA-SynthHead']['wts']
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, pretrained_dir + pretrained_synth_head_wts, quiet=False)
    pretrained = torch.load(pretrained_dir + pretrained_synth_head_wts)[ModelWeights['VFA-SynthHead']['wts_key']]
    model.load_state_dict(pretrained)
    model.cuda()
    
    '''
    Validation
    '''
    synth_steps = 2
    with torch.no_grad():
        exp = torch.from_numpy(nib.load(f'/scratch/jchen/DATA/Myelin/t2w_affine/sub-720320_t2w.nii.gz').get_fdata()[None, None, ...]).cuda().float()
        exp_real = torch.from_numpy(nib.load(f'/scratch/jchen/DATA/Myelin/t1w_affine/sub-720320_t1w.nii.gz').get_fdata()[None, None, ...]).cuda().float()
        exp = normalize_01(exp)
        exp_real = normalize_01(exp_real)
        print(exp.shape)
        vfa.eval()
        model.eval()
        exp_ = exp.clone()
        for _ in range(synth_steps):
            output = VFA_synth(exp_, model, vfa)
            exp_ = output
        plt.figure(figsize=(6,6), dpi=150)
        plt.subplot(3, 3, 1)
        plt.imshow(output[0, 0, :, :, 96].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=0.9)
        plt.subplot(3, 3, 2)
        plt.imshow(exp_real[0, 0, :, :, 96].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=0.9)
        plt.subplot(3, 3, 3)
        plt.imshow(exp[0, 0, :, :, 96].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=0.9)
        plt.subplot(3, 3, 4)
        plt.imshow(output[0, 0, :, 112, :].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=0.9)
        plt.subplot(3, 3, 5)
        plt.imshow(exp_real[0, 0, :, 112, :].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=0.9)
        plt.subplot(3, 3, 6)
        plt.imshow(exp[0, 0, :, 112, :].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=0.9)
        plt.subplot(3, 3, 7)
        plt.imshow(output[0, 0, 80, :, :].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=0.9)
        plt.subplot(3, 3, 8)
        plt.imshow(exp_real[0, 0, 80, :, :].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=0.9)
        plt.subplot(3, 3, 9)
        plt.imshow(exp[0, 0, 80, :, :].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=0.9)
        plt.savefig('VFA_synth.png')
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