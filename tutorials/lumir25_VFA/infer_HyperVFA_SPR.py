import os, gdown
import json
from torch.utils.data import DataLoader
import numpy as np
import torch
from natsort import natsorted
from MIR.models import VFA, HyperVFASPR
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

class L2RLUMIRJSONDataset(Dataset):
    def __init__(self, base_dir, json_path, stage='train'):
        with open(json_path) as f:
            d = json.load(f)
        if stage.lower()=='train':
            self.imgs = d['training']
        elif stage.lower()=='validation':
            self.imgs = d['validation']
        else:
            self.imgs = d['validation']
        self.base_dir = base_dir
        self.stage = stage
    
    def __getitem__(self, index):
        if self.stage == 'train':
            mov_dict = self.imgs[index]
            fix_dicts = self.imgs.copy()
            fix_dicts.remove(mov_dict)
            random.shuffle(fix_dicts)
            fix_dict = fix_dicts[0]
            x = nib.load(self.base_dir+mov_dict['image'])
            y = nib.load(self.base_dir+fix_dict['image'])

        else:
            img_dict = self.imgs[index]
            mov_path = img_dict['moving']
            fix_path = img_dict['fixed']
            x = nib.load(self.base_dir + mov_path)
            y = nib.load(self.base_dir + fix_path)
        x = x.get_fdata() / 255.
        y = y.get_fdata() / 255.
        x, y = x[None, ...], y[None, ...]
        x = np.ascontiguousarray(x)  # [channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x.float(), y.float()

    def __len__(self):
        return len(self.imgs)

def save_nii(img, file_name, pix_dim=[1., 1., 1.]):
    x_nib = nib.Nifti1Image(img, np.eye(4))
    x_nib.header.get_xyzt_units()
    x_nib.header['pixdim'][1:4] = pix_dim
    x_nib.to_filename('{}.nii.gz'.format(file_name))

def main():
    batch_size = 1
    val_dir = '/scratch/jchen/DATA/LUMIR/LUMIR25/'
    scale_factor = 1
    output_dir = 'LUMIR_HyperVFA_ValPhase/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    '''
    Initialize model
    '''
    H, W, D = 160, 224, 192
    config = CONFIGS_VFA.get_VFA_default_config()
    config.img_size = (H//scale_factor, W//scale_factor, D//scale_factor)
    print(config)
    model = HyperVFASPR(config, device='cuda:0')
    pretrained_dir = '/scratch/jchen/python_projects/LUMIR/experiments/HyperVFASPR/dsc0.7430.pth.tar'
    pretrained_mult = torch.load(pretrained_dir)['state_dict']
    pretrained_dir = '/scratch/jchen/python_projects/LUMIR/experiments/HyperVFASPR_Mono/dsc0.7717.pth.tar'
    pretrained_mono = torch.load(pretrained_dir)['state_dict']
    
    if not os.path.isfile('LUMIR25_dataset.json'):
        # download dataset json file
        file_id = DatasetJSONs['LUMIR25']
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, 'LUMIR25_dataset.json', quiet=False)

    #pretrained = torch.load(pretrained_dir+pretrained_wts)[ModelWeights['VFA-LUMIR24-MonoModal']['wts_key']]
    #model.load_state_dict(pretrained)
    #print('Pretrained Weights: {} loaded!'.format(pretrained_dir+pretrained_wts))
    model.cuda()
    
    '''
    Initialize training
    '''
    val_set = L2RLUMIRJSONDataset(base_dir=val_dir, json_path='LUMIR25_dataset.json', stage='test')
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    val_files = val_set.imgs
    '''
    Validation
    '''
    
    for i, data in enumerate(val_loader):
        model.eval()
        mv_id = val_files[i]['moving'].split('_')[-2]
        fx_id = val_files[i]['fixed'].split('_')[-2]
        mv_id = val_files[i]['moving'].split('_')[-2]
        mv_mod = val_files[i]['moving'].split('_')[-1]
        fx_id = val_files[i]['fixed'].split('_')[-2]
        fx_mod = val_files[i]['fixed'].split('_')[-1]
        if '0000' in mv_mod and '0000' in fx_mod:
            model.load_state_dict(pretrained_mono)
            model.cuda()
            print(f'Processing: disp_{fx_id}_{mv_id} Modality: {fx_mod} vs {mv_mod}. Using Mono-modal Pretrained Weights')
            alpha = 1
            lambda_wt = 0.2
        else:
            model.load_state_dict(pretrained_mult)
            model.cuda()
            print(f'Processing: disp_{fx_id}_{mv_id} Modality: {fx_mod} vs {mv_mod}. Using Multi-modal Pretrained Weights')
            alpha = 0.78
            lambda_wt = 0.14
        
        with torch.no_grad():
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            beta_code = torch.tensor([alpha], dtype=x.dtype, device=x.device).unsqueeze(dim=0)
            reg_code = torch.tensor([lambda_wt], dtype=x.dtype, device=x.device).unsqueeze(dim=0)
            hyper_code = torch.cat((beta_code, reg_code), -1)
            flow, wts = model((x, y), hyper_code)
        flow = fit_warp_to_svf(flow, nb_steps=7, iters=500, lr=0.1, output_type='svf')
        flow = flow.squeeze(0).detach().cpu().numpy()
        save_nii(flow, output_dir + 'disp_{}_{}'.format(fx_id, mv_id))
        print('disp_{}_{}.nii.gz saved to {}'.format(fx_id, mv_id, output_dir))

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