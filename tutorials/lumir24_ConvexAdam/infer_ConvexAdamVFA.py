import os, gdown
import json
from torch.utils.data import DataLoader
import numpy as np
import torch
from natsort import natsorted
from MIR.models import convex_adam_vfa
import MIR.models.convexAdam.configs_ConvexAdam_MIND as CONFIGS_CVXAdam
import MIR.models.configs_VFA as CONFIGS_VFA
from MIR import ModelWeights, DatasetJSONs
import matplotlib
matplotlib.use('Agg')
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
import random
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
    val_dir = '/scratch2/jchen/DATA/LUMIR/'
    output_dir = 'LUMIR_ConvexAdam_ValPhase/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    '''
    Initialize model
    '''
    H, W, D = 160, 224, 192
    config = CONFIGS_CVXAdam.get_ConvexAdam_MIND_brain_default_config()
    vfa_config = CONFIGS_VFA.get_VFA_default_config()
    vfa_config.img_size = (H, W, D)
    print(config)
    print(vfa_config)
    
    if not os.path.isfile('LUMIR_dataset.json'):
        # download dataset json file
        file_id = DatasetJSONs['LUMIR24']
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, 'LUMIR_dataset.json', quiet=False)

    pretrained_dir = 'pretrained_wts/'
    pretrained_wts = 'VFA_LUMIR24.pth'
    if not os.path.isdir(pretrained_dir):
        os.makedirs(pretrained_dir)
    if not os.path.isfile(pretrained_dir + pretrained_wts):
        file_id = ModelWeights['VFA-LUMIR24-MonoModal']['wts']
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, pretrained_dir + pretrained_wts, quiet=False)
    vfa_weights_key = ModelWeights['VFA-LUMIR24-MonoModal']['wts_key']
    
    '''
    Initialize training
    '''
    val_set = L2RLUMIRJSONDataset(base_dir=val_dir, json_path='LUMIR_dataset.json', stage='validation')
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    val_files = val_set.imgs
    '''
    Validation
    '''
    for i, data in enumerate(val_loader):
        mv_id = val_files[i]['moving'].split('_')[-2]
        fx_id = val_files[i]['fixed'].split('_')[-2]
        
        data = [t.cuda() for t in data]
        x = data[0]
        y = data[1]
        flow = convex_adam_vfa(
            img_fixed=y,
            img_moving=x,
            convex_config=config,
            vfa_config=vfa_config,
            vfa_weights_path=pretrained_dir + pretrained_wts,
            vfa_weights_key=vfa_weights_key,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        )
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