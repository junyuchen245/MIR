import os, gdown
import json
from torch.utils.data import DataLoader
import numpy as np
import torch
from natsort import natsorted
from MIR.models import TransMorphTVF
from MIR import ModelWeights, DatasetJSONs
import MIR.models.configs_TransMorph as configs_TransMorph
import matplotlib

from MIR.pretrained_wts import ValEvalModules
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
    scale_factor = 2
    output_dir = 'LUMIR_TransMorph_ValPhase/'
    eval_dir = 'output_eval/'+output_dir
    os.makedirs(eval_dir, exist_ok=True)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    '''
    Initialize model
    '''
    H, W, D = 160, 224, 192
    config = configs_TransMorph.get_3DTransMorph3Lvl_config()
    config.img_size = (H//scale_factor, W//scale_factor, D//scale_factor)
    config.window_size = (H // 64, W // 64, D // 64)
    config.out_chan = 3
    print(config)
    model = TransMorphTVF(config, time_steps=7)
    pretrained_dir = 'pretrained_wts/'
    pretrained_wts = 'TransMorphTVF_LUMIR24.pth.tar'
    if not os.path.isdir("pretrained_wts/"):
        os.makedirs("pretrained_wts/")
    if not os.path.isfile(pretrained_dir+pretrained_wts):
        # download model
        file_id = ModelWeights['TransMorphTVF-LUMIR24-MonoModal']['wts']
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, pretrained_dir+pretrained_wts, quiet=False)
    
    if not os.path.isfile('LUMIR_dataset.json'):
        # download dataset json file
        file_id = DatasetJSONs['LUMIR24']
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, 'LUMIR_dataset.json', quiet=False)
        
    if not os.path.isfile('lumir24_eval'):
        file_id = ValEvalModules['LUMIR24']
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, 'lumir24_eval.zip', quiet=False)
        import zipfile
        with zipfile.ZipFile('lumir24_eval.zip', 'r') as zip:
            zip.extractall('./')
        os.remove('lumir24_eval.zip')
    os.system('chmod +x lumir24_evaluate')

    pretrained = torch.load(pretrained_dir+pretrained_wts)[ModelWeights['TransMorphTVF-LUMIR24-MonoModal']['wts_key']]
    model.load_state_dict(pretrained)
    print('Pretrained Weights: {} loaded!'.format(pretrained_dir+pretrained_wts))
    model.cuda()
    
    '''
    Initialize training
    '''
    val_set = L2RLUMIRJSONDataset(base_dir=val_dir, json_path=val_dir+'LUMIR_dataset.json', stage='test')
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    val_files = val_set.imgs
    '''
    Validation
    '''
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            model.eval()
            mv_id = val_files[i]['moving'].split('_')[-2]
            fx_id = val_files[i]['fixed'].split('_')[-2]
            
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_half = F.avg_pool3d(x, 2).cuda()
            y_half = F.avg_pool3d(y, 2).cuda()
            flow = model((x_half, y_half))
            flow = F.interpolate(flow.cuda(), scale_factor=2, mode='trilinear', align_corners=False) * 2
            flow = flow.squeeze(0).detach().cpu().numpy()
            save_nii(flow, output_dir + 'disp_{}_{}'.format(fx_id, mv_id))
            print('disp_{}_{}.nii.gz saved to {}'.format(fx_id, mv_id, output_dir))

    os.system(f'./lumir24_evaluate --input-path {output_dir} --output-path {eval_dir}')

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