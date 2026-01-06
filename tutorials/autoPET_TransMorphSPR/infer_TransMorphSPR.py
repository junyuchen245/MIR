import glob
import os
from torch.utils.data import DataLoader
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from MIR.models import TransMorphTVFSPR, SpatialTransformer
import MIR.models.configs_TransMorph as configs_TransMorph
import nibabel as nib
from torch.utils.data import Dataset
from MIR.accuracy_measures import dice_val_VOI, dice_val_substruct, dice_val_all
from MIR.accuracy_measures import calc_J_i, calc_Jstar_1, calc_Jstar_2, calc_jac_dets, get_identity_grid, calc_measurements
from MIR.utils import Logger, AverageMeter
import torch.nn as nn
import torch.nn.functional as F
from MIR import ModelWeights
import gdown

class AutoPETTrainDataset(Dataset):
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

    def remap_lbl(self, lbl):
        #grouping_table = [[1,], [2, 3,], [4,], [5,], [6,], [7,], [8,], [9,], [10,], [11,], [12,], [13, 14,],
        #                  [15,], [16,], [17,], [18,], [19,], [20,], [21,], [22,], [23,], [24, 25, 26,], [27,],
        #                  [28, 29, 30, 31, 32,], [33, 34,], [35, 36,], [37, 38,], [39,]]
        grouping_table = [[1,], [2,], [3,], [4,], [5,], [6,], [7,], [8,], [9,], [10,], [11,], [12,], [13, 14,], [15, 16, 17,],
                  [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41], [42,],
                  [43,], [44,], [45,], [46,], [47,], [48,], [49,], [55,], [56,], [57,], [58, 59, 60, 61, 62, 63, 64, 65,
                                                                                         66, 67, 68, 69, 70, 71, 72, 73,
                                                                                         74, 75, 76, 77, 78, 79, 80, 81,],
                  [82, 84, 86,], [83, 85, 87,], [88, 90,], [89, 91], [92], [94, 96, 98,], [95, 97, 99,], [100,], [101],
                  [102,], [103,], [104,]]
        label_out = np.zeros_like(lbl)

        for idx, item in enumerate(grouping_table):
            for seg_i in item:
                label_out[lbl == seg_i] = idx + 1
        return label_out

    def __getitem__(self, index):
        mov_name = self.data_names[index]
        x = nib.load(self.path + '{}_CTRes.nii.gz'.format(mov_name))
        x = self.norm_img(x.get_fdata())
        x_suv = nib.load(self.path + '{}_SUV.nii.gz'.format(mov_name))
        x_suv = self.norm_suv(x_suv.get_fdata())
        x_seg = nib.load(self.path + '{}_CTRes_seg.nii.gz'.format(mov_name))#_segsimple#seg
        x_seg = x_seg.get_fdata()
        x_seg = self.remap_lbl(x_seg)#Affine reg does not have this.
        x = x[None, ...]
        x_suv = x_suv[None, ...]
        x_seg = x_seg[None, ...]
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        x_suv = np.ascontiguousarray(x_suv)  # [Bsize,channelsHeight,,Width,Depth]
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        x = torch.from_numpy(x)
        x_seg = torch.from_numpy(x_seg)
        x_suv = torch.from_numpy(x_suv)
        return x, x_suv, x_seg

    def __len__(self):
        return len(self.data_names)

def label_names():
    grouping_table1 = [[1,], [2,], [3,], [4,], [5,], [6,], [7,], [8,], [9,], [10,], [11,], [12,], [13, 14,], [15, 16, 17,],
                  [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41], [42,],
                  [43,], [44,], [45,], [46,], [47,], [48,], [49,], [55,], [56,], [57,], [58, 59, 60, 61, 62, 63, 64, 65,
                                                                                         66, 67, 68, 69, 70, 71, 72, 73,
                                                                                         74, 75, 76, 77, 78, 79, 80, 81,],
                  [82, 84, 86,], [83, 85, 87,], [88, 90,], [89, 91], [92], [94, 96, 98,], [95, 97, 99,], [100,], [101],
                  [102,], [103,], [104,]]
    grouping_table2 =  [[1,], [2, 3,], [4,], [5,], [6,], [7,], [8,], [9,], [10,], [11,], [12,], [13, 14,],
                          [15,], [16,], [17,], [18,], [19,], [20,], [21,], [22,], [23,], [24, 25, 26,], [27,],
                          [28, 29, 30, 31, 32,], [33, 34,], [35, 36,], [37, 38,], [39,]]
    organ_names = {0: "bkg", 1:"spleen", 2:"kidney_right", 3:"kidney_left", 4:"gallbladder", 5:"liver", 6:"stomach", 7:"aorta", 8:"inferior_vena_cava", 9:"portal_vein_and_splenic_vein", 10:"pancreas", 
                   11:"adrenal_gland_right", 12:"adrenal_gland_left", 13:"lung_left", 14:"lung_right", 15:"vertebrae", 16:"esophagus", 17:"trachea", 18:"heart_myocardium", 19:"heart_atrium_left", 20:"heart_ventricle_left",
                   21:"heart_atrium_right", 22:"heart_ventricle_right", 23:"pulmonary_artery", 24:"small_bowel", 25:"duodenum", 26:"colon", 27:"rib", 28:"humerus_scapula_clavicula_left", 29:"humerus_scapula_clavicula_right",
                   30:"femur_hip_left", 31:"femur_hip_right", 32:"sacrum", 33:"gluteus_left", 34:"gluteus_right", 35:"autochthon_left", 36:"autochthon_right", 37:"iliopsoas_left", 38:"iliopsoas_right", 39:"urinary_bladder"}
    organ_names2= {}
    return organ_names

def remap_lbl(lbl):
        grouping_table = [[1,], [2,], [3,], [4,], [5,], [6,], [7,], [8,], [9,], [10,], [11,], [12,], [13, 14,], [15, 16, 17,],
                  [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41], [42,],
                  [43,], [44,], [45,], [46,], [47,], [48,], [49,], [55,], [56,], [57,], [58, 59, 60, 61, 62, 63, 64, 65,
                                                                                         66, 67, 68, 69, 70, 71, 72, 73,
                                                                                         74, 75, 76, 77, 78, 79, 80, 81,],
                  [82, 84, 86,], [83, 85, 87,], [88, 90,], [89, 91], [92], [94, 96, 98,], [95, 97, 99,], [100,], [101],
                  [102,], [103,], [104,]]
        label_out = np.zeros_like(lbl)

        for idx, item in enumerate(grouping_table):
            for seg_i in item:
                label_out[lbl == seg_i] = idx + 1
        return label_out

def main():
    base_dir = '/scratch/jchen/python_projects/AutoPET/'
    test_dir = '/scratch2/jchen/DATA/AutoPET/affine_aligned/network/test/'
    model_folder = 'TransMorphSPR/'
    atlas_dir = 'TransMorphAtlas_MAE_1_MS_1_diffusion_1/'
    ct_atlas_dir = base_dir + 'atlas/ct/'+atlas_dir
    pt_atlas_dir = base_dir + 'atlas/pet/' + atlas_dir
    model_dir = base_dir + 'experiments/' + model_folder
    model_idx = -1
    if not os.path.exists('Quantitative_Results/'):
        os.makedirs('Quantitative_Results/')
    if os.path.exists('Quantitative_Results/'+model_folder[:-1]+'.csv'):
        os.remove('Quantitative_Results/'+model_folder[:-1]+'.csv')
    csv_writter(model_folder[:-1], 'Quantitative_Results/' + model_folder[:-1])
    dicts = label_names()
    line = 'pat_idx'
    for i in range(40):
        line = line + ',' + dicts[i]
    csv_writter(line +','+'ndv'+','+'ndp', 'Quantitative_Results/' + model_folder[:-1])

    '''
    Initialize model
    '''
    H, W, D = 192, 192, 256
    num_clus = 40
    config = configs_TransMorph.get_3DTransMorphAutoPET3Lvl_config()
    config.img_size = (H//2, W//2, D//2)
    config.window_size = (H // 32, W // 32, D // 32)
    config.out_chan = 3
    model = TransMorphTVFSPR(config, time_steps=5)
    model.cuda()
    
    pretrained_dir = 'pretrained_wts/'
    pretrained_wts = 'autoPET_TM_SPR_BETA.pth.tar'
    if not os.path.isdir("pretrained_wts/"):
        os.makedirs("pretrained_wts/")
    if not os.path.isfile(pretrained_dir+pretrained_wts):
        # download model
        file_id = ModelWeights['MedIA-TM-SPR-Beta-autoPET']['wts']
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, pretrained_dir+pretrained_wts, quiet=False)
    
    best_model = torch.load(pretrained_dir+pretrained_wts)[ModelWeights['MedIA-TM-SPR-Beta-autoPET']['wts_key']]
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    '''
    Initialize spatial transformation function
    '''
    reg_model = SpatialTransformer((H, W, D))
    reg_model.cuda()
    
    '''
    Initialize training
    '''
    test_names = glob.glob(test_dir + '*segsimple*')
    test_names = [name.split('/')[-1].split('_')[0] for name in test_names]
    test_set = AutoPETTrainDataset(test_dir, test_names)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    x_ct = nib.load(ct_atlas_dir + natsorted(os.listdir(ct_atlas_dir))[model_idx])
    x_ct = x_ct.get_fdata()[None, None, ...]
    x_ct = torch.from_numpy(x_ct).cuda().float()
    x_pt = nib.load(pt_atlas_dir + natsorted(os.listdir(pt_atlas_dir))[model_idx])
    x_pt = x_pt.get_fdata()[None, None, ...]
    x_pt = torch.from_numpy(x_pt).cuda().float()
    x_seg = nib.load(base_dir + 'seg_atlas_from_reg_rib_pp.nii.gz')
    x_seg = x_seg.get_fdata()[None, None, ...]
    x_seg = remap_lbl(x_seg)
    x_seg = torch.from_numpy(x_seg).cuda().float()
    x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=num_clus)
    x_seg_oh = torch.squeeze(x_seg_oh, 1)
    x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
    
    eval_dsc_def = AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_det = AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            model.eval()
            y = data[0].cuda().float()
            x_half = F.avg_pool3d(x_ct, 2).cuda()
            y_half = F.avg_pool3d(y, 2).cuda()
            y_suv = data[1].cuda().float()
            y_seg = data[2].cuda().float()
            #print(len(np.unique(y_seg.cpu().detach().numpy())), len(np.unique(x_seg.cpu().detach().numpy())))
            flow,_,_ = model((x_half, y_half))
            flow = nn.functional.interpolate(flow.cuda(), scale_factor=2, mode='trilinear', align_corners=False) * 2
            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=num_clus)
            x_seg_oh = torch.squeeze(x_seg_oh, 1)
            x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            x_segs = []
            for i in range(num_clus):
                def_seg = reg_model(x_seg_oh[:, i:i + 1, ...].float(), flow.float())
                x_segs.append(def_seg)
            x_segs = torch.cat(x_segs, dim=1)
            def_out = torch.argmax(x_segs, dim=1, keepdim=True)
            del x_segs, x_seg_oh
            
            mask = x_ct.detach().cpu().numpy()[0, 0, 1:-1, 1:-1, 1:-1]
            mask = mask > 0
            disp_field = flow.cpu().detach().numpy()[0]
            trans_ = disp_field + get_identity_grid(disp_field)
            jac_dets = calc_jac_dets(trans_)
            non_diff_voxels, non_diff_tetrahedra, non_diff_volume, non_diff_volume_map = calc_measurements(jac_dets, mask)
            total_voxels = np.sum(mask)
            ndv = non_diff_volume / total_voxels * 100
            ndp = non_diff_voxels / total_voxels * 100
            
            line = dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            line = line +','+str(ndv)+','+str(ndp)
            csv_writter(line, 'Quantitative_Results/' + model_folder[:-1])
            dsc_trans = dice_val_all(def_out.long(), y_seg.long(), num_clus)
            dsc_raw = dice_val_all(x_seg.long(), y_seg.long(), num_clus)
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x_ct.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x_ct.size(0))
            stdy_idx += 1

        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

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