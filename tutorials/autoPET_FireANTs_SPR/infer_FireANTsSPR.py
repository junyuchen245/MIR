import glob
import os
from torch.utils.data import DataLoader
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from MIR.models.fireants.io.image import Image, BatchedImages
from MIR.models.fireants.registration.affine import AffineRegistration
from MIR.models.fireants.registration.greedy_SPR import GreedyRegistration
from MIR.models.fireants.registration.syn_SPR import SyNRegistration
from MIR.models import SpatialTransformer
import nibabel as nib
from torch.utils.data import Dataset
from MIR.accuracy_measures import dice_val_VOI
from MIR.accuracy_measures import calc_J_i, calc_Jstar_1, calc_Jstar_2, calc_jac_dets, get_identity_grid, calc_measurements
from MIR.utils import Logger, AverageMeter
import torch.nn as nn
import torch.nn.functional as F
from MIR import ModelWeights
import gdown
import SimpleITK as sitk
from MIR.models.fireants.io.image import Image

def torch_to_fireants_image(tensor, spacing=(1,1,1), origin=(0,0,0), direction=None, device='cuda'):
    vol = tensor.detach().cpu().squeeze().numpy().astype(np.float32)
    itk = sitk.GetImageFromArray(vol)  # assumes array is [Z,Y,X]
    itk.SetSpacing(spacing)
    itk.SetOrigin(origin)
    if direction is not None:
        itk.SetDirection(np.array(direction).reshape(len(spacing), len(spacing)).ravel())
    return Image(itk, device=device)

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
        grouping_table = [[1,], [2, 3,], [4,], [5,], [6,], [7,], [8,], [9,], [10,], [11,], [12,], [13, 14,],
                          [15,], [16,], [17,], [18,], [19,], [20,], [21,], [22,], [23,], [24, 25, 26,], [27,],
                          [28, 29, 30, 31, 32,], [33, 34,], [35, 36,], [37, 38,], [39,]]
        #grouping_table = [[1,], [2,], [3,], [4,], [5,], [6,], [7,], [8,], [9,], [10,], [11,], [12,], [13, 14,], [15, 16, 17,],
        #          [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41], [42,],
        #          [43,], [44,], [45,], [46,], [47,], [48,], [49,], [55,], [56,], [57,], [58, 59, 60, 61, 62, 63, 64, 65,
        #                                                                                 66, 67, 68, 69, 70, 71, 72, 73,
        #                                                                                 74, 75, 76, 77, 78, 79, 80, 81,],
        #          [82, 84, 86,], [83, 85, 87,], [88, 90,], [89, 91], [92], [94, 96, 98,], [95, 97, 99,], [100,], [101],
        #          [102,], [103,], [104,]]
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
        x_seg = nib.load(self.path + '{}_CTRes_segsimple.nii.gz'.format(mov_name))#_segsimple#seg
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
    test_dir = '/scratch/jchen/DATA/AutoPET/affine_aligned/network/test/'
    atlas_dir = '/scratch/jchen/DATA/AutoPET/'
    ct_atlas_dir = '/scratch/jchen/DATA/AutoPET/CT_atlas.nii.gz'
    pt_atlas_dir = '/scratch/jchen/DATA/AutoPET/PET_atlas.nii.gz'
    FireANTs_option = 'syn_SPR'
    

    '''
    Initialize model
    '''
    H, W, D = 192, 192, 256
    num_clus = 29
    
    '''
    Initialize spatial transformation function
    '''
    reg_model = SpatialTransformer((H, W, D))
    reg_model.cuda()
    
    '''
    Initialize training
    '''
    # Generate the optimizers.    
    
    test_names = glob.glob(test_dir + '*segsimple*')
    test_names = [name.split('/')[-1].split('_')[0] for name in test_names]
    test_set = AutoPETTrainDataset(test_dir, test_names)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    H, W, D = 192, 192, 256
    num_clus = 29
    x = nib.load(ct_atlas_dir)
    x = x.get_fdata()[None, None, ...]
    x = torch.from_numpy(x).cuda().float()
    x_suv = nib.load(pt_atlas_dir)
    x_suv = x_suv.get_fdata()[None, None, ...]
    x_suv = torch.from_numpy(x_suv).cuda().float()
    x_seg = nib.load(atlas_dir+'seg_simple_atlas_from_reg.nii.gz')
    x_seg = x_seg.get_fdata()[None, None, ...]
    x_seg = torch.from_numpy(x_seg).cuda().float()
    x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=num_clus)
    x_seg_oh = torch.squeeze(x_seg_oh, 1)
    x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
    
    eval_dsc_def = AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_det = AverageMeter()
    stdy_idx = 0
    for data in test_loader:
        data = [t.cuda() for t in data]
        y = data[0].cuda().float()
        y_suv = data[1].cuda().float()
        y_seg = data[2].cuda().float()
    
        moving_image = torch_to_fireants_image(x, device='cuda')
        fixed_image = torch_to_fireants_image(y, device='cuda')
        
        fixed_image, moving_image = BatchedImages(fixed_image), BatchedImages(moving_image)
        
        if FireANTs_option == "greedy_SPR":
            deformable = GreedyRegistration([4, 2, 1], [250, 200, 200],
                                fixed_image, moving_image, deformation_type='compositive',
                                optimizer='adam', optimizer_lr=0.5, cc_kernel_size=7,
                                smooth_grad_sigma=1,
                                smooth_warp_sigma=0.5, spr_reg=True, spr_lambda=1, spr_logbeta_weight=0.01, spr_lr=0.05)
        elif FireANTs_option == "syn_SPR":
            deformable = SyNRegistration([4, 2, 1], [250, 200, 100],
                                            fixed_image, moving_image, deformation_type='compositive',
                                            optimizer='adam', optimizer_lr=0.1, cc_kernel_size=5,   # 2k + 1
                                            smooth_grad_sigma=0.5,
                                            smooth_warp_sigma=0.25, spr_reg=True, spr_logbeta_weight=0.01, spr_lr=0.05)
        else:
            raise NotImplementedError('Registration model not implemented')
        
        deformable.optimize()
        warp = deformable.get_warped_coordinates(fixed_image, moving_image)
        spatial_weight = deformable.spr_wt_param
        spatial_weight = (spatial_weight-spatial_weight.min())/(spatial_weight.max()-spatial_weight.min()+1e-6)
        
        flow = flow2disp(warp, size=(H, W, D))
        def_x_ct = reg_model(x.float(), flow.float())
        
        plt.figure(figsize=(12,4), dpi=150)
        plt.subplot(1, 3, 1)
        plt.imshow(def_x_ct[0, 0, :, :, D//2].cpu().detach().numpy(), cmap='gray')
        plt.title('Deformed CT Slice')
        plt.subplot(1, 3, 2)
        plt.imshow(y[0, 0, :, :, D//2].cpu().detach().numpy(), cmap='gray')
        plt.title('Fixed CT Slice')
        plt.subplot(1, 3, 3)
        plt.imshow(spatial_weight[0, 0, :, :, D//2].cpu().detach().numpy(), cmap='jet')
        plt.colorbar()
        plt.title('Spatial Weight Map Slice')
        plt.savefig('tmp_def_ct.png')
        plt.close()
        
        x_segs = []
        for i in range(num_clus):
            def_seg = reg_model(x_seg_oh[:, i:i + 1, ...].float(), flow.float())
            x_segs.append(def_seg)
        x_segs = torch.cat(x_segs, dim=1)
        def_out = torch.argmax(x_segs, dim=1, keepdim=True)
        
        mask = x.detach().cpu().numpy()[0, 0, 1:-1, 1:-1, 1:-1]
        mask = mask > 0
        disp_field = flow.cpu().detach().numpy()[0]
        trans_ = disp_field + get_identity_grid(disp_field)
        jac_dets = calc_jac_dets(trans_)
        non_diff_voxels, non_diff_tetrahedra, non_diff_volume, non_diff_volume_map = calc_measurements(jac_dets, mask)
        total_voxels = np.sum(mask)
        ndv = non_diff_volume / total_voxels * 100
        ndp = non_diff_voxels / total_voxels * 100
        
        dsc_trans = dice_val_VOI(def_out.long(), y_seg.long(), num_clus)
        dsc_raw = dice_val_VOI(x_seg.long(), y_seg.long(), num_clus)
        print('Trans dsc: {:.4f}, Raw dsc: {:.4f}, ndv: {:.4f}, ndp: {:.4f}'.format(dsc_trans.item(),dsc_raw.item(), ndv, ndp))
        eval_dsc_def.update(dsc_trans.item(), x.size(0))
        eval_dsc_raw.update(dsc_raw.item(), x.size(0))
        stdy_idx += 1

def flow2disp(flow, size=(160, 224, 192)): 
    vectors = [torch.arange(0, s) for s in size]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0)
    grid = grid.type(torch.FloatTensor).cuda()
    
    flow = flow[..., [2, 1, 0]].permute(0, 4, 1, 2, 3)
    shape = flow.shape[2:]
    for i in range(len(shape)):
        flow_ = flow[:, i, ...]
        flow[:, i, ...] = (flow_/2+0.5)*(shape[i] - 1)
    return flow-grid

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