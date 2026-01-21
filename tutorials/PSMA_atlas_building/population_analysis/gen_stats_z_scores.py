from torch.utils.tensorboard import SummaryWriter
import os, glob, re, gc
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
from MIR.models import VFASPR, SpatialTransformer, VFA, TemplateCreation
import MIR.models.configs_VFA as CONFIGS_VFA
from MIR.image_similarity import SSIM3D
from MIR.deformation_regularizer import Grad3d
from MIR.utils import Logger, AverageMeter
import nibabel as nib
import utils
import json
from datetime import datetime, timezone
import torch.nn.functional as F
from MIR.statistical_analysis import GLM, winsorize, janmahasatian_lbm, robust_bp_ref
import MIR.label_reference as lbl_ref
import math
import os, json, math, re
import numpy as np
import nibabel as nib
import torch
from dataset import JHUPSMADataset
from datetime import datetime
import utils
from utils import gaussian_kernel3d, smooth_inside_mask, morph_close3d, save_vector_as_vol, save_vol
from cox_analysis import benjamini_hochberg, build_clinical_matrix, cox_fit, concordance_index, orthogonalize_against, plot_km_three_groups, plot_km_two_groups
import csv

def main():
    batch_size = 1
    _scale = 0.777047
    train_dir = '/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/JHU/'
    val_dir = '/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/JHU/'
    save_dir = 'VFAAtlas_SSIM_1_MS_1_diffusion_1/'
    model_dir = '../experiments/' + save_dir
    population_stats_dir = 'population_stats_feats/'
    os.makedirs(population_stats_dir, exist_ok=True)
    # ---- toggles / options ----
    USE_robust_atlas = True
    '''
    Initialize model
    '''
    H, W, D = 192, 192, 256
    scale_factor=1
    config = CONFIGS_VFA.get_VFA_default_config()
    config.img_size = (H//scale_factor, W//scale_factor, D//scale_factor)
    print(config)
    model = VFA(config, device='cuda:0', SVF=True, return_full=True).cuda()
    template_model = TemplateCreation(model, (H, W, D))
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[0])['state_dict']
    template_model.load_state_dict(best_model)
    model = template_model.reg_model
    model.cuda()

    ct_atlas_dir = '../atlas/ct/VFAAtlas_SSIM_1_MS_1_diffusion_1/'
    pt_atlas_dir = '../atlas/suv/VFAAtlas_SSIM_1_MS_1_diffusion_1/'
    seg_atlas_path = '../atlas/seg/suv_seg_atlas_w_reg_14lbls.nii.gz'
    ct_seg_atlas_path = '../atlas/seg/ct_seg_atlas_w_reg_40lbls.nii.gz'
    ct_full_atlas_path = '../atlas/seg/ct_seg_atlas_w_reg_118lbls.nii.gz'
    x_ct = nib.load(ct_atlas_dir + natsorted(os.listdir(ct_atlas_dir))[-1])
    x_ct = x_ct.get_fdata()[None, None, ...]
    x_ct = torch.from_numpy(x_ct).cuda().float()
    x_pt = nib.load(pt_atlas_dir + natsorted(os.listdir(pt_atlas_dir))[-1])
    x_pt_nib_aff = x_pt.affine
    x_pt = x_pt.get_fdata()[None, None, ...]
    x_pt = torch.from_numpy(x_pt).cuda().float()
    x_pt_seg = nib.load(seg_atlas_path)
    x_pt_seg = x_pt_seg.get_fdata()[None, None, ...]
    x_pt_seg = torch.from_numpy(x_pt_seg).cuda().long()
    liver_mask = (x_pt_seg==1).float().repeat(2, 1, 1, 1, 1)
    
    x_ct_seg = nib.load(ct_seg_atlas_path)
    x_ct_seg = x_ct_seg.get_fdata()[None, None, ...]
    x_ct_seg = torch.from_numpy(x_ct_seg).cuda().long()
    aorta_mask = (x_ct_seg==7).float().repeat(2, 1, 1, 1, 1)
    
    x_ct_full_seg = nib.load(ct_full_atlas_path)
    x_ct_full_seg = x_ct_full_seg.get_fdata()[None, None, ...]
    x_ct_full_seg = lbl_ref.remap_totalsegmentator_lbls(x_ct_full_seg, total_seg_version='v2_biological_meaningful', label_scheme='v2_biological_meaningful')
    #print(np.unique(x_ct_full_seg))
    #sys.exit()
    x_ct_full_seg = torch.from_numpy(x_ct_full_seg).cuda().long()
    
    
    '''
    Initialize training
    '''
    mask_3d = ((x_ct_seg[0,0] > 0.01) | (x_ct[0,0] > 0.01)).bool()
    # Optional: close small holes inside the mask to avoid edge artifacts
    CLOSE_MASK_RADIUS = 2  # set to 0 to disable; typical 1-3 vox
    if CLOSE_MASK_RADIUS and CLOSE_MASK_RADIUS > 0:
        mask_3d = morph_close3d(mask_3d, radius=CLOSE_MASK_RADIUS)
    # Optional: save the atlas mask for QA
    # nib.save(nib.Nifti1Image(mask_3d.cpu().detach().float().numpy(), np.eye(4)), 'temp_mask.nii.gz')
    # Build list of one scan per patient:
    # - If a patient has multiple scans, keep the earliest (baseline)
    # - If a patient has only one scan, keep that single scan
    full_paths = glob.glob(os.path.join(train_dir, '*_CT_seg*'))
    bases = [os.path.basename(p).split('_CT_seg')[0] for p in full_paths]

    # Parse patid and date (YYYY-MM-DD) via regex and group by patid
    pid_to_dates = {}
    for b in bases:
        m = re.match(r'^(?P<pid>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})$', b)
        if not m:
            continue
        pid = m.group('pid')
        date = m.group('date')
        pid_to_dates.setdefault(pid, set()).add(date)

    # Select exactly one base name per patient: baseline (min date) or the single date
    selected = []
    for pid, dates in pid_to_dates.items():
        if not dates:
            continue
        bl_date = sorted(dates)[0]
        selected.append(f"{pid}_{bl_date}")

    full_names = natsorted(selected)
    full_names = [nm for nm in full_names if 'PSMA' not in nm]
    #print(full_names, len(full_names))
    #sys.exit(0)
    full_names = natsorted(full_names)
    val_names = full_names
    val_set = JHUPSMADataset(val_dir, val_names)
    # Custom collate to handle non-tensor fields (e.g., strings/None) safely
    from torch.utils.data._utils.collate import default_collate as _default_collate
    def _safe_collate(batch):
        if not isinstance(batch, list) or len(batch) == 0:
            return batch
        elem0 = batch[0]
        if not isinstance(elem0, dict):
            return _default_collate(batch)
        keep_as_list = {'SubjectID', 'scan_date', 'r1_new_lesion_location', 'r2_new_lesion_location'}
        keep_first = {'covname2idx'}
        out = {}
        for k in elem0.keys():
            vals = [b[k] for b in batch]
            if k in keep_as_list:
                out[k] = [("" if (v is None) else v) for v in vals]
            elif k in keep_first:
                out[k] = elem0[k]
            else:
                try:
                    out[k] = _default_collate(vals)
                except TypeError:
                    out[k] = vals
        return out
    # Use batch_size=1 because we only need baseline (or the single scan) per patient
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False, collate_fn=_safe_collate)

    lesion_stats = utils.RunningStats3D(shape=(H, W, D), device='cuda:0', dtype=torch.float32)
    '''
    Validation
    '''
    # Collect patients (each batch has a single scan: baseline or the only scan)
    patient_ids = []       # to align optional VPS by SubjectID
    # Build robust atlas using per-voxel median and MAD-based std (no mask)
    if (not os.path.isfile(population_stats_dir + 'sulr_atlas_median.nii.gz') or
        not os.path.isfile(population_stats_dir + 'sulr_atlas_madstd.nii.gz')):
        idx = 0
        sulr_stack = []  # store per-patient normalized SUL volumes
        with torch.no_grad():
            for data in val_loader:
                idx += 1
                print(f'Prognostic pass, patient {idx}, {data["SubjectID"]}')
                # capture patient id in order for optional VPS alignment
                try:
                    sid = data['SubjectID'][0] if isinstance(data['SubjectID'], (list, tuple)) else str(data['SubjectID'])
                except Exception:
                    sid = f'case_{idx}'
                patient_ids.append(str(sid))

                model.eval()
                pat_ct = data['CT'].cuda().float()          # [B,1,H,W,D], B=1 here
                pat_suv_org = data['SUV_Org'].cuda().float()
                cov_raw = data['covariates_raw'].cpu().float()  # [B, P], B=1 here
                cov_names = val_set.covariate_names
                # forward: warp scan(s) into atlas (B can be 1)
                B = pat_ct.shape[0]
                def_atlas, def_image, pos_flow, neg_flow = model((x_ct.repeat(B,1,1,1,1), pat_ct))
                def_pat_suv = model.spatial_trans(pat_suv_org, neg_flow)  # [B,1,H,W,D]
                suv_bl = def_pat_suv[0, 0]  # use baseline (or the only) scan

                # blood pool normalization
                Rb = robust_bp_ref(suv_bl, liver_mask[0,0])
                if not torch.isfinite(Rb): Rb = torch.tensor(1.0, device=suv_bl.device)

                # LBM scaling if available
                def get_idx(nm): return cov_names.index(nm) if nm in cov_names else None
                try:
                    ix_h = cov_names.index('height_m'); ix_w = cov_names.index('weight_kg')
                    h0 = float(cov_raw[0, ix_h].item()); w0 = float(cov_raw[0, ix_w].item())
                    if math.isfinite(h0) and math.isfinite(w0) and w0 > 0 and h0 > 0:
                        lbm0 = 9270.0 * w0 / (6680.0 + 216.0 * w0 / (h0*h0))
                        sul_bl = suv_bl * (lbm0 / (w0 + 1e-8))
                    else:
                        sul_bl = suv_bl
                except Exception:
                    sul_bl = suv_bl

                sulr_bl = sul_bl / (Rb + 1e-6)
                # store numpy volume [H,W,D]; sulr_bl is already [H,W,D], no extra indexing
                sulr_stack.append(sulr_bl.detach().cpu().numpy())
        if idx == 0:
            raise RuntimeError('No patients found to build robust atlas stats.')
        sulr_arr = np.stack(sulr_stack, axis=0)  # [N,H,W,D] when N>1; [H,W,D] when N==1
        if sulr_arr.ndim == 3:
            sulr_arr = np.expand_dims(sulr_arr, axis=0)
        if USE_robust_atlas:
            # per-voxel median across patients
            sulr_atlas_median = np.median(sulr_arr, axis=0).astype(np.float32)
            # per-voxel MAD across patients
            mad_vol = np.median(np.abs(sulr_arr - sulr_atlas_median[None, ...]), axis=0).astype(np.float32)
            sulr_atlas_madstd = np.maximum(1.4826 * mad_vol, 1e-6).astype(np.float32)
            # save robust atlas
            save_vol(sulr_atlas_median, population_stats_dir+'sulr_atlas_median.nii.gz', x_pt_nib_aff)
            save_vol(sulr_atlas_madstd, population_stats_dir+'sulr_atlas_madstd.nii.gz', x_pt_nib_aff)
            sulr_bl_med = torch.from_numpy(sulr_atlas_median).cuda().float()
            sulr_bl_madstd = torch.from_numpy(sulr_atlas_madstd).cuda().float()
        else:
            # per-voxel mean/std across patients
            sulr_atlas_mean = np.mean(sulr_arr, axis=0).astype(np.float32)
            std_vol = np.std(sulr_arr, axis=0).astype(np.float32)
            sulr_atlas_std = np.maximum(std_vol, 1e-6).astype(np.float32)
            # save mean/std atlas
            save_vol(sulr_atlas_mean, population_stats_dir+'sulr_atlas_mean.nii.gz', x_pt_nib_aff)
            save_vol(sulr_atlas_std, population_stats_dir+'sulr_atlas_std.nii.gz', x_pt_nib_aff)
            sulr_bl_med = torch.from_numpy(sulr_atlas_mean).cuda().float()
            sulr_bl_madstd = torch.from_numpy(sulr_atlas_std).cuda().float()
    else:
        if USE_robust_atlas:
            sulr_bl_med_nib = nib.load(population_stats_dir+'sulr_atlas_median.nii.gz')
            sulr_bl_med = sulr_bl_med_nib.get_fdata().astype(np.float32)
            sulr_bl_med = torch.from_numpy(sulr_bl_med).cuda().float()
            sulr_bl_madstd_nib = nib.load(population_stats_dir+'sulr_atlas_madstd.nii.gz')
            sulr_bl_madstd = sulr_bl_madstd_nib.get_fdata().astype(np.float32)
            sulr_bl_madstd = torch.from_numpy(np.maximum(sulr_bl_madstd, 1e-6)).cuda().float()
        else:
            sulr_bl_med_nib = nib.load(population_stats_dir+'sulr_atlas_mean.nii.gz')
            sulr_bl_med = sulr_bl_med_nib.get_fdata().astype(np.float32)
            sulr_bl_med = torch.from_numpy(sulr_bl_med).cuda().float()
            sulr_bl_madstd_nib = nib.load(population_stats_dir+'sulr_atlas_std.nii.gz')
            sulr_bl_madstd = sulr_bl_madstd_nib.get_fdata().astype(np.float32)
            sulr_bl_madstd = torch.from_numpy(np.maximum(sulr_bl_madstd, 1e-6)).cuda().float()
    
    spatial_trans_nn = SpatialTransformer((H, W, D), mode='nearest').cuda()
    # compute z-scores
    idx = 0
    sulr_bl_all = 0
    with torch.no_grad():
        for data in val_loader:
            idx += 1
            if idx < 20:
                continue
            print(f'Prognostic pass, patient {idx}, {data["SubjectID"]}')
            # capture patient id in order for optional VPS alignment
            try:
                sid = data['SubjectID'][0] if isinstance(data['SubjectID'], (list, tuple)) else str(data['SubjectID'])
            except Exception:
                sid = f'case_{idx}'
            patient_ids.append(str(sid))

            model.eval()
            pat_ct = data['CT'].cuda().float()          # [B,1,H,W,D], B=1 here
            pat_suv_org = data['SUV_Org'].cuda().float()
            pat_ct_seg = data['CT_seg'].cuda().long()
            
            y_seg = data['SUV_seg'].cuda().long()
            print(torch.sum(y_seg==14).item())
            if torch.sum(y_seg==14).item() > 1000:
                print('yes')
            else:
                print('no')
                continue
            
            cov_raw = data['covariates_raw'].cpu().float()  # [B, P], B=1 here
            cov_names = val_set.covariate_names
            # forward: warp scan(s) into atlas (B can be 1)
            B = pat_ct.shape[0]
            def_atlas, def_image, pos_flow, neg_flow = model((x_ct.repeat(B,1,1,1,1), pat_ct))
            def_pat_lbl = spatial_trans_nn(y_seg.float(), neg_flow)[0, 0]
            def_pat_ctlbl = spatial_trans_nn(pat_ct_seg.float(), neg_flow)[0, 0]
            def_pat_suv = model.spatial_trans(pat_suv_org, neg_flow)  # [B,1,H,W,D]
            suv_bl = def_pat_suv[0, 0]  # use baseline (or the only) scan

            # blood pool normalization
            Rb = robust_bp_ref(suv_bl, liver_mask[0,0])
            if not torch.isfinite(Rb): Rb = torch.tensor(1.0, device=suv_bl.device)

            # LBM scaling if available
            def get_idx(nm): return cov_names.index(nm) if nm in cov_names else None
            try:
                ix_h = cov_names.index('height_m'); ix_w = cov_names.index('weight_kg')
                h0 = float(cov_raw[0, ix_h].item()); w0 = float(cov_raw[0, ix_w].item())
                if math.isfinite(h0) and math.isfinite(w0) and w0 > 0 and h0 > 0:
                    lbm0 = 9270.0 * w0 / (6680.0 + 216.0 * w0 / (h0*h0))
                    sul_bl = suv_bl * (lbm0 / (w0 + 1e-8))
                else:
                    sul_bl = suv_bl
            except Exception:
                sul_bl = suv_bl
            sulr_bl = sul_bl / (Rb + 1e-6)
            # Per-voxel robust z-score using atlas median and MAD-based std
            zscore = (sulr_bl - sulr_bl_med) / (sulr_bl_madstd + 1e-6)
            save_vol(zscore.detach().cpu().numpy()*mask_3d.detach().cpu().numpy(), population_stats_dir+'zscore_map_pat_{}.nii.gz'.format(sid), x_pt_nib_aff)
            save_vol(sulr_bl.detach().cpu().numpy()*mask_3d.detach().cpu().numpy(), population_stats_dir+'spa_norm_pat_{}.nii.gz'.format(sid), x_pt_nib_aff)
            save_vol(pat_suv_org.detach().cpu().numpy(), population_stats_dir+'suv_pat_{}.nii.gz'.format(sid), x_pt_nib_aff)
            save_vol(def_pat_ctlbl.detach().cpu().numpy(), population_stats_dir+'seg_ct_warped_pat_{}.nii.gz'.format(sid), x_pt_nib_aff)
            save_vol(def_pat_lbl.detach().cpu().numpy(), population_stats_dir+'seg_warped_pat_{}.nii.gz'.format(sid), x_pt_nib_aff)
            sys.exit(0)


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