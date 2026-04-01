from torch.utils.tensorboard import SummaryWriter
import os, glob, re
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

def gaussian_kernel3d(sigma_vox, size_vox):
    ax = [torch.arange(-(s//2), s//2+1, dtype=torch.float32) for s in size_vox]
    zz, yy, xx = torch.meshgrid(ax[0], ax[1], ax[2], indexing='ij')
    g = torch.exp(-(xx**2+yy**2+zz**2)/(2*sigma_vox**2))
    g = g / g.sum()
    return g

class JHUPSMADataset(Dataset):
    def __init__(self, data_path, data_names, data_json='/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/JHU_dicom_info/', normalize_covariates=True):
        self.path = data_path
        self.data_names = data_names
        self.data_json = data_json
        self.normalize_covariates = normalize_covariates

        # --- Build covariate names: continuous + one-hot categorical (vendor) ---
        self.covariate_names = [
            'age', 'height', 'weight', 'dose', 'rel_time_days',
            'vendor_siemens', 'vendor_ge', 'vendor_philips', 'vendor_other'
        ]
        self._cont_idx = np.array([0, 1, 2, 3, 4], dtype=int)

        # Precompute raw covariates for all subjects
        cov_rows = []
        # --- helpers for robust parsing ---
        def _safe_float(v, default=0.0):
            try:
                return float(v)
            except Exception:
                return float(default)
        def _extract_dose(sj):
            seq = sj.get('RadiopharmaceuticalInformationSequence', None)
            if isinstance(seq, list) and len(seq) > 0 and isinstance(seq[0], dict):
                raw = seq[0].get('RadionuclideTotalDose', 0.0)
            elif isinstance(seq, dict):
                raw = seq.get('RadionuclideTotalDose', 0.0)
            else:
                raw = 0.0
            return _safe_float(raw, 0.0)
        def _parse_study_ts(sj):
            date_raw = str(sj.get('StudyDate', ''))
            time_raw = str(sj.get('StudyTime', ''))
            date_str = date_raw.replace('-', '')
            if len(date_str) >= 8 and date_str[:8].isdigit():
                yyyy, mm, dd = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
            else:
                yyyy, mm, dd = 1970, 1, 1
            time_digits = ''.join(ch for ch in time_raw if ch.isdigit())
            HH = int(time_digits[0:2]) if len(time_digits) >= 2 else 0
            MM = int(time_digits[2:4]) if len(time_digits) >= 4 else 0
            SS = int(time_digits[4:6]) if len(time_digits) >= 6 else 0
            dt_utc = datetime(yyyy, mm, dd, HH, MM, SS, tzinfo=timezone.utc)
            return float(dt_utc.timestamp())

        # --- first pass: compute study_ts and patient baselines ---
        study_ts_map = {}
        pid_to_ts = {}
        for name in self.data_names:
            sj = json.load(open(os.path.join(self.data_json, f'{name}.json'), 'r'))
            ts = _parse_study_ts(sj)
            study_ts_map[name] = ts
            pid = name.split('_')[0]
            pid_to_ts.setdefault(pid, []).append(ts)
        pid_baseline = {pid: min(ts_list) for pid, ts_list in pid_to_ts.items()}

        # --- second pass: build covariate rows using relative time (days since first study per patient) ---
        for name in self.data_names:
            sj = json.load(open(os.path.join(self.data_json, f'{name}.json'), 'r'))
            age = _safe_float(sj.get('PatientAge', '0').rstrip('Yy'))
            height = _safe_float(sj.get('PatientSize', 0.0))
            weight = _safe_float(sj.get('PatientWeight', 0.0))
            dose = _extract_dose(sj)
            vendor = str(sj.get('Manufacturer', 'unknown')).lower()
            if 'siemens' in vendor:
                vid = 0
            else:
                vid = 1
            vendor_oh = np.zeros(2, dtype=np.float32)
            vendor_oh[vid] = 1.0

            pid = name.split('_')[0]
            ts = study_ts_map.get(name, 0.0)
            base = pid_baseline.get(pid, ts)
            rel_time_days = max(0.0, (ts - base) / 86400.0)

            cov = np.array([age, height, weight, dose, rel_time_days], dtype=np.float32)
            cov = np.concatenate([cov, vendor_oh], axis=0)
            cov_rows.append(cov)
        self.covariates_raw = np.stack(cov_rows, axis=0) if len(cov_rows) > 0 else np.zeros((0, len(self.covariate_names)), dtype=np.float32)

        # Normalize continuous covariates (z-score), keep one-hot as-is
        if self.normalize_covariates and self.covariates_raw.shape[0] > 0:
            self.cov_mean = self.covariates_raw[:, self._cont_idx].mean(axis=0)
            self.cov_std = self.covariates_raw[:, self._cont_idx].std(axis=0)
            self.cov_std[self.cov_std < 1e-6] = 1.0
            cov_norm = self.covariates_raw.copy()
            cov_norm[:, self._cont_idx] = (cov_norm[:, self._cont_idx] - self.cov_mean) / self.cov_std
            self.covariates = cov_norm.astype(np.float32)
        else:
            self.cov_mean = np.zeros(len(self._cont_idx), dtype=np.float32)
            self.cov_std = np.ones(len(self._cont_idx), dtype=np.float32)
            self.covariates = self.covariates_raw.astype(np.float32)

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
        x_ = x.get_fdata()
        x = self.norm_img(x_)
        x_suv = nib.load(self.path + '{}_SUV.nii.gz'.format(mov_name))
        x_suv_ = x_suv.get_fdata()
        x_suv = self.norm_suv(x_suv_)
        x_seg = nib.load(self.path + '{}_CT_seg.nii.gz'.format(mov_name))
        x_seg = utils.remap_totalseg_lbl(x_seg.get_fdata())
        x_suv_seg = nib.load(self.path + '{}_SUV_seg.nii.gz'.format(mov_name))
        x_suv_seg = utils.remap_suv_lbl(x_suv_seg.get_fdata(), include_lesions=True)
        x, x_ = x[None, ...], x_[None, ...]
        x_suv, x_suv_ = x_suv[None, ...], x_suv_[None, ...]
        x_seg = x_seg[None, ...]
        x_suv_seg = x_suv_seg[None, ...]
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        x_ = np.ascontiguousarray(x_)
        x_suv = np.ascontiguousarray(x_suv)  # [Bsize,channelsHeight,,Width,Depth]
        x_suv_ = np.ascontiguousarray(x_suv_)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        x_suv_seg = np.ascontiguousarray(x_suv_seg)  # [Bsize,channelsHeight,,Width,Depth]
        x_ = torch.from_numpy(x_)
        x_suv_ = torch.from_numpy(x_suv_)
        x = torch.from_numpy(x)
        x_suv = torch.from_numpy(x_suv)
        x_seg = torch.from_numpy(x_seg)
        x_suv_seg = torch.from_numpy(x_suv_seg)
        '''
        subject_json = json.load(open(self.data_json + f'{mov_name}.json', 'r'))
        age = float(subject_json['PatientAge'][:-1])
        height = float(subject_json['PatientHeight'])
        weight = float(subject_json['PatientWeight'])
        dose = float(subject_json['RadiopharmaceuticalInformationSequence']['RadionuclideTotalDose'])
        vendor = subject_json['Manufacturer']
        if 'siemens' in vendor.lower():
            vendor_id = 0.
        elif 'ge' in vendor.lower():
            vendor_id = 1.
        elif 'philips' in vendor.lower():
            vendor_id = 2.
        else:
            vendor_id = 3.
        # Parse StudyDate (+ optional StudyTime) to UTC Unix epoch seconds (float)
        study_date_raw = subject_json.get('StudyDate', '')
        study_time_raw = subject_json.get('StudyTime', '')
        date_str = study_date_raw.replace('-', '')
        if len(date_str) >= 8 and date_str[:8].isdigit():
            yyyy, mm, dd = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
        else:
            yyyy, mm, dd = 1970, 1, 1
        time_digits = ''.join(ch for ch in study_time_raw if ch.isdigit())
        HH = int(time_digits[0:2]) if len(time_digits) >= 2 else 0
        MM = int(time_digits[2:4]) if len(time_digits) >= 4 else 0
        SS = int(time_digits[4:6]) if len(time_digits) >= 6 else 0
        dt_utc = datetime(yyyy, mm, dd, HH, MM, SS, tzinfo=timezone.utc)
        study_ts = float(dt_utc.timestamp())
        '''
        # Retrieve precomputed normalized covariates
        covariates_vec = torch.from_numpy(self.covariates[index]).float()
        covariates_raw_vec = torch.from_numpy(self.covariates_raw[index]).float()

        return {
            'CT': x.float(),
            'SUV': x_suv.float(),
            'CT_seg': x_seg.long(),
            'SUV_seg': x_suv_seg.long(),
            'CT_Org': x_.float(),
            'SUV_Org': x_suv_.float(),
            'covariates': covariates_vec,
            'covariates_raw': covariates_raw_vec
        }

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
    save_dir = 'VFAAtlas_SSIM_1_MS_1_diffusion_1/'
    model_dir = 'experiments/' + save_dir
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

    ct_atlas_dir = 'atlas/ct/VFAAtlas_SSIM_1_MS_1_diffusion_1/'
    pt_atlas_dir = 'atlas/suv/VFAAtlas_SSIM_1_MS_1_diffusion_1/'
    seg_atlas_path = 'atlas/seg/suv_seg_atlas_w_reg_14lbls.nii.gz'
    ct_seg_atlas_path = 'atlas/seg/ct_seg_atlas_w_reg_40lbls.nii.gz'
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
    
    '''
    Initialize training
    '''
    # Replace the way we build and filter full_names to drop singles (patients with only one date)
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

    # Keep only entries whose pid has at least two distinct dates
    full_names = [b for b in bases
                  if (m := re.match(r'^(?P<pid>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})$', b))
                  and len(pid_to_dates.get(m.group('pid'), set())) >= 2]

    full_names = natsorted(full_names)
    val_names = full_names
    val_set = JHUPSMADataset(val_dir, val_names)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    lesion_stats = utils.RunningStats3D(shape=(H, W, D), device='cuda:0', dtype=torch.float32)
    '''
    Validation
    '''
    delta_stack = []      # list of (X*Y*Z,) vectors
    cov_stack   = []      # list of covariate vectors (e.g., age, BMI, vendor, Δtime)
    idx = 0
    with torch.no_grad():
        for data in val_loader:
            idx += 1
            print(f'Processing patient: {idx}')
            model.eval()
            pat_ct = data['CT'].cuda().float()
            pat_suv = data['SUV'].cuda().float()
            pat_ct_org = data['CT_Org'].cuda().float()
            pat_suv_org = data['SUV_Org'].cuda().float()
            pat_covariates = data['covariates'].cuda().float()
            pat_covariates_raw = data['covariates_raw'].cuda().float()
            print(pat_covariates_raw)

            y_seg = data['SUV_seg'].cuda().long()
            if torch.sum(y_seg==14).item() == 0:
                pass#continue
            else:
                print('no')
                pass
            
            
            def_atlas, def_image, pos_flow, neg_flow = model((x_ct.repeat(2, 1, 1, 1, 1), pat_ct))
            
            def_pat_ct = model.spatial_trans(pat_ct_org.cuda().float(), neg_flow.cuda())

            def_pat_suv = model.spatial_trans(pat_suv_org.cuda().float(), neg_flow.cuda())
            
            suv_baseline = def_pat_suv[0,0]
            suv_followup = def_pat_suv[1,0]
            
            Rb = robust_bp_ref(suv_baseline, aorta_mask[0,0])  # baseline blood-pool SUV
            Rf = robust_bp_ref(suv_followup, aorta_mask[0,0])  # follow-up blood-pool SUV
            
            # ---- compute SUL per timepoint ----
            # get raw covariates for each timepoint to compute LBM
            cov0_raw = data['covariates_raw'][0].cpu().numpy()  # [age, height(m), weight(kg), dose, rel_days, ...]
            cov1_raw = data['covariates_raw'][1].cpu().numpy()
            h0, w0 = float(cov0_raw[1]), float(cov0_raw[2])
            h1, w1 = float(cov1_raw[1]), float(cov1_raw[2])
            
            # PSMA cohort is male; if you do have sex per subject, pass it here
            lbm0 = janmahasatian_lbm(h0, w0, sex_is_male=True)
            lbm1 = janmahasatian_lbm(h1, w1, sex_is_male=True)

            # SUL = SUV * (LBM / Weight)
            sul_bl = suv_baseline * (lbm0 / (w0 + 1e-8))
            sul_fu = suv_followup * (lbm1 / (w1 + 1e-8))

            # ---- SULR: normalize to blood-pool per timepoint ----
            sulr_bl = sul_bl / (Rb + 1e-6)
            sulr_fu = sul_fu / (Rf + 1e-6)

            # ---- target: log-delta SULR (percent-change friendly, variance-stabilizing) ----
            mask_3d  = ((x_ct_seg[0, 0] > 0.01).float() + (x_ct[0,0]>0.01).float())>0                   # torso-ish threshold; adjust if needed
            pct_delta = sulr_fu - sulr_bl
            pct_delta = smooth_inside_mask(pct_delta, mask_3d.float(), sigma=0.85)

            pct_delta = winsorize(pct_delta, 1.0, 99.0)
            
            sulr_bl_smooth = smooth_inside_mask(sulr_bl, mask_3d.float(), sigma=0.85)
            baseline_sulr_summary = torch.median(sulr_bl_smooth[mask_3d.bool()]).item()
            
            # Derive useful continuous covariates for this pair
            cov0_raw, cov1_raw = data['covariates_raw'][0].cpu(), data['covariates_raw'][1].cpu()
            # raw layout: [age, height(m), weight(kg), dose(Bq), rel_time_days, vendor one-hot...]
            age_mean = 0.5*(cov0_raw[0].item() + cov1_raw[0].item())
            dose_mean = 0.5*(cov0_raw[3].item() + cov1_raw[3].item())   # raw Bq
            rel_days = cov1_raw[4].item() - cov0_raw[4].item()          # true days between scans
            v_bl_oh = data['covariates'][0, 5:7].cpu().float()  # baseline vendor one-hot
            v_fu_oh = data['covariates'][1, 5:7].cpu().float()  # follow-up vendor one-hot
            vendor_switch = float((v_bl_oh.argmax() != v_fu_oh.argmax()))

            cov_row = torch.tensor([age_mean, baseline_sulr_summary, rel_days, vendor_switch], dtype=torch.float32)
            cov_stack.append(cov_row.tolist())
            
            mask_vec = mask_3d.reshape(-1).bool().to(pct_delta.device)
            delta_vec = pct_delta.reshape(-1)[mask_vec>0].detach().cpu()
            delta_stack.append(delta_vec)
                    
        # Split continuous vs categorical vendor parts
        Y = torch.stack(delta_stack, dim=0)           # percent change target
        Y = torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

        # X subject-level covariates
        X_cont = torch.tensor([row for row in cov_stack], dtype=torch.float32)  # shape N x 3  [age, baseline_sulr_summary, vendor_switch]

        # Standardize only continuous columns: age and baseline summary (cols 0, 1)
        means = X_cont[:, :3].mean(0)
        stds  = X_cont[:, :3].std(0).clamp_min(1e-6)
        X_cont_std = X_cont.clone()
        X_cont_std[:, 0:3] = (X_cont[:, 0:3] - means) / stds

        # Build final X with intercept
        X_ = torch.cat([torch.ones(X_cont_std.shape[0], 1), X_cont_std], dim=1)  # columns: [1, age_z, baseSULR_z, vendor_switch]
        
        betas, t_maps = GLM(X_, Y)
        
        t_age       = t_maps[1]
        t_days      = t_maps[2]
        t_baseSULR  = t_maps[3]
        t_switch    = t_maps[4]

        # Save coefficient and t maps
        save_masked_map(betas[0], 'beta_intercept.nii.gz',   x_pt_nib_aff, (H, W, D), mask_3d.bool())
        save_masked_map(betas[1], 'beta_ageZ.nii.gz',        x_pt_nib_aff, (H, W, D), mask_3d.bool())
        save_masked_map(betas[2], 'beta_baseSULRZ.nii.gz',   x_pt_nib_aff, (H, W, D), mask_3d.bool())
        save_masked_map(betas[3], 'beta_daysZ.nii.gz',        x_pt_nib_aff, (H, W, D), mask_3d.bool())
        save_masked_map(betas[4], 'beta_vendor_switchZ.nii.gz', x_pt_nib_aff, (H, W, D), mask_3d.bool())

        save_masked_map(t_age,      't_ageZ.nii.gz',          x_pt_nib_aff, (H, W, D), mask_3d.bool())
        save_masked_map(t_baseSULR, 't_baseSULRZ.nii.gz',     x_pt_nib_aff, (H, W, D), mask_3d.bool())
        save_masked_map(t_days,     't_daysZ.nii.gz',         x_pt_nib_aff, (H, W, D), mask_3d.bool())
        save_masked_map(t_switch,   't_vendor_switchZ.nii.gz', x_pt_nib_aff, (H, W, D), mask_3d.bool())


# ---- helper: smooth inside mask to avoid edge bleed ----
def smooth_inside_mask(vol, m_float, sigma=0.85):
    k = gaussian_kernel3d(sigma, (5, 5, 5)).to(vol.device)[None, None, ...]
    num = F.conv3d((vol * m_float)[None, None, ...], k, padding="same")
    den = F.conv3d(m_float[None, None, ...], k, padding="same").clamp_min(1e-6)
    return (num / den)[0, 0]

def save_masked_map(vec_masked, name, affine, shape, mask_bool3d):
    """
    vec_masked: tensor of shape (Vmask,)
    mask_bool3d: bool numpy or torch array of shape (H,W,D) with True inside body
    """
    H, W, D = shape
    out = torch.zeros(H*W*D, dtype=vec_masked.dtype)
    mv = torch.as_tensor(mask_bool3d).reshape(-1).bool()
    out[mv] = vec_masked.detach().cpu()
    vol = out.reshape(H, W, D)
    nib.save(nib.Nifti1Image(vol.numpy(), affine), name)

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