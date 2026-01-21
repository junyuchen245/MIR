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
import math
def gaussian_kernel3d(sigma_vox, size_vox):
    ax = [torch.arange(-(s//2), s//2+1, dtype=torch.float32) for s in size_vox]
    zz, yy, xx = torch.meshgrid(ax[0], ax[1], ax[2], indexing='ij')
    g = torch.exp(-(xx**2+yy**2+zz**2)/(2*sigma_vox**2))
    g = g / g.sum()
    return g

import os, json, math, re
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from datetime import datetime
import utils

class JHUPSMADataset(Dataset):
    def __init__(self, data_path, data_names,
                 data_json='/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/clinical_jsons/',
                 normalize_covariates=True):
        self.path = data_path
        self.data_names = data_names
        self.data_json = data_json
        self.normalize_covariates = normalize_covariates

        # Final covariates we will feed the model (order matters)
        # Continuous: age, bmi, log_psa_at_scan, grade_group, t_ord
        # One-hots: race, indication, T1..T4, pre/post therapies
        self.covariate_names = [
            'age', 'bmi', 'log_psa_at_scan', 'grade_group', 't_ord',
            'race_white','race_black','race_asian','race_other',
            'ind_primary','ind_recurrence','ind_metastatic','ind_therapy','ind_research','ind_other',
            't1','t2','t3','t4',
            'pre_local','pre_focal','pre_at','pre_cyto',
            'post_local','post_focal','post_at','post_cyto',
            # add raw aux and outcomes at the end
            'height_m','weight_kg',
            'relapse_event','relapsetime_days','death_event','survival_time_days'
        ]
        # Only standardize true continuous predictors used in models
        self.cont_idx = [0, 1, 2, 3, 4]  # do NOT include aux/outcomes here

        # Pre-parse and cache clinical rows and covariate vectors for all scans
        cov_rows = []
        self._clin_cache = {}
        for nm in self.data_names:
            clin_path = os.path.join(self.data_json, f'{nm}.json')
            scan_date = self._scan_date_from_name(nm)
            if os.path.isfile(clin_path):
                with open(clin_path, 'r') as f:
                    rec = json.load(f)
            else:
                rec = {}
            parsed = self.parse_clinical_record(rec, scan_date)
            self._clin_cache[nm] = parsed

            # pick PSA for the scan time
            log_psa_at_scan = parsed.get('log_psa_pre', np.nan)
            if np.isnan(log_psa_at_scan):
                log_psa_at_scan = parsed.get('log_psa_initial', np.nan)

            row_vals = [
                parsed.get('age', np.nan),
                parsed.get('bmi', np.nan),
                log_psa_at_scan,
                parsed.get('grade_group', np.nan),
                parsed.get('t_ord', np.nan),
                parsed.get('race_white', 0),
                parsed.get('race_black', 0),
                parsed.get('race_asian', 0),
                parsed.get('race_other', 0),
                parsed.get('ind_primary', 0),
                parsed.get('ind_recurrence', 0),
                parsed.get('ind_metastatic', 0),
                parsed.get('ind_therapy', 0),
                parsed.get('ind_research', 0),
                parsed.get('ind_other', 0),
                parsed.get('t1', 0),
                parsed.get('t2', 0),
                parsed.get('t3', 0),
                parsed.get('t4', 0),
                parsed.get('pre_local', 0),
                parsed.get('pre_focal', 0),
                parsed.get('pre_at', 0),
                parsed.get('pre_cyto', 0),
                parsed.get('post_local', 0),
                parsed.get('post_focal', 0),
                parsed.get('post_at', 0),
                parsed.get('post_cyto', 0),
                # new aux and outcomes
                parsed.get('height_m', np.nan),
                parsed.get('weight_kg', np.nan),
                parsed.get('relapse_event', np.nan),
                parsed.get('relapsetime_days', np.nan),
                parsed.get('death_event', np.nan),
                parsed.get('survival_time_days', np.nan),
            ]
            cov_rows.append(np.array(row_vals, dtype=np.float32))

        self.covariates_raw = np.stack(cov_rows, axis=0) if cov_rows else np.zeros((0, len(self.covariate_names)), dtype=np.float32)
        self.covname2idx = {k: i for i, k in enumerate(self.covariate_names)}

        # Normalize continuous columns
        if self.normalize_covariates and self.covariates_raw.shape[0] > 0:
            self.cov_mean = np.nanmean(self.covariates_raw[:, self.cont_idx], axis=0)
            self.cov_std = np.nanstd(self.covariates_raw[:, self.cont_idx], axis=0)
            self.cov_std[self.cov_std < 1e-6] = 1.0
            # ensure consistent dtype for later scalar use
            self.cov_mean = self.cov_mean.astype(np.float32)
            self.cov_std = self.cov_std.astype(np.float32)
            cov_norm = self.covariates_raw.copy()
            cont = cov_norm[:, self.cont_idx]
            # replace NaN with column means before z-score to avoid NaN propagation
            for j in range(cont.shape[1]):
                m = self.cov_mean[j]
                nanmask = np.isnan(cont[:, j])
                cont[nanmask, j] = m
            cont = (cont - self.cov_mean) / self.cov_std
            cov_norm[:, self.cont_idx] = cont
            self.covariates = cov_norm.astype(np.float32)
        else:
            self.cov_mean = np.zeros(len(self.cont_idx), dtype=np.float32)
            self.cov_std = np.ones(len(self.cont_idx), dtype=np.float32)
            self.covariates = self.covariates_raw.astype(np.float32)

    # ---------- parsing helpers ----------
    def _scan_date_from_name(self, name):
        m = re.search(r'(\d{4}-\d{2}-\d{2})', name)
        if m:
            try:
                return datetime.strptime(m.group(1), '%Y-%m-%d')
            except Exception:
                return None
        return None

    def _to_date_any(self, s):
        if s is None: return None
        s = str(s).strip()
        if not s: return None
        fmts = ["%m.%d.%Y","%m.%d.%y","%m/%d/%Y","%m/%d/%y","%Y-%m-%d","%Y/%m/%d"]
        for f in fmts:
            try:
                return datetime.strptime(s, f)
            except Exception:
                pass
        try:
            return datetime.strptime(s, "%d.%m.%Y")
        except Exception:
            return None

    def _to_float(self, s):
        try:
            return float(s)
        except Exception:
            return np.nan

    def _bin_12(self, x):
        try:
            v = int(float(x))
            return 1 if v == 1 else 0
        except Exception:
            return 0

    def _race_onehot(self, x):
        try:
            v = int(float(x))
        except Exception:
            v = 4
        return {
            "race_white": 1 if v == 1 else 0,
            "race_black": 1 if v == 2 else 0,
            "race_asian": 1 if v == 3 else 0,
            "race_other": 1 if v not in (1,2,3) else 0,
        }

    def _parse_psa(self, s):
        if s is None or str(s).strip() == "":
            return np.nan, 0
        txt = str(s).strip()
        m = re.match(r"^\s*<\s*([0-9.]+)\s*$", txt)
        if m:
            thr = float(m.group(1))
            return 0.5 * thr, 1
        try:
            return float(txt), 0
        except Exception:
            return np.nan, 0

    def _parse_gleason(self, s):
        if not s:
            return dict(g_pri=np.nan, g_sec=np.nan, g_sum=np.nan, grade_group=np.nan)
        txt = str(s).strip().replace(" ", "")
        m = re.match(r"^(\d)\+(\d)(?:=(\d+))?$", txt)
        if m:
            g1 = int(m.group(1)); g2 = int(m.group(2))
            gsum = int(m.group(3)) if m.group(3) else g1 + g2
        else:
            m2 = re.match(r"^(\d+)$", txt)
            if not m2:
                return dict(g_pri=np.nan, g_sec=np.nan, g_sum=np.nan, grade_group=np.nan)
            gsum = int(m2.group(1))
            if gsum == 6:   g1, g2 = 3, 3
            elif gsum == 7: g1, g2 = 3, 4
            elif gsum == 8: g1, g2 = 4, 4
            elif gsum >= 9: g1, g2 = 4, 5
            else:           g1, g2 = np.nan, np.nan
        if gsum <= 6: gg = 1
        elif gsum == 7 and g1 == 3 and g2 == 4: gg = 2
        elif gsum == 7 and g1 == 4 and g2 == 3: gg = 3
        elif gsum == 8: gg = 4
        elif gsum in (9, 10): gg = 5
        else: gg = np.nan
        return dict(g_pri=g1, g_sec=g2, g_sum=gsum, grade_group=gg)

    def _parse_t_stage(self, s):
        if not s:
            return dict(t_ord=np.nan, t1=0,t2=0,t3=0,t4=0)
        txt = str(s).lower().strip()
        txt = txt[1:] if txt.startswith('t') else txt
        m = re.match(r"^(\d)([abc])?$", txt)
        if not m:
            return dict(t_ord=np.nan, t1=0,t2=0,t3=0,t4=0)
        tnum = int(m.group(1))
        let = m.group(2) or ""
        submap = {"":0.0,"a":0.1,"b":0.2,"c":0.3}
        t_ord = tnum + submap.get(let, 0.0)
        oh = {f"t{k}": 1 if tnum == k else 0 for k in [1,2,3,4]}
        return dict(t_ord=t_ord, **oh)

    def _map_indication(self, x):
        try:
            v = int(float(x))
        except Exception:
            v = 6
        if v == 1: main = "primary"
        elif v == 2: main = "recurrence"
        elif v == 3: main = "metastatic"
        elif v == 4: main = "therapy"
        elif v == 5: main = "research"
        else: main = "other"
        keys = ["primary","recurrence","metastatic","therapy","research","other"]
        return {"ind_"+k: 1 if k == main else 0 for k in keys}, main

    def _lbm_janma(self, height_m, weight_kg, sex_is_male=True):
        if any([height_m is None, weight_kg is None]):
            return np.nan
        try:
            h = float(height_m); w = float(weight_kg)
        except Exception:
            return np.nan
        if sex_is_male:
            return 9270 * w / (6680 + 216 * w / (h**2))
        else:
            return 9270 * w / (8780 + 244 * w / (h**2))

    # ---------- clinical parser ----------
    def parse_clinical_record(self, rec: dict, scan_date: datetime):
        get = lambda k: rec.get(k, {}).get("data", None)

        age = self._to_float(get("Age"))
        race_oh = self._race_onehot(get("Race"))

        height_m = self._to_float(get("height"))
        weight_lbs = self._to_float(get("weight"))
        weight_kg = weight_lbs * 0.453592 if not math.isnan(weight_lbs) else np.nan
        bmi = weight_kg / (height_m**2) if (height_m and height_m > 0) else np.nan
        lbm = self._lbm_janma(height_m, weight_kg, sex_is_male=True)

        pre_local  = self._bin_12(get("Pre PSMA Local"))
        pre_focal  = self._bin_12(get("Pre PSMA Focal"))
        pre_at     = self._bin_12(get("Pre PSMA Systemic Androgen Targeted"))
        pre_cyto   = self._bin_12(get("Pre PSMA Systemic and cytotxic"))
        post_local = self._bin_12(get("Post PSMA Local"))
        post_focal = self._bin_12(get("Post PSMA Focal"))
        post_at    = self._bin_12(get("Post PSMA Systemic Androgen Targeted"))
        post_cyto  = self._bin_12(get("Post PSMA Systemic and cytotxic"))

        psa_initial, _ = self._parse_psa(get("Initial PSA"))
        psa_pre, _     = self._parse_psa(get("PRE PSMA PSA"))
        psa_post, _    = self._parse_psa(get("Post PSMA PSA"))
        log_psa_initial = np.log1p(psa_initial) if not np.isnan(psa_initial) else np.nan
        log_psa_pre     = np.log1p(psa_pre) if not np.isnan(psa_pre) else np.nan
        log_psa_post    = np.log1p(psa_post) if not np.isnan(psa_post) else np.nan

        ind_hot, ind_main = self._map_indication(get("Indications for PSMA-PET"))
        g = self._parse_gleason(get("Gleason score"))
        t = self._parse_t_stage(get("T stage"))

        psa_time_dt = self._to_date_any(get("psa time"))
        relapse_event = self._bin_12(get("relapse"))
        death_event   = self._bin_12(get("survival"))
        relapse_date = self._to_date_any(get("relapsetime"))
        last_date    = self._to_date_any(get("Date for survival"))

        def days_between(a, b):
            if a is None or b is None: return np.nan
            return (b - a).days

        relapsetime_days = days_between(scan_date, relapse_date) if relapse_event == 1 else days_between(scan_date, last_date)
        if isinstance(relapsetime_days, (int, float)) and not np.isnan(relapsetime_days):
            relapsetime_days = max(0, relapsetime_days)

        survival_time_days = days_between(scan_date, last_date)
        if isinstance(survival_time_days, (int, float)) and not np.isnan(survival_time_days):
            survival_time_days = max(0, survival_time_days)

        out = dict(
            age=age,
            height_m=height_m, weight_kg=weight_kg, bmi=bmi, lbm=lbm,
            **race_oh,
            pre_local=pre_local, pre_focal=pre_focal, pre_at=pre_at, pre_cyto=pre_cyto,
            post_local=post_local, post_focal=post_focal, post_at=post_at, post_cyto=post_cyto,
            psa_initial=psa_initial, log_psa_initial=log_psa_initial,
            psa_pre=psa_pre, log_psa_pre=log_psa_pre,
            psa_post=psa_post, log_psa_post=log_psa_post,
            indication_main=ind_main, **ind_hot,
            g_primary=g["g_pri"], g_secondary=g["g_sec"], g_sum=g["g_sum"], grade_group=g["grade_group"],
            t_ord=t["t_ord"], t1=t["t1"], t2=t["t2"], t3=t["t3"], t4=t["t4"],
            psa_time_dt=psa_time_dt,
            relapse_event=relapse_event, relapsetime_days=relapsetime_days,
            death_event=death_event, survival_time_days=survival_time_days,
        )
        return out

    # ---------- image helpers ----------
    def norm_img(self, img):
        img = img.copy()
        img[img < -300] = -300
        img[img > 300] = 300
        mn, mx = img.min(), img.max()
        if mx <= mn: return np.zeros_like(img, dtype=np.float32)
        return ((img - mn) / (mx - mn)).astype(np.float32)

    def norm_suv(self, img):
        img = img.copy().astype(np.float32)
        img_max, img_min = 15.0, 0.0
        norm = (img - img_min) / (img_max - img_min)
        norm[norm < 0] = 0
        norm[norm > 1] = 1
        return norm.astype(np.float32)

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]), dtype=np.float32)
        for i in range(C):
            out[i, ...] = (img == i)
        return out

    # ---------- dataset dunder ----------
    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, index):
        mov_name = self.data_names[index]

        # images
        x_ct_n = nib.load(os.path.join(self.path, f'{mov_name}_CT.nii.gz'))
        x_ct = x_ct_n.get_fdata()
        x_ct_norm = self.norm_img(x_ct)

        x_suv_n = nib.load(os.path.join(self.path, f'{mov_name}_SUV.nii.gz'))
        x_suv = x_suv_n.get_fdata()
        x_suv_norm = self.norm_suv(x_suv)

        x_ct_seg_n = nib.load(os.path.join(self.path, f'{mov_name}_CT_seg.nii.gz'))
        x_ct_seg = utils.remap_totalseg_lbl(x_ct_seg_n.get_fdata())

        x_suv_seg_n = nib.load(os.path.join(self.path, f'{mov_name}_SUV_seg.nii.gz'))
        x_suv_seg = utils.remap_suv_lbl(x_suv_seg_n.get_fdata(), include_lesions=True)

        # shapes to [1, ...]
        ct = torch.from_numpy(x_ct_norm[None, ...])
        suv = torch.from_numpy(x_suv_norm[None, ...])
        ct_seg = torch.from_numpy(x_ct_seg[None, ...]).long()
        suv_seg = torch.from_numpy(x_suv_seg[None, ...]).long()
        ct_org = torch.from_numpy(x_ct[None, ...]).float()
        suv_org = torch.from_numpy(x_suv[None, ...]).float()
        # covariates
        parsed = self._clin_cache.get(mov_name, {})
        log_psa_at_scan = parsed.get('log_psa_pre', np.nan)
        if np.isnan(log_psa_at_scan):
            log_psa_at_scan = parsed.get('log_psa_initial', np.nan)

        row = [
            parsed.get('age', np.nan),
            parsed.get('bmi', np.nan),
            log_psa_at_scan,
            parsed.get('grade_group', np.nan),
            parsed.get('t_ord', np.nan),
            parsed.get('race_white', 0),
            parsed.get('race_black', 0),
            parsed.get('race_asian', 0),
            parsed.get('race_other', 0),
            parsed.get('ind_primary', 0),
            parsed.get('ind_recurrence', 0),
            parsed.get('ind_metastatic', 0),
            parsed.get('ind_therapy', 0),
            parsed.get('ind_research', 0),
            parsed.get('ind_other', 0),
            parsed.get('t1', 0),
            parsed.get('t2', 0),
            parsed.get('t3', 0),
            parsed.get('t4', 0),
            parsed.get('pre_local', 0),
            parsed.get('pre_focal', 0),
            parsed.get('pre_at', 0),
            parsed.get('pre_cyto', 0),
            parsed.get('post_local', 0),
            parsed.get('post_focal', 0),
            parsed.get('post_at', 0),
            parsed.get('post_cyto', 0),
            # aux and outcomes, kept RAW
            parsed.get('height_m', np.nan),
            parsed.get('weight_kg', np.nan),
            parsed.get('relapse_event', np.nan),
            parsed.get('relapsetime_days', np.nan),
            parsed.get('death_event', np.nan),
            parsed.get('survival_time_days', np.nan),
        ]
        cov_raw = torch.tensor(row, dtype=torch.float32)

        # normalize only selected continuous columns
        cov_norm = cov_raw.clone()
        if self.normalize_covariates and len(self.cont_idx) > 0:
            for j, colidx in enumerate(self.cont_idx):
                m = float(self.cov_mean[j])
                s = float(self.cov_std[j]) if float(self.cov_std[j]) != 0.0 else 1.0
                v = cov_norm[colidx].item()
                if np.isnan(v):
                    v = m
                cov_norm[colidx] = float((v - m) / s)

        # optionally provide scan date for exact day deltas
        scan_date_str = None
        try:
            scan_date_str = mov_name.split('_')[1]
        except Exception:
            pass

        return {
            'CT': ct.float(),
            'SUV': suv.float(),
            'CT_seg': ct_seg,
            'SUV_seg': suv_seg,
            'CT_Org': ct_org,
            'SUV_Org': suv_org,
            'covariates': cov_norm,
            'covariates_raw': cov_raw,
            'covname2idx': self.covname2idx,     # <-- use this downstream
            'SubjectID': mov_name,
            'scan_date': scan_date_str,
            }   


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
    mask_3d = ((x_ct_seg[0,0] > 0.01) | (x_ct[0,0] > 0.01)).bool()
    # Optional: close small holes inside the mask to avoid edge artifacts
    CLOSE_MASK_RADIUS = 2  # set to 0 to disable; typical 1-3 vox
    if CLOSE_MASK_RADIUS and CLOSE_MASK_RADIUS > 0:
        mask_3d = morph_close3d(mask_3d, radius=CLOSE_MASK_RADIUS)
    # Optional: save the atlas mask for QA
    # nib.save(nib.Nifti1Image(mask_3d.cpu().detach().float().numpy(), np.eye(4)), 'temp_mask.nii.gz')
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
    # Collect patients (each batch has 2 scans: assume index 0 = baseline)
    baseline_X_list = []   # baseline SULR vectorized within mask
    time_to_event = []     # days to relapse (or survival) from baseline
    event_indicator = []   # 1 if event occurred
    clinical_rows = []     # clinical covariates per patient
    keep_mask = None       # atlas mask (bool) to map back to volume

    idx = 0
    with torch.no_grad():
        for data in val_loader:
            idx += 1
            print(f'Prognostic pass, patient {idx}, {data["SubjectID"]}')

            model.eval()
            pat_ct = data['CT'].cuda().float()          # [2,1,H,W,D]
            pat_suv_org = data['SUV_Org'].cuda().float()
            cov_raw = data['covariates_raw'].cpu().float()  # [2, P]
            cov_names = val_set.covariate_names
            # forward: warp both timepoints into atlas
            def_atlas, def_image, pos_flow, neg_flow = model((x_ct.repeat(2,1,1,1,1), pat_ct))
            def_pat_suv = model.spatial_trans(pat_suv_org, neg_flow)  # [2,1,H,W,D]
            suv_bl = def_pat_suv[0, 0]  # baseline predictor map

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
            # body mask + gentle smoothing
            
            sulr_bl_s = smooth_inside_mask(sulr_bl, mask_3d.float(), sigma=1.0)
            #nib.save(nib.Nifti1Image(sulr_bl_s.cpu().detach().numpy(), x_pt_nib_aff), 'tmp_img.nii.gz')
            #sys.exit(0)
            body_vals = sulr_bl_s[mask_3d.bool()]
            
            if body_vals.numel() == 0:
                print('Skip: empty mask')
                continue

            # vectorize baseline predictor within mask
            mv = mask_3d.reshape(-1).bool()
            if keep_mask is None:
                keep_mask = mv.detach().cpu().numpy()
            Xv = sulr_bl_s.reshape(-1)[mv]        # [Vmask]
            # Preserve NaNs here; we'll handle them with nanmean/nanstd per-voxel later
            Xv = Xv.cpu().numpy()
            baseline_X_list.append(Xv)

            # choose outcome: relapse time; fallback to survival time if relapse missing
            # These are computed relative to scan_date in your parser
            # Use baseline row (index 0)
            # Avoid using collated covname2idx (which becomes batched); resolve via cov_names instead
            def _get_val(row, name):
                try:
                    j = cov_names.index(name)
                except ValueError:
                    return np.nan
                v = row[j]
                if isinstance(v, torch.Tensor):
                    # handle scalar or length-1; if accidentally length>1, take first
                    v = v.item() if v.numel() == 1 else v.view(-1)[0].item()
                return float(v)

            # baseline row = 0 of the batch
            t_rel = _get_val(cov_raw[0], 'relapsetime_days')
            e_rel = _get_val(cov_raw[0], 'relapse_event')
            t_sur = _get_val(cov_raw[0], 'survival_time_days')
            e_dea = _get_val(cov_raw[0], 'death_event')

            print(f'  Relapse time: {t_rel} days, event: {e_rel}')
            print(f'  Survival time: {t_sur} days, event: {e_dea}')
            
            if np.isfinite(t_rel) and e_rel in (0.0, 1.0):
                time_to_event.append(t_rel)
                event_indicator.append(int(e_rel))
            elif np.isfinite(t_sur) and e_dea in (0.0, 1.0):
                time_to_event.append(t_sur)
                event_indicator.append(int(e_dea))
            else:
                # no usable outcome; drop this patient
                baseline_X_list.pop()
                print('Skip: missing outcome')
                continue

            # build clinical covariates row for multivariable Cox
            clinical_rows.append(build_clinical_matrix(cov_raw[0, :].numpy(), cov_names))

    # stack patient-by-voxel predictor matrix
    Ximg = np.stack(baseline_X_list, axis=0)   # [N, Vmask]
    time_np = np.asarray(time_to_event, dtype=float)
    event_np = np.asarray(event_indicator, dtype=int)
    Xclin = np.stack(clinical_rows, axis=0)    # [N, Pclin]

    print(f'Dataset for prognostic analysis: N={Ximg.shape[0]} patients, Vmask={Ximg.shape[1]} voxels')

    # ---------- Voxelwise univariate Cox (baseline SULR -> time-to-event) ----------
    # Hyperparameters (adjust if needed)
    VOXEL_L2 = 1e-4       # ridge for univariate fits
    MAX_ITER = 40
    # Coverage and stability thresholds
    MIN_COVERAGE = 15          # require at least this many patients contributing
    MIN_EVENTS   = 5           # and at least this many events among contributing patients
    MIN_STD      = 1e-6        # minimal across-patient std to attempt a fit

    # Run in chunks to limit RAM
    V = Ximg.shape[1]
    chunk = 100000
    betas = np.zeros(V, dtype=float)
    ses   = np.zeros(V, dtype=float)
    zs    = np.zeros(V, dtype=float)
    ps    = np.ones(V, dtype=float)
    cover = np.zeros(V, dtype=np.int32)
    stdev = np.zeros(V, dtype=float)

    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        Xi = Ximg[:, start:end]  # [N, W]
        for j in range(Xi.shape[1]):
            xcol = Xi[:, j]
            # basic coverage/stability checks
            mask_fin = np.isfinite(xcol)
            n_cov = int(mask_fin.sum())
            e_cov = int(event_np[mask_fin].sum())
            s = np.nanstd(xcol)
            cover[start + j] = n_cov
            stdev[start + j] = s if np.isfinite(s) else np.nan
            if (n_cov < MIN_COVERAGE) or (e_cov < MIN_EVENTS) or (not np.isfinite(s)) or (s < MIN_STD):
                betas[start + j] = np.nan; ses[start + j] = np.nan; zs[start + j] = np.nan; ps[start + j] = np.nan
                continue
            # standardize predictor for numeric stability
            m = np.nanmean(xcol)
            xz = (np.where(np.isfinite(xcol), xcol, m) - m) / (s + 1e-9)
            # tame extreme standardized values to avoid quasi-separation / runaway betas
            xz = np.clip(xz, -8.0, 8.0)
            b, se, z, p = cox_fit(xz, time_np, event_np, l2=VOXEL_L2, max_iter=MAX_ITER)
            # make outputs at least 1-D to avoid indexing a scalar
            b = np.atleast_1d(b); se = np.atleast_1d(se); z = np.atleast_1d(z); p = np.atleast_1d(p)
            betas[start + j] = float(b[0]); ses[start + j] = float(se[0]); zs[start + j] = float(z[0]); ps[start + j] = float(p[0])
        print(f'Cox progress: {end}/{V} voxels')
        gc.collect()

    # BH-FDR at q=0.05 (primary) and q=0.10 (exploratory)
    sig_mask05, qvals05 = benjamini_hochberg(ps, q=0.05)
    sig_mask10, qvals10 = benjamini_hochberg(ps, q=0.10)

    # ---------- Save maps ----------
    def save_vector_as_vol(vec, name, affine, shape, mv_bool):
        H, W, D = shape
        out = np.zeros(H*W*D, dtype=np.float32)
        out[mv_bool.reshape(-1)] = vec.astype(np.float32)
        vol = out.reshape(H, W, D)
        nib.save(nib.Nifti1Image(vol, affine), name)

    # Safe exponentiation to avoid under/overflow when saving as float32
    HR = np.exp(np.clip(betas, -50.0, 50.0))
    # mark invalid fits clearly
    HR[~np.isfinite(betas)] = np.nan
    zs[~np.isfinite(zs)] = np.nan
    ps[~np.isfinite(ps)] = np.nan
    # Additional views helpful for interpretation/visualization
    beta_map = betas.copy()                              # log(HR)
    risk_ratio = np.exp(np.clip(np.abs(betas), 0.0, 50.0))  # exp(|beta|) >= 1
    hr_gt1 = np.where(betas >= 0, HR, np.nan)            # show only risk-increasing
    inv_hr_protective = np.where(betas < 0, np.exp(np.clip(-betas, 0.0, 50.0)), np.nan)  # protective strength >1
    save_vector_as_vol(HR,   'hr_map.nii.gz',      x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(beta_map.astype(np.float32), 'beta_map.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(risk_ratio.astype(np.float32), 'risk_ratio_map.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(hr_gt1.astype(np.float32), 'hr_gt1_map.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(inv_hr_protective.astype(np.float32), 'inv_hr_protective_map.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(zs,   'z_map.nii.gz',       x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(ps,   'p_map.nii.gz',       x_pt_nib_aff, (H, W, D), keep_mask)
    # FDR masks for q=0.05 and q=0.10
    save_vector_as_vol(sig_mask05.astype(np.float32), 'sig_mask_fdr_q05.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(sig_mask10.astype(np.float32), 'sig_mask_fdr_q10.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)

    # Publication-friendly HR variants masked by FDR
    hr_gt1_q05 = np.where(sig_mask05, hr_gt1, np.nan)
    inv_hr_protective_q05 = np.where(sig_mask05, inv_hr_protective, np.nan)
    hr_gt1_q10 = np.where(sig_mask10, hr_gt1, np.nan)
    inv_hr_protective_q10 = np.where(sig_mask10, inv_hr_protective, np.nan)
    save_vector_as_vol(hr_gt1_q05.astype(np.float32), 'hr_gt1_fdr_q05.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(inv_hr_protective_q05.astype(np.float32), 'inv_hr_protective_fdr_q05.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(hr_gt1_q10.astype(np.float32), 'hr_gt1_fdr_q10.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(inv_hr_protective_q10.astype(np.float32), 'inv_hr_protective_fdr_q10.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)

    # Optional: store q-values for visualization (both thresholds)
    save_vector_as_vol(qvals05, 'q_map_q05.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(qvals10, 'q_map_q10.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    # Also provide viewer-friendly versions with NaNs filled as 1.0 (non-significant)
    save_vector_as_vol(np.nan_to_num(qvals05, nan=1.0).astype(np.float32), 'q_map_vis_q05.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(np.nan_to_num(qvals10, nan=1.0).astype(np.float32), 'q_map_vis_q10.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    # Save diagnostics: coverage and std across patients (helps explain zeros)
    save_vector_as_vol(cover.astype(np.float32), 'coverage_map.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(np.nan_to_num(stdev, nan=0.0).astype(np.float32), 'std_map.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)

    # ---------- Build patient-level Voxel Prognostic Score (VPS) ----------
    # Use q=0.05 mask as the default for VPS construction
    sig_idx = np.where(sig_mask05)[0]
    if sig_idx.size == 0:
        print('No FDR-significant voxels at q=0.05; consider reporting top-k or relax q to 0.10.')
        VPS = Ximg.mean(axis=1)  # fallback global mean
    else:
        # Weighted by Cox beta (positive weights increase risk)
        w = betas[sig_idx]
        # Use standardized effect scale for better transferability
        # Standardize each significant voxel using cohort-wide mean/std
        m_sig = np.nanmean(Ximg[:, sig_idx], axis=0)
        s_sig = np.nanstd(Ximg[:, sig_idx], axis=0); s_sig[s_sig < 1e-9] = 1.0
        Xz_sig = (np.where(np.isfinite(Ximg[:, sig_idx]), Ximg[:, sig_idx], m_sig) - m_sig) / s_sig
        Xz_sig = np.clip(Xz_sig, -8.0, 8.0)
        VPS = (Xz_sig * w[None, :]).sum(axis=1) / (np.abs(w).sum() + 1e-8)

    # Optional: Out-of-fold VPS to avoid double-dipping (disabled by default)
    USE_OOF_VPS = True
    K_FOLDS_OOF = 5

    def _stratified_kfold_indices(events, k, seed=42):
        rng = np.random.RandomState(seed)
        idx_pos = np.where(events == 1)[0]
        idx_neg = np.where(events == 0)[0]
        rng.shuffle(idx_pos); rng.shuffle(idx_neg)
        folds = [[] for _ in range(k)]
        for i, ix in enumerate(idx_pos):
            folds[i % k].append(ix)
        for i, ix in enumerate(idx_neg):
            folds[i % k].append(ix)
        return [np.array(sorted(f), dtype=int) for f in folds]

    def _compute_oof_vps(Xall, time_all, event_all, k=5, l2=1e-4, max_iter=40, q=0.05, seed=42):
        N, V = Xall.shape
        folds = _stratified_kfold_indices(event_all, k, seed)
        oof = np.zeros(N, dtype=float)
        for fi, val_idx in enumerate(folds):
            train_idx = np.setdiff1d(np.arange(N, dtype=int), val_idx)
            t_tr = time_all[train_idx]; e_tr = event_all[train_idx]
            X_tr = Xall[train_idx, :]
            X_va = Xall[val_idx, :]

            bet = np.zeros(V, dtype=float)
            pva = np.ones(V, dtype=float)
            mV  = np.zeros(V, dtype=float)
            sV  = np.ones(V, dtype=float)
            for start in range(0, V, 100000):
                end = min(start + 100000, V)
                Xchunk = X_tr[:, start:end]
                m = np.nanmean(Xchunk, axis=0)
                s = np.nanstd(Xchunk, axis=0); s[s < 1e-9] = 1.0
                mV[start:end] = m; sV[start:end] = s
                Xz = (np.where(np.isfinite(Xchunk), Xchunk, m) - m) / s
                Xz = np.clip(Xz, -8.0, 8.0)
                for j in range(end - start):
                    col = Xz[:, j]
                    if not np.isfinite(col).all():
                        col = np.where(np.isfinite(col), col, 0.0)
                    b, _, _, p = cox_fit(col, t_tr, e_tr, l2=l2, max_iter=max_iter)
                    b = np.atleast_1d(b); p = np.atleast_1d(p)
                    bet[start + j] = float(b[0]); pva[start + j] = float(p[0])
                gc.collect()
            sig_m, _ = benjamini_hochberg(pva, q=q)
            sidx = np.where(sig_m)[0]
            if sidx.size == 0:
                # fallback: simple global mean within validation
                oof[val_idx] = np.nanmean(X_va, axis=1)
            else:
                w = bet[sidx]
                m_s = mV[sidx]; s_s = sV[sidx]
                Xva_s = X_va[:, sidx]
                Xz_va = (np.where(np.isfinite(Xva_s), Xva_s, m_s) - m_s) / s_s
                Xz_va = np.clip(Xz_va, -8.0, 8.0)
                oof[val_idx] = (Xz_va * w[None, :]).sum(axis=1) / (np.abs(w).sum() + 1e-8)
            print(f'OOF fold {fi+1}/{k}: valN={len(val_idx)}, sigVox={sidx.size}')
        return oof

    VPS_oof = None
    if USE_OOF_VPS:
        VPS_oof = _compute_oof_vps(Ximg, time_np, event_np, k=K_FOLDS_OOF, l2=VOXEL_L2, max_iter=MAX_ITER, q=0.05, seed=DEFAULT_RANDOM_SEED)

    # ---------- Multivariable Cox: Clinical vs Clinical + VPS ----------
    # Standardize columns
    def zscore(A):
        m = np.nanmean(A, axis=0); s = np.nanstd(A, axis=0); s[s < 1e-9] = 1.0
        A = np.where(np.isfinite(A), A, m)
        return (A - m) / s, m, s

    Xclin_z, _, _ = zscore(Xclin)
    VPS_z, _, _   = zscore(VPS[:, None])
    X_clin_only   = Xclin_z
    X_with_vps    = np.hstack([Xclin_z, VPS_z])

    # Helper: orthogonalize a vector against covariates (remove linear redundancy)
    def orthogonalize_against(y_vec, Xmat):
        y = y_vec.reshape(-1, 1)
        try:
            coef, _, _, _ = np.linalg.lstsq(Xmat, y, rcond=None)
            resid = y - Xmat @ coef
            return resid.reshape(-1, 1)
        except Exception:
            return y

    b_c, se_c, z_c, p_c = cox_fit(X_clin_only, time_np, event_np, l2=1e-4, max_iter=60)
    b_cv, se_cv, z_cv, p_cv = cox_fit(X_with_vps, time_np, event_np, l2=1e-4, max_iter=60)

    # C-index
    c_clin = concordance_index(time_np, event_np, risk=(X_clin_only @ b_c))
    c_all  = concordance_index(time_np, event_np, risk=(X_with_vps @ b_cv))

    # Clinical + VPS (orthogonalized)
    VPS_orth = orthogonalize_against(VPS_z[:, 0], Xclin_z)
    X_with_vps_orth = np.hstack([Xclin_z, VPS_orth])
    b_cvo, _, _, _ = cox_fit(X_with_vps_orth, time_np, event_np, l2=1e-4, max_iter=60)
    c_all_orth = concordance_index(time_np, event_np, risk=(X_with_vps_orth @ b_cvo))

    # If OOF VPS computed, evaluate as well
    c_all_oof = np.nan
    c_all_oof_orth = np.nan
    if VPS_oof is not None:
        VPS_oof_z, _, _ = zscore(VPS_oof[:, None])
        X_with_oof = np.hstack([Xclin_z, VPS_oof_z])
        b_cvoof, _, _, _ = cox_fit(X_with_oof, time_np, event_np, l2=1e-4, max_iter=60)
        c_all_oof = concordance_index(time_np, event_np, risk=(X_with_oof @ b_cvoof))
        VPS_oof_orth = orthogonalize_against(VPS_oof_z[:, 0], Xclin_z)
        X_with_oof_orth = np.hstack([Xclin_z, VPS_oof_orth])
        b_cvoofo, _, _, _ = cox_fit(X_with_oof_orth, time_np, event_np, l2=1e-4, max_iter=60)
        c_all_oof_orth = concordance_index(time_np, event_np, risk=(X_with_oof_orth @ b_cvoofo))

    print('\n=== Multivariable Cox Results ===')
    print(f'Clinical only:      C-index={c_clin:.3f}  (P={X_clin_only.shape[1]} covariates)')
    print(f'Clinical + VPS:     C-index={c_all:.3f}  (adds 1 imaging covariate)')
    print(f'Clinical + VPS⊥:   C-index={c_all_orth:.3f}  (VPS orthogonalized to clinical)')
    if VPS_oof is not None:
        print(f'Clinical + VPS(OOF):     C-index={c_all_oof:.3f}')
        print(f'Clinical + VPS(OOF)⊥:   C-index={c_all_oof_orth:.3f}')

    # Save a small report for reproducibility
    with open('cox_summary.txt', 'w') as f:
        f.write(f'Patients N={Ximg.shape[0]}, Vmask={Ximg.shape[1]}\n')
        f.write(f'Events={int(event_np.sum())}, Median time={np.nanmedian(time_np):.1f} days\n')
        f.write(f'Clinical C-index={c_clin:.3f}\n')
        f.write(f'Clinical+VPS C-index={c_all:.3f}\n')
        f.write(f'Clinical+VPS(orthogonalized) C-index={c_all_orth:.3f}\n')
        if VPS_oof is not None:
            f.write(f'Clinical+VPS(OOF) C-index={c_all_oof:.3f}\n')
            f.write(f'Clinical+VPS(OOF, orthogonalized) C-index={c_all_oof_orth:.3f}\n')
        f.write(f'Significant voxels (FDR q<=0.05)={sig_idx.size}\n')

    print('Saved: hr_map.nii.gz, z_map.nii.gz, p_map.nii.gz, '
        'q_map_q05.nii.gz, q_map_q10.nii.gz, q_map_vis_q05.nii.gz, q_map_vis_q10.nii.gz, '
        'sig_mask_fdr_q05.nii.gz, sig_mask_fdr_q10.nii.gz, '
        'hr_gt1_fdr_q05.nii.gz, inv_hr_protective_fdr_q05.nii.gz, '
        'hr_gt1_fdr_q10.nii.gz, inv_hr_protective_fdr_q10.nii.gz, cox_summary.txt')

def build_clinical_matrix(cov_row_np, cov_names):
    """
    Returns a 1D array of clinical covariates for multivariable Cox.
    Suggested set for PSMA:
      age, log_psa_at_scan (fallback log_psa_pre/initial),
      grade_group, t_ord, BMI, indication (one-hot: primary/recurrence/metastatic),
      pre_androgen_targeted, pre_cytotoxic
    """
    def get(nm, default=np.nan):
        try:
            j = cov_names.index(nm); return cov_row_np[j]
        except ValueError:
            return default

    # PSA at scan
    lpsa = get('log_psa_at_scan')
    if not np.isfinite(lpsa):
        lpsa = get('log_psa_pre')
    if not np.isfinite(lpsa):
        lpsa = get('log_psa_initial')

    age = get('age'); gg = get('grade_group'); tord = get('t_ord'); bmi = get('bmi')

    ind_primary    = get('ind_primary', 0.0)
    ind_recurrence = get('ind_recurrence', 0.0)
    ind_metastatic = get('ind_metastatic', 0.0)

    pre_at   = get('pre_at', 0.0)
    pre_cyto = get('pre_cyto', 0.0)

    row = np.array([age, lpsa, gg, tord, bmi,
                    ind_primary, ind_recurrence, ind_metastatic,
                    pre_at, pre_cyto], dtype=float)

    # replace NaNs with column means (computed later in zscore)
    return row

def _cox_sort_by_time(time, event, X=None):
    idx = np.argsort(time)  # ascending time
    time_s = time[idx]
    event_s = event[idx]
    if X is None:
        return time_s, event_s, None, idx
    return time_s, event_s, X[idx, ...], idx

def _cox_partial_grad_hess(beta, X, time, event, l2=0.0):
    # Returns gradient and Fisher information (positive semidefinite)
    if X.ndim == 1:
        X = X[:, None]
    N, P = X.shape
    xb = X @ beta
    r  = np.exp(xb)
    order = np.argsort(time)   # ascending
    time = time[order]
    event = event[order]
    X = X[order, :]
    r = r[order]

    cr = np.cumsum(r[::-1])[::-1]                                  # S0
    cX = np.cumsum((r[:, None] * X)[::-1, :], axis=0)[::-1, :]     # S1

    g = np.zeros(P)
    I = np.zeros((P, P))  # Fisher information

    i = 0
    while i < N:
        t = time[i]
        j = i
        while j < N and time[j] == t:
            j += 1
        d = int(event[i:j].sum())
        if d > 0:
            s0 = cr[i]
            s1 = cX[i, :]
            mu = s1 / max(s0, 1e-12)
            xe_sum = X[i:j, :][event[i:j] == 1].sum(axis=0)
            g += xe_sum - d * mu

            X_risk = X[i:, :]
            r_risk = r[i:]
            s2 = (X_risk.T * r_risk) @ X_risk / max(s0, 1e-12)
            I += d * (s2 - np.outer(mu, mu))
        i = j

    # Ridge penalty for penalized log-likelihood: l(beta) - (l2/2)||beta||^2
    if l2 > 0:
        g -= l2 * beta
        I += l2 * np.eye(P)
    return g, I

def cox_fit(X, time, event, l2=1e-4, max_iter=50, tol=1e-6,
            return_diag=False, cond_thresh=1e8, l2_max=1e-2):
    """
    Newton/Fisher scoring for Cox PH with adaptive ridge if I is ill-conditioned.
    Returns: beta[P], se[P], z[P], p[P] (and diag if return_diag)
    """
    X = np.asarray(X, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    if X.ndim == 1:
        X = X[:, None]
    mask = np.isfinite(time) & np.isfinite(event) & np.all(np.isfinite(X), axis=1)
    X, time, event = X[mask], time[mask], event[mask]
    if X.shape[0] < 10 or event.sum() < 5:
        P = X.shape[1]
        out = (np.zeros(P), np.full(P, np.inf), np.zeros(P), np.ones(P))
        if return_diag:
            return (*out, dict(n_iter=0, converged=False, cond=np.nan, l2_used=l2, step_max=np.nan))
        return out

    time_s, event_s, X_s, _ = _cox_sort_by_time(time, event, X)
    P = X_s.shape[1]
    beta = np.zeros(P)
    last_step_max = np.nan
    last_cond = np.nan
    l2_used = float(l2)

    for it in range(max_iter):
        g, I = _cox_partial_grad_hess(beta, X_s, time_s, event_s, l2=l2_used)
        # Adaptive ridge to control conditioning of I
        try:
            last_cond = np.linalg.cond(I)
        except np.linalg.LinAlgError:
            last_cond = np.inf
        if not np.isfinite(last_cond) or last_cond > cond_thresh:
            # bump ridge up to l2_max
            bump = l2_used
            while bump < l2_max:
                bump = min(l2_max, max(bump * 10.0, 1e-6))
                I = I + (bump - l2_used) * np.eye(P)
                try:
                    last_cond = np.linalg.cond(I)
                except np.linalg.LinAlgError:
                    last_cond = np.inf
                if np.isfinite(last_cond) and last_cond <= cond_thresh:
                    l2_used = bump
                    break
            # if still bad, proceed with pinv solve below

        try:
            step = np.linalg.solve(I, g)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(I) @ g
        beta_new = beta + step
        last_step_max = float(np.max(np.abs(step)))
        beta = beta_new
        if last_step_max < tol:
            break

    # Variance from Fisher information at the solution (use final l2_used)
    _, Ifinal = _cox_partial_grad_hess(beta, X_s, time_s, event_s, l2=l2_used)
    try:
        var = np.linalg.inv(Ifinal)
    except np.linalg.LinAlgError:
        var = np.linalg.pinv(Ifinal)
    diagv = np.diag(var)
    diagv = np.where(diagv > 1e-12, diagv, 1e-12)
    se = np.sqrt(diagv)

    z = beta / se
    def norm_cdf_vec(x):
        x = np.asarray(x, dtype=float)
        try:
            from scipy.special import ndtr as _ndtr
            return _ndtr(x)
        except Exception:
            from math import erf, sqrt
            return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))
    p = 2.0 * (1.0 - norm_cdf_vec(np.abs(z)))  # keep two-sided

    if return_diag:
        return beta, se, z, p, dict(n_iter=it+1, converged=last_step_max < tol,
                                    cond=last_cond, l2_used=l2_used, step_max=last_step_max)
    return beta, se, z, p

def benjamini_hochberg(pvals, q=0.05):
    """BH-FDR that ignores NaNs and returns NaN q-values where p is NaN."""
    p = np.asarray(pvals, dtype=float)
    valid = np.isfinite(p)
    m = int(valid.sum())
    sig = np.zeros_like(p, dtype=bool)
    qvals = np.full_like(p, np.nan, dtype=float)
    if m == 0:
        return sig, qvals

    p_valid = p[valid]
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    thresh = q * (np.arange(1, m+1) / m)
    passed = ranked <= thresh
    if passed.any():
        k = int(np.max(np.where(passed)[0]))
        cutoff = ranked[k]
        sig_valid = p_valid <= cutoff
    else:
        sig_valid = np.zeros_like(p_valid, dtype=bool)

    # standard monotone q-values on valid entries only
    q_tmp = ranked * m / np.arange(1, m+1)
    q_tmp = np.minimum.accumulate(q_tmp[::-1])[::-1]
    out_valid = np.clip(q_tmp, 0, 1)
    qvals_valid = np.empty_like(p_valid)
    qvals_valid[order] = out_valid

    qvals[valid] = qvals_valid
    sig[valid] = sig_valid
    return sig, qvals

def concordance_index(time, event, risk):
    # Harrell’s C (naive O(N^2), fine for N~100-300)
    n = len(time)
    num = 0; den = 0
    for i in range(n):
        for j in range(n):
            if time[i] < time[j] and event[i] == 1:
                den += 1
                if risk[i] > risk[j]:
                    num += 1
                elif risk[i] == risk[j]:
                    num += 0.5
    return num / den if den > 0 else np.nan

# ---------- Build baseline design ----------

def to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t

def safe_median_torch(x):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.median(x)

# ---- helper: nan-robust stats for older PyTorch ----
def torch_nanmean(x: torch.Tensor, dim=None, keepdim=False):
    mask = torch.isfinite(x)
    safe = torch.where(mask, x, torch.zeros(1, dtype=x.dtype, device=x.device))
    if dim is None:
        count = mask.sum().clamp_min(1)
        return safe.sum() / count
    count = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1)
    return safe.sum(dim=dim, keepdim=keepdim) / count

def torch_nanstd(x: torch.Tensor, dim=None, keepdim=False):
    if dim is None:
        m = torch_nanmean(x)
        mask = torch.isfinite(x)
        diff2 = torch.where(mask, (x - m)**2, torch.zeros_like(x))
        count = mask.sum().clamp_min(1)
        var = diff2.sum() / count
        return torch.sqrt(var.clamp_min(0))
    m = torch_nanmean(x, dim=dim, keepdim=True)
    mask = torch.isfinite(x)
    diff2 = torch.where(mask, (x - m)**2, torch.zeros_like(x))
    count = mask.sum(dim=dim, keepdim=True).clamp_min(1)
    var = diff2.sum(dim=dim, keepdim=True) / count
    if not keepdim:
        var = var.squeeze(dim)
    return torch.sqrt(var.clamp_min(0))

def zscore_inplace(X, cont_cols):
    if X.numel() == 0:
        return X, torch.zeros(len(cont_cols)), torch.ones(len(cont_cols))
    # use nan-robust helpers
    means = torch_nanmean(X[:, cont_cols], dim=0)
    stds  = torch_nanstd(X[:, cont_cols],  dim=0).clamp_min(1e-6)
    Xz = X.clone()
    for j, c in enumerate(cont_cols):
        col = Xz[:, c]
        col = torch.where(torch.isfinite(col), col, means[j])
        Xz[:, c] = (col - means[j]) / stds[j]
    return Xz, means, stds

# ---- helper: smooth inside mask to avoid edge bleed ----
def smooth_inside_mask(vol, m_float, sigma=0.85):
    """
    Gaussian smoothing inside mask with NaN-awareness (normalized convolution).
    - vol: torch.Tensor [H,W,D], may contain NaNs.
    - m_float: torch.Tensor [H,W,D], mask in {0,1} (float).
    - sigma: Gaussian sigma in voxels. Kernel size is set to ~6*sigma+1.
    Returns a float volume with NaNs locally averaged (if neighbors exist) and
    no bleed outside mask.
    """
    # Pick kernel size based on sigma (at least 5 and odd)
    rad = max(2, int(math.ceil(3.0 * float(sigma))))
    ksz = 2 * rad + 1
    k = gaussian_kernel3d(float(sigma), (ksz, ksz, ksz)).to(vol.device)[None, None, ...]
    # Build finite-data weight to avoid NaN propagation
    finite = torch.isfinite(vol)
    w = (finite & (m_float > 0)).float()
    v = torch.where(finite, vol, torch.zeros_like(vol))
    num = F.conv3d((v * w)[None, None, ...], k, padding="same")
    den = F.conv3d(w[None, None, ...], k, padding="same").clamp_min(1e-6)
    out = (num / den)[0, 0]
    return out

# ---- helper: morphological closing to fill small holes in a binary mask ----
def morph_close3d(mask_bool: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """
    Binary morphological closing (dilation then erosion) on a 3D boolean mask.
    Uses max-pooling to implement dilation and erosion efficiently on GPU.
    radius: number of voxels for the structuring element (kernel size = 2*radius+1).
    Returns bool mask of the same shape/device.
    """
    if radius is None or radius <= 0:
        return mask_bool
    k = int(2 * radius + 1)
    m = mask_bool.float()[None, None, ...]
    # Dilation
    dil = F.max_pool3d(m, kernel_size=k, stride=1, padding=radius)
    # Erosion via dilation of the complement: erode(A) = not dilate(not A)
    er = 1.0 - F.max_pool3d(1.0 - dil, kernel_size=k, stride=1, padding=radius)
    return (er[0, 0] > 0.5)

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