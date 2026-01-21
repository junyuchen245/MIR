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
    model_dir = '../../experiments/' + save_dir
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

    ct_atlas_dir = '../../atlas/ct/VFAAtlas_SSIM_1_MS_1_diffusion_1/'
    pt_atlas_dir = '../../atlas/suv/VFAAtlas_SSIM_1_MS_1_diffusion_1/'
    seg_atlas_path = '../../atlas/seg/suv_seg_atlas_w_reg_14lbls.nii.gz'
    ct_seg_atlas_path = '../../atlas/seg/ct_seg_atlas_w_reg_40lbls.nii.gz'
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
    # Collect patients (each batch has 2 scans: assume index 0 = baseline, 1 = follow-up)
    baseline_X_list = []   # baseline SULR vectorized within mask
    delta_X_list = []      # delta SULR (follow-up - baseline) within mask
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
            suv_fu = def_pat_suv[1, 0]  # follow-up predictor map

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

            # Follow-up LBM scaling
            try:
                ix_h = cov_names.index('height_m'); ix_w = cov_names.index('weight_kg')
                h1 = float(cov_raw[1, ix_h].item()); w1 = float(cov_raw[1, ix_w].item())
                if math.isfinite(h1) and math.isfinite(w1) and w1 > 0 and h1 > 0:
                    lbm1 = 9270.0 * w1 / (6680.0 + 216.0 * w1 / (h1*h1))
                    sul_fu = suv_fu * (lbm1 / (w1 + 1e-8))
                else:
                    sul_fu = suv_fu
            except Exception:
                sul_fu = suv_fu

            # Blood pool normalization per timepoint
            sulr_bl = sul_bl / (Rb + 1e-6)
            Rb_fu = robust_bp_ref(sul_fu, liver_mask[0,0])
            if not torch.isfinite(Rb_fu): Rb_fu = torch.tensor(1.0, device=suv_bl.device)
            sulr_fu = sul_fu / (Rb_fu + 1e-6)
            # body mask + gentle smoothing
            
            sulr_bl_s = smooth_inside_mask(sulr_bl, mask_3d.float(), sigma=1.0)
            sulr_fu_s = smooth_inside_mask(sulr_fu, mask_3d.float(), sigma=1.0)
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

            # vectorize delta (follow-up - baseline)
            dXv = (sulr_fu_s.reshape(-1)[mv] - sulr_bl_s.reshape(-1)[mv]).cpu().numpy()
            delta_X_list.append(dXv)

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

    # stack patient-by-voxel predictor matrices
    Ximg = np.stack(baseline_X_list, axis=0)   # [N, Vmask] baseline
    Ximg_delta = np.stack(delta_X_list, axis=0)  # [N, Vmask] follow-up minus baseline
    time_np = np.asarray(time_to_event, dtype=float)
    event_np = np.asarray(event_indicator, dtype=int)
    Xclin = np.stack(clinical_rows, axis=0)    # [N, Pclin]

    print(f'Dataset for prognostic analysis: N={Ximg.shape[0]} patients, Vmask={Ximg.shape[1]} voxels')

    # ---------- Voxelwise multivariable Cox (baseline SULR and delta SULR -> time-to-event) ----------
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
    # Per-voxel coefficients for baseline and delta
    betas_base = np.zeros(V, dtype=float)
    ses_base   = np.zeros(V, dtype=float)
    zs_base    = np.zeros(V, dtype=float)
    ps_base    = np.ones(V, dtype=float)
    betas_delta = np.zeros(V, dtype=float)
    ses_delta   = np.zeros(V, dtype=float)
    zs_delta    = np.zeros(V, dtype=float)
    ps_delta    = np.ones(V, dtype=float)
    cover = np.zeros(V, dtype=np.int32)
    stdev_base = np.zeros(V, dtype=float)
    stdev_delta = np.zeros(V, dtype=float)

    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        Xb = Ximg[:, start:end]        # [N, W]
        Xd = Ximg_delta[:, start:end]  # [N, W]
        for j in range(Xb.shape[1]):
            x_base = Xb[:, j]
            x_delt = Xd[:, j]
            # basic coverage/stability checks
            mask_fin = np.isfinite(x_base) | np.isfinite(x_delt)
            n_cov = int(mask_fin.sum())
            e_cov = int(event_np[mask_fin].sum())
            s_b = np.nanstd(x_base)
            s_d = np.nanstd(x_delt)
            cover[start + j] = n_cov
            stdev_base[start + j] = s_b if np.isfinite(s_b) else np.nan
            stdev_delta[start + j] = s_d if np.isfinite(s_d) else np.nan
            if (n_cov < MIN_COVERAGE) or (e_cov < MIN_EVENTS):
                betas_base[start + j] = np.nan; ses_base[start + j] = np.nan; zs_base[start + j] = np.nan; ps_base[start + j] = np.nan
                betas_delta[start + j] = np.nan; ses_delta[start + j] = np.nan; zs_delta[start + j] = np.nan; ps_delta[start + j] = np.nan
                continue
            # standardize predictor for numeric stability
            m_b = np.nanmean(x_base); s_b = np.nanstd(x_base); s_b = 1.0 if not np.isfinite(s_b) or s_b < MIN_STD else s_b
            m_d = np.nanmean(x_delt); s_d = np.nanstd(x_delt); s_d = 1.0 if not np.isfinite(s_d) or s_d < MIN_STD else s_d
            xb_z = (np.where(np.isfinite(x_base), x_base, m_b) - m_b) / (s_b + 1e-9)
            xd_z = (np.where(np.isfinite(x_delt), x_delt, m_d) - m_d) / (s_d + 1e-9)
            # tame extreme standardized values to avoid quasi-separation / runaway betas
            xb_z = np.clip(xb_z, -8.0, 8.0)
            xd_z = np.clip(xd_z, -8.0, 8.0)
            X2 = np.stack([xb_z, xd_z], axis=1)
            b, se, z, p = cox_fit(X2, time_np, event_np, l2=VOXEL_L2, max_iter=MAX_ITER)
            # make outputs at least 1-D to avoid indexing a scalar
            b = np.atleast_1d(b); se = np.atleast_1d(se); z = np.atleast_1d(z); p = np.atleast_1d(p)
            betas_base[start + j] = float(b[0]); ses_base[start + j] = float(se[0]); zs_base[start + j] = float(z[0]); ps_base[start + j] = float(p[0])
            if b.shape[0] > 1:
                betas_delta[start + j] = float(b[1]); ses_delta[start + j] = float(se[1]); zs_delta[start + j] = float(z[1]); ps_delta[start + j] = float(p[1])
            else:
                betas_delta[start + j] = np.nan; ses_delta[start + j] = np.nan; zs_delta[start + j] = np.nan; ps_delta[start + j] = np.nan
        print(f'Cox progress: {end}/{V} voxels')
        gc.collect()

    # BH-FDR for baseline and delta at q=0.05 (primary) and q=0.10 (exploratory)
    sig_mask05_base, qvals05_base = benjamini_hochberg(ps_base, q=0.05)
    sig_mask10_base, qvals10_base = benjamini_hochberg(ps_base, q=0.10)
    sig_mask05_delta, qvals05_delta = benjamini_hochberg(ps_delta, q=0.05)
    sig_mask10_delta, qvals10_delta = benjamini_hochberg(ps_delta, q=0.10)

    # ---------- Save maps ----------
    def save_vector_as_vol(vec, name, affine, shape, mv_bool):
        H, W, D = shape
        out = np.zeros(H*W*D, dtype=np.float32)
        out[mv_bool.reshape(-1)] = vec.astype(np.float32)
        vol = out.reshape(H, W, D)
        nib.save(nib.Nifti1Image(vol, affine), name)

    # Safe exponentiation to avoid under/overflow when saving as float32
    HR_base = np.exp(np.clip(betas_base, -50.0, 50.0))
    HR_delta = np.exp(np.clip(betas_delta, -50.0, 50.0))
    # mark invalid fits clearly
    HR_base[~np.isfinite(betas_base)] = np.nan
    HR_delta[~np.isfinite(betas_delta)] = np.nan
    zs_base[~np.isfinite(zs_base)] = np.nan
    zs_delta[~np.isfinite(zs_delta)] = np.nan
    ps_base[~np.isfinite(ps_base)] = np.nan
    ps_delta[~np.isfinite(ps_delta)] = np.nan
    # Additional views helpful for interpretation/visualization
    # Baseline outputs
    beta_map_base = betas_base.copy()
    risk_ratio_base = np.exp(np.clip(np.abs(betas_base), 0.0, 50.0))
    hr_gt1_base = np.where(betas_base >= 0, HR_base, np.nan)
    inv_hr_protective_base = np.where(betas_base < 0, np.exp(np.clip(-betas_base, 0.0, 50.0)), np.nan)
    save_vector_as_vol(HR_base,   'hr_map_baseline.nii.gz',      x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(beta_map_base.astype(np.float32), 'beta_map_baseline.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(risk_ratio_base.astype(np.float32), 'risk_ratio_map_baseline.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(hr_gt1_base.astype(np.float32), 'hr_gt1_baseline.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(inv_hr_protective_base.astype(np.float32), 'inv_hr_protective_baseline.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(zs_base,   'z_map_baseline.nii.gz',       x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(ps_base,   'p_map_baseline.nii.gz',       x_pt_nib_aff, (H, W, D), keep_mask)
    # Delta outputs
    beta_map_delta = betas_delta.copy()
    risk_ratio_delta = np.exp(np.clip(np.abs(betas_delta), 0.0, 50.0))
    hr_gt1_delta = np.where(betas_delta >= 0, HR_delta, np.nan)
    inv_hr_protective_delta = np.where(betas_delta < 0, np.exp(np.clip(-betas_delta, 0.0, 50.0)), np.nan)
    save_vector_as_vol(HR_delta,   'hr_map_delta.nii.gz',      x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(beta_map_delta.astype(np.float32), 'beta_map_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(risk_ratio_delta.astype(np.float32), 'risk_ratio_map_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(hr_gt1_delta.astype(np.float32), 'hr_gt1_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(inv_hr_protective_delta.astype(np.float32), 'inv_hr_protective_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(zs_delta,   'z_map_delta.nii.gz',       x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(ps_delta,   'p_map_delta.nii.gz',       x_pt_nib_aff, (H, W, D), keep_mask)
    # Combined (baseline + delta) illustrative maps: effect of simultaneous 1 SD increase in both baseline and delta SULR.
    # Note: No separate p-value/FDR is computed for this combined coefficient (it's a linear combination).
    beta_map_combined = betas_base + betas_delta
    HR_combined = np.exp(np.clip(beta_map_combined, -50.0, 50.0))
    hr_gt1_combined = np.where(beta_map_combined >= 0, HR_combined, np.nan)
    inv_hr_protective_combined = np.where(beta_map_combined < 0, np.exp(np.clip(-beta_map_combined, 0.0, 50.0)), np.nan)
    # Union and intersection of q<=0.05 significance across baseline and delta voxels
    sig_union_q05 = (sig_mask05_base | sig_mask05_delta).astype(np.float32)
    sig_intersect_q05 = (sig_mask05_base & sig_mask05_delta).astype(np.float32)
    save_vector_as_vol(HR_combined.astype(np.float32), 'hr_map_combined_baseline_plus_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(beta_map_combined.astype(np.float32), 'beta_map_combined_baseline_plus_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(hr_gt1_combined.astype(np.float32), 'hr_gt1_combined_baseline_plus_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(inv_hr_protective_combined.astype(np.float32), 'inv_hr_protective_combined_baseline_plus_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(sig_union_q05, 'sig_union_fdr_q05_baseline_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(sig_intersect_q05, 'sig_intersect_fdr_q05_baseline_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    # FDR masks for q=0.05 and q=0.10
    save_vector_as_vol(sig_mask05_base.astype(np.float32), 'sig_mask_fdr_q05_baseline.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(sig_mask10_base.astype(np.float32), 'sig_mask_fdr_q10_baseline.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(sig_mask05_delta.astype(np.float32), 'sig_mask_fdr_q05_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(sig_mask10_delta.astype(np.float32), 'sig_mask_fdr_q10_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)

    # Publication-friendly HR variants masked by FDR
    # Publication-friendly HR variants masked by FDR (baseline)
    hr_gt1_q05_base = np.where(sig_mask05_base, hr_gt1_base, np.nan)
    inv_hr_protective_q05_base = np.where(sig_mask05_base, inv_hr_protective_base, np.nan)
    hr_gt1_q10_base = np.where(sig_mask10_base, hr_gt1_base, np.nan)
    inv_hr_protective_q10_base = np.where(sig_mask10_base, inv_hr_protective_base, np.nan)
    save_vector_as_vol(hr_gt1_q05_base.astype(np.float32), 'hr_gt1_fdr_q05_baseline.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(inv_hr_protective_q05_base.astype(np.float32), 'inv_hr_protective_fdr_q05_baseline.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(hr_gt1_q10_base.astype(np.float32), 'hr_gt1_fdr_q10_baseline.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(inv_hr_protective_q10_base.astype(np.float32), 'inv_hr_protective_fdr_q10_baseline.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    # and (delta)
    hr_gt1_q05_delta = np.where(sig_mask05_delta, hr_gt1_delta, np.nan)
    inv_hr_protective_q05_delta = np.where(sig_mask05_delta, inv_hr_protective_delta, np.nan)
    hr_gt1_q10_delta = np.where(sig_mask10_delta, hr_gt1_delta, np.nan)
    inv_hr_protective_q10_delta = np.where(sig_mask10_delta, inv_hr_protective_delta, np.nan)
    save_vector_as_vol(hr_gt1_q05_delta.astype(np.float32), 'hr_gt1_fdr_q05_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(inv_hr_protective_q05_delta.astype(np.float32), 'inv_hr_protective_fdr_q05_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(hr_gt1_q10_delta.astype(np.float32), 'hr_gt1_fdr_q10_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(inv_hr_protective_q10_delta.astype(np.float32), 'inv_hr_protective_fdr_q10_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)

    # Optional: store q-values for visualization (both thresholds)
    save_vector_as_vol(qvals05_base, 'q_map_q05_baseline.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(qvals10_base, 'q_map_q10_baseline.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(qvals05_delta, 'q_map_q05_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(qvals10_delta, 'q_map_q10_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    # Also provide viewer-friendly versions with NaNs filled as 1.0 (non-significant)
    save_vector_as_vol(np.nan_to_num(qvals05_base, nan=1.0).astype(np.float32), 'q_map_vis_q05_baseline.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(np.nan_to_num(qvals10_base, nan=1.0).astype(np.float32), 'q_map_vis_q10_baseline.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(np.nan_to_num(qvals05_delta, nan=1.0).astype(np.float32), 'q_map_vis_q05_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(np.nan_to_num(qvals10_delta, nan=1.0).astype(np.float32), 'q_map_vis_q10_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    # Save diagnostics: coverage and std across patients (helps explain zeros)
    save_vector_as_vol(cover.astype(np.float32), 'coverage_map.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(np.nan_to_num(stdev_base, nan=0.0).astype(np.float32), 'std_map_baseline.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(np.nan_to_num(stdev_delta, nan=0.0).astype(np.float32), 'std_map_delta.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)

    # ---------- Build patient-level Voxel Prognostic Score (VPS) ----------
    # Use q=0.05 masks as defaults for VPS construction (baseline and delta separately)
    # Baseline VPS
    sig_idx_base = np.where(sig_mask05_base)[0]
    if sig_idx_base.size == 0:
        print('No FDR-significant baseline voxels at q=0.05; using global mean baseline SULR as fallback.')
        VPS_base = np.nanmean(Ximg, axis=1)
    else:
        w_b = betas_base[sig_idx_base]
        m_sig_b = np.nanmean(Ximg[:, sig_idx_base], axis=0)
        s_sig_b = np.nanstd(Ximg[:, sig_idx_base], axis=0); s_sig_b[s_sig_b < 1e-9] = 1.0
        Xz_sig_b = (np.where(np.isfinite(Ximg[:, sig_idx_base]), Ximg[:, sig_idx_base], m_sig_b) - m_sig_b) / s_sig_b
        Xz_sig_b = np.clip(Xz_sig_b, -8.0, 8.0)
        VPS_base = (Xz_sig_b * w_b[None, :]).sum(axis=1) / (np.abs(w_b).sum() + 1e-8)
    # Delta VPS
    sig_idx_delta = np.where(sig_mask05_delta)[0]
    if sig_idx_delta.size == 0:
        print('No FDR-significant delta voxels at q=0.05; using global mean delta SULR as fallback.')
        VPS_delta = np.nanmean(Ximg_delta, axis=1)
    else:
        w_d = betas_delta[sig_idx_delta]
        m_sig_d = np.nanmean(Ximg_delta[:, sig_idx_delta], axis=0)
        s_sig_d = np.nanstd(Ximg_delta[:, sig_idx_delta], axis=0); s_sig_d[s_sig_d < 1e-9] = 1.0
        Xz_sig_d = (np.where(np.isfinite(Ximg_delta[:, sig_idx_delta]), Ximg_delta[:, sig_idx_delta], m_sig_d) - m_sig_d) / s_sig_d
        Xz_sig_d = np.clip(Xz_sig_d, -8.0, 8.0)
        VPS_delta = (Xz_sig_d * w_d[None, :]).sum(axis=1) / (np.abs(w_d).sum() + 1e-8)

    # Optional: Out-of-fold VPS to avoid double-dipping (disabled by default)
    USE_OOF_VPS = False
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
    VPS_base_z, _, _   = zscore(VPS_base[:, None])
    VPS_delta_z, _, _  = zscore(VPS_delta[:, None])
    X_clin_only        = Xclin_z
    X_with_vps_base    = np.hstack([Xclin_z, VPS_base_z])
    X_with_vps_delta   = np.hstack([Xclin_z, VPS_delta_z])
    X_with_vps_both    = np.hstack([Xclin_z, VPS_base_z, VPS_delta_z])

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
    b_cv_b, se_cv_b, z_cv_b, p_cv_b = cox_fit(X_with_vps_base, time_np, event_np, l2=1e-4, max_iter=60)
    b_cv_d, se_cv_d, z_cv_d, p_cv_d = cox_fit(X_with_vps_delta, time_np, event_np, l2=1e-4, max_iter=60)
    b_cv_bd, se_cv_bd, z_cv_bd, p_cv_bd = cox_fit(X_with_vps_both, time_np, event_np, l2=1e-4, max_iter=60)

    # C-index
    c_clin = concordance_index(time_np, event_np, risk=(X_clin_only @ b_c))
    c_all_base   = concordance_index(time_np, event_np, risk=(X_with_vps_base @ b_cv_b))
    c_all_delta  = concordance_index(time_np, event_np, risk=(X_with_vps_delta @ b_cv_d))
    c_all_both   = concordance_index(time_np, event_np, risk=(X_with_vps_both @ b_cv_bd))

    # Clinical + VPS (orthogonalized)
    VPS_base_orth = orthogonalize_against(VPS_base_z[:, 0], Xclin_z)
    VPS_delta_orth = orthogonalize_against(VPS_delta_z[:, 0], Xclin_z)
    X_with_vps_base_orth = np.hstack([Xclin_z, VPS_base_orth])
    X_with_vps_delta_orth = np.hstack([Xclin_z, VPS_delta_orth])
    b_cvo_b, _, _, _ = cox_fit(X_with_vps_base_orth, time_np, event_np, l2=1e-4, max_iter=60)
    b_cvo_d, _, _, _ = cox_fit(X_with_vps_delta_orth, time_np, event_np, l2=1e-4, max_iter=60)
    c_all_base_orth = concordance_index(time_np, event_np, risk=(X_with_vps_base_orth @ b_cvo_b))
    c_all_delta_orth = concordance_index(time_np, event_np, risk=(X_with_vps_delta_orth @ b_cvo_d))

    # Clinical + both VPS (orthogonalized sequentially: base ⟂ clinical, delta ⟂ [clinical, base⊥])
    VPS_base_orth2 = VPS_base_orth  # already orth to clinical
    X_for_delta_orth = np.hstack([Xclin_z, VPS_base_orth2])
    VPS_delta_orth2 = orthogonalize_against(VPS_delta_z[:, 0], X_for_delta_orth)
    X_with_vps_both_orth = np.hstack([Xclin_z, VPS_base_orth2, VPS_delta_orth2])
    b_cvo_bd, _, _, _ = cox_fit(X_with_vps_both_orth, time_np, event_np, l2=1e-4, max_iter=60)
    c_all_both_orth = concordance_index(time_np, event_np, risk=(X_with_vps_both_orth @ b_cvo_bd))

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
    print(f'Clinical + VPS (baseline):     C-index={c_all_base:.3f}  (adds 1 imaging covariate)')
    print(f'Clinical + VPS⊥ (baseline):   C-index={c_all_base_orth:.3f}  (VPS orthogonalized to clinical)')
    print(f'Clinical + VPS (delta):        C-index={c_all_delta:.3f}  (adds 1 imaging covariate)')
    print(f'Clinical + VPS⊥ (delta):      C-index={c_all_delta_orth:.3f}  (VPS orthogonalized to clinical)')
    print(f'Clinical + VPS (baseline + delta): C-index={c_all_both:.3f}  (adds 2 imaging covariates)')
    print(f'Clinical + VPS⊥ (baseline + delta): C-index={c_all_both_orth:.3f}  (both VPS orthogonalized)')
    if VPS_oof is not None:
        print(f'Clinical + VPS(OOF):     C-index={c_all_oof:.3f}')
        print(f'Clinical + VPS(OOF)⊥:   C-index={c_all_oof_orth:.3f}')

    # ---------- Kaplan–Meier curves (median splits) ----------
    try:
        risk_clin = X_clin_only @ b_c
        med_c = np.nanmedian(risk_clin)
        grp_c_hi = risk_clin >= med_c
        plot_km_two_groups(time_np, event_np, grp_c_hi, out_path='km_clinical_median.png', title='KM: clinical risk (median split)')
        print('Saved: km_clinical_median.png')
    except Exception as ex:
        print(f'KM clinical plot skipped due to error: {ex}')

    try:
        vps_b = VPS_base_z[:, 0]
        med_b = np.nanmedian(vps_b)
        grp_b_hi = vps_b >= med_b
        plot_km_two_groups(time_np, event_np, grp_b_hi, out_path='km_vps_baseline_median.png', title='KM: imaging VPS baseline (median split)')
        print('Saved: km_vps_baseline_median.png')
    except Exception as ex:
        print(f'KM VPS baseline plot skipped due to error: {ex}')

    try:
        vps_d = VPS_delta_z[:, 0]
        med_d = np.nanmedian(vps_d)
        grp_d_hi = vps_d >= med_d
        plot_km_two_groups(time_np, event_np, grp_d_hi, out_path='km_vps_delta_median.png', title='KM: imaging VPS delta (median split)')
        print('Saved: km_vps_delta_median.png')
    except Exception as ex:
        print(f'KM VPS delta plot skipped due to error: {ex}')

    # Combined clinical + both VPS KM using median split of combined linear predictor
    try:
        risk_combined = X_with_vps_both @ b_cv_bd
        med_bd = np.nanmedian(risk_combined)
        grp_bd_hi = risk_combined >= med_bd
        plot_km_two_groups(time_np, event_np, grp_bd_hi, out_path='km_clinical_plus_vps_both_median.png', title='KM: clinical + VPS(baseline, delta) (median split)')
        print('Saved: km_clinical_plus_vps_both_median.png')
    except Exception as ex:
        print(f'KM clinical+both VPS plot skipped due to error: {ex}')

    # ---------- Optional: KM with three groups (tertiles) ----------
    try:
        risk_c = (X_clin_only @ b_c).ravel()
        plot_km_three_groups(time_np, event_np, risk_c, out_path='km_clinical_tertiles.png', title='KM: clinical risk (tertiles)')
        print('Saved: km_clinical_tertiles.png')
    except Exception as ex:
        print(f'KM clinical tertiles plot skipped due to error: {ex}')

    try:
        vps_b = VPS_base_z[:, 0]
        plot_km_three_groups(time_np, event_np, vps_b, out_path='km_vps_baseline_tertiles.png', title='KM: imaging VPS baseline (tertiles)')
        print('Saved: km_vps_baseline_tertiles.png')
    except Exception as ex:
        print(f'KM VPS baseline tertiles plot skipped due to error: {ex}')

    try:
        vps_d = VPS_delta_z[:, 0]
        plot_km_three_groups(time_np, event_np, vps_d, out_path='km_vps_delta_tertiles.png', title='KM: imaging VPS delta (tertiles)')
        print('Saved: km_vps_delta_tertiles.png')
    except Exception as ex:
        print(f'KM VPS delta tertiles plot skipped due to error: {ex}')

    try:
        risk_combined = (X_with_vps_both @ b_cv_bd).ravel()
        plot_km_three_groups(time_np, event_np, risk_combined, out_path='km_clinical_plus_vps_both_tertiles.png', title='KM: clinical + VPS(baseline, delta) (tertiles)')
        print('Saved: km_clinical_plus_vps_both_tertiles.png')
    except Exception as ex:
        print(f'KM clinical+both VPS tertiles plot skipped due to error: {ex}')

    # Save a small report for reproducibility
    with open('cox_summary.txt', 'w') as f:
        f.write(f'Patients N={Ximg.shape[0]}, Vmask={Ximg.shape[1]}\n')
        f.write(f'Events={int(event_np.sum())}, Median time={np.nanmedian(time_np):.1f} days\n')
        f.write(f'Clinical C-index={c_clin:.3f}\n')
        f.write(f'Clinical+VPS(baseline) C-index={c_all_base:.3f}\n')
        f.write(f'Clinical+VPS⊥(baseline) C-index={c_all_base_orth:.3f}\n')
        f.write(f'Clinical+VPS(delta) C-index={c_all_delta:.3f}\n')
        f.write(f'Clinical+VPS⊥(delta) C-index={c_all_delta_orth:.3f}\n')
        f.write(f'Clinical+VPS(baseline+delta) C-index={c_all_both:.3f}\n')
        f.write(f'Clinical+VPS⊥(baseline+delta) C-index={c_all_both_orth:.3f}\n')
        if VPS_oof is not None:
            f.write(f'Clinical+VPS(OOF baseline) C-index={c_all_oof:.3f}\n')
            f.write(f'Clinical+VPS(OOF baseline, orthogonalized) C-index={c_all_oof_orth:.3f}\n')
        f.write(f'Significant voxels baseline (FDR q<=0.05)={sig_idx_base.size}\n')
        f.write(f'Significant voxels delta (FDR q<=0.05)={sig_idx_delta.size}\n')
        f.write('Clinical covariates included: age, log_psa_at_scan (fallback pre/initial), '
                'grade_group, t_ord, BMI, indication (primary/recurrence/metastatic), '
                'pre_androgen_targeted, pre_cytotoxic.\n')
        f.write('Excluded post-PSMA covariates (post_local, post_focal, post_at, post_cyto).\n')

    print('Saved baseline maps: hr_map_baseline.nii.gz, z_map_baseline.nii.gz, p_map_baseline.nii.gz, '
        'beta_map_baseline.nii.gz, q_map_q05_baseline.nii.gz, q_map_q10_baseline.nii.gz, '
        'q_map_vis_q05_baseline.nii.gz, q_map_vis_q10_baseline.nii.gz, '
        'sig_mask_fdr_q05_baseline.nii.gz, sig_mask_fdr_q10_baseline.nii.gz, '
        'hr_gt1_fdr_q05_baseline.nii.gz, inv_hr_protective_fdr_q05_baseline.nii.gz, '
        'hr_gt1_fdr_q10_baseline.nii.gz, inv_hr_protective_fdr_q10_baseline.nii.gz')
    print('Saved delta maps:    hr_map_delta.nii.gz, z_map_delta.nii.gz, p_map_delta.nii.gz, '
        'beta_map_delta.nii.gz, q_map_q05_delta.nii.gz, q_map_q10_delta.nii.gz, '
        'q_map_vis_q05_delta.nii.gz, q_map_vis_q10_delta.nii.gz, '
        'sig_mask_fdr_q05_delta.nii.gz, sig_mask_fdr_q10_delta.nii.gz, '
        'hr_gt1_fdr_q05_delta.nii.gz, inv_hr_protective_fdr_q05_delta.nii.gz, '
        'hr_gt1_fdr_q10_delta.nii.gz, inv_hr_protective_fdr_q10_delta.nii.gz')
    print('Also saved diagnostics: coverage_map.nii.gz, std_map_baseline.nii.gz, std_map_delta.nii.gz, and cox_summary.txt')

def build_clinical_matrix(cov_row_np, cov_names):
    """
    Returns a 1D array of clinical covariates for multivariable Cox.
    Suggested set for PSMA:
      age, log_psa_at_scan (fallback log_psa_pre/initial),
      grade_group, t_ord, BMI, indication (one-hot: primary/recurrence/metastatic),
            pre_androgen_targeted, pre_cytotoxic

        Note:
            We intentionally EXCLUDE any post-PSMA covariates (e.g., post_local, post_focal,
            post_at, post_cyto) to avoid including variables that are downstream of the PSMA
            imaging itself.
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

# ---------- Kaplan–Meier utilities ----------
def _km_step(time, event):
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    order = np.argsort(t)
    t = t[order]; e = e[order]
    uniq = np.unique(t[e == 1])
    if uniq.size == 0:
        return np.array([0.0], dtype=float), np.array([1.0], dtype=float)
    S = 1.0
    times_plot = [0.0]
    surv_plot = [1.0]
    for ti in uniq:
        at_risk = np.sum(t >= ti)
        d_i = np.sum((t == ti) & (e == 1))
        if at_risk <= 0:
            continue
        times_plot.append(ti)
        surv_plot.append(surv_plot[-1])
        S = S * (1.0 - d_i / float(at_risk))
        times_plot.append(ti)
        surv_plot.append(S)
    return np.asarray(times_plot, dtype=float), np.asarray(surv_plot, dtype=float)

def plot_km_two_groups(time, event, group_bool, out_path, title='Kaplan–Meier: High vs Low risk', to_years=True, ci=True, alpha_band=0.20):
    """KM curves for two groups with optional Greenwood 95% CI shaded bands."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    g = np.asarray(group_bool, dtype=bool)
    if time.shape[0] == 0 or (~np.isfinite(time)).all():
        return
    m_hi = g & np.isfinite(time) & np.isfinite(event)
    m_lo = (~g) & np.isfinite(time) & np.isfinite(event)
    if m_hi.sum() == 0 or m_lo.sum() == 0:
        return

    def km_with_var(t_in, e_in):
        t = np.asarray(t_in, dtype=float); e = np.asarray(e_in, dtype=int)
        ord_idx = np.argsort(t); t = t[ord_idx]; e = e[ord_idx]
        uniq = np.unique(t[e == 1])
        if uniq.size == 0:
            return np.array([0.0]), np.array([1.0]), np.array([0.0])
        S = 1.0
        times_plot = [0.0]; surv_plot = [1.0]
        var_terms_cum = []
        acc = 0.0
        for ti in uniq:
            at_risk = np.sum(t >= ti)
            d_i = np.sum((t == ti) & (e == 1))
            if at_risk <= 0:
                continue
            times_plot.append(ti); surv_plot.append(surv_plot[-1])
            S = S * (1.0 - d_i / float(at_risk))
            times_plot.append(ti); surv_plot.append(S)
            if at_risk > d_i:
                acc += d_i / (at_risk * (at_risk - d_i))
            var_terms_cum.append(acc)
        times = np.asarray(times_plot, dtype=float)
        surv = np.asarray(surv_plot, dtype=float)
        var_cum = np.zeros_like(surv)
        if len(var_terms_cum):
            vi = 0
            for k in range(2, len(surv)):
                if surv[k] != surv[k-1] and vi < len(var_terms_cum):
                    vi_curr = var_terms_cum[vi]; vi += 1
                var_cum[k] = vi_curr if 'vi_curr' in locals() else 0.0
            var_cum[0:2] = 0.0
        var_surv = (surv ** 2) * var_cum
        return times, surv, var_surv

    t_hi, s_hi, v_hi = km_with_var(time[m_hi], event[m_hi])
    t_lo, s_lo, v_lo = km_with_var(time[m_lo], event[m_lo])

    if to_years:
        t_hi = t_hi / 365.25; t_lo = t_lo / 365.25; xlab = 'Time (years)'
    else:
        xlab = 'Time (days)'

    plt.figure(figsize=(6.5,5.0), dpi=140)
    plt.step(t_hi, s_hi, where='post', color='#d62728', linewidth=2.0, label=f'High (N={m_hi.sum()}, events={int(event[m_hi].sum())})')
    plt.step(t_lo, s_lo, where='post', color='#1f77b4', linewidth=2.0, label=f'Low (N={m_lo.sum()}, events={int(event[m_lo].sum())})')

    if ci:
        se_hi = np.sqrt(np.clip(v_hi, 0.0, None))
        se_lo = np.sqrt(np.clip(v_lo, 0.0, None))
        ci_hi_low = np.clip(s_hi - 1.96 * se_hi, 0.0, 1.0)
        ci_hi_high = np.clip(s_hi + 1.96 * se_hi, 0.0, 1.0)
        ci_lo_low = np.clip(s_lo - 1.96 * se_lo, 0.0, 1.0)
        ci_lo_high = np.clip(s_lo + 1.96 * se_lo, 0.0, 1.0)
        plt.fill_between(t_hi, ci_hi_low, ci_hi_high, color='#d62728', alpha=alpha_band, linewidth=0)
        plt.fill_between(t_lo, ci_lo_low, ci_lo_high, color='#1f77b4', alpha=alpha_band, linewidth=0)

    plt.ylim(0,1); plt.xlim(left=0)
    plt.grid(alpha=0.25, linestyle='--')
    plt.xlabel(xlab); plt.ylabel('Survival probability'); plt.title(title)
    plt.legend(loc='best', frameon=False); plt.tight_layout()
    try:
        plt.savefig(out_path)
    finally:
        plt.close()

def plot_km_three_groups(time, event, risk, out_path,
                         title='Kaplan–Meier: Low vs Medium vs High risk (tertiles)',
                         to_years=True, ci=True, alpha_band=0.20):
    """Tertile-based three-group KM with optional Greenwood 95% CI bands."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    r = np.asarray(risk, dtype=float)
    m_valid = np.isfinite(time) & np.isfinite(event) & np.isfinite(r)
    if m_valid.sum() == 0:
        return
    rv = r[m_valid]
    q1, q2 = np.nanquantile(rv, [1/3, 2/3])
    if not np.isfinite(q1) or not np.isfinite(q2) or q1 >= q2:
        q1 = q2 = np.nanmedian(rv)
    m_low = (r <= q1) & m_valid
    m_high = (r >= q2) & m_valid
    m_med = (~m_low & ~m_high) & m_valid
    if m_low.sum() == 0 or m_med.sum() == 0 or m_high.sum() == 0:
        return

    def km_with_var(t_in, e_in):
        t = np.asarray(t_in, dtype=float); e = np.asarray(e_in, dtype=int)
        ord_idx = np.argsort(t); t = t[ord_idx]; e = e[ord_idx]
        uniq = np.unique(t[e == 1])
        if uniq.size == 0:
            return np.array([0.0]), np.array([1.0]), np.array([0.0])
        S = 1.0
        times_plot = [0.0]; surv_plot = [1.0]
        var_terms_cum = []
        acc = 0.0
        for ti in uniq:
            at_risk = np.sum(t >= ti); d_i = np.sum((t == ti) & (e == 1))
            if at_risk <= 0:
                continue
            times_plot.append(ti); surv_plot.append(surv_plot[-1])
            S = S * (1.0 - d_i / float(at_risk))
            times_plot.append(ti); surv_plot.append(S)
            if at_risk > d_i:
                acc += d_i / (at_risk * (at_risk - d_i))
            var_terms_cum.append(acc)
        times = np.asarray(times_plot, dtype=float)
        surv = np.asarray(surv_plot, dtype=float)
        var_cum = np.zeros_like(surv)
        if len(var_terms_cum):
            vi = 0
            for k in range(2, len(surv)):
                if surv[k] != surv[k-1] and vi < len(var_terms_cum):
                    vi_curr = var_terms_cum[vi]; vi += 1
                var_cum[k] = vi_curr if 'vi_curr' in locals() else 0.0
            var_cum[0:2] = 0.0
        var_surv = (surv ** 2) * var_cum
        return times, surv, var_surv

    t_lo, s_lo, v_lo = km_with_var(time[m_low], event[m_low])
    t_md, s_md, v_md = km_with_var(time[m_med], event[m_med])
    t_hi, s_hi, v_hi = km_with_var(time[m_high], event[m_high])

    if to_years:
        t_lo, t_md, t_hi = t_lo/365.25, t_md/365.25, t_hi/365.25
        xlab = 'Time (years)'
    else:
        xlab = 'Time (days)'

    plt.figure(figsize=(7.0,5.2), dpi=140)
    plt.step(t_lo, s_lo, where='post', color='#1f77b4', linewidth=2.0, label=f'Low (N={m_low.sum()}, events={int(event[m_low].sum())})')
    plt.step(t_md, s_md, where='post', color='#ff7f0e', linewidth=2.0, label=f'Medium (N={m_med.sum()}, events={int(event[m_med].sum())})')
    plt.step(t_hi, s_hi, where='post', color='#d62728', linewidth=2.0, label=f'High (N={m_high.sum()}, events={int(event[m_high].sum())})')

    if ci:
        for t, s, v, col in [(t_lo, s_lo, v_lo, '#1f77b4'), (t_md, s_md, v_md, '#ff7f0e'), (t_hi, s_hi, v_hi, '#d62728')]:
            se = np.sqrt(np.clip(v, 0.0, None))
            lo = np.clip(s - 1.96 * se, 0.0, 1.0); hi = np.clip(s + 1.96 * se, 0.0, 1.0)
            plt.fill_between(t, lo, hi, color=col, alpha=alpha_band, linewidth=0)

    plt.ylim(0,1); plt.xlim(left=0)
    plt.grid(alpha=0.25, linestyle='--')
    plt.xlabel(xlab); plt.ylabel('Survival probability'); plt.title(title)
    plt.legend(loc='best', frameon=False); plt.tight_layout()
    try:
        plt.savefig(out_path)
    finally:
        plt.close()

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