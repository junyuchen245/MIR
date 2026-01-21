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
            'post_local','post_focal','post_at','post_cyto'
        ]
        self.cont_idx = [0, 1, 2, 3, 4]  # indices in covariate_names that are continuous

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
            ]
            cov_rows.append(np.array(row_vals, dtype=np.float32))

        self.covariates_raw = np.stack(cov_rows, axis=0) if cov_rows else np.zeros((0, len(self.covariate_names)), dtype=np.float32)

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
        # expects 'PID_YYYY-MM-DD'
        try:
            return datetime.strptime(name.split('_')[1], '%Y-%m-%d')
        except Exception:
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
        # choose log_psa_at_scan here to keep consistency at item level as well
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
        ]
        cov_raw = torch.tensor(row, dtype=torch.float32)

        # normalize continuous on the fly using dataset stats
        cov_norm = cov_raw.clone()
        if self.normalize_covariates and len(self.cont_idx) > 0:
            for j, colidx in enumerate(self.cont_idx):
                m = float(self.cov_mean[j])
                s = float(self.cov_std[j]) if float(self.cov_std[j]) != 0.0 else 1.0
                v = cov_norm[colidx].item()
                if np.isnan(v):
                    v = m
                z = float((v - m) / s)
                cov_norm[colidx] = z
        return {
            'CT': ct.float(),
            'SUV': suv.float(),
            'CT_seg': ct_seg,
            'SUV_seg': suv_seg,
            'CT_Org': ct_org,
            'SUV_Org': suv_org,
            'covariates': cov_norm,           # normalized
            'covariates_raw': cov_raw,        # original scale
            'covariate_names': self.covariate_names,
            'SubjectID': mov_name
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
            print(f'Processing patient pair: {idx}, {data["SubjectID"]}')
            model.eval()

            # images
            pat_ct      = data['CT'].cuda().float()          # [2,1,H,W,D]
            pat_suv     = data['SUV'].cuda().float()
            pat_ct_org  = data['CT_Org'].cuda().float()
            pat_suv_org = data['SUV_Org'].cuda().float()

            # clinical
            cov_raw      = data['covariates_raw'].cpu().float()   # [2, P]
            cov_names    = data['covariate_names']                # list of length P
            cov_norm     = data['covariates'].cpu().float()

            # optional dates if you add them to the dataset
            scan_dates = data.get('scan_date', None)  # list[str] or None
            if scan_dates is not None:
                # expect format 'YYYY-MM-DD'
                import datetime as _dt
                try:
                    dt0 = _dt.datetime.strptime(scan_dates[0], '%Y-%m-%d')
                    dt1 = _dt.datetime.strptime(scan_dates[1], '%Y-%m-%d')
                    rel_days = (dt1 - dt0).days
                except Exception:
                    rel_days = float('nan')
            else:
                rel_days = float('nan')

            # sanity: need two timepoints
            if pat_ct.shape[0] < 2:
                print('Skipping: not a pair')
                continue

            # register to atlas then warp subject to atlas
            try:
                def_atlas, def_image, pos_flow, neg_flow = model((x_ct.repeat(2,1,1,1,1), pat_ct))
                def_pat_suv = model.spatial_trans(pat_suv_org.cuda().float(), neg_flow.cuda())
            except Exception as e:
                print(f'Skipping due to registration error: {e}')
                continue

            suv_bl = def_pat_suv[0, 0]
            suv_fu = def_pat_suv[1, 0]

            # blood pool refs with fallback
            Rb = robust_bp_ref(suv_bl, aorta_mask[0,0])    # scalar
            Rf = robust_bp_ref(suv_fu, aorta_mask[0,0])    # scalar
            if not torch.isfinite(Rb): Rb = torch.tensor(1.0, device=suv_bl.device)
            if not torch.isfinite(Rf): Rf = torch.tensor(1.0, device=suv_bl.device)

            # get height and weight for LBM per timepoint if available
            def _get_idx(name):
                return cov_names.index(name) if name in cov_names else None

            ix_bmi  = _get_idx('bmi')
            ix_age  = _get_idx('age')
            ix_logp = _get_idx('log_psa_at_scan') if _get_idx('log_psa_at_scan') is not None else \
                    (_get_idx('log_psa_pre') if _get_idx('log_psa_pre') is not None else _get_idx('log_psa_initial'))
            ix_gg   = _get_idx('grade_group')
            ix_tord = _get_idx('t_ord')

            # If you kept height_m and weight_kg in covariates_raw, use them for LBM.
            try:
                ix_h = cov_names.index('height_m')
                ix_w = cov_names.index('weight_kg')
                h0, w0 = float(cov_raw[0, ix_h].item()), float(cov_raw[0, ix_w].item())
                h1, w1 = float(cov_raw[1, ix_h].item()), float(cov_raw[1, ix_w].item())
            except ValueError:
                # fallback to BMI to approximate weight if height present, else no LBM scaling
                h0 = h1 = w0 = w1 = float('nan')

            def _janma_lbm(h, w):
                if not (math.isfinite(h) and math.isfinite(w)) or h <= 0:
                    return float('nan')
                return 9270.0 * w / (6680.0 + 216.0 * w / (h*h))

            lbm0 = _janma_lbm(h0, w0)
            lbm1 = _janma_lbm(h1, w1)

            # build SUL with safe fallbacks
            def _safe_div(num, den, eps=1e-8):
                den = den if isinstance(den, torch.Tensor) else torch.tensor(den, device=num.device, dtype=num.dtype)
                return num / (den + eps)

            if math.isfinite(lbm0) and math.isfinite(w0) and w0 > 0:
                sul_bl = suv_bl * (lbm0 / (w0 + 1e-8))
            else:
                sul_bl = suv_bl  # fallback to SUV if LBM unavailable

            if math.isfinite(lbm1) and math.isfinite(w1) and w1 > 0:
                sul_fu = suv_fu * (lbm1 / (w1 + 1e-8))
            else:
                sul_fu = suv_fu

            sulr_bl = _safe_div(sul_bl, Rb)
            sulr_fu = _safe_div(sul_fu, Rf)

            # target map: delta SULR with light smoothing and winsorization
            mask_3d = ((x_ct_seg[0,0] > 0.01).float() + (x_ct[0,0] > 0.01).float()) > 0
            pct_delta = smooth_inside_mask(sulr_fu - sulr_bl, mask_3d.float(), sigma=0.85)
            pct_delta = winsorize(pct_delta, 1.0, 99.0)

            # skip if target is degenerate
            if not torch.isfinite(pct_delta).any():
                print('Skipping: non-finite target')
                continue

            # summarize baseline SULR for covariate
            sulr_bl_smooth = smooth_inside_mask(sulr_bl, mask_3d.float(), sigma=0.85)
            body_vals = sulr_bl_smooth[mask_3d.bool()]
            if body_vals.numel() == 0:
                print('Skipping: empty mask')
                continue
            # replace nanmedian with median after sanitizing values
            baseline_sulr_summary = torch.median(torch.nan_to_num(body_vals, nan=0.0)).item()

            # build subject-level covariate row
            # choose clinically meaningful columns if present
            age_mean = float(torch_nanmean(cov_raw[:, ix_age]).item()) if ix_age is not None else float('nan')
            logpsa_mean = float(torch_nanmean(cov_raw[:, ix_logp]).item()) if ix_logp is not None else float('nan')
            gg_mean = float(torch_nanmean(cov_raw[:, ix_gg]).item()) if ix_gg is not None else float('nan')
            tord_mean = float(torch_nanmean(cov_raw[:, ix_tord]).item()) if ix_tord is not None else float('nan')

            # race one-hots
            def _get_hot(name):
                try:
                    return float(cov_raw[0, cov_names.index(name)].item())
                except Exception:
                    return 0.0
            race_white = _get_hot('race_white')
            race_black = _get_hot('race_black')
            race_asian = _get_hot('race_asian')
            race_other = _get_hot('race_other')

            # indication one-hots
            ind_primary     = _get_hot('ind_primary')
            ind_recurrence  = _get_hot('ind_recurrence')
            ind_metastatic  = _get_hot('ind_metastatic')
            ind_therapy     = _get_hot('ind_therapy')
            ind_research    = _get_hot('ind_research')
            ind_other       = _get_hot('ind_other')

            # pre therapies
            pre_local = _get_hot('pre_local')
            pre_focal = _get_hot('pre_focal')
            pre_at    = _get_hot('pre_at')
            pre_cyto  = _get_hot('pre_cyto')

            # assemble row: [age, bmi, logpsa, grade_group, t_ord, baseline_sulr, rel_days, race..., ind..., pre...]
            bmi_mean = float(torch_nanmean(cov_raw[:, cov_names.index('bmi')]).item()) if 'bmi' in cov_names else float('nan')

            cov_row = [
                age_mean, bmi_mean, logpsa_mean, gg_mean, tord_mean,
                baseline_sulr_summary, float(rel_days),
                race_white, race_black, race_asian, race_other,
                ind_primary, ind_recurrence, ind_metastatic, ind_therapy, ind_research, ind_other,
                pre_local, pre_focal, pre_at, pre_cyto,
            ]
            cov_stack.append(cov_row)

            # vectorize target inside mask
            mv = mask_3d.reshape(-1).bool().to(pct_delta.device)
            delta_vec = pct_delta.reshape(-1)[mv]
            # replace non-finite values
            delta_vec = torch.nan_to_num(delta_vec, nan=0.0, posinf=0.0, neginf=0.0).cpu()
            delta_stack.append(delta_vec)

    # stack Y and X
    if len(delta_stack) == 0:
        raise RuntimeError('No valid pairs for analysis')

    Y = torch.stack(delta_stack, dim=0)  # shape [N, Vmask]
    Y = torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    X = torch.tensor(cov_stack, dtype=torch.float32)  # shape [N, P]

    # define which columns are continuous for z-scoring
    # [age, bmi, logpsa, grade_group, t_ord, baseline_sulr, rel_days]
    cont_cols = [0, 1, 2, 3, 4, 5, 6]
    Xz, means, stds = zscore_inplace(X, cont_cols)

    # add intercept
    X_ = torch.cat([torch.ones(Xz.shape[0], 1), Xz], dim=1)

    # run GLM
    betas, t_maps = GLM(X_, Y)

    # unpack t-maps by your column order above
    t_age        = t_maps[1]
    t_bmi        = t_maps[2]
    t_logpsa     = t_maps[3]
    t_gradegroup = t_maps[4]
    t_tord       = t_maps[5]
    t_baseSULR   = t_maps[6]
    t_reldays    = t_maps[7]

    # save maps
    save_masked_map(betas[0], 'beta_intercept.nii.gz',       x_pt_nib_aff, (H, W, D), mask_3d.bool())
    save_masked_map(betas[1], 'beta_ageZ.nii.gz',            x_pt_nib_aff, (H, W, D), mask_3d.bool())
    save_masked_map(betas[2], 'beta_bmiZ.nii.gz',            x_pt_nib_aff, (H, W, D), mask_3d.bool())
    save_masked_map(betas[3], 'beta_logpsaZ.nii.gz',         x_pt_nib_aff, (H, W, D), mask_3d.bool())
    save_masked_map(betas[4], 'beta_gradegroupZ.nii.gz',     x_pt_nib_aff, (H, W, D), mask_3d.bool())
    save_masked_map(betas[5], 'beta_tordZ.nii.gz',           x_pt_nib_aff, (H, W, D), mask_3d.bool())
    save_masked_map(betas[6], 'beta_baseSULRZ.nii.gz',       x_pt_nib_aff, (H, W, D), mask_3d.bool())
    save_masked_map(betas[7], 'beta_relDaysZ.nii.gz',        x_pt_nib_aff, (H, W, D), mask_3d.bool())

    save_masked_map(t_age,        't_ageZ.nii.gz',           x_pt_nib_aff, (H, W, D), mask_3d.bool())
    save_masked_map(t_bmi,        't_bmiZ.nii.gz',           x_pt_nib_aff, (H, W, D), mask_3d.bool())
    save_masked_map(t_logpsa,     't_logpsaZ.nii.gz',        x_pt_nib_aff, (H, W, D), mask_3d.bool())
    save_masked_map(t_gradegroup, 't_gradeGroupZ.nii.gz',    x_pt_nib_aff, (H, W, D), mask_3d.bool())
    save_masked_map(t_tord,       't_tOrdZ.nii.gz',          x_pt_nib_aff, (H, W, D), mask_3d.bool())
    save_masked_map(t_baseSULR,   't_baseSULRZ.nii.gz',      x_pt_nib_aff, (H, W, D), mask_3d.bool())
    save_masked_map(t_reldays,    't_relDaysZ.nii.gz',       x_pt_nib_aff, (H, W, D), mask_3d.bool())


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