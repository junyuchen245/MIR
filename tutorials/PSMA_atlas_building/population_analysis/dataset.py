from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib
import torch
import re
import json
import math
import utils
from datetime import datetime

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
            # radiologist assessments (numeric)
            'r1_recip_score','r1_new_lesion','r2_recip_score','r2_new_lesion',
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
                # radiologist numeric fields
                parsed.get('r1_recip_score', np.nan),
                parsed.get('r1_new_lesion', 0),
                parsed.get('r2_recip_score', np.nan),
                parsed.get('r2_new_lesion', 0),
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

        # Radiologist RECIP/new lesion (R1, R2)
        def _to_float_safe(x):
            try:
                return float(x)
            except Exception:
                return np.nan
        r1_recip_score = _to_float_safe(get("R1 RECIP score"))
        r1_new_lesion = self._bin_12(get("R1 New lesion"))
        r1_new_lesion_loc = get("R1 Location of new lesion")
        r2_recip_score = _to_float_safe(get("R2 RECIP score"))
        r2_new_lesion = self._bin_12(get("R2 New lesion"))
        r2_new_lesion_loc = get("R2 Location of new lesion")
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
            r1_recip_score=r1_recip_score, r1_new_lesion=r1_new_lesion,
            r1_new_lesion_location=r1_new_lesion_loc,
            r2_recip_score=r2_recip_score, r2_new_lesion=r2_new_lesion,
            r2_new_lesion_location=r2_new_lesion_loc,
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
            # radiologist numeric fields
            parsed.get('r1_recip_score', np.nan),
            parsed.get('r1_new_lesion', 0),
            parsed.get('r2_recip_score', np.nan),
            parsed.get('r2_new_lesion', 0),
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
            # expose radiologist string fields for downstream reporting
            'r1_new_lesion_location': parsed.get('r1_new_lesion_location', None),
            'r2_new_lesion_location': parsed.get('r2_new_lesion_location', None),
            }   