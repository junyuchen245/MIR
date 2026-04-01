#!/usr/bin/env python3
"""
Region-wise baseline comparison in atlas space (PSMA PET).

What it does
- Selects one baseline scan per patient.
- Warps PET to atlas (unless --prewarped) and computes SULR (LBM + liver BP normalization).
- Builds/loads a cohort SULR atlas (mean/std or median/MAD) and computes z-score maps per subject.
- Extracts region-wise metrics per organ label for z-scores and/or SULR.
- 2-group comparison: Welch t-test + BH-FDR per metric.
- Optional multivariate per-ROI: Hotelling's T^2 over selected metrics (2-group).
- Optional multi-group comparison: one-way ANOVA + BH-FDR per metric.
- If --multigroup and --multivariate are both set, also runs PERMANOVA per ROI
    on the multivariate metrics.
    In this case, multigroup ANOVA uses the same metric set as PERMANOVA.

Examples
    # Two-group (treated vs untreated)
    python3.8 -u gen_group_compare_regionwise.py --group-by pre_at --out-dir output_region_pre_at

    # Multivariate (vector) per ROI
    python3.8 -u gen_group_compare_regionwise.py --multivariate --radiomics-space both --mv-metrics mean_z,p90_z,mean_sulr,iqr_sulr

    # Multi-group ANOVA across pre-therapy types (skip 2-group output)
    python3.8 -u gen_group_compare_regionwise.py --multigroup --multigroup-by pre_types --multigroup-include-none --multigroup-assign first --skip-binary

    # Multi-group multivariate (PERMANOVA) using mv-metrics across pre-therapy types
    python3.8 -u gen_group_compare_regionwise.py --multigroup --multivariate --radiomics-space both --mv-metrics mean_z,p90_z,mean_sulr,iqr_sulr --multigroup-by pre_types --multigroup-include-none --multigroup-assign first
    python3.8 -u gen_group_compare_regionwise.py --multigroup --multivariate --mv-metrics auto --radiomics-space z --multigroup-by pre_types --multigroup-include-none --multigroup-assign first --skip-binary
"""

import os
import re
import json
import math
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader
from natsort import natsorted

from dataset import JHUPSMADataset
from cox_analysis import benjamini_hochberg
from MIR.models import VFA, TemplateCreation
import MIR.models.configs_VFA as CONFIGS_VFA
from MIR.statistical_analysis import robust_bp_ref
import MIR.label_reference as lbl_ref
import utils  # local population_analysis/utils.py

DEFAULT_Z_METRICS = [
    'mean_z', 'median_z', 'p10_z', 'p90_z', 'iqr_z', 'min_z', 'max_z', 'range_z',
    'std_z', 'skew_z', 'kurt_z', 'entropy_z', 'frac_z_gt'
]
DEFAULT_SULR_METRICS = [
    'mean_sulr', 'median_sulr', 'p10_sulr', 'p90_sulr', 'iqr_sulr', 'min_sulr',
    'max_sulr', 'range_sulr', 'std_sulr', 'skew_sulr', 'kurt_sulr', 'entropy_sulr'
]
DEFAULT_MV_PREF = [
    'mean_z', 'p90_z', 'std_z', 'frac_z_gt',
    'mean_sulr', 'median_sulr', 'iqr_sulr', 'p90_sulr'
]


def select_baseline_names_by_patient(data_dir: str) -> list:
    """Select one scan per patient (earliest date) using *_CT_seg files as anchors."""
    import glob
    full_paths = glob.glob(os.path.join(data_dir, '*_CT_seg*'))
    bases = [os.path.basename(p).split('_CT_seg')[0] for p in full_paths]
    pid_to_dates = {}
    for b in bases:
        m = re.match(r'^(?P<pid>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})$', b)
        if not m:
            continue
        pid = m.group('pid'); date = m.group('date')
        pid_to_dates.setdefault(pid, set()).add(date)
    selected = []
    for pid, dates in pid_to_dates.items():
        if not dates:
            continue
        bl_date = sorted(dates)[0]
        selected.append(f"{pid}_{bl_date}")
    selected = [nm for nm in natsorted(selected) if 'PSMA' not in nm]
    return selected


def prepare_vfa_and_atlas(device: str = 'cuda:0') -> Dict[str, Any]:
    """Load VFA model and atlas resources used for warping and masking."""
    H, W, D = 192, 192, 256
    save_dir = 'VFAAtlas_SSIM_1_MS_1_diffusion_1/'
    model_dir = os.path.join('..', 'experiments', save_dir)
    ct_atlas_dir = os.path.join('..', 'atlas', 'ct', save_dir)
    pt_atlas_dir = os.path.join('..', 'atlas', 'suv', save_dir)
    seg_atlas_path = os.path.join('..', 'atlas', 'seg', 'suv_seg_atlas_w_reg_14lbls.nii.gz')
    ct_seg_atlas_path = os.path.join('..', 'atlas', 'seg', 'ct_seg_atlas_w_reg_40lbls.nii.gz')

    config = CONFIGS_VFA.get_VFA_default_config()
    config.img_size = (H, W, D)
    model = VFA(config, device=device, SVF=True, return_full=True).to(device)
    template_model = TemplateCreation(model, (H, W, D))
    ckpts = natsorted(os.listdir(model_dir))
    if not ckpts:
        raise RuntimeError(f"No model checkpoints found in {model_dir}")
    best_model = torch.load(os.path.join(model_dir, ckpts[0]), map_location=device)['state_dict']
    template_model.load_state_dict(best_model)
    reg_model = template_model.reg_model.to(device)
    reg_model.eval()

    # Atlas volumes and segmentations
    x_ct_ni = nib.load(os.path.join(ct_atlas_dir, natsorted(os.listdir(ct_atlas_dir))[-1]))
    x_ct = torch.from_numpy(x_ct_ni.get_fdata()[None, None, ...]).to(device).float()
    x_pt_ni = nib.load(os.path.join(pt_atlas_dir, natsorted(os.listdir(pt_atlas_dir))[-1]))
    x_pt_aff = x_pt_ni.affine
    x_pt_seg = nib.load(seg_atlas_path)
    x_pt_seg_t = torch.from_numpy(x_pt_seg.get_fdata()[None, None, ...]).to(device).long()
    x_ct_seg = nib.load(ct_seg_atlas_path)
    x_ct_seg_t = torch.from_numpy(x_ct_seg.get_fdata()[None, None, ...]).to(device).long()

    # Masks
    liver_mask = (x_pt_seg_t == 1).float()  # [1,1,H,W,D]
    mask_3d = ((x_ct_seg_t[0, 0] > 0.01) | (x_ct[0, 0] > 0.01)).bool()
    CLOSE_MASK_RADIUS = 2
    if CLOSE_MASK_RADIUS and CLOSE_MASK_RADIUS > 0:
        mask_3d = utils.morph_close3d(mask_3d, radius=CLOSE_MASK_RADIUS)

    return dict(reg_model=reg_model, x_ct=x_ct, liver_mask=liver_mask,
                mask_3d=mask_3d, pt_affine=x_pt_aff, shape=(H, W, D))


def build_stack_via_warp(ds: JHUPSMADataset, names: list, device: str = 'cuda:0', print_every: int = 25) -> Tuple[np.ndarray, nib.Nifti1Image, Dict[str, np.ndarray]]:
    """Warp each subject's SUV to atlas space and compute SULR; return stack [N,H,W,D] with NaNs outside mask."""
    ctx = prepare_vfa_and_atlas(device=device)
    reg_model = ctx['reg_model']
    x_ct = ctx['x_ct']
    liver_mask = ctx['liver_mask']
    mask_3d = ctx['mask_3d']
    H, W, D = ctx['shape']
    ref_img = nib.Nifti1Image(np.zeros((H, W, D), dtype=np.float32), affine=ctx['pt_affine'])

    sub_ds = JHUPSMADataset(ds.path, names, data_json=ds.data_json, normalize_covariates=False)
    # Custom collate to handle None/strings in clinical fields
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
    loader = DataLoader(sub_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False, collate_fn=_safe_collate)

    stack = np.full((len(names), H, W, D), np.nan, dtype=np.float32)
    covname2idx = sub_ds.covname2idx
    total = len(names)
    print(f"[info] Warping to atlas: N={total}, device={device}")
    for i, batch in enumerate(loader):
        pat_ct = batch['CT'].to(device).float()         # [1,1,H,W,D]
        pat_suv_org = batch['SUV_Org'].to(device).float()
        with torch.no_grad():
            def_atlas, def_image, pos_flow, neg_flow = reg_model((x_ct.repeat(1,1,1,1,1), pat_ct))
            def_pat_suv = reg_model.spatial_trans(pat_suv_org, neg_flow)  # [1,1,H,W,D]
            suv_bl = def_pat_suv[0, 0]
        # blood pool normalization
        Rb = robust_bp_ref(suv_bl, liver_mask[0, 0])
        if not torch.isfinite(Rb):
            Rb = torch.tensor(1.0, device=suv_bl.device)
        # LBM scaling
        h = float(batch['covariates_raw'][0, covname2idx.get('height_m')].item()) if 'height_m' in covname2idx else float('nan')
        w = float(batch['covariates_raw'][0, covname2idx.get('weight_kg')].item()) if 'weight_kg' in covname2idx else float('nan')
        if math.isfinite(h) and math.isfinite(w) and w > 0 and h > 0:
            lbm = 9270.0 * w / (6680.0 + 216.0 * w / (h*h))
            sul = suv_bl * (lbm / (w + 1e-8))
        else:
            sul = suv_bl
        sulr = sul / (Rb + 1e-6)
        # smooth inside body mask
        sulr_s = utils.smooth_inside_mask(sulr, mask_3d.float(), sigma=1.0)
        vol = sulr_s.detach().cpu().numpy()
        # apply mask: outside -> NaN
        arr = np.full((H, W, D), np.nan, dtype=np.float32)
        mask_np = mask_3d.cpu().numpy()
        arr[mask_np] = vol[mask_np]
        stack[i] = arr
        if print_every and ((i + 1) % print_every == 0 or (i + 1) == total):
            print(f"[info]  warped {i + 1}/{total}")

    return stack, ref_img, dict(mask_3d=mask_3d.cpu().numpy())


def load_prewarped_stack(data_dir: str, names: list, image_suffix: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """Load atlas-space volumes into stack [N,H,W,D]. Missing volumes -> NaNs."""
    imgs = []
    ref = None
    missing = 0
    for nm in names:
        p = os.path.join(data_dir, f"{nm}_{image_suffix}.nii.gz")
        if not os.path.isfile(p):
            missing += 1
            imgs.append(None)
            continue
        ni = nib.load(p)
        arr = ni.get_fdata().astype(np.float32)
        if ref is None:
            ref = ni
        imgs.append(arr)
    if ref is None:
        raise RuntimeError("No volumes found for the given suffix.")
    shp = ref.shape
    stack = np.full((len(names),) + shp, np.nan, dtype=np.float32)
    for i, arr in enumerate(imgs):
        if arr is None:
            continue
        if arr.shape != shp:
            raise RuntimeError(f"Shape mismatch for {names[i]}: got {arr.shape}, expected {shp}")
        stack[i] = arr
    if missing > 0:
        print(f"[warn] Missing {missing}/{len(names)} volumes with suffix '{image_suffix}'.")
    print(f"[info] Loaded prewarped stack: shape={stack.shape}")
    return stack, ref


def load_or_compute_atlas_stats(stack: np.ndarray, out_dir: str, use_robust: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Load or compute atlas mean/std (or median/madstd) from stack [N,H,W,D]."""
    os.makedirs(out_dir, exist_ok=True)
    if use_robust:
        median_path = os.path.join(out_dir, 'sulr_atlas_median.nii.gz')
        madstd_path = os.path.join(out_dir, 'sulr_atlas_madstd.nii.gz')
        if os.path.isfile(median_path) and os.path.isfile(madstd_path):
            print(f"[info] Loading robust atlas from {out_dir}")
            med = nib.load(median_path).get_fdata().astype(np.float32)
            madstd = nib.load(madstd_path).get_fdata().astype(np.float32)
            return med, madstd
        # compute robust atlas from stack
        print(f"[info] Computing robust atlas (median/MAD) from stack: {stack.shape}")
        med = np.nanmedian(stack, axis=0).astype(np.float32)
        mad = np.nanmedian(np.abs(stack - med[None, ...]), axis=0).astype(np.float32)
        madstd = np.maximum(1.4826 * mad, 1e-6).astype(np.float32)
        nib.save(nib.Nifti1Image(med, np.eye(4)), median_path)
        nib.save(nib.Nifti1Image(madstd, np.eye(4)), madstd_path)
        return med, madstd
    else:
        mean_path = os.path.join(out_dir, 'sulr_atlas_mean.nii.gz')
        std_path = os.path.join(out_dir, 'sulr_atlas_std.nii.gz')
        if os.path.isfile(mean_path) and os.path.isfile(std_path):
            print(f"[info] Loading atlas mean/std from {out_dir}")
            mean = nib.load(mean_path).get_fdata().astype(np.float32)
            std = nib.load(std_path).get_fdata().astype(np.float32)
            return mean, std
        print(f"[info] Computing atlas mean/std from stack: {stack.shape}")
        mean = np.nanmean(stack, axis=0).astype(np.float32)
        std = np.nanstd(stack, axis=0).astype(np.float32)
        std = np.where(np.isfinite(std) & (std >= 1e-6), std, 1.0).astype(np.float32)
        nib.save(nib.Nifti1Image(mean, np.eye(4)), mean_path)
        nib.save(nib.Nifti1Image(std, np.eye(4)), std_path)
        return mean, std


def get_treatment_groups(ds: JHUPSMADataset, names: list, group_by: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Build treatment-based groups from covariates.
    Group A = treated (value==1), Group B = not treated (value==0)."""
    cov = ds.covariates_raw
    idx = ds.covname2idx
    n = len(names)
    A = np.zeros(n, dtype=bool)
    B = np.zeros(n, dtype=bool)
    detail = {"group_by": group_by}

    def getv(row, key, default=np.nan):
        j = idx.get(key, None)
        if j is None:
            return default
        return float(cov[row, j])

    def norm01(v):
        """Normalize binary coding. Accepts 1=yes, 2=no, or 1/0."""
        if not np.isfinite(v):
            return np.nan
        iv = int(v)
        if iv == 2:
            return 0
        return iv

    def assign_from_single(key, label_on="Treated", label_off="Untreated"):
        for i in range(n):
            v = norm01(getv(i, key, np.nan))
            if not np.isfinite(v):
                continue
            if int(v) == 1:
                A[i] = True
            elif int(v) == 0:
                B[i] = True
        detail["A_label"] = label_on
        detail["B_label"] = label_off

    if group_by in ('pre_local','pre_focal','pre_at','pre_cyto','post_local','post_focal','post_at','post_cyto'):
        assign_from_single(group_by, label_on=f"{group_by}=1", label_off=f"{group_by}=0")
    elif group_by == 'pre_any':
        for i in range(n):
            vals = [norm01(getv(i, k, np.nan)) for k in ('pre_local','pre_focal','pre_at','pre_cyto')]
            vals = [v for v in vals if np.isfinite(v)]
            if len(vals) == 0:
                continue
            if any(int(v) == 1 for v in vals):
                A[i] = True
            elif all(int(v) == 0 for v in vals):
                B[i] = True
        detail["A_label"] = "Any pre-treatment"
        detail["B_label"] = "No pre-treatment"
    elif group_by == 'post_any':
        for i in range(n):
            vals = [norm01(getv(i, k, np.nan)) for k in ('post_local','post_focal','post_at','post_cyto')]
            vals = [v for v in vals if np.isfinite(v)]
            if len(vals) == 0:
                continue
            if any(int(v) == 1 for v in vals):
                A[i] = True
            elif all(int(v) == 0 for v in vals):
                B[i] = True
        detail["A_label"] = "Any post-treatment"
        detail["B_label"] = "No post-treatment"
    elif group_by == 'any_therapy':
        for i in range(n):
            vals = [norm01(getv(i, k, np.nan)) for k in ('pre_local','pre_focal','pre_at','pre_cyto','post_local','post_focal','post_at','post_cyto')]
            vals = [v for v in vals if np.isfinite(v)]
            if len(vals) == 0:
                continue
            if any(int(v) == 1 for v in vals):
                A[i] = True
            elif all(int(v) == 0 for v in vals):
                B[i] = True
        detail["A_label"] = "Any therapy (pre or post)"
        detail["B_label"] = "No therapy"
    else:
        raise ValueError(f"Unknown group_by: {group_by}")

    return A, B, detail


def welch_t_and_p(mean1, var1, n1, mean2, var2, n2):
    eps = 1e-12
    denom = np.sqrt(np.maximum(var1 / np.maximum(n1, 1), 0) + np.maximum(var2 / np.maximum(n2, 1), 0) + eps)
    tstat = (mean1 - mean2) / denom
    num = (var1 / np.maximum(n1, 1) + var2 / np.maximum(n2, 1)) ** 2
    den = (np.maximum(var1, 0) ** 2) / (np.maximum(n1, 1) ** 2 * np.maximum(n1 - 1, 1)) + \
          (np.maximum(var2, 0) ** 2) / (np.maximum(n2, 1) ** 2 * np.maximum(n2 - 1, 1))
    df = num / np.maximum(den, eps)
    # Normal approximation for p-values (keeps dependencies minimal)
    from math import erf, sqrt
    z = np.abs(tstat)
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))
    return tstat, p, df


def get_multigroup_labels(ds: JHUPSMADataset, names: list, group_by: str, assign_mode: str = 'exclude', include_none: bool = True) -> Tuple[List[str], List[str]]:
    """Assign multi-group labels based on therapy types.
    group_by: pre_types, post_types, any_types
    assign_mode: exclude (drop multi-therapy), first (use first in order)
    include_none: include patients with no therapy as 'none'
    """
    cov = ds.covariates_raw
    idx = ds.covname2idx
    if group_by == 'pre_types':
        keys = ['pre_local', 'pre_focal', 'pre_at', 'pre_cyto']
    elif group_by == 'post_types':
        keys = ['post_local', 'post_focal', 'post_at', 'post_cyto']
    elif group_by == 'any_types':
        keys = ['pre_local', 'pre_focal', 'pre_at', 'pre_cyto', 'post_local', 'post_focal', 'post_at', 'post_cyto']
    else:
        raise ValueError(f"Unknown multigroup: {group_by}")
    def norm01(v):
        if not np.isfinite(v):
            return np.nan
        iv = int(v)
        if iv == 2:
            return 0
        return iv

    labels = []
    for i in range(len(names)):
        active = []
        for k in keys:
            j = idx.get(k, None)
            if j is None:
                continue
            v = norm01(cov[i, j])
            if np.isfinite(v) and int(v) == 1:
                active.append(k)
        if len(active) == 0:
            labels.append('none' if include_none else None)
        elif len(active) == 1:
            labels.append(active[0])
        else:
            if assign_mode == 'first':
                labels.append(active[0])
            else:
                labels.append(None)
    group_list = [k for k in keys]
    if include_none:
        group_list = ['none'] + group_list
    return labels, group_list


def oneway_anova(groups: List[np.ndarray]) -> Tuple[float, float, int, int]:
    """One-way ANOVA from group arrays. Returns (F, p, df_between, df_within)."""
    try:
        from scipy.stats import f as fdist
    except Exception:
        raise RuntimeError("scipy is required for multi-group ANOVA (missing scipy.stats)")
    clean_groups = [g[np.isfinite(g)] for g in groups]
    clean_groups = [g for g in clean_groups if g.size > 0]
    k = len(clean_groups)
    if k < 2:
        return np.nan, np.nan, 0, 0
    ns = np.array([g.size for g in clean_groups], dtype=float)
    means = np.array([np.mean(g) for g in clean_groups], dtype=float)
    N = float(np.sum(ns))
    overall = float(np.sum(ns * means) / max(N, 1.0))
    ss_between = float(np.sum(ns * (means - overall) ** 2))
    ss_within = float(np.sum([np.sum((g - m) ** 2) for g, m in zip(clean_groups, means)]))
    df_between = k - 1
    df_within = int(N - k)
    if df_within <= 0 or ss_within <= 0:
        return np.nan, np.nan, df_between, df_within
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    F = ms_between / ms_within
    p = float(1.0 - fdist.cdf(F, df_between, df_within))
    return float(F), float(p), int(df_between), int(df_within)


def permanova_oneway(X: np.ndarray, labels: List[str], n_perm: int = 999, seed: int = 123) -> Tuple[float, float, int, int]:
    """One-way PERMANOVA using squared Euclidean distances.
    Returns (F, p, df_between, df_within).
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n = X.shape[0]
    if n < 3:
        return np.nan, np.nan, 0, 0
    groups = np.array(labels)
    uniq = np.unique(groups)
    k = len(uniq)
    if k < 2:
        return np.nan, np.nan, 0, 0

    # squared Euclidean distance matrix
    G = np.dot(X, X.T)
    sq = np.diag(G)[:, None]
    D2 = sq + sq.T - 2 * G
    D2 = np.maximum(D2, 0.0)

    def _ssw(grps):
        ssw = 0.0
        for g in np.unique(grps):
            idx = np.where(grps == g)[0]
            ng = len(idx)
            if ng <= 1:
                continue
            ssw += np.sum(D2[np.ix_(idx, idx)]) / (2.0 * ng)
        return ssw

    sst = np.sum(D2) / (2.0 * n)
    ssw = _ssw(groups)
    ssb = sst - ssw
    df_between = k - 1
    df_within = n - k
    if df_within <= 0 or ssw <= 0:
        return np.nan, np.nan, df_between, df_within
    F_obs = (ssb / df_between) / (ssw / df_within)

    rng = np.random.RandomState(seed)
    if n_perm <= 0:
        return float(F_obs), np.nan, int(df_between), int(df_within)
    perm_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(groups)
        ssw_p = _ssw(perm)
        ssb_p = sst - ssw_p
        if ssw_p <= 0:
            continue
        F_p = (ssb_p / df_between) / (ssw_p / df_within)
        if F_p >= F_obs:
            perm_ge += 1
    p = (perm_ge + 1) / (n_perm + 1)
    return float(F_obs), float(p), int(df_between), int(df_within)


def hotellings_t2(A: np.ndarray, B: np.ndarray, ridge: float = 1e-6) -> Tuple[float, float, int, int]:
    """Hotelling's T^2 test for two groups. Returns (T2, pval, df1, df2)."""
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D arrays")
    nA, p = A.shape
    nB, pB = B.shape
    if pB != p:
        raise ValueError("A and B must have same number of features")
    if nA <= p or nB <= p:
        return np.nan, np.nan, p, max(nA + nB - p - 1, 0)
    mA = np.mean(A, axis=0)
    mB = np.mean(B, axis=0)
    Sa = np.cov(A, rowvar=False, ddof=1)
    Sb = np.cov(B, rowvar=False, ddof=1)
    Sp = ((nA - 1) * Sa + (nB - 1) * Sb) / max(nA + nB - 2, 1)
    if Sp.ndim == 0:
        Sp = np.array([[Sp]])
    reg = ridge * (np.trace(Sp) / max(p, 1) + 1e-12)
    Sp = Sp + reg * np.eye(p)
    try:
        invSp = np.linalg.inv(Sp)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, p, max(nA + nB - p - 1, 0)
    diff = (mA - mB).reshape(-1, 1)
    T2 = float((nA * nB) / max(nA + nB, 1) * (diff.T @ invSp @ diff).item())
    df1 = p
    df2 = nA + nB - p - 1
    if df2 <= 0:
        return T2, np.nan, df1, df2
    # Convert to F statistic
    F = (df2 * T2) / (df1 * (nA + nB - 2))
    try:
        from scipy.stats import f as fdist
        pval = float(1.0 - fdist.cdf(F, df1, df2))
    except Exception:
        pval = np.nan
    return T2, pval, df1, df2


def region_metrics(z_vol: np.ndarray, sulr_vol: np.ndarray, labels: np.ndarray, roi_ids: List[int], z_thr: float) -> Dict[str, np.ndarray]:
    """Compute per-ROI metrics for a single subject."""
    mean_z = np.full(len(roi_ids), np.nan, dtype=float)
    median_z = np.full(len(roi_ids), np.nan, dtype=float)
    p10_z = np.full(len(roi_ids), np.nan, dtype=float)
    p90_z = np.full(len(roi_ids), np.nan, dtype=float)
    iqr_z = np.full(len(roi_ids), np.nan, dtype=float)
    min_z = np.full(len(roi_ids), np.nan, dtype=float)
    max_z = np.full(len(roi_ids), np.nan, dtype=float)
    range_z = np.full(len(roi_ids), np.nan, dtype=float)
    std_z = np.full(len(roi_ids), np.nan, dtype=float)
    skew_z = np.full(len(roi_ids), np.nan, dtype=float)
    kurt_z = np.full(len(roi_ids), np.nan, dtype=float)
    entropy_z = np.full(len(roi_ids), np.nan, dtype=float)
    frac_z_gt = np.full(len(roi_ids), np.nan, dtype=float)
    mean_sulr = np.full(len(roi_ids), np.nan, dtype=float)
    median_sulr = np.full(len(roi_ids), np.nan, dtype=float)
    p10_sulr = np.full(len(roi_ids), np.nan, dtype=float)
    p90_sulr = np.full(len(roi_ids), np.nan, dtype=float)
    iqr_sulr = np.full(len(roi_ids), np.nan, dtype=float)
    min_sulr = np.full(len(roi_ids), np.nan, dtype=float)
    max_sulr = np.full(len(roi_ids), np.nan, dtype=float)
    range_sulr = np.full(len(roi_ids), np.nan, dtype=float)
    std_sulr = np.full(len(roi_ids), np.nan, dtype=float)
    skew_sulr = np.full(len(roi_ids), np.nan, dtype=float)
    kurt_sulr = np.full(len(roi_ids), np.nan, dtype=float)
    entropy_sulr = np.full(len(roi_ids), np.nan, dtype=float)
    for i, lab in enumerate(roi_ids):
        m = (labels == lab)
        if not m.any():
            continue
        zvals = z_vol[m]
        zvals = zvals[np.isfinite(zvals)]
        if zvals.size == 0:
            continue
        mean_z[i] = float(np.nanmean(zvals))
        median_z[i] = float(np.nanmedian(zvals))
        p10_z[i] = float(np.nanpercentile(zvals, 10.0))
        p90_z[i] = float(np.nanpercentile(zvals, 90.0))
        iqr_z[i] = float(p90_z[i] - p10_z[i])
        min_z[i] = float(np.nanmin(zvals))
        max_z[i] = float(np.nanmax(zvals))
        range_z[i] = float(max_z[i] - min_z[i])
        std_z[i] = float(np.nanstd(zvals))
        if std_z[i] > 0:
            skew_z[i] = float(np.nanmean(((zvals - mean_z[i]) / std_z[i]) ** 3))
            kurt_z[i] = float(np.nanmean(((zvals - mean_z[i]) / std_z[i]) ** 4) - 3.0)
        nbins_z = int(max(8, min(256, np.sqrt(zvals.size))))
        if nbins_z > 1:
            hist_z, _ = np.histogram(zvals, bins=nbins_z, density=False)
            pz = hist_z.astype(np.float64)
            pz = pz[pz > 0]
            pz = pz / pz.sum()
            entropy_z[i] = float(-np.sum(pz * np.log(pz + 1e-12)))
        frac_z_gt[i] = float((zvals > z_thr).sum() / max(zvals.size, 1))
        sul = sulr_vol[m]
        sul = sul[np.isfinite(sul)]
        if sul.size:
            mean_sulr[i] = float(np.nanmean(sul))
            median_sulr[i] = float(np.nanmedian(sul))
            p10_sulr[i] = float(np.nanpercentile(sul, 10.0))
            p90_sulr[i] = float(np.nanpercentile(sul, 90.0))
            iqr_sulr[i] = float(p90_sulr[i] - p10_sulr[i])
            min_sulr[i] = float(np.nanmin(sul))
            max_sulr[i] = float(np.nanmax(sul))
            range_sulr[i] = float(max_sulr[i] - min_sulr[i])
            mu = mean_sulr[i]
            std_sulr[i] = float(np.nanstd(sul))
            if std_sulr[i] > 0:
                skew_sulr[i] = float(np.nanmean(((sul - mu) / std_sulr[i]) ** 3))
                kurt_sulr[i] = float(np.nanmean(((sul - mu) / std_sulr[i]) ** 4) - 3.0)
            # Shannon entropy using adaptive bins
            nbins = int(max(8, min(256, np.sqrt(sul.size))))
            if nbins > 1:
                hist, _ = np.histogram(sul, bins=nbins, density=False)
                p = hist.astype(np.float64)
                p = p[p > 0]
                p = p / p.sum()
                entropy_sulr[i] = float(-np.sum(p * np.log(p + 1e-12)))
        else:
            mean_sulr[i] = np.nan
    return dict(
        mean_z=mean_z,
        median_z=median_z,
        p10_z=p10_z,
        p90_z=p90_z,
        iqr_z=iqr_z,
        min_z=min_z,
        max_z=max_z,
        range_z=range_z,
        std_z=std_z,
        skew_z=skew_z,
        kurt_z=kurt_z,
        entropy_z=entropy_z,
        frac_z_gt=frac_z_gt,
        mean_sulr=mean_sulr,
        median_sulr=median_sulr,
        p10_sulr=p10_sulr,
        p90_sulr=p90_sulr,
        iqr_sulr=iqr_sulr,
        min_sulr=min_sulr,
        max_sulr=max_sulr,
        range_sulr=range_sulr,
        std_sulr=std_sulr,
        skew_sulr=skew_sulr,
        kurt_sulr=kurt_sulr,
        entropy_sulr=entropy_sulr,
    )


def main():
    ap = argparse.ArgumentParser(description="Region-wise baseline comparison between treatment groups")
    ap.add_argument('--data-dir', default='/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/JHU', help='Data dir (CT/SUV files)')
    ap.add_argument('--json-dir', default='/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/clinical_jsons/', help='Clinical JSON dir')
    ap.add_argument('--out-dir', default='output_regionwise_treatment', help='Output directory')
    ap.add_argument('--group-by', default='pre_at',
                    choices=['pre_local','pre_focal','pre_at','pre_cyto','post_local','post_focal','post_at','post_cyto',
                             'pre_any','post_any','any_therapy'],
                    help='Treatment grouping rule')
    ap.add_argument('--z-thr', type=float, default=3.0, help='Z threshold for fraction-of-lesion metric')
    ap.add_argument('--radiomics-space', default='both', choices=['z', 'sulr', 'both'],
                    help='Compute radiomics on z-scores, SULR, or both')
    ap.add_argument('--radiomics-metrics', default='',
                    help='Comma-separated metric names to compute (empty = defaults by space)')
    ap.add_argument('--label-atlas', default='../atlas/seg/ct_seg_atlas_w_reg_40lbls.nii.gz', help='Atlas label map (organ ROIs)')
    ap.add_argument('--use-robust-atlas', action='store_true', help='Use median/MAD atlas (if available or compute)')
    ap.add_argument('--atlas-stats-dir', default='population_stats_feats', help='Folder for mean/std (or median/madstd) atlas stats')
    ap.add_argument('--names-list', default=None, help='Optional text file with subject basenames to include')
    ap.add_argument('--prewarped', action='store_true', help='If set, use prewarped atlas-space volumes')
    ap.add_argument('--image-suffix', default='SULR', help='Suffix for prewarped atlas-space volumes')
    ap.add_argument('--device', default='cuda:0', help='CUDA device for warping')
    ap.add_argument('--print-every', type=int, default=25, help='Progress print interval during warping/feature extraction')
    ap.add_argument('--multivariate', action='store_true', help='Run Hotelling T2 per ROI using selected metrics')
    ap.add_argument('--mv-metrics', default='auto',
                    help='Comma-separated metric names for multivariate test (auto = pick from available metrics)')
    ap.add_argument('--mv-perm', type=int, default=999, help='Permutations for multigroup PERMANOVA')
    ap.add_argument('--multigroup', action='store_true', help='Run multi-group ANOVA per ROI')
    ap.add_argument('--multigroup-by', default='pre_types', choices=['pre_types', 'post_types', 'any_types'],
                    help='Multi-group therapy types grouping rule')
    ap.add_argument('--multigroup-assign', default='first', choices=['exclude', 'first'],
                    help='How to handle multiple therapies: exclude or assign first in order')
    ap.add_argument('--multigroup-include-none', action='store_true', help='Include no-therapy group')
    ap.add_argument('--skip-binary', action='store_true', help='Skip 2-group comparison output/logs')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Subjects
    if args.names_list and os.path.isfile(args.names_list):
        with open(args.names_list, 'r') as f:
            names = [ln.strip() for ln in f if ln.strip()]
    else:
        names = select_baseline_names_by_patient(args.data_dir)
    if len(names) == 0:
        raise RuntimeError("No subjects found. Check --data-dir or --names-list.")
    print(f"[info] Baseline subjects found: {len(names)}")

    ds = JHUPSMADataset(data_path=args.data_dir, data_names=names, data_json=args.json_dir, normalize_covariates=False)
    A, B, detail = get_treatment_groups(ds, names, args.group_by)
    keep = (A | B)
    names_keep = [nm for (nm, m) in zip(names, keep) if m]
    if len(names_keep) < 10:
        raise RuntimeError(f"Too few subjects after grouping (n={len(names_keep)}).")
    A = A[keep]; B = B[keep]
    if not args.skip_binary:
        print(f"[info] Grouping: {detail.get('A_label','A')} n={int(A.sum())} vs {detail.get('B_label','B')} n={int(B.sum())}")

    # Pre-check multigroup availability before spatial normalization
    mg_info = None
    if args.multigroup or (args.multigroup and args.multivariate):
        labels_all, group_order = get_multigroup_labels(
            ds, names_keep,
            group_by=args.multigroup_by,
            assign_mode=args.multigroup_assign,
            include_none=bool(args.multigroup_include_none),
        )
        valid_mask = np.array([lbl is not None for lbl in labels_all], dtype=bool)
        n_valid = int(valid_mask.sum())
        if n_valid < 10:
            raise RuntimeError(f"Too few subjects for multigroup after filtering (n={n_valid}).")
        labels_valid = [lbl for (lbl, m) in zip(labels_all, valid_mask) if m]
        groups_present = [g for g in group_order if g in set(labels_valid)]
        if len(groups_present) < 2:
            raise RuntimeError(f"Need >=2 groups for multigroup analysis, found {groups_present}")
        counts_all = {g: labels_valid.count(g) for g in group_order}
        counts_present = {g: labels_valid.count(g) for g in groups_present}
        print("[info] Multigroup precheck:", json.dumps({
            "multigroup_by": args.multigroup_by,
            "assign": args.multigroup_assign,
            "include_none": bool(args.multigroup_include_none),
            "groups_present": groups_present,
            "counts": counts_present,
            "counts_all": counts_all,
            "n_valid": n_valid,
        }, indent=2))
        mg_info = dict(
            labels=labels_valid,
            group_order=group_order,
            groups_present=groups_present,
            valid_mask=valid_mask,
            n_valid=n_valid,
            counts_all=counts_all,
        )

    # Load/warp atlas-space SULR stack
    if args.prewarped:
        stack, ref = load_prewarped_stack(args.data_dir, names_keep, args.image_suffix)
        mask_3d = np.isfinite(stack[0])
    else:
        stack, ref, aux = build_stack_via_warp(ds, names_keep, device=args.device, print_every=args.print_every)
        mask_3d = aux['mask_3d']
    print(f"[info] Stack ready: shape={stack.shape}, mask voxels={int(mask_3d.sum())}")

    # Atlas stats for z-score
    mean_map, std_map = load_or_compute_atlas_stats(stack, args.atlas_stats_dir, use_robust=args.use_robust_atlas)
    std_map = np.where(np.isfinite(std_map) & (std_map >= 1e-6), std_map, 1.0)

    # Label atlas
    lab_img = nib.load(args.label_atlas)
    labels = np.round(lab_img.get_fdata()).astype(np.int32)
    if labels.shape != stack.shape[1:]:
        raise RuntimeError(f"Label atlas shape {labels.shape} does not match data shape {stack.shape[1:]}")
    roi_ids = sorted([int(v) for v in np.unique(labels) if v != 0])
    roi_names_map = getattr(lbl_ref, 'totalseg_merged_40labels', {})
    print(f"[info] ROI labels loaded: {len(roi_ids)}")

    # Per-subject ROI metrics
    if args.radiomics_space == 'z':
        default_metrics = DEFAULT_Z_METRICS
    elif args.radiomics_space == 'sulr':
        default_metrics = DEFAULT_SULR_METRICS
    else:
        default_metrics = DEFAULT_Z_METRICS + DEFAULT_SULR_METRICS

    if args.radiomics_metrics:
        selected_metrics = [s.strip() for s in args.radiomics_metrics.split(',') if s.strip()]
    else:
        selected_metrics = list(default_metrics)

    metrics = {k: [] for k in selected_metrics}
    total = stack.shape[0]
    for i in range(total):
        sulr = stack[i]
        z = (sulr - mean_map) / std_map
        z = np.where(mask_3d, z, np.nan)
        sulr = np.where(mask_3d, sulr, np.nan)
        m = region_metrics(z, sulr, labels, roi_ids, z_thr=args.z_thr)
        for k in metrics.keys():
            if k not in m:
                raise RuntimeError(f"Unknown radiomics metric: {k}")
            metrics[k].append(m[k])
        if args.print_every and ((i + 1) % args.print_every == 0 or (i + 1) == total):
            print(f"[info]  features {i + 1}/{total}")
    for k in metrics.keys():
        metrics[k] = np.stack(metrics[k], axis=0)  # [N, R]

    # Resolve multivariate metric list once (shared across analyses)
    mv_list = None
    if args.multivariate:
        if args.mv_metrics.strip().lower() == 'auto':
            mv_list = list(metrics.keys())
        else:
            mv_list = [s.strip() for s in args.mv_metrics.split(',') if s.strip()]
        missing = [m for m in mv_list if m not in metrics]
        if missing:
            raise RuntimeError(f"Unknown mv metrics: {missing}. Available: {list(metrics.keys())}")

    # Optional multivariate comparison per ROI (2-group)
    mv_rows = []
    if args.multivariate and not args.skip_binary:
        for k, roi in enumerate(roi_ids):
            XA = np.stack([metrics[m][A, k] for m in mv_list], axis=1)
            XB = np.stack([metrics[m][B, k] for m in mv_list], axis=1)
            XA = XA[np.isfinite(XA).all(axis=1)]
            XB = XB[np.isfinite(XB).all(axis=1)]
            T2, pval, df1, df2 = hotellings_t2(XA, XB)
            mv_rows.append({
                'roi_id': int(roi),
                'roi_name': roi_names_map.get(int(roi), f'label_{int(roi)}'),
                'nA': int(XA.shape[0]),
                'nB': int(XB.shape[0]),
                'df1': int(df1) if np.isfinite(df1) else 0,
                'df2': int(df2) if np.isfinite(df2) else 0,
                'T2': float(T2) if np.isfinite(T2) else np.nan,
                'p': float(pval) if np.isfinite(pval) else np.nan,
                'metrics': ','.join(mv_list),
            })

    # Compare groups per metric with Welch t-test + FDR
    rows = []
    if not args.skip_binary:
        for metric_name, mat in metrics.items():
            vals_A = mat[A]
            vals_B = mat[B]
            # Compute stats per ROI
            t_list = []; p_list = []; nA_list = []; nB_list = []; d_list = []
            meanA_list = []; meanB_list = []; diff_list = []
            for j in range(mat.shape[1]):
                a = vals_A[:, j]; b = vals_B[:, j]
                a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
                nA = int(a.size); nB = int(b.size)
                nA_list.append(nA); nB_list.append(nB)
                if nA < 3 or nB < 3:
                    t_list.append(np.nan); p_list.append(np.nan); d_list.append(np.nan)
                    meanA_list.append(np.nan); meanB_list.append(np.nan); diff_list.append(np.nan)
                    continue
                mA = float(np.mean(a)); mB = float(np.mean(b))
                vA = float(np.var(a, ddof=1)); vB = float(np.var(b, ddof=1))
                t, p, _df = welch_t_and_p(mA, vA, nA, mB, vB, nB)
                # Cohen's d (pooled SD)
                s_pooled = math.sqrt(((nA - 1) * vA + (nB - 1) * vB) / max(nA + nB - 2, 1))
                d = (mA - mB) / s_pooled if s_pooled > 0 else np.nan
                t_list.append(float(t)); p_list.append(float(p)); d_list.append(float(d))
                meanA_list.append(mA); meanB_list.append(mB); diff_list.append(mA - mB)

            t_arr = np.asarray(t_list, dtype=float)
            p_arr = np.asarray(p_list, dtype=float)
            # FDR per metric
            sig05, q05 = benjamini_hochberg(p_arr, q=0.05) if np.isfinite(p_arr).any() else (np.zeros_like(p_arr, dtype=bool), np.ones_like(p_arr))
            sig10, q10 = benjamini_hochberg(p_arr, q=0.10) if np.isfinite(p_arr).any() else (np.zeros_like(p_arr, dtype=bool), np.ones_like(p_arr))

            for k, roi in enumerate(roi_ids):
                rows.append({
                    'metric': metric_name,
                    'roi_id': int(roi),
                    'roi_name': roi_names_map.get(int(roi), f'label_{int(roi)}'),
                    'nA': int(nA_list[k]),
                    'nB': int(nB_list[k]),
                    'meanA': float(meanA_list[k]) if np.isfinite(meanA_list[k]) else np.nan,
                    'meanB': float(meanB_list[k]) if np.isfinite(meanB_list[k]) else np.nan,
                    'diff': float(diff_list[k]) if np.isfinite(diff_list[k]) else np.nan,
                    't': float(t_arr[k]) if np.isfinite(t_arr[k]) else np.nan,
                    'p': float(p_arr[k]) if np.isfinite(p_arr[k]) else np.nan,
                    'q05': float(q05[k]) if np.isfinite(p_arr[k]) else np.nan,
                    'q10': float(q10[k]) if np.isfinite(p_arr[k]) else np.nan,
                    'sig_q05': bool(sig05[k]) if np.isfinite(p_arr[k]) else False,
                    'sig_q10': bool(sig10[k]) if np.isfinite(p_arr[k]) else False,
                    'cohens_d': float(d_list[k]) if np.isfinite(d_list[k]) else np.nan,
                })

    # Multi-group ANOVA per metric (optional)
    mg_rows = []
    if args.multigroup and mg_info:
        labels = mg_info['labels']
        valid_mask = mg_info['valid_mask']
        groups_present = mg_info['groups_present']
        group_order = mg_info['group_order']
        anova_metrics = mv_list if (args.multigroup and args.multivariate and mv_list) else list(metrics.keys())

        if groups_present:
            for metric_name in anova_metrics:
                mat = metrics[metric_name]
                mat = mat[valid_mask]
                # ANOVA per ROI
                p_list = []
                f_list = []
                dfb_list = []
                dfw_list = []
                ns_list = []
                means_list = []
                for j in range(mat.shape[1]):
                    grp_vals = []
                    grp_ns = []
                    grp_means = []
                    for g in group_order:
                        vals = mat[np.array([lbl == g for lbl in labels]), j]
                        vals = vals[np.isfinite(vals)]
                        grp_vals.append(vals)
                        grp_ns.append(int(vals.size))
                        grp_means.append(float(np.mean(vals)) if vals.size else np.nan)
                    F, p, dfb, dfw = oneway_anova(grp_vals)
                    f_list.append(float(F) if np.isfinite(F) else np.nan)
                    p_list.append(float(p) if np.isfinite(p) else np.nan)
                    dfb_list.append(int(dfb))
                    dfw_list.append(int(dfw))
                    ns_list.append(grp_ns)
                    means_list.append(grp_means)
                p_arr = np.asarray(p_list, dtype=float)
                sig05, q05 = benjamini_hochberg(p_arr, q=0.05) if np.isfinite(p_arr).any() else (np.zeros_like(p_arr, dtype=bool), np.ones_like(p_arr))
                sig10, q10 = benjamini_hochberg(p_arr, q=0.10) if np.isfinite(p_arr).any() else (np.zeros_like(p_arr, dtype=bool), np.ones_like(p_arr))

                for k, roi in enumerate(roi_ids):
                    mg_rows.append({
                        'metric': metric_name,
                        'roi_id': int(roi),
                        'roi_name': roi_names_map.get(int(roi), f'label_{int(roi)}'),
                        'groups': ','.join(group_order),
                        'n_by_group': json.dumps(ns_list[k]),
                        'mean_by_group': json.dumps(means_list[k]),
                        'F': float(f_list[k]) if np.isfinite(f_list[k]) else np.nan,
                        'p': float(p_arr[k]) if np.isfinite(p_arr[k]) else np.nan,
                        'q05': float(q05[k]) if np.isfinite(p_arr[k]) else np.nan,
                        'q10': float(q10[k]) if np.isfinite(p_arr[k]) else np.nan,
                        'sig_q05': bool(sig05[k]) if np.isfinite(p_arr[k]) else False,
                        'sig_q10': bool(sig10[k]) if np.isfinite(p_arr[k]) else False,
                        'df_between': int(dfb_list[k]),
                        'df_within': int(dfw_list[k]),
                    })

    # Multi-group multivariate PERMANOVA (optional)
    mg_mv_rows = []
    if args.multigroup and args.multivariate and mg_info:
        labels = mg_info['labels']
        valid_mask = mg_info['valid_mask']
        groups_present = mg_info['groups_present']
        if len(groups_present) >= 2 and valid_mask.sum() >= 5:
            for k, roi in enumerate(roi_ids):
                X = np.stack([metrics[m][valid_mask, k] for m in mv_list], axis=1)
                good = np.isfinite(X).all(axis=1)
                X = X[good]
                lbl = [l for (l, g) in zip(labels, good) if g]
                if len(set(lbl)) < 2 or X.shape[0] < 5:
                    F, p, dfb, dfw = np.nan, np.nan, 0, 0
                else:
                    F, p, dfb, dfw = permanova_oneway(X, lbl, n_perm=args.mv_perm)
                mg_mv_rows.append({
                    'roi_id': int(roi),
                    'roi_name': roi_names_map.get(int(roi), f'label_{int(roi)}'),
                    'groups': ','.join(groups_present),
                    'n': int(X.shape[0]),
                    'df_between': int(dfb),
                    'df_within': int(dfw),
                    'F': float(F) if np.isfinite(F) else np.nan,
                    'p': float(p) if np.isfinite(p) else np.nan,
                    'metrics': ','.join(mv_list),
                    'n_perm': int(args.mv_perm),
                })
        else:
            print(f"[warn] Multigroup PERMANOVA skipped: need >=2 groups with data, found {groups_present}")

    # Save CSV
    out_csv = None
    import csv
    if rows:
        out_csv = os.path.join(args.out_dir, 'regionwise_treatment_stats.csv')
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    mv_csv = None
    if args.multivariate and mv_rows:
        mv_csv = os.path.join(args.out_dir, 'regionwise_multivariate_hotelling.csv')
        with open(mv_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(mv_rows[0].keys()))
            w.writeheader(); w.writerows(mv_rows)

    mg_csv = None
    if args.multigroup and mg_rows:
        mg_csv = os.path.join(args.out_dir, 'regionwise_multigroup_anova.csv')
        with open(mg_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(mg_rows[0].keys()))
            w.writeheader(); w.writerows(mg_rows)

    mg_mv_csv = None
    if args.multigroup and args.multivariate and mg_mv_rows:
        mg_mv_csv = os.path.join(args.out_dir, 'regionwise_multigroup_permanova.csv')
        with open(mg_mv_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(mg_mv_rows[0].keys()))
            w.writeheader(); w.writerows(mg_mv_rows)

    # Summary JSON
    summary = dict(
        N_total=len(names),
        N_used=len(names_keep),
        N_A=int(A.sum()),
        N_B=int(B.sum()),
        group_by=args.group_by,
        group_A_label=detail.get('A_label', 'A'),
        group_B_label=detail.get('B_label', 'B'),
        z_thr=float(args.z_thr),
        radiomics_space=args.radiomics_space,
        radiomics_metrics=selected_metrics,
        skip_binary=bool(args.skip_binary),
        binary_csv=out_csv,
        use_robust_atlas=bool(args.use_robust_atlas),
        atlas_stats_dir=args.atlas_stats_dir,
        label_atlas=args.label_atlas,
        prewarped=bool(args.prewarped),
        image_suffix=args.image_suffix,
        multivariate=bool(args.multivariate),
        mv_metrics=args.mv_metrics,
        mv_perm=int(args.mv_perm),
        multivariate_csv=mv_csv,
        multigroup=bool(args.multigroup),
        multigroup_by=args.multigroup_by,
        multigroup_assign=args.multigroup_assign,
        multigroup_include_none=bool(args.multigroup_include_none),
        multigroup_csv=mg_csv,
        multigroup_multivariate_csv=mg_mv_csv,
    )
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("=== Region-wise treatment comparison done ===")
    print(json.dumps(summary, indent=2))
    if out_csv:
        print(f"Saved CSV: {out_csv}")


if __name__ == '__main__':
    main()
