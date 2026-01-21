#!/usr/bin/env python3
"""
Voxel-wise and region-wise group comparison in atlas space for PSMA PET.

What it does
- Splits the cohort into clinically meaningful groups (RECIP responders vs non-responders,
  new lesion vs none, early relapse vs not, high vs low PSA).
- Loads atlas-space PET volumes (default: *_SUV.nii.gz) for all subjects.
- Performs voxel-wise Welch's t-test between two groups with NaN coverage handling.
- Multiple-comparison control via BH-FDR; optional permutation-based cluster-extent correction.
- Optional region-wise statistics if a region atlas is provided.
- Saves NIfTI maps (t, p, q, effect size, group means, significant masks) and summary CSV/JSON.

Usage (examples)
  python3 -u gen_group_compare_voxelwise.py \
    --data-dir /path/to/atlas_space_volumes \
    --out-dir  /path/to/output/group_compare \
    --group-by recip_any \
    --image-suffix SUV \
    --voxel-p 0.01 --permutations 0

  python3 -u gen_group_compare_voxelwise.py \
    --data-dir /path/to/atlas_space_volumes \
    --out-dir  /path/to/output/newlesion \
    --group-by new_lesion_any \
    --voxel-p 0.01 --permutations 500 --seed 13

Notes
- Region-wise stats: provide --region-atlas to compute label-wise means/diffs/t/p and save a CSV.
- This script reuses dataset parsing and BH-FDR utilities already in this repo.
"""

import os
import sys
import json
import math
import argparse
from typing import Tuple, Dict, Any, List
from scipy import ndimage
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader
from natsort import natsorted

try:
    from scipy.ndimage import label as cc_label
except Exception:
    cc_label = None

try:
    from scipy.stats import t as _student_t
except Exception:
    _student_t = None

# Local imports
from dataset import JHUPSMADataset
from cox_analysis import benjamini_hochberg
from MIR.models import VFA, TemplateCreation
import MIR.models.configs_VFA as CONFIGS_VFA
from MIR.statistical_analysis import robust_bp_ref
from utils import smooth_inside_mask, morph_close3d
from MIR.statistical_analysis.GLM import GLM as TORCH_GLM


def list_subject_basenames(data_dir: str, image_suffix: str) -> list:
    """Find basenames where {name}_{suffix}.nii.gz exists in data_dir."""
    names = []
    for fn in os.listdir(data_dir):
        if not fn.endswith(f"_{image_suffix}.nii.gz"):
            continue
        base = fn[: -len(f"_{image_suffix}.nii.gz")]
        names.append(base)
    names = sorted(list(set(names)))
    return names


def select_baseline_names_by_patient(data_dir: str) -> list:
    """Select one scan per patient (earliest date) using *_CT_seg files as anchors."""
    import glob, re
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


def build_groups(ds: JHUPSMADataset, names: list, group_by: str,
                 early_days: int = 365, psa_split: str = 'median') -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return boolean masks (A, B) over subjects for the chosen grouping and a small info dict.
    Groups are mutually exclusive; subjects with missing criteria are excluded from both.

    group_by options:
      - 'recip_any': responders (RECIP 1–2) vs non-responders (RECIP 3–4), using any reader available.
      - 'recip_r1'  : same but only R1.
      - 'recip_r2'  : same but only R2.
      - 'new_lesion_any': any reader marks new lesion vs none.
      - 'new_lesion_r1' : only R1.
      - 'new_lesion_r2' : only R2.
      - 'early_relapse': relapse within early_days vs otherwise (no relapse or late).
      - 'psa_highlow': high vs low by split (median by default) on log_psa_at_scan.
    """
    cov = ds.covariates_raw  # [N, K]
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

    if group_by.startswith('recip'):
        use_r1 = (group_by in ('recip_any', 'recip_r1'))
        use_r2 = (group_by in ('recip_any', 'recip_r2'))
        for i in range(n):
            r1 = getv(i, 'r1_recip_score', np.nan) if use_r1 else np.nan
            r2 = getv(i, 'r2_recip_score', np.nan) if use_r2 else np.nan
            vals = [v for v in [r1, r2] if np.isfinite(v)]
            if len(vals) == 0:
                continue
            # responders: 1–2, non-responders: 3–4
            any_nonresp = any(v >= 3 for v in vals)
            any_resp = any((v == 1) or (v == 2) for v in vals)
            if any_nonresp and not any_resp:
                B[i] = True  # group B: non-responders
            elif any_resp and not any_nonresp:
                A[i] = True  # group A: responders
            else:
                # conflicting scores (e.g., 2 vs 3) -> exclude; could also choose a tie-breaker.
                pass
        detail["A_label"] = "RECIP 1–2 (responders)"
        detail["B_label"] = "RECIP 3–4 (non-responders)"

    elif group_by.startswith('new_lesion'):
        use_r1 = (group_by in ('new_lesion_any', 'new_lesion_r1'))
        use_r2 = (group_by in ('new_lesion_any', 'new_lesion_r2'))
        for i in range(n):
            r1 = getv(i, 'r1_new_lesion', np.nan) if use_r1 else np.nan
            r2 = getv(i, 'r2_new_lesion', np.nan) if use_r2 else np.nan
            vals = [v for v in [r1, r2] if np.isfinite(v)]
            if len(vals) == 0:
                continue
            any_yes = any(v == 1 for v in vals)
            any_no = any(v == 0 for v in vals)
            if any_yes and not any_no:
                A[i] = True  # A: has new lesion
            elif any_no and not any_yes:
                B[i] = True  # B: no new lesion
            else:
                pass
        detail["A_label"] = "New lesion present"
        detail["B_label"] = "No new lesion"

    elif group_by == 'early_relapse':
        for i in range(n):
            e = getv(i, 'relapse_event', np.nan)
            t = getv(i, 'relapsetime_days', np.nan)
            if not np.isfinite(e) or not np.isfinite(t):
                continue
            if int(e) == 1 and t <= early_days:
                A[i] = True  # early relapse
            else:
                B[i] = True  # no relapse or late relapse
        detail["A_label"] = f"Early relapse (<= {early_days} d)"
        detail["B_label"] = "No/late relapse"

    elif group_by == 'psa_highlow':
        # Use log_psa_at_scan (or fallback). Split by median.
        vals = []
        for i in range(n):
            v = getv(i, 'log_psa_at_scan', np.nan)
            if not np.isfinite(v):
                v = getv(i, 'log_psa_pre', np.nan)
            if not np.isfinite(v):
                v = getv(i, 'log_psa_initial', np.nan)
            vals.append(v)
        vals = np.asarray(vals, dtype=float)
        mask = np.isfinite(vals)
        if psa_split == 'median':
            thr = np.nanmedian(vals[mask])
        else:
            # custom numeric threshold (expects float-able string)
            try:
                thr = float(psa_split)
            except Exception:
                thr = np.nanmedian(vals[mask])
        for i in range(n):
            v = vals[i]
            if not np.isfinite(v):
                continue
            if v >= thr:
                A[i] = True  # high
            else:
                B[i] = True  # low
        detail["A_label"] = f"High PSA (>= {thr:.3f} log-unit)"
        detail["B_label"] = f"Low PSA (< {thr:.3f} log-unit)"

    else:
        raise ValueError(f"Unknown group_by: {group_by}")

    # Drop subjects unassigned to either group
    return A, B, detail


def load_stack(data_dir: str, names: list, suffix: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """Load volumes into a 4D array [N, X, Y, Z]; return stack and a reference NIfTI for affine/header."""
    imgs = []
    ref = None
    for nm in names:
        p = os.path.join(data_dir, f"{nm}_{suffix}.nii.gz")
        if not os.path.isfile(p):
            imgs.append(None)
            continue
        ni = nib.load(p)
        arr = ni.get_fdata().astype(np.float32)
        if ref is None:
            ref = ni
        imgs.append(arr)
    # ensure shapes consistent; missing volumes -> NaN array of ref shape
    if ref is None:
        raise RuntimeError("No volumes found matching the given suffix.")
    shp = ref.shape
    stack = np.full((len(names),) + shp, np.nan, dtype=np.float32)
    for i, arr in enumerate(imgs):
        if arr is None:
            continue
        if arr.shape != shp:
            raise RuntimeError(f"Shape mismatch for {names[i]}: got {arr.shape}, expected {shp}")
        stack[i] = arr
    return stack, ref


def prepare_vfa_and_atlas(device: str = 'cuda:0'):
    """Load VFA model and atlas resources used for on-the-fly warping and masking."""
    # Atlas/model config consistent with Cox script
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
    # pick first file in experiments dir (replicates Cox script behavior)
    ckpts = natsorted(os.listdir(model_dir))
    if not ckpts:
        raise RuntimeError(f"No model checkpoints found in {model_dir}")
    best_model = torch.load(os.path.join(model_dir, ckpts[0]), map_location=device)['state_dict']
    template_model.load_state_dict(best_model)
    reg_model = template_model.reg_model.to(device)
    reg_model.eval()

    # Load atlas volumes and segmentations
    x_ct_ni = nib.load(os.path.join(ct_atlas_dir, natsorted(os.listdir(ct_atlas_dir))[-1]))
    x_ct = torch.from_numpy(x_ct_ni.get_fdata()[None, None, ...]).to(device).float()
    x_pt_ni = nib.load(os.path.join(pt_atlas_dir, natsorted(os.listdir(pt_atlas_dir))[-1]))
    x_pt_aff = x_pt_ni.affine
    x_pt_seg = nib.load(seg_atlas_path)
    x_pt_seg_t = torch.from_numpy(x_pt_seg.get_fdata()[None, None, ...]).to(device).long()
    x_ct_seg = nib.load(ct_seg_atlas_path)
    x_ct_seg_t = torch.from_numpy(x_ct_seg.get_fdata()[None, None, ...]).to(device).long()

    # Masks: liver from PET seg label==1, aorta from CT seg label==7; body mask for analysis
    liver_mask = (x_pt_seg_t == 1).float()  # [1,1,H,W,D]
    mask_3d = ((x_ct_seg_t[0, 0] > 0.01) | (x_ct[0, 0] > 0.01)).bool()
    CLOSE_MASK_RADIUS = 2
    if CLOSE_MASK_RADIUS and CLOSE_MASK_RADIUS > 0:
        mask_3d = morph_close3d(mask_3d, radius=CLOSE_MASK_RADIUS)

    return dict(reg_model=reg_model, x_ct=x_ct, liver_mask=liver_mask,
                mask_3d=mask_3d, pt_affine=x_pt_aff, shape=(H, W, D))


def build_stack_via_warp(ds: JHUPSMADataset, names: list, device: str = 'cuda:0') -> Tuple[np.ndarray, nib.Nifti1Image, Dict[str, np.ndarray]]:
    """Warp each subject's SUV to atlas space and compute SULR with LBM scaling and liver blood-pool normalization,
    smoothing inside body mask. Returns stack [N,H,W,D] with NaNs outside mask, a NIfTI ref, and aux maps used.
    """
    ctx = prepare_vfa_and_atlas(device=device)
    reg_model = ctx['reg_model']
    x_ct = ctx['x_ct']
    liver_mask = ctx['liver_mask']
    mask_3d = ctx['mask_3d']
    H, W, D = ctx['shape']
    # Build a simple NIfTI ref using the PET atlas affine
    ref_img = nib.Nifti1Image(np.zeros((H, W, D), dtype=np.float32), affine=ctx['pt_affine'])

    # Build loader in the given order of names
    sub_ds = JHUPSMADataset(ds.path, names, data_json=ds.data_json, normalize_covariates=False)
    loader = DataLoader(sub_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    stack = np.full((len(names), H, W, D), np.nan, dtype=np.float32)
    covname2idx = sub_ds.covname2idx
    for i, batch in enumerate(loader):
        pat_ct = batch['CT'].to(device).float()         # [1,1,H,W,D] native
        pat_suv_org = batch['SUV_Org'].to(device).float()
        # register to atlas
        with torch.no_grad():
            def_atlas, def_image, pos_flow, neg_flow = reg_model((x_ct.repeat(1,1,1,1,1), pat_ct))
            def_pat_suv = reg_model.spatial_trans(pat_suv_org, neg_flow)  # [1,1,H,W,D] atlas space
            suv_bl = def_pat_suv[0, 0]
        # blood pool normalization from liver mask
        Rb = robust_bp_ref(suv_bl, liver_mask[0, 0])
        if not torch.isfinite(Rb):
            Rb = torch.tensor(1.0, device=suv_bl.device)
        # LBM scaling if available
        h = float(batch['covariates_raw'][0, covname2idx.get('height_m')].item()) if 'height_m' in covname2idx else float('nan')
        w = float(batch['covariates_raw'][0, covname2idx.get('weight_kg')].item()) if 'weight_kg' in covname2idx else float('nan')
        if math.isfinite(h) and math.isfinite(w) and w > 0 and h > 0:
            lbm = 9270.0 * w / (6680.0 + 216.0 * w / (h*h))
            sul = suv_bl * (lbm / (w + 1e-8))
        else:
            sul = suv_bl
        sulr = sul / (Rb + 1e-6)
        # smooth inside body mask
        sulr_s = smooth_inside_mask(sulr, mask_3d.float(), sigma=1.0)
        vol = sulr_s.detach().cpu().numpy()
        # Apply mask: keep inside mask, set outside to NaN
        arr = np.full((H, W, D), np.nan, dtype=np.float32)
        mask_np = mask_3d.cpu().numpy()
        arr[mask_np] = vol[mask_np]
        stack[i] = arr

    return stack, ref_img, dict(mask_3d=mask_3d.cpu().numpy())


def welch_t_and_p(mean1, var1, n1, mean2, var2, n2):
    """Return t-statistic and two-sided p-value for Welch's t-test with fallback if SciPy is absent."""
    # t = (m1 - m2)/sqrt(v1/n1 + v2/n2)
    eps = 1e-12
    denom = np.sqrt(np.maximum(var1 / np.maximum(n1, 1), 0) + np.maximum(var2 / np.maximum(n2, 1), 0) + eps)
    tstat = (mean1 - mean2) / denom
    # Welch-Satterthwaite df
    num = (var1 / np.maximum(n1, 1) + var2 / np.maximum(n2, 1)) ** 2
    den = (np.maximum(var1, 0) ** 2) / (np.maximum(n1, 1) ** 2 * np.maximum(n1 - 1, 1)) + \
          (np.maximum(var2, 0) ** 2) / (np.maximum(n2, 1) ** 2 * np.maximum(n2 - 1, 1))
    df = num / np.maximum(den, eps)
    if _student_t is not None:
        with np.errstate(invalid='ignore'):
            p = 2.0 * _student_t.sf(np.abs(tstat), df)
    else:
        # Normal approximation fallback
        from math import erf, sqrt
        z = np.abs(tstat)
        p = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))
    return tstat, p, df


def voxelwise_ttest(stack: np.ndarray, A: np.ndarray, B: np.ndarray,
                    min_coverage: int = 15, min_std: float = 1e-6) -> Dict[str, np.ndarray]:
    """Compute voxel-wise Welch t-test between group A and B.
    Returns dict with maps: t, p, df, nA, nB, meanA, meanB, diff, d (Cohen's d), std_pooled.
    NaNs in stack are ignored per voxel.
    """
    # Stack: [N, X, Y, Z]
    N = stack.shape[0]
    A = np.asarray(A, dtype=bool)
    B = np.asarray(B, dtype=bool)
    print(f"Group A size: {A.sum()}, Group B size: {B.sum()}", file=sys.stderr)
    if A.sum() == 0 or B.sum() == 0:
        raise RuntimeError("Empty group: ensure both groups have at least one subject.")

    # reshape to [N, V]
    V = int(np.prod(stack.shape[1:]))
    data = stack.reshape(N, V)
    # coverage masks per voxel
    validA = np.isfinite(data[A])
    validB = np.isfinite(data[B])
    nA = validA.sum(axis=0)
    nB = validB.sum(axis=0)
    covmask = (nA >= min_coverage) & (nB >= min_coverage)
    # means/vars with NaN-safe computations
    def nanmean(a, axis):
        with np.errstate(invalid='ignore'):
            s = np.nansum(a, axis=axis)
            c = np.sum(np.isfinite(a), axis=axis)
            c = np.maximum(c, 1)
            return s / c
    def nanvar(a, axis):
        m = nanmean(a, axis)
        with np.errstate(invalid='ignore'):
            diff = a - m
            diff[~np.isfinite(a)] = 0.0
            s2 = np.nansum(diff * diff, axis=axis)
            c = np.sum(np.isfinite(a), axis=axis)
            c = np.maximum(c - 1, 1)
            return s2 / c

    meanA = nanmean(data[A], axis=0)
    meanB = nanmean(data[B], axis=0)
    varA = nanvar(data[A], axis=0)
    varB = nanvar(data[B], axis=0)

    # gate by minimal variability
    std_all = np.sqrt(0.5 * (varA + varB))
    covmask = covmask & (std_all >= min_std)

    tstat = np.full(V, np.nan, dtype=np.float32)
    pval = np.full(V, np.nan, dtype=np.float32)
    df = np.full(V, np.nan, dtype=np.float32)
    if covmask.any():
        ts, ps, dfs = welch_t_and_p(meanA[covmask], varA[covmask], nA[covmask],
                                     meanB[covmask], varB[covmask], nB[covmask])
        tstat[covmask] = ts.astype(np.float32)
        pval[covmask] = ps.astype(np.float32)
        df[covmask] = dfs.astype(np.float32)

    # effect sizes
    diff = (meanA - meanB).astype(np.float32)
    # Cohen's d (Hedges g correction could be added), pooled SD using Welch-style avg
    sd_pooled = np.sqrt(((nA - 1) * varA + (nB - 1) * varB) / np.maximum(nA + nB - 2, 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        d_eff = diff / np.where(sd_pooled > 0, sd_pooled, np.nan)

    out = dict(
        t=tstat.reshape(stack.shape[1:]),
        p=pval.reshape(stack.shape[1:]),
        df=df.reshape(stack.shape[1:]),
        nA=nA.reshape(stack.shape[1:]).astype(np.int16),
        nB=nB.reshape(stack.shape[1:]).astype(np.int16),
        meanA=meanA.reshape(stack.shape[1:]).astype(np.float32),
        meanB=meanB.reshape(stack.shape[1:]).astype(np.float32),
        diff=diff.reshape(stack.shape[1:]),
        d=d_eff.reshape(stack.shape[1:]).astype(np.float32),
        sd_pooled=sd_pooled.reshape(stack.shape[1:]).astype(np.float32),
    )
    return out


def voxelwise_glm_with_covariates(stack: np.ndarray,
                                  A: np.ndarray,
                                  B: np.ndarray,
                                  cov_raw: np.ndarray,
                                  covname2idx: Dict[str, int],
                                  covariate_names: List[str],
                                  standardize: bool = True,
                                  min_coverage: int = 15,
                                  min_std: float = 1e-6) -> Dict[str, np.ndarray]:
    """Voxel-wise GLM adjusting for clinical covariates.
    Builds design X = [1, group, covariates...] and tests the group coefficient.
    Returns maps: t, p, df, nA, nB, meanA, meanB, diff, d, sd_pooled, and beta_group.
    """
    # Prepare data
    N = stack.shape[0]
    V = int(np.prod(stack.shape[1:]))
    data = stack.reshape(N, V)
    A = np.asarray(A, dtype=bool)
    B = np.asarray(B, dtype=bool)
    if A.sum() == 0 or B.sum() == 0:
        raise RuntimeError("Empty group for GLM: both groups must have samples.")

    # Coverage gating identical to t-test path to avoid NaNs in GLM
    validA = np.isfinite(data[A])
    validB = np.isfinite(data[B])
    nA = validA.sum(axis=0)
    nB = validB.sum(axis=0)
    covmask = (nA >= min_coverage) & (nB >= min_coverage)

    # Mean/var for effect sizes and std gating
    def nanmean(a, axis):
        with np.errstate(invalid='ignore'):
            s = np.nansum(a, axis=axis)
            c = np.sum(np.isfinite(a), axis=axis)
            c = np.maximum(c, 1)
            return s / c
    def nanvar(a, axis):
        m = nanmean(a, axis)
        with np.errstate(invalid='ignore'):
            diff = a - m
            diff[~np.isfinite(a)] = 0.0
            s2 = np.nansum(diff * diff, axis=axis)
            c = np.sum(np.isfinite(a), axis=axis)
            c = np.maximum(c - 1, 1)
            return s2 / c

    meanA = nanmean(data[A], axis=0)
    meanB = nanmean(data[B], axis=0)
    varA = nanvar(data[A], axis=0)
    varB = nanvar(data[B], axis=0)
    std_all = np.sqrt(0.5 * (varA + varB))
    covmask = covmask & (std_all >= min_std)

    # Select voxels to analyze with GLM
    vox_idx = np.where(covmask)[0]
    if vox_idx.size == 0:
        # Nothing to test; return empty-like maps
        shp = stack.shape[1:]
        nanmap = np.full(shp, np.nan, dtype=np.float32)
        zeromap = np.zeros(shp, dtype=np.int16)
        return dict(t=nanmap, p=nanmap, df=nanmap, nA=zeromap, nB=zeromap,
                    meanA=nanmap, meanB=nanmap, diff=nanmap, d=nanmap,
                    sd_pooled=nanmap, beta_group=nanmap)

    Y = data[:, vox_idx]  # [N, V_sel]

    # Build design matrix X: intercept + group + covariates
    g = np.zeros(N, dtype=np.float32)
    g[A] = 1.0
    # Collect specified covariates
    cov_cols = []
    cov_used = []
    for nm in (covariate_names or []):
        j = covname2idx.get(nm, None)
        if j is None:
            print(f"[warn][GLM] covariate '{nm}' not found; skipping.")
            continue
        col = cov_raw[:, j].astype(np.float32)
        cov_cols.append(col)
        cov_used.append(nm)
    if len(cov_cols) > 0:
        C = np.stack(cov_cols, axis=1)  # [N, P]
        # Mean-impute NaNs per column
        col_means = np.nanmean(C, axis=0)
        inds = np.where(~np.isfinite(C))
        if len(inds[0]) > 0:
            C[inds] = np.take(col_means, inds[1])
        # Optional standardization
        if standardize:
            mu = C.mean(axis=0, keepdims=True)
            sd = C.std(axis=0, keepdims=True)
            sd[sd < 1e-8] = 1.0
            C = (C - mu) / sd
        X = np.column_stack([np.ones(N, dtype=np.float32), g, C])
    else:
        X = np.column_stack([np.ones(N, dtype=np.float32), g])

    # Torch GLM on CPU to conserve GPU mem
    X_t = torch.from_numpy(X.astype(np.float32))
    Y_t = torch.from_numpy(Y.astype(np.float32))

    betas, t_maps = TORCH_GLM(X_t, Y_t)
    # Column indices: 0=intercept, 1=group, others=covariates
    t_group = t_maps[1].detach().cpu().numpy().astype(np.float32)  # [V_sel]

    # Degrees of freedom: N - rank(X)
    try:
        rank = int(torch.linalg.matrix_rank(X_t).item())
    except Exception:
        rank = np.linalg.matrix_rank(X)
    df_val = max(int(N - rank), 1)
    if _student_t is not None:
        with np.errstate(invalid='ignore'):
            p_group = 2.0 * _student_t.sf(np.abs(t_group), df_val)
    else:
        from math import erf, sqrt
        z = np.abs(t_group)
        p_group = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))

    # Beta for group effect
    beta_group = betas[1].detach().cpu().numpy().astype(np.float32)  # [V_sel]

    # Effect sizes for reference
    sd_pooled = np.sqrt(((nA - 1) * varA + (nB - 1) * varB) / np.maximum(nA + nB - 2, 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        d_eff = (meanA - meanB) / np.where(sd_pooled > 0, sd_pooled, np.nan)

    # Scatter back into full-size maps
    shp = stack.shape[1:]
    t_full = np.full(V, np.nan, dtype=np.float32)
    p_full = np.full(V, np.nan, dtype=np.float32)
    df_full = np.full(V, np.nan, dtype=np.float32)
    beta_full = np.full(V, np.nan, dtype=np.float32)
    t_full[vox_idx] = t_group
    p_full[vox_idx] = p_group
    df_full[vox_idx] = float(df_val)
    beta_full[vox_idx] = beta_group

    out = dict(
        t=t_full.reshape(shp),
        p=p_full.reshape(shp),
        df=df_full.reshape(shp),
        nA=nA.reshape(shp).astype(np.int16),
        nB=nB.reshape(shp).astype(np.int16),
        meanA=meanA.reshape(shp).astype(np.float32),
        meanB=meanB.reshape(shp).astype(np.float32),
        diff=(meanA - meanB).reshape(shp).astype(np.float32),
        d=d_eff.reshape(shp).astype(np.float32),
        sd_pooled=sd_pooled.reshape(shp).astype(np.float32),
        beta_group=beta_full.reshape(shp).astype(np.float32),
    )
    return out


def cluster_correction(p_map: np.ndarray, voxel_p: float, A: np.ndarray, B: np.ndarray,
                       stack: np.ndarray, n_perm: int = 0, seed: int = 0,
                       connectivity: int = 26) -> Dict[str, Any]:
    """Simple cluster-extent correction using permutations.
    - Threshold p_map at voxel_p to define suprathreshold clusters.
    - If n_perm > 0 and SciPy ndimage is available, compute the null distribution of max cluster size
      by shuffling group labels and recomputing t-tests; return the cluster-size 95th percentile as
      the extent threshold, and produce a cluster-corrected mask. If unavailable, returns only the
      uncorrected mask.
    Returns dict with: mask_uncorr, labels_uncorr, extent_thr, mask_corr, labels_corr.
    """
    if cc_label is None:
        # No connected components available
        supr = (p_map <= voxel_p) & np.isfinite(p_map)
        return dict(mask_uncorr=supr, labels_uncorr=None, extent_thr=None,
                    mask_corr=None, labels_corr=None)

    # Uncorrected suprathreshold clusters
    structure = None
    if connectivity == 6:
        structure = np.array([[[0,0,0],[0,1,0],[0,0,0]],
                              [[0,1,0],[1,1,1],[0,1,0]],
                              [[0,0,0],[0,1,0],[0,0,0]]], dtype=np.int8)
    elif connectivity in (18, 26):
        structure = np.ones((3,3,3), dtype=np.int8)
    supr = (p_map <= voxel_p) & np.isfinite(p_map)
    labels_uncorr, nlab = cc_label(supr.astype(np.int8), structure=structure)

    if n_perm <= 0:
        return dict(mask_uncorr=supr, labels_uncorr=labels_uncorr, extent_thr=None,
                    mask_corr=None, labels_corr=None)

    # Build permutation null of max cluster size
    rng = np.random.default_rng(seed)
    N = stack.shape[0]
    idx_all = np.arange(N)
    A0 = np.asarray(A, dtype=bool)
    B0 = np.asarray(B, dtype=bool)
    nA = int(A0.sum()); nB = int(B0.sum())
    if nA == 0 or nB == 0:
        return dict(mask_uncorr=supr, labels_uncorr=labels_uncorr, extent_thr=None,
                    mask_corr=None, labels_corr=None)

    V = int(np.prod(stack.shape[1:]))
    data = stack.reshape(N, V)
    max_sizes = []
    for _ in range(int(n_perm)):
        rng.shuffle(idx_all)
        A_perm = np.zeros(N, dtype=bool); A_perm[idx_all[:nA]] = True
        B_perm = ~A_perm
        # fast means/vars
        validA = np.isfinite(data[A_perm]); validB = np.isfinite(data[B_perm])
        nAp = validA.sum(axis=0); nBp = validB.sum(axis=0)
        # If either group too small at a voxel, skip by setting p=1 there
        with np.errstate(invalid='ignore'):
            mAp = np.nansum(np.where(validA, data[A_perm], 0.0), axis=0) / np.maximum(nAp, 1)
            mBp = np.nansum(np.where(validB, data[B_perm], 0.0), axis=0) / np.maximum(nBp, 1)
            diff = mAp - mBp
            vAp = np.nansum(np.where(validA, (data[A_perm]-mAp)**2, 0.0), axis=0) / np.maximum(nAp-1, 1)
            vBp = np.nansum(np.where(validB, (data[B_perm]-mBp)**2, 0.0), axis=0) / np.maximum(nBp-1, 1)
            denom = np.sqrt(np.maximum(vAp/np.maximum(nAp,1),0) + np.maximum(vBp/np.maximum(nBp,1),0) + 1e-12)
            t = diff / denom
        # Use normal approx for speed to map t -> p
        from math import erf, sqrt
        z = np.abs(t)
        p = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))
        supr_perm = (p.reshape(stack.shape[1:]) <= voxel_p)
        _, nlab_perm = cc_label(supr_perm.astype(np.int8), structure=structure)
        if nlab_perm == 0:
            max_sizes.append(0)
        else:
            # compute sizes via bincount
            labs = cc_label(supr_perm.astype(np.int8), structure=structure)[0]
            bc = np.bincount(labs.ravel())
            max_sizes.append(int(bc[1:].max()) if bc.size > 1 else 0)
    extent_thr = int(np.percentile(max_sizes, 95)) if len(max_sizes) else None
    if extent_thr is None or extent_thr <= 0:
        return dict(mask_uncorr=supr, labels_uncorr=labels_uncorr, extent_thr=None,
                    mask_corr=None, labels_corr=None)
    # Apply extent threshold to observed clusters
    if nlab == 0:
        return dict(mask_uncorr=supr, labels_uncorr=labels_uncorr, extent_thr=extent_thr,
                    mask_corr=np.zeros_like(supr, dtype=bool), labels_corr=None)
    bc_obs = np.bincount(labels_uncorr.ravel())
    keep = np.where(bc_obs >= extent_thr)[0]
    keep = keep[keep != 0]
    mask_corr = np.isin(labels_uncorr, keep)
    return dict(mask_uncorr=supr, labels_uncorr=labels_uncorr, extent_thr=extent_thr,
                mask_corr=mask_corr, labels_corr=None)


def tfce_transform(stat_map: np.ndarray, mask: np.ndarray = None, E: float = 0.5, H: float = 2.0,
                   dh: float = 0.1, connectivity: int = 26, two_sided: bool = True) -> np.ndarray:
    """Threshold-Free Cluster Enhancement (TFCE) for a 3D statistic map.
    Returns TFCE-enhanced map. If two_sided, combine positive and negative TFCE maps.
    """
    if cc_label is None:
        raise RuntimeError("TFCE requires scipy.ndimage.label (SciPy). Please install SciPy.")
    S = stat_map.copy()
    if mask is None:
        mask = np.isfinite(S)
    else:
        mask = mask & np.isfinite(S)
    S = np.where(mask, S, 0.0)

    def _tfce_one(img_pos):
        m = img_pos > 0
        if not m.any():
            return np.zeros_like(img_pos, dtype=np.float32)
        t_max = float(img_pos.max())
        if t_max <= 0:
            return np.zeros_like(img_pos, dtype=np.float32)
        # connectivity structure
        structure = np.ones((3,3,3), dtype=np.int8) if connectivity in (18,26) else None
        out = np.zeros_like(img_pos, dtype=np.float32)
        h = dh
        while h <= t_max + 1e-8:
            thr = img_pos >= h
            if thr.any():
                labels, nlab = cc_label(thr.astype(np.int8), structure=structure)
                if nlab > 0:
                    # cluster sizes
                    bc = np.bincount(labels.ravel())
                    if bc.size > 1:
                        # For each cluster label, add contribution (size^E * h^H * dh)
                        contrib = (bc ** E) * (h ** H) * dh
                        # Map per-voxel by indexing contrib[labels]
                        out += contrib[labels].astype(out.dtype)
            h += dh
        return out

    if two_sided:
        pos = np.clip(S, 0, None)
        neg = np.clip(-S, 0, None)
        tfce_pos = _tfce_one(pos)
        tfce_neg = _tfce_one(neg)
        # Combine as signed TFCE: positive minus negative
        tfce_map = tfce_pos - tfce_neg
    else:
        tfce_map = _tfce_one(S)
    # Zero outside mask
    tfce_map = np.where(mask, tfce_map, 0.0).astype(np.float32)
    return tfce_map


def tfce_permutation_pvals(stack: np.ndarray,
                           A: np.ndarray,
                           B: np.ndarray,
                           t_obs: np.ndarray,
                           mask_valid: np.ndarray,
                           model: str = 'ttest',
                           glm_ctx: Dict[str, Any] = None,
                           n_perm: int = 0,
                           seed: int = 0,
                           E: float = 0.5,
                           H: float = 2.0,
                           dh: float = 0.1,
                           connectivity: int = 26,
                           two_sided: bool = True,
                           min_coverage: int = 15,
                           min_std: float = 1e-6) -> Dict[str, np.ndarray]:
    """Compute TFCE map for observed t and FWE-corrected p-values via permutation of labels.
    For each permutation, compute t-map and TFCE, record max |TFCE| to form null.
    p_fwe(voxel) = Pr_null(max|TFCE| >= |TFCE_obs(voxel)|).
    model: 'ttest' or 'glm'. For 'glm', glm_ctx must include keys:
        cov_raw_keep (N,K), covname2idx, covariate_names (list[str]), standardize (bool).
    """
    rng = np.random.default_rng(seed)
    N = stack.shape[0]
    V = int(np.prod(stack.shape[1:]))
    data = stack.reshape(N, V)

    # Observed TFCE
    tfce_obs = tfce_transform(t_obs, mask=mask_valid.reshape(stack.shape[1:]), E=E, H=H, dh=dh,
                              connectivity=connectivity, two_sided=two_sided)
    tfce_obs_flat = tfce_obs.reshape(-1)

    if n_perm <= 0:
        return dict(tfce_map=tfce_obs.astype(np.float32), tfce_p_fwe=None)

    # Helper to compute t under permutation
    def compute_t_ttest(A_mask: np.ndarray) -> np.ndarray:
        validA = np.isfinite(data[A_mask])
        validB = np.isfinite(data[~A_mask])
        nA = validA.sum(axis=0)
        nB = validB.sum(axis=0)
        covmask = (nA >= min_coverage) & (nB >= min_coverage)
        # means/vars
        with np.errstate(invalid='ignore'):
            mA = np.nansum(np.where(validA, data[A_mask], 0.0), axis=0) / np.maximum(nA, 1)
            mB = np.nansum(np.where(validB, data[~A_mask], 0.0), axis=0) / np.maximum(nB, 1)
            vA = np.nansum(np.where(validA, (data[A_mask]-mA)**2, 0.0), axis=0) / np.maximum(nA-1, 1)
            vB = np.nansum(np.where(validB, (data[~A_mask]-mB)**2, 0.0), axis=0) / np.maximum(nB-1, 1)
            denom = np.sqrt(np.maximum(vA/np.maximum(nA,1),0) + np.maximum(vB/np.maximum(nB,1),0) + 1e-12)
            t = (mA - mB) / denom
        # std gating
        std_all = np.sqrt(0.5 * (vA + vB))
        covmask &= (std_all >= min_std)
        t[~covmask] = 0.0
        t[~np.isfinite(t)] = 0.0
        return t

    def compute_t_glm(A_mask: np.ndarray) -> np.ndarray:
        # Use same selected voxels as mask_valid
        idx_sel = np.where(mask_valid.ravel())[0]
        if idx_sel.size == 0:
            return np.zeros(V, dtype=np.float32)
        Y = data[:, idx_sel]
        g = np.zeros(N, dtype=np.float32)
        g[A_mask] = 1.0
        cov_raw_keep = glm_ctx['cov_raw_keep']
        covname2idx = glm_ctx['covname2idx']
        covariate_names = glm_ctx['covariate_names']
        standardize = glm_ctx['standardize']
        cov_cols = []
        for nm in (covariate_names or []):
            j = covname2idx.get(nm, None)
            if j is None:
                continue
            col = cov_raw_keep[:, j].astype(np.float32)
            # mean impute
            m = np.nanmean(col)
            col = np.where(np.isfinite(col), col, m)
            cov_cols.append(col)
        if len(cov_cols) > 0:
            C = np.stack(cov_cols, axis=1)
            if standardize:
                mu = C.mean(axis=0, keepdims=True)
                sd = C.std(axis=0, keepdims=True); sd[sd<1e-8] = 1.0
                C = (C - mu) / sd
            X = np.column_stack([np.ones(N, dtype=np.float32), g, C])
        else:
            X = np.column_stack([np.ones(N, dtype=np.float32), g])
        X_t = torch.from_numpy(X.astype(np.float32))
        Y_t = torch.from_numpy(Y.astype(np.float32))
        betas, t_maps = TORCH_GLM(X_t, Y_t)
        t_group = t_maps[1].detach().cpu().numpy().astype(np.float32)
        t_full = np.zeros(V, dtype=np.float32)
        t_full[idx_sel] = t_group
        # zero out outside mask_valid already implied
        return t_full

    use_glm = (model == 'glm')
    max_null = []
    idx_all = np.arange(N)
    nA = int(A.sum());
    if nA == 0 or nA == N:
        return dict(tfce_map=tfce_obs.astype(np.float32), tfce_p_fwe=None)

    for _ in range(int(n_perm)):
        rng.shuffle(idx_all)
        A_perm = np.zeros(N, dtype=bool); A_perm[idx_all[:nA]] = True
        if use_glm:
            t_perm = compute_t_glm(A_perm)
        else:
            t_perm = compute_t_ttest(A_perm)
        tfce_perm = tfce_transform(t_perm.reshape(stack.shape[1:]), mask=mask_valid.reshape(stack.shape[1:]),
                                   E=E, H=H, dh=dh, connectivity=connectivity, two_sided=two_sided)
        max_null.append(float(np.nanmax(np.abs(tfce_perm))))

    max_null = np.asarray(max_null, dtype=np.float32)
    # Compute FWE-corrected p-values via max-null
    max_null_sorted = np.sort(max_null)
    abs_obs = np.abs(tfce_obs_flat)
    # For each voxel, p = (1 + #null >= obs) / (n_perm + 1)
    # Use searchsorted to vectorize
    # Position of first value >= obs in sorted null
    idx = np.searchsorted(max_null_sorted, abs_obs, side='left')
    # Count of null >= obs = n_perm - idx + (null equals obs handled by side='left')
    ge = (n_perm - idx)
    p_fwe = (1.0 + ge) / (n_perm + 1.0)
    p_fwe = p_fwe.reshape(stack.shape[1:]).astype(np.float32)
    return dict(tfce_map=tfce_obs.astype(np.float32), tfce_p_fwe=p_fwe)


def save_nifti_like(ref_img: nib.Nifti1Image, arr: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ni = nib.Nifti1Image(arr.astype(np.float32), affine=ref_img.affine, header=ref_img.header)
    nib.save(ni, out_path)


def regionwise_stats(region_atlas_path: str, ref_img: nib.Nifti1Image,
                     maps: Dict[str, np.ndarray], out_csv: str):
    import csv
    atlas = nib.load(region_atlas_path).get_fdata()
    atlas = np.round(atlas).astype(np.int32)
    labels = np.unique(atlas)
    labels = labels[labels != 0]
    rows = []
    for lab in labels:
        m = (atlas == lab)
        def mean_of(x):
            v = x[m]
            v = v[np.isfinite(v)]
            return float(np.nanmean(v)) if v.size else np.nan
        rows.append(dict(
            label=int(lab),
            meanA=mean_of(maps['meanA']),
            meanB=mean_of(maps['meanB']),
            diff=mean_of(maps['diff']),
            t=mean_of(maps['t']),
            p=mean_of(maps['p']),
            q=mean_of(maps['q']),
            d=mean_of(maps['d']),
            nA=int(np.nanmean(maps['nA'][m])) if np.isfinite(maps['nA'][m]).any() else 0,
            nB=int(np.nanmean(maps['nB'][m])) if np.isfinite(maps['nB'][m]).any() else 0,
        ))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Voxel-wise/region-wise group comparison in atlas space")
    ap.add_argument('--data-dir', required=False, help='Folder with native-space {name}_{suffix}.nii.gz volumes (CT/SUV/seg)')
    ap.add_argument('--out-dir', required=False, help='Output directory')
    ap.add_argument('--image-suffix', default='SUV', help='Which image to analyze per subject (default: SUV)')
    ap.add_argument('--group-by', required=False,
                    choices=['recip_any','recip_r1','recip_r2','new_lesion_any','new_lesion_r1','new_lesion_r2',
                             'early_relapse','psa_highlow'],
                    help='Clinical grouping rule')
    ap.add_argument('--names-list', default=None, help='Optional text file with subject basenames to include (one per line)')
    ap.add_argument('--json-dir', default='/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/clinical_jsons/',
                    help='Folder containing {name}.json clinical files (defaults to project path)')
    ap.add_argument('--min-coverage', type=int, default=15, help='Min subjects per group per voxel')
    ap.add_argument('--min-std', type=float, default=1e-6, help='Min pooled std to test a voxel')
    ap.add_argument('--voxel-p', type=float, default=0.01, help='Primary voxel p-threshold for cluster maps')
    ap.add_argument('--permutations', type=int, default=0, help='Number of permutations for cluster-extent correction (0 to skip)')
    ap.add_argument('--seed', type=int, default=0, help='Random seed for permutations')
    ap.add_argument('--early-days', type=int, default=365, help='Threshold in days for early relapse grouping')
    ap.add_argument('--psa-split', default='median', help='PSA split: "median" or numeric threshold (on log_psa)')
    ap.add_argument('--region-atlas', default=None, help='Optional region atlas NIfTI in atlas space (integer labels) for ROI stats')
    ap.add_argument('--prefilter-p', type=float, default=None, help='Optional uncorrected p-value prefilter before FDR (e.g. 0.01)')
    ap.add_argument('--prefilter-d', type=float, default=None, help='Optional absolute effect size (|d|) prefilter before FDR (e.g. 0.30)')
    ap.add_argument('--prewarped', action='store_true', help='If set, assume inputs are already in atlas space and skip warping')
    ap.add_argument('--device', default='cuda:0', help='CUDA device for warping (default cuda:0)')
    # GLM options
    ap.add_argument('--model', choices=['ttest','glm'], default='ttest', help='Statistical model: voxel-wise Welch t-test or GLM with covariates')
    ap.add_argument('--glm-covars', default=None, help='Comma-separated covariate names to adjust for (e.g., "age,log_psa_at_scan,gleason_total")')
    ap.add_argument('--glm-standardize', action='store_true', help='Standardize covariates (zero-mean, unit-variance)')
    # TFCE options
    ap.add_argument('--tfce', action='store_true', help='Enable Threshold-Free Cluster Enhancement (TFCE) on the t-map')
    ap.add_argument('--tfce-permutations', type=int, default=0, help='Number of permutations for TFCE FWE p-values (0 to skip)')
    ap.add_argument('--tfce-E', type=float, default=0.5, dest='tfce_E', help='TFCE extent exponent E (default 0.5)')
    ap.add_argument('--tfce-H', type=float, default=2.0, dest='tfce_H', help='TFCE height exponent H (default 2.0)')
    ap.add_argument('--tfce-dh', type=float, default=0.1, help='TFCE threshold step dh (default 0.1)')
    ap.add_argument('--tfce-two-sided', action='store_true', help='Use two-sided TFCE (separate pos/neg)')
    ap.add_argument('--roi-mask', default=None, help='Optional atlas-space NIfTI mask (binary) restricting analysis to ROI voxels')
    args = ap.parse_args()

    # Defaults for your environment if not provided
    if not args.data_dir:
        args.data_dir = '/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/JHU'
    if not args.out_dir:
        args.out_dir = os.path.join(os.path.dirname(__file__), 'output_recip_any')
    if not args.group_by:
        args.group_by = 'recip_any'
    if not args.psa_split:
        args.psa_split = 'median'
    if args.prefilter_p is None and args.prefilter_d is None:
        args.prefilter_p = None  # default prefilter to speed up FDR
        args.prefilter_d = None
    args.model = 'glm'
    args.glm_covars = 'age,log_psa_at_scan,gleason_total'
    args.glm_standardize = True
    args.tfce = True
    args.tfce_two_sided = True
    args.tfce_permutations = 0
    args.seed = 13
    args.roi_mask = '/scratch/jchen/python_projects/custom_packages/MIR/tutorials/PSMA_atlas_building/atlas/seg/ct_seg_atlas_w_reg_40lbls.nii.gz'
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Subjects
    if args.names_list and os.path.isfile(args.names_list):
        with open(args.names_list, 'r') as f:
            names = [ln.strip() for ln in f if ln.strip()]
    else:
        # Choose one baseline scan per patient for fair group contrasts
        names = select_baseline_names_by_patient(args.data_dir)
    if len(names) == 0:
        raise RuntimeError("No subjects found. Check --data-dir and --image-suffix or provide --names-list.")

    # Clinical dataset
    ds = JHUPSMADataset(data_path=args.data_dir, data_names=names, data_json=args.json_dir, normalize_covariates=False)

    # Build groups
    A, B, detail = build_groups(ds, names, args.group_by, early_days=args.early_days, psa_split=args.psa_split)
    keep = (A | B)
    names_keep = [nm for (nm, m) in zip(names, keep) if m]
    if len(names_keep) < 10:
        raise RuntimeError(f"Too few subjects after grouping (n={len(names_keep)}).")
    A = A[keep]; B = B[keep]

    # Load stack for kept subjects: warp to atlas and compute SULR unless prewarped
    if args.prewarped:
        stack, ref = load_stack(args.data_dir, names_keep, args.image_suffix)
    else:
        stack, ref, aux = build_stack_via_warp(ds, names_keep, device=args.device)

    # Optional ROI mask restriction (atlas space)
    roi_mask = None
    if args.roi_mask:
        #if not os.path.isfile(args.roi_mask):
        #    raise RuntimeError(f"ROI mask not found: {args.roi_mask}")
        #roi_ni = nib.load(args.roi_mask)
        #roi_arr = roi_ni.get_fdata()
        #if roi_arr.shape != stack.shape[1:]:
        #    raise RuntimeError(f"ROI mask shape {roi_arr.shape} does not match data shape {stack.shape[1:]}")
        # Accept any positive value as inside ROI
        #roi_mask = roi_arr > 0.5
        atlas_labels = '/scratch/jchen/python_projects/custom_packages/MIR/tutorials/PSMA_atlas_building/atlas/seg/ct_seg_atlas_w_reg_40lbls.nii.gz'
        # Load atlas for reference
        atlas_img = nib.load(atlas_labels).get_fdata()
        roi_mask = (atlas_img == 15) #+ (atlas_img == 27) + (atlas_img == 28) + (atlas_img == 29) + (atlas_img == 30) + (atlas_img == 31) + (atlas_img == 32) # bone label in this atlas
        roi_mask = roi_mask > 0
        roi_mask = ndimage.binary_dilation(roi_mask, iterations=2)
        # Apply: set outside-ROI voxels to NaN for all subjects
        outside = ~roi_mask
        stack[:, outside] = np.nan
        print(f"[info] Applied ROI mask: kept {int(roi_mask.sum())} voxels ({roi_mask.sum()/roi_mask.size*100:.2f}% of volume)")

    # Build covariate subset aligned to names_keep
    cov_raw_keep = ds.covariates_raw[keep]

    # Choose model
    if args.model == 'glm':
        covar_list = None
        if args.glm_covars is not None and len(args.glm_covars.strip()) > 0:
            covar_list = [s.strip() for s in args.glm_covars.split(',') if s.strip()]
        else:
            covar_list = []
        print(f"[info] Running GLM with covariates: {covar_list}")
        maps = voxelwise_glm_with_covariates(
            stack=stack,
            A=A,
            B=B,
            cov_raw=cov_raw_keep,
            covname2idx=ds.covname2idx,
            covariate_names=covar_list,
            standardize=bool(args.glm_standardize),
            min_coverage=args.min_coverage,
            min_std=args.min_std,
        )
    else:
        # Voxel-wise t-tests
        maps = voxelwise_ttest(stack, A, B, min_coverage=args.min_coverage, min_std=args.min_std)

    # --- Diagnostics before FDR ---
    p_flat = maps['p'].ravel()
    finite_mask = np.isfinite(p_flat)
    n_finite = int(finite_mask.sum())
    n_total = p_flat.size
    if n_finite == 0:
        print('[diagnostic] No finite p-values. Likely all voxels failed coverage/STD gating. Consider lowering --min-coverage or --min-std.')
    else:
        # Quick uncorrected counts
        p10 = int((p_flat[finite_mask] < 0.10).sum())
        p05 = int((p_flat[finite_mask] < 0.05).sum())
        p01 = int((p_flat[finite_mask] < 0.01).sum())
        p001 = int((p_flat[finite_mask] < 0.001).sum())
        print(f'[diagnostic] Tested voxels (finite p): {n_finite}/{n_total}')
        print(f'[diagnostic] Uncorrected counts: p<0.10={p10}, p<0.05={p05}, p<0.01={p01}, p<0.001={p001}')
        if p05 == 0:
            print('[diagnostic] No voxels pass uncorrected p<0.05; FDR mask will almost certainly be empty.')
        # Coverage distribution
        covA = maps['nA'].ravel(); covB = maps['nB'].ravel()
        cov_valid = np.isfinite(p_flat)  # only those tested
        if cov_valid.any():
            print('[diagnostic] Coverage A median/mean:', np.median(covA[cov_valid]), np.mean(covA[cov_valid]))
            print('[diagnostic] Coverage B median/mean:', np.median(covB[cov_valid]), np.mean(covB[cov_valid]))
        # Effect size preview
        d_flat = maps['d'].ravel()
        d_valid = d_flat[np.isfinite(d_flat) & np.isfinite(p_flat)]
        if d_valid.size:
            print('[diagnostic] Effect size d median/95th:', np.median(d_valid), np.percentile(d_valid,95))

    # FDR correction (optionally after prefiltering to reduce multiplicity)
    pvec = maps['p'].ravel()
    dvec = maps['d'].ravel()
    prefilter_mask = np.isfinite(pvec)
    if args.prefilter_p is not None:
        prefilter_mask &= (pvec <= args.prefilter_p)
    if args.prefilter_d is not None:
        prefilter_mask &= (np.abs(dvec) >= args.prefilter_d)

    m_before = int(np.isfinite(pvec).sum())
    m_after = int(prefilter_mask.sum())
    if m_after == 0 and m_before > 0:
        print(f'[diagnostic] Prefilter removed all voxels (m_before={m_before}). Consider relaxing --prefilter-p/--prefilter-d.')

    def apply_fdr_local(p_full, mask, q):
        # Apply BH only on masked subset; others get q=1 and sig=False
        sig_out = np.zeros_like(p_full, dtype=bool)
        q_out = np.ones_like(p_full, dtype=float)
        if mask.any():
            sig_sub, q_sub = benjamini_hochberg(p_full[mask], q=q)
            sig_out[mask] = sig_sub
            q_out[mask] = q_sub
        return sig_out, q_out

    sig, qvals = apply_fdr_local(pvec, prefilter_mask, q=0.05)
    maps['q'] = qvals.reshape(maps['p'].shape).astype(np.float32)
    maps['fdr_mask_q05'] = sig.reshape(maps['p'].shape)
    sig10, qvals10 = apply_fdr_local(pvec, prefilter_mask, q=0.10)
    maps['q_q10'] = qvals10.reshape(maps['p'].shape).astype(np.float32)
    maps['fdr_mask_q10'] = sig10.reshape(maps['p'].shape)

    # Warn if FDR mask empty but uncorrected differences exist
    if not sig.any() and np.isfinite(pvec).any():
        min_p = np.nanmin(pvec)
        print(f'[diagnostic] FDR q<=0.05 mask empty. Smallest p={min_p:.3e}. Multiplicity m={m_before} (prefiltered m={m_after}). Options: reduce mask (effect-size filter), report uncorrected + effect sizes, try cluster permutations.')
    else:
        print(f'[diagnostic] FDR q<=0.05 significant voxels: {int(sig.sum())}')
    if not sig10.any():
        print('[diagnostic] FDR q<=0.10 mask also empty.')
    else:
        print(f'[diagnostic] FDR q<=0.10 significant voxels: {int(sig10.sum())}')
    print(f'[diagnostic] FDR multiplicity m_before={m_before}, m_after_prefilter={m_after}')

    # Cluster correction (optional)
    cl = cluster_correction(maps['p'], voxel_p=args.voxel_p, A=A, B=B, stack=stack,
                            n_perm=args.permutations, seed=args.seed)

    # Save NIfTI maps
    def save_map(nm, arr):
        save_nifti_like(ref, arr, os.path.join(args.out_dir, f"{nm}.nii.gz"))

    save_map('t_map', maps['t'])
    save_map('p_map', maps['p'])
    save_map('q_map', maps['q'])          # q<=0.05 values
    save_map('q_map_q05', maps['q'])      # alias for clarity
    save_map('fdr_mask_q05', maps['fdr_mask_q05'].astype(np.float32))
    save_map('q_map_q10', maps['q_q10'])
    save_map('fdr_mask_q10', maps['fdr_mask_q10'].astype(np.float32))
    save_map('meanA', maps['meanA'])
    save_map('meanB', maps['meanB'])
    save_map('diff_meanA_minus_meanB', maps['diff'])
    save_map('effect_size_d', maps['d'])
    if 'beta_group' in maps:
        save_map('beta_group', maps['beta_group'])
    save_map('nA', maps['nA'].astype(np.float32))
    save_map('nB', maps['nB'].astype(np.float32))

    # Cluster outputs
    if cl.get('mask_uncorr') is not None:
        save_map(f"cluster_uncorr_p{args.voxel_p:.3f}", cl['mask_uncorr'].astype(np.float32))
    if cl.get('extent_thr') is not None:
        # Save corrected mask if present
        if cl.get('mask_corr') is not None:
            save_map(f"cluster_corr_p{args.voxel_p:.3f}_extent{cl['extent_thr']}", cl['mask_corr'].astype(np.float32))

    # TFCE (optional)
    if args.tfce or args.tfce_permutations > 0:
        try:
            t_map = maps['t']
            # Valid mask where p is finite (tested voxels)
            mask_valid = np.isfinite(maps['p'])
            if roi_mask is not None:
                mask_valid = mask_valid & roi_mask.astype(bool)
            glm_ctx = None
            if args.model == 'glm':
                glm_ctx = dict(
                    cov_raw_keep=cov_raw_keep,
                    covname2idx=ds.covname2idx,
                    covariate_names=covar_list,
                    standardize=bool(args.glm_standardize),
                )
            tfce_res = tfce_permutation_pvals(
                stack=stack,
                A=A,
                B=B,
                t_obs=t_map,
                mask_valid=mask_valid,
                model=args.model,
                glm_ctx=glm_ctx,
                n_perm=int(args.tfce_permutations),
                seed=int(args.seed),
                E=float(args.tfce_E),
                H=float(args.tfce_H),
                dh=float(args.tfce_dh),
                connectivity=26,
                two_sided=bool(args.tfce_two_sided),
                min_coverage=args.min_coverage,
                min_std=args.min_std,
            )
            save_map('tfce_map', tfce_res['tfce_map'])
            if tfce_res['tfce_p_fwe'] is not None:
                save_map('tfce_p_fwe', tfce_res['tfce_p_fwe'])
        except Exception as e:
            print(f"[warn] TFCE failed: {e}")

    # Region-wise CSV (optional)
    if args.region_atlas and os.path.isfile(args.region_atlas):
        roi_csv = os.path.join(args.out_dir, 'regionwise_stats.csv')
        try:
            regionwise_stats(args.region_atlas, ref, maps, roi_csv)
        except Exception as e:
            print(f"[warn] region-wise stats failed: {e}")

    # Summary JSON
    summary = dict(
        N_total=len(names), N_used=len(names_keep),
        N_A=int(A.sum()), N_B=int(B.sum()),
        group_A_label=detail.get('A_label', 'A'),
        group_B_label=detail.get('B_label', 'B'),
        group_by=args.group_by,
        image_suffix=args.image_suffix,
        voxel_p=args.voxel_p,
        permutations=args.permutations,
        extent_thr=cl.get('extent_thr', None),
        min_coverage=args.min_coverage,
        min_std=args.min_std,
        N_sig_q05=int(maps['fdr_mask_q05'].sum()),
        N_sig_q10=int(maps['fdr_mask_q10'].sum()),
        m_before=m_before,
        m_after_prefilter=m_after,
        prefilter_p=args.prefilter_p,
        prefilter_d=args.prefilter_d,
        tfce=bool(args.tfce or args.tfce_permutations>0),
        tfce_permutations=int(args.tfce_permutations),
        tfce_E=float(args.tfce_E),
        tfce_H=float(args.tfce_H),
        tfce_dh=float(args.tfce_dh),
        roi_mask_path=args.roi_mask,
        roi_voxels=int(roi_mask.sum()) if roi_mask is not None else None,
        roi_fraction=(float(roi_mask.sum())/roi_mask.size) if roi_mask is not None else None,
    )
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("=== Group comparison done ===")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
