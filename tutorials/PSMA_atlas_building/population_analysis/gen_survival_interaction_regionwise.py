#!/usr/bin/env python3
"""
Region-wise survival interaction modeling (baseline PET features x therapy).

Fit Cox model per ROI/metric:
    Survival ~ feature + therapy + feature*therapy + covariates

This is a first-step interaction analysis to assess whether baseline patterns
modify the relative benefit of a therapy. Outputs interaction coefficients and
p-values per ROI/metric (with optional bootstrap sign stability).

Running instructions
1) Basic univariate interaction per metric (default behavior)
     - Uses baseline PET radiomics per ROI, shows feature/therapy/interaction terms.
     - Example (relapse survival, z-metrics):
         python3.8 -u gen_survival_interaction_regionwise.py \
             --group-by pre_at --endpoint relapse \
             --radiomics-space z --radiomics-metrics mean_z,p90_z,std_z

2) Multivariate interaction per ROI (metrics jointly)
     - Fits one Cox model per ROI with all selected metrics + therapy + interactions.
     - Example (use the selected metrics jointly):
         python3.8 -u gen_survival_interaction_regionwise.py --group-by pre_at --endpoint relapse --radiomics-space z --radiomics-metrics mean_z,p90_z,std_z --multivariate --skip-univariate

3) Specify multivariate metric list explicitly
     - Example:
         python3.8 -u gen_survival_interaction_regionwise.py --group-by pre_at --endpoint relapse --radiomics-space both --radiomics-metrics mean_z,p90_z,mean_sulr,iqr_sulr --multivariate --mv-metrics mean_z,p90_z,mean_sulr,iqr_sulr

4) Use prewarped atlas-space volumes (skip warping)
     - Provide atlas-space SULR volumes as {SubjectID}_SULR.nii.gz in --data-dir.
     - Example:
         python3.8 -u gen_survival_interaction_regionwise.py \
             --prewarped --image-suffix SULR --group-by pre_at --endpoint relapse

5) Relapse endpoint
     - Example:
         python3.8 -u gen_survival_interaction_regionwise.py \
             --group-by pre_at --endpoint relapse --radiomics-space z

Outputs
    - Univariate CSV (if not skipped): output_regionwise_survival_interaction/regionwise_survival_interaction.csv
    - Multivariate CSV (if enabled): output_regionwise_survival_interaction/regionwise_survival_interaction_multivariate.csv
    - Summary JSON: output_regionwise_survival_interaction/summary.json
"""

import os
import json
import argparse
import math
from typing import Dict, Any, List, Tuple

import numpy as np
import nibabel as nib

from dataset import JHUPSMADataset
from cox_analysis import cox_fit, benjamini_hochberg, build_clinical_matrix
import MIR.label_reference as lbl_ref

from gen_group_compare_regionwise import (
    select_baseline_names_by_patient,
    build_stack_via_warp,
    load_prewarped_stack,
    load_or_compute_atlas_stats,
    region_metrics,
    DEFAULT_Z_METRICS,
    DEFAULT_SULR_METRICS,
    get_treatment_groups,
)


def _zscore_cols(X: np.ndarray) -> np.ndarray:
    X = X.astype(float, copy=True)
    for j in range(X.shape[1]):
        col = X[:, j]
        m = np.nanmean(col)
        s = np.nanstd(col)
        if not np.isfinite(s) or s < 1e-8:
            s = 1.0
        col = np.where(np.isfinite(col), col, m)
        X[:, j] = (col - m) / s
    return X


def _get_endpoint_arrays(ds: JHUPSMADataset, keep_mask: np.ndarray, endpoint: str) -> Tuple[np.ndarray, np.ndarray]:
    idx = ds.covname2idx
    cov = ds.covariates_raw
    if endpoint == 'overall':
        t = cov[:, idx.get('survival_time_days')]
        e = cov[:, idx.get('death_event')]
    else:
        t = cov[:, idx.get('relapsetime_days')]
        e = cov[:, idx.get('relapse_event')]
    t = t.astype(float)
    e = e.astype(float)
    if keep_mask is None:
        return t, e
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if keep_mask.shape[0] != t.shape[0]:
        raise RuntimeError(f"keep mask length {keep_mask.shape[0]} does not match covariates length {t.shape[0]}")
    return t[keep_mask], e[keep_mask]


def _bootstrap_interaction_sign(feature: np.ndarray, therapy: np.ndarray, covar: np.ndarray,
                                time: np.ndarray, event: np.ndarray, n_boot: int, seed: int = 123) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    n = len(feature)
    if n < 10 or n_boot <= 0:
        return np.nan, np.nan
    signs = []
    betas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        f = feature[idx]
        t = therapy[idx]
        c = covar[idx]
        tm = time[idx]
        ev = event[idx]
        inter = f * t
        X = np.column_stack([f, t, inter, c])
        beta, _, _, _ = cox_fit(X, tm, ev, l2=1e-4)
        if beta.size >= 3 and np.isfinite(beta[2]):
            betas.append(beta[2])
            signs.append(1 if beta[2] > 0 else (-1 if beta[2] < 0 else 0))
    if len(betas) == 0:
        return np.nan, np.nan
    betas = np.asarray(betas, dtype=float)
    signs = np.asarray(signs, dtype=float)
    # fraction of consistent sign with median
    med = np.nanmedian(betas)
    if med > 0:
        frac = float(np.mean(signs > 0))
    elif med < 0:
        frac = float(np.mean(signs < 0))
    else:
        frac = float(np.mean(signs == 0))
    return float(frac), float(med)


def _fisher_combined_p(pvals: np.ndarray) -> float:
    pvals = np.asarray(pvals, dtype=float)
    pvals = pvals[np.isfinite(pvals) & (pvals > 0)]
    if pvals.size == 0:
        return np.nan
    stat = -2.0 * np.sum(np.log(pvals))
    try:
        from scipy.stats import chi2 as _chi2
        return float(1.0 - _chi2.cdf(stat, 2 * pvals.size))
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser(description="Region-wise survival interaction modeling")
    ap.add_argument('--data-dir', default='/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/JHU', help='Data dir (CT/SUV files)')
    ap.add_argument('--json-dir', default='/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/clinical_jsons/', help='Clinical JSON dir')
    ap.add_argument('--out-dir', default='output_regionwise_survival_interaction', help='Output directory')
    ap.add_argument('--group-by', default='pre_at',
                    choices=['pre_local','pre_focal','pre_at','pre_cyto','post_local','post_focal','post_at','post_cyto',
                             'pre_any','post_any','any_therapy'],
                    help='Therapy grouping rule (treated vs untreated)')
    ap.add_argument('--endpoint', default='overall', choices=['overall', 'relapse'],
                    help='Survival endpoint: overall (death) or relapse')
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
    ap.add_argument('--min-n', type=int, default=10, help='Minimum subjects for fitting')
    ap.add_argument('--multivariate', action='store_true', help='Run multivariate Cox per ROI (metrics jointly)')
    ap.add_argument('--mv-metrics', default='auto',
                    help='Comma-separated metric names for multivariate model (auto = use selected metrics)')
    ap.add_argument('--skip-univariate', action='store_true', help='Skip univariate (per-metric) Cox models')
    ap.add_argument('--multivariate-bootstrap', type=int, default=0, help='Bootstrap repetitions for interaction sign stability (univariate only)')
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
    if len(names_keep) < args.min_n:
        raise RuntimeError(f"Too few subjects after grouping (n={len(names_keep)}).")
    A = A[keep]; B = B[keep]
    therapy = A.astype(float)
    print(f"[info] Therapy grouping: {detail.get('A_label','A')} n={int(A.sum())} vs {detail.get('B_label','B')} n={int(B.sum())}")

    # Load/warp atlas-space SULR stack
    if args.prewarped:
        stack, _ref = load_prewarped_stack(args.data_dir, names_keep, args.image_suffix)
        mask_3d = np.isfinite(stack[0])
    else:
        stack, _ref, aux = build_stack_via_warp(ds, names_keep, device=args.device, print_every=args.print_every)
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

    # Determine metrics
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

    # Per-subject ROI metrics
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

    # Clinical covariates
    cov_rows = []
    for i in range(len(names_keep)):
        row = build_clinical_matrix(ds.covariates_raw[keep][i], ds.covariate_names)
        cov_rows.append(row)
    cov_mat = np.stack(cov_rows, axis=0)
    cov_mat = _zscore_cols(cov_mat)

    # Endpoint arrays (aligned to filtered subjects)
    time_all, event_all = _get_endpoint_arrays(ds, keep, endpoint=args.endpoint)

    # Main modeling (univariate per metric)
    rows = []
    if not args.skip_univariate:
        for metric_name, mat in metrics.items():
            p_int_list = []
            for k, roi in enumerate(roi_ids):
                f = mat[:, k].astype(float)
                # standardize feature
                f = (f - np.nanmean(f)) / (np.nanstd(f) + 1e-8)
                # build valid mask
                valid = np.isfinite(f) & np.isfinite(time_all) & np.isfinite(event_all)
                valid &= np.isfinite(therapy) & np.isfinite(cov_mat).all(axis=1)
                if valid.sum() < args.min_n or int(event_all[valid].sum()) < 5:
                    rows.append({
                        'metric': metric_name,
                        'roi_id': int(roi),
                        'roi_name': roi_names_map.get(int(roi), f'label_{int(roi)}'),
                        'n': int(valid.sum()),
                        'events': int(event_all[valid].sum()),
                        'beta_feature': np.nan,
                        'beta_therapy': np.nan,
                        'beta_interaction': np.nan,
                        'p_feature': np.nan,
                        'p_therapy': np.nan,
                        'p_interaction': np.nan,
                        'hr_feature': np.nan,
                        'hr_therapy': np.nan,
                        'hr_interaction': np.nan,
                        'interaction_sign_frac': np.nan,
                        'interaction_boot_median': np.nan,
                    })
                    p_int_list.append(np.nan)
                    continue

                f_v = f[valid]
                t_v = therapy[valid]
                c_v = cov_mat[valid]
                tm_v = time_all[valid]
                ev_v = event_all[valid].astype(int)
                inter = f_v * t_v
                X = np.column_stack([f_v, t_v, inter, c_v])

                beta, se, z, p = cox_fit(X, tm_v, ev_v, l2=1e-4)
                if beta.size < 3:
                    b_feat = b_th = b_int = np.nan
                    p_feat = p_th = p_int = np.nan
                else:
                    b_feat, b_th, b_int = float(beta[0]), float(beta[1]), float(beta[2])
                    p_feat, p_th, p_int = float(p[0]), float(p[1]), float(p[2])

                sign_frac = np.nan
                boot_med = np.nan
                if args.multivariate_bootstrap and args.multivariate_bootstrap > 0:
                    sign_frac, boot_med = _bootstrap_interaction_sign(
                        f_v, t_v, c_v, tm_v, ev_v, n_boot=args.multivariate_bootstrap
                    )

                rows.append({
                    'metric': metric_name,
                    'roi_id': int(roi),
                    'roi_name': roi_names_map.get(int(roi), f'label_{int(roi)}'),
                    'n': int(valid.sum()),
                    'events': int(ev_v.sum()),
                    'beta_feature': b_feat,
                    'beta_therapy': b_th,
                    'beta_interaction': b_int,
                    'p_feature': p_feat,
                    'p_therapy': p_th,
                    'p_interaction': p_int,
                    'hr_feature': float(np.exp(b_feat)) if np.isfinite(b_feat) else np.nan,
                    'hr_therapy': float(np.exp(b_th)) if np.isfinite(b_th) else np.nan,
                    'hr_interaction': float(np.exp(b_int)) if np.isfinite(b_int) else np.nan,
                    'interaction_sign_frac': sign_frac,
                    'interaction_boot_median': boot_med,
                })
                p_int_list.append(p_int)

            # FDR on interaction p-values per metric
            p_int_arr = np.asarray(p_int_list, dtype=float)
            sig05, q05 = benjamini_hochberg(p_int_arr, q=0.05) if np.isfinite(p_int_arr).any() else (np.zeros_like(p_int_arr, dtype=bool), np.ones_like(p_int_arr))
            sig10, q10 = benjamini_hochberg(p_int_arr, q=0.10) if np.isfinite(p_int_arr).any() else (np.zeros_like(p_int_arr, dtype=bool), np.ones_like(p_int_arr))
            # write q-values into rows for this metric
            offset = len(rows) - len(roi_ids)
            for i in range(len(roi_ids)):
                rows[offset + i]['q05_interaction'] = float(q05[i]) if np.isfinite(p_int_arr[i]) else np.nan
                rows[offset + i]['q10_interaction'] = float(q10[i]) if np.isfinite(p_int_arr[i]) else np.nan
                rows[offset + i]['sig_q05_interaction'] = bool(sig05[i]) if np.isfinite(p_int_arr[i]) else False
                rows[offset + i]['sig_q10_interaction'] = bool(sig10[i]) if np.isfinite(p_int_arr[i]) else False

    # Multivariate modeling (joint metrics per ROI)
    mv_rows = []
    mv_list = None
    if args.multivariate:
        if args.mv_metrics.strip().lower() == 'auto':
            mv_list = list(selected_metrics)
        else:
            mv_list = [s.strip() for s in args.mv_metrics.split(',') if s.strip()]
        missing = [m for m in mv_list if m not in metrics]
        if missing:
            raise RuntimeError(f"Unknown mv metrics: {missing}. Available: {list(metrics.keys())}")

        for k, roi in enumerate(roi_ids):
            Xfeat = np.stack([metrics[m][:, k] for m in mv_list], axis=1).astype(float)
            # z-score each feature column
            for j in range(Xfeat.shape[1]):
                col = Xfeat[:, j]
                Xfeat[:, j] = (col - np.nanmean(col)) / (np.nanstd(col) + 1e-8)
            # build valid mask
            valid = np.isfinite(Xfeat).all(axis=1) & np.isfinite(time_all) & np.isfinite(event_all)
            valid &= np.isfinite(therapy) & np.isfinite(cov_mat).all(axis=1)
            if valid.sum() < args.min_n or int(event_all[valid].sum()) < 5:
                mv_rows.append({
                    'roi_id': int(roi),
                    'roi_name': roi_names_map.get(int(roi), f'label_{int(roi)}'),
                    'n': int(valid.sum()),
                    'events': int(event_all[valid].sum()),
                    'metrics': ','.join(mv_list),
                    'p_interaction_fisher': np.nan,
                    'fit_l2': np.nan,
                    'fit_cond': np.nan,
                    'fit_converged': False,
                })
                continue

            Xf = Xfeat[valid]
            t_v = therapy[valid]
            c_v = cov_mat[valid]
            tm_v = time_all[valid]
            ev_v = event_all[valid].astype(int)
            inter = Xf * t_v[:, None]
            X = np.column_stack([Xf, t_v, inter, c_v])

            beta, se, z, p, diag = cox_fit(X, tm_v, ev_v, l2=1e-4, return_diag=True)
            # interaction block p-values follow after: features (M), therapy (1), interactions (M)
            m = Xf.shape[1]
            p_int = np.full(m, np.nan, dtype=float)
            if p.size >= (2 * m + 1):
                p_int = np.asarray(p[m + 1:m + 1 + m], dtype=float)
            # Retry with stronger ridge if interaction p-values are non-finite
            if not np.isfinite(p_int).all():
                for l2_try in [1e-3, 1e-2, 1e-1, 1.0]:
                    beta, se, z, p, diag = cox_fit(X, tm_v, ev_v, l2=l2_try, return_diag=True)
                    if p.size >= (2 * m + 1):
                        p_int = np.asarray(p[m + 1:m + 1 + m], dtype=float)
                    if np.isfinite(p_int).all():
                        break

            row = {
                'roi_id': int(roi),
                'roi_name': roi_names_map.get(int(roi), f'label_{int(roi)}'),
                'n': int(valid.sum()),
                'events': int(ev_v.sum()),
                'metrics': ','.join(mv_list),
                'p_interaction_fisher': _fisher_combined_p(p_int),
                'fit_l2': float(diag.get('l2_used')) if isinstance(diag, dict) else np.nan,
                'fit_cond': float(diag.get('cond')) if isinstance(diag, dict) else np.nan,
                'fit_converged': bool(diag.get('converged')) if isinstance(diag, dict) else False,
            }
            for j, mname in enumerate(mv_list):
                row[f'p_interaction_{mname}'] = float(p_int[j]) if np.isfinite(p_int[j]) else np.nan
            mv_rows.append(row)

    # Save CSV
    import csv
    out_csv = None
    if rows:
        out_csv = os.path.join(args.out_dir, 'regionwise_survival_interaction.csv')
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    mv_csv = None
    if mv_rows:
        mv_csv = os.path.join(args.out_dir, 'regionwise_survival_interaction_multivariate.csv')
        with open(mv_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(mv_rows[0].keys()))
            w.writeheader(); w.writerows(mv_rows)

    # Summary JSON
    summary = dict(
        N_total=len(names),
        N_used=int(therapy.size),
        N_treated=int(A.sum()),
        N_untreated=int(B.sum()),
        group_by=args.group_by,
        group_A_label=detail.get('A_label', 'A'),
        group_B_label=detail.get('B_label', 'B'),
        endpoint=args.endpoint,
        z_thr=float(args.z_thr),
        radiomics_space=args.radiomics_space,
        radiomics_metrics=selected_metrics,
        use_robust_atlas=bool(args.use_robust_atlas),
        atlas_stats_dir=args.atlas_stats_dir,
        label_atlas=args.label_atlas,
        prewarped=bool(args.prewarped),
        image_suffix=args.image_suffix,
        multivariate=bool(args.multivariate),
        mv_metrics=args.mv_metrics,
        skip_univariate=bool(args.skip_univariate),
        bootstrap_n=int(args.multivariate_bootstrap),
        output_csv=out_csv,
        output_mv_csv=mv_csv,
    )
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("=== Region-wise survival interaction modeling done ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved CSV: {out_csv}")


if __name__ == '__main__':
    main()
