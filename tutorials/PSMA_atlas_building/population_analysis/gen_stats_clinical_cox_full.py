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
import os, json, math, re
import numpy as np
import nibabel as nib
import torch
from dataset import JHUPSMADataset
from datetime import datetime
import utils
from utils import gaussian_kernel3d, smooth_inside_mask, morph_close3d, save_vector_as_vol
from cox_analysis import benjamini_hochberg, build_clinical_matrix, cox_fit, concordance_index, orthogonalize_against, plot_km_three_groups, plot_km_two_groups

def main():
    batch_size = 1
    _scale = 0.777047
    train_dir = '/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/JHU/'
    val_dir = '/scratch2/jchen/DATA/PSMA_JHU/Preprocessed/JHU/'
    save_dir = 'VFAAtlas_SSIM_1_MS_1_diffusion_1/'
    model_dir = '../experiments/' + save_dir
    population_stats_dir = 'population_stats/'
    os.makedirs(population_stats_dir, exist_ok=True)
    # Option: use per-voxel Z-score maps (across the cohort) instead of raw SULR values
    # when constructing the Voxel Prognostic Score (VPS). Enabled by default.
    USE_Z_MAP_FOR_VPS = True
    # Option: choose endpoint for survival analysis. If True, use Overall Survival (time to death, death_event).
    # If False, use Relapse (time to relapse, relapse_event). Enabled by default.
    USE_OVERALL_SURVIVAL = True
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
    # Keeps tensors stacked as usual; for string/None fields, returns lists (replacing None with "")
    # For mapping field 'covname2idx' (same for all items), just keep the first copy.
    from torch.utils.data._utils.collate import default_collate as _default_collate
    def _safe_collate(batch):
        if not isinstance(batch, list) or len(batch) == 0:
            return batch
        elem0 = batch[0]
        # If dataset returns non-dict, fall back to default
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
                    # As a last resort, keep raw list to avoid crashing
                    out[k] = vals
        return out

    # Use batch_size=1 because we only need baseline (or the single scan) per patient
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False, collate_fn=_safe_collate)

    # Running stats to build cohort mean/std of SULR in atlas space
    lesion_stats = utils.RunningStats3D(shape=(H, W, D), device='cuda:0', dtype=torch.float32)
    # Keep per-patient volumes for later special-case saving
    subj_ids = []
    sulr_volumes = []   # smoothed, masked SULR in atlas space [H,W,D]
    pet_volumes = []    # warped PET in atlas space [H,W,D]
    '''
    Validation
    '''
    # Collect patients (each batch has a single scan: baseline or the only scan)
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

            # Update cohort running stats and stash volumes for later
            try:
                lesion_stats.update(sulr_bl_s)  # expects full 3D tensor
            except Exception:
                # Fallback: ensure tensor device/dtype
                lesion_stats.update(sulr_bl_s.to(dtype=torch.float32, device='cuda:0'))
            subj_ids.append(str(data['SubjectID'][0]))
            sulr_volumes.append(sulr_bl_s.detach().cpu().numpy())
            pet_volumes.append(def_pat_suv[0, 0].detach().cpu().numpy())

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
            
            if USE_OVERALL_SURVIVAL:
                if np.isfinite(t_sur) and e_dea in (0.0, 1.0):
                    time_to_event.append(t_sur)
                    event_indicator.append(int(e_dea))
                else:
                    # no usable overall survival outcome; drop this patient
                    baseline_X_list.pop()
                    print('Skip: missing overall survival outcome')
                    continue
            else:
                if np.isfinite(t_rel) and e_rel in (0.0, 1.0):
                    time_to_event.append(t_rel)
                    event_indicator.append(int(e_rel))
                else:
                    # no usable relapse outcome; drop this patient
                    baseline_X_list.pop()
                    print('Skip: missing relapse outcome')
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
    events_cover = np.zeros(V, dtype=np.int32)

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
            events_cover[start + j] = e_cov
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
    save_vector_as_vol(HR,   population_stats_dir+'hr_map.nii.gz',      x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(beta_map.astype(np.float32), population_stats_dir+'beta_map.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(risk_ratio.astype(np.float32), population_stats_dir+'risk_ratio_map.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(hr_gt1.astype(np.float32), population_stats_dir+'hr_gt1_map.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(inv_hr_protective.astype(np.float32), population_stats_dir+'inv_hr_protective_map.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(zs,   population_stats_dir+'z_map.nii.gz',       x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(ps,   population_stats_dir+'p_map.nii.gz',       x_pt_nib_aff, (H, W, D), keep_mask)
    # FDR masks for q=0.05 and q=0.10
    save_vector_as_vol(sig_mask05.astype(np.float32), population_stats_dir+'sig_mask_fdr_q05.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(sig_mask10.astype(np.float32), population_stats_dir+'sig_mask_fdr_q10.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)

    # Publication-friendly HR variants masked by FDR
    hr_gt1_q05 = np.where(sig_mask05, hr_gt1, np.nan)
    inv_hr_protective_q05 = np.where(sig_mask05, inv_hr_protective, np.nan)
    hr_gt1_q10 = np.where(sig_mask10, hr_gt1, np.nan)
    inv_hr_protective_q10 = np.where(sig_mask10, inv_hr_protective, np.nan)
    save_vector_as_vol(hr_gt1_q05.astype(np.float32), population_stats_dir+'hr_gt1_fdr_q05.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(inv_hr_protective_q05.astype(np.float32), population_stats_dir+'inv_hr_protective_fdr_q05.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(hr_gt1_q10.astype(np.float32), population_stats_dir+'hr_gt1_fdr_q10.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(inv_hr_protective_q10.astype(np.float32), population_stats_dir+'inv_hr_protective_fdr_q10.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)

    # Optional: store q-values for visualization (both thresholds)
    save_vector_as_vol(qvals05, population_stats_dir+'q_map_q05.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(qvals10, population_stats_dir+'q_map_q10.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    # Also provide viewer-friendly versions with NaNs filled as 1.0 (non-significant)
    save_vector_as_vol(np.nan_to_num(qvals05, nan=1.0).astype(np.float32), population_stats_dir+'q_map_vis_q05.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(np.nan_to_num(qvals10, nan=1.0).astype(np.float32), population_stats_dir+'q_map_vis_q10.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    # Save diagnostics: coverage and std across patients (helps explain zeros)
    save_vector_as_vol(cover.astype(np.float32), population_stats_dir+'coverage_map.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    save_vector_as_vol(np.nan_to_num(stdev, nan=0.0).astype(np.float32), population_stats_dir+'std_map.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    # Valid-fit mask (1 where z/beta/p are finite)
    valid_fit = np.isfinite(zs) & np.isfinite(betas) & np.isfinite(ps)
    save_vector_as_vol(valid_fit.astype(np.float32), population_stats_dir+'valid_fit_map.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)
    # Events-per-voxel (among contributing patients)
    save_vector_as_vol(events_cover.astype(np.float32), population_stats_dir+'events_map.nii.gz', x_pt_nib_aff, (H, W, D), keep_mask)

    # ---------- Build patient-level Voxel Prognostic Score (VPS) ----------
    # Use q=0.05 mask as the default for VPS construction
    sig_idx = np.where(sig_mask05)[0]
    # Choose the matrix to aggregate for VPS: either raw SULR (Ximg) or cohort Z-score per voxel
    X_for_vps = Ximg
    if USE_Z_MAP_FOR_VPS:
        m_all = np.nanmean(Ximg, axis=0)
        s_all = np.nanstd(Ximg, axis=0)
        s_all[s_all < 1e-9] = 1.0
        X_for_vps = (np.where(np.isfinite(Ximg), Ximg, m_all) - m_all) / s_all
        X_for_vps = np.clip(X_for_vps, -8.0, 8.0)
    if sig_idx.size == 0:
        print('No FDR-significant voxels at q=0.05; consider reporting top-k or relax q to 0.10.')
        VPS = X_for_vps.mean(axis=1)  # fallback global mean
    else:
        # Weighted by Cox beta (positive weights increase risk)
        w = betas[sig_idx]
        # Aggregate either Z-scored or raw matrix (as selected above)
        X_sel = X_for_vps[:, sig_idx]
        VPS = (X_sel * w[None, :]).sum(axis=1) / (np.abs(w).sum() + 1e-8)

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
    VPS_z, _, _   = zscore(VPS[:, None])
    X_clin_only   = Xclin_z
    X_with_vps    = np.hstack([Xclin_z, VPS_z])

    b_c, se_c, z_c, p_c = cox_fit(X_clin_only, time_np, event_np, l2=1e-4, max_iter=60)
    b_cv, se_cv, z_cv, p_cv = cox_fit(X_with_vps, time_np, event_np, l2=1e-4, max_iter=60)
    # VPS-only model (record C-index for imaging score alone)
    b_v, se_v, z_v, p_v = cox_fit(VPS_z, time_np, event_np, l2=1e-4, max_iter=60)

    # C-index
    c_clin = concordance_index(time_np, event_np, risk=(X_clin_only @ b_c))
    c_all  = concordance_index(time_np, event_np, risk=(X_with_vps @ b_cv))
    c_vps  = concordance_index(time_np, event_np, risk=(VPS_z @ b_v))

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

    print('\n=== Cox Results ===')
    print(f'VPS only:           C-index={c_vps:.3f}  (1 imaging covariate)')
    print(f'Clinical only:      C-index={c_clin:.3f}  (P={X_clin_only.shape[1]} covariates)')
    print(f'Clinical + VPS:     C-index={c_all:.3f}  (adds 1 imaging covariate)')
    print(f'Clinical + VPS⊥:   C-index={c_all_orth:.3f}  (VPS orthogonalized to clinical)')
    if VPS_oof is not None:
        # If OOF VPS available, also report VPS-only OOF C-index
        b_voof, _, _, _ = cox_fit(VPS_oof_z, time_np, event_np, l2=1e-4, max_iter=60)
        c_vps_oof = concordance_index(time_np, event_np, risk=(VPS_oof_z @ b_voof))
        print(f'VPS only (OOF):     C-index={c_vps_oof:.3f}')
        print(f'Clinical + VPS(OOF):     C-index={c_all_oof:.3f}')
        print(f'Clinical + VPS(OOF)⊥:   C-index={c_all_oof_orth:.3f}')

    # ---------- Kaplan–Meier curves (median/tertiles) with censor ticks ----------
    def _km_curve_arrays(t, e):
        import numpy as _np
        order = _np.argsort(t)
        t = _np.asarray(t, dtype=float)[order]
        e = _np.asarray(e, dtype=int)[order]
        uniq_times = _np.unique(t)
        n_at_risk = float(len(t))
        surv = 1.0
        xs = [0.0]; ys = [1.0]
        censor_times = []
        censor_levels = []
        var_acc = 0.0
        ci_x = [0.0]; ci_lo = [1.0]; ci_hi = [1.0]
        for ut in uniq_times:
            m = (t == ut)
            d = float((e[m] == 1).sum())
            c = float((e[m] == 0).sum())
            if n_at_risk > 0 and d > 0:
                surv *= (1.0 - d / n_at_risk)
                xs.append(ut); ys.append(ys[-1])
                xs.append(ut); ys.append(surv)
                # Greenwood variance accumulates at event times
                var_acc += (d / (n_at_risk * max(n_at_risk - d, 1.0)))
                se = (surv ** 2) * var_acc
                delta = 1.96 * _np.sqrt(max(se, 0.0))
                lo = float(max(0.0, surv - delta))
                hi = float(min(1.0, surv + delta))
                ci_x.append(ut); ci_lo.append(ci_lo[-1]); ci_hi.append(ci_hi[-1])
                ci_x.append(ut); ci_lo.append(lo);            ci_hi.append(hi)
            if c > 0:
                censor_times.extend([ut] * int(c))
                censor_levels.extend([surv] * int(c))
            n_at_risk -= (d + c)
        return (np.asarray(xs), np.asarray(ys),
                np.asarray(censor_times), np.asarray(censor_levels),
                np.asarray(ci_x), np.asarray(ci_lo), np.asarray(ci_hi))

    def plot_km_with_ticks_two_groups(time_days, event, grp_high, out_path, title='KM: two groups'):
        import numpy as _np
        import matplotlib.pyplot as _plt
        grp_high = _np.asarray(grp_high, dtype=bool)
        t = _np.asarray(time_days, dtype=float)
        e = _np.asarray(event, dtype=int)
        mask_fin = _np.isfinite(t) & _np.isfinite(e)
        t = t[mask_fin]; e = e[mask_fin]; grp_high = grp_high[mask_fin]
        xH,yH,cHt,cHy,ciHx,ciHlo,ciHhi = _km_curve_arrays(t[grp_high], e[grp_high])
        xL,yL,cLt,cLy,ciLx,ciLlo,ciLhi = _km_curve_arrays(t[~grp_high], e[~grp_high])
        _plt.figure(figsize=(6.5,4.5), dpi=140)
        _plt.step(xH, yH, where='post', color='tab:red', label='High')
        _plt.step(xL, yL, where='post', color='tab:blue', label='Low')
        # 95% CI bands
        if ciHx.size:
            _plt.fill_between(ciHx, ciHlo, ciHhi, color='tab:red', alpha=0.15, step='post')
        if ciLx.size:
            _plt.fill_between(ciLx, ciLlo, ciLhi, color='tab:blue', alpha=0.15, step='post')
        if cHt.size:
            _plt.scatter(cHt, cHy, marker='|', color='tab:red', s=80, linewidths=1.5)
        if cLt.size:
            _plt.scatter(cLt, cLy, marker='|', color='tab:blue', s=80, linewidths=1.5)
        _plt.ylim(0.0, 1.05)
        _plt.xlabel('Time (days)')
        _plt.ylabel('Survival probability')
        _plt.title(title)
        _plt.grid(alpha=0.25)
        _plt.legend(loc='best')
        try:
            _plt.savefig(out_path, bbox_inches='tight')
        finally:
            _plt.close()

    def plot_km_with_ticks_three_groups(time_days, event, score, out_path, title='KM: three groups (tertiles)'):
        import numpy as _np
        import matplotlib.pyplot as _plt
        t = _np.asarray(time_days, dtype=float)
        e = _np.asarray(event, dtype=int)
        s = _np.asarray(score, dtype=float)
        mask_fin = _np.isfinite(t) & _np.isfinite(e) & _np.isfinite(s)
        t = t[mask_fin]; e = e[mask_fin]; s = s[mask_fin]
        q1 = _np.nanpercentile(s, 33.33); q2 = _np.nanpercentile(s, 66.67)
        grp_lo = s < q1; grp_md = (s >= q1) & (s < q2); grp_hi = s >= q2
        xLo,yLo,cLo_t,cLo_y,ciLox,ciLolo,ciLohi = _km_curve_arrays(t[grp_lo], e[grp_lo])
        xMd,yMd,cMd_t,cMd_y,ciMdx,ciMdlo,ciMdhi = _km_curve_arrays(t[grp_md], e[grp_md])
        xHi,yHi,cHi_t,cHi_y,ciHix,ciHilo,ciHihi = _km_curve_arrays(t[grp_hi], e[grp_hi])
        _plt.figure(figsize=(6.8,4.8), dpi=140)
        _plt.step(xLo, yLo, where='post', color='tab:green', label='Low')
        _plt.step(xMd, yMd, where='post', color='tab:orange', label='Mid')
        _plt.step(xHi, yHi, where='post', color='tab:red', label='High')
        # 95% CI bands
        if ciLox.size:
            _plt.fill_between(ciLox, ciLolo, ciLohi, color='tab:green', alpha=0.15, step='post')
        if ciMdx.size:
            _plt.fill_between(ciMdx, ciMdlo, ciMdhi, color='tab:orange', alpha=0.15, step='post')
        if ciHix.size:
            _plt.fill_between(ciHix, ciHilo, ciHihi, color='tab:red', alpha=0.15, step='post')
        if cLo_t.size:
            _plt.scatter(cLo_t, cLo_y, marker='|', color='tab:green', s=80, linewidths=1.5)
        if cMd_t.size:
            _plt.scatter(cMd_t, cMd_y, marker='|', color='tab:orange', s=80, linewidths=1.5)
        if cHi_t.size:
            _plt.scatter(cHi_t, cHi_y, marker='|', color='tab:red', s=80, linewidths=1.5)
        _plt.ylim(0.0, 1.05)
        _plt.xlabel('Time (days)')
        _plt.ylabel('Survival probability')
        _plt.title(title)
        _plt.grid(alpha=0.25)
        _plt.legend(loc='best')
        try:
            _plt.savefig(out_path, bbox_inches='tight')
        finally:
            _plt.close()
    # Imaging-only VPS risk split (with ticks)
    try:
        vps_scores = VPS_z[:, 0]
        med_vps = np.nanmedian(vps_scores)
        grp_vps_hi = vps_scores >= med_vps
        plot_km_with_ticks_two_groups(
            time_np, event_np, grp_vps_hi,
            out_path='km_vps_median.png',
            title='KM: imaging VPS (median split)')
        print('Saved: km_vps_median.png')
    except Exception as ex:
        print(f'KM VPS plot skipped due to error: {ex}')

    # Clinical-only linear predictor split (with ticks)
    try:
        risk_clin = X_clin_only @ b_c
        med_c = np.nanmedian(risk_clin)
        grp_c_hi = risk_clin >= med_c
        plot_km_with_ticks_two_groups(
            time_np, event_np, grp_c_hi,
            out_path='km_clinical_median.png',
            title='KM: clinical risk (median split)')
        print('Saved: km_clinical_median.png')
    except Exception as ex:
        print(f'KM clinical plot skipped due to error: {ex}')

    # Clinical + VPS linear predictor split (with ticks)
    try:
        risk_all = X_with_vps @ b_cv
        med_all = np.nanmedian(risk_all)
        grp_all_hi = risk_all >= med_all
        plot_km_with_ticks_two_groups(
            time_np, event_np, grp_all_hi,
            out_path='km_clinical_plus_vps_median.png',
            title='KM: clinical + VPS risk (median split)')
        print('Saved: km_clinical_plus_vps_median.png')
    except Exception as ex:
        print(f'KM clinical+VPS plot skipped due to error: {ex}')

    # ---------- Optional: KM with three risk groups (tertiles) ----------
    try:
        # VPS tertiles (with ticks)
        vps_scores = VPS_z[:, 0]
        plot_km_with_ticks_three_groups(
            time_np, event_np, vps_scores,
            out_path='km_vps_tertiles.png',
            title='KM: imaging VPS (tertiles)')
        print('Saved: km_vps_tertiles.png')
    except Exception as ex:
        print(f'KM VPS tertiles plot skipped due to error: {ex}')

    try:
        # Clinical tertiles (with ticks)
        risk_clin = (X_clin_only @ b_c).ravel()
        plot_km_with_ticks_three_groups(
            time_np, event_np, risk_clin,
            out_path='km_clinical_tertiles.png',
            title='KM: clinical risk (tertiles)')
        print('Saved: km_clinical_tertiles.png')
    except Exception as ex:
        print(f'KM clinical tertiles plot skipped due to error: {ex}')

    try:
        # Clinical + VPS tertiles (with ticks)
        risk_all = (X_with_vps @ b_cv).ravel()
        plot_km_with_ticks_three_groups(
            time_np, event_np, risk_all,
            out_path='km_clinical_plus_vps_tertiles.png',
            title='KM: clinical + VPS risk (tertiles)')
        print('Saved: km_clinical_plus_vps_tertiles.png')
    except Exception as ex:
        print(f'KM clinical+VPS tertiles plot skipped due to error: {ex}')

    # Save a small report for reproducibility
    with open('cox_summary.txt', 'w') as f:
        f.write(f'Patients N={Ximg.shape[0]}, Vmask={Ximg.shape[1]}\n')
        f.write(f'Events={int(event_np.sum())}, Median time={np.nanmedian(time_np):.1f} days\n')
        f.write('Endpoint=' + ('Overall Survival' if USE_OVERALL_SURVIVAL else 'Relapse') + '\n')
        f.write(f'VPS-only C-index={c_vps:.3f}\n')
        f.write(f'Clinical C-index={c_clin:.3f}\n')
        f.write(f'Clinical+VPS C-index={c_all:.3f}\n')
        f.write(f'Clinical+VPS(orthogonalized) C-index={c_all_orth:.3f}\n')
        if VPS_oof is not None:
            f.write(f'VPS-only(OOF) C-index={c_vps_oof:.3f}\n')
            f.write(f'Clinical+VPS(OOF) C-index={c_all_oof:.3f}\n')
            f.write(f'Clinical+VPS(OOF, orthogonalized) C-index={c_all_oof_orth:.3f}\n')
        f.write(f'Significant voxels (FDR q<=0.05)={sig_idx.size}\n')
        inv_frac = 1.0 - float(np.mean(valid_fit))
        f.write(f'Invalid voxels (NaN in z/beta/p)={int((~valid_fit).sum())}  ({inv_frac*100:.1f}%)\n')
        f.write(f'Median coverage (valid voxels)={np.nanmedian(cover[valid_fit]):.1f}\n')
        f.write(f'Median events (valid voxels)={np.nanmedian(events_cover[valid_fit]):.1f}\n')
        f.write('Clinical covariates included: age, log_psa_at_scan (fallback pre/initial), '
                'grade_group, t_ord, BMI, indication (primary/recurrence/metastatic), '
                'pre_androgen_targeted, pre_cytotoxic.\n')
        f.write('Excluded post-PSMA covariates (post_local, post_focal, post_at, post_cyto).\n')

    print('Saved: hr_map.nii.gz, z_map.nii.gz, p_map.nii.gz, '
        'q_map_q05.nii.gz, q_map_q10.nii.gz, q_map_vis_q05.nii.gz, q_map_vis_q10.nii.gz, '
        'sig_mask_fdr_q05.nii.gz, sig_mask_fdr_q10.nii.gz, '
        'hr_gt1_fdr_q05.nii.gz, inv_hr_protective_fdr_q05.nii.gz, '
        'hr_gt1_fdr_q10.nii.gz, inv_hr_protective_fdr_q10.nii.gz, cox_summary.txt')

    # ---------- Save PET and z-score images for special cases (high/low risk) ----------
    try:
        # Prefer clinical+VPS risk; if unavailable, fall back to imaging-only VPS
        if 'risk_all' in locals():
            risks = (X_with_vps @ b_cv).ravel()
        else:
            risks = VPS_z[:, 0]
        if len(risks) != len(sulr_volumes):
            print('Special-case saving skipped: risk length mismatch')
        else:
            idx_hi = int(np.nanargmax(risks))
            idx_lo = int(np.nanargmin(risks))
            # Cohort mean/std from running stats
            mean_vol = lesion_stats.mean().detach().cpu().numpy()
            std_vol = lesion_stats.std().detach().cpu().numpy()
            std_vol = np.where(std_vol < 1e-6, 1.0, std_vol)
            # Compose z-score maps
            z_hi = (sulr_volumes[idx_hi] - mean_vol) / std_vol
            z_lo = (sulr_volumes[idx_lo] - mean_vol) / std_vol
            # Outside mask, set to 0 for viewer friendliness
            msk = mask_3d.detach().cpu().numpy()
            z_hi[~msk] = 0.0; z_lo[~msk] = 0.0
            pet_hi = pet_volumes[idx_hi]; pet_lo = pet_volumes[idx_lo]
            sid_hi = subj_ids[idx_hi]; sid_lo = subj_ids[idx_lo]
            out_dir = os.path.join(population_stats_dir, 'special_cases')
            os.makedirs(out_dir, exist_ok=True)
            nib.save(nib.Nifti1Image(pet_hi.astype(np.float32), x_pt_nib_aff), os.path.join(out_dir, f'{sid_hi}_PET_high_risk.nii.gz'))
            nib.save(nib.Nifti1Image(z_hi.astype(np.float32),   x_pt_nib_aff), os.path.join(out_dir, f'{sid_hi}_Z_high_risk.nii.gz'))
            nib.save(nib.Nifti1Image(pet_lo.astype(np.float32), x_pt_nib_aff), os.path.join(out_dir, f'{sid_lo}_PET_low_risk.nii.gz'))
            nib.save(nib.Nifti1Image(z_lo.astype(np.float32),   x_pt_nib_aff), os.path.join(out_dir, f'{sid_lo}_Z_low_risk.nii.gz'))
            print(f'Saved special cases to: {out_dir}\n  High-risk: {sid_hi}\n  Low-risk: {sid_lo}')
    except Exception as ex:
        print(f'Special-case PET/Z saving failed: {ex}')

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