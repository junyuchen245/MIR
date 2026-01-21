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
from utils import gaussian_kernel3d, smooth_inside_mask, morph_close3d, save_vector_as_vol
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
    USE_VOXELWISE = False          # set False to skip voxelwise Cox maps to speed up
    USE_ATLAS_FEATURES = False     # keep atlas-based features analysis
    EVAL_VPS_INCREMENTAL = False  # set True to test a precomputed VPS vs clinical+atlas
    VPS_PATH = None               # '/path/to/vps.npy' or '/path/to/vps.csv' with columns [SubjectID,VPS]
    USE_LESION_FEATURES = False    # add global lesion-burden features (TL-PSMA-like)
    USE_ATLAS_Z_DETECTION = True  # build/load statistical SULR atlas and use z>thr to define lesions
    LESION_Z_THR = 3.0            # z-score threshold relative to atlas mean/std
    LESION_MIN_CC = 20            # minimum connected-component size (voxels)
    USE_ZSCORE_ROI_FEATURES = True # derive ROI percentiles from z-score map instead of raw SULR
    ROI_FEAT_PERCENTILES = [0.5, 0.9]
    # Use robust atlas (median + MAD-based std) for z-scoring instead of mean/std
    USE_ROBUST_Z_ATLAS = False
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
    baseline_X_list = []   # baseline SULR vectorized within mask
    atlas_feat_list = []   # atlas-based SULR features per patient
    feat_names = None      # feature names (set once from first patient)
    patient_ids = []       # to align optional VPS by SubjectID
    time_to_event = []     # days to relapse (or survival) from baseline
    event_indicator = []   # 1 if event occurred
    clinical_rows = []     # clinical covariates per patient
    keep_mask = None       # atlas mask (bool) to map back to volume
    # Track per-patient volumes for special-case saving
    subj_ids = []
    sulr_volumes = []   # SULR volumes in atlas space [H,W,D] for patients included in analysis
    pet_volumes = []    # original PET SUV volumes in atlas space [H,W,D] for patients included in analysis

    idx = 0
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
            subj_ids.append(str(sid))

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

            # choose outcome based on endpoint toggle
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
                    pet_volumes.append(suv_bl.detach().cpu().numpy().astype(np.float32))
                    sulr_volumes.append(sulr_bl.detach().cpu().numpy().astype(np.float32))
                    subj_ids.append(str(sid))
                else:
                    # no usable overall survival outcome; drop this patient
                    baseline_X_list.pop()
                    print('Skip: missing overall survival outcome')
                    continue
            else:
                if np.isfinite(t_rel) and e_rel in (0.0, 1.0):
                    time_to_event.append(t_rel)
                    event_indicator.append(int(e_rel))
                    # Track volumes for special-case saving only for included patients
                    pet_volumes.append(suv_bl.detach().cpu().numpy().astype(np.float32))
                    sulr_volumes.append(sulr_bl.detach().cpu().numpy().astype(np.float32))
                    subj_ids.append(str(sid))
                else:
                    # no usable relapse outcome; drop this patient
                    baseline_X_list.pop()
                    print('Skip: missing relapse outcome')
                    continue

            # build clinical covariates row for multivariable Cox
            clinical_rows.append(build_clinical_matrix(cov_raw[0, :].numpy(), cov_names))

            # Now that the patient is confirmed (has outcome), extract atlas features
            if USE_ATLAS_FEATURES:
                # Atlas-based biomarker features from remapped TotalSegmentator labels
                ct_lbl_vol = x_ct_full_seg[0,0]  # [H,W,D], long
                suv_lbl_vol = x_pt_seg[0,0]
                FEAT_PERCENTILES = [0.5, 0.90]  # configurable percentiles per ROI
                feats_ct, feat_names_ct = extract_quantification_features(sulr_bl_s, ct_lbl_vol, FEAT_PERCENTILES, feat_prefix='CT')
                feats_suv, feat_names_suv = extract_quantification_features(sulr_bl_s, suv_lbl_vol, FEAT_PERCENTILES, feat_prefix='SUV')
                # Optional: global lesion-burden features (TL-PSMA-like)
                # If using atlas z-detection, defer lesion features until after atlas stats are built
                feats_les = np.array([], dtype=float); feat_names_les = []
                # set names once (CT+SUV for now; lesion names will be appended later if enabled)
                if feat_names is None:
                    feat_names = feat_names_ct + feat_names_suv
                # concatenate CT- and SUV-derived features per patient (lesion features appended later if enabled)
                fvec = np.concatenate([feats_ct, feats_suv, feats_les], axis=0)
                atlas_feat_list.append(fvec)

    # stack patient-by-voxel predictor matrix
    Ximg = np.stack(baseline_X_list, axis=0)   # [N, Vmask]
    time_np = np.asarray(time_to_event, dtype=float)
    event_np = np.asarray(event_indicator, dtype=int)
    Xclin = np.stack(clinical_rows, axis=0)    # [N, Pclin]

    print(f'Dataset for prognostic analysis: N={Ximg.shape[0]} patients, Vmask={Ximg.shape[1]} voxels')

    # ---------- Voxelwise univariate Cox (baseline SULR -> time-to-event) ----------
    ran_voxelwise = False
    if USE_VOXELWISE:
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
        ran_voxelwise = True
    else:
        # set placeholders for summary
        V = Ximg.shape[1]
        cover = np.zeros(V, dtype=np.int32)
        events_cover = np.zeros(V, dtype=np.int32)
        valid_fit = np.zeros(V, dtype=bool)

    # ---------- Atlas-based biomarker / derived features ----------
    # Stack per-patient atlas features (may be empty if USE_ATLAS_FEATURES=False)
    if len(atlas_feat_list) == 0:
        atlas_X = np.zeros((Ximg.shape[0], 0), dtype=float)  # ensure correct N rows even if no atlas ROI features
    else:
        atlas_X = np.vstack([f for f in atlas_feat_list])  # [N, F]

    # Build/load statistical SULR atlas and compute lesion / z-score ROI features regardless of USE_ATLAS_FEATURES
    if USE_ATLAS_Z_DETECTION:
        mean_vec = None; std_vec = None
        if USE_ROBUST_Z_ATLAS:
            # Prefer robust atlas: median + MAD-based std
            median_path = os.path.join(population_stats_dir, 'sulr_atlas_median.nii.gz')
            madstd_path = os.path.join(population_stats_dir, 'sulr_atlas_madstd.nii.gz')
            if os.path.exists(median_path) and os.path.exists(madstd_path):
                try:
                    med_vol = nib.load(median_path).get_fdata().astype(np.float32)
                    mad_vol = nib.load(madstd_path).get_fdata().astype(np.float32)
                    mv_bool = keep_mask.reshape(-1)
                    mean_vec = med_vol.reshape(-1)[mv_bool]
                    std_vec  = mad_vol.reshape(-1)[mv_bool]
                except Exception as _e:
                    mean_vec = None; std_vec = None
            if mean_vec is None or std_vec is None:
                # compute from Ximg across patients (nanmedian + MAD per voxel)
                med = np.nanmedian(Ximg, axis=0).astype(np.float32)
                mad = np.nanmedian(np.abs(Ximg - med[None, :]), axis=0).astype(np.float32)
                madstd = np.maximum(1.4826 * mad, 1e-6).astype(np.float32)
                mean_vec = med
                std_vec  = madstd
                # save to disk as volumes for reuse
                save_vector_as_vol(mean_vec, median_path, x_pt_nib_aff, (H, W, D), keep_mask)
                save_vector_as_vol(std_vec,  madstd_path,  x_pt_nib_aff, (H, W, D), keep_mask)
        else:
            # Classic atlas: mean + std
            mean_path = os.path.join(population_stats_dir, 'sulr_atlas_mean.nii.gz')
            std_path  = os.path.join(population_stats_dir, 'sulr_atlas_std.nii.gz')
            if os.path.exists(mean_path) and os.path.exists(std_path):
                try:
                    mean_vol = nib.load(mean_path).get_fdata().astype(np.float32)
                    std_vol  = nib.load(std_path).get_fdata().astype(np.float32)
                    mv_bool = keep_mask.reshape(-1)
                    mean_vec = mean_vol.reshape(-1)[mv_bool]
                    std_vec  = std_vol.reshape(-1)[mv_bool]
                except Exception as _e:
                    mean_vec = None; std_vec = None
            if mean_vec is None or std_vec is None:
                # compute from Ximg across patients (nanmean/std per voxel)
                mean_vec = np.nanmean(Ximg, axis=0).astype(np.float32)
                std_vec  = np.nanstd(Ximg, axis=0).astype(np.float32)
                # floor std
                std_vec = np.where(np.isfinite(std_vec) & (std_vec >= 1e-6), std_vec, 1.0).astype(np.float32)
                # save to disk as volumes for reuse
                save_vector_as_vol(mean_vec, mean_path, x_pt_nib_aff, (H, W, D), keep_mask)
                save_vector_as_vol(std_vec,  std_path,  x_pt_nib_aff, (H, W, D), keep_mask)

        # Compute per-patient features from z-thresholded mask and/or ROI-level z percentiles
        lesion_feat_list = []
        lesion_feat_names = None
        roi_feat_list = []
        roi_ct_names = None
        roi_suv_names = None
        body_mask_np = mask_3d.detach().cpu().numpy().astype(bool)
        from scipy import ndimage as _ndi
        # Prepare labels on CPU for ROI extraction
        ct_lbl_cpu = x_ct_full_seg[0,0].cpu()
        suv_lbl_cpu = x_pt_seg[0,0].cpu()
        for n in range(Ximg.shape[0]):
            x = Ximg[n]
            # z-score vs atlas
            z = (x - mean_vec) / np.where(std_vec >= 1e-6, std_vec, 1.0)
            # rebuild 3D arrays
            sul3d = np.full((H, W, D), np.nan, dtype=np.float32)
            z3d   = np.zeros((H, W, D), dtype=np.float32)
            mv_bool = keep_mask.reshape(-1)
            sul3d.reshape(-1)[mv_bool] = x.astype(np.float32)
            z3d.reshape(-1)[mv_bool]   = z.astype(np.float32)
            # 1) Lesion features from z-thresholded mask
            if USE_LESION_FEATURES:
                lesion_mask = np.isfinite(z3d) & (z3d > float(LESION_Z_THR)) & body_mask_np
                if lesion_mask.any() and LESION_MIN_CC > 1:
                    lab, nlab = _ndi.label(lesion_mask.astype(np.uint8), structure=np.ones((3,3,3), dtype=np.uint8))
                    if nlab > 0:
                        sizes = _ndi.sum(np.ones_like(lesion_mask, dtype=np.float32), labels=lab, index=np.arange(1, nlab+1))
                        keep = (sizes >= float(LESION_MIN_CC))
                        if keep.any():
                            keep_idx = np.nonzero(keep)[0] + 1
                            lesion_mask = np.isin(lab, keep_idx)
                        else:
                            lesion_mask = np.zeros_like(lesion_mask, dtype=bool)
                feats_les, names_les = extract_lesion_burden_from_mask(sul3d, lesion_mask, x_pt_nib_aff)
                if lesion_feat_names is None:
                    lesion_feat_names = list(names_les)
                lesion_feat_list.append(feats_les.astype(float))
            # 2) ROI z-score percentiles
            if USE_ZSCORE_ROI_FEATURES:
                z_t = torch.from_numpy(z3d)
                feats_ct_z, names_ct_z = extract_quantification_features(z_t, ct_lbl_cpu, ROI_FEAT_PERCENTILES, feat_prefix='CTz')
                feats_suv_z, names_suv_z = extract_quantification_features(z_t, suv_lbl_cpu, ROI_FEAT_PERCENTILES, feat_prefix='SUVz')
                if roi_ct_names is None:
                    roi_ct_names = list(names_ct_z)
                if roi_suv_names is None:
                    roi_suv_names = list(names_suv_z)
                roi_feat = np.concatenate([feats_ct_z, feats_suv_z], axis=0)
                roi_feat_list.append(roi_feat.astype(float))

        # Append newly created blocks to atlas_X and extend feature names
        if USE_LESION_FEATURES and len(lesion_feat_list) == Ximg.shape[0]:
            lesion_X = np.vstack(lesion_feat_list)
            atlas_X = np.hstack([atlas_X, lesion_X]) if atlas_X.size else lesion_X
            if lesion_feat_names is not None:
                feat_names = (feat_names or []) + lesion_feat_names
        if USE_ZSCORE_ROI_FEATURES and len(roi_feat_list) == Ximg.shape[0]:
            roi_X = np.vstack(roi_feat_list)
            atlas_X = np.hstack([atlas_X, roi_X]) if atlas_X.size else roi_X
            roi_names_all = (roi_ct_names or []) + (roi_suv_names or [])
            if roi_names_all:
                # Place ROI z features before lesion names for readability
                feat_names = (roi_names_all) + (feat_names or [])

    # Univariate Cox per feature with BH-FDR across features
    def zscore(A):
        """Column-wise z-score with robust NaN handling.
        - Replaces NaNs with column means when available.
        - If an entire column is NaN, returns zeros for that column (mean=0, std=1).
        - Guards tiny/NaN stds.
        """
        A = np.asarray(A, dtype=float)
        finite_mask = np.isfinite(A)
        counts = finite_mask.sum(axis=0)
        m = np.nanmean(A, axis=0)
        s = np.nanstd(A, axis=0)
        # Columns with no finite values: set mean=0, std=1 later force zeros
        m = np.where(counts > 0, m, 0.0)
        s = np.where(np.isfinite(s) & (s >= 1e-9), s, 1.0)
        A_filled = np.where(finite_mask, A, m)
        Z = (A_filled - m) / s
        # For columns entirely NaN originally, force zeros to avoid NaNs downstream
        Z[:, counts == 0] = 0.0
        return Z, m, s

    # Compute clinical-only model regardless of feature availability
    try:
        Xclin_z_all, _, _ = zscore(Xclin)
        b_c_only, _, _, _ = cox_fit(Xclin_z_all, time_np, event_np, l2=1e-4, max_iter=60)
        c_clin = concordance_index(time_np, event_np, risk=(Xclin_z_all @ b_c_only))
        print(f'Clinical only: C-index={c_clin:.3f}')
    except Exception as _e:
        c_clin = None

    feat_beta = []; feat_p = []; feat_z = []; feat_cov = []; feat_evt = []
    if atlas_X.shape[0] == time_np.shape[0] and atlas_X.shape[0] > 0:
        Xf = atlas_X.copy()
        # coverage per feature
        for j in range(Xf.shape[1]):
            col = Xf[:, j]
            mfin = np.isfinite(col)
            n_cov = int(mfin.sum()); e_cov = int(event_np[mfin].sum())
            feat_cov.append(n_cov); feat_evt.append(e_cov)
            if n_cov < 15 or e_cov < 5:
                # insufficient coverage or events -> mark as null effect
                feat_beta.append(0.0); feat_z.append(0.0); feat_p.append(1.0)
                continue
            m = np.nanmean(col); s = np.nanstd(col); s = s if np.isfinite(s) and s >= 1e-9 else 1.0
            xz = (np.where(np.isfinite(col), col, m) - m) / s
            xz = np.clip(xz, -8.0, 8.0)
            b, se, z, p = cox_fit(xz, time_np, event_np, l2=1e-4, max_iter=60)
            b = np.atleast_1d(b); z = np.atleast_1d(z); p = np.atleast_1d(p)
            b0 = float(b[0]); z0 = float(z[0]); p0 = float(p[0])
            # If solver produced non-finite (separation/divergence), fall back to stronger ridge
            if not (np.isfinite(b0) and np.isfinite(z0) and np.isfinite(p0)):
                for l2_try in [1e-3, 1e-2, 1e-1, 1.0]:
                    b2, se2, z2, p2 = cox_fit(xz, time_np, event_np, l2=l2_try, max_iter=80)
                    b2 = np.atleast_1d(b2); z2 = np.atleast_1d(z2); p2 = np.atleast_1d(p2)
                    b0, z0, p0 = float(b2[0]), float(z2[0]), float(p2[0])
                    if np.isfinite(b0) and np.isfinite(z0) and np.isfinite(p0):
                        break
            if not (np.isfinite(b0) and np.isfinite(z0) and np.isfinite(p0)):
                # final safety: treat as null effect to avoid NaNs propagating
                b0, z0, p0 = 0.0, 0.0, 1.0
            feat_beta.append(b0); feat_z.append(z0); feat_p.append(p0)
        feat_beta = np.asarray(feat_beta); feat_z = np.asarray(feat_z); feat_p = np.asarray(feat_p)
        feat_cov = np.asarray(feat_cov); feat_evt = np.asarray(feat_evt)
        sig_feat05, q_feat05 = benjamini_hochberg(feat_p, q=0.05)
        sig_feat10, q_feat10 = benjamini_hochberg(feat_p, q=0.10)
        # Save CSV for per-feature results
        feat_csv = os.path.join(population_stats_dir, 'atlas_features_univariate_cox.csv')
        os.makedirs(population_stats_dir, exist_ok=True)
        with open(feat_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['feature','beta','HR','z','p','q05','q10','coverage','events'])
            names_iter = feat_names if feat_names is not None else [f'feat_{k}' for k in range(len(feat_beta))]
            for k, nm in enumerate(names_iter):
                HRk = np.exp(np.clip(feat_beta[k], -50.0, 50.0)) if np.isfinite(feat_beta[k]) else np.nan
                w.writerow([nm, feat_beta[k], HRk, feat_z[k], feat_p[k],
                            (q_feat05[k] if np.isfinite(q_feat05[k]) else np.nan),
                            (q_feat10[k] if np.isfinite(q_feat10[k]) else np.nan),
                            int(feat_cov[k]), int(feat_evt[k])])
        print(f'Saved atlas feature univariate Cox: {feat_csv}')

        # Multivariable: Clinical vs Clinical + top-K features
        TOPK = 20
        finite_idx = np.where(np.isfinite(feat_p))[0]
        order = finite_idx[np.argsort(feat_p[finite_idx])]
        sel = order[:min(TOPK, order.size)]
        Xclin_z, _, _ = zscore(Xclin)
        Xfeat_z, _, _ = zscore(atlas_X[:, sel]) if sel.size > 0 else (np.zeros((Xclin_z.shape[0], 0)), None, None)
        # Use NaN-free matrices for risk computation
        X_clin_only = Xclin_z
        X_with_feats = np.hstack([Xclin_z, Xfeat_z]) if Xfeat_z.shape[1] > 0 else Xclin_z
        X_clin_only_safe = np.nan_to_num(X_clin_only, nan=0.0, posinf=0.0, neginf=0.0)
        X_with_feats_safe = np.nan_to_num(X_with_feats, nan=0.0, posinf=0.0, neginf=0.0)
        b_c, _, _, _ = cox_fit(X_clin_only_safe, time_np, event_np, l2=1e-4, max_iter=60)
        b_c = np.nan_to_num(b_c, nan=0.0, posinf=0.0, neginf=0.0)
        b_cf, _, _, _ = cox_fit(X_with_feats_safe, time_np, event_np, l2=1e-3, max_iter=80)
        b_cf = np.nan_to_num(b_cf, nan=0.0, posinf=0.0, neginf=0.0)
        c_clin = concordance_index(time_np, event_np, risk=(X_clin_only_safe @ b_c))
        c_all  = concordance_index(time_np, event_np, risk=(X_with_feats_safe @ b_cf))
        # Features-only C-index (imaging only) if any features selected
        if Xfeat_z.shape[1] > 0:
            Xfeat_z_safe = np.nan_to_num(Xfeat_z, nan=0.0, posinf=0.0, neginf=0.0)
            b_f, _, _, _ = cox_fit(Xfeat_z_safe, time_np, event_np, l2=1e-3, max_iter=80)
            b_f = np.nan_to_num(b_f, nan=0.0, posinf=0.0, neginf=0.0)
            c_feat = concordance_index(time_np, event_np, risk=(Xfeat_z_safe @ b_f))
        else:
            c_feat = None
        print('\n=== Multivariable Cox (atlas features) ===')
        print(f'Clinical only: C-index={c_clin:.3f}')
        if c_feat is not None:
            print(f'Features only (top{sel.size}): C-index={c_feat:.3f}')
        else:
            print('Features only: N/A (no features selected)')
        print(f'Clinical + top{sel.size} atlas feats: C-index={c_all:.3f}')

        # Imaging-only ABS (Atlas Biomarker Score) and KM
        if sel.size > 0:
            # Build imaging-only linear predictor from selected features
            b_only = b_cf[-sel.size:]  # coefficients for imaging block
            b_only = np.nan_to_num(b_only, nan=0.0, posinf=0.0, neginf=0.0)
            Xfeat_z_safe = np.nan_to_num(Xfeat_z, nan=0.0, posinf=0.0, neginf=0.0)
            ABS = (Xfeat_z_safe @ b_only)
            try:
                abs_z, _, _ = zscore(ABS[:, None])
                grp_abs_hi = abs_z[:, 0] >= np.nanmedian(abs_z[:, 0])
                plot_km_with_ticks_two_groups(time_np, event_np, grp_abs_hi, out_path=population_stats_dir + 'km_abs_median.png', title='KM: Atlas Biomarker Score (median split)')
                print('Saved: km_abs_median.png')
            except Exception as ex:
                print(f'KM ABS plot skipped due to error: {ex}')

            # Clinical + ABS median split
            try:
                risk_all = (X_with_feats_safe @ b_cf).ravel()
                grp_all_hi = risk_all >= np.nanmedian(risk_all)
                plot_km_with_ticks_two_groups(time_np, event_np, grp_all_hi, out_path=population_stats_dir + 'km_clinical_plus_abs_median.png', title='KM: clinical + ABS (median split)')
                print('Saved: km_clinical_plus_abs_median.png')
            except Exception as ex:
                print(f'KM clinical+ABS plot skipped due to error: {ex}')

    # ---------- Optional: evaluate precomputed VPS incremental value ----------
    if EVAL_VPS_INCREMENTAL and VPS_PATH is not None:
        def _load_vps_vector(path, ids_order):
            arr = None
            if path.lower().endswith('.npy'):
                arr = np.load(path)
                arr = np.asarray(arr, dtype=float).reshape(-1)
                if arr.shape[0] != len(ids_order):
                    print(f'VPS .npy length {arr.shape[0]} does not match N={len(ids_order)}; skipping VPS eval.')
                    return None
                return arr
            if path.lower().endswith('.csv'):
                try:
                    with open(path, 'r') as f:
                        rdr = csv.DictReader(f)
                        mp = {}
                        for row in rdr:
                            sid = str(row.get('SubjectID', '')).strip()
                            try:
                                v = float(row.get('VPS', 'nan'))
                            except Exception:
                                v = np.nan
                            if sid:
                                mp[sid] = v
                        vec = [mp.get(str(sid), np.nan) for sid in ids_order]
                        return np.asarray(vec, dtype=float)
                except Exception as ex:
                    print(f'Failed to parse VPS CSV: {ex}')
                    return None
            print('Unsupported VPS file type; use .npy or .csv with SubjectID,VPS columns.')
            return None

        vps_vec = _load_vps_vector(VPS_PATH, patient_ids)
        if vps_vec is not None:
            # Build clinical+atlas design (use the top-K selected features if available)
            Xclin_z, _, _ = zscore(Xclin)
            if 'Xfeat_z' in locals():
                Xfeat_use = Xfeat_z
            else:
                Xfeat_use = np.zeros((Xclin_z.shape[0], 0))
            X_ca = np.hstack([Xclin_z, Xfeat_use])
            VPS_z, _, _ = zscore(vps_vec[:, None])
            # Model with VPS added
            X_with_vps = np.hstack([X_ca, VPS_z])
            b_cav, _, _, _ = cox_fit(X_with_vps, time_np, event_np, l2=1e-4, max_iter=60)
            c_cav = concordance_index(time_np, event_np, risk=(X_with_vps @ b_cav))
            # Orthogonalized VPS
            VPS_res = orthogonalize_against(VPS_z[:, 0], X_ca)
            X_with_vps_orth = np.hstack([X_ca, VPS_res])
            b_cav_o, _, _, _ = cox_fit(X_with_vps_orth, time_np, event_np, l2=1e-4, max_iter=60)
            c_cav_o = concordance_index(time_np, event_np, risk=(X_with_vps_orth @ b_cav_o))
            print('\n=== Optional VPS incremental evaluation ===')
            print(f'Clinical+atlas: C-index={concordance_index(time_np, event_np, risk=(X_ca @ cox_fit(X_ca, time_np, event_np)[0])):.3f}')
            print(f'+ VPS:         C-index={c_cav:.3f}')
            print(f'+ VPS⊥:        C-index={c_cav_o:.3f}')
        else:
            print('VPS not evaluated (failed to load).')

    # Save a small report for reproducibility
    # compute voxelwise significant count for reporting
    try:
        sig_idx = np.where(sig_mask05)[0]
    except Exception:
        sig_idx = np.array([], dtype=int)
    with open(population_stats_dir + 'cox_summary.txt', 'w') as f:
        f.write(f'Patients N={Ximg.shape[0]}, Vmask={Ximg.shape[1]}\n')
        f.write(f'Events={int(event_np.sum())}, Median time={np.nanmedian(time_np):.1f} days\n')
        # Record selected endpoint for reproducibility
        try:
            f.write('Endpoint=' + ('Overall Survival' if USE_OVERALL_SURVIVAL else 'Relapse') + '\n')
        except Exception:
            pass
        try:
            f.write(f'Clinical C-index={c_clin:.3f}\n')
        except Exception:
            pass
        # Append atlas-features Cox summary if available
        try:
            f.write(f'Clinical+topK_atlas_features C-index={c_all:.3f}\n')
            f.write(f'Number of atlas features considered={atlas_X.shape[1]} (selected topK by p-value)\n')
            try:
                # If features-only C-index was computed
                if 'c_feat' in locals() and c_feat is not None:
                    f.write(f'Features-only (topK) C-index={c_feat:.3f}\n')
                    f.write(f'Delta C-index (clinical+features vs clinical)={c_all - c_clin:.3f}\n')
                    f.write(f'Delta C-index (clinical+features vs features-only)={c_all - c_feat:.3f}\n')
                else:
                    f.write('Features-only C-index=N/A (no features selected)\n')
            except Exception:
                pass
        except Exception:
            pass
        if ran_voxelwise:
            f.write(f'Significant voxels (FDR q<=0.05)={sig_idx.size}\n')
            inv_frac = 1.0 - float(np.mean(valid_fit)) if valid_fit.size else float('nan')
            f.write(f'Invalid voxels (NaN in z/beta/p)={int((~valid_fit).sum()) if valid_fit.size else 0}  ({(inv_frac*100 if np.isfinite(inv_frac) else float("nan")):.1f}%)\n')
            try:
                f.write(f'Median coverage (valid voxels)={np.nanmedian(cover[valid_fit]):.1f}\n')
                f.write(f'Median events (valid voxels)={np.nanmedian(events_cover[valid_fit]):.1f}\n')
            except Exception:
                f.write('Median coverage (valid voxels)=N/A\n')
                f.write('Median events (valid voxels)=N/A\n')
        else:
            f.write('Voxelwise Cox: skipped (USE_VOXELWISE=False)\n')
        f.write('Clinical covariates included: age, log_psa_at_scan (fallback pre/initial), '
                'grade_group, t_ord, BMI, indication (primary/recurrence/metastatic), '
                'pre_androgen_targeted, pre_cytotoxic.\n')
        f.write('Excluded post-PSMA covariates (post_local, post_focal, post_at, post_cyto).\n')
    if ran_voxelwise:
        print('Saved: hr_map.nii.gz, z_map.nii.gz, p_map.nii.gz, '
            'q_map_q05.nii.gz, q_map_q10.nii.gz, q_map_vis_q05.nii.gz, q_map_vis_q10.nii.gz, '
            'sig_mask_fdr_q05.nii.gz, sig_mask_fdr_q10.nii.gz, '
            'hr_gt1_fdr_q05.nii.gz, inv_hr_protective_fdr_q05.nii.gz, '
            'hr_gt1_fdr_q10.nii.gz, inv_hr_protective_fdr_q10.nii.gz, cox_summary.txt')
    else:
        print('Saved: cox_summary.txt (voxelwise maps skipped)')
        
    # ---------- Save PET and z-score images for special cases (high/low risk) ----------
    try:
        # Build a risk vector: prefer Clinical + top-K atlas features, else clinical-only
        if 'b_cf' in locals():
            risk_all = (X_with_feats_safe @ b_cf).ravel()
        else:
            risk_all = (X_clin_only_safe @ b_c).ravel()
        if len(risk_all) != len(sulr_volumes):
            # Fallback: align by the number of analyzed patients (Ximg rows)
            Nuse = min(len(risk_all), len(sulr_volumes))
            risk_use = np.asarray(risk_all[:Nuse])
            vols_idx_hi = int(np.nanargmax(risk_use))
            vols_idx_lo = int(np.nanargmin(risk_use))
        else:
            idx_hi = int(np.nanargmax(risk_all))
            idx_lo = int(np.nanargmin(risk_all))
            vols_idx_hi = idx_hi
            vols_idx_lo = idx_lo
            # Choose atlas stats based on robust toggle
            if USE_ATLAS_Z_DETECTION:
                if 'mean_vec' in locals() and 'std_vec' in locals():
                    # Rebuild full 3D mean/std volumes from vectors
                    mv_bool = keep_mask.reshape(-1)
                    mean_vol = np.full((H, W, D), np.nan, dtype=np.float32)
                    std_vol  = np.ones((H, W, D), dtype=np.float32)
                    mean_vol.reshape(-1)[mv_bool] = mean_vec
                    std_vol.reshape(-1)[mv_bool]  = np.where(std_vec >= 1e-6, std_vec, 1.0)
                else:
                    # Fallback: cohort stats from running lesion_stats
                    mean_vol = lesion_stats.mean().detach().cpu().numpy().astype(np.float32)
                    std_tmp  = lesion_stats.std().detach().cpu().numpy().astype(np.float32)
                    std_vol  = np.where(std_tmp >= 1e-6, std_tmp, 1.0)
            else:
                # If atlas z detection is disabled, still use running stats
                mean_vol = lesion_stats.mean().detach().cpu().numpy().astype(np.float32)
                std_tmp  = lesion_stats.std().detach().cpu().numpy().astype(np.float32)
                std_vol  = np.where(std_tmp >= 1e-6, std_tmp, 1.0)

            # Compose z-score maps
            z_hi = (sulr_volumes[vols_idx_hi] - mean_vol) / std_vol
            z_lo = (sulr_volumes[vols_idx_lo] - mean_vol) / std_vol
            # Outside mask, set to 0 for viewer friendliness
            msk = mask_3d.detach().cpu().numpy()
            z_hi[~msk] = 0.0; z_lo[~msk] = 0.0
            pet_hi = pet_volumes[vols_idx_hi]; pet_lo = pet_volumes[vols_idx_lo]
            sid_hi = subj_ids[vols_idx_hi]; sid_lo = subj_ids[vols_idx_lo]
            out_dir = os.path.join(population_stats_dir, 'special_cases')
            os.makedirs(out_dir, exist_ok=True)
            nib.save(nib.Nifti1Image(pet_hi.astype(np.float32), x_pt_nib_aff), os.path.join(out_dir, f'{sid_hi}_PET_high_risk.nii.gz'))
            nib.save(nib.Nifti1Image(z_hi.astype(np.float32),   x_pt_nib_aff), os.path.join(out_dir, f'{sid_hi}_Z_high_risk.nii.gz'))
            nib.save(nib.Nifti1Image(pet_lo.astype(np.float32), x_pt_nib_aff), os.path.join(out_dir, f'{sid_lo}_PET_low_risk.nii.gz'))
            nib.save(nib.Nifti1Image(z_lo.astype(np.float32),   x_pt_nib_aff), os.path.join(out_dir, f'{sid_lo}_Z_low_risk.nii.gz'))
            print(f'Saved special cases to: {out_dir}\n  High-risk: {sid_hi}\n  Low-risk: {sid_lo}')
    except Exception as ex:
        print(f'Special-case PET/Z saving failed: {ex}')

def extract_quantification_features(sulr_img: torch.Tensor, lbls: torch.Tensor, percentiles, feat_prefix='ct'):
    """Extract per-ROI percentiles from a spatially normalized SULR volume.

    Args:
        sulr_img: torch.Tensor [H,W,D], float (CPU or CUDA).
        lbls: torch.Tensor [H,W,D], long atlas labels (0=background).
        percentiles: list of floats in (0,1], e.g., [0.5, 0.9].

    Returns:
        feats_np: 1D numpy array of length (n_labels_without_bg * len(percentiles))
        names:    list of feature names aligned with feats_np
    """
    device = sulr_img.device
    lbls = lbls.to(device)
    sul = sulr_img
    uniq = torch.unique(lbls)
    uniq = uniq[uniq != 0]
    uniq = uniq.sort().values
    names = []
    vals = []
    # ensure numeric stability: mask finite
    finite = torch.isfinite(sul)
    for lbl in uniq.tolist():
        if "SUV" in feat_prefix.upper() and lbl > 13:
            continue  # skip tumor labels for SUV features
        if lbl == 0:
            continue  # skip background
        m = (lbls == int(lbl)) & finite
        v = sul[m]
        if v.numel() == 0:
            for p in percentiles:
                names.append(f'{feat_prefix}_label{int(lbl)}_p{int(p*100)}')
                vals.append(float('nan'))
            continue
        # quantiles for this ROI
        for p in percentiles:
            try:
                qv = torch.quantile(v, float(p))
                qvf = float(qv.detach().cpu())
            except Exception:
                qvf = float('nan')
            names.append(f'{feat_prefix}_label{int(lbl)}_p{int(p*100)}')
            vals.append(qvf)
    feats_np = np.array(vals, dtype=float)
    return feats_np, names
    
def extract_lesion_burden_features(sulr_img: torch.Tensor, body_mask: torch.Tensor, affine: np.ndarray,
                                   thresh: float = 1.5, min_cc: int = 20):
    """Compute global PSMA lesion burden features from SULR image.

    Features (8):
      - LESION_count: number of connected components above threshold
      - LESION_PSMA_TV: total lesion volume (voxels * voxel_vol)
      - LESION_TLPSMA: sum of SULR over lesion voxels times voxel_vol
      - LESION_SULRmax: maximum SULR in lesions
      - LESION_p95: 95th percentile of SULR in lesions
      - LESION_largest_vol: volume of largest lesion component
      - LESION_mean: mean SULR within lesions
      - LESION_spread: mean pairwise centroid distance between lesions (in mm)
    """
    import numpy as _np
    from scipy import ndimage as _ndi
    sul = sulr_img.detach().float().cpu().numpy()
    m = body_mask.detach().cpu().numpy().astype(bool)
    # thresholded tumor mask in SULR space
    tmask = (_np.isfinite(sul)) & m & (sul > float(thresh))
    # 26-connectivity structure
    struct = _np.ones((3,3,3), dtype=_np.uint8)
    lab, nlab = _ndi.label(tmask.astype(_np.uint8), structure=struct)
    names = ['LESION_count','LESION_PSMA_TV','LESION_TLPSMA','LESION_SULRmax','LESION_p95','LESION_largest_vol','LESION_mean','LESION_spread']
    if nlab == 0:
        return _np.array([0, 0.0, 0.0, 0.0, 0.0, 0.0, _np.nan, 0.0], dtype=float), names
    sizes = _ndi.sum(_np.ones_like(sul, dtype=_np.float32), labels=lab, index=_np.arange(1, nlab+1))
    keep = (sizes >= float(min_cc))
    if not keep.any():
        return _np.array([0, 0.0, 0.0, _np.nan, _np.nan, 0.0, _np.nan, 0.0], dtype=float), names
    keep_idx = _np.nonzero(keep)[0] + 1
    keep_mask = _np.isin(lab, keep_idx)
    lab2, nlab2 = _ndi.label(keep_mask.astype(_np.uint8), structure=struct)
    # voxel volume from affine (mm^3 if affine encodes mm)
    try:
        voxel_vol = float(abs(_np.linalg.det(affine[:3, :3])))
    except Exception:
        voxel_vol = 1.0
    lesion_vals = sul[lab2 > 0]
    lesion_count = int(nlab2)
    psma_tv = float(lesion_vals.size * voxel_vol)
    tl_psma = float(_np.nansum(lesion_vals) * voxel_vol)
    smax = float(_np.nanmax(lesion_vals)) if lesion_vals.size > 0 else 0.0
    p95 = float(_np.nanpercentile(lesion_vals, 95.0)) if lesion_vals.size > 0 else 0.0
    largest_vol = float(_np.max(_ndi.sum(_np.ones_like(sul, dtype=_np.float32), labels=lab2, index=_np.arange(1, nlab2+1))) * voxel_vol)
    smean = float(_np.nanmean(lesion_vals)) if lesion_vals.size > 0 else _np.nan
    # centroid spread in mm (approx)
    centroids = _ndi.center_of_mass(_np.ones_like(sul, dtype=_np.float32), labels=lab2, index=_np.arange(1, nlab2+1))
    centroids = _np.array(centroids, dtype=float)  # (z,y,x)
    try:
        spac = _np.sqrt((_np.asarray(affine[:3, :3])**2).sum(axis=0))
        centroids_mm = centroids[:, ::-1] * spac  # convert to (x,y,z) spacing; centroids (z,y,x)
    except Exception:
        centroids_mm = centroids
    spread = 0.0
    if lesion_count >= 2:
        dsum = 0.0; k = 0
        for i in range(lesion_count):
            for j in range(i+1, lesion_count):
                di = centroids_mm[i] - centroids_mm[j]
                dsum += float(_np.sqrt((di**2).sum()))
                k += 1
        spread = float(dsum / max(k, 1))
    feats = _np.array([lesion_count, psma_tv, tl_psma, smax, p95, largest_vol, smean, spread], dtype=float)
    return feats, names

def extract_lesion_burden_from_mask(sul_np3d: np.ndarray, lesion_mask3d: np.ndarray, affine: np.ndarray):
    """Compute lesion-burden features using a provided binary lesion mask.

    Inputs:
      - sul_np3d: 3D numpy array of SULR in atlas space (H,W,D), NaNs allowed outside mask
      - lesion_mask3d: 3D boolean array, same shape, True for lesion voxels
      - affine: 4x4 affine for voxel spacing

    Returns:
      feats: np.array of shape (8,)
      names: list of 8 feature names
    """
    import numpy as _np
    from scipy import ndimage as _ndi
    names = ['LESION_count','LESION_PSMA_TV','LESION_TLPSMA','LESION_SULRmax','LESION_p95','LESION_largest_vol','LESION_mean','LESION_spread']
    if sul_np3d is None or lesion_mask3d is None:
        return _np.array([0,0.0,0.0,0.0,0.0,0.0,_np.nan,0.0], dtype=float), names
    try:
        voxel_vol = float(abs(_np.linalg.det(affine[:3, :3])))
    except Exception:
        voxel_vol = 1.0
    tmask = _np.asarray(lesion_mask3d, dtype=bool)
    if not tmask.any():
        return _np.array([0, 0.0, 0.0, 0.0, 0.0, 0.0, _np.nan, 0.0], dtype=float), names
    struct = _np.ones((3,3,3), dtype=_np.uint8)
    lab, nlab = _ndi.label(tmask.astype(_np.uint8), structure=struct)
    if nlab == 0:
        return _np.array([0, 0.0, 0.0, 0.0, 0.0, 0.0, _np.nan, 0.0], dtype=float), names
    lesion_vals = sul_np3d[lab > 0]
    lesion_vals = lesion_vals[_np.isfinite(lesion_vals)]
    lesion_count = int(nlab)
    psma_tv = float(int(tmask.sum()) * voxel_vol)
    tl_psma = float(_np.nansum(lesion_vals) * voxel_vol)
    smax = float(_np.nanmax(lesion_vals)) if lesion_vals.size > 0 else 0.0
    p95 = float(_np.nanpercentile(lesion_vals, 95.0)) if lesion_vals.size > 0 else 0.0
    sizes = _ndi.sum(_np.ones_like(sul_np3d, dtype=_np.float32), labels=lab, index=_np.arange(1, nlab+1))
    largest_vol = float((_np.max(sizes) if sizes.size>0 else 0.0) * voxel_vol)
    smean = float(_np.nanmean(lesion_vals)) if lesion_vals.size > 0 else _np.nan
    centroids = _ndi.center_of_mass(_np.ones_like(sul_np3d, dtype=_np.float32), labels=lab, index=_np.arange(1, nlab+1))
    centroids = _np.array(centroids, dtype=float)  # (z,y,x)
    try:
        spac = _np.sqrt((_np.asarray(affine[:3, :3])**2).sum(axis=0))
        centroids_mm = centroids[:, ::-1] * spac
    except Exception:
        centroids_mm = centroids
    spread = 0.0
    if lesion_count >= 2:
        dsum = 0.0; k = 0
        for i in range(lesion_count):
            for j in range(i+1, lesion_count):
                di = centroids_mm[i] - centroids_mm[j]
                dsum += float(_np.sqrt((di**2).sum()))
                k += 1
        spread = float(dsum / max(k, 1))
    feats = _np.array([lesion_count, psma_tv, tl_psma, smax, p95, largest_vol, smean, spread], dtype=float)
    return feats, names
    
def binary_dilation_3d_max_pool(binary_volume, kernel_size, padding):
    """
    Performs 3D binary dilation using max_pool3d.

    Args:
        binary_volume (torch.Tensor): Input binary 3D tensor (e.g., shape (C, D, H, W)).
                                      Values should be 0 or 1.
        kernel_size (int or tuple): Size of the structuring element (e.g., 3 for a 3x3x3 cube).
        padding (int or tuple): Padding to apply.

    Returns:
        torch.Tensor: Dilated binary 3D tensor.
    """
    # Ensure input is float and has the correct dimensions (e.g., C, D, H, W for max_pool3d)
    if binary_volume.ndim == 3:
        binary_volume = binary_volume.unsqueeze(0).unsqueeze(0) # Add batch and channel dim
    elif binary_volume.ndim == 4 and binary_volume.shape[0] != 1:
        # If it's (B, D, H, W), add a channel dimension
        binary_volume = binary_volume.unsqueeze(1)
    
    dilated_volume = F.max_pool3d(binary_volume.float(), kernel_size=kernel_size, stride=1, padding=padding)
    return (dilated_volume > 0).float() # Convert back to binary (0 or 1)

def plot_km_with_ticks_two_groups(time_days: np.ndarray, event: np.ndarray, grp_high: np.ndarray,
                                  out_path: str, title: str = 'KM: two groups with censor ticks'):
    """Plot KM curves for two groups (high vs low) with censor tick marks.

    Args:
        time_days: array of times in days (N,)
        event: array of 1=event, 0=censored (N,)
        grp_high: boolean array (N,) True for high-risk group, False for low-risk
        out_path: file path to save PNG
        title: plot title
    """
    import numpy as _np
    import matplotlib.pyplot as _plt

    def _km_curve(t, e):
        # Sort by time
        order = _np.argsort(t)
        t = _np.asarray(t, dtype=float)[order]
        e = _np.asarray(e, dtype=int)[order]
        # Unique event times
        uniq_times = _np.unique(t)
        n_at_risk = float(len(t))
        surv = 1.0
        xs = [0.0]; ys = [1.0]
        censor_times = []
        censor_levels = []
        # Greenwood variance accumulator: sum(d / (n*(n-d))) over event times
        var_acc = 0.0
        ci_x = [0.0]; ci_lo = [1.0]; ci_hi = [1.0]
        i = 0
        for ut in uniq_times:
            # individuals at this exact time
            mask_ut = (t == ut)
            # number of events at ut
            d = float((e[mask_ut] == 1).sum())
            # number censored at ut
            c = float((e[mask_ut] == 0).sum())
            if n_at_risk > 0 and d > 0:
                surv *= (1.0 - d / n_at_risk)
                xs.append(ut); ys.append(ys[-1])  # horizontal step
                xs.append(ut); ys.append(surv)    # drop at event time
                # Greenwood variance at current step
                var_acc += (d / (n_at_risk * max(n_at_risk - d, 1.0)))
                se = (surv ** 2) * var_acc
                # Normal approx 95% CI; clip to [0,1]
                delta = 1.96 * _np.sqrt(max(se, 0.0))
                lo = float(max(0.0, surv - delta))
                hi = float(min(1.0, surv + delta))
                ci_x.append(ut); ci_lo.append(ci_lo[-1])
                ci_hi.append(ci_hi[-1])
                ci_x.append(ut); ci_lo.append(lo)
                ci_hi.append(hi)
            # record censored positions (tick marks)
            if c > 0:
                censor_times.extend([ut] * int(c))
                censor_levels.extend([surv] * int(c))
            # update risk set size after handling all at ut
            n_at_risk -= (d + c)
        return (_np.asarray(xs), _np.asarray(ys),
                _np.asarray(censor_times), _np.asarray(censor_levels),
                _np.asarray(ci_x), _np.asarray(ci_lo), _np.asarray(ci_hi))

    # Prepare groups
    grp_high = _np.asarray(grp_high, dtype=bool)
    t = _np.asarray(time_days, dtype=float)
    e = _np.asarray(event, dtype=int)
    # Guard: remove NaNs
    mask_fin = _np.isfinite(t) & _np.isfinite(e)
    t = t[mask_fin]; e = e[mask_fin]; grp_high = grp_high[mask_fin]
    # Curves
    xH, yH, cH_t, cH_y, ciHx, ciHlo, ciHhi = _km_curve(t[grp_high], e[grp_high])
    xL, yL, cL_t, cL_y, ciLx, ciLlo, ciLhi = _km_curve(t[~grp_high], e[~grp_high])
    # Plot
    _plt.figure(figsize=(6.5, 4.5), dpi=140)
    _plt.step(xH, yH, where='post', color='tab:red', label='High group')
    _plt.step(xL, yL, where='post', color='tab:blue', label='Low group')
    # 95% CI bands (Greenwood, normal approx)
    if ciHx.size:
        _plt.fill_between(ciHx, ciHlo, ciHhi, color='tab:red', alpha=0.15, step='post')
    if ciLx.size:
        _plt.fill_between(ciLx, ciLlo, ciLhi, color='tab:blue', alpha=0.15, step='post')
    # Censor ticks: short vertical ticks at censor times
    if cH_t.size > 0:
        _plt.scatter(cH_t, cH_y, marker='|', color='tab:red', s=80, linewidths=1.5)
    if cL_t.size > 0:
        _plt.scatter(cL_t, cL_y, marker='|', color='tab:blue', s=80, linewidths=1.5)
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