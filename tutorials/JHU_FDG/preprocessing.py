import numpy as np
import nibabel as nib
import os, subprocess, sys
import glob
import tempfile
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.ndimage import zoom
from MIR.models import SpatialTransformer, AffineReg3D
import torch
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.filters import threshold_otsu
from scipy import ndimage
import MIR.utils.other_utils as utils
from nibabel.filebasedimages import ImageFileError

def zoom_img(img, pixel_dims, order=3):
    img_pixdim, tar_pix = pixel_dims
    ratio = np.array(img_pixdim) / np.array(tar_pix)
    img = zoom(img, ratio, order=order)
    return img

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC

def remove_bed(img):
    img_ = img.copy()
    img_ = (img_-img_.min())/(img_.max()-img_.min())
    threshold_global_otsu = threshold_otsu(img_)
    mask = img >= -800#threshold_global_otsu
    #mask = (mask + seg_simp>0)>0
    mask = ndimage.binary_erosion(mask).astype(mask.dtype)
    mask = ndimage.binary_fill_holes(mask).astype(mask.dtype)
    mask = ndimage.binary_dilation(mask).astype(mask.dtype)
    mask3D = getLargestCC(mask)
    mask3D = ndimage.binary_dilation(mask3D).astype(mask3D.dtype)
    mask3D = ndimage.binary_closing(mask3D).astype(mask3D.dtype)
    img[mask3D == 0] = np.percentile(img, 0.5)
    return img

def norm_ct(img):
    img[img < -300] = -300
    img[img > 300] = 300
    norm = (img - img.min()) / (img.max() - img.min())
    return norm

def make_affine_from_pixdim(pixdim):
    # Create a 4x4 affine with spacing along the diagonal
    affine = np.eye(4)
    affine[0, 0] = pixdim[0]
    affine[1, 1] = pixdim[1]
    affine[2, 2] = pixdim[2]
    return affine

BASE_DIR = '/scratch2/jchen/DATA/JHU_FDG/batch7/'

SEG_DIR = BASE_DIR + 'seg/ct/'
os.makedirs(SEG_DIR, exist_ok=True)
AFF_DIR = BASE_DIR + 'affine/'
os.makedirs(AFF_DIR, exist_ok=True)
OUTPUT_DIR = BASE_DIR + 'preprocessed/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOTALSEG_WORKERS = 5
TAR_PIXDIM = [2.8, 2.8, 3.8]
MIN_FOV_RATIO_TO_ATLAS = 0.7
EXCLUDE_LOG_PATH = BASE_DIR + 'excluded_fov_cases.txt'

# TotalSegmentator multilabel IDs used for FOV QC/cropping.
LABEL_BRAIN = 90
LABEL_SKULL = 91
LABEL_VERTEBRAE_S1 = 26
LABEL_HIP_LEFT = 77
LABEL_HIP_RIGHT = 78
LABEL_FEMUR_LEFT = 75
LABEL_FEMUR_RIGHT = 76

SUPERIOR_LABELS = [LABEL_SKULL, LABEL_BRAIN]
INFERIOR_ANCHOR_LABELS = [LABEL_FEMUR_LEFT, LABEL_FEMUR_RIGHT, LABEL_HIP_LEFT, LABEL_HIP_RIGHT, LABEL_VERTEBRAE_S1]
FEMUR_LABELS = [LABEL_FEMUR_LEFT, LABEL_FEMUR_RIGHT]
HIP_LABELS = [LABEL_HIP_LEFT, LABEL_HIP_RIGHT]

CT_ATLAS = '/scratch/jchen/python_projects/AutoPET/atlas/ct/TransMorphAtlas_MAE_1_MS_1_diffusion_1/dsc0.5425.nii.gz'
CT_ATLAS_NIB = nib.load(CT_ATLAS)
CT_ATLAS_NPY = CT_ATLAS_NIB.get_fdata()
CT_ATLAS_NPY = np.flip(CT_ATLAS_NPY, 1).copy()
CT_ATLAS_TORCH = torch.from_numpy(CT_ATLAS_NPY).float().unsqueeze(0).unsqueeze(0).cuda(0)
spatial_trans_ct = SpatialTransformer(size=CT_ATLAS_TORCH.shape[2:], mode='bilinear').cuda(0)
spatial_trans_nn = SpatialTransformer(size=CT_ATLAS_TORCH.shape[2:], mode='nearest').cuda(0)


def fill_outside_valid_region(warped_img, flow, pad_value=-1000, eps=1e-4, valid_thr=0.995):
    # Use bilinear valid-weight map, then keep only fully valid voxels to avoid
    # partial-volume boundary speckles/rings from original FOV edges.
    ones = torch.ones_like(warped_img)
    valid_w = spatial_trans_ct(ones, flow)
    corrected = warped_img / torch.clamp(valid_w, min=eps)
    valid_mask = valid_w > valid_thr
    corrected = torch.clamp(corrected, min=-1024.0, max=3071.0)
    return torch.where(valid_mask, corrected, torch.full_like(warped_img, pad_value))


def _axis_median_coord(mask, axis):
    idx = np.where(mask)
    if len(idx[0]) == 0:
        return None
    return float(np.median(idx[axis]))


def _find_si_axis_and_direction(seg_labels):
    sup_mask = np.isin(seg_labels, SUPERIOR_LABELS)
    inf_mask = np.isin(seg_labels, INFERIOR_ANCHOR_LABELS)
    if sup_mask.sum() == 0 or inf_mask.sum() == 0:
        return None, None

    best_axis = None
    best_sep = -1.0
    best_dir = None
    for axis in range(3):
        sup_c = _axis_median_coord(sup_mask, axis)
        inf_c = _axis_median_coord(inf_mask, axis)
        if sup_c is None or inf_c is None:
            continue
        sep = abs(sup_c - inf_c)
        if sep > best_sep:
            best_sep = sep
            best_axis = axis
            best_dir = 1 if sup_c > inf_c else -1  # +1 means index increases towards superior.
    return best_axis, best_dir


def _compute_mid_thigh_to_vertex_bounds(seg_labels, axis, si_dir):
    sup_mask = np.isin(seg_labels, SUPERIOR_LABELS)
    if sup_mask.sum() == 0:
        return None, None

    sup_idx = np.where(sup_mask)[axis]
    top_idx = int(sup_idx.max()) if si_dir == 1 else int(sup_idx.min())

    femur_mask = np.isin(seg_labels, FEMUR_LABELS)
    if femur_mask.sum() > 0:
        fem_idx = np.where(femur_mask)[axis]
        fem_min, fem_max = int(fem_idx.min()), int(fem_idx.max())
        bottom_idx = int(round((fem_min + fem_max) / 2.0))  # mid-thigh approximation.
    else:
        # Fallback if femur is not available: use inferior hip end.
        hip_mask = np.isin(seg_labels, HIP_LABELS)
        if hip_mask.sum() == 0:
            return None, None
        hip_idx = np.where(hip_mask)[axis]
        bottom_idx = int(hip_idx.min()) if si_dir == 1 else int(hip_idx.max())

    if si_dir == 1:
        lo, hi = bottom_idx, top_idx
    else:
        lo, hi = top_idx, bottom_idx
    lo = max(0, lo)
    hi = min(seg_labels.shape[axis] - 1, hi)
    if hi <= lo:
        return None, None
    return lo, hi


def _crop_by_axis_bounds(vol, axis, lo, hi):
    slicer = [slice(None)] * vol.ndim
    slicer[axis] = slice(lo, hi + 1)
    return vol[tuple(slicer)]


def _map_bounds_between_shapes(src_len, dst_len, lo, hi):
    lo_r = float(lo) / float(src_len)
    hi_r = float(hi + 1) / float(src_len)
    dst_lo = int(np.floor(lo_r * dst_len))
    dst_hi = int(np.ceil(hi_r * dst_len)) - 1
    dst_lo = max(0, min(dst_len - 1, dst_lo))
    dst_hi = max(dst_lo, min(dst_len - 1, dst_hi))
    return dst_lo, dst_hi


def quality_control_and_crop(ct_samp, pt_samp, ct_seg_samp, atlas_shape):
    seg_lbl = np.rint(ct_seg_samp).astype(np.int16)
    axis, si_dir = _find_si_axis_and_direction(seg_lbl)
    if axis is None:
        return None, None, None, False, 'Missing superior/inferior anchors in CT segmentation.', None

    lo, hi = _compute_mid_thigh_to_vertex_bounds(seg_lbl, axis, si_dir)
    if lo is None:
        return None, None, None, False, 'Cannot determine crop bounds (missing skull/brain or hips/femurs).', None

    crop_len = hi - lo + 1
    atlas_len = atlas_shape[axis]
    fov_ratio = float(crop_len) / float(max(atlas_len, 1))
    # Mask atlas whenever subject support is not full-length in atlas space.
    # Keep MIN_FOV_RATIO_TO_ATLAS only for QC reporting; no hard exclusion here.
    is_small_fov = crop_len < atlas_len

    ct_len = ct_samp.shape[axis]
    pt_len = pt_samp.shape[axis]
    pt_lo, pt_hi = _map_bounds_between_shapes(ct_len, pt_len, lo, hi)

    ct_crop = _crop_by_axis_bounds(ct_samp, axis, lo, hi)
    seg_crop = _crop_by_axis_bounds(ct_seg_samp, axis, lo, hi)
    pt_crop = _crop_by_axis_bounds(pt_samp, axis, pt_lo, pt_hi)
    info = {
        'axis': axis,
        'si_dir': si_dir,
        'ct_bounds': (lo, hi),
        'pet_bounds': (pt_lo, pt_hi),
        'crop_len': crop_len,
        'atlas_len': int(atlas_len),
        'fov_ratio': fov_ratio,
        'is_small_fov': is_small_fov,
    }
    tag = 'MASK_ATLAS_FOV' if is_small_fov else 'OK_FOV'
    return ct_crop, pt_crop, seg_crop, True, f'{tag} axis={axis}, ct[{lo}:{hi}], pet[{pt_lo}:{pt_hi}], ratio={fov_ratio:.3f}', info


def _is_nonempty_volume(vol):
    return vol is not None and np.prod(vol.shape) > 0 and all(s > 0 for s in vol.shape)


def _safe_percentile(vol, q, fallback=0.0):
    if not _is_nonempty_volume(vol):
        return float(fallback)
    return float(np.percentile(vol, q))


def _pad_or_crop_np(np_vol, target_shape, fill_value=0.0):
    t = torch.from_numpy(np_vol).float().unsqueeze(0).unsqueeze(0)
    out = AffineReg3D._pad_or_crop(t, target_shape, 'constant', float(fill_value))
    return out[0, 0].detach().cpu().numpy()


def _run_antspy_affine(ct_mov_np, fixed_np):
    try:
        import ants
    except Exception as e:
        raise RuntimeError(f'ANTsPy is not available: {e}')

    spacing = tuple(float(x) for x in TAR_PIXDIM)
    fixed_img = ants.from_numpy(fixed_np.astype(np.float32), spacing=spacing)
    ct_mov_img = ants.from_numpy(ct_mov_np.astype(np.float32), spacing=spacing)

    reg = ants.registration(
        fixed=fixed_img,
        moving=ct_mov_img,
        type_of_transform='Affine',
    )

    return reg


@contextlib.contextmanager
def _temporary_antspy_workspace(prefix='antspy_'):
    """Create an isolated temp workspace for ANTsPy and clean it automatically."""
    with tempfile.TemporaryDirectory(prefix=prefix) as tmpdir:
        env_keys = ('TMPDIR', 'TMP', 'TEMP')
        prev_env = {k: os.environ.get(k, None) for k in env_keys}
        try:
            os.environ['TMPDIR'] = tmpdir
            os.environ['TMP'] = tmpdir
            os.environ['TEMP'] = tmpdir
            yield tmpdir
        finally:
            for k, v in prev_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _extract_antspy_affine_matrix_voxel_from_reg(reg, spacing):
    import ants
    import pandas as pd

    sp = np.array(spacing, dtype=np.float64)  # assumed (sx, sy, sz)

    # NOTE: numpy volumes here are in native axis order (x, y, z), and MIR
    # SpatialTransformer flow channels follow the same axis order as tensor spatial dims.
    # Therefore do NOT swap axes here.
    def xyz_vox_to_xyz_phys(v_xyz):
        x, y, z = v_xyz
        return np.array([x * sp[0], y * sp[1], z * sp[2]], dtype=np.float64)

    def xyz_phys_to_xyz_vox(p_xyz):
        x, y, z = p_xyz
        return np.array([x / sp[0], y / sp[1], z / sp[2]], dtype=np.float64)

    # Fit fixed_voxel(xyz) -> moving_voxel(xyz) affine from mapped points.
    # Use more than 4 points for robustness.
    src_fixed_xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    pts_xyz = np.array([xyz_vox_to_xyz_phys(p) for p in src_fixed_xyz], dtype=np.float64)
    pts_df = pd.DataFrame({'x': pts_xyz[:, 0], 'y': pts_xyz[:, 1], 'z': pts_xyz[:, 2]})

    inv_list = reg.get('invtransforms', [])
    if len(inv_list) == 0:
        fwd_list = reg.get('fwdtransforms', [])
        if len(fwd_list) == 0:
            raise RuntimeError('No ANTs transforms found in registration result.')
        # Affine-only fallback: use fwd with inversion to map fixed->moving.
        mapped_df = ants.apply_transforms_to_points(
            dim=3,
            points=pts_df,
            transformlist=fwd_list,
            whichtoinvert=[True] * len(fwd_list),
        )
    else:
        mapped_df = ants.apply_transforms_to_points(
            dim=3,
            points=pts_df,
            transformlist=inv_list,
        )

    dst_moving_xyz = mapped_df[['x', 'y', 'z']].to_numpy(dtype=np.float64)
    dst_moving_xyz_vox = np.array([xyz_phys_to_xyz_vox(p) for p in dst_moving_xyz], dtype=np.float64)

    X = np.concatenate([src_fixed_xyz, np.ones((src_fixed_xyz.shape[0], 1), dtype=np.float64)], axis=1)
    B = np.linalg.lstsq(X, dst_moving_xyz_vox, rcond=None)[0]  # (4,3)

    M = np.eye(4, dtype=np.float64)
    M[:3, :4] = B.T
    return M.astype(np.float32)


def _apply_voxel_affine_with_st(
    affine_model,
    affine4x4_vox,
    ct_mov_full_np,
    ct_seg_mov_full_np,
    pt_mov_full_np,
    atlas_np,
):
    device = CT_ATLAS_TORCH.device
    dtype = CT_ATLAS_TORCH.dtype
    shape = tuple(CT_ATLAS_TORCH.shape[2:])

    ct_t = torch.from_numpy(ct_mov_full_np).float().unsqueeze(0).unsqueeze(0).to(device)
    seg_t = torch.from_numpy(ct_seg_mov_full_np).float().unsqueeze(0).unsqueeze(0).to(device)
    pt_t = torch.from_numpy(pt_mov_full_np).float().unsqueeze(0).unsqueeze(0).to(device)

    M = np.array(affine4x4_vox, dtype=np.float64)
    M_use = M

    # MIR _affine_matrix_to_flow applies transform around volume center:
    # x' = A(x-c) + c + t. Convert absolute voxel affine x' = A x + b
    # into MIR-centered form by setting t = b + (A - I)c.
    A_np = M_use[:3, :3]
    b_np = M_use[:3, 3]
    c_np = (np.array(shape, dtype=np.float64) - 1.0) / 2.0
    t_centered_np = b_np + (A_np - np.eye(3, dtype=np.float64)) @ c_np

    M_centered = np.eye(4, dtype=np.float64)
    M_centered[:3, :3] = A_np
    M_centered[:3, 3] = t_centered_np
    affine_t = torch.from_numpy(M_centered).to(device=device, dtype=dtype).unsqueeze(0)

    flow = affine_model._affine_matrix_to_flow(
        affine=affine_t,
        shape=shape,
        batch_size=1,
        device=device,
        dtype=dtype,
        invert=False,
    )
    
    ct_w_raw = spatial_trans_ct(ct_t, flow)
    ct_w = fill_outside_valid_region(ct_w_raw, flow, -1000.0)
    seg_w = spatial_trans_nn(seg_t, flow)
    pt_w = spatial_trans_ct(pt_t, flow)

    ct_np = ct_w.detach().cpu().numpy()[0, 0]

    return {
        'affine_matrix': affine_t,
        'ct': ct_np,
        'ct_raw': ct_w_raw.detach().cpu().numpy()[0, 0],
        'seg': seg_w.detach().cpu().numpy()[0, 0],
        'pt': pt_w.detach().cpu().numpy()[0, 0],
    }

def _run_totalsegmentation_case(pat_name, ct_path, seg_path, device='gpu:0'):
    cmd = [
        "TotalSegmentator",
        "--ml",
        "-i", ct_path,
        "-o", seg_path,
        "--device", device,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return pat_name, result.returncode, result.stdout, result.stderr


def totalsegmentation(max_workers=None, device='gpu:0'):
    if max_workers is None:
        max_workers = max(TOTALSEG_WORKERS, 1)

    cases_to_run = []
    n_skipped = 0
    for img_folder_path in sorted(glob.glob(BASE_DIR + 'img/*/')):
        ct_candidates = glob.glob(img_folder_path + '*_CT.nii.gz')
        if len(ct_candidates) == 0:
            print(f'No CT found in {img_folder_path}, skipping.')
            continue

        ct_path = ct_candidates[0]
        pat_name = os.path.basename(ct_path).replace('_CT.nii.gz', '')
        seg_path = SEG_DIR + f'{pat_name}_CTSeg.nii.gz'

        if os.path.exists(seg_path) and os.path.getsize(seg_path) > 0:
            print(f'Skipping {pat_name}: segmentation exists.')
            n_skipped += 1
            continue

        cases_to_run.append((pat_name, ct_path, seg_path))

    print(f'Total cases: {len(cases_to_run) + n_skipped} | Skip existing: {n_skipped} | To run: {len(cases_to_run)}')
    if len(cases_to_run) == 0:
        return

    n_success, n_failed = 0, 0
    max_workers = max(1, int(max_workers))
    print(f'Running TotalSegmentator with workers={max_workers}, device={device}')
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_run_totalsegmentation_case, pat_name, ct_path, seg_path, device): pat_name
            for pat_name, ct_path, seg_path in cases_to_run
        }
        for fut in as_completed(futures):
            pat_name, code, _stdout, stderr = fut.result()
            if code == 0:
                n_success += 1
                print(f'[OK] {pat_name}')
            else:
                n_failed += 1
                print(f'[FAIL] {pat_name} (exit={code})')
                if stderr:
                    print(stderr.strip().split('\n')[-1])

    print(f'TotalSegmentator done. Success: {n_success}, Failed: {n_failed}, Skipped: {n_skipped}')

def preprocess():
    excluded_cases = []
    for img_folder_path in glob.glob(BASE_DIR+'img/*/'):
        try:
            pat_name = glob.glob(img_folder_path+'*_CT.nii.gz')[0].split('/')[-1].split('_CT.nii.gz')[0]
        except IndexError:
            print(f"No CT file found in {img_folder_path}, skipping.")
            continue

        out_ct_path = OUTPUT_DIR + f'{pat_name}_CT.nii.gz'
        out_seg_path = OUTPUT_DIR + f'{pat_name}_CTSeg.nii.gz'
        out_pet_path = OUTPUT_DIR + f'{pat_name}_PET.nii.gz'
        out_aff_path = AFF_DIR + f'{pat_name}_affine.pkl'
        if (
            os.path.exists(out_ct_path)
            and os.path.exists(out_seg_path)
            and os.path.exists(out_pet_path)
            and os.path.exists(out_aff_path)
        ):
            print(f'Skipping {pat_name}: already preprocessed outputs exist.')
            continue

        print(f'Preprocessing patient: {pat_name}')
        try:
            pt_nib = nib.load(img_folder_path+f'{pat_name}_PET.nii.gz')
        except FileNotFoundError:
            print(f"PET file not found for patient {pat_name}")
            continue
        except ImageFileError as e:
            msg = f'Excluded {pat_name}: invalid PET file ({e})'
            print(msg)
            excluded_cases.append(msg)
            continue
        except Exception as e:
            msg = f'Excluded {pat_name}: failed loading PET ({e})'
            print(msg)
            excluded_cases.append(msg)
            continue
        
        try:
            ct_nib = nib.load(img_folder_path+f'{pat_name}_CT.nii.gz')
        except ImageFileError as e:
            msg = f'Excluded {pat_name}: invalid CT file ({e})'
            print(msg)
            excluded_cases.append(msg)
            continue
        except Exception as e:
            msg = f'Excluded {pat_name}: failed loading CT ({e})'
            print(msg)
            excluded_cases.append(msg)
            continue

        ct_npy = ct_nib.get_fdata()
        header_info = ct_nib.header
        ct_pixdim = header_info['pixdim'][1:4]

        seg_path = SEG_DIR+f'{pat_name}_CTSeg.nii.gz'
        if not os.path.exists(seg_path):
            msg = f'Missing CT segmentation for {pat_name}: {seg_path}'
            print(msg)
            excluded_cases.append(msg)
            continue
        try:
            ct_seg_nib = nib.load(seg_path)
        except ImageFileError as e:
            msg = f'Excluded {pat_name}: invalid CT segmentation file ({e})'
            print(msg)
            excluded_cases.append(msg)
            continue
        except Exception as e:
            msg = f'Excluded {pat_name}: failed loading CT segmentation ({e})'
            print(msg)
            excluded_cases.append(msg)
            continue

        ct_seg_npy = ct_seg_nib.get_fdata()
        ct_seg_npy_samp = zoom_img(ct_seg_npy, (ct_pixdim, TAR_PIXDIM), 0)

        header_info = pt_nib.header
        pt_npy = pt_nib.get_fdata()
        pt_pixdim = header_info['pixdim'][1:4]
        pt_npy_samp = zoom_img(pt_npy, (pt_pixdim, TAR_PIXDIM), 3)
        
        ct_npy_samp = zoom_img(ct_npy, (ct_pixdim, TAR_PIXDIM), 3)

        if not _is_nonempty_volume(ct_npy_samp) or not _is_nonempty_volume(pt_npy_samp) or not _is_nonempty_volume(ct_seg_npy_samp):
            msg = (
                f'Excluded {pat_name}: empty resampled volume '
                f'(ct={ct_npy_samp.shape}, pet={pt_npy_samp.shape}, seg={ct_seg_npy_samp.shape})'
            )
            print(msg)
            excluded_cases.append(msg)
            continue

        ct_npy_samp, pt_npy_samp, ct_seg_npy_samp, keep_case, qc_msg, qc_info = quality_control_and_crop(
            ct_npy_samp,
            pt_npy_samp,
            ct_seg_npy_samp,
            CT_ATLAS_NPY.shape,
        )
        if not keep_case:
            msg = f'Excluded {pat_name}: {qc_msg}'
            print(msg)
            excluded_cases.append(msg)
            continue

        if not _is_nonempty_volume(ct_npy_samp) or not _is_nonempty_volume(pt_npy_samp) or not _is_nonempty_volume(ct_seg_npy_samp):
            msg = (
                f'Excluded {pat_name}: empty cropped volume after QC '
                f'(ct={ct_npy_samp.shape}, pet={pt_npy_samp.shape}, seg={ct_seg_npy_samp.shape})'
            )
            print(msg)
            excluded_cases.append(msg)
            continue
        print(f'FOV QC {pat_name}: {qc_msg}')

        ct_npy_samp_bd = remove_bed(ct_npy_samp)
        ct_fill = _safe_percentile(ct_npy_samp_bd, 0.5, fallback=-1000.0)
        pt_fill = _safe_percentile(pt_npy_samp, 0.5, fallback=0.0)

        ct_norm_np = norm_ct(ct_npy_samp_bd.copy())
        ct_norm_torch = torch.from_numpy(ct_norm_np).float().unsqueeze(0).unsqueeze(0).cuda(0)

        affine_model = AffineReg3D(vol_shape=CT_ATLAS_TORCH.shape[2:], dof = "affine", scales = (0.25, 0.5, 1), loss_funcs = ("mse", "mse", "mse"), ).cuda(0)

        atlas_for_opt_torch = CT_ATLAS_TORCH
        if qc_info is not None and qc_info.get('is_small_fov', False):
            # Keep small-FOV subjects by masking atlas support to subject support
            # instead of discarding the case.
            valid_moving = torch.ones_like(ct_norm_torch)
            atlas_support_mask = affine_model._pad_or_crop(
                valid_moving,
                CT_ATLAS_TORCH.shape[2:],
                'constant',
                0.0,
            )
            atlas_for_opt_torch = CT_ATLAS_TORCH * atlas_support_mask
            print(
                f"Using atlas support masking for {pat_name}: "
                f"fov_ratio={qc_info.get('fov_ratio', 0):.3f}"
            )

        atlas_for_opt_np = atlas_for_opt_torch.detach().cpu().numpy()[0, 0]

        # Prepare padded/cropped arrays for ANTs path and fair metric comparison.
        ct_mov_full_np = _pad_or_crop_np(ct_npy_samp_bd, CT_ATLAS_TORCH.shape[2:], fill_value=ct_fill)
        ct_seg_mov_full_np = _pad_or_crop_np(ct_seg_npy_samp, CT_ATLAS_TORCH.shape[2:], fill_value=0.0)
        pt_mov_full_np = _pad_or_crop_np(pt_npy_samp, CT_ATLAS_TORCH.shape[2:], fill_value=pt_fill)

        # ANTsPy affine is the only backend, then convert affine matrix to flow
        # to keep warping with MIR SpatialTransformer.
        # Use a per-case temp workspace so ANTs files are removed immediately.
        try:
            with _temporary_antspy_workspace(prefix=f'antspy_{pat_name}_') as ants_tmpdir:
                try:
                    import ants
                except Exception as e:
                    raise RuntimeError(f'ANTsPy is not available: {e}')

                spacing = tuple(float(x) for x in TAR_PIXDIM)
                fixed_img = ants.from_numpy(atlas_for_opt_np.astype(np.float32), spacing=spacing)
                ct_mov_img = ants.from_numpy(ct_mov_full_np.astype(np.float32), spacing=spacing)
                outprefix = os.path.join(ants_tmpdir, f'{pat_name}_')

                ants_reg = ants.registration(
                    fixed=fixed_img,
                    moving=ct_mov_img,
                    type_of_transform='Affine',
                    outprefix=outprefix,
                )

                fwd_list = ants_reg.get('fwdtransforms', [])
                if len(fwd_list) == 0:
                    raise RuntimeError('No ANTs forward transform returned.')

                affine_vox_4x4 = _extract_antspy_affine_matrix_voxel_from_reg(ants_reg, TAR_PIXDIM)

            utils.savepkl(affine_vox_4x4, AFF_DIR+f'{pat_name}_affine_ants_voxel.pkl')
            #np.savetxt(AFF_DIR+f'{pat_name}_affine_ants_voxel.txt', affine_vox_4x4, fmt='%.8f')

            outputs = _apply_voxel_affine_with_st(
                affine_model=affine_model,
                affine4x4_vox=affine_vox_4x4,
                ct_mov_full_np=ct_mov_full_np,
                ct_seg_mov_full_np=ct_seg_mov_full_np,
                pt_mov_full_np=pt_mov_full_np,
                atlas_np=atlas_for_opt_np,
            )
            
            ct_aff_npy = outputs['ct']
            ct_seg_aff_npy = outputs['seg']
            pt_aff_npy = outputs['pt']
            affine_mat = outputs['affine_matrix'].detach().cpu().numpy()[0]
            utils.savepkl(affine_mat, AFF_DIR+f'{pat_name}_affine.pkl')
        except Exception as e:
            print(f'ANTsPy affine failed for {pat_name}: {e}')
            ct_aff_npy = None
            ct_seg_aff_npy = None
            pt_aff_npy = None

        if ct_aff_npy is None or ct_seg_aff_npy is None or pt_aff_npy is None:
            msg = f'Affine registration failed for {pat_name} with backend=ants'
            print(msg)
            excluded_cases.append(msg)
            continue

        ct_aff_nib = nib.Nifti1Image(ct_aff_npy, make_affine_from_pixdim(TAR_PIXDIM))
        nib.save(ct_aff_nib, OUTPUT_DIR+f'{pat_name}_CT.nii.gz')

        ct_seg_aff_nib = nib.Nifti1Image(ct_seg_aff_npy, make_affine_from_pixdim(TAR_PIXDIM))
        nib.save(ct_seg_aff_nib, OUTPUT_DIR+f'{pat_name}_CTSeg.nii.gz')

        pt_aff_nib = nib.Nifti1Image(pt_aff_npy, make_affine_from_pixdim(TAR_PIXDIM))
        nib.save(pt_aff_nib, OUTPUT_DIR+f'{pat_name}_PET.nii.gz')
        
        
        # Safe middle-slice indices per volume (avoid OOB when cropped input is smaller than atlas).
        y_in = ct_norm_torch.shape[3] // 2
        y_aff = ct_aff_npy.shape[1] // 2
        y_atlas = atlas_for_opt_np.shape[1] // 2

        plt.figure(figsize=(20, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(ct_norm_torch[0, 0, :, y_in, :].detach().cpu().numpy(), cmap='gray')
        plt.title('Input')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(ct_aff_npy[:, y_aff, :], cmap='gray')
        plt.title('ST from ANTs affine')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(atlas_for_opt_np[:, y_atlas, :], cmap='gray')
        plt.title('Atlas')
        plt.axis('off')
        plt.savefig(f'affine.png')
        plt.close()
        
        
        

    if len(excluded_cases) > 0:
        with open(EXCLUDE_LOG_PATH, 'w') as f:
            f.write('\n'.join(excluded_cases) + '\n')
        print(f'Excluded cases logged to: {EXCLUDE_LOG_PATH} (n={len(excluded_cases)})')
        
if __name__ == "__main__":
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
    torch.manual_seed(0)
    
    #totalsegmentation()
    preprocess()