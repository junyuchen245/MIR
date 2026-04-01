import contextlib
import glob
import os
import subprocess
import tempfile
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from nibabel.filebasedimages import ImageFileError
from scipy import ndimage
from scipy.ndimage import zoom

import MIR.utils.other_utils as utils
from MIR.models import AffineReg3D, SpatialTransformer


def zoom_img(img, pixel_dims, order=3):
	img_pixdim, tar_pix = pixel_dims
	ratio = np.array(img_pixdim) / np.array(tar_pix)
	return zoom(img, ratio, order=order)


def make_affine_from_pixdim(pixdim):
	affine = np.eye(4)
	affine[0, 0] = pixdim[0]
	affine[1, 1] = pixdim[1]
	affine[2, 2] = pixdim[2]
	return affine


def remove_bed(img):
	mask = img >= -800
	mask = ndimage.binary_erosion(mask)
	mask = ndimage.binary_fill_holes(mask)
	mask = ndimage.binary_dilation(mask)
	labels, n = ndimage.label(mask)
	if n > 0:
		sizes = ndimage.sum(mask, labels, range(1, n + 1))
		largest = int(np.argmax(sizes) + 1)
		mask = labels == largest
		mask = ndimage.binary_dilation(mask)
		mask = ndimage.binary_closing(mask)
	out = img.copy()
	out[~mask] = np.percentile(out, 0.5)
	return out


def norm_ct(img):
	x = img.copy()
	x[x < -300] = -300
	x[x > 300] = 300
	return (x - x.min()) / (x.max() - x.min() + 1e-8)


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


def _infer_si_axis_and_direction_from_atlas_seg(seg_v1):
	brain_like = np.isin(seg_v1, [50, 93])
	inferior_like = np.isin(seg_v1, [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])
	if brain_like.sum() == 0 or inferior_like.sum() == 0:
		return 2, 1

	best_axis = 2
	best_sep = -1.0
	best_dir = 1
	for axis in range(3):
		brain_idx = np.where(brain_like)[axis]
		inf_idx = np.where(inferior_like)[axis]
		brain_c = float(np.median(brain_idx))
		inf_c = float(np.median(inf_idx))
		sep = abs(brain_c - inf_c)
		if sep > best_sep:
			best_sep = sep
			best_axis = axis
			best_dir = 1 if brain_c > inf_c else -1
	return best_axis, best_dir


def _build_fixed_upper_torso_head_crop(shape, si_axis, si_dir, target_shape):
	bbox = []
	for axis in range(3):
		if axis == si_axis:
			crop_len = target_shape[axis]
			if si_dir == 1:
				lo = shape[axis] - crop_len
				hi = shape[axis] - 1
			else:
				lo = 0
				hi = crop_len - 1
		else:
			crop_len = target_shape[axis]
			lo = max(0, (shape[axis] - crop_len) // 2)
			hi = lo + crop_len - 1
		bbox.append((int(lo), int(hi)))
	return tuple(bbox)


def _crop_np(vol, bbox):
	return vol[
		bbox[0][0]:bbox[0][1] + 1,
		bbox[1][0]:bbox[1][1] + 1,
		bbox[2][0]:bbox[2][1] + 1,
	]


def _fit_volume_to_target(np_vol, target_shape, fill_value, si_axis, si_dir, return_support_mask=False):
	out = np.full(target_shape, fill_value, dtype=np_vol.dtype)
	support = np.zeros(target_shape, dtype=np.uint8) if return_support_mask else None

	src_slices = []
	dst_slices = []
	for axis in range(3):
		src_len = np_vol.shape[axis]
		dst_len = target_shape[axis]
		if src_len >= dst_len:
			if axis == si_axis:
				if si_dir == 1:
					src_lo = src_len - dst_len
				else:
					src_lo = 0
			else:
				src_lo = (src_len - dst_len) // 2
			src_hi = src_lo + dst_len
			dst_lo, dst_hi = 0, dst_len
		else:
			src_lo, src_hi = 0, src_len
			if axis == si_axis:
				if si_dir == 1:
					dst_lo = dst_len - src_len
				else:
					dst_lo = 0
			else:
				dst_lo = (dst_len - src_len) // 2
			dst_hi = dst_lo + src_len

		src_slices.append(slice(src_lo, src_hi))
		dst_slices.append(slice(dst_lo, dst_hi))

	out[tuple(dst_slices)] = np_vol[tuple(src_slices)]
	if return_support_mask:
		support[tuple(dst_slices)] = 1
		return out, support
	return out


def _build_slice_only_atlas_mask(patient_support_mask, si_axis):
	slice_seen = np.any(patient_support_mask > 0, axis=tuple(ax for ax in range(3) if ax != si_axis))
	if not np.any(slice_seen):
		return np.ones(patient_support_mask.shape, dtype=np.float32)

	idx = np.where(slice_seen)[0]
	lo = int(idx.min())
	hi = int(idx.max())
	slice_mask = np.zeros_like(slice_seen, dtype=np.float32)
	slice_mask[lo:hi + 1] = 1.0

	reshape = [1, 1, 1]
	reshape[si_axis] = slice_mask.shape[0]
	return np.broadcast_to(slice_mask.reshape(reshape), patient_support_mask.shape).astype(np.float32)


def _make_preview(input_ct_np, aff_ct_np, atlas_np, preview_path, case_id):
	def _mid_slice(vol):
		return vol[:, vol.shape[1] // 2, :]

	plt.figure(figsize=(18, 6))
	plt.subplot(1, 3, 1)
	plt.imshow(_mid_slice(input_ct_np), cmap='gray')
	plt.title(f'{case_id} input')
	plt.axis('off')

	plt.subplot(1, 3, 2)
	plt.imshow(_mid_slice(aff_ct_np), cmap='gray')
	plt.title('Affine CT')
	plt.axis('off')

	plt.subplot(1, 3, 3)
	plt.imshow(_mid_slice(atlas_np), cmap='gray')
	plt.title('Cropped atlas')
	plt.axis('off')

	plt.tight_layout()
	plt.savefig(preview_path, dpi=120)
	plt.close()


BASE_DIR = '/scratch2/jchen/DATA/HECKTOR25/'
IMG_DIR = BASE_DIR + 'img/'
OUTPUT_DIR = BASE_DIR + 'preprocessed/'
AFF_DIR = OUTPUT_DIR + 'affine/'
QC_DIR = OUTPUT_DIR + 'qc/'
SEG_DIR = OUTPUT_DIR + 'seg/ct/'
SAVE_QC_DEFAULT = False
PREVIEW_PATH = OUTPUT_DIR + 'preview_latest.png'
RTDOSE_INTERP_ORDER = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AFF_DIR, exist_ok=True)
os.makedirs(SEG_DIR, exist_ok=True)

EXCLUDE_LOG_PATH = OUTPUT_DIR + 'excluded_cases.txt'
TAR_PIXDIM = [2.8, 2.8, 3.8]
TOTALSEG_WORKERS = 3

# Reuse whole-body atlas, but crop atlas support per H&N subject FOV.
CT_ATLAS = '/scratch/jchen/python_projects/AutoPET/atlas/ct/TransMorphAtlas_MAE_1_MS_1_diffusion_1/dsc0.5425.nii.gz'
ATLAS_SEG_V1 = '/scratch/jchen/python_projects/AutoPET/seg_atlas_from_reg_rib_pp.nii.gz'
CT_ATLAS_NIB = nib.load(CT_ATLAS)
CT_ATLAS_NPY = CT_ATLAS_NIB.get_fdata()
CT_ATLAS_NPY = np.flip(CT_ATLAS_NPY, 1).copy()
CT_ATLAS_NPY = np.clip(CT_ATLAS_NPY.astype(np.float32), 0.0, 1.0)
ATLAS_SEG_V1_NIB = nib.load(ATLAS_SEG_V1)
ATLAS_SEG_V1_NPY = ATLAS_SEG_V1_NIB.get_fdata()
ATLAS_SEG_V1_NPY = np.flip(ATLAS_SEG_V1_NPY, 1).copy()
ATLAS_TARGET_SHAPE = (192, 192, 144)
ATLAS_SI_AXIS, ATLAS_SI_DIR = _infer_si_axis_and_direction_from_atlas_seg(ATLAS_SEG_V1_NPY)
ATLAS_CROP_BBOX = _build_fixed_upper_torso_head_crop(CT_ATLAS_NPY.shape, ATLAS_SI_AXIS, ATLAS_SI_DIR, ATLAS_TARGET_SHAPE)
CT_ATLAS_FIXED_NPY = _crop_np(CT_ATLAS_NPY, ATLAS_CROP_BBOX).astype(np.float32)
ATLAS_SEG_FIXED_NPY = _crop_np(ATLAS_SEG_V1_NPY, ATLAS_CROP_BBOX)


def _to_device_torch(x_np):
	return torch.from_numpy(x_np).float().unsqueeze(0).unsqueeze(0).cuda(0)


CT_ATLAS_TORCH = _to_device_torch(CT_ATLAS_FIXED_NPY)
spatial_trans_ct = SpatialTransformer(size=CT_ATLAS_TORCH.shape[2:], mode='bilinear').cuda(0)
spatial_trans_nn = SpatialTransformer(size=CT_ATLAS_TORCH.shape[2:], mode='nearest').cuda(0)


def fill_outside_valid_region(warped_img, flow, pad_value=-1000, eps=1e-4, valid_thr=0.995):
	ones = torch.ones_like(warped_img)
	valid_w = spatial_trans_ct(ones, flow)
	corrected = warped_img / torch.clamp(valid_w, min=eps)
	valid_mask = valid_w > valid_thr
	corrected = torch.clamp(corrected, min=-1024.0, max=3071.0)
	return torch.where(valid_mask, corrected, torch.full_like(warped_img, pad_value))


def _make_affine_model():
	return AffineReg3D(
		vol_shape=CT_ATLAS_TORCH.shape[2:],
		dof='affine',
		scales=(0.25, 0.5, 1),
		loss_funcs=('mse', 'mse', 'mse'),
	).cuda(0)


def _center_voxel_affine_matrix(affine4x4_vox, shape):
	M = np.array(affine4x4_vox, dtype=np.float64)
	A_np = M[:3, :3]
	b_np = M[:3, 3]
	c_np = (np.array(shape, dtype=np.float64) - 1.0) / 2.0
	t_centered_np = b_np + (A_np - np.eye(3, dtype=np.float64)) @ c_np

	M_centered = np.eye(4, dtype=np.float64)
	M_centered[:3, :3] = A_np
	M_centered[:3, 3] = t_centered_np
	return M_centered.astype(np.float32)


def _compute_affine_flow(affine_model, centered_affine4x4, shape):
	device = CT_ATLAS_TORCH.device
	dtype = CT_ATLAS_TORCH.dtype
	affine_t = torch.from_numpy(np.array(centered_affine4x4, dtype=np.float32)).to(device=device, dtype=dtype).unsqueeze(0)
	flow = affine_model._affine_matrix_to_flow(
		affine=affine_t,
		shape=shape,
		batch_size=1,
		device=device,
		dtype=dtype,
		invert=False,
	)
	return affine_t, flow


def _warp_volume_with_st(
	affine_model,
	centered_affine4x4,
	volume_np,
	mode='bilinear',
	pad_value=0.0,
	correct_valid_region=False,
):
	shape = tuple(CT_ATLAS_TORCH.shape[2:])
	volume_t = _to_device_torch(volume_np)
	_, flow = _compute_affine_flow(affine_model, centered_affine4x4, shape)
	transformer = spatial_trans_nn if mode == 'nearest' else spatial_trans_ct
	warped_raw = transformer(volume_t, flow)
	if correct_valid_region:
		warped = fill_outside_valid_region(warped_raw, flow, pad_value=pad_value)
	else:
		warped = warped_raw
	return warped.detach().cpu().numpy()[0, 0]


def _load_saved_centered_affine(case_id):
	centered_aff_path = os.path.join(AFF_DIR, f'{case_id}_affine.pkl')
	if os.path.exists(centered_aff_path):
		return np.array(utils.pkload(centered_aff_path), dtype=np.float32)

	ants_voxel_aff_path = os.path.join(AFF_DIR, f'{case_id}_affine_ants_voxel.pkl')
	if os.path.exists(ants_voxel_aff_path):
		affine_vox = utils.pkload(ants_voxel_aff_path)
		return _center_voxel_affine_matrix(affine_vox, ATLAS_TARGET_SHAPE)

	raise FileNotFoundError(f'No saved affine found for {case_id} in {AFF_DIR}')


@contextlib.contextmanager
def _temporary_antspy_workspace(prefix='antspy_'):
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

	sp = np.array(spacing, dtype=np.float64)

	def xyz_vox_to_xyz_phys(v_xyz):
		x, y, z = v_xyz
		return np.array([x * sp[0], y * sp[1], z * sp[2]], dtype=np.float64)

	def xyz_phys_to_xyz_vox(p_xyz):
		x, y, z = p_xyz
		return np.array([x / sp[0], y / sp[1], z / sp[2]], dtype=np.float64)

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
	B = np.linalg.lstsq(X, dst_moving_xyz_vox, rcond=None)[0]

	M = np.eye(4, dtype=np.float64)
	M[:3, :4] = B.T
	return M.astype(np.float32)


def _apply_voxel_affine_with_st(
	affine_model,
	affine4x4_vox,
	ct_mov_full_np,
	pt_mov_full_np,
	seg_mov_full_np=None,
):
	shape = tuple(CT_ATLAS_TORCH.shape[2:])
	centered_affine = _center_voxel_affine_matrix(affine4x4_vox, shape)
	affine_t, _ = _compute_affine_flow(affine_model, centered_affine, shape)
	ct_w = _warp_volume_with_st(
		affine_model,
		centered_affine,
		ct_mov_full_np,
		mode='bilinear',
		pad_value=-1000.0,
		correct_valid_region=True,
	)
	pt_w = _warp_volume_with_st(
		affine_model,
		centered_affine,
		pt_mov_full_np,
		mode='bilinear',
	)
	seg_w = None
	if seg_mov_full_np is not None:
		seg_w = _warp_volume_with_st(
			affine_model,
			centered_affine,
			seg_mov_full_np,
			mode='nearest',
		)

	outputs = {
		'affine_matrix': affine_t,
		'ct': ct_w,
		'pt': pt_w,
	}
	if seg_w is not None:
		outputs['seg'] = seg_w
	return outputs


def _process_rtdose_case(case_id, affine_model=None, centered_affine=None, overwrite=False):
	case_dir = os.path.join(IMG_DIR, case_id)
	rtdose_path = os.path.join(case_dir, f'{case_id}__RTDOSE.nii.gz')
	out_rtdose_path = os.path.join(OUTPUT_DIR, f'{case_id}_RTDOSE.nii.gz')

	if not os.path.exists(rtdose_path):
		return None
	if os.path.exists(out_rtdose_path) and not overwrite:
		print(f'Skipping {case_id}: RTDose output already exists.')
		return None

	try:
		rtdose_nib = nib.load(rtdose_path)
	except ImageFileError as e:
		return f'Excluded {case_id}: invalid RTDose file ({e})'
	except Exception as e:
		return f'Excluded {case_id}: failed loading RTDose ({e})'

	rtdose_npy = rtdose_nib.get_fdata()
	rtdose_pixdim = rtdose_nib.header['pixdim'][1:4]
	rtdose_samp = zoom_img(rtdose_npy, (rtdose_pixdim, TAR_PIXDIM), RTDOSE_INTERP_ORDER).astype(np.float32)
	if not _is_nonempty_volume(rtdose_samp):
		return f'Excluded {case_id}: empty resampled RTDose ({rtdose_samp.shape})'

	rtdose_mov_full_np = _fit_volume_to_target(
		rtdose_samp,
		ATLAS_TARGET_SHAPE,
		fill_value=0.0,
		si_axis=ATLAS_SI_AXIS,
		si_dir=ATLAS_SI_DIR,
	)

	if centered_affine is None:
		centered_affine = _load_saved_centered_affine(case_id)
	else:
		centered_affine = np.array(centered_affine, dtype=np.float32)

	local_affine_model = affine_model if affine_model is not None else _make_affine_model()
	rtdose_aff_npy = _warp_volume_with_st(
		local_affine_model,
		centered_affine,
		rtdose_mov_full_np,
		mode='bilinear',
		pad_value=0.0,
		correct_valid_region=False,
	).astype(np.float32)

	nib.save(nib.Nifti1Image(rtdose_aff_npy, make_affine_from_pixdim(TAR_PIXDIM)), out_rtdose_path)
	print(f'Saved RTDose for {case_id}: {out_rtdose_path}')
	return None


def preprocess_rtdose_only(overwrite=False):
	excluded_cases = []
	case_ids = _find_case_ids()
	print(f'Total candidate folders for RTDose-only pass: {len(case_ids)}')
	affine_model = _make_affine_model()

	for case_id in case_ids:
		try:
			msg = _process_rtdose_case(case_id, affine_model=affine_model, overwrite=overwrite)
		except Exception as e:
			msg = f'Excluded {case_id}: RTDose preprocessing failed ({e})'
		if msg:
			print(msg)
			excluded_cases.append(msg)

	if len(excluded_cases) > 0:
		with open(EXCLUDE_LOG_PATH, 'w') as f:
			f.write('\n'.join(excluded_cases) + '\n')
		print(f'Excluded RTDose cases logged to: {EXCLUDE_LOG_PATH} (n={len(excluded_cases)})')


def _find_case_ids():
	case_ids = []
	for d in sorted(glob.glob(IMG_DIR + '*/')):
		case_id = os.path.basename(os.path.normpath(d))
		if case_id == 'preprocessed':
			continue
		case_ids.append(case_id)
	return case_ids


def _run_totalsegmentation_case(case_id, ct_path, seg_path, device='gpu:0'):
	cmd = [
		'TotalSegmentator',
		'--ml',
		'-i', ct_path,
		'-o', seg_path,
		'--device', device,
	]
	result = subprocess.run(cmd, capture_output=True, text=True)
	return case_id, result.returncode, result.stdout, result.stderr


def totalsegmentation(max_workers=None, device='gpu:0'):
	if max_workers is None:
		max_workers = max(TOTALSEG_WORKERS, 1)

	cases_to_run = []
	n_skipped = 0
	for case_id in _find_case_ids():
		case_dir = os.path.join(IMG_DIR, case_id)
		ct_path = os.path.join(case_dir, f'{case_id}__CT.nii.gz')
		seg_path = os.path.join(SEG_DIR, f'{case_id}_CTSeg.nii.gz')

		if not os.path.exists(ct_path):
			print(f'Skipping {case_id}: missing CT for TotalSegmentator.')
			continue

		if os.path.exists(seg_path) and os.path.getsize(seg_path) > 0:
			print(f'Skipping {case_id}: segmentation exists.')
			n_skipped += 1
			continue

		cases_to_run.append((case_id, ct_path, seg_path))

	print(
		f'Total cases: {len(cases_to_run) + n_skipped} '
		f'| Skip existing: {n_skipped} | To run: {len(cases_to_run)}'
	)
	if len(cases_to_run) == 0:
		return

	n_success, n_failed = 0, 0
	max_workers = max(1, int(max_workers))
	print(f'Running TotalSegmentator with workers={max_workers}, device={device}')
	with ThreadPoolExecutor(max_workers=max_workers) as ex:
		futures = {
			ex.submit(_run_totalsegmentation_case, case_id, ct_path, seg_path, device): case_id
			for case_id, ct_path, seg_path in cases_to_run
		}
		for fut in as_completed(futures):
			case_id, code, _stdout, stderr = fut.result()
			if code == 0:
				n_success += 1
				print(f'[OK] {case_id}')
			else:
				n_failed += 1
				print(f'[FAIL] {case_id} (exit={code})')
				if stderr:
					print(stderr.strip().split('\n')[-1])

	print(f'TotalSegmentator done. Success: {n_success}, Failed: {n_failed}, Skipped: {n_skipped}')


def preprocess(save_qc=SAVE_QC_DEFAULT):
	excluded_cases = []
	case_ids = _find_case_ids()
	print(f'Total candidate folders: {len(case_ids)}')

	for case_id in case_ids:
		case_dir = os.path.join(IMG_DIR, case_id)
		ct_path = os.path.join(case_dir, f'{case_id}__CT.nii.gz')
		pt_path = os.path.join(case_dir, f'{case_id}__PT.nii.gz')
		rtdose_path = os.path.join(case_dir, f'{case_id}__RTDOSE.nii.gz')
		seg_path = os.path.join(case_dir, f'{case_id}.nii.gz')
		ct_seg_path = os.path.join(SEG_DIR, f'{case_id}_CTSeg.nii.gz')

		out_ct_path = os.path.join(OUTPUT_DIR, f'{case_id}_CT.nii.gz')
		out_pt_path = os.path.join(OUTPUT_DIR, f'{case_id}_PET.nii.gz')
		out_rtdose_path = os.path.join(OUTPUT_DIR, f'{case_id}_RTDOSE.nii.gz')
		out_seg_path = os.path.join(OUTPUT_DIR, f'{case_id}_SEG.nii.gz')
		out_ctseg_path = os.path.join(OUTPUT_DIR, f'{case_id}_CTSeg.nii.gz')
		out_aff_path = os.path.join(AFF_DIR, f'{case_id}_affine.pkl')

		if os.path.exists(out_ct_path) and os.path.exists(out_pt_path) and os.path.exists(out_aff_path):
			seg_done = (not os.path.exists(seg_path)) or os.path.exists(out_seg_path)
			ctseg_done = (not os.path.exists(ct_seg_path)) or os.path.exists(out_ctseg_path)
			rtdose_done = (not os.path.exists(rtdose_path)) or os.path.exists(out_rtdose_path)
			if seg_done and ctseg_done and rtdose_done:
				print(f'Skipping {case_id}: outputs already exist.')
				continue
			if seg_done and ctseg_done and (not rtdose_done):
				print(f'Reusing saved affine for RTDose only: {case_id}')
				try:
					msg = _process_rtdose_case(case_id, overwrite=False)
				except Exception as e:
					msg = f'Excluded {case_id}: RTDose-only reuse failed ({e})'
				if msg:
					print(msg)
					excluded_cases.append(msg)
				continue

		if not os.path.exists(ct_path) or not os.path.exists(pt_path):
			msg = f'Excluded {case_id}: missing CT/PT ({ct_path}, {pt_path})'
			print(msg)
			excluded_cases.append(msg)
			continue

		print(f'Preprocessing case: {case_id}')
		try:
			ct_nib = nib.load(ct_path)
			pt_nib = nib.load(pt_path)
		except ImageFileError as e:
			msg = f'Excluded {case_id}: invalid input file ({e})'
			print(msg)
			excluded_cases.append(msg)
			continue
		except Exception as e:
			msg = f'Excluded {case_id}: failed loading CT/PT ({e})'
			print(msg)
			excluded_cases.append(msg)
			continue

		seg_nib = None
		ct_seg_nib = None
		if os.path.exists(seg_path):
			try:
				seg_nib = nib.load(seg_path)
			except Exception:
				print(f'Warning: cannot read segmentation for {case_id}, continuing without SEG.')

		if os.path.exists(ct_seg_path):
			try:
				ct_seg_nib = nib.load(ct_seg_path)
			except Exception:
				print(f'Warning: cannot read TotalSegmentator CT seg for {case_id}, continuing without CTSeg.')

		ct_npy = ct_nib.get_fdata()
		pt_npy = pt_nib.get_fdata()
		ct_pixdim = ct_nib.header['pixdim'][1:4]
		pt_pixdim = pt_nib.header['pixdim'][1:4]

		ct_samp = zoom_img(ct_npy, (ct_pixdim, TAR_PIXDIM), 3)
		pt_samp = zoom_img(pt_npy, (pt_pixdim, TAR_PIXDIM), 3)
		seg_samp = zoom_img(seg_nib.get_fdata(), (ct_pixdim, TAR_PIXDIM), 0) if seg_nib is not None else None
		ct_seg_samp = zoom_img(ct_seg_nib.get_fdata(), (ct_pixdim, TAR_PIXDIM), 0) if ct_seg_nib is not None else None

		if not _is_nonempty_volume(ct_samp) or not _is_nonempty_volume(pt_samp):
			msg = f'Excluded {case_id}: empty resampled CT/PT ({ct_samp.shape}, {pt_samp.shape})'
			print(msg)
			excluded_cases.append(msg)
			continue

		ct_samp_bd = remove_bed(ct_samp)
		ct_samp_norm = norm_ct(ct_samp_bd.copy()).astype(np.float32)
		ct_fill = _safe_percentile(ct_samp_bd, 0.5, fallback=-1000.0)
		pt_fill = _safe_percentile(pt_samp, 0.5, fallback=0.0)

		ct_mov_full_np, patient_support_mask = _fit_volume_to_target(
			ct_samp_bd,
			ATLAS_TARGET_SHAPE,
			fill_value=ct_fill,
			si_axis=ATLAS_SI_AXIS,
			si_dir=ATLAS_SI_DIR,
			return_support_mask=True,
		)
		ct_mov_full_norm_np = _fit_volume_to_target(
			ct_samp_norm,
			ATLAS_TARGET_SHAPE,
			fill_value=0.0,
			si_axis=ATLAS_SI_AXIS,
			si_dir=ATLAS_SI_DIR,
		)
		pt_mov_full_np = _fit_volume_to_target(pt_samp, ATLAS_TARGET_SHAPE, pt_fill, ATLAS_SI_AXIS, ATLAS_SI_DIR)
		seg_mov_full_np = _fit_volume_to_target(seg_samp, ATLAS_TARGET_SHAPE, 0.0, ATLAS_SI_AXIS, ATLAS_SI_DIR) if seg_samp is not None else None
		ct_seg_mov_full_np = _fit_volume_to_target(ct_seg_samp, ATLAS_TARGET_SHAPE, 0.0, ATLAS_SI_AXIS, ATLAS_SI_DIR) if ct_seg_samp is not None else None

		atlas_slice_mask_np = _build_slice_only_atlas_mask(patient_support_mask, ATLAS_SI_AXIS)
		atlas_for_opt_np = CT_ATLAS_FIXED_NPY * atlas_slice_mask_np

		if np.any(atlas_slice_mask_np == 0):
			print(f'Atlas slice masking applied for {case_id}: missing patient CT coverage in part of fixed atlas z-range.')

		affine_model = _make_affine_model()

		try:
			with _temporary_antspy_workspace(prefix=f'antspy_{case_id}_') as ants_tmpdir:
				try:
					import ants
				except Exception as e:
					raise RuntimeError(f'ANTsPy is not available: {e}')

				spacing = tuple(float(x) for x in TAR_PIXDIM)
				fixed_img = ants.from_numpy(atlas_for_opt_np.astype(np.float32), spacing=spacing)
				moving_img = ants.from_numpy(ct_mov_full_norm_np.astype(np.float32), spacing=spacing)

				ants_reg = ants.registration(
					fixed=fixed_img,
					moving=moving_img,
					type_of_transform='Affine',
					outprefix=os.path.join(ants_tmpdir, f'{case_id}_'),
					aff_metric='mattes',
					aff_iterations=(600, 600, 250), aff_shrink_factors=(4, 2, 1), aff_smoothing_sigmas=(2, 1, 0)
					)

				affine_vox_4x4 = _extract_antspy_affine_matrix_voxel_from_reg(ants_reg, TAR_PIXDIM)

			utils.savepkl(affine_vox_4x4, os.path.join(AFF_DIR, f'{case_id}_affine_ants_voxel.pkl'))

			outputs = _apply_voxel_affine_with_st(
				affine_model=affine_model,
				affine4x4_vox=affine_vox_4x4,
				ct_mov_full_np=ct_mov_full_np,
				pt_mov_full_np=pt_mov_full_np,
				seg_mov_full_np=seg_mov_full_np,
			)

			ct_aff_npy = outputs['ct']
			pt_aff_npy = outputs['pt']
			seg_aff_npy = outputs.get('seg', None)

			ctseg_aff_npy = None
			if ct_seg_mov_full_np is not None:
				ctseg_outputs = _apply_voxel_affine_with_st(
					affine_model=affine_model,
					affine4x4_vox=affine_vox_4x4,
					ct_mov_full_np=ct_mov_full_np,
					pt_mov_full_np=pt_mov_full_np,
					seg_mov_full_np=ct_seg_mov_full_np,
				)
				ctseg_aff_npy = ctseg_outputs.get('seg', None)

			affine_mat = outputs['affine_matrix'].detach().cpu().numpy()[0]
			utils.savepkl(affine_mat, out_aff_path)
		except Exception as e:
			msg = f'Affine registration failed for {case_id}: {e}'
			print(msg)
			excluded_cases.append(msg)
			continue

		nib.save(nib.Nifti1Image(ct_aff_npy, make_affine_from_pixdim(TAR_PIXDIM)), out_ct_path)
		nib.save(nib.Nifti1Image(pt_aff_npy, make_affine_from_pixdim(TAR_PIXDIM)), out_pt_path)

		if seg_aff_npy is not None:
			nib.save(nib.Nifti1Image(seg_aff_npy, make_affine_from_pixdim(TAR_PIXDIM)), out_seg_path)
		if ctseg_aff_npy is not None:
			nib.save(nib.Nifti1Image(ctseg_aff_npy, make_affine_from_pixdim(TAR_PIXDIM)), out_ctseg_path)

		if os.path.exists(rtdose_path):
			msg = _process_rtdose_case(
				case_id,
				affine_model=affine_model,
				centered_affine=affine_mat,
				overwrite=False,
			)
			if msg:
				print(msg)
				excluded_cases.append(msg)

		_make_preview(ct_mov_full_np, ct_aff_npy, atlas_for_opt_np, PREVIEW_PATH, case_id)

		# Save atlas support mask used for H&N-cropped affine fitting.
		if save_qc:
			os.makedirs(QC_DIR, exist_ok=True)
			nib.save(
				nib.Nifti1Image(atlas_for_opt_np.astype(np.float32), make_affine_from_pixdim(TAR_PIXDIM)),
				os.path.join(QC_DIR, f'{case_id}_atlas_masked_template.nii.gz'),
			)
			nib.save(
				nib.Nifti1Image(atlas_slice_mask_np.astype(np.float32), make_affine_from_pixdim(TAR_PIXDIM)),
				os.path.join(QC_DIR, f'{case_id}_atlas_slice_mask.nii.gz'),
			)
        
		#sys.exit(0)
	if len(excluded_cases) > 0:
		with open(EXCLUDE_LOG_PATH, 'w') as f:
			f.write('\n'.join(excluded_cases) + '\n')
		print(f'Excluded cases logged to: {EXCLUDE_LOG_PATH} (n={len(excluded_cases)})')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--save-qc', dest='save_qc', action='store_true', help='Save QC outputs (atlas mask).')
	parser.add_argument('--no-save-qc', dest='save_qc', action='store_false', help='Do not save QC outputs.')
	parser.add_argument('--rtdose-only', action='store_true', help='Reuse existing saved affine transforms and preprocess RTDose only.')
	parser.add_argument('--overwrite-rtdose', action='store_true', help='Overwrite existing RTDose outputs when running RTDose processing.')
	parser.set_defaults(save_qc=SAVE_QC_DEFAULT)
	args = parser.parse_args()

	GPU_iden = 0
	GPU_num = torch.cuda.device_count()
	print('Number of GPU: ' + str(GPU_num))
	for GPU_idx in range(GPU_num):
		GPU_name = torch.cuda.get_device_name(GPU_idx)
		print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
	torch.cuda.set_device(GPU_iden)
	print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
	print('If the GPU is available? ' + str(torch.cuda.is_available()))
	torch.manual_seed(0)

	# 1) Generate CT multi-label anatomy with TotalSegmentator.
	# 2) Run affine preprocessing for CT/PT/(optional GT SEG and CTSeg).
	#totalsegmentation()
	if args.rtdose_only:
		preprocess_rtdose_only(overwrite=args.overwrite_rtdose)
	else:
		preprocess(save_qc=args.save_qc)
