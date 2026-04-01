import argparse
import glob
import os

import nibabel as nib
import numpy as np
import torch
from nibabel.filebasedimages import ImageFileError

from MIR.models import AffineReg3D
import MIR.utils.other_utils as utils

import preprocessing as base


DEFAULT_BATCH_DIRS = [
    '/scratch2/jchen/DATA/JHU_FDG/batch7',
    '/scratch2/jchen/DATA/JHU_FDG/batch8',
]

VISTA_SUFFIX = '_CT_vista3d_seg.nii.gz'
OUTPUT_SUFFIX = '_vista_CTSeg.nii.gz'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess JHU FDG VISTA labels using existing crop logic and saved affines.',
    )
    parser.add_argument(
        '--batch-dirs',
        nargs='+',
        default=DEFAULT_BATCH_DIRS,
        help='Batch directories to process.',
    )
    parser.add_argument(
        '--vista-dirname',
        default='seg/vista3d_ct_breast_fat_muscle_combined',
        help='Relative subdirectory containing VISTA segmentations inside each batch.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing *_vista_CTSeg outputs.',
    )
    return parser.parse_args()


def case_id_from_vista_path(vista_path):
    name = os.path.basename(vista_path)
    if name.endswith(VISTA_SUFFIX):
        return name[:-len(VISTA_SUFFIX)]
    if name.endswith('.nii.gz'):
        return name[:-7]
    return os.path.splitext(name)[0]


def crop_ct_space_volume_like_preprocess(vol_samp, ct_seg_samp):
    seg_lbl = np.rint(ct_seg_samp).astype(np.int16)
    axis, si_dir = base._find_si_axis_and_direction(seg_lbl)
    if axis is None:
        raise RuntimeError('Missing superior/inferior anchors in CT segmentation.')

    lo, hi = base._compute_mid_thigh_to_vertex_bounds(seg_lbl, axis, si_dir)
    if lo is None:
        raise RuntimeError('Cannot determine crop bounds (missing skull/brain or hips/femurs).')

    return base._crop_by_axis_bounds(vol_samp, axis, lo, hi), {
        'axis': axis,
        'ct_bounds': (lo, hi),
        'vol_bounds': (lo, hi),
    }


def process_case(vista_path, batch_dir, affine_model, zero_atlas_np, overwrite=False):
    case_id = case_id_from_vista_path(vista_path)
    out_path = os.path.join(batch_dir, 'preprocessed', f'{case_id}{OUTPUT_SUFFIX}')
    aff_path = os.path.join(batch_dir, 'affine', f'{case_id}_affine_ants_voxel.pkl')
    ct_seg_path = os.path.join(batch_dir, 'seg', 'ct', f'{case_id}_CTSeg.nii.gz')

    if os.path.exists(out_path) and not overwrite:
        print(f'Skipping {case_id}: output exists.')
        return 'skipped'

    if not os.path.exists(aff_path):
        print(f'Skipping {case_id}: missing affine {aff_path}')
        return 'missing-affine'

    if not os.path.exists(ct_seg_path):
        print(f'Skipping {case_id}: missing CT segmentation {ct_seg_path}')
        return 'missing-ct-seg'

    try:
        vista_nib = nib.load(vista_path)
        ct_seg_nib = nib.load(ct_seg_path)
    except ImageFileError as exc:
        print(f'Skipping {case_id}: invalid NIfTI input ({exc})')
        return 'invalid-input'
    except Exception as exc:
        print(f'Skipping {case_id}: failed loading inputs ({exc})')
        return 'load-failed'

    vista_np = vista_nib.get_fdata()
    ct_seg_np = ct_seg_nib.get_fdata()

    vista_pixdim = vista_nib.header['pixdim'][1:4]
    ct_seg_pixdim = ct_seg_nib.header['pixdim'][1:4]

    vista_samp = base.zoom_img(vista_np, (vista_pixdim, base.TAR_PIXDIM), order=0)
    ct_seg_samp = base.zoom_img(ct_seg_np, (ct_seg_pixdim, base.TAR_PIXDIM), order=0)

    if not base._is_nonempty_volume(vista_samp) or not base._is_nonempty_volume(ct_seg_samp):
        print(
            f'Skipping {case_id}: empty resampled volume '
            f'(vista={vista_samp.shape}, ct_seg={ct_seg_samp.shape})'
        )
        return 'empty-resampled'

    try:
        vista_crop, crop_info = crop_ct_space_volume_like_preprocess(vista_samp, ct_seg_samp)
    except Exception as exc:
        print(f'Skipping {case_id}: crop failed ({exc})')
        return 'crop-failed'

    if not base._is_nonempty_volume(vista_crop):
        print(f'Skipping {case_id}: empty cropped VISTA volume {vista_crop.shape}')
        return 'empty-cropped'

    vista_full_np = base._pad_or_crop_np(vista_crop, base.CT_ATLAS_TORCH.shape[2:], fill_value=0.0)
    affine_vox_4x4 = utils.pkload(aff_path)

    outputs = base._apply_voxel_affine_with_st(
        affine_model=affine_model,
        affine4x4_vox=affine_vox_4x4,
        ct_mov_full_np=zero_atlas_np,
        ct_seg_mov_full_np=vista_full_np,
        pt_mov_full_np=zero_atlas_np,
        atlas_np=base.CT_ATLAS_NPY,
    )

    vista_aff_np = np.rint(outputs['seg']).astype(np.int16)
    nib.save(
        nib.Nifti1Image(vista_aff_np, base.make_affine_from_pixdim(base.TAR_PIXDIM)),
        out_path,
    )
    print(
        f'Processed {case_id}: '
        f'crop axis={crop_info["axis"]}, '
        f'ct={crop_info["ct_bounds"]}, '
        f'vista={crop_info["vol_bounds"]}'
    )
    return 'processed'


def process_batch(batch_dir, affine_model, args, zero_atlas_np):
    vista_dir = os.path.join(batch_dir, args.vista_dirname)
    output_dir = os.path.join(batch_dir, 'preprocessed')
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(vista_dir):
        print(f'Skipping batch {batch_dir}: missing VISTA directory {vista_dir}')
        return {}

    vista_paths = sorted(glob.glob(os.path.join(vista_dir, '*.nii.gz')))
    if len(vista_paths) == 0:
        print(f'No VISTA segmentations found in {vista_dir}')
        return {}

    counts = {}
    for vista_path in vista_paths:
        status = process_case(
            vista_path=vista_path,
            batch_dir=batch_dir,
            affine_model=affine_model,
            zero_atlas_np=zero_atlas_np,
            overwrite=args.overwrite,
        )
        counts[status] = counts.get(status, 0) + 1
    return counts


def print_gpu_info():
    gpu_id = 0
    gpu_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(gpu_num))
    for gpu_idx in range(gpu_num):
        gpu_name = torch.cuda.get_device_name(gpu_idx)
        print('     GPU #' + str(gpu_idx) + ': ' + gpu_name)
    torch.cuda.set_device(gpu_id)
    gpu_available = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(gpu_id))
    print('If the GPU is available? ' + str(gpu_available))
    torch.manual_seed(0)


def main():
    args = parse_args()
    print_gpu_info()

    affine_model = AffineReg3D(
        vol_shape=base.CT_ATLAS_TORCH.shape[2:],
        dof='affine',
        scales=(0.25, 0.5, 1),
        loss_funcs=('mse', 'mse', 'mse'),
    ).cuda(0)
    zero_atlas_np = np.zeros(base.CT_ATLAS_TORCH.shape[2:], dtype=np.float32)

    for batch_dir in args.batch_dirs:
        print(f'Processing batch: {batch_dir}')
        counts = process_batch(batch_dir, affine_model, args, zero_atlas_np)
        if counts:
            summary = ', '.join(f'{key}={value}' for key, value in sorted(counts.items()))
            print(f'Batch summary {batch_dir}: {summary}')


if __name__ == '__main__':
    main()