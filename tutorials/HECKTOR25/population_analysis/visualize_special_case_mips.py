import argparse
import glob
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPECIAL_CASE_DIR = os.path.join(THIS_DIR, 'population_stats', 'special_cases')
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_SPECIAL_CASE_DIR, 'mips')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create PET/Z-score MIP visualizations and overlays for HECKTOR special cases with zero-valued background.'
    )
    parser.add_argument('--input-dir', default=DEFAULT_SPECIAL_CASE_DIR, help='Directory containing *_PET_* and *_Z_* NIfTI files.')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Directory to save PNG outputs.')
    parser.add_argument('--axis', type=int, default=1, choices=[0, 1, 2], help='Projection axis. Default 1 gives a coronal-like MIP.')
    parser.add_argument('--margin', type=int, default=6, help='Crop margin in voxels around the non-background box.')
    parser.add_argument('--pet-min', type=float, default=1e-6, help='PET threshold used to define foreground/background.')
    parser.add_argument('--pet-vmax-percentile', type=float, default=99.5, help='Upper PET display percentile.')
    parser.add_argument('--z-min', type=float, default=2.5, help='Minimum positive Z-score shown in standalone/overlay maps.')
    parser.add_argument('--z-vmax', type=float, default=0.0, help='Maximum displayed Z-score for color scaling. Use <=0 for auto scaling.')
    return parser.parse_args()


def discover_cases(input_dir: str) -> List[Dict[str, str]]:
    pet_paths = sorted(glob.glob(os.path.join(input_dir, '*_PET_*.nii.gz')))
    cases: List[Dict[str, str]] = []
    for pet_path in pet_paths:
        base = os.path.basename(pet_path)
        if '_PET_' not in base:
            continue
        prefix, suffix = base.split('_PET_', 1)
        z_path = os.path.join(input_dir, f'{prefix}_Z_{suffix}')
        if not os.path.exists(z_path):
            continue
        cases.append({
            'case_id': prefix,
            'tag': suffix.replace('.nii.gz', ''),
            'pet_path': pet_path,
            'z_path': z_path,
        })
    return cases


def load_volume(path: str) -> np.ndarray:
    return nib.load(path).get_fdata().astype(np.float32)


def build_pet_foreground_mask(pet: np.ndarray, pet_min: float) -> np.ndarray:
    mask = np.isfinite(pet) & (pet > float(pet_min))
    if not mask.any():
        return mask
    labels, n_labels = ndimage.label(mask.astype(np.uint8), structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n_labels <= 1:
        return mask
    sizes = ndimage.sum(mask.astype(np.float32), labels=labels, index=np.arange(1, n_labels + 1))
    largest_idx = int(np.argmax(sizes)) + 1
    return labels == largest_idx


def compute_crop_slices(mask: np.ndarray, margin: int) -> Tuple[slice, slice, slice]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return tuple(slice(0, dim) for dim in mask.shape)
    mins = np.maximum(coords.min(axis=0) - int(margin), 0)
    maxs = np.minimum(coords.max(axis=0) + int(margin) + 1, np.asarray(mask.shape))
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))


def crop_volume(vol: np.ndarray, crop_slices: Tuple[slice, slice, slice]) -> np.ndarray:
    return vol[crop_slices[0], crop_slices[1], crop_slices[2]]


def make_mip(vol: np.ndarray, axis: int) -> np.ndarray:
    mip = np.nanmax(vol, axis=axis)
    mip = np.nan_to_num(mip, nan=0.0, posinf=0.0, neginf=0.0)
    mip = np.rot90(mip)
    return mip


def normalize_pet_for_display(pet_mip: np.ndarray, vmax_percentile: float) -> np.ndarray:
    positive = pet_mip[np.isfinite(pet_mip) & (pet_mip > 0)]
    if positive.size == 0:
        return np.zeros_like(pet_mip, dtype=np.float32)
    vmax = float(np.percentile(positive, vmax_percentile))
    vmax = max(vmax, float(positive.max()), 1e-6) if vmax <= 0 else vmax
    return np.clip(pet_mip / vmax, 0.0, 1.0).astype(np.float32)


def normalize_z_for_display(z_mip: np.ndarray, z_min: float, z_vmax: float) -> np.ndarray:
    positive = z_mip[np.isfinite(z_mip) & (z_mip > float(z_min))]
    if positive.size == 0:
        return np.zeros_like(z_mip, dtype=np.float32)
    if z_vmax is None or float(z_vmax) <= float(z_min):
        vmax = float(np.max(positive))
    else:
        vmax = float(z_vmax)
    vmax = max(vmax, float(z_min) + 1e-6)
    z_pos = np.clip(z_mip, z_min, vmax)
    z_pos[z_mip <= z_min] = 0.0
    return ((z_pos - z_min) / max(vmax - z_min, 1e-6)).astype(np.float32)


def _colorize_with_zero_background(img: np.ndarray, mask: np.ndarray, cmap: str, background_rgb=(0.0, 0.0, 0.0)) -> np.ndarray:
    cm = plt.get_cmap(cmap)
    rgba = cm(np.clip(img, 0.0, 1.0))
    background_rgb = np.asarray(background_rgb, dtype=np.float32)
    rgba[..., :3] = rgba[..., :3] * mask[..., None] + background_rgb[None, None, :] * (1.0 - mask[..., None])
    rgba[..., 3] = 1.0
    return rgba


def save_single_map(img: np.ndarray, mask: np.ndarray, out_path: str, cmap: str, background_rgb=(0.0, 0.0, 0.0)) -> None:
    rgb = _colorize_with_zero_background(img, mask, cmap, background_rgb=background_rgb)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    ax.imshow(rgb, interpolation='nearest')
    ax.set_axis_off()
    fig.patch.set_facecolor(background_rgb)
    ax.set_facecolor(background_rgb)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    try:
        fig.savefig(out_path, facecolor=background_rgb, bbox_inches='tight', pad_inches=0)
    finally:
        plt.close(fig)


def save_overlay(pet_img: np.ndarray, pet_mask: np.ndarray, z_img: np.ndarray, z_mask: np.ndarray, out_path: str) -> None:
    pet_rgb = _colorize_with_zero_background(pet_img, pet_mask, 'gray_r', background_rgb=(1.0, 1.0, 1.0))
    z_rgb = _colorize_with_zero_background(z_img, z_mask, 'turbo', background_rgb=(1.0, 1.0, 1.0))
    overlay = pet_rgb.copy()
    blend_mask = z_mask > 0
    overlay[..., :3][blend_mask] = 0.45 * pet_rgb[..., :3][blend_mask] + 0.55 * z_rgb[..., :3][blend_mask]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    ax.imshow(overlay, interpolation='nearest')
    ax.set_axis_off()
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    try:
        fig.savefig(out_path, facecolor='white', bbox_inches='tight', pad_inches=0)
    finally:
        plt.close(fig)


def save_summary_figure(rows: List[Dict[str, str]], out_path: str) -> None:
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3.2 * max(n_rows, 1)), dpi=180)
    if n_rows == 1:
        axes = np.asarray([axes])
    column_titles = ['PET MIP', 'Z-score MIP', 'Overlay']
    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title, fontsize=10)
    for row_idx, row in enumerate(rows):
        pet_png = plt.imread(row['pet_png'])
        z_png = plt.imread(row['z_png'])
        ov_png = plt.imread(row['overlay_png'])
        for col_idx, arr in enumerate([pet_png, z_png, ov_png]):
            axes[row_idx, col_idx].imshow(arr)
            axes[row_idx, col_idx].set_axis_off()
        axes[row_idx, 0].text(
            0.02,
            0.98,
            f"{row['case_id']} ({row['tag'].replace('_', ' ')})",
            transform=axes[row_idx, 0].transAxes,
            ha='left',
            va='top',
            fontsize=9,
            color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=(0, 0, 0, 0.55), edgecolor='none'),
        )
    fig.patch.set_facecolor('white')
    plt.tight_layout(pad=0.5)
    try:
        fig.savefig(out_path, facecolor='white', bbox_inches='tight', pad_inches=0.05)
    finally:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cases = discover_cases(args.input_dir)
    if not cases:
        raise RuntimeError(f'No paired PET/Z special-case files found in {args.input_dir}')

    summary_rows: List[Dict[str, str]] = []
    for case in cases:
        pet = load_volume(case['pet_path'])
        z = load_volume(case['z_path'])
        foreground = build_pet_foreground_mask(pet, args.pet_min)
        crop_slices = compute_crop_slices(foreground, args.margin)
        pet_c = crop_volume(np.where(foreground, pet, 0.0), crop_slices)
        z_masked = np.where(foreground, z, 0.0)
        z_c = crop_volume(z_masked, crop_slices)

        pet_mip = make_mip(pet_c, axis=args.axis)
        z_mip = make_mip(np.where(z_c > float(args.z_min), z_c, 0.0), axis=args.axis)

        pet_img = normalize_pet_for_display(pet_mip, args.pet_vmax_percentile)
        z_img = normalize_z_for_display(z_mip, args.z_min, args.z_vmax)

        pet_mask = (pet_img > 0).astype(np.float32)
        z_mask = (z_img > 0).astype(np.float32)

        stem = f"{case['case_id']}_{case['tag']}"
        pet_png = os.path.join(args.output_dir, f'{stem}_pet_mip.png')
        z_png = os.path.join(args.output_dir, f'{stem}_z_mip.png')
        overlay_png = os.path.join(args.output_dir, f'{stem}_overlay_mip.png')

        save_single_map(pet_img, pet_mask, pet_png, cmap='gray_r', background_rgb=(1.0, 1.0, 1.0))
        save_single_map(z_img, z_mask, z_png, cmap='turbo', background_rgb=(1.0, 1.0, 1.0))
        save_overlay(pet_img, pet_mask, z_img, z_mask, overlay_png)

        summary_rows.append({
            'case_id': case['case_id'],
            'tag': case['tag'],
            'pet_png': pet_png,
            'z_png': z_png,
            'overlay_png': overlay_png,
        })
        print(f"Saved MIPs for {case['case_id']} ({case['tag']})")

    summary_png = os.path.join(args.output_dir, 'special_cases_mip_summary.png')
    save_summary_figure(summary_rows, summary_png)
    print(f'Saved summary figure: {summary_png}')


if __name__ == '__main__':
    main()
