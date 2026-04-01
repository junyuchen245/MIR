'''
Wrapper for Convex Adam with MIND features and SVF parameterization.
This code is based on the original Convex Adam code from:
Siebert, Hanna, et al. "ConvexAdam: Self-Configuring Dual-Optimisation-Based 3D Multitask Medical Image Registration." IEEE Transactions on Medical Imaging (2024).
https://github.com/multimodallearning/convexAdam

Modified and extended by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''

import argparse
import os
import warnings
from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np
import torch

from MIR.models.convexAdam.convex_adam_features import convex_adam_feats_svf
from MIR.models.convexAdam.convex_adam_MIND import extract_features

warnings.filterwarnings("ignore")


def convex_adam_pt_svf(
    img_fixed: Union[torch.Tensor, np.ndarray, nib.Nifti1Image],
    img_moving: Union[torch.Tensor, np.ndarray, nib.Nifti1Image],
    mind_r: int = 1,
    mind_d: int = 2,
    lambda_weight: float = 1.25,
    grid_sp: int = 6,
    disp_hw: int = 4,
    selected_niter: int = 80,
    selected_smooth: int = 0,
    grid_sp_adam: int = 2,
    ic: bool = True,
    use_mask: bool = False,
    path_fixed_mask: Optional[Union[Path, str]] = None,
    path_moving_mask: Optional[Union[Path, str]] = None,
    svf_steps: int = 7,
    dtype: torch.dtype = torch.float16,
    verbose: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_disp: bool = True,
    return_velocity: bool = False,
):
    """ConvexAdam MIND registration with SVF optimization and forward/reverse displacement outputs."""
    img_fixed = img_fixed.float()
    img_moving = img_moving.float()

    if dtype == torch.float16 and device == torch.device("cpu"):
        print("Warning: float16 is not supported on CPU, using float32 instead")
        dtype = torch.float32

    if use_mask:
        mask_fixed = torch.from_numpy(nib.load(path_fixed_mask).get_fdata()).float()
        mask_moving = torch.from_numpy(nib.load(path_moving_mask).get_fdata()).float()
    else:
        mask_fixed = None
        mask_moving = None

    with torch.no_grad():
        features_fix, features_mov = extract_features(
            img_fixed=img_fixed,
            img_moving=img_moving,
            mind_r=mind_r,
            mind_d=mind_d,
            use_mask=use_mask,
            mask_fixed=mask_fixed,
            mask_moving=mask_moving,
            device=device,
            dtype=dtype,
        )

    return convex_adam_feats_svf(
        img_fixed=features_fix,
        img_moving=features_mov,
        lambda_weight=lambda_weight,
        grid_sp=grid_sp,
        disp_hw=disp_hw,
        selected_niter=selected_niter,
        selected_smooth=selected_smooth,
        grid_sp_adam=grid_sp_adam,
        ic=ic,
        svf_steps=svf_steps,
        dtype=dtype,
        verbose=verbose,
        device=device,
        save_disp=save_disp,
        return_velocity=return_velocity,
    )


def convex_adam_svf(
    path_img_fixed: Union[Path, str],
    path_img_moving: Union[Path, str],
    mind_r: int = 1,
    mind_d: int = 2,
    lambda_weight: float = 1.25,
    grid_sp: int = 6,
    disp_hw: int = 4,
    selected_niter: int = 80,
    selected_smooth: int = 0,
    grid_sp_adam: int = 2,
    ic: bool = True,
    use_mask: bool = False,
    path_fixed_mask: Optional[Union[Path, str]] = None,
    path_moving_mask: Optional[Union[Path, str]] = None,
    svf_steps: int = 7,
    result_path: Union[Path, str] = './',
    verbose: bool = False,
) -> None:
    """Run ConvexAdam MIND-SVF from image paths and save both directions."""

    img_fixed = torch.from_numpy(nib.load(path_img_fixed).get_fdata()).float()
    img_moving = torch.from_numpy(nib.load(path_img_moving).get_fdata()).float()

    disp_fwd, disp_rev = convex_adam_pt_svf(
        img_fixed=img_fixed,
        img_moving=img_moving,
        mind_r=mind_r,
        mind_d=mind_d,
        lambda_weight=lambda_weight,
        grid_sp=grid_sp,
        disp_hw=disp_hw,
        selected_niter=selected_niter,
        selected_smooth=selected_smooth,
        grid_sp_adam=grid_sp_adam,
        ic=ic,
        use_mask=use_mask,
        path_fixed_mask=path_fixed_mask,
        path_moving_mask=path_moving_mask,
        svf_steps=svf_steps,
        verbose=verbose,
        save_disp=True,
    )

    affine = nib.load(path_img_fixed).affine
    nib.save(nib.Nifti1Image(disp_fwd, affine), os.path.join(result_path, 'disp_fwd.nii.gz'))
    nib.save(nib.Nifti1Image(disp_rev, affine), os.path.join(result_path, 'disp_rev.nii.gz'))


def convex_adam_MIND_SVF(
    img_moving,
    img_fixed,
    configs,
):
    """ConvexAdam MIND-SVF wrapper returning forward and reverse displacement tensors."""

    return convex_adam_pt_svf(
        img_fixed=img_fixed[0, 0],
        img_moving=img_moving[0, 0],
        mind_r=configs.mind_r,
        mind_d=configs.mind_d,
        lambda_weight=configs.lambda_weight,
        grid_sp=configs.grid_sp,
        disp_hw=configs.disp_hw,
        selected_niter=configs.selected_niter,
        selected_smooth=configs.selected_smooth,
        grid_sp_adam=configs.grid_sp_adam,
        ic=configs.ic,
        use_mask=configs.use_mask,
        path_fixed_mask=configs.path_fixed_mask,
        path_moving_mask=configs.path_moving_mask,
        svf_steps=getattr(configs, 'svf_steps', 7),
        verbose=configs.verbose,
        save_disp=False,
        return_velocity=getattr(configs, 'return_velocity', False),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--path_img_fixed", type=str, required=True)
    parser.add_argument("-m", "--path_img_moving", type=str, required=True)
    parser.add_argument('--mind_r', type=int, default=1)
    parser.add_argument('--mind_d', type=int, default=2)
    parser.add_argument('--lambda_weight', type=float, default=1.25)
    parser.add_argument('--grid_sp', type=int, default=6)
    parser.add_argument('--disp_hw', type=int, default=4)
    parser.add_argument('--selected_niter', type=int, default=80)
    parser.add_argument('--selected_smooth', type=int, default=0)
    parser.add_argument('--grid_sp_adam', type=int, default=2)
    parser.add_argument('--svf_steps', type=int, default=7)
    parser.add_argument('--ic', choices=('True', 'False'), default='True')
    parser.add_argument('--use_mask', choices=('True', 'False'), default='False')
    parser.add_argument('--path_mask_fixed', type=str, default=None)
    parser.add_argument('--path_mask_moving', type=str, default=None)
    parser.add_argument('--result_path', type=str, default='./')

    args = parser.parse_args()

    convex_adam_svf(
        path_img_fixed=args.path_img_fixed,
        path_img_moving=args.path_img_moving,
        mind_r=args.mind_r,
        mind_d=args.mind_d,
        lambda_weight=args.lambda_weight,
        grid_sp=args.grid_sp,
        disp_hw=args.disp_hw,
        selected_niter=args.selected_niter,
        selected_smooth=args.selected_smooth,
        grid_sp_adam=args.grid_sp_adam,
        svf_steps=args.svf_steps,
        ic=(args.ic == 'True'),
        use_mask=(args.use_mask == 'True'),
        path_fixed_mask=args.path_mask_fixed,
        path_moving_mask=args.path_mask_moving,
        result_path=args.result_path,
    )