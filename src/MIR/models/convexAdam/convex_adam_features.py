'''
Wrapper for Convex Adam with MIND features
This code is based on the original Convex Adam code from:
Siebert, Hanna, et al. "ConvexAdam: Self-Configuring Dual-Optimisation-Based 3D Multitask Medical Image Registration." IEEE Transactions on Medical Imaging (2024).
https://github.com/multimodallearning/convexAdam

Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''

import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as edt
from typing import Tuple
from MIR.models.registration_utils import VecInt
from MIR.models.convexAdam.convex_adam_utils import (MINDSSC, correlate, coupled_convex,
                                          inverse_consistency, validate_image)

warnings.filterwarnings("ignore")


def extract_features(
    img_fixed: torch.Tensor,
    img_moving: torch.Tensor,
    mind_r: int,
    mind_d: int,
    use_mask: bool,
    mask_fixed: torch.Tensor,
    mask_moving: torch.Tensor,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract MIND and/or semantic nnUNet features"""

    # MIND features
    if use_mask:
        H,W,D = img_fixed.shape[-3:]

        #replicate masking
        avg3 = nn.Sequential(nn.ReplicationPad3d(1),nn.AvgPool3d(3,stride=1))
        avg3.to(device)

        mask = (avg3(mask_fixed.view(1,1,H,W,D).to(device))>0.9).float()
        _,idx = edt((mask[0,0,::2,::2,::2]==0).squeeze().cpu().numpy(),return_indices=True)
        fixed_r = F.interpolate((img_fixed[::2,::2,::2].to(device).reshape(-1)[idx[0]*D//2*W//2+idx[1]*D//2+idx[2]]).unsqueeze(0).unsqueeze(0),scale_factor=2,mode='trilinear')
        fixed_r.view(-1)[mask.view(-1)!=0] = img_fixed.to(device).reshape(-1)[mask.view(-1)!=0]

        mask = (avg3(mask_moving.view(1,1,H,W,D).to(device))>0.9).float()
        _,idx = edt((mask[0,0,::2,::2,::2]==0).squeeze().cpu().numpy(),return_indices=True)
        moving_r = F.interpolate((img_moving[::2,::2,::2].to(device).reshape(-1)[idx[0]*D//2*W//2+idx[1]*D//2+idx[2]]).unsqueeze(0).unsqueeze(0),scale_factor=2,mode='trilinear')
        moving_r.view(-1)[mask.view(-1)!=0] = img_moving.to(device).reshape(-1)[mask.view(-1)!=0]

        features_fix = MINDSSC(fixed_r.to(device),mind_r,mind_d,device=device).to(dtype)
        features_mov = MINDSSC(moving_r.to(device),mind_r,mind_d,device=device).to(dtype)
    else:
        img_fixed = img_fixed.unsqueeze(0).unsqueeze(0)
        img_moving = img_moving.unsqueeze(0).unsqueeze(0)
        features_fix = MINDSSC(img_fixed.to(device),mind_r,mind_d,device=device).to(dtype)
        features_mov = MINDSSC(img_moving.to(device),mind_r,mind_d,device=device).to(dtype)

    return features_fix, features_mov


def _compute_convex_initial_disp(
    features_fix: torch.Tensor,
    features_mov: torch.Tensor,
    grid_sp: int,
    disp_hw: int,
    ic: bool,
    device: torch.device,
    dtype: torch.dtype,
    full_shape: Tuple[int, int, int],
) -> torch.Tensor:
    H, W, D = full_shape
    n_ch = features_fix.shape[1]

    ssd, ssd_argmin = correlate(features_fix, features_mov, disp_hw, grid_sp, (H, W, D), n_ch)
    disp_mesh_t = F.affine_grid(
        disp_hw * torch.eye(3, 4).to(device).to(dtype).unsqueeze(0),
        (1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1),
        align_corners=True,
    ).permute(0, 4, 1, 2, 3).reshape(3, -1, 1)

    disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, (H, W, D))

    if not ic:
        return disp_soft

    scale = torch.tensor([H // grid_sp - 1, W // grid_sp - 1, D // grid_sp - 1]).view(1, 3, 1, 1, 1).to(device).to(dtype) / 2
    ssd_rev, ssd_argmin_rev = correlate(features_mov, features_fix, disp_hw, grid_sp, (H, W, D), n_ch)
    disp_soft_rev = coupled_convex(ssd_rev, ssd_argmin_rev, disp_mesh_t, grid_sp, (H, W, D))
    disp_ice, _ = inverse_consistency((disp_soft / scale).flip(1), (disp_soft_rev / scale).flip(1), iter=15)
    return F.interpolate(disp_ice.flip(1) * scale * grid_sp, size=(H, W, D), mode='trilinear', align_corners=False)


def _triple_average(field: torch.Tensor, kernel_size: int = 3, padding: int = 1) -> torch.Tensor:
    return F.avg_pool3d(
        F.avg_pool3d(
            F.avg_pool3d(field, kernel_size, stride=1, padding=padding),
            kernel_size,
            stride=1,
            padding=padding,
        ),
        kernel_size,
        stride=1,
        padding=padding,
    )


def _diffusion_regularization(field: torch.Tensor, lambda_weight: float) -> torch.Tensor:
    return (
        lambda_weight * (field[:, :, 1:, :, :] - field[:, :, :-1, :, :]).pow(2).mean()
        + lambda_weight * (field[:, :, :, 1:, :] - field[:, :, :, :-1, :]).pow(2).mean()
        + lambda_weight * (field[:, :, :, :, 1:] - field[:, :, :, :, :-1]).pow(2).mean()
    )


def convex_adam_feats(
    img_fixed: Union[torch.Tensor, np.ndarray, nib.Nifti1Image],
    img_moving: Union[torch.Tensor, np.ndarray, nib.Nifti1Image],
    initial_disp: Optional[torch.Tensor] = None,
    lambda_weight: float = 1.25,
    grid_sp: int = 6,
    disp_hw: int = 4,
    selected_niter: int = 80,
    selected_smooth: int = 0,
    grid_sp_adam: int = 2,
    ic: bool = True,
    dtype: torch.dtype = torch.float16,
    verbose: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_disp: bool = True,
) -> np.ndarray:
    """Coupled convex optimisation with adam instance optimisation"""
    features_fix = img_fixed.float()
    features_mov = img_moving.float()
    if dtype == torch.float16 and device == torch.device("cpu"):
        print("Warning: float16 is not supported on CPU, using float32 instead")
        dtype = torch.float32

    _, _, H, W, D = img_fixed.shape

    t0 = time.time()

    # compute features and downsample (using average pooling)
    with torch.no_grad():
        features_fix_smooth = F.avg_pool3d(features_fix, grid_sp, stride=grid_sp)
        features_mov_smooth = F.avg_pool3d(features_mov, grid_sp, stride=grid_sp)

    if initial_disp is not None:
        disp_hr = initial_disp
    else:
        disp_hr = _compute_convex_initial_disp(
            features_fix=features_fix_smooth,
            features_mov=features_mov_smooth,
            grid_sp=grid_sp,
            disp_hw=disp_hw,
            ic=ic,
            device=device,
            dtype=dtype,
            full_shape=(H, W, D),
        )
    
    # run Adam instance optimisation
    if lambda_weight > 0:
        with torch.no_grad():
            patch_features_fix = F.avg_pool3d(features_fix,grid_sp_adam,stride=grid_sp_adam)
            patch_features_mov = F.avg_pool3d(features_mov,grid_sp_adam,stride=grid_sp_adam)

        #create optimisable displacement grid
        disp_lr = F.interpolate(disp_hr,size=(H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam),mode='trilinear',align_corners=False)

        net = nn.Sequential(nn.Conv3d(3,1,(H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam),bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data/grid_sp_adam
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1)

        grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0).to(device),(1,1,H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam),align_corners=False)

        #run Adam optimisation with diffusion regularisation and B-spline smoothing
        for iter in range(selected_niter):
            optimizer.zero_grad()

            disp_sample = _triple_average(net[0].weight).permute(0,2,3,4,1)
            reg_loss = lambda_weight*((disp_sample[0,:,1:,:]-disp_sample[0,:,:-1,:])**2).mean()+\
            lambda_weight*((disp_sample[0,1:,:,:]-disp_sample[0,:-1,:,:])**2).mean()+\
            lambda_weight*((disp_sample[0,:,:,1:]-disp_sample[0,:,:,:-1])**2).mean()

            scale = torch.tensor([(H//grid_sp_adam-1)/2,(W//grid_sp_adam-1)/2,(D//grid_sp_adam-1)/2]).to(device).unsqueeze(0)
            grid_disp = grid0.view(-1,3).to(device).float()+((disp_sample.view(-1,3))/scale).flip(1).float()

            patch_mov_sampled = F.grid_sample(patch_features_mov.float(),grid_disp.view(1,H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam,3).to(device),align_corners=False,mode='bilinear')

            sampled_cost = (patch_mov_sampled-patch_features_fix).pow(2).mean(1)*12
            loss = sampled_cost.mean()
            (loss+reg_loss).backward()
            optimizer.step()

        fitted_grid = disp_sample.detach().permute(0,4,1,2,3)
        disp_hr = F.interpolate(fitted_grid*grid_sp_adam,size=(H,W,D),mode='trilinear',align_corners=False)

        if selected_smooth > 0:
            if selected_smooth % 2 == 0:
                kernel_smooth = selected_smooth+1
                print('selected_smooth should be an odd number, adding 1')

            kernel_smooth = selected_smooth
            padding_smooth = kernel_smooth//2
            disp_hr = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(disp_hr,kernel_smooth,padding=padding_smooth,stride=1),kernel_smooth,padding=padding_smooth,stride=1),kernel_smooth,padding=padding_smooth,stride=1)

    t1 = time.time()
    case_time = t1-t0
    if verbose:
        print(f'case time: {case_time}')
    if save_disp:
        x = disp_hr[0,0,:,:,:].cpu().to(dtype).data.numpy()
        y = disp_hr[0,1,:,:,:].cpu().to(dtype).data.numpy()
        z = disp_hr[0,2,:,:,:].cpu().to(dtype).data.numpy()
        displacements = np.stack((x,y,z),3).astype(float)
    else:
        displacements = disp_hr
    return displacements


def convex_adam_feats_svf(
    img_fixed: Union[torch.Tensor, np.ndarray, nib.Nifti1Image],
    img_moving: Union[torch.Tensor, np.ndarray, nib.Nifti1Image],
    initial_disp: Optional[torch.Tensor] = None,
    initial_velocity: Optional[torch.Tensor] = None,
    lambda_weight: float = 1.25,
    grid_sp: int = 6,
    disp_hw: int = 4,
    selected_niter: int = 80,
    selected_smooth: int = 0,
    grid_sp_adam: int = 2,
    ic: bool = True,
    svf_steps: int = 7,
    dtype: torch.dtype = torch.float16,
    verbose: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_disp: bool = True,
    return_velocity: bool = False,
):
    """ConvexAdam optimization with an SVF parameterization and forward/reverse outputs."""
    features_fix = img_fixed.float()
    features_mov = img_moving.float()
    if dtype == torch.float16 and device == torch.device("cpu"):
        print("Warning: float16 is not supported on CPU, using float32 instead")
        dtype = torch.float32

    _, _, H, W, D = features_fix.shape
    h_lr, w_lr, d_lr = H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam

    t0 = time.time()

    with torch.no_grad():
        features_fix_smooth = F.avg_pool3d(features_fix, grid_sp, stride=grid_sp)
        features_mov_smooth = F.avg_pool3d(features_mov, grid_sp, stride=grid_sp)

    if initial_velocity is None:
        if initial_disp is None:
            initial_disp = _compute_convex_initial_disp(
                features_fix=features_fix_smooth,
                features_mov=features_mov_smooth,
                grid_sp=grid_sp,
                disp_hw=disp_hw,
                ic=ic,
                device=device,
                dtype=dtype,
                full_shape=(H, W, D),
            )
        initial_velocity = initial_disp

    with torch.no_grad():
        patch_features_fix = F.avg_pool3d(features_fix, grid_sp_adam, stride=grid_sp_adam)
        patch_features_mov = F.avg_pool3d(features_mov, grid_sp_adam, stride=grid_sp_adam)

    velocity_lr = F.interpolate(initial_velocity, size=(h_lr, w_lr, d_lr), mode='trilinear', align_corners=False)

    net = nn.Sequential(nn.Conv3d(3, 1, (h_lr, w_lr, d_lr), bias=False))
    net[0].weight.data[:] = velocity_lr.float().cpu().data / grid_sp_adam
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1)

    integrator_lr = VecInt((h_lr, w_lr, d_lr), svf_steps).to(device)
    transformer_lr = integrator_lr.transformer

    for iter in range(selected_niter):
        optimizer.zero_grad()

        velocity_sample = _triple_average(net[0].weight)
        reg_loss = _diffusion_regularization(velocity_sample, lambda_weight)
        disp_sample = integrator_lr(velocity_sample.float())

        patch_mov_sampled = transformer_lr(patch_features_mov.float(), disp_sample, padding_mode='border')
        sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1) * 12
        loss = sampled_cost.mean()
        (loss + reg_loss).backward()
        optimizer.step()

    fitted_velocity = velocity_sample.detach()

    if selected_smooth > 0:
        if selected_smooth % 2 == 0:
            kernel_smooth = selected_smooth + 1
            print('selected_smooth should be an odd number, adding 1')
        else:
            kernel_smooth = selected_smooth
        fitted_velocity = _triple_average(fitted_velocity, kernel_size=kernel_smooth, padding=kernel_smooth // 2)

    velocity_hr = F.interpolate(fitted_velocity * grid_sp_adam, size=(H, W, D), mode='trilinear', align_corners=False)
    integrator_hr = VecInt((H, W, D), svf_steps).to(device)
    disp_fwd = integrator_hr(velocity_hr.float())
    disp_rev = integrator_hr((-velocity_hr).float())

    t1 = time.time()
    case_time = t1 - t0
    if verbose:
        print(f'case time: {case_time}')

    if save_disp:
        fwd = np.stack(
            (
                disp_fwd[0, 0].cpu().to(dtype).data.numpy(),
                disp_fwd[0, 1].cpu().to(dtype).data.numpy(),
                disp_fwd[0, 2].cpu().to(dtype).data.numpy(),
            ),
            3,
        ).astype(float)
        rev = np.stack(
            (
                disp_rev[0, 0].cpu().to(dtype).data.numpy(),
                disp_rev[0, 1].cpu().to(dtype).data.numpy(),
                disp_rev[0, 2].cpu().to(dtype).data.numpy(),
            ),
            3,
        ).astype(float)
        if return_velocity:
            vel = np.stack(
                (
                    velocity_hr[0, 0].cpu().to(dtype).data.numpy(),
                    velocity_hr[0, 1].cpu().to(dtype).data.numpy(),
                    velocity_hr[0, 2].cpu().to(dtype).data.numpy(),
                ),
                3,
            ).astype(float)
            return fwd, rev, vel
        return fwd, rev

    if return_velocity:
        return disp_fwd, disp_rev, velocity_hr

    return disp_fwd, disp_rev


def convex_adam_features_svf(
    img_moving,
    img_fixed,
    configs,
    initial_disp=None,
    initial_velocity=None,
):
    """Feature-space ConvexAdam variant parameterized by an SVF."""

    return convex_adam_feats_svf(
        img_fixed=img_fixed,
        img_moving=img_moving,
        initial_disp=initial_disp,
        initial_velocity=initial_velocity,
        lambda_weight=configs.lambda_weight,
        grid_sp=configs.grid_sp,
        disp_hw=configs.disp_hw,
        selected_niter=configs.selected_niter,
        selected_smooth=configs.selected_smooth,
        grid_sp_adam=configs.grid_sp_adam,
        ic=configs.ic,
        svf_steps=getattr(configs, 'svf_steps', 7),
        verbose=configs.verbose,
        save_disp=False,
        return_velocity=getattr(configs, 'return_velocity', False),
    )

def convex_adam_features(
    img_moving,
    img_fixed,
    configs,
    initial_disp=None
) -> None:
    """Coupled convex optimisation with adam instance optimisation"""

    displacements = convex_adam_feats(
        img_fixed=img_fixed,
        img_moving=img_moving,
        initial_disp=initial_disp,
        lambda_weight=configs.lambda_weight,
        grid_sp=configs.grid_sp,
        disp_hw=configs.disp_hw,
        selected_niter=configs.selected_niter,
        selected_smooth=configs.selected_smooth,
        grid_sp_adam=configs.grid_sp_adam,
        ic=configs.ic,
        verbose=configs.verbose,
        save_disp=False,
    )
    
    return displacements


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--path_img_fixed", type=str, required=True)
    parser.add_argument("-m",'--path_img_moving', type=str, required=True)
    parser.add_argument('--mind_r', type=int, default=1)
    parser.add_argument('--mind_d', type=int, default=2)
    parser.add_argument('--lambda_weight', type=float, default=1.25)
    parser.add_argument('--grid_sp', type=int, default=6)
    parser.add_argument('--disp_hw', type=int, default=4)
    parser.add_argument('--selected_niter', type=int, default=80)
    parser.add_argument('--selected_smooth', type=int, default=0)
    parser.add_argument('--grid_sp_adam', type=int, default=2)
    parser.add_argument('--ic', choices=('True','False'), default='True')
    parser.add_argument('--use_mask', choices=('True','False'), default='False')
    parser.add_argument('--path_mask_fixed', type=str, default=None)
    parser.add_argument('--path_mask_moving', type=str, default=None)
    parser.add_argument('--result_path', type=str, default='./')

    args = parser.parse_args()

    convex_adam_features(
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
        ic=(args.ic == 'True'),
        use_mask=(args.use_mask == 'True'),
        path_fixed_mask=args.path_mask_fixed,
        path_moving_mask=args.path_mask_moving,
        result_path=args.result_path,
    )
