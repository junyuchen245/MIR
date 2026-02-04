"""3D affine registration module using MIR SpatialTransformer."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path
import importlib.resources as resources

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import nibabel as nib
from MIR.models.registration_utils import SpatialTransformer
from MIR.models.convexAdam.convex_adam_utils import MINDSSC, correlate, coupled_convex
from MIR.image_similarity import (
    NCC_vxm,
    PCC,
    FastNCC,
    LocalCorrRatio,
    CorrRatio,
    SSIM3D,
    MutualInformation,
    localMutualInformation,
    MattesMutualInformation,
    NormalizedMutualInformation,
    MIND_loss,
    NormalizedGradientFieldLoss,
)


class AffineReg3D(nn.Module):
    """3D affine registration with configurable degrees of freedom. This is an optimization-based approach, not deep learning-based.

    Args:
        vol_shape: Spatial shape tuple (D, H, W).
        dof: Degrees of freedom, one of: "affine", "rigid", "translation", "scaling".
        scales: Multi-scale factors (e.g., (0.25, 0.5, 1)).
        loss_funcs: Loss names per scale ("mse", "l1", "ncc", "fastncc", "pcc", "localcorrratio", "corrratio", "ssim3d", "mutualinformation", "localmutualinformation", "mind", "mattes", "nmi", "ngf").
        loss_weights: Optional loss weights per scale.
        mode: Sampling mode for SpatialTransformer.
        batch_size: Parameter batch size (defaults to 1).
        match_fixed: If True, pad/crop inputs to fixed image shape.
        pad_mode: Padding mode for size matching.
        pad_value: Constant value for padding (if pad_mode == "constant").
    """

    def __init__(
        self,
        vol_shape: Sequence[int],
        dof: str = "affine",
        scales: Sequence[float] = (1.0,),
        loss_funcs: Sequence[str] = ("ncc",),
        loss_weights: Optional[Sequence[float]] = None,
        mode: str = "bilinear",
        batch_size: int = 1,
        match_fixed: bool = True,
        pad_mode: str = "constant",
        pad_value: float = 0.0,
    ) -> None:
        super().__init__()
        if len(vol_shape) != 3:
            raise ValueError("vol_shape must be a 3-tuple (D, H, W).")
        self.vol_shape = tuple(int(s) for s in vol_shape)
        self.dof = dof.lower()
        self.scales = tuple(float(s) for s in scales)
        self.loss_funcs = tuple(loss_funcs)
        self.loss_weights = loss_weights
        self.mode = mode
        self.batch_size = int(batch_size)
        self.match_fixed = match_fixed
        self.pad_mode = pad_mode
        self.pad_value = float(pad_value)

        enable_rotation = self.dof in {"affine", "rigid"}
        enable_translation = self.dof in {"affine", "rigid", "translation", "scaling"}
        enable_scale = self.dof in {"affine", "scaling"}
        enable_shear = self.dof in {"affine"}

        if enable_rotation:
            self.rot = nn.Parameter(torch.zeros(self.batch_size, 3))
        else:
            self.register_buffer("rot", torch.zeros(self.batch_size, 3))

        if enable_translation:
            self.trans = nn.Parameter(torch.zeros(self.batch_size, 3))
        else:
            self.register_buffer("trans", torch.zeros(self.batch_size, 3))

        if enable_scale:
            self.log_scale = nn.Parameter(torch.zeros(self.batch_size, 3))
        else:
            self.register_buffer("log_scale", torch.zeros(self.batch_size, 3))

        if enable_shear:
            self.shear = nn.Parameter(torch.zeros(self.batch_size, 6))
        else:
            self.register_buffer("shear", torch.zeros(self.batch_size, 6))

        self._grid_cache: Dict[
            Tuple[Tuple[int, int, int], torch.device, torch.dtype], torch.Tensor
        ] = {}
        self._transformers: Dict[Tuple[int, int, int], SpatialTransformer] = {}

    def reset_parameters(self) -> None:
        """Reset learnable parameters to identity transform."""
        if isinstance(self.rot, nn.Parameter):
            nn.init.zeros_(self.rot)
        if isinstance(self.trans, nn.Parameter):
            nn.init.zeros_(self.trans)
        if isinstance(self.log_scale, nn.Parameter):
            nn.init.zeros_(self.log_scale)
        if isinstance(self.shear, nn.Parameter):
            nn.init.zeros_(self.shear)

    def _get_transformer(self, shape: Tuple[int, int, int]) -> SpatialTransformer:
        if shape not in self._transformers:
            self._transformers[shape] = SpatialTransformer(shape, mode=self.mode)
        return self._transformers[shape]

    def _get_grid(
        self,
        shape: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (shape, device, dtype)
        if key in self._grid_cache:
            return self._grid_cache[key]
        vectors = [torch.arange(0, s, device=device, dtype=dtype) for s in shape]
        try:
            grids = torch.meshgrid(vectors, indexing="ij")
        except TypeError:
            grids = torch.meshgrid(vectors)
        grid = torch.stack(grids, dim=0).unsqueeze(0)
        self._grid_cache[key] = grid
        return grid

    def _expand_params(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        def _expand(param: torch.Tensor) -> torch.Tensor:
            if param.shape[0] == batch_size:
                return param
            if param.shape[0] == 1:
                return param.expand(batch_size, -1)
            raise ValueError("Parameter batch size does not match input batch size.")

        rot = _expand(self.rot)
        trans = _expand(self.trans)
        log_scale = _expand(self.log_scale)
        shear = _expand(self.shear)
        return rot, trans, log_scale, shear

    @staticmethod
    def _rotation_matrices(rot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rx, ry, rz = rot[:, 0], rot[:, 1], rot[:, 2]
        ones = torch.ones_like(rx)
        zeros = torch.zeros_like(rx)

        rot_x = torch.stack(
            [
                torch.stack([ones, zeros, zeros], dim=1),
                torch.stack([zeros, torch.cos(rx), -torch.sin(rx)], dim=1),
                torch.stack([zeros, torch.sin(rx), torch.cos(rx)], dim=1),
            ],
            dim=2,
        )
        rot_y = torch.stack(
            [
                torch.stack([torch.cos(ry), zeros, torch.sin(ry)], dim=1),
                torch.stack([zeros, ones, zeros], dim=1),
                torch.stack([-torch.sin(ry), zeros, torch.cos(ry)], dim=1),
            ],
            dim=2,
        )
        rot_z = torch.stack(
            [
                torch.stack([torch.cos(rz), -torch.sin(rz), zeros], dim=1),
                torch.stack([torch.sin(rz), torch.cos(rz), zeros], dim=1),
                torch.stack([zeros, zeros, ones], dim=1),
            ],
            dim=2,
        )
        return rot_x, rot_y, rot_z

    def _affine_matrix(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        rot, trans, log_scale, shear = self._expand_params(batch_size)
        rot = rot.to(device=device, dtype=dtype)
        trans = trans.to(device=device, dtype=dtype)
        log_scale = log_scale.to(device=device, dtype=dtype)
        shear = shear.to(device=device, dtype=dtype)

        scale = torch.exp(log_scale)
        rot_x, rot_y, rot_z = self._rotation_matrices(rot)

        scale_mat = torch.stack(
            [
                torch.stack([scale[:, 0], torch.zeros_like(scale[:, 0]), torch.zeros_like(scale[:, 0])], dim=1),
                torch.stack([torch.zeros_like(scale[:, 1]), scale[:, 1], torch.zeros_like(scale[:, 1])], dim=1),
                torch.stack([torch.zeros_like(scale[:, 2]), torch.zeros_like(scale[:, 2]), scale[:, 2]], dim=1),
            ],
            dim=2,
        )

        shear_mat = torch.stack(
            [
                torch.stack([torch.ones_like(shear[:, 0]), shear[:, 0], shear[:, 1]], dim=1),
                torch.stack([shear[:, 2], torch.ones_like(shear[:, 0]), shear[:, 3]], dim=1),
                torch.stack([shear[:, 4], shear[:, 5], torch.ones_like(shear[:, 0])], dim=1),
            ],
            dim=2,
        )

        mat = torch.bmm(shear_mat, torch.bmm(scale_mat, torch.bmm(rot_z, torch.bmm(rot_y, rot_x))))
        mat = torch.cat([mat, trans.unsqueeze(2)], dim=-1)
        return mat

    def _affine_to_flow(
        self,
        shape: Tuple[int, int, int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        grid = self._get_grid(shape, device=device, dtype=dtype)
        grid_flat = grid.view(1, 3, -1).expand(batch_size, -1, -1)

        center = torch.tensor([(s - 1) / 2 for s in shape], device=device, dtype=dtype).view(1, 3, 1)
        grid_centered = grid_flat - center

        mat = self._affine_matrix(batch_size, device, dtype)
        linear = mat[:, :, :3]
        trans = mat[:, :, 3:].contiguous()

        transformed = torch.bmm(linear, grid_centered) + center + trans
        flow_flat = transformed - grid_flat
        flow = flow_flat.view(batch_size, 3, *shape)
        return flow

    def _affine_matrix_to_flow(
        self,
        affine: torch.Tensor,
        shape: Tuple[int, int, int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        invert: bool = False,
    ) -> torch.Tensor:
        if affine.dim() == 2:
            affine = affine.unsqueeze(0)
        if affine.shape[-2:] == (4, 4):
            affine_4x4 = affine
        elif affine.shape[-2:] == (3, 4):
            last_row = torch.tensor([0, 0, 0, 1], device=affine.device, dtype=affine.dtype)
            last_row = last_row.view(1, 1, 4).expand(affine.shape[0], -1, -1)
            affine_4x4 = torch.cat([affine, last_row], dim=1)
        else:
            raise ValueError("affine must have shape (B,3,4), (3,4), (B,4,4), or (4,4).")

        if invert:
            affine_4x4 = torch.inverse(affine_4x4)

        affine_4x4 = affine_4x4.to(device=device, dtype=dtype)
        linear = affine_4x4[:, :3, :3]
        trans = affine_4x4[:, :3, 3:].contiguous()

        grid = self._get_grid(shape, device=device, dtype=dtype)
        grid_flat = grid.view(1, 3, -1).expand(batch_size, -1, -1)
        center = torch.tensor([(s - 1) / 2 for s in shape], device=device, dtype=dtype).view(1, 3, 1)
        grid_centered = grid_flat - center

        transformed = torch.bmm(linear, grid_centered) + center + trans
        flow_flat = transformed - grid_flat
        flow = flow_flat.view(batch_size, 3, *shape)
        return flow

    def _resolve_loss(self, name: str):
        name = name.lower()
        if name == "mse":
            return nnf.mse_loss
        if name == "l1":
            return nnf.l1_loss
        if name == "ncc":
            return NCC_vxm()
        if name == "pcc":
            return PCC()
        if name == "fastncc":
            return FastNCC()
        if name == "localcorrratio":
            return LocalCorrRatio()
        if name == "corrratio":
            return CorrRatio()
        if name == "ssim3d":
            return SSIM3D()
        if name == "mutualinformation":
            return MutualInformation()
        if name == "localmutualinformation":
            return localMutualInformation()
        if name == "mattes":
            return MattesMutualInformation()
        if name in {"nmi", "normalizedmutualinformation"}:
            return NormalizedMutualInformation()
        if name in {"ngf", "normalizedgradientfield"}:
            return NormalizedGradientFieldLoss()
        if name == "mind":
            return MIND_loss()
        raise ValueError(f"Unsupported loss name: {name}")

    @staticmethod
    def _pad_or_crop(
        tensor: torch.Tensor,
        target_shape: Tuple[int, int, int],
        pad_mode: str = "constant",
        pad_value: float = 0.0,
    ) -> torch.Tensor:
        current_shape = tensor.shape[2:]
        cropped = tensor
        for dim, (cur, tgt) in enumerate(zip(current_shape, target_shape)):
            if cur > tgt:
                start = (cur - tgt) // 2
                end = start + tgt
                slices = [slice(None)] * cropped.dim()
                slices[dim + 2] = slice(start, end)
                cropped = cropped[tuple(slices)]
        pads: List[int] = []
        for cur, tgt in reversed(list(zip(cropped.shape[2:], target_shape))):
            if cur < tgt:
                pad_left = (tgt - cur) // 2
                pad_right = tgt - cur - pad_left
            else:
                pad_left = 0
                pad_right = 0
            pads.extend([pad_left, pad_right])
        if any(pads):
            cropped = nnf.pad(cropped, pads, mode=pad_mode, value=pad_value)
        return cropped

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        target_shape: Optional[Sequence[int]] = None,
    ):
        """Compute affine registration at full resolution.

        Args:
            moving: Moving image tensor (B, C, D, H, W).
            fixed: Fixed image tensor (B, C, D, H, W).

        Returns:
            Dict with keys: warped, flow, affine, loss, losses (single entry).
        """
        if moving.shape != fixed.shape:
            if not self.match_fixed and target_shape is None:
                raise ValueError("moving and fixed must have the same shape.")
        if moving.dim() != 5:
            raise ValueError("Expected 5D tensors (B, C, D, H, W).")

        if target_shape is None:
            target = tuple(fixed.shape[2:])
        else:
            if len(target_shape) != 3:
                raise ValueError("target_shape must be a 3-tuple (D, H, W).")
            target = tuple(int(s) for s in target_shape)

        if self.match_fixed or target_shape is not None:
            moving = self._pad_or_crop(moving, target, self.pad_mode, self.pad_value)
            fixed = self._pad_or_crop(fixed, target, self.pad_mode, self.pad_value)

        batch_size = moving.shape[0]
        shape = tuple(moving.shape[2:])
        flow = self._affine_to_flow(shape, batch_size, moving.device, moving.dtype)
        transformer = self._get_transformer(shape).to(moving.device)
        warped = transformer(moving, flow)

        loss_name = self.loss_funcs[0] if self.loss_funcs else "ncc"
        loss_fn = self._resolve_loss(loss_name)
        loss = loss_fn(fixed, warped)
        if self.loss_weights is not None:
            loss = loss * float(self.loss_weights[0])

        losses = [loss]

        affine = self._affine_matrix(batch_size, moving.device, moving.dtype)
        return {
            "warped": warped,
            "flow": flow,
            "affine": affine,
            "loss": loss,
            "losses": losses,
        }

    def apply_affine(
        self,
        moving: torch.Tensor,
        affine: torch.Tensor,
        invert: bool = False,
        target_shape: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Apply a given affine matrix to the moving image.

        Args:
            moving: Moving image tensor (B, C, D, H, W).
            affine: Affine matrix with shape (3,4), (B,3,4), (4,4), or (B,4,4).
            invert: If True, apply the inverse affine.
            target_shape: Optional target shape for padding/cropping.

        Returns:
            Warped image tensor (B, C, D, H, W).
        """
        if moving.dim() != 5:
            raise ValueError("Expected 5D tensor (B, C, D, H, W).")

        if target_shape is None:
            target = tuple(moving.shape[2:])
        else:
            if len(target_shape) != 3:
                raise ValueError("target_shape must be a 3-tuple (D, H, W).")
            target = tuple(int(s) for s in target_shape)

        if self.match_fixed or target_shape is not None:
            moving = self._pad_or_crop(moving, target, self.pad_mode, self.pad_value)

        batch_size = moving.shape[0]
        shape = tuple(moving.shape[2:])
        flow = self._affine_matrix_to_flow(
            affine=affine,
            shape=shape,
            batch_size=batch_size,
            device=moving.device,
            dtype=moving.dtype,
            invert=invert,
        )
        transformer = self._get_transformer(shape).to(moving.device)
        warped = transformer(moving, flow)
        return warped

    def _normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize image intensities to [0, 1] range."""
        img_min = torch.quantile(img, 0.02)
        img_max = torch.quantile(img, 0.98)
        normalized = (img - img_min) / (img_max - img_min + 1e-8)
        normalized = torch.clamp(normalized, 0.0, 1.0)
        return normalized

    def _estimate_affine_from_flow(
        self,
        flow: torch.Tensor,
        sample_stride: int = 4,
    ) -> torch.Tensor:
        """Estimate a 3x4 affine matrix from a displacement field.

        Args:
            flow: Displacement field (B, 3, D, H, W).
            sample_stride: Subsampling stride for least-squares fitting.

        Returns:
            Affine matrix tensor (B, 3, 4) in voxel coordinates.
        """
        if flow.dim() != 5:
            raise ValueError("flow must have shape (B, 3, D, H, W)")
        batch_size, _, d, h, w = flow.shape
        device = flow.device
        dtype = flow.dtype

        zs = torch.arange(0, d, device=device, dtype=dtype)
        ys = torch.arange(0, h, device=device, dtype=dtype)
        xs = torch.arange(0, w, device=device, dtype=dtype)
        grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing="ij")
        grid = torch.stack([grid_z, grid_y, grid_x], dim=0)

        grid = grid[:, ::sample_stride, ::sample_stride, ::sample_stride]
        flow_sub = flow[:, :, ::sample_stride, ::sample_stride, ::sample_stride]

        grid_flat = grid.reshape(3, -1).T  # (N, 3)
        ones = torch.ones((grid_flat.shape[0], 1), device=device, dtype=dtype)
        x_aug = torch.cat([grid_flat, ones], dim=1)  # (N, 4)

        affines = []
        for b in range(batch_size):
            y = (grid_flat + flow_sub[b].reshape(3, -1).T)  # (N, 3)
            sol = torch.linalg.lstsq(x_aug, y).solution  # (4, 3)
            affines.append(sol.T)
        return torch.stack(affines, dim=0)

    def _init_params_from_affine(
        self,
        affine: torch.Tensor,
        shape: Tuple[int, int, int],
        steps: int = 25,
        lr: float = 1e-2,
    ) -> None:
        """Initialize affine parameters from a target affine matrix.

        Args:
            affine: Affine matrix (B, 3, 4) in voxel coordinates.
            shape: Spatial shape (D, H, W).
            steps: Optimization steps for parameter fitting.
            lr: Learning rate for parameter fitting.
        """
        if affine.dim() == 2:
            affine = affine.unsqueeze(0)
        batch_size = affine.shape[0]
        device = affine.device
        dtype = affine.dtype

        linear_target = affine[:, :, :3]
        trans_target = affine[:, :, 3]
        center = torch.tensor([(s - 1) / 2 for s in shape], device=device, dtype=dtype).view(1, 3)
        trans_centered = trans_target + torch.bmm(linear_target, center.unsqueeze(-1)).squeeze(-1) - center

        if isinstance(self.trans, nn.Parameter):
            with torch.no_grad():
                self.trans.copy_(trans_centered)

        params = []
        for p in (self.rot, self.log_scale, self.shear):
            if isinstance(p, nn.Parameter) and p.requires_grad:
                params.append(p)

        if not params:
            return

        optimizer = torch.optim.Adam(params, lr=lr)
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            mat = self._affine_matrix(batch_size, device=device, dtype=dtype)
            loss = nnf.mse_loss(mat[:, :, :3], linear_target)
            loss.backward()
            optimizer.step()

    def _convex_mind_affine_init(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        grid_sp: int = 6,
        disp_hw: int = 4,
        mind_r: int = 1,
        mind_d: int = 2,
        sample_stride: int = 4,
    ) -> None:
        """Initialize affine parameters using convex MIND-based coarse flow."""
        if moving.dim() != 5 or fixed.dim() != 5:
            raise ValueError("Expected 5D tensors (B, C, D, H, W).")

        device = moving.device
        dtype = moving.dtype
        shape = tuple(moving.shape[2:])

        with torch.no_grad():
            mind_fix = MINDSSC(fixed, radius=mind_r, dilation=mind_d, device=device)
            mind_mov = MINDSSC(moving, radius=mind_r, dilation=mind_d, device=device)

            mind_fix_s = nnf.avg_pool3d(mind_fix, grid_sp, stride=grid_sp)
            mind_mov_s = nnf.avg_pool3d(mind_mov, grid_sp, stride=grid_sp)
            n_ch = mind_fix_s.shape[1]

            ssd, ssd_argmin = correlate(mind_fix_s, mind_mov_s, disp_hw, grid_sp, shape, n_ch)

            disp_mesh_t = nnf.affine_grid(
                disp_hw * torch.eye(3, 4, device=device, dtype=dtype).unsqueeze(0),
                (1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1),
                align_corners=True,
            ).permute(0, 4, 1, 2, 3).reshape(3, -1, 1)

            disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, shape)
            disp_hr = nnf.interpolate(disp_soft, size=shape, mode="trilinear", align_corners=False) * grid_sp

        affine_init = self._estimate_affine_from_flow(disp_hr, sample_stride=sample_stride)
        self._init_params_from_affine(affine_init, shape)

    def optimize(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        target_shape: Optional[Sequence[int]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizer_name: str = "lbfgs",
        lr: float = 1e-2,
        steps: int = 200,
        steps_per_scale: Optional[Sequence[int]] = None,
        return_history: bool = False,
        verbose: bool = False,
        normalize: bool = True,
        lbfgs_history_size: int = 10,
        mind_init: bool = False,
        mind_init_grid_sp: int = 6,
        mind_init_disp_hw: int = 4,
        mind_init_mind_r: int = 1,
        mind_init_mind_d: int = 2,
        mind_init_sample_stride: int = 4,
    ):
        """Optimize affine parameters to align moving to fixed.

        Args:
            moving: Moving image tensor (B, C, D, H, W).
            fixed: Fixed image tensor (B, C, D, H, W).
            target_shape: Optional target spatial shape for padding/cropping.
            optimizer: Optional optimizer. If None, optimizer_name is used.
            optimizer_name: "lbfgs" or "adam" when optimizer is None.
            lr: Learning rate for optimizer if optimizer is None.
            steps: Number of optimization steps (used if steps_per_scale is None).
            steps_per_scale: Optional per-scale step counts matching self.scales.
            return_history: Whether to return loss history.
            normalize: Whether to normalize images before optimization.
            lbfgs_history_size: History size for LBFGS.
            mind_init: Whether to run a short MIND-based pre-alignment stage.
            mind_init_grid_sp: Grid spacing for convex MIND initialization.
            mind_init_disp_hw: Displacement half-width for convex MIND initialization.
            mind_init_mind_r: MIND radius for convex MIND initialization.
            mind_init_mind_d: MIND dilation for convex MIND initialization.
            mind_init_sample_stride: Subsampling stride when fitting affine from convex flow.

        Returns:
            Tuple of (output_dict, loss_history) if return_history else output_dict.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No learnable parameters found for the selected DOF.")
        if optimizer is None:
            optimizer_name = optimizer_name.lower()
            if optimizer_name == "lbfgs":
                optimizer = torch.optim.LBFGS(
                    params,
                    lr=lr,
                    max_iter=1,
                    history_size=lbfgs_history_size,
                    line_search_fn="strong_wolfe",
                )
            elif optimizer_name == "adam":
                optimizer = torch.optim.Adam(params, lr=lr)
            else:
                raise ValueError("optimizer_name must be 'lbfgs' or 'adam'.")

        loss_history: List[float] = []
        if target_shape is None:
            target = tuple(fixed.shape[2:])
        else:
            if len(target_shape) != 3:
                raise ValueError("target_shape must be a 3-tuple (D, H, W).")
            target = tuple(int(s) for s in target_shape)
        if self.match_fixed or target_shape is not None:
            moving = self._pad_or_crop(moving, target, self.pad_mode, self.pad_value)
            fixed = self._pad_or_crop(fixed, target, self.pad_mode, self.pad_value)
        moving_ = moving.clone()
        if normalize:
            moving = self._normalize_image(moving)
            fixed = self._normalize_image(fixed)
        def _run_optimization(
            scales: Sequence[float],
            steps_per_scale_local: Optional[Sequence[int]],
            loss_funcs_local: Sequence[str],
            optimizer_local: torch.optim.Optimizer,
            log_prefix: str,
        ) -> None:
            if steps_per_scale_local is not None and len(steps_per_scale_local) != len(scales):
                raise ValueError("steps_per_scale must match number of scales.")

            for scale_idx, scale in enumerate(scales):
                scale_steps = steps_per_scale_local[scale_idx] if steps_per_scale_local is not None else steps
                for step in range(int(scale_steps)):
                    if scale == 1.0:
                        mov = moving
                        fix = fixed
                    else:
                        mov = nnf.interpolate(moving, scale_factor=scale, mode="trilinear", align_corners=False)
                        fix = nnf.interpolate(fixed, scale_factor=scale, mode="trilinear", align_corners=False)
                        mov = self._pad_or_crop(mov, tuple(fix.shape[2:]), self.pad_mode, self.pad_value)

                    shape = tuple(mov.shape[2:])
                    transformer = self._get_transformer(shape).to(mov.device)
                    loss_name = loss_funcs_local[min(scale_idx, len(loss_funcs_local) - 1)]
                    loss_fn = self._resolve_loss(loss_name)

                    def closure():
                        optimizer_local.zero_grad(set_to_none=True)
                        flow = self._affine_to_flow(shape, mov.shape[0], mov.device, mov.dtype)
                        warped = transformer(mov, flow)
                        loss_val = loss_fn(fix, warped)
                        loss_val.backward()
                        return loss_val

                    if isinstance(optimizer_local, torch.optim.LBFGS):
                        loss = optimizer_local.step(closure)
                    else:
                        loss = closure()
                        optimizer_local.step()

                    if return_history:
                        loss_history.append(float(loss.detach().cpu()))
                    if verbose and (step == 0 or step == scale_steps - 1 or (step + 1) % 10 == 0):
                        print(
                            f"{log_prefix} scale {scale_idx + 1}/{len(scales)} step {step + 1}/{scale_steps} "
                            f"loss={loss.item():.6f}"
                        )

        if mind_init:
            self._convex_mind_affine_init(
                moving=moving,
                fixed=fixed,
                grid_sp=mind_init_grid_sp,
                disp_hw=mind_init_disp_hw,
                mind_r=mind_init_mind_r,
                mind_d=mind_init_mind_d,
                sample_stride=mind_init_sample_stride,
            )

        _run_optimization(
            scales=self.scales,
            steps_per_scale_local=steps_per_scale,
            loss_funcs_local=self.loss_funcs,
            optimizer_local=optimizer,
            log_prefix="[AffineReg3D]",
        )

        output = self.forward(moving_, fixed, target_shape=target_shape)

        if return_history:
            return output, loss_history
        return output


class PreAffineToTemplate(nn.Module):
    '''
    Affine pre-alignment to a fixed template before deformable registration.

    Args:
        mode: Interpolation mode for SpatialTransformer.
        template_type: Template name to load ("lumir" or "mni").
        batch_size: Batch size for affine parameters.
        dof: Degrees of freedom for affine model.
        scales: Multi-scale factors for loss computation.
        loss_funcs: Loss names per scale.
        device: Device for the affine model and template.
    '''

    def __init__(
        self,
        mode: str = "bilinear",
        template_type: str = "lumir",
        batch_size: int = 1,
        dof: str = "affine",
        scales: Sequence[float] = (0.25, 0.5, 1.0),
        loss_funcs: Sequence[str] = ("mse", "mse", "mse"),
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if template_type == "lumir":
            template_name = "LUMIR_template.nii.gz"
        elif template_type == "mni":
            template_name = "mni_icbm152_2009c_t1_1mm_masked_img.nii.gz"
        else:
            template_name = template_type
        self.atlas_nib = nib.load(self._resolve_template_path(template_name))
        self.atlas = torch.from_numpy(self.atlas_nib.get_fdata()).unsqueeze(0).unsqueeze(0).float()
        self.reg_model = AffineReg3D(
            vol_shape=self.atlas.shape[2:],
            dof=dof,
            scales=scales,
            loss_funcs=loss_funcs,
            mode=mode,
            batch_size=batch_size,
            match_fixed=True,
            pad_mode="constant",
            pad_value=0.0,
        ).to(device)

    def _resolve_template_path(self, name: str) -> Path:
        """Resolve template path from package resources with source-tree fallback."""
        try:
            package_root = resources.files("MIR")
            candidate = package_root.joinpath("templates", name)
            if candidate.is_file():
                return Path(candidate)
        except Exception:
            pass
        repo_templates = Path(__file__).resolve().parents[2] / "templates" / name
        if repo_templates.exists():
            return repo_templates
        raise FileNotFoundError(f"Template not found: {name}")

    def forward(self, x: torch.Tensor, optimize: bool = True, verbose: bool = False):
        '''
        Forward pass of the affine registration module
        Args:
            x: Input image tensor (B, C, H, W, D)
        Returns:
            Affine registered image tensor (B, C, H, W, D)
        '''
        if x.ndim != 5:
            raise ValueError("Input must have shape (B,C,H,W,D)")
        batch_size = x.shape[0]

        fixed = self.atlas.repeat(batch_size, 1, 1, 1, 1).to(x.device)
        if optimize:
            output = self.reg_model.optimize(x, fixed, verbose=verbose, steps_per_scale=[50, 50, 50])
        else:
            output = self.reg_model(x, fixed)
        return output