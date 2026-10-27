"""Global regularizers for deformation regularization."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as nnf
import MIR.utils.registration_utils as reg_utils


def _apply_penalty(tensor: torch.Tensor, penalty: str) -> torch.Tensor:
    """Apply the configured penalty while preserving legacy behavior."""
    if penalty == 'l2':
        return tensor.pow(2)
    return tensor


def _gradient_loss_2d(y_pred: torch.Tensor, penalty: str) -> torch.Tensor:
    """Compute the mean 2D finite-difference gradient penalty."""
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = _apply_penalty(dx, penalty)
    dy = _apply_penalty(dy, penalty)
    return (torch.mean(dx) + torch.mean(dy)) / 2.0


def _gradient_loss_3d(y_pred: torch.Tensor, penalty: str) -> torch.Tensor:
    """Compute the mean 3D finite-difference gradient penalty."""
    dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
    dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
    dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

    dx = _apply_penalty(dx, penalty)
    dy = _apply_penalty(dy, penalty)
    dz = _apply_penalty(dz, penalty)
    return (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3.0


class Grad2D(torch.nn.Module):
    """2D gradient (diffusion) loss on a displacement field.

    Mean of the forward differences along each spatial axis (``|d|`` or ``d^2``), averaged over the two axes.

    Args:
        penalty: ``'l2'`` squares the differences; any other value uses absolute differences (``'l1'``).
        loss_mult: Optional multiplier applied to the final loss.
    """

    def __init__(self, penalty: str = 'l1', loss_mult: Optional[float] = None):
        super().__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred: torch.Tensor, y_true) -> torch.Tensor:
        """Compute 2D gradient regularization loss.

        Args:
            y_pred: Predicted displacement/field tensor (B, C, H, W).
            y_true: Unused placeholder for API compatibility.

        Returns:
            Scalar gradient penalty.

        """
        # del y_true
        grad = _gradient_loss_2d(y_pred, self.penalty)

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class Grad3d(torch.nn.Module):
    """3D gradient (diffusion) loss on a displacement field.

    Mean of the forward differences along each spatial axis (``|d|`` or ``d^2``), averaged over the three axes.

    Args:
        penalty: ``'l2'`` squares the differences; any other value uses absolute differences (``'l1'``).
        loss_mult: Optional multiplier applied to the final loss.
    """

    def __init__(self, penalty: str = 'l1', loss_mult: Optional[float] = None):
        super().__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred: torch.Tensor, y_true=None) -> torch.Tensor:
        """Compute 3D gradient regularization loss.

        Args:
            y_pred: Predicted displacement/field tensor (B, C, H, W, D), e.g. a flow (B, 3, H, W, D).
            y_true: Unused placeholder for API compatibility.

        Returns:
            Scalar gradient penalty.

        """
        # del y_true
        grad = _gradient_loss_3d(y_pred, self.penalty)

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class Grad3DiTV(torch.nn.Module):
    """3D isotropic total-variation (TV) loss.

    Computes ``mean(sqrt(dx^2 + dy^2 + dz^2 + 1e-6)) / 3`` from forward differences evaluated on the common
    ``[1:, 1:, 1:]`` sub-grid. The constructor takes no arguments.
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true) -> torch.Tensor:
        """Compute isotropic total-variation loss in 3D.

        Args:
            y_pred: Predicted displacement/field tensor (B, C, H, W, D).
            y_true: Unused placeholder for API compatibility.

        Returns:
            Scalar TV penalty.

        """
        # del y_true
        dy = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, :-1, 1:, 1:])
        dx = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, 1:, :-1, 1:])
        dz = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, 1:, 1:, :-1])

        dy = dy.pow(2)
        dx = dx.pow(2)
        dz = dz.pow(2)
        return torch.mean(torch.sqrt(dx + dy + dz + 1e-6)) / 3.0


class DisplacementRegularizer(torch.nn.Module):
    """Compute displacement-field regularization energies.

    Derivatives are central differences on the interior voxels (one voxel is dropped at every border).

    Args:
        energy_type: ``'bending'`` (bending energy), ``'gradient-l2'`` (mean squared first derivatives) or
            ``'gradient-l1'`` (mean absolute first derivatives).
    """

    def __init__(self, energy_type: str):
        super().__init__()
        self.energy_type = energy_type

    @staticmethod
    def gradient_dx(fv: torch.Tensor) -> torch.Tensor:
        """Central difference along the first spatial axis.

        Args:
            fv: Single-component field (B, H, W, D).

        Returns:
            Derivative on the interior grid (B, H-2, W-2, D-2).
        """
        return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

    @staticmethod
    def gradient_dy(fv: torch.Tensor) -> torch.Tensor:
        """Central difference along the second spatial axis.

        Args:
            fv: Single-component field (B, H, W, D).

        Returns:
            Derivative on the interior grid (B, H-2, W-2, D-2).
        """
        return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

    @staticmethod
    def gradient_dz(fv: torch.Tensor) -> torch.Tensor:
        """Central difference along the third spatial axis.

        Args:
            fv: Single-component field (B, H, W, D).

        Returns:
            Derivative on the interior grid (B, H-2, W-2, D-2).
        """
        return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

    @staticmethod
    def gradient_txyz(txyz: torch.Tensor, gradient_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Apply ``gradient_fn`` to each of the three components of a vector field.

        Args:
            txyz: Vector field (B, 3, H, W, D).
            gradient_fn: One of :meth:`gradient_dx`, :meth:`gradient_dy`, :meth:`gradient_dz`.

        Returns:
            Stacked derivatives (B, 3, H-2, W-2, D-2).
        """
        return torch.stack([gradient_fn(txyz[:, axis, ...]) for axis in (0, 1, 2)], dim=1)

    def compute_gradient_norm(self, displacement: torch.Tensor, flag_l1: bool = False) -> torch.Tensor:
        """Compute L1/L2 gradient norm of a displacement field.

        Args:
            displacement: Tensor (B, 3, H, W, D).
            flag_l1: If True, uses absolute derivatives; otherwise squared derivatives.

        Returns:
            Scalar: mean over all elements of the summed (absolute or squared) x/y/z derivatives, divided by 3.

        """
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        if flag_l1:
            norms = torch.abs(dTdx) + torch.abs(dTdy) + torch.abs(dTdz)
        else:
            norms = dTdx.pow(2) + dTdy.pow(2) + dTdz.pow(2)
        return torch.mean(norms) / 3.0

    def compute_bending_energy(self, displacement: torch.Tensor) -> torch.Tensor:
        """Compute bending energy of a displacement field.

        Args:
            displacement: Tensor (B, 3, H, W, D).

        Returns:
            Scalar bending energy: mean of ``u_xx^2 + u_yy^2 + u_zz^2 + 2 (u_xy^2 + u_xz^2 + u_yz^2)``.

        """
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        dTdxx = self.gradient_txyz(dTdx, self.gradient_dx)
        dTdyy = self.gradient_txyz(dTdy, self.gradient_dy)
        dTdzz = self.gradient_txyz(dTdz, self.gradient_dz)
        dTdxy = self.gradient_txyz(dTdx, self.gradient_dy)
        dTdyz = self.gradient_txyz(dTdy, self.gradient_dz)
        dTdxz = self.gradient_txyz(dTdx, self.gradient_dz)
        return torch.mean(
            dTdxx.pow(2) + dTdyy.pow(2) + dTdzz.pow(2) + 2 * dTdxy.pow(2) + 2 * dTdxz.pow(2) + 2 * dTdyz.pow(2)
        )

    def forward(self, disp: torch.Tensor, _) -> torch.Tensor:
        """Compute the configured regularization energy.

        Args:
            disp: Displacement field (B, 3, H, W, D).
            _: Unused placeholder for API compatibility (e.g. ``None`` or the fixed image).

        Returns:
            Scalar energy.

        Raises:
            Exception: If ``energy_type`` is not recognised.
        """
        if self.energy_type == 'bending':
            energy = self.compute_bending_energy(disp)
        elif self.energy_type == 'gradient-l2':
            energy = self.compute_gradient_norm(disp)
        elif self.energy_type == 'gradient-l1':
            energy = self.compute_gradient_norm(disp, flag_l1=True)
        else:
            raise Exception('Not recognised local regulariser!')
        return energy


class GradICON3d(torch.nn.Module):
    """Gradient-ICON loss for 3D displacement fields.

    Composes the two flows on the voxel grid (``flow_inv + warp(flow_fwd, flow_inv)``) and applies the finite-difference
    gradient penalty of :class:`Grad3d` to the composed displacement, which is zero when the Jacobian of the composed
    map is the identity.

    Args:
        flow_shape: Spatial shape (H, W, D) of the flows, used to build the spatial transformer.
        penalty: ``'l1'`` or ``'l2'``.
        loss_mult: Optional multiplier applied to the final loss.
        both_dirs: If True, also penalise the reverse composition and average both directions.
        device: Device on which the spatial transformer grid is created.
    """

    def __init__(
        self,
        flow_shape: Sequence[int],
        penalty: str = 'l2',
        loss_mult: Optional[float] = None,
        both_dirs: bool = False,
        device: Union[str, torch.device] = 'cpu',
    ):
        """Initialize the GradICON penalty module."""
        super().__init__()
        if penalty not in ('l1', 'l2'):
            raise ValueError("penalty must be 'l1' or 'l2'")
        self.stn = reg_utils.SpatialTransformer(flow_shape).to(device)
        self.penalty = penalty
        self.loss_mult = loss_mult
        self.both_dirs = both_dirs

    @staticmethod
    def _grad3d(disp, penalty):
        """Compute a finite-difference gradient loss for one displacement."""
        return _gradient_loss_3d(disp, penalty)

    def forward(self, flow_fwd: torch.Tensor, flow_inv: torch.Tensor) -> torch.Tensor:
        """Compute the GradICON penalty for forward and inverse flows.

        Args:
            flow_fwd: Forward displacement field (B, 3, H, W, D) in voxels.
            flow_inv: Inverse (backward) displacement field (B, 3, H, W, D) in voxels.

        Returns:
            Scalar penalty.
        """
        comp_f = flow_inv + self.stn(flow_fwd, flow_inv)
        loss = self._grad3d(comp_f, self.penalty)

        if self.both_dirs:
            comp_b = flow_fwd + self.stn(flow_inv, flow_fwd)
            loss = 0.5 * (loss + self._grad3d(comp_b, self.penalty))

        if self.loss_mult is not None:
            loss *= self.loss_mult
        return loss


class GradICONExact3d(torch.nn.Module):
    """Paper-faithful Gradient-ICON for 3D flows.

    At every call a random subset of ``N / 8`` voxel locations is drawn, the Jacobian of the composed map is estimated
    there by forward finite differences (step ``1e-3 * (size - 1)`` voxels) with ``grid_sample``, and the mean
    (squared or absolute) Frobenius deviation from the identity is returned.

    Args:
        vol_shape: Spatial shape of the flows (three ints).
        penalty: ``'l1'`` (sum of absolute deviations) or ``'l2'`` (squared Frobenius norm).
        both_dirs: If True, also penalise the reverse composition and average both directions.
        device: Device for the sampling grid and spatial transformer.
    """

    def __init__(
        self,
        vol_shape: Sequence[int],
        penalty: str = 'l2',
        both_dirs: bool = False,
        device: Union[str, torch.device] = 'cpu',
    ):
        super().__init__()
        if penalty not in ('l1', 'l2'):
            raise ValueError("penalty must be 'l1' or 'l2'")
        self.D, self.H, self.W = vol_shape
        self.penalty = penalty
        self.both_dirs = both_dirs
        self.device = device

        self.stn = reg_utils.SpatialTransformer(vol_shape).to(device)

        # finite-difference step per point coordinate; points are stored as (x, y, z) = (axis 2, axis 1, axis 0)
        self.dx_vox = (
            torch.tensor(
                [size - 1 for size in reversed(tuple(vol_shape))],
                dtype=torch.float32,
                device=device,
            )
            * 1e-3
        )

        z = torch.linspace(0, self.D - 1, self.D, device=device)
        y = torch.linspace(0, self.H - 1, self.H, device=device)
        x = torch.linspace(0, self.W - 1, self.W, device=device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        self.grid_full = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)
        self.Nsub = self.grid_full.size(0) // 8

        self.register_buffer('eye3', torch.eye(3))

    @staticmethod
    def _fro(diff, penalty):
        if penalty == 'l1':
            return diff.abs().sum((-2, -1))
        return diff.pow(2).sum((-2, -1))

    def _compose_disp(self, flow_fwd, flow_inv):
        """Return the displacement of the composed transform on the voxel grid."""
        return flow_inv + self.stn(flow_fwd, flow_inv)

    def _jacobian_samples(self, disp, pts_vox):
        """Sample a finite-difference Jacobian at random voxel locations."""
        B, N, _ = pts_vox.shape

        def sample(f, p):
            p_norm = p.clone()
            p_norm[..., 0] = 2 * (p[..., 0] / (self.W - 1) - 0.5)
            p_norm[..., 1] = 2 * (p[..., 1] / (self.H - 1) - 0.5)
            p_norm[..., 2] = 2 * (p[..., 2] / (self.D - 1) - 0.5)
            g = p_norm.view(B, N, 1, 1, 3)
            v = nnf.grid_sample(
                f,
                g,
                align_corners=True,  # matches the normalisation above (p = 0 and p = size - 1 map to -1 and +1)
                mode='bilinear',
                padding_mode='border',
            )
            # displacement channels follow the tensor axes (0, 1, 2); reorder to the point order (axis 2, 1, 0)
            return v.view(B, 3, N).permute(0, 2, 1)[..., [2, 1, 0]]

        x = pts_vox
        phi_x = x + sample(disp, x)

        grads = []
        for axis, dx in enumerate(self.dx_vox):
            x_shift = x.clone()
            x_shift[..., axis] += dx
            phi_shift = x_shift + sample(disp, x_shift)
            grads.append((phi_shift - phi_x) / dx)

        return torch.stack(grads, dim=-1)

    def forward(self, flow_fwd: torch.Tensor, flow_inv: torch.Tensor) -> torch.Tensor:
        """Compute the exact GradICON penalty for sampled voxel locations.

        Args:
            flow_fwd: Forward displacement field (B, 3, *vol_shape) in voxels.
            flow_inv: Inverse (backward) displacement field (B, 3, *vol_shape) in voxels.

        Returns:
            Scalar penalty averaged over the batch and sampled locations.
        """
        B = flow_fwd.size(0)

        idx = torch.randperm(self.grid_full.size(0), device=self.device)[: self.Nsub]
        pts0 = self.grid_full[idx].unsqueeze(0).repeat(B, 1, 1)

        comp_disp = self._compose_disp(flow_fwd, flow_inv)

        J = self._jacobian_samples(comp_disp, pts0)
        diff = J - self.eye3.to(J.device)
        loss = self._fro(diff, self.penalty).mean()

        if self.both_dirs:
            comp_disp_b = self._compose_disp(flow_inv, flow_fwd)
            J_b = self._jacobian_samples(comp_disp_b, pts0)
            diff_b = J_b - self.eye3
            loss = 0.5 * (loss + self._fro(diff_b, self.penalty).mean())

        return loss
