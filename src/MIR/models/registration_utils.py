"""Model-level registration utilities (transformers and geometry)."""

from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as nnf


class SpatialTransformer(nn.Module):
    """N-D spatial transformer that warps an image with a voxel displacement field.

    Adapted from https://github.com/voxelmorph/voxelmorph. An identity sampling grid of shape ``[1, ndim, *size]``
    (float32) is stored as a non-persistent buffer; displacements are added to it and the result is sampled with
    ``torch.nn.functional.grid_sample(align_corners=True)``.

    Args:
        size: Spatial size of the input tensor, ``(H, W)`` or ``(H, W, D)``.
        mode: Interpolation mode passed to ``grid_sample``, e.g. ``'bilinear'`` or ``'nearest'``.
    """

    def __init__(self, size: Sequence[int], mode: str = 'bilinear'):
        """Initialize a spatial transformer.

        Args:
            size: Spatial size tuple (H, W[, D]).
            mode: Interpolation mode ('bilinear' or 'nearest').

        """
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid, persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints saved before the sampling grid became a non-persistent buffer still contain it. The grid is
        # rebuilt from ``size``, so a stored copy is dropped; this keeps ``load_state_dict(strict=True)`` working for
        # those checkpoints (e.g. the public TransMorphTVF LUMIR24 weights).
        state_dict.pop(prefix + 'grid', None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, src: torch.Tensor, flow: torch.Tensor, padding_mode: str = 'zeros') -> torch.Tensor:
        """Warp a source tensor with a displacement field.

        Args:
            src: Source tensor ``[B, C, H, W]`` or ``[B, C, H, W, D]``.
            flow: Displacement field in voxels, ``[B, ndim, H, W]`` or ``[B, ndim, H, W, D]``; channel ``i`` displaces
                spatial axis ``i``.
            padding_mode: Padding mode for out-of-bounds samples passed to ``grid_sample`` (``'zeros'``,
                ``'border'`` or ``'reflection'``).

        Returns:
            Warped tensor with the batch/channel size of ``src`` and the spatial size of ``flow``.

        """
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode, padding_mode=padding_mode)


class VecInt(nn.Module):
    """Integrates a stationary velocity field via scaling and squaring.

    Args:
        inshape: Spatial shape of the field, ``(H, W)`` or ``(H, W, D)``.
        nsteps: Number of squaring steps (must be >= 0); the field is first scaled by ``1 / 2**nsteps``.

    """

    def __init__(self, inshape: Sequence[int], nsteps: int):
        """Initialize vector field integrator.

        Args:
            inshape: Spatial shape of the field.
            nsteps: Number of scaling-and-squaring steps.

        """
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2**self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        """Integrate a vector field via scaling and squaring.

        Args:
            vec: Velocity field tensor ``[B, ndim, *inshape]`` in voxel units.

        Returns:
            Integrated displacement field with the same shape as ``vec``.

        """
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class AffineTransformer(nn.Module):
    """3-D affine transformer built on ``affine_grid``/``grid_sample`` (``align_corners=False``).

    Args:
        mode: Interpolation mode passed to ``grid_sample``, e.g. ``'bilinear'`` or ``'nearest'``.
    """

    def __init__(self, mode: str = 'bilinear'):
        """Initialize affine transformer.

        Args:
            mode: Interpolation mode ('bilinear' or 'nearest').

        """
        super().__init__()
        self.mode = mode

    def apply_affine(self, src: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
        """Apply an affine matrix to a source volume.

        Args:
            src: Source tensor ``[B, C, H, W, D]``.
            mat: Affine matrix ``[B, 3, 4]`` in the normalized ``[-1, 1]`` coordinates used by ``affine_grid``.

        Returns:
            Warped tensor ``[B, C, H, W, D]``.

        """
        grid = nnf.affine_grid(mat, [src.shape[0], 3, src.shape[2], src.shape[3], src.shape[4]], align_corners=False)
        return nnf.grid_sample(src, grid, align_corners=False, mode=self.mode)

    def forward(
        self,
        src: torch.Tensor,
        affine: torch.Tensor,
        scale: torch.Tensor,
        translate: torch.Tensor,
        shear: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply composed affine parameters to a volume.

        The linear part is ``Shear @ Scale @ Rz @ Ry @ Rx``; the translation is appended as the last column.

        Args:
            src: Source tensor ``[B, C, H, W, D]``.
            affine: Rotation angles in radians about x, y, z, ``[B, 3]``.
            scale: Scale factors, ``[B, 3]``.
            translate: Translations in normalized ``[-1, 1]`` grid units, ``[B, 3]``.
            shear: Shear angles in radians (``xy, xz, yx, yz, zx, zy``; the matrix uses their tangents), ``[B, 6]``.

        Returns:
            Tuple ``(warped, mat, inv_mat)``: the warped tensor ``[B, C, H, W, D]``, the affine matrix ``[B, 3, 4]``
            and its inverse ``[B, 3, 4]``.

        """
        theta_x = affine[:, 0]
        theta_y = affine[:, 1]
        theta_z = affine[:, 2]
        scale_x = scale[:, 0]
        scale_y = scale[:, 1]
        scale_z = scale[:, 2]
        trans_x = translate[:, 0]
        trans_y = translate[:, 1]
        trans_z = translate[:, 2]
        shear_xy = shear[:, 0]
        shear_xz = shear[:, 1]
        shear_yx = shear[:, 2]
        shear_yz = shear[:, 3]
        shear_zx = shear[:, 4]
        shear_zy = shear[:, 5]

        rot_mat_x = torch.stack(
            [
                torch.stack([torch.ones_like(theta_x), torch.zeros_like(theta_x), torch.zeros_like(theta_x)], dim=1),
                torch.stack([torch.zeros_like(theta_x), torch.cos(theta_x), -torch.sin(theta_x)], dim=1),
                torch.stack([torch.zeros_like(theta_x), torch.sin(theta_x), torch.cos(theta_x)], dim=1),
            ],
            dim=2,
        ).to(theta_x.device)
        rot_mat_y = torch.stack(
            [
                torch.stack([torch.cos(theta_y), torch.zeros_like(theta_y), torch.sin(theta_y)], dim=1),
                torch.stack([torch.zeros_like(theta_y), torch.ones_like(theta_x), torch.zeros_like(theta_x)], dim=1),
                torch.stack([-torch.sin(theta_y), torch.zeros_like(theta_y), torch.cos(theta_y)], dim=1),
            ],
            dim=2,
        ).to(theta_x.device)
        rot_mat_z = torch.stack(
            [
                torch.stack([torch.cos(theta_z), -torch.sin(theta_z), torch.zeros_like(theta_y)], dim=1),
                torch.stack([torch.sin(theta_z), torch.cos(theta_z), torch.zeros_like(theta_y)], dim=1),
                torch.stack([torch.zeros_like(theta_y), torch.zeros_like(theta_y), torch.ones_like(theta_x)], dim=1),
            ],
            dim=2,
        ).to(theta_x.device)
        scale_mat = torch.stack(
            [
                torch.stack([scale_x, torch.zeros_like(theta_z), torch.zeros_like(theta_y)], dim=1),
                torch.stack([torch.zeros_like(theta_z), scale_y, torch.zeros_like(theta_y)], dim=1),
                torch.stack([torch.zeros_like(theta_y), torch.zeros_like(theta_y), scale_z], dim=1),
            ],
            dim=2,
        ).to(theta_x.device)
        shear_mat = torch.stack(
            [
                torch.stack([torch.ones_like(theta_x), torch.tan(shear_xy), torch.tan(shear_xz)], dim=1),
                torch.stack([torch.tan(shear_yx), torch.ones_like(theta_x), torch.tan(shear_yz)], dim=1),
                torch.stack([torch.tan(shear_zx), torch.tan(shear_zy), torch.ones_like(theta_x)], dim=1),
            ],
            dim=2,
        ).to(theta_x.device)
        trans = torch.stack([trans_x, trans_y, trans_z], dim=1).unsqueeze(dim=2)
        mat = torch.bmm(shear_mat, torch.bmm(scale_mat, torch.bmm(rot_mat_z, torch.matmul(rot_mat_y, rot_mat_x))))
        inv_mat = torch.inverse(mat)
        mat = torch.cat([mat, trans], dim=-1)
        inv_trans = torch.bmm(-inv_mat, trans)
        inv_mat = torch.cat([inv_mat, inv_trans], dim=-1)
        grid = nnf.affine_grid(mat, [src.shape[0], 3, src.shape[2], src.shape[3], src.shape[4]], align_corners=False)
        return nnf.grid_sample(src, grid, align_corners=False, mode=self.mode), mat, inv_mat


def _to_tensor(arr, device):
    """Convert array to float tensor on a device.

    Args:
        arr: NumPy array or tensor.
        device: Target torch device.

    Returns:
        Tensor on device.

    """
    if isinstance(arr, torch.Tensor):
        return arr.to(device).float()
    return torch.from_numpy(arr).to(device).float()


def jacobian_determinant(disp: torch.Tensor) -> torch.Tensor:
    """Compute the determinant of the spatial gradient of a displacement field.

    The determinant is taken of the matrix of derivatives ``d u_i / d x_j`` of the displacement itself (no identity is
    added). The derivatives come from ``torch.gradient`` (central differences, unit spacing) applied to
    ``disp[:, i]``, whose axis ``j`` is selected; note that axis 0 of that tensor is the batch axis.

    Args:
        disp: Displacement field ``[B, C, *vol_shape]`` with ``C`` equal to the number of spatial dims (2 or 3).

    Returns:
        Determinant map ``[B, 1, *vol_shape]``.

    """
    dims = disp.shape[1]
    # d u_i / d x_j along the spatial axes (axis j + 1 of disp[:, i]; axis 0 is the batch), plus the identity, so
    # that the result is the Jacobian determinant of the transform x -> x + u(x).
    if hasattr(torch, "gradient"):
        grads = [[torch.gradient(disp[:, i], spacing=1.0, dim=j + 1)[0] for j in range(dims)] for i in range(dims)]
    else:  # PyTorch without torch.gradient
        grads = [[_torch_gradient(disp[:, i : i + 1], spacing=1.0)[j][:, 0] for j in range(dims)] for i in range(dims)]
    J = [[grads[i][j] + (1.0 if i == j else 0.0) for j in range(dims)] for i in range(dims)]

    if dims == 2:
        det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
    else:  # 3‑D
        a = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
        b = J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
        c = J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0])
        det = a - b + c

    return det.unsqueeze(1)  # (B,1,*)


def _torch_gradient(x, spacing=1):
    """Fallback torch gradient for older PyTorch versions.

    Args:
        x: Tensor to differentiate.
        spacing: Grid spacing.

    Returns:
        List of gradients per spatial dimension.

    """
    # minimal torch‑gradient fallback for < 2.1
    grads = []
    for dim in range(x.ndim - 2):
        d = dim + 2
        pad = [0, 0] * (x.ndim - d - 1) + [1, 1]
        g = nnf.pad(x, pad, mode='replicate')  # Neumann
        fwd = g.movedim(d, -1)[..., 2:]
        bwd = g.movedim(d, -1)[..., :-2]
        grads.append((fwd - bwd) / (2 * spacing))
    return grads


# What fit_warp_to_svf(_fast) returns for each ``output_type``. The legacy names are inverted relative to what they
# return ("disp" -> the fitted velocity, "svf" -> the displacement exp(v)); they are kept unchanged because existing
# scripts depend on them. New code should use the explicit "velocity" / "displacement".
_FIT_OUTPUT_TYPES = {"velocity": "velocity", "displacement": "displacement", "disp": "velocity", "svf": "displacement"}


def fit_warp_to_svf(
    warp_t: torch.Tensor,
    nb_steps: int = 7,
    iters: int = 500,
    min_delta: float = 1e-5,
    lr: float = 0.1,
    objective: str = "mse",  # "mse" | "l1"
    init: str = "warp",  # "warp" | "jacwt"
    output_type: str = "disp",  # "svf" | "disp"
    verbose: bool = True,
    device: str = 'cpu',
) -> torch.Tensor:
    """Fit a stationary velocity field ``v`` so that ``exp(v)`` approximates a given displacement field.

    ``v`` is optimized with Adam on the device of ``warp_t``; ``exp`` is :class:`VecInt` scaling and squaring.
    Parameters mirror the original TensorFlow implementation.

    Args:
        warp_t: Displacement field tensor ``[B, ndim, *vol_shape]`` in voxels.
        nb_steps: Number of scaling-and-squaring steps.
        iters: Maximum number of optimization iterations.
        min_delta: Early-stopping threshold on the absolute change of the loss between iterations.
        lr: Adam learning rate.
        objective: Fitting loss, ``"mse"`` or ``"l1"``.
        init: Initialization of ``v``: ``"warp"`` (copy of ``warp_t``) or ``"jacwt"`` (``warp_t`` adjusted by
            :func:`jacobian_determinant`); any other value leaves ``v`` at zero.
        output_type: What to return. ``"velocity"`` returns the fitted stationary velocity field ``v``;
            ``"displacement"`` returns the displacement ``exp(v)`` from the last iteration (a diffeomorphic
            approximation of ``warp_t``). The legacy names ``"disp"`` (the default; returns the velocity) and
            ``"svf"`` (returns the displacement) are kept for backward compatibility; note that they are inverted
            relative to what they return.
        verbose: Print the loss every 20 iterations and on convergence.
        device: Unused; the device of ``warp_t`` is used instead.

    Returns:
        Tensor with the shape of ``warp_t`` (an ``nn.Parameter`` for ``output_type="disp"``).

    Raises:
        ValueError: If ``objective`` or ``output_type`` is not one of the accepted values.

    """
    device = warp_t.device

    vol_shape, ndims = warp_t.shape[2:], warp_t.shape[1]

    # -- optimisation variable ------------------
    vel = nn.Parameter(torch.zeros_like(warp_t, device=device))

    # init velocity
    if init == "warp":
        vel.data.copy_(warp_t)
    elif init == "jacwt":
        jac = jacobian_determinant(warp_t)  # (1,1,*)
        jac3 = jac.repeat(1, ndims, *[1] * len(vol_shape))
        vel.data.copy_(1.02 * warp_t + (1.0 - jac3) * 0.05)

    # integrator (re‑use your VecInt)
    integrator = VecInt(vol_shape, nb_steps).to(device)

    opt = torch.optim.Adam([vel], lr=lr)
    last_loss = None
    if objective == "mse":
        criterion = nnf.mse_loss
    elif objective == "l1":
        criterion = nnf.l1_loss
    else:
        raise ValueError(f"unknown objective {objective!r}, expected 'mse' or 'l1'")
    if output_type not in _FIT_OUTPUT_TYPES:
        raise ValueError(f"unknown output_type {output_type!r}, expected one of {sorted(_FIT_OUTPUT_TYPES)}")
    for epoch in range(iters):
        opt.zero_grad(set_to_none=True)
        disp_pred = integrator(vel)  # exp(v)
        loss = criterion(disp_pred, warp_t)
        loss.backward()
        opt.step()

        # early stop
        if last_loss is not None and abs(last_loss - loss.item()) < min_delta:
            if verbose:
                print(f"Converged @ {epoch:4d}  loss={loss.item():.4e}")
            break
        last_loss = loss.item()

        if verbose and epoch % 20 == 0:
            print(f"[{epoch:03d}]  loss={loss.item():.4e}")
    return vel if _FIT_OUTPUT_TYPES[output_type] == "velocity" else disp_pred


def fit_warp_to_svf_fast(
    warp_t: torch.Tensor,
    nb_steps: int = 5,
    iters: int = 50,
    lr: float = 0.1,
    downsample_factor: int = 2,
    refine_iters: int = 10,
    min_delta: float = 1e-5,
    warm_start: Optional[torch.Tensor] = None,
    use_amp: bool = False,
    output_type: str = "disp",  # "svf" | "disp"
    verbose: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Fast approximation to :func:`fit_warp_to_svf` using coarse-to-fine fitting.

    The velocity is first fitted (MSE loss, Adam) on a trilinearly downsampled copy of the field, upsampled back to
    full resolution and then refined at full resolution.

    Args:
        warp_t: 3-D displacement field tensor ``[B, 3, H, W, D]`` in voxels.
        nb_steps: :class:`VecInt` steps for exponentiation.
        iters: Optimization iterations for the coarse fit.
        lr: Learning rate for the coarse fit and the refinement.
        downsample_factor: Integer downsample factor for the coarse fit (>1 recommended; <= 1 fits at full
            resolution).
        refine_iters: Full-resolution refinement iterations (0 disables refinement).
        min_delta: Early-stopping threshold on the absolute change of the loss between iterations.
        warm_start: Optional initial velocity with the shape of ``warp_t``; defaults to ``warp_t`` itself.
        use_amp: Use CUDA automatic mixed precision (only when ``warp_t`` is on a CUDA device).
        output_type: ``"velocity"`` returns the fitted velocity; ``"displacement"`` returns ``exp(v)``. The legacy
            names ``"disp"`` (default, returns the velocity) and ``"svf"`` (returns the displacement) are kept for
            backward compatibility, as in :func:`fit_warp_to_svf`.
        verbose: Print progress during fitting.
        **kwargs: Accepted for call compatibility with :func:`fit_warp_to_svf`; ignored.

    Returns:
        Tensor of same shape as ``warp_t``, either SVF (velocity) or displacement.

    Raises:
        ValueError: If ``output_type`` is not ``"disp"`` or ``"svf"``.

    """
    device = warp_t.device

    def _optimize(warp, init_vel, nsteps, niter, lr_val):
        vol_shape = warp.shape[2:]
        integrator = VecInt(vol_shape, nsteps).to(device)
        vel = nn.Parameter(init_vel.clone())
        opt = torch.optim.Adam([vel], lr=lr_val)
        last_loss = None
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and warp.is_cuda)
        for epoch in range(niter):
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp and warp.is_cuda):
                disp_pred = integrator(vel)
                loss = nnf.mse_loss(disp_pred, warp)
            if use_amp and warp.is_cuda:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            if last_loss is not None and abs(last_loss - loss.item()) < min_delta:
                if verbose:
                    print(f"Converged @ {epoch:4d}  loss={loss.item():.4e}")
                break
            last_loss = loss.item()
            if verbose and epoch % 20 == 0:
                print(f"[{epoch:03d}]  loss={loss.item():.4e}")
        return vel

    if downsample_factor > 1:
        scale = 1.0 / float(downsample_factor)
        warp_ds = (
            nnf.interpolate(
                warp_t,
                scale_factor=scale,
                mode="trilinear",
                align_corners=False,
            )
            * scale
        )
        if warm_start is not None:
            warm_ds = (
                nnf.interpolate(
                    warm_start,
                    scale_factor=scale,
                    mode="trilinear",
                    align_corners=False,
                )
                * scale
            )
        else:
            warm_ds = warp_ds.detach()
        vel_ds = _optimize(warp_ds, warm_ds, nb_steps, iters, lr)
        vel = (
            nnf.interpolate(
                vel_ds,
                size=warp_t.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
            * downsample_factor
        )
    else:
        init_vel = warm_start if warm_start is not None else warp_t.detach()
        vel = _optimize(warp_t, init_vel, nb_steps, iters, lr)

    if refine_iters > 0:
        vel = _optimize(warp_t, vel.detach(), nb_steps, refine_iters, lr)

    if output_type not in _FIT_OUTPUT_TYPES:
        raise ValueError(f"unknown output_type {output_type!r}, expected one of {sorted(_FIT_OUTPUT_TYPES)}")
    if _FIT_OUTPUT_TYPES[output_type] == "velocity":
        return vel
    return VecInt(warp_t.shape[2:], nb_steps).to(device)(vel)


# -----------------------  inverse via velocity  --------------------------- #


def invert_warp_via_velocity(
    warp: torch.Tensor,
    nb_steps: int = 5,
    iters: int = 100,
    **kwargs: Any,
) -> torch.Tensor:
    """Approximate the inverse of a displacement field by fitting ``v`` and integrating ``-v``.

    Args:
        warp: Displacement field tensor ``[B, ndim, *vol_shape]`` in voxels.
        nb_steps: Number of scaling-and-squaring steps.
        iters: Iterations of :func:`fit_warp_to_svf`.
        **kwargs: Further keyword arguments for :func:`fit_warp_to_svf`.

    Returns:
        Inverse displacement field tensor with the same shape as ``warp``, on the device of the fitted velocity.

    """
    vel = fit_warp_to_svf(warp, nb_steps, iters, **kwargs)

    # integrate ‑v
    device = vel.device  # integrate where the fitted velocity lives
    inv_disp = VecInt(vel.shape[2:], nb_steps).to(device)(-vel)
    return inv_disp


def ensure_svf(field: Any, field_type: str, nb_steps: int = 5, iters: int = 100, **fit_kw: Any) -> Any:
    """Return a stationary velocity field, fitting one if the input is a displacement.

    Args:
        field: Velocity or displacement field. A ``'svf'`` field is returned unchanged; a ``'disp'`` field must be a
            tensor ``[B, ndim, *vol_shape]`` accepted by :func:`fit_warp_to_svf`.
        field_type: ``'svf'`` (already a velocity) or ``'disp'`` (plain displacement).
        nb_steps: Number of scaling-and-squaring steps for the fit.
        iters: Iterations of :func:`fit_warp_to_svf`.
        **fit_kw: Further keyword arguments for :func:`fit_warp_to_svf`.

    Returns:
        ``field`` itself for ``'svf'``, otherwise the fitted velocity tensor with the shape of ``field``.

    Raises:
        ValueError: If ``field_type`` is not ``'svf'`` or ``'disp'``.

    """
    if field_type == 'svf':
        return field  # nothing to do
    if field_type == 'disp':
        # fit_warp_to_svf defined earlier
        return fit_warp_to_svf(field, nb_steps=nb_steps, iters=iters, **fit_kw)
    raise ValueError(f"unknown field_type {field_type}")


def ensemble_average(
    fields: Sequence[Any],  # list of np arrays, each (*vol, ndims)
    field_types: Sequence[str],  # parallel list, 'svf' or 'disp'
    weights: Optional[Sequence[float]] = None,  # list of floats or None  (defaults to equal)
    nb_steps: int = 5,
    iters: int = 100,
    device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> torch.Tensor:
    """Ensemble average of displacement fields or SVFs in the log (velocity) domain.

    Every displacement field is converted to a stationary velocity field with :func:`ensure_svf`, the first batch
    element of each velocity is taken, the velocities are averaged with the normalized weights and the average is
    exponentiated once with :class:`VecInt`: ``disp_bar = exp(sum_i w_i v_i)``.

    Args:
        fields: Displacement fields or SVFs, each a 3-D field tensor ``[B, 3, H, W, D]`` (index 0 of each is used for
            the average; ``'disp'`` entries are fitted with :func:`fit_warp_to_svf`).
        field_types: Parallel list of ``'svf'`` or ``'disp'``.
        weights: Per-field weights, normalized to sum to 1; ``None`` gives equal weights.
        nb_steps: Number of integration steps for :class:`VecInt`.
        iters: Number of iterations for :func:`fit_warp_to_svf`.
        device: Device on which the velocities are averaged and integrated (default: CUDA if available, chosen at
            import time).

    Returns:
        Averaged displacement field tensor ``[3, H, W, D]`` (channel-first, no batch axis) on ``device``.

    Example:
        Usage sketch (array layouts follow the original example)::

            # assume you already have three predictions:
            svf_A   = np.load("model_A_velocity.npy")    # (*vol, 3)
            disp_B  = np.load("model_B_disp.npy")        # (*vol, 3)
            disp_C  = np.load("model_C_disp.npy")

            fields      = [svf_A, disp_B, disp_C]
            field_types = ['svf', 'disp', 'disp']
            # e.g. let Dice on a small val-set decide the weights
            weights     = [0.45, 0.35, 0.20]

            disp_mean = ensemble_average(
                fields,
                field_types,
                weights,
                nb_steps=7,          # same nsteps you used during training/inference
                iters=150,           # iterations for fit_warp_to_svf on the disp fields
            )

            # now warp any tensor (image, mask, logits) with the consensus deformation
            vol_shape = disp_mean.shape[:-1]
            transform = SpatialTransformer(vol_shape).cuda()         # if GPU used above
            moving_img_t = torch.from_numpy(moving_img)[None, None].float().cuda()
            disp_t = torch.from_numpy(disp_mean).permute(3,0,1,2)[None].float().cuda()
            warped = transform(moving_img_t, disp_t)                 # (1,1,*vol)

    """
    assert len(fields) == len(field_types), "length mismatch"
    N = len(fields)
    if weights is None:
        weights = [1.0 / N] * N
    else:
        wsum = sum(weights)
        weights = [w / wsum for w in weights]  # normalize to 1

    # 1. make sure everything is an SVF
    v_list = [
        ensure_svf(f, t, nb_steps=nb_steps, iters=iters)  # returns (*vol, ndims)
        for f, t in zip(fields, field_types)
    ]

    # 2. stack tensors (B,C,*) for averaging
    v_stack = torch.stack([v[0] for v in v_list], dim=0).float().to(device)  # (N,C,*)
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    v_bar = torch.sum(w[:, None, None, None, None] * v_stack, dim=0, keepdim=True)
    # v_bar shape (1,C,*)

    # 3. exponentiate once → displacement
    integrator = VecInt(v_bar.shape[2:], nb_steps).to(device)
    disp_bar = integrator(v_bar).squeeze(0)  # (C,*)

    return disp_bar
