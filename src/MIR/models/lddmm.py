"""Model-agnostic LDDMM (time-varying velocity field) components for MIR.

* :class:`FourierKernel`   - the smoothing operator K = L^{-1} defining the velocity space V.
* :class:`LDDMMIntegrator` - integrates v_t to a diffeomorphism (forward and inverse map) either by
                             geodesic shooting from an initial momentum or by relaxation.
* :class:`LDDMMWrapper`    - turns ANY registration backbone that maps (mov, fix) to a dense
                             3-channel field into an end-to-end LDDMM model: the backbone output is
                             used as the initial momentum (geodesic shooting) or as the per-step
                             pre-velocity (relaxation), and the flow is integrated inside the model.

Conventions follow the rest of MIR: a displacement field ``u`` warps the moving image as
``mov(x + u(x))`` (``SpatialTransformer``), ``psi = id + u`` is the LDDMM flow phi_1 and
``id + w`` its inverse.

References:
    - Beg, Miller, Trouve, Younes. Computing large deformation metric mappings via geodesic
      flows of diffeomorphisms. IJCV 2005.
    - Vialard, Risser, Rueckert, Cotter. Diffeomorphic 3D image registration via geodesic
      shooting using an efficient adjoint calculation. IJCV 2012.
    - Zhang & Fletcher. Fast diffeomorphic image registration via Fourier-approximated Lie
      algebras (FLASH). IJCV 2019.
Yang, Kwitt, Styner, Niethammer. Quicksilver: Fast predictive image registration.
    NeuroImage 2017.

Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University

"""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import MIR.models.registration_utils as utils


# --------------------------------------------------------------------------- #
# Smoothing kernel K = L^{-1}
# --------------------------------------------------------------------------- #
class FourierKernel(nn.Module):
    """Sobolev smoothing operator applied in the Fourier domain (periodic boundaries).

    ``kernel='helmholtz'``: L = (-alpha * Laplacian + gamma)^power, K = L^{-1}.
        This is the classic LDDMM metric (Beg et al.; FLASH). ``sqrt(alpha/gamma)`` is the
        characteristic length (in voxels) and ``power`` controls how fast high frequencies
        are damped.
    ``kernel='gaussian'``: K is a (sum of) Gaussian(s) with standard deviation ``sigma``
        (a float or a list of floats, with optional ``weights``), as used by Quicksilver.

    The inner product ``<p, K p>`` (returned by :meth:`inner`) is the squared V-norm of
    ``v = K p``; it is the LDDMM regularizer.

    Args:
        kernel: ``'helmholtz'`` or ``'gaussian'``.
        alpha, gamma, power: Helmholtz operator parameters.
        sigma: Gaussian standard deviation(s) in voxels.
        weights: Optional mixture weights for a multi-Gaussian kernel.

    Forward inputs:
        p: Field of shape [B, C, D, H, W] (momentum / pre-velocity).

    Forward outputs:
        v = K p with the same shape (float32).

    """

    def __init__(self, kernel='helmholtz', alpha=2.0, gamma=1.0, power=2, sigma=2.0, weights=None):
        super().__init__()
        if kernel not in ('helmholtz', 'gaussian'):
            raise ValueError(f'Unknown kernel type: {kernel}')
        self.kernel = kernel
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.power = float(power)
        self.sigmas = [float(s) for s in (sigma if isinstance(sigma, (list, tuple)) else [sigma])]
        if weights is None:
            weights = [1.0 / len(self.sigmas)] * len(self.sigmas)
        if len(weights) != len(self.sigmas):
            raise ValueError('weights must have the same length as sigma')
        self.weights = [float(w) for w in weights]
        self._cache_shape = None
        self.register_buffer('k_hat', torch.zeros(1), persistent=False)

    def scaled(self, scale):
        """Return a copy whose length scales are expressed on a grid ``scale`` times finer."""
        return FourierKernel(
            kernel=self.kernel,
            alpha=self.alpha * scale * scale,
            gamma=self.gamma,
            power=self.power,
            sigma=[s * scale for s in self.sigmas],
            weights=self.weights,
        )

    def _build(self, shape, device):
        """Build the Fourier multiplier K_hat on the rfft grid of ``shape``."""
        lam = None  # eigenvalues of the discrete negative Laplacian (periodic)
        om2 = None  # continuous |omega|^2
        nd = len(shape)
        for d, n in enumerate(shape):
            freq = torch.fft.rfftfreq(n) if d == nd - 1 else torch.fft.fftfreq(n)
            freq = freq.to(device=device, dtype=torch.float32)
            view = [1] * nd
            view[d] = -1
            lam_d = (2.0 - 2.0 * torch.cos(2.0 * math.pi * freq)).view(view)
            om2_d = ((2.0 * math.pi * freq) ** 2).view(view)
            lam = lam_d if lam is None else lam + lam_d
            om2 = om2_d if om2 is None else om2 + om2_d
        if self.kernel == 'helmholtz':
            k_hat = (self.alpha * lam + self.gamma) ** (-self.power)
        else:
            k_hat = torch.zeros_like(om2)
            for s, w in zip(self.sigmas, self.weights):
                k_hat = k_hat + w * torch.exp(-0.5 * (s**2) * om2)
        self.k_hat = k_hat
        self._cache_shape = (tuple(shape), str(device))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Apply the smoothing operator, ``v = K p``, by FFT (periodic boundaries).

        The Fourier multiplier is cached and rebuilt only when the spatial shape or device changes.

        Args:
            p: Field [B, C, *spatial] (e.g. a momentum).

        Returns:
            torch.Tensor: Smoothed field ``K p`` of the same shape (float32).
        """
        shape = tuple(p.shape[2:])
        if self._cache_shape != (shape, str(p.device)):
            self._build(shape, p.device)
        dims = tuple(range(2, p.dim()))
        p_hat = torch.fft.rfftn(p.float(), dim=dims)
        v = torch.fft.irfftn(p_hat * self.k_hat, s=shape, dim=dims)
        return v

    @staticmethod
    def inner(p, v, weight=None):
        """Mean (over voxels and batch) of the pointwise inner product <p, v>.

        When ``v = K p`` this equals ||v||_V^2 / (number of voxels).  ``weight`` ([B, 1, ...],
        optional) multiplies the integrand pointwise (SPR-style weighted energy).
        """
        e = (p.float() * v).sum(dim=1, keepdim=True)
        if weight is not None:
            e = e * weight
        return e.mean()

    def impulse_response(self, shape, device='cpu'):
        """Return K applied to a unit impulse at the centre of ``shape`` (for diagnostics)."""
        delta = torch.zeros((1, 1) + tuple(shape), device=device)
        delta[(0, 0) + tuple(s // 2 for s in shape)] = 1.0
        return self.forward(delta)[0, 0]

    def fwhm(self, shape=(64, 64, 64)):
        """Full width at half maximum (voxels) of the impulse response along the last axis."""
        ir = self.impulse_response(shape)
        c = tuple(s // 2 for s in shape)
        prof = ir[c[0], c[1], :]
        return float((prof > ir[c] / 2).sum())


# --------------------------------------------------------------------------- #
# Spatially varying (region-specific) kernel, RDMM style
# --------------------------------------------------------------------------- #
class SpatiallyVaryingKernel(nn.Module):
    """Multi-Gaussian kernel with per-voxel mixture weights (Shen, Vialard, Niethammer, NeurIPS 2019).

        K_w m = sum_i sqrt(w_i) * G_{sigma_i} * (sqrt(w_i) m),      w_i(x) >= 0, sum_i w_i(x) = 1

    Each term is symmetric positive semi-definite, so ``<m, K_w m>`` is a valid (spatially
    varying) kinetic energy.  Where the weights favour a large ``sigma`` the deformation is
    smooth, where they favour a small ``sigma`` it can be sharp: this is the LDDMM counterpart
    of spatially varying regularization (SPR).  Along a geodesic the weights are advected with
    the flow (``w_t = w_0 o psi_t^{-1}``), which :class:`LDDMMIntegrator` does automatically.

    Args:
        sigmas: Standard deviations (voxels) of the Gaussians, e.g. ``[1, 2, 4, 8]``.
        weight_smooth_sigma: Gaussian smoothing (voxels) applied to the weight maps produced by
            :meth:`weights_from_logits` (0 = none).
        omt_power: Exponent ``p`` of the optimal-mass-transport penalty.

    Forward inputs:
        p: Field [B, C, D, H, W];  w: weights [B, n_kernels, D, H, W].

    Forward outputs:
        v = K_w p (float32).

    """

    def __init__(self, sigmas=(1.0, 2.0, 4.0, 8.0), weight_smooth_sigma=1.0, omt_power=2.0):
        super().__init__()
        if len(sigmas) < 2:
            raise ValueError('SpatiallyVaryingKernel needs at least two sigmas')
        self.sigmas = [float(s) for s in sigmas]
        self.weight_smooth_sigma = float(weight_smooth_sigma)
        self.omt_power = float(omt_power)
        self.gaussians = nn.ModuleList([FourierKernel('gaussian', sigma=s) for s in self.sigmas])
        self.weight_smoother = (
            FourierKernel('gaussian', sigma=self.weight_smooth_sigma) if self.weight_smooth_sigma > 0 else None
        )
        self.eps = 1e-6

    @property
    def n_kernels(self) -> int:
        """Number of Gaussians in the mixture."""
        return len(self.sigmas)

    def scaled(self, scale):
        """Return a copy whose length scales are expressed on a grid ``scale`` times finer."""
        return SpatiallyVaryingKernel(
            [s * scale for s in self.sigmas], self.weight_smooth_sigma * scale, self.omt_power
        )

    def forward(self, p: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Apply the spatially varying kernel, ``v = K_w p``.

        Args:
            p: Field [B, C, D, H, W].
            w: Kernel weights [B, n_kernels, D, H, W] (negative values are clamped to 0).

        Returns:
            torch.Tensor: ``sum_i sqrt(w_i) * G_{sigma_i} * (sqrt(w_i) p)`` (float32), same shape as ``p``.

        Raises:
            ValueError: If ``w`` does not have ``n_kernels`` channels.
        """
        if w.shape[1] != self.n_kernels:
            raise ValueError(f'expected {self.n_kernels} weight channels, got {w.shape[1]}')
        p = p.float()
        sw = w.float().clamp_min(0).sqrt()
        v = None
        for i, g in enumerate(self.gaussians):
            s_i = sw[:, i : i + 1]
            v_i = s_i * g(s_i * p)
            v = v_i if v is None else v + v_i
        return v

    inner = staticmethod(FourierKernel.inner)

    def weights_from_logits(self, logits):
        """Softmax over the kernel axis, optional smoothing, renormalization (sum to 1)."""
        w = torch.softmax(logits.float(), dim=1)
        if self.weight_smoother is not None:
            w = self.weight_smoother(w).clamp_min(self.eps)
            w = w / w.sum(dim=1, keepdim=True)
        return w

    def weights_from_spr(self, lam):
        """Kernel weights from an SPR-style regularization-strength map ``lam`` in [0, 1].

        ``lam = 1`` selects the smoothest Gaussian (largest sigma), ``lam = 0`` the sharpest one,
        intermediate values interpolate linearly between neighbouring sigmas (for two sigmas the
        weights are simply ``[1 - lam, lam]``).  This turns the single weight map of TransMorph-SPR /
        VFA-SPR into a spatially varying LDDMM metric.
        """
        lam = lam.float().clamp(0.0, 1.0)
        n = self.n_kernels
        t = lam[:, 0] * (n - 1)  # position on the sigma axis
        idx = torch.arange(n, device=lam.device, dtype=torch.float32).view(1, n, 1, 1, 1)
        w = (1.0 - (t.unsqueeze(1) - idx).abs()).clamp_min(0.0)  # linear hat functions
        if self.weight_smoother is not None:
            w = self.weight_smoother(w).clamp_min(self.eps)
        else:
            w = w.clamp_min(self.eps)
        return w / w.sum(dim=1, keepdim=True)

    def omt(self, w):
        """Optimal-mass-transport penalty: cost of moving weight away from the largest sigma.

        ``mean_x sum_i w_i(x) ((sigma_max - sigma_i) / (sigma_max - sigma_min))^p`` in [0, 1].
        """
        s = torch.tensor(self.sigmas, device=w.device, dtype=torch.float32)
        cost = ((s.max() - s) / (s.max() - s.min())) ** self.omt_power
        return (w * cost.view(1, -1, 1, 1, 1)).sum(dim=1).mean()

    def effective_sigma(self, w):
        """Per-voxel weighted mean sigma [B, 1, D, H, W] (for visualization)."""
        s = torch.tensor(self.sigmas, device=w.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        return (w * s).sum(dim=1, keepdim=True)

    def fwhm(self, shape=(64, 64, 64)):
        """FWHM (voxels) of the smoothest and sharpest Gaussian, for logging."""
        return self.gaussians[-1].fwhm(shape), self.gaussians[0].fwhm(shape)


class KernelWeightHead(nn.Module):
    """Default model-agnostic predictor of the kernel-weight logits.

    A small 3-D CNN applied to ``cat(mov, fix, field)`` (field = momentum / pre-velocity) at the
    field resolution.  The last layer is zero-initialized with a bias that puts most of the
    weight on the smoothest kernel (``init='smoothest'``) or on all kernels equally
    (``init='uniform'``), so training starts from a globally smooth metric.

    Args:
        n_kernels: Number of Gaussians of the :class:`SpatiallyVaryingKernel`.
        in_channels: Channels of the input (default 2 images + 3 field channels).
        width: Hidden channels.
        init: ``'smoothest'`` or ``'uniform'``.

    """

    def __init__(self, n_kernels, in_channels=5, width=16, init='smoothest'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, width, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(width, width, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(width, n_kernels, 3, padding=1),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        bias = torch.zeros(n_kernels)
        if init == 'smoothest':
            bias[-1] = 3.0  # softmax -> ~0.87 on the largest sigma for 4 kernels
        elif init != 'uniform':
            raise ValueError("init must be 'smoothest' or 'uniform'")
        with torch.no_grad():
            last.bias.copy_(bias)

    def forward(self, mov: torch.Tensor, fix: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        """Predict kernel-weight logits.

        Args:
            mov: Moving image [B, 1, ...]; resized (trilinear) to the field resolution if needed.
            fix: Fixed image [B, 1, ...]; resized (trilinear) to the field resolution if needed.
            field: Backbone field (momentum / pre-velocity) [B, 3, D, H, W].

        Returns:
            torch.Tensor: Logits [B, n_kernels, D, H, W] (see :meth:`SpatiallyVaryingKernel.weights_from_logits`).
        """
        size = field.shape[2:]
        if tuple(mov.shape[2:]) != tuple(size):
            mov = F.interpolate(mov, size=size, mode='trilinear', align_corners=False)
            fix = F.interpolate(fix, size=size, mode='trilinear', align_corners=False)
        return self.net(torch.cat((mov, fix, field), dim=1))


class SPRHead(nn.Module):
    """Model-agnostic spatially varying regularization (SPR) head.

    Same structure as :class:`KernelWeightHead` but with one sigmoid output ``lam`` in
    ``(eps, 1]``, the per-voxel regularization strength of TransMorph-SPR / VFA-SPR (1 = full
    regularization).  The last layer is zero-initialized with ``init_bias`` so that training
    starts from ``lam = sigmoid(init_bias)`` everywhere (0.88 by default).

    Args:
        in_channels: Channels of ``cat(mov, fix, field)``.
        width: Hidden channels.
        init_bias: Bias of the last layer at initialization.
        eps: Lower clamp of the output ``lam``.

    """

    def __init__(self, in_channels=5, width=16, init_bias=2.0, eps=1e-6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, width, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(width, width, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(width, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, float(init_bias))
        self.eps = eps

    def forward(self, mov: torch.Tensor, fix: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        """Predict the SPR regularization-strength map.

        Args:
            mov: Moving image [B, 1, ...]; resized (trilinear) to the field resolution if needed.
            fix: Fixed image [B, 1, ...]; resized (trilinear) to the field resolution if needed.
            field: Backbone field (momentum / pre-velocity) [B, 3, D, H, W].

        Returns:
            torch.Tensor: ``lam`` [B, 1, D, H, W] in ``[eps, 1]``.
        """
        size = field.shape[2:]
        if tuple(mov.shape[2:]) != tuple(size):
            mov = F.interpolate(mov, size=size, mode='trilinear', align_corners=False)
            fix = F.interpolate(fix, size=size, mode='trilinear', align_corners=False)
        return torch.clamp(torch.sigmoid(self.net(torch.cat((mov, fix, field), dim=1))), self.eps, 1.0)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def jacobian_fd(disp):
    """Jacobian of the map ``id + disp`` by central finite differences.

    Args:
        disp: Displacement field [B, 3, D, H, W].

    Returns:
        Tensor [B, 3, 3, D, H, W] with entry [:, i, j] = d(x_i + disp_i) / d x_j.

    """
    padded = F.pad(disp, (1, 1, 1, 1, 1, 1), mode='replicate')
    cols = []
    for j in range(3):
        fwd = [slice(None), slice(None)] + [slice(1, -1)] * 3
        bwd = [slice(None), slice(None)] + [slice(1, -1)] * 3
        fwd[2 + j] = slice(2, None)
        bwd[2 + j] = slice(None, -2)
        cols.append(0.5 * (padded[tuple(fwd)] - padded[tuple(bwd)]))
    jac = torch.stack(cols, dim=2)  # [B, 3(i), 3(j), ...]
    eye = torch.eye(3, device=disp.device, dtype=jac.dtype).view(1, 3, 3, 1, 1, 1)
    return jac + eye


def det3x3(jac):
    """Determinant of a field of 3x3 matrices [B, 3, 3, ...] -> [B, ...]."""
    a = jac
    return (
        a[:, 0, 0] * (a[:, 1, 1] * a[:, 2, 2] - a[:, 1, 2] * a[:, 2, 1])
        - a[:, 0, 1] * (a[:, 1, 0] * a[:, 2, 2] - a[:, 1, 2] * a[:, 2, 0])
        + a[:, 0, 2] * (a[:, 1, 0] * a[:, 2, 1] - a[:, 1, 1] * a[:, 2, 0])
    )


# --------------------------------------------------------------------------- #
# Time integration of the flow
# --------------------------------------------------------------------------- #
class LDDMMIntegrator(nn.Module):
    """Integrate a time-varying velocity field to a diffeomorphism (and its inverse).

    The forward map ``psi = id + u`` and its inverse ``psi^{-1} = id + w`` are advanced with
    an explicit Euler step of size ``dt = 1 / time_steps``:

        u <- u + dt * v_t o (id + u)
        w <- -dt * v_t + w o (id - dt * v_t)

    With a :class:`SpatiallyVaryingKernel` the kernel weights ``w_0`` are advected with the
    flow, ``w_t = w_0 o psi_t^{-1}``, before the velocity ``v_t = K_{w_t} m_t`` is formed.

    Args:
        size: Spatial size (D, H, W) of the fields.
        time_steps: Number of time steps.
        kernel: A :class:`FourierKernel` (K = L^{-1}) or a :class:`SpatiallyVaryingKernel`.

    """

    def __init__(self, size, time_steps, kernel):
        super().__init__()
        if time_steps < 1:
            raise ValueError('time_steps must be >= 1')
        self.size = tuple(size)
        self.time_steps = int(time_steps)
        self.dt = 1.0 / self.time_steps
        self.kernel = kernel
        self.transformer = utils.SpatialTransformer(self.size)

    @property
    def spatially_varying(self) -> bool:
        """Whether ``self.kernel`` is a :class:`SpatiallyVaryingKernel`."""
        return isinstance(self.kernel, SpatiallyVaryingKernel)

    def step(self, u, w, v):
        """One Euler step of the forward and inverse maps with velocity ``v``."""
        dv = self.dt * v
        u = u + self.transformer(dv, u)
        w = -dv + self.transformer(w, -dv)
        return u, w

    def _velocity(self, p, weights, w_inv, t):
        """V = K p, advecting the kernel weights with the inverse map when the kernel varies in space."""
        if weights is None:
            if self.spatially_varying:
                raise ValueError('SpatiallyVaryingKernel needs kernel weights (pass weights=...)')
            return self.kernel(p)
        w_t = weights if t == 0 else self.transformer(weights, w_inv, padding_mode='border')
        return self.kernel(p, w_t)

    def transport_momentum(self, m0, w):
        """Push the initial momentum forward with the inverse map ``id + w``.

        ``m_t = |D psi^{-1}| (D psi^{-1})^T (m_0 o psi^{-1})``
        """
        jac = jacobian_fd(w)  # [B, 3, 3, ...]
        det = det3x3(jac)  # [B, ...]
        m_pull = self.transformer(m0, w)  # m_0 o psi^{-1}
        m_t = torch.einsum('bjidhw,bjdhw->bidhw', jac, m_pull)  # (J^T m)_i = sum_j J_ji m_j
        return det.unsqueeze(1) * m_t

    def shoot(self, m0, weights=None, return_path=False, energy_weight=None):
        """Geodesic shooting from the initial momentum ``m0``.

        Args:
            m0: Initial momentum [B, 3, D, H, W].
            weights: Kernel weights ``w_0`` [B, n_kernels, D, H, W] (spatially varying kernel only).
            return_path: Also return the list of velocities v_t.
            energy_weight: Optional pointwise weight [B, 1, D, H, W] of the returned ``energy``
                (SPR-weighted kinetic energy); the diagnostic ``energies`` stay unweighted.

        Returns:
            dict with ``flow`` (u), ``inv_flow`` (w), ``energy`` (<m_0, K m_0>, differentiable),
            ``energies`` (detached per-step kinetic energies, constant along the geodesic),
            and optionally ``velocities``.

        """
        m0 = m0.float()
        u = torch.zeros_like(m0)
        w = torch.zeros_like(m0)
        energies = []
        velocities = []
        flows = []
        energy = None
        for t in range(self.time_steps):
            m_t = m0 if t == 0 else self.transport_momentum(m0, w)
            v_t = self._velocity(m_t, weights, w, t)
            e_t = self.kernel.inner(m_t, v_t)
            if t == 0:
                energy = e_t if energy_weight is None else self.kernel.inner(m_t, v_t, energy_weight)
            energies.append(e_t.detach())
            u, w = self.step(u, w, v_t)
            if return_path:
                velocities.append(v_t)
                flows.append(u)
        out = {'flow': u, 'inv_flow': w, 'energy': energy, 'energies': energies}
        if return_path:
            out['velocities'] = velocities
            out['flows'] = flows  # phi_t for t = 1/T ... 1 (displacements)
        return out

    def relax(self, predict_fn, weights=None, return_path=False, energy_weight=None):
        """Relaxation (time-varying field) integration.

        Args:
            predict_fn: Callable ``(t, u) -> p_t`` returning the pre-kernel field at step ``t``
                given the current forward displacement ``u`` (``None`` at ``t = 0``).
            weights: Kernel weights ``w_0`` [B, n_kernels, D, H, W] (spatially varying kernel only).
            return_path: Also return the list of velocities v_t.
            energy_weight: Optional pointwise weight [B, 1, D, H, W] applied to every step's
                energy (SPR-weighted path energy); ``energies`` stay unweighted.

        Returns:
            dict with ``flow``, ``inv_flow``, ``energy`` (sum_t dt <p_t, K p_t>, differentiable),
            ``energies`` (detached per-step values) and optionally ``velocities``.

        """
        u = w = None
        energy = 0.0
        energies = []
        velocities = []
        flows = []
        for t in range(self.time_steps):
            if u is None:
                p_t = predict_fn(t, None)
                u = torch.zeros((p_t.shape[0], 3) + self.size, device=p_t.device, dtype=torch.float32)
                w = torch.zeros_like(u)
            else:
                p_t = predict_fn(t, u)
            v_t = self._velocity(p_t, weights, w, t)
            e_t = self.kernel.inner(p_t, v_t)
            energy = energy + self.dt * (e_t if energy_weight is None else self.kernel.inner(p_t, v_t, energy_weight))
            energies.append(e_t.detach())
            u, w = self.step(u, w, v_t)
            if return_path:
                velocities.append(v_t)
                flows.append(u)
        out = {'flow': u, 'inv_flow': w, 'energy': energy, 'energies': energies}
        if return_path:
            out['velocities'] = velocities
            out['flows'] = flows
        return out


# --------------------------------------------------------------------------- #
# Backbone adapters
# --------------------------------------------------------------------------- #
def adapter_pair(backbone, mov, fix):
    """Default: ``backbone((mov, fix))`` (TransMorph, VFA, VoxelMorph, LessNet, Fourier-Net, ...)."""
    return backbone((mov, fix))


def adapter_sitreg(backbone, mov, fix):
    """SITReg: ``backbone(mov, fix, ...)`` returns mapping pairs; take the level-0 forward displacement."""
    from MIR.models.SITReg.composable_mapping import DataFormat

    pair = backbone(mov, fix, mappings_for_levels=((0, False),))[0]
    return pair.forward_mapping.sample(data_format=DataFormat.voxel_displacements()).generate_values()


ADAPTERS = {'pair': adapter_pair, 'sitreg': adapter_sitreg}


# --------------------------------------------------------------------------- #
# Model-agnostic wrapper
# --------------------------------------------------------------------------- #
class LDDMMWrapper(nn.Module):
    """Turn any dense registration backbone into an end-to-end LDDMM model.

    The backbone must map a moving/fixed pair to a dense 3-channel field at some resolution
    (any MIR registration network with its own SVF integration switched off qualifies).  That
    field is *not* used as a displacement: it is the initial momentum ``m_0`` (``param='shooting'``)
    or the pre-velocity ``p_t`` (``param='tvf'``), and the diffeomorphism is obtained by
    integrating the resulting time-varying velocity field.  Backbone and integration are trained
    jointly; the regularizer is the LDDMM kinetic energy returned in the output dict.

    * ``param='shooting'``: one backbone call; ``m_0`` is transported with the conservation
      law and ``energy = <m_0, K m_0>``.
    * ``param='tvf'``: the backbone is re-applied at every time step to ``(mov o psi_t, fix)``
      with shared weights (``time_steps`` backbone calls); ``energy = sum_t dt <p_t, K p_t>``.
      The backbone field must have the resolution of ``mov``.

    **Spatially varying kernel (region-specific LDDMM / SPR).**  Pass ``sigmas=[1, 2, 4, 8]`` to
    replace the stationary kernel by a :class:`SpatiallyVaryingKernel`.  A :class:`KernelWeightHead`
    (or a user-supplied ``weight_head(mov, fix, field) -> logits``) predicts per-voxel mixture
    weights from the images and the backbone field; the weights are advected with the flow and the
    output dict gains ``kernel_weights`` [B, n_kernels, ...] and the ``omt`` penalty that should be
    added to the loss (it keeps the kernel smooth unless the data demand otherwise).

    **SPR integration.**  ``spr='kernel'`` uses a TransMorph-SPR style weight map ``lam`` in
    (0, 1] (one channel, 1 = full regularization) to set the kernel scale per voxel through
    :meth:`SpatiallyVaryingKernel.weights_from_spr` (``sigmas`` required): the SPR map becomes a
    spatially varying LDDMM metric.  ``spr='energy'`` keeps the stationary kernel and weights the
    kinetic-energy integrand by ``lam`` instead, i.e. exactly the role of ``lam`` in the
    ``LocalGrad3d`` penalty of TransMorph-SPR.  ``lam`` comes from a model-agnostic :class:`SPRHead`
    (``spr_source='head'``) or from the backbone itself when it is an SPR model returning
    ``(field, lam)`` such as ``TransMorphTVFSPR(SVF=False)`` or ``VFASPR(SVF=False)``
    (``spr_source='backbone'``).  The output dict then contains ``spr_weight``; apply the Beta
    prior ``logBeta(spr_weight, 1 + wt)`` in the loss as in the SPR tutorials.

    Args:
        backbone: ``nn.Module`` producing the field.
        param: ``'shooting'`` or ``'tvf'``.
        time_steps: Number of integration steps.
        kernel, alpha, gamma, power, sigma, kernel_weights: See :class:`FourierKernel`
            (parameters in voxels of the backbone output grid). Ignored when ``sigmas`` is given.
        spr: ``None``, ``'kernel'`` or ``'energy'`` (see above).
        spr_source: ``'head'`` (default :class:`SPRHead`, or the module passed as ``spr_head``) or
            ``'backbone'`` (the adapter output must be ``(field, lam)``).
        spr_head: Optional module/callable ``(mov, fix, field) -> lam`` for ``spr_source='head'``.
        sigmas: Gaussian standard deviations of the spatially varying kernel (None = stationary kernel).
        weight_head: Module/callable ``(mov, fix, field) -> logits [B, len(sigmas), ...]``; default
            :class:`KernelWeightHead`.
        weight_smooth_sigma, omt_power: See :class:`SpatiallyVaryingKernel`.
        int_scale: Integrate on a grid ``int_scale`` times finer than the backbone output
            (``'shooting'`` only); the returned fields are at that resolution.
        adapter: ``'pair'`` (default, ``backbone((mov, fix))``), ``'sitreg'`` or a callable
            ``(backbone, mov, fix) -> field``.  If the backbone returns a tuple/list the last
            element is used (e.g. ``(warped, flow)`` from ``gen_output=True``).
        field_scale: Initial value of a learnable scalar multiplying the backbone field before it
            is used as momentum / pre-velocity (``None`` = no scaling).  Useful when warm-starting
            from a backbone trained to output displacements: a trained displacement field is far
            too large and rough to be a momentum (the geodesic blows up), so start small, e.g. 0.2,
            and let training grow it.
        return_path: Return the intermediate flows and velocities (``flows``, ``velocities``) in the dict.

    Forward inputs:
        inputs: Tuple ``(mov, fix)`` of tensors [B, 1, D, H, W].
        return_dict: If True, return a dict with ``flow``, ``inv_flow``, ``energy``, ``energies``,
            ``momentum`` (shooting) / ``pre_velocities`` (tvf) and, with a spatially varying
            kernel, ``kernel_weights`` and ``omt``.

    Forward outputs:
        Dense displacement field [B, 3, D', H', W'] (or the dict).

    Example::

        backbone = VFA(config, device='cuda:0')           # SVF=False (default)
        model = LDDMMWrapper(backbone, param='shooting', time_steps=7, sigmas=[1, 2, 4, 8]).cuda()
        out = model((mov, fix), return_dict=True)
        loss = ncc(warp(mov, out['flow']), fix) + 0.01 * out['energy'] + 0.1 * out['omt']

    """

    def __init__(
        self,
        backbone,
        param='shooting',
        time_steps=7,
        kernel='helmholtz',
        alpha=2.0,
        gamma=1.0,
        power=2,
        sigma=2.0,
        kernel_weights=None,
        sigmas=None,
        weight_head=None,
        weight_smooth_sigma=1.0,
        omt_power=2.0,
        spr=None,
        spr_source='head',
        spr_head=None,
        int_scale=1,
        adapter=None,
        field_scale=None,
        return_path=False,
    ):
        super().__init__()
        if param not in ('shooting', 'tvf'):
            raise ValueError(f"param must be 'shooting' or 'tvf', got {param}")
        if param == 'tvf' and int_scale != 1:
            raise ValueError("int_scale > 1 is only supported with param='shooting'")
        if spr not in (None, 'kernel', 'energy'):
            raise ValueError("spr must be None, 'kernel' or 'energy'")
        if spr == 'kernel' and sigmas is None:
            raise ValueError("spr='kernel' needs the sigmas of a spatially varying kernel")
        if spr_source not in ('head', 'backbone'):
            raise ValueError("spr_source must be 'head' or 'backbone'")
        if getattr(backbone, 'SVF', False):
            warnings.warn(
                'The backbone has SVF integration enabled; LDDMMWrapper expects the raw '
                '(un-integrated) field. Construct the backbone with SVF=False.',
                stacklevel=2,
            )
        self.backbone = backbone
        self.param = param
        self.time_steps = int(time_steps)
        self.int_scale = int(int_scale)
        if adapter is None:
            adapter = 'pair'
        self.adapter = ADAPTERS[adapter] if isinstance(adapter, str) else adapter
        self.adapter_name = adapter if isinstance(adapter, str) else getattr(adapter, '__name__', 'custom')
        self.spr = spr
        self.spr_source = spr_source
        if sigmas is not None:
            self.kernel = SpatiallyVaryingKernel(sigmas, weight_smooth_sigma, omt_power).scaled(self.int_scale)
            # with spr='kernel' the kernel weights come from the SPR map, not from a logits head
            self.weight_head = (
                (weight_head if weight_head is not None else KernelWeightHead(len(sigmas))) if spr != 'kernel' else None
            )
        else:
            self.kernel = FourierKernel(
                kernel=kernel, alpha=alpha, gamma=gamma, power=power, sigma=sigma, weights=kernel_weights
            ).scaled(self.int_scale)
            self.weight_head = None
        if spr is not None and spr_source == 'head':
            self.spr_head = spr_head if spr_head is not None else SPRHead()
        else:
            self.spr_head = None
        self._integrators = nn.ModuleDict()
        self.int_size = None  # set after the first forward pass
        self.field_scale = nn.Parameter(torch.tensor(float(field_scale))) if field_scale is not None else None
        self.return_path = return_path

    @property
    def spatially_varying(self) -> bool:
        """Whether ``self.kernel`` is a :class:`SpatiallyVaryingKernel`."""
        return isinstance(self.kernel, SpatiallyVaryingKernel)

    def predict_weights(self, mov, fix, field, lam=None):
        """Kernel weights w_0 [B, n_kernels, ...] at the resolution of ``field`` (spatially varying only)."""
        if self.spr == 'kernel':
            return self.kernel.weights_from_spr(lam)
        return self.kernel.weights_from_logits(self.weight_head(mov, fix, field))

    # -- helpers -------------------------------------------------------------------------- #
    def _get_integrator(self, shape, device):
        key = 'x'.join(str(int(s)) for s in shape)
        if key not in self._integrators:
            self._integrators[key] = LDDMMIntegrator(shape, self.time_steps, self.kernel)
        self.int_size = tuple(shape)
        return self._integrators[key].to(device)

    @property
    def integrator(self):
        """The integrator used in the last forward pass (``None`` before the first call)."""
        if self.int_size is None:
            return None
        return self._integrators['x'.join(str(s) for s in self.int_size)]

    def predict_field(self, mov, fix, with_spr=False):
        """Run the backbone and return its dense 3-channel field (and the SPR map if requested).

        With ``spr_source='backbone'`` the adapter output must be ``(field, lam)``.
        """
        out = self.adapter(self.backbone, mov, fix)
        lam = None
        if self.spr is not None and self.spr_source == 'backbone':
            if not isinstance(out, (tuple, list)) or len(out) < 2:
                raise ValueError("spr_source='backbone' expects the backbone to return (field, lam)")
            field, lam = out[0], out[1]
            if not torch.is_tensor(lam) or lam.dim() != 5 or lam.shape[1] != 1:
                raise ValueError('The SPR map must have shape [B, 1, D, H, W]')
            lam = torch.clamp(lam.float(), 1e-6, 1.0)
        elif isinstance(out, (tuple, list)):
            field = out[-1]
        else:
            field = out
        if not torch.is_tensor(field) or field.dim() != 5 or field.shape[1] != 3:
            raise ValueError(
                'The backbone must return a [B, 3, D, H, W] field (got '
                f'{type(field).__name__}{tuple(field.shape) if torch.is_tensor(field) else ""})'
            )
        if self.field_scale is not None:
            field = field * self.field_scale
        if not with_spr:
            return field
        if self.spr is not None and self.spr_source == 'head':
            lam = self.spr_head(mov, fix, field)
        return field, lam

    # -- forward -------------------------------------------------------------------------- #
    def forward(self, inputs, return_dict=False):
        """Predict the field with the backbone and integrate it to a diffeomorphism.

        Args:
            inputs: Tuple ``(mov, fix)`` of tensors [B, 1, D, H, W].
            return_dict: If True, return the full output dict (see the class docstring) instead of the flow.

        Returns:
            The forward displacement field ``flow`` [B, 3, D', H', W'] or, with ``return_dict=True``, the dict
            produced by :meth:`LDDMMIntegrator.shoot` / :meth:`LDDMMIntegrator.relax` extended with
            ``momentum`` (shooting) or ``pre_velocities`` (tvf) and, when applicable, ``kernel_weights``,
            ``omt`` and ``spr_weight``.

        Raises:
            ValueError: With ``param='tvf'``, if the backbone field resolution differs from that of ``mov``.
        """
        mov, fix = inputs
        if self.param == 'shooting':
            m0, lam = self.predict_field(mov, fix, with_spr=True)
            w0 = self.predict_weights(mov, fix, m0, lam) if self.spatially_varying else None
            if self.int_scale > 1:
                m0 = (
                    F.interpolate(m0, scale_factor=self.int_scale, mode='trilinear', align_corners=False)
                    * self.int_scale
                )
                if w0 is not None:
                    w0 = F.interpolate(w0, scale_factor=self.int_scale, mode='trilinear', align_corners=False)
                if lam is not None:
                    lam = F.interpolate(lam, scale_factor=self.int_scale, mode='trilinear', align_corners=False)
            integ = self._get_integrator(m0.shape[2:], m0.device)
            out = integ.shoot(
                m0, weights=w0, energy_weight=lam if self.spr == 'energy' else None, return_path=self.return_path
            )
            out['momentum'] = m0
        else:
            p0, lam = self.predict_field(mov, fix, with_spr=True)
            if tuple(p0.shape[2:]) != tuple(mov.shape[2:]):
                raise ValueError("param='tvf' requires the backbone field to have the resolution of the input images")
            w0 = self.predict_weights(mov, fix, p0, lam) if self.spatially_varying else None
            integ = self._get_integrator(p0.shape[2:], p0.device)
            pre_vel = [p0]

            def predict(t, u):
                if t == 0:
                    return p0
                p_t = self.predict_field(integ.transformer(mov, u), fix)
                pre_vel.append(p_t)
                return p_t

            out = integ.relax(
                predict, weights=w0, energy_weight=lam if self.spr == 'energy' else None, return_path=self.return_path
            )
            out['pre_velocities'] = pre_vel
        if w0 is not None:
            out['kernel_weights'] = w0
            if self.spr != 'kernel':
                out['omt'] = self.kernel.omt(w0)
        if lam is not None:
            out['spr_weight'] = lam
        if return_dict:
            return out
        return out['flow']


def load_backbone_weights(model, state_dict, verbose=True):
    """Copy every tensor of ``state_dict`` whose name and shape match into ``model``.

    Useful to initialize the backbone of an LDDMM model from a checkpoint of the plain
    registration network (e.g. a TransMorph / VFA checkpoint); non-matching tensors keep their
    random initialization.  For an :class:`LDDMMWrapper` pass ``model.backbone``.

    Returns:
        (number of tensors loaded, number of tensors in the model).

    """
    own = model.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(matched, strict=False)
    if verbose:
        skipped = [k for k in own if k not in matched]
        print(
            f'Loaded {len(matched)}/{len(own)} tensors from checkpoint; '
            f'{len(skipped)} left at random init (e.g. {skipped[:3]})'
        )
    return len(matched), len(own)
