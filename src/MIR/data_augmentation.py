"""Data augmentation utilities for MIR training and preprocessing."""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F


def _should_transform_labels(im_label_1, im_label_2) -> bool:
    """Return whether both label tensors are present for paired augmentation."""
    return im_label_1 is not None and im_label_2 is not None


def _normalize_to_unit_interval(image: torch.Tensor, lower_q: float, upper_q: float) -> torch.Tensor:
    """Normalize an image to ``[0, 1]`` using robust quantiles."""
    value_min = torch.quantile(image, lower_q)
    value_max = torch.quantile(image, upper_q)
    scale = (value_max - value_min).clamp_min(torch.finfo(image.dtype).eps)
    normalized = (image - value_min) / scale
    return torch.clamp(normalized, 0, 1)


def _normalize_from_minmax(image: torch.Tensor) -> torch.Tensor:
    """Normalize an image to ``[0, 1]`` using its min and max values."""
    value_min = image.min()
    value_max = image.max()
    scale = (value_max - value_min).clamp_min(torch.finfo(image.dtype).eps)
    normalized = (image - value_min) / scale
    return torch.clamp(normalized, 0, 1)


def _random_triplet(value_range):
    """Sample three random values from a sequence of ranges."""
    return [random.uniform(*bounds) for bounds in value_range]


def _build_affine_matrix_3d(
    angle_xyz,
    scale_xyz,
    trans_xyz,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Construct a batched 3D affine matrix in homogeneous coordinates."""
    angle_x, angle_y, angle_z = angle_xyz
    scale_x, scale_y, scale_z = scale_xyz
    trans_x, trans_y, trans_z = trans_xyz

    rotation_x = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, math.cos(angle_x), -math.sin(angle_x), 0.0],
            [0.0, math.sin(angle_x), math.cos(angle_x), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    ).unsqueeze(0)

    rotation_y = torch.tensor(
        [
            [math.cos(angle_y), 0.0, math.sin(angle_y), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-math.sin(angle_y), 0.0, math.cos(angle_y), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    ).unsqueeze(0)

    rotation_z = torch.tensor(
        [
            [math.cos(angle_z), -math.sin(angle_z), 0.0, 0.0],
            [math.sin(angle_z), math.cos(angle_z), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    ).unsqueeze(0)

    scale_translate = torch.tensor(
        [
            [1.0 + scale_x, 0.0, 0.0, trans_x],
            [0.0, 1.0 + scale_y, 0.0, trans_y],
            [0.0, 0.0, 1.0 + scale_z, trans_z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    ).unsqueeze(0)

    affine_matrix = torch.matmul(rotation_x, rotation_y)
    affine_matrix = torch.matmul(affine_matrix, rotation_z)
    affine_matrix = torch.matmul(affine_matrix, scale_translate)
    return affine_matrix


def flip_aug(im_1, im_2, im_label_1=None, im_label_2=None):
    """Flip paired images along randomly selected spatial axes."""
    with torch.no_grad():
        has_labels = _should_transform_labels(im_label_1, im_label_2)

        for dim in (2, 3, 4):
            if not np.random.choice([True, False]):
                continue

            im_1 = torch.flip(im_1, dims=[dim])
            im_2 = torch.flip(im_2, dims=[dim])
            if has_labels:
                im_label_1 = torch.flip(im_label_1, dims=[dim])
                im_label_2 = torch.flip(im_label_2, dims=[dim])

        if has_labels:
            return im_1, im_2, im_label_1, im_label_2
        return im_1, im_2

def affine_aug(im, im_label=None, seed=10, angle_range=10, trans_range=0.1, scale_range=0.1):
    """Apply a random 3D affine transformation to an image and optional label."""
    with torch.no_grad():
        if seed is not None:
            random.seed(seed)

        angle_xyz = tuple(
            random.uniform(-angle_range, angle_range) * math.pi / 180 for _ in range(3)
        )
        scale_xyz = tuple(random.uniform(-scale_range, scale_range) for _ in range(3))
        trans_xyz = tuple(random.uniform(-trans_range, trans_range) for _ in range(3))

        theta_final = _build_affine_matrix_3d(
            angle_xyz,
            scale_xyz,
            trans_xyz,
            device=im.device,
            dtype=im.dtype,
        )
        output_grid = F.affine_grid(theta_final[:, 0:3, :], im.shape, align_corners=False)

        im = F.grid_sample(im, output_grid, mode='bilinear', padding_mode='border', align_corners=False)

        if im_label is not None:
            im_label = F.grid_sample(
                im_label,
                output_grid,
                mode='nearest',
                padding_mode='border',
                align_corners=False,
            )
            return im, im_label
        return im


# --------------------------------------------------------------------------
# helper 1 :   1‑D three‑class Gaussian mixture (EM, 20 iters)
# --------------------------------------------------------------------------
def gmm_soft_masks(brain_vox, iters: int = 40):
    """Estimate three soft tissue masks with a 1D Gaussian mixture model."""
    x = brain_vox[:, None]

    centers = torch.linspace(
        torch.quantile(x, 0.05),
        torch.quantile(x, 0.95),
        3,
        device=x.device,
    )[:, None]
    labels = torch.argmin(torch.cdist(x, centers), dim=1)

    mixture_weights = []
    means = []
    variances = []
    for cluster_index in range(3):
        cluster_values = x[labels == cluster_index]
        if cluster_values.numel() == 0:
            cluster_values = x
        mixture_weights.append((labels == cluster_index).float().mean())
        means.append(cluster_values.mean())
        variances.append(cluster_values.var(unbiased=False).clamp_min(1e-4))

    pi = torch.stack(mixture_weights)
    mu = torch.stack(means)
    var = torch.stack(variances)

    num_voxels = x.shape[0]
    for _ in range(iters):
        pdf = torch.exp(-0.5 * (x - mu) ** 2 / var) / torch.sqrt(2 * torch.pi * var)
        pdf = (pi[:, None] * pdf.T).clamp_min(1e-12)
        post = pdf / pdf.sum(0, keepdim=True)

        Nk = post.sum(1).clamp_min(1e-12)
        pi = Nk / num_voxels
        mu = (post * x.T).sum(1) / Nk
        var = (post * (x.T - mu[:, None]) ** 2).sum(1) / Nk
        var = var.clamp_min(1e-4)

    order = torch.argsort(mu)
    return post[order].T.reshape(brain_vox.shape + (-1,))

def fcm_soft_masks(
    brain_vox: torch.Tensor,
    n_clusters: int = 3,
    m: float = 2.,
    iters: int = 30,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Fuzzy C-Means on 1-D intensities, k-means init, correct broadcasting.

    brain_vox : Tensor, shape (N,) of [0,1] intensities
    n_clusters: number of tissue classes (3→CSF/GM/WM)
    m         : fuzziness exponent (>1, larger→softer)
    iters     : EM updates
    eps       : small constant to avoid zeros
    Returns U of shape (N, n_clusters) with soft memberships.
    """

    x = brain_vox.flatten().to(brain_vox.device)
    x = x.unsqueeze(1)

    qs = torch.quantile(x.flatten(), torch.linspace(0.05, 0.95, n_clusters, device=x.device))
    centers = qs.unsqueeze(0)

    dist = torch.abs(x - centers).clamp_min(eps)
    exponent = 2.0 / (m - 1.0)
    ratio = dist.unsqueeze(2) / dist.unsqueeze(1)
    denom = (ratio ** exponent).sum(dim=2)
    U = 1.0 / denom

    for _ in range(iters):
        Um = U ** m
        centers = (Um * x).sum(dim=0, keepdim=True) / Um.sum(dim=0, keepdim=True).clamp_min(eps)

        dist = torch.abs(x - centers).clamp_min(eps)
        ratio = dist.unsqueeze(2) / dist.unsqueeze(1)
        denom = (ratio ** exponent).sum(dim=2)
        U = 1.0 / denom

    centers_flat = centers.flatten()
    order = torch.argsort(centers_flat)
    U = U[:, order]

    return U

def mask_edge(mask: torch.BoolTensor, kernel_size: int = 3) -> torch.BoolTensor:
    """Compute the one-voxel edge of a 3D binary mask."""
    if mask.dtype != torch.bool:
        raise ValueError('mask must be a BoolTensor')
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError('kernel_size must be an odd positive integer')

    m = mask.float().unsqueeze(0).unsqueeze(0)
    inv = 1.0 - m
    pad = kernel_size // 2
    dilated = F.max_pool3d(inv, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = (dilated < 1.0).squeeze(0).squeeze(0)
    edge = mask & ~eroded
    return edge[None, None]


def fill_holes_torch(mask: torch.BoolTensor, closing_size: int = 3) -> torch.BoolTensor:
    """Remove small gaps and fill internal holes in a 3D mask."""
    if mask.dtype != torch.bool:
        raise ValueError('mask must be a BoolTensor')
    if closing_size % 2 == 0 or closing_size < 1:
        raise ValueError('closing_size must be an odd positive integer')

    squeeze_result = False
    if mask.dim() == 3:
        mask = mask.unsqueeze(0).unsqueeze(0)
        squeeze_result = True
    elif mask.dim() != 5:
        raise ValueError('mask must have shape (D, H, W) or (B, C, D, H, W)')

    m = mask.float()
    k = closing_size
    pad = k // 2
    dil = F.max_pool3d(m, kernel_size=k, stride=1, padding=pad)
    inv = 1.0 - dil
    er_inv = F.max_pool3d(inv, kernel_size=k, stride=1, padding=pad)
    closed = (1.0 - (er_inv > 0).float()).bool()

    inv_closed = ~closed

    outside = torch.zeros_like(inv_closed)
    outside[:, :, 0, ...] = inv_closed[:, :, 0, ...]
    outside[:, :, -1, ...] = inv_closed[:, :, -1, ...]
    outside[:, :, :, 0, :] = inv_closed[:, :, :, 0, :]
    outside[:, :, :, -1, :] = inv_closed[:, :, :, -1, :]
    outside[:, :, :, :, 0] = inv_closed[:, :, :, :, 0]
    outside[:, :, :, :, -1] = inv_closed[:, :, :, :, -1]

    curr = outside.clone()
    while True:
        dilated = F.max_pool3d(curr.float(), 3, 1, 1).bool()
        dilated = dilated & inv_closed
        if torch.equal(dilated, curr):
            break
        curr = dilated

    holes = inv_closed & ~curr
    filled = closed | holes
    if squeeze_result:
        return filled.squeeze(0).squeeze(0)
    return filled


class SimulatedWidthThickness(torch.nn.Module):
    """Simulate coarse resolution along the width dimension."""

    def __init__(self, min_factor=2, max_factor=5, p=0.5):
        super().__init__()
        if min_factor < 1 or max_factor < min_factor:
            raise ValueError('Invalid factor range')
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (C, D, H, W) or (B, C, D, H, W)
        Returns:
            Tensor of same shape with simulated coarse width resolution
        """
        if x.dim() not in (4, 5):
            raise ValueError('x must have shape (C, D, H, W) or (B, C, D, H, W)')

        squeeze_batch = False
        if x.dim() == 4:
            x = x.unsqueeze(0)
            squeeze_batch = True

        if not self.training or random.random() > self.p:
            return x.squeeze(0) if squeeze_batch else x

        _, _, depth, height, width = x.shape
        factor = random.randint(self.min_factor, self.max_factor)

        pad_w = (factor - (width % factor)) % factor
        x_padded = F.pad(x, (0, pad_w, 0, 0, 0, 0))

        pooled = F.avg_pool3d(
            x_padded,
            kernel_size=(1, 1, factor),
            stride=(1, 1, factor),
            padding=0,
        )

        x_coarse = F.interpolate(
            pooled,
            size=(depth, height, width),
            mode='trilinear',
            align_corners=False,
        )
        if squeeze_batch:
            return x_coarse.squeeze(0)
        return x_coarse

def random_contrast_lut(
    img: torch.Tensor,
    num_bins: int = 128,
    sigma_bins: float = 128 // 10,
    chunk_size: int = 64,
    use_half: bool = True,
) -> torch.Tensor:
    """Apply a smooth random LUT to a 2D or 3D image tensor in ``[0, 1]``."""

    device = img.device
    dtype = img.dtype
    idx_dtype = torch.float16 if use_half else torch.float32

    lut = torch.rand(num_bins, device=device)

    k = int(6 * sigma_bins + 1) | 1
    xs = torch.arange(k, device=device) - (k // 2)
    gauss = torch.exp(-0.5 * (xs / sigma_bins) ** 2)
    gauss /= gauss.sum()
    lut = lut.view(1, 1, -1)
    gauss = gauss.view(1, 1, -1)
    lut_smooth = F.conv1d(F.pad(lut, (k // 2,) * 2), gauss).view(-1)
    lut_smooth = _normalize_from_minmax(lut_smooth)
    lut_smooth[0] = 0.0
    lut_smooth[num_bins - 1] = 1.0
    if use_half:
        lut_smooth = lut_smooth.half()

    out = torch.empty_like(img)

    height = img.shape[-2]
    for start_idx in range(0, height, chunk_size):
        end_idx = min(start_idx + chunk_size, height)
        patch = img[..., start_idx:end_idx, :]
        idx = (patch * (num_bins - 1)).to(idx_dtype)
        lo = idx.floor().long().clamp(0, num_bins - 2)
        hi    = lo + 1
        w = idx - lo.to(idx_dtype)

        v_lo = lut_smooth[lo]
        v_hi  = lut_smooth[hi]
        mapped = v_lo + (v_hi - v_lo) * w

        out[..., start_idx:end_idx, :] = mapped.to(dtype)

    return out

class RandomMultiContrastRemap(torch.nn.Module):
    """
    Augment a skull‑stripped T1‑w brain MRI so that the same anatomy can resemble
    T2, FLAIR, T2*, or CT.  Background stays zero; brain voxels are re‑normalized
    to [0,1] at the end.

    Args:
        p: Probability of applying the augmentation
        noise_std: Standard deviation of Gaussian noise added to the image
        downsample_p: Probability of applying simulated coarse width resolution
    """

    def __init__(
        self,
        p=0.9,
        noise_std=0.03,
        downsample_p=0.5,
        recipe=['t2', 'flair', 't2star', 'pd', 'ct', 'random_remap', 'random_lut'],
    ):
        super().__init__()
        self.p = p
        self.noise_std = noise_std
        self.slice_down = SimulatedWidthThickness(4, 8, p=downsample_p)
        self.recipe_list = list(recipe)

    def _gamma_map(self, x, g):
        return x.pow(g)

    def _invert(self, x):
        return 1.0 - x

    def _piecewise_linear(self, x, knots, values):
        """
        Safe piece-wise linear mapping that never indexes out of bounds.
        x, knots, values : 1-D (knots and values same length, strictly increasing knots)
        """
        idx = torch.searchsorted(knots, x)
        idx = torch.clamp(idx, max=len(knots) - 1)

        idx0 = torch.clamp(idx - 1, min=0)
        x0, x1 = knots[idx0], knots[idx]
        y0, y1 = values[idx0], values[idx]

        t = (x - x0) / (x1 - x0 + 1e-6)
        return torch.lerp(y0, y1, t)

    def _poly_bias(self, shape, order, device, strength):
        z, y, x = [torch.linspace(-1, 1, size, device=device) for size in shape]
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        grid = torch.stack([xx, yy, zz])
        field = torch.zeros_like(grid[0])
        for order_idx in range(1, order + 1):
            field += torch.randn(1, device=device) * (
                grid[0] ** order_idx + grid[1] ** order_idx + grid[2] ** order_idx
            )
        field = 1.0 + strength * field / field.abs().max().clamp_min(1e-6)
        return field

    def _gmm_remap(self, image, mask_bool, value_ranges, fuzzy_range=(0.4, 0.6)):
        """Remap masked voxels by Gaussian-mixture soft assignments."""
        voxels = image[0, 0][mask_bool]
        fuzzy_factor = random.uniform(*fuzzy_range)
        post = gmm_soft_masks(voxels) ** fuzzy_factor
        targets = torch.tensor(_random_triplet(value_ranges), device=voxels.device, dtype=voxels.dtype)
        image[0, 0][mask_bool] = (post * voxels[:, None] * targets).sum(1)
        return image

    def _normalize_brain(self, image):
        """Apply robust normalization used by the recipe transforms."""
        return _normalize_to_unit_interval(image, 0.05, 0.95)

    def forward(self, img_, mask):
        mask = fill_holes_torch(mask, closing_size=5)
        img = img_.clone()
        if random.random() > self.p:
            return img * mask.float()

        single = img.dim() == 4
        if single:
            img, mask = img.unsqueeze(0), mask.unsqueeze(0)

        img = img.clone()
        _, _, _, _, _ = img.shape
        device = img.device
        mask_bool = mask.bool()
        brain_mask = mask[0, 0].bool()

        recipe = random.choice(self.recipe_list)

        if recipe == 't2':
            img = self._gmm_remap(img, brain_mask, [(4, 5), (0.3, 0.35), (0.15, 0.25)])
            img = self._normalize_brain(img)

        elif recipe == 'flair':
            img = self._gmm_remap(img, brain_mask, [(0.1, 0.1), (0.3, 0.3), (0.1, 0.1)])
            img = self._normalize_brain(img)

        elif recipe == 't2star':
            img = self._gmm_remap(img, brain_mask, [(3, 4), (0.7, 0.9), (0.3, 0.4)])
            img = self._normalize_brain(img)

        elif recipe == 'ct':
            img = self._gmm_remap(img, brain_mask, [(0.06, 0.1), (0.06, 0.08), (0.025, 0.04)])
            edge = mask_edge(mask[0, 0], kernel_size=3)
            thickness = 1
            k = 2 * thickness + 1
            e = edge.float()
            e = F.max_pool3d(e, kernel_size=k, stride=1, padding=thickness) > 0
            edge = e.squeeze(0).squeeze(0)

            channel = img[0, 0]
            channel[edge] = 0.1 + channel[edge]
            img[0, 0] = channel
            img = self._normalize_brain(img)

        elif recipe == 'pd':
            img[mask_bool] = self._invert(img[mask_bool])
            g = random.uniform(1.1, 1.6)
            img[mask_bool] = self._gamma_map(img[mask_bool], g)
            knots = torch.tensor([0.0, 0.3, 0.8, 1.0], device=device)
            values = torch.tensor([0.0, 0.35, 0.85, 1.0], device=device)
            img[mask_bool] = self._piecewise_linear(img[mask_bool], knots, values)
            img = self._normalize_brain(img)

        elif recipe == 'random_remap':
            img = self._gmm_remap(img, brain_mask, [(0.01, 1), (0.01, 1), (0.01, 1)])
            img = self._normalize_brain(img)
            if random.random() > 0.5:
                img[mask_bool] = self._invert(img[mask_bool])

        elif recipe == 'random_lut':
            img = _normalize_from_minmax(img)
            img = random_contrast_lut(img, num_bins=256)
            img = _normalize_from_minmax(img)

        if self.noise_std > 0:
            sigma = random.uniform(0, self.noise_std)
            img[mask_bool] += sigma * torch.randn_like(img)[mask_bool]

        img = _normalize_to_unit_interval(img, 0.01, 0.99)
        img[~mask_bool] = 0
        img = torch.nan_to_num(img)
        if single:
            img = img.squeeze(0)
        if recipe in ('t2star', 'flair'):
            img = self.slice_down(img)
        return img
        