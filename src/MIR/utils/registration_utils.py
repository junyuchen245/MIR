"""Registration utility functions and spatial transformer helpers."""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
import ants
import nibabel as nib
from scipy.ndimage import zoom


class SpatialTransformer(nn.Module):
    """N-D spatial transformer that warps an image with a voxel displacement field.

    Adapted from https://github.com/voxelmorph/voxelmorph. Same as
    :class:`MIR.models.registration_utils.SpatialTransformer` but without a ``padding_mode`` argument
    (``grid_sample`` uses zero padding).

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
        # See MIR.models.registration_utils.SpatialTransformer: drop stale grids stored by older checkpoints.
        state_dict.pop(prefix + 'grid', None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp a source image with a displacement field.

        Args:
            src: Source tensor ``[B, C, H, W]`` or ``[B, C, H, W, D]``.
            flow: Displacement field in voxels, ``[B, ndim, H, W]`` or ``[B, ndim, H, W, D]``.

        Returns:
            Warped source tensor.

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

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


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


def make_affine_from_pixdim(pixdim: Sequence[float]) -> np.ndarray:
    """Create a 4x4 affine matrix from pixel spacing.

    Args:
        pixdim: Sequence of spacing values (dx, dy, dz).

    Returns:
        4x4 float64 affine with the spacings on the diagonal and zero translation.

    """
    # Create a 4x4 affine with spacing along the diagonal
    affine = np.eye(4)
    affine[0, 0] = pixdim[0]
    affine[1, 1] = pixdim[1]
    affine[2, 2] = pixdim[2]
    return affine


def get_pixdim(nib_img: nib.Nifti1Image) -> np.ndarray:
    """Extract voxel spacing from a nibabel image.

    Args:
        nib_img: nibabel image object with a NIfTI-style header (``header['pixdim']``).

    Returns:
        Pixel spacing float array ``(dx, dy, dz)`` taken from ``header['pixdim'][1:4]``.

    """
    return nib_img.header['pixdim'][1:4].astype(float)


def resample_by_pixdim(
    arr: np.ndarray, from_pixdim: Sequence[float], to_pixdim: Sequence[float], order: int
) -> np.ndarray:
    """Resample a numpy array by voxel spacing with ``scipy.ndimage.zoom``.

    Args:
        arr: Input numpy array.
        from_pixdim: Source spacing.
        to_pixdim: Target spacing; the zoom factor is ``from_pixdim / to_pixdim``.
        order: Spline interpolation order.

    Returns:
        Resampled numpy array.

    """
    ratio = np.array(from_pixdim, dtype=float) / np.array(to_pixdim, dtype=float)
    return zoom(arr, ratio, order=order)


def crop_or_pad(volume: np.ndarray, target_shape: Sequence[int], pad_value: float = 0) -> np.ndarray:
    """Center-crop or center-pad a NumPy array to exactly match target_shape.

    Args:
        volume: Input array of shape (d0, d1, ..., dN).
        target_shape: Desired shape (t0, t1, ..., tN).
        pad_value: Constant value to use for padding.

    Returns:
        An array of shape target_shape (may be a view of ``volume`` if only cropping was needed).

    Raises:
        ValueError: If ``len(target_shape)`` differs from ``volume.ndim``.

    """
    output = volume
    ndim = output.ndim

    if ndim != len(target_shape):
        raise ValueError(f"volume.ndim={ndim} but target_shape has length {len(target_shape)}")

    for axis, tgt in enumerate(target_shape):
        cur = output.shape[axis]

        if cur < tgt:
            # need to pad
            total_pad = tgt - cur
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pad_width = [(0, 0)] * ndim
            pad_width[axis] = (pad_before, pad_after)
            output = np.pad(output, pad_width=pad_width, mode="constant", constant_values=pad_value)

        elif cur > tgt:
            # need to crop
            start = (cur - tgt) // 2
            end = start + tgt
            slicer = [slice(None)] * ndim
            slicer[axis] = slice(start, end)
            output = output[tuple(slicer)]

        # if cur == tgt, nothing to do on this axis

    return output


def resample_to_orginal_space_and_save(
    deformed_img: torch.Tensor,
    ants_affine_mat_path: str,
    img_orig_path: str,
    out_back_dir: str,
    img_pixdim: Sequence[float],
    if_flip: bool = True,
    flip_axis: int = 1,
    interpolater: str = 'nearestNeighbor',
) -> np.ndarray:
    """Resample a deformed image back to the original image space and save it as NIfTI.

    The preprocessing grid is reconstructed by resampling the original image to ``img_pixdim`` (and flipping it
    along axis 1 if ``if_flip``). ``deformed_img[0, 0]`` is mapped onto that grid with the inverse of the ANTs affine,
    flipped back along ``flip_axis``, resampled to the original spacing, center-cropped/padded to the original shape
    (padding with the image minimum) and saved with the original affine.

    Args:
        deformed_img: Deformed image tensor (B, 1, H, W, D); only the first batch element is used.
        ants_affine_mat_path: Path to the ANTs affine matrix.
        img_orig_path: Path to the original image (.nii.gz).
        out_back_dir: Output file path for the resampled image (passed to ``nib.save``).
        img_pixdim: Pixel spacing used during preprocessing.
        if_flip: Whether to flip the image along the specified axis.
        flip_axis: Axis along which to flip the image back if if_flip is True.
        interpolater: ANTs interpolator ('linear', 'nearestNeighbor', 'bSpline'); also selects the scipy zoom order
            (0, 1 or 3).

    Returns:
        img_final: The resampled image in the original space (int16 for ``'nearestNeighbor'``, float32 otherwise).

    Raises:
        ValueError: If ``interpolater`` is not supported.

    """
    if interpolater == 'nearestNeighbor':
        order = 0
    elif interpolater == 'linear':
        order = 1
    elif interpolater == 'bSpline':
        order = 3
    else:
        raise ValueError(f"Unsupported interpolater: {interpolater}")
    # deformed_img is on the atlas grid at this point
    deformed_img_np = deformed_img.cpu().numpy()[0, 0]  # [H,W,D] labels

    # Paths to original Image and inverse affine
    orig_img_nib = nib.load(img_orig_path)
    orig_affine = orig_img_nib.affine
    orig_shape = orig_img_nib.shape
    orig_pixdim = get_pixdim(orig_img_nib)

    # Match the spacing you used in preprocessing
    target_pixdim = np.array(img_pixdim, dtype=float)

    # Reconstruct the pre-affine Image grid used in preprocessing:
    #   resample original Image to target spacing, then flip along axis 1
    img_pre = resample_by_pixdim(
        orig_img_nib.get_fdata(), from_pixdim=orig_pixdim, to_pixdim=target_pixdim, order=order
    )
    if if_flip:
        img_pre = np.flip(img_pre, 1)

    # Atlas → pre-affine Image grid via inverse ANTs affine (nearest for labels)
    mov_img_im = ants.from_numpy(deformed_img_np.astype(np.float32))
    img_pre_im = ants.from_numpy(img_pre.astype(np.float32))
    img_on_ctpre = ants.apply_transforms(
        fixed=img_pre_im,
        moving=mov_img_im,
        transformlist=[ants_affine_mat_path],
        interpolator=interpolater,
        whichtoinvert=[True],  # inverse of the stored (image -> atlas) affine
    ).numpy()

    # Undo preprocessing to original Image space:
    #   unflip, resample spacing back, center crop or pad, save with original affine
    if if_flip:
        # Flip back to original orientation
        img_unflipped = np.flip(img_on_ctpre, flip_axis)
    else:
        img_unflipped = img_on_ctpre
    img_orig_sp = resample_by_pixdim(img_unflipped, from_pixdim=target_pixdim, to_pixdim=orig_pixdim, order=order)
    if order == 0:
        img_final = crop_or_pad(img_orig_sp, orig_shape, pad_value=img_orig_sp.min()).astype(np.int16)
    else:
        img_final = crop_or_pad(img_orig_sp, orig_shape, pad_value=img_orig_sp.min()).astype(np.float32)

    nib.save(nib.Nifti1Image(img_final, orig_affine), out_back_dir)
    return img_final
