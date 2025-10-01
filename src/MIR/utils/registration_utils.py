import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
import ants
import nibabel as nib
import os
from scipy.ndimage import zoom

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
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

        return nnf.grid_sample(src, new_locs, align_corners=False, mode=self.mode)
    
class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

def make_affine_from_pixdim(pixdim):
    # Create a 4x4 affine with spacing along the diagonal
    affine = np.eye(4)
    affine[0, 0] = pixdim[0]
    affine[1, 1] = pixdim[1]
    affine[2, 2] = pixdim[2]
    return affine

def get_pixdim(nib_img):
    return nib_img.header['pixdim'][1:4].astype(float)

def resample_by_pixdim(arr, from_pixdim, to_pixdim, order):
    ratio = np.array(from_pixdim, dtype=float) / np.array(to_pixdim, dtype=float)
    return zoom(arr, ratio, order=order)

def crop_or_pad(volume: np.ndarray,
                target_shape,
                pad_value: float = 0) -> np.ndarray:
    """
    Center-crop or center-pad a NumPy array to exactly match target_shape.

    Args:
        volume: Input array of shape (d0, d1, ..., dN).
        target_shape: Desired shape (t0, t1, ..., tN).
        pad_value: Constant value to use for padding.

    Returns:
        A new array of shape target_shape.
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
            output = np.pad(output,
                            pad_width=pad_width,
                            mode="constant",
                            constant_values=pad_value)

        elif cur > tgt:
            # need to crop
            start = (cur - tgt) // 2
            end = start + tgt
            slicer = [slice(None)] * ndim
            slicer[axis] = slice(start, end)
            output = output[tuple(slicer)]

        # if cur == tgt, nothing to do on this axis

    return output

def resample_to_orginal_space_and_save(deformed_img, ants_affine_mat_path, img_orig_path, out_back_dir, img_pixdim, if_flip=True, flip_axis=1, interpolater='nearestNeighbor'):
    '''
    Resample a deformed image to the original image space and save it.
    Args:
        deformed_img: The deformed image tensor.
        ants_affine_mat_path: Path to the ANTs affine matrix.
        img_orig_path: Path to the original image in .nii.gz.
        out_back_dir: Output directory to save the resampled image.
        img_pixdim: Pixel dimensions of the deformed image.
        if_flip: Whether to flip the image along the specified axis.
        flip_axis: Axis along which to flip the image if if_flip is True.
        interpolater: Interpolator type for ANTs resampling. Options are 'linear', 'nearestNeighbor', 'bSpline' etc.
    Returns:
        img_final: The final resampled image in the original space.
    '''
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
    orig_img_nib  = nib.load(img_orig_path)
    orig_affine  = orig_img_nib.affine
    orig_shape   = orig_img_nib.shape
    orig_pixdim  = get_pixdim(orig_img_nib)

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
    img_pre_im    = ants.from_numpy(img_pre.astype(np.float32))
    img_on_ctpre = ants.apply_transforms(
        fixed=img_pre_im,
        moving=mov_img_im,
        transformlist=[ants_affine_mat_path],
        interpolator=interpolater,
        whichtoinvert=[True]  # forward transform
    ).numpy()

    # Undo preprocessing to original Image space:
    #   unflip, resample spacing back, center crop or pad, save with original affine
    if if_flip:
        # Flip back to original orientation
        img_unflipped = np.flip(img_on_ctpre, flip_axis)
    else:
        img_unflipped = img_on_ctpre
    img_orig_sp   = resample_by_pixdim(
        img_unflipped, from_pixdim=target_pixdim, to_pixdim=orig_pixdim, order=order
    )
    if order == 0:
        img_final = crop_or_pad(img_orig_sp, orig_shape, pad_value=img_orig_sp.min()).astype(np.int16)
    else:
        img_final = crop_or_pad(img_orig_sp, orig_shape, pad_value=img_orig_sp.min()).astype(np.float32)

    nib.save(nib.Nifti1Image(img_final, orig_affine), out_back_dir)
    return img_final