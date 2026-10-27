"""Digital diffeomorphism metrics and Jacobian determinant utilities.

Finite-difference Jacobian determinants of a 3-D transformation under several stencils, and the non-diffeomorphic
voxel / tetrahedra / volume measures built from them (digital diffeomorphism criteria). Can also be run as a script
on a ``.npz`` transformation.
"""

from typing import Dict, Tuple

import nibabel as nib
import numpy as np
import scipy.ndimage
import argparse


def calc_J_i(trans: np.ndarray, grad_args: str) -> np.ndarray:
    """Compute the Jacobian determinant using a specific finite-difference stencil.

    Each axis is differentiated with a forward (``'+'``), backward (``'-'``) or central (``'0'``) difference
    (``scipy.ndimage.correlate``, ``mode='nearest'``); the one-voxel boundary is removed from the result.

    Args:
        trans: Deformation (absolute voxel coordinate) field (3, H, W, D), i.e. identity + displacement.
        grad_args: Gradient stencil spec, a direction (``'+'``, ``'-'``, ``'0'``) per axis ``x, y, z``
            (e.g., '0x0y0z', '+x-y+z').

    Returns:
        Jacobian determinant volume (H-2, W-2, D-2).

    """
    kernel = {}
    grad_args = list(grad_args)
    grad_args = [(grad_args[i] + grad_args[i + 1]) for i in range(0, len(grad_args), 2)]

    # example: [('-', 'x'), ('+', 'y'), ('-', 'z')]
    for direction, axis in grad_args:
        if direction == '+':
            kernel[axis] = np.array([0, -1, 1])
        elif direction == '0':
            kernel[axis] = np.array([-0.5, 0, 0.5])
        elif direction == '-':
            kernel[axis] = np.array([-1, 1, 0])
    kernel['x'] = kernel['x'].reshape(1, 3, 1, 1)
    kernel['y'] = kernel['y'].reshape(1, 1, 3, 1)
    kernel['z'] = kernel['z'].reshape(1, 1, 1, 3)

    trans = trans[None, ...]
    gradx = np.stack(
        [
            scipy.ndimage.correlate(trans[:, 0, :, :, :], kernel['x'], mode='nearest'),
            scipy.ndimage.correlate(trans[:, 1, :, :, :], kernel['x'], mode='nearest'),
            scipy.ndimage.correlate(trans[:, 2, :, :, :], kernel['x'], mode='nearest'),
        ],
        axis=1,
    )

    grady = np.stack(
        [
            scipy.ndimage.correlate(trans[:, 0, :, :, :], kernel['y'], mode='nearest'),
            scipy.ndimage.correlate(trans[:, 1, :, :, :], kernel['y'], mode='nearest'),
            scipy.ndimage.correlate(trans[:, 2, :, :, :], kernel['y'], mode='nearest'),
        ],
        axis=1,
    )

    gradz = np.stack(
        [
            scipy.ndimage.correlate(trans[:, 0, :, :, :], kernel['z'], mode='nearest'),
            scipy.ndimage.correlate(trans[:, 1, :, :, :], kernel['z'], mode='nearest'),
            scipy.ndimage.correlate(trans[:, 2, :, :, :], kernel['z'], mode='nearest'),
        ],
        axis=1,
    )

    jacobian = np.concatenate([gradx, grady, gradz], 0)

    jac_det = (
        jacobian[0, 0, :, :, :]
        * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :])
        - jacobian[1, 0, :, :, :]
        * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :])
        + jacobian[2, 0, :, :, :]
        * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
    )

    jac_det = jac_det[1:-1, 1:-1, 1:-1]  # remove boundary voxels

    return jac_det


def calc_Jstar_1(trans: np.ndarray) -> np.ndarray:
    """Compute the Jacobian determinant ``J*_1`` with the first diagonal (tetrahedral) difference stencil.

    Args:
        trans: Deformation (absolute voxel coordinate) field (3, H, W, D), i.e. identity + displacement.

    Returns:
        Jacobian determinant volume (H-2, W-2, D-2).

    """
    kernel = {}
    kernel['x'] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]).reshape(1, 3, 3, 1)
    kernel['y'] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]).reshape(1, 3, 1, 3)
    kernel['z'] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]).reshape(1, 1, 3, 3)

    trans = trans[None, ...]
    gradx = np.stack(
        [
            scipy.ndimage.correlate(trans[:, 0, :, :, :], kernel['x'], mode='nearest'),
            scipy.ndimage.correlate(trans[:, 1, :, :, :], kernel['x'], mode='nearest'),
            scipy.ndimage.correlate(trans[:, 2, :, :, :], kernel['x'], mode='nearest'),
        ],
        axis=1,
    )

    grady = np.stack(
        [
            scipy.ndimage.correlate(trans[:, 0, :, :, :], kernel['y'], mode='nearest'),
            scipy.ndimage.correlate(trans[:, 1, :, :, :], kernel['y'], mode='nearest'),
            scipy.ndimage.correlate(trans[:, 2, :, :, :], kernel['y'], mode='nearest'),
        ],
        axis=1,
    )

    gradz = np.stack(
        [
            scipy.ndimage.correlate(trans[:, 0, :, :, :], kernel['z'], mode='nearest'),
            scipy.ndimage.correlate(trans[:, 1, :, :, :], kernel['z'], mode='nearest'),
            scipy.ndimage.correlate(trans[:, 2, :, :, :], kernel['z'], mode='nearest'),
        ],
        axis=1,
    )

    jacobian = np.concatenate([gradx, grady, gradz], 0)

    jac_det = (
        jacobian[0, 0, :, :, :]
        * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :])
        - jacobian[1, 0, :, :, :]
        * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :])
        + jacobian[2, 0, :, :, :]
        * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
    )

    jac_det = jac_det[1:-1, 1:-1, 1:-1]  # remove boundary voxels

    return jac_det


def calc_Jstar_2(trans: np.ndarray) -> np.ndarray:
    """Compute the Jacobian determinant ``J*_2`` with the second diagonal (tetrahedral) difference stencil.

    Args:
        trans: Deformation (absolute voxel coordinate) field (3, H, W, D), i.e. identity + displacement.

    Returns:
        Jacobian determinant volume (H-2, W-2, D-2).

    """
    kernel = {}
    kernel['x'] = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]]).reshape(1, 3, 3, 1)
    kernel['y'] = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]]).reshape(1, 1, 3, 3)
    kernel['z'] = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]]).reshape(1, 3, 1, 3)

    trans = trans[None, ...]
    gradx = np.stack(
        [
            scipy.ndimage.correlate(trans[:, 0, :, :, :], kernel['x'], mode='nearest'),
            scipy.ndimage.correlate(trans[:, 1, :, :, :], kernel['x'], mode='nearest'),
            scipy.ndimage.correlate(trans[:, 2, :, :, :], kernel['x'], mode='nearest'),
        ],
        axis=1,
    )

    grady = np.stack(
        [
            scipy.ndimage.correlate(trans[:, 0, :, :, :], kernel['y'], mode='nearest'),
            scipy.ndimage.correlate(trans[:, 1, :, :, :], kernel['y'], mode='nearest'),
            scipy.ndimage.correlate(trans[:, 2, :, :, :], kernel['y'], mode='nearest'),
        ],
        axis=1,
    )

    gradz = np.stack(
        [
            scipy.ndimage.correlate(trans[:, 0, :, :, :], kernel['z'], mode='nearest'),
            scipy.ndimage.correlate(trans[:, 1, :, :, :], kernel['z'], mode='nearest'),
            scipy.ndimage.correlate(trans[:, 2, :, :, :], kernel['z'], mode='nearest'),
        ],
        axis=1,
    )

    jacobian = np.concatenate([gradx, grady, gradz], 0)

    jac_det = (
        jacobian[0, 0, :, :, :]
        * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :])
        - jacobian[1, 0, :, :, :]
        * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :])
        + jacobian[2, 0, :, :, :]
        * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
    )

    jac_det = jac_det[1:-1, 1:-1, 1:-1]  # remove boundary voxels

    return jac_det


def calc_jac_dets(trans: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute multiple Jacobian determinant variants and a consistency mask.

    Args:
        trans: Deformation (absolute voxel coordinate) field (3, H, W, D), i.e. identity + displacement.

    Returns:
        Dict of Jacobian determinant volumes (H-2, W-2, D-2) keyed by stencil string: ``'0x0y0z'`` (central),
        the eight one-sided stencils ``'+x+y+z'`` ... ``'-x-y-z'``, ``'Jstar_1'`` and ``'Jstar_2'``, plus
        ``'all J_i>0'``, a 0/1 mask of voxels where all eight one-sided determinants are positive.

    Raises:
        AssertionError: If a voxel with all one-sided determinants positive has a non-positive central determinant.

    """
    jac_det = {}
    # calculate all finite difference based |J|'s
    for grad_args in ['0x0y0z', '+x+y+z', '+x+y-z', '+x-y+z', '+x-y-z', '-x+y+z', '-x+y-z', '-x-y+z', '-x-y-z']:
        jac_det[grad_args] = calc_J_i(trans, grad_args)

    # calc 'all |J_i| > 0'
    jac_det['all J_i>0'] = np.ones_like(jac_det['0x0y0z'])
    for grad_args in ['+x+y+z', '+x+y-z', '+x-y+z', '+x-y-z', '-x+y+z', '-x+y-z', '-x-y+z', '-x-y-z']:
        jac_det['all J_i>0'] *= jac_det[grad_args] > 0

    # sanity check: if 'all J_i > 0', the central difference must be positive
    assert np.sum((jac_det['all J_i>0'] > 0) * (jac_det['0x0y0z'] <= 0)) == 0

    jac_det['Jstar_1'] = calc_Jstar_1(trans)
    jac_det['Jstar_2'] = calc_Jstar_2(trans)

    return jac_det


def get_identity_grid(array: np.ndarray) -> np.ndarray:
    """Return the identity transformation of the same size as the input.

    Args:
        array: Field of shape (3, H, W, S); only its shape is used.

    Returns:
        float32 grid (3, H, W, S) whose channel ``i`` holds the voxel index along spatial axis ``i``.

    """
    dims = array.shape[1:]
    vectors = [np.arange(0, dim, 1) for dim in dims]

    grids = np.meshgrid(*vectors)
    grids = [np.transpose(x, axes=(1, 0, 2)) for x in grids]
    grid = np.stack(grids).astype('float32')

    return grid


def calc_measurements(jac_dets: Dict[str, np.ndarray], mask: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    """Compute non-diffeomorphic voxel, tetrahedra and volume measures.

    Args:
        jac_dets: Output of :func:`calc_jac_dets`.
        mask: Mask (H-2, W-2, D-2) (boundary removed) weighting the voxels to count.

    Returns:
        Tuple ``(non_diff_voxels, non_diff_tetrahedra, non_diff_volume, non_diff_volume_map)``: the masked count of
        voxels with central-difference ``|J| <= 0``; the weighted count (weight 1/10 each) of non-positive
        determinants over the ten tetrahedral stencils; the total non-diffeomorphic volume
        ``sum(-0.5 * min(|J|, 0) / 6)`` over the same stencils; and the voxel-wise map of that volume.

    """
    # calculate non-diffeomorphic voxels
    non_diff_voxels = np.sum((jac_dets['0x0y0z'] <= 0) * mask)

    # calculate non-diffeomorphic voxels
    non_diff_tetrahedra = 0
    weights = [1 / 5 / 2] * 10  # equally weighted, five tetrahedra, two dividing schemes
    # weights = [1/6/2] * 8 + [1/3/2] * 2 # weighted based on volume
    for w, grad_args in zip(
        weights, ['+x+y+z', '+x+y-z', '+x-y+z', '+x-y-z', '-x+y+z', '-x+y-z', '-x-y+z', '-x-y-z', 'Jstar_1', 'Jstar_2']
    ):
        non_diff_tetrahedra += np.sum((jac_dets[grad_args] <= 0) * mask) * w

    # calculate non-diffeomorphic volume
    non_diff_volume = 0
    non_diff_volume_map = 0
    for grad_args in [
        '+x+y+z',
        '+x+y-z',
        '+x-y+z',
        '+x-y-z',
        'Jstar_1',
        'Jstar_2',
        '-x+y+z',
        '-x+y-z',
        '-x-y+z',
        '-x-y-z',
    ]:
        # each |J| equals six times the signed volume, two schemes considered
        non_diff_volume += np.sum(-0.5 * np.minimum(jac_dets[grad_args], 0) * mask / 6)
        non_diff_volume_map += -0.5 * np.minimum(jac_dets[grad_args], 0) * mask / 6

    return non_diff_voxels, non_diff_tetrahedra, non_diff_volume, non_diff_volume_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trans",
        required=True,
        help="Path of the input transformation, in '.npz' format. Expected dimension: 3xHxWxS.",
    )
    parser.add_argument("--mask", help="Path of the mask or label image, in '.nii.gz' format.")
    parser.add_argument("--scale", action='store_true', help="Flag for upsample the transformation.")
    parser.add_argument("--disp", action='store_true', help="Flag for displacement input.")
    args = parser.parse_args()

    # load transformation from file
    trans = np.load(args.trans)['arr_0'].astype('float32')

    if args.scale:
        # upsample the input to match the brain label image
        trans = np.array([scipy.ndimage.zoom(trans[i], 2, order=2) for i in range(3)])

    if args.disp:
        # convert displacement field to deformation field if necessary
        trans += get_identity_grid(trans)

    # calculate the Jacobian determinants
    jac_dets = calc_jac_dets(trans)

    # load the label image for mask
    if args.mask:
        mask = nib.load(args.mask).get_fdata().astype('float32')
        mask = (mask[1:-1, 1:-1, 1:-1] > 0).astype('float32')  # remove boundary voxels
    else:
        mask = np.ones_like(trans[0, 1:-1, 1:-1, 1:-1])
    total_voxels = np.sum(mask)

    # calculate non-diffeomorphic voxels, non-diffeomorphic tetrahedra,
    # and non-diffeomorphic volume
    non_diff_voxels, non_diff_tetrahedra, non_diff_volume, _ = calc_measurements(jac_dets, mask)

    print(f'Non-diffeomorphic Voxels: {non_diff_voxels:.2f}({non_diff_voxels / total_voxels * 100:.2f}%)')
    print(f'Non-diffeomorphic Tetrahedra: {non_diff_tetrahedra:.2f}({non_diff_tetrahedra / total_voxels * 100:.2f}%)')
    print(f'Non-diffeomorphic Volume: {non_diff_volume:.2f}({non_diff_volume / total_voxels * 100:.2f}%)')
