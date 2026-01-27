"""Core functions for convolution sampling."""

from math import ceil, floor
from typing import List, Optional, Tuple

from torch import Tensor, cat
from torch import device as torch_device
from torch import empty, linspace, tensor, unique, zeros_like
from torch.jit import script

from MIR.models.SITReg.composable_mapping.util import get_spatial_dims


@script
def permute_sequence(sequence: List[int], permutation: List[int]) -> List[int]:
    """Permute a sequence according to a permutation.

    Args:
        sequence: List to permute.
        permutation: Permutation to apply.

    Returns:
        Permuted sequence.
    """
    if len(sequence) != len(permutation):
        raise ValueError("Sequence has wrong length.")
    permuted: List[int] = []
    for permuted_dim in permutation:
        permuted.append(sequence[permuted_dim])
    return permuted


@script
def normalize_sampling_grid(
    grid_spatial_shape: List[int],
    grid_affine_matrix: Tensor,
) -> Optional[Tuple[List[int], Tensor, List[int], List[int]]]:
    """Normalize sampling grid to a diagonal matrix with positive diagonal elements.

    Args:
        grid_spatial_shape: Shape of the spatial dimensions of the grid.
        grid_affine_matrix: Affine matrix applied to the voxel grid to obtain
            the sampling grid.

    Returns:
        Normalized sampling grid (the affine matrix will have the last row
        omitted), and permutation and flips which can be applied to the final
        sampled volume to obtain the same result as sampling the original volume
        with the unnormalized sampling grid. Return None if the normalization is
        not possible.
    """
    if grid_affine_matrix.shape[-2] != grid_affine_matrix.shape[-1]:
        return None
    n_dims = grid_affine_matrix.shape[-1] - 1
    grid_affine_matrix = grid_affine_matrix.view(-1, n_dims + 1, n_dims + 1)[:, :-1, :]
    inverse_permutation = grid_affine_matrix[0, :, :-1].abs().argmax(dim=0)
    if len(unique(inverse_permutation)) != n_dims:
        return None
    inverse_permutation_list: List[int] = inverse_permutation.tolist()
    flipped_spatial_dims: List[int] = []
    for column, largest_row in enumerate(inverse_permutation_list):
        if grid_affine_matrix[0, largest_row, column] < 0:
            flipped_spatial_dims.append(largest_row)
    permutation = inverse_permutation.argsort()
    permutation_list: List[int] = permutation.tolist()

    grid_affine_matrix = grid_affine_matrix[
        :, :, cat((permutation, permutation.new_full((1,), n_dims)), dim=0)
    ]
    grid_spatial_shape = permute_sequence(grid_spatial_shape, permutation_list)
    grid_shape_tensor = tensor(
        grid_spatial_shape, device=grid_affine_matrix.device, dtype=grid_affine_matrix.dtype
    )

    if flipped_spatial_dims:
        linear_matrix = grid_affine_matrix[:, :, :-1]
        diagonal = grid_affine_matrix[0, :, :-1].diagonal()
        flip_vector = diagonal.sign()
        linear_matrix = (linear_matrix * flip_vector[:, None]).contiguous()
        grid_affine_matrix = cat(
            [
                linear_matrix,
                (grid_affine_matrix[:, :, -1] + (grid_shape_tensor - 1) * diagonal.clamp(max=0.0))[
                    :, :, None
                ],
            ],
            dim=2,
        )

    return grid_spatial_shape, grid_affine_matrix, inverse_permutation_list, flipped_spatial_dims


@script
def _infer_kernel_min_and_max_coordinates(
    kernel_support: Tuple[float, float],
    is_zero_on_bounds: Tuple[bool, bool],
    limit_direction: str,
    limit_tol: float,
) -> Tuple[float, float, bool, bool]:
    kernel_min, kernel_max = kernel_support
    left_zero, right_zero = is_zero_on_bounds
    if left_zero:
        kernel_min = kernel_min + limit_tol
        inclusive_min = False
    else:
        if limit_direction == "left":
            kernel_min = kernel_min - limit_tol
            inclusive_min = True
        elif limit_direction == "right":
            kernel_min = kernel_min + limit_tol
            inclusive_min = False
        elif limit_direction == "average":
            kernel_min = kernel_min - limit_tol
            inclusive_min = True
        else:
            raise ValueError(f"Invalid limit direction {limit_direction}")
    if right_zero:
        kernel_max = kernel_max - limit_tol
        inclusive_max = False
    else:
        if limit_direction == "left":
            kernel_max = kernel_max - limit_tol
            inclusive_max = False
        elif limit_direction == "right":
            kernel_max = kernel_max + limit_tol
            inclusive_max = True
        elif limit_direction == "average":
            kernel_max = kernel_max + limit_tol
            inclusive_max = True
        else:
            raise ValueError(f"Invalid limit direction {limit_direction}")
    return kernel_min, kernel_max, inclusive_min, inclusive_max


@script
def calculate_convolutional_sampling_args(
    grid_spatial_shape: List[int],
    grid_affine_matrix: Tensor,
    is_interpolating_kernel: List[bool],
    conv_tol: float,
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
    """Extract parameters which can be used to sample the grid using
    convolutional interpolation.

    Args:
        grid_spatial_shape: Shape of the spatial dimensions of the grid
            (normalized by normalize_sampling_grid).
        grid_affine_matrix: Affine matrix applied to the voxel grid (normalized
            by normalize_sampling_grid).
        is_interpolating_kernel: Whether the kernel is interpolating in each spatial dimension.
        conv_tol: Maximum allowed difference in coordinates
            when using convolution-based sampling (the difference might be upper
            bounded when doing the decision).

    Returns:
        Tuple with the following elements: Downsampling factor, and translation,
        as well as masks for which approach to use (slicing, convolution, or
        transposed convolution) for each spatial dimension.
    """
    device = grid_affine_matrix.device
    dtype = grid_affine_matrix.dtype
    scale = grid_affine_matrix[0, :, :-1].diagonal()
    if scale.any() == 0.0:
        return None
    transposed_convolution_mask = scale < 1 / (1 / 2 + 1 / 1)
    downsampling_factor = empty(scale.shape, device=device, dtype=scale.dtype)
    downsampling_factor[~transposed_convolution_mask] = scale[~transposed_convolution_mask].round()
    downsampling_factor[transposed_convolution_mask] = (
        1 / (1 / scale[transposed_convolution_mask]).round()
    )
    rounded_stride_matrix = downsampling_factor.diag()
    difference_matrix = grid_affine_matrix[:, :, :-1] - rounded_stride_matrix
    shape_tensor = tensor(grid_spatial_shape, device=device, dtype=dtype)
    max_diagonal_coordinate_difference_upper_bound = (
        (difference_matrix * shape_tensor).abs().amax(dim=(0, 2))
    )
    if grid_affine_matrix.size(0) == 1:
        max_translation_coordinate_difference: Tensor = tensor(0.0, device=device, dtype=dtype)
    else:
        max_translation_coordinate_difference = (
            (grid_affine_matrix[1:, :, -1] - grid_affine_matrix[0, :, -1]).abs().amax(dim=0)
        )
    max_conv_coordinate_difference_upper_bound = (
        max_diagonal_coordinate_difference_upper_bound + max_translation_coordinate_difference
    )
    if (max_conv_coordinate_difference_upper_bound > conv_tol).any():
        return None
    interpolating_kernel = tensor(
        is_interpolating_kernel,
        device=device,
    )
    translation = grid_affine_matrix[0, :, -1].clone().contiguous()
    rounded_translation = translation.round()
    small_translation = (
        (translation - rounded_translation).abs() + max_conv_coordinate_difference_upper_bound
    ) < conv_tol
    if (interpolating_kernel & (~transposed_convolution_mask)).any():
        convolution_mask = ((~interpolating_kernel) | (~small_translation)) & (
            ~transposed_convolution_mask
        )
        slicing_mask = (~transposed_convolution_mask) & (~convolution_mask)
        translation[slicing_mask] = translation[slicing_mask].round()
    else:
        convolution_mask = ~transposed_convolution_mask
        slicing_mask = zeros_like(convolution_mask)
    return (
        downsampling_factor,
        translation,
        slicing_mask,
        convolution_mask,
        transposed_convolution_mask,
    )


def apply_flips_and_permutation_to_volume(
    volume: Tensor,
    n_channel_dims: int,
    spatial_permutation: List[int],
    flipped_spatial_dims: List[int],
) -> Tensor:
    """Apply the flipping and permutation to a volume.

    Args:
        volume: Volume to transform with shape (*batch_shape, *channels_shape, *spatial_shape).
        spatial_permutation: Spatial permutation to apply.
        flipped_spatial_dims: Spatial dimensions to flip.
    """
    spatial_dims = get_spatial_dims(volume.ndim, n_channel_dims=n_channel_dims)
    if flipped_spatial_dims:
        flipped_dims = [spatial_dims[spatial_dim] for spatial_dim in flipped_spatial_dims]
        volume = volume.flip(dims=flipped_dims)
    permuted_spatial_dims = permute_sequence(spatial_dims, spatial_permutation)
    volume = volume.permute(tuple(range(spatial_dims[0])) + tuple(permuted_spatial_dims))
    return volume


@script
def _optionally_inclusive_floor(
    value: float,
    inclusive: bool,
) -> int:
    if inclusive:
        return int(floor(value))
    return int(ceil(value - 1))


@script
def calculate_convolutional_sampling_parameters(
    volume_spatial_shape: List[int],
    grid_spatial_shape: List[int],
    grid_affine_matrix: Tensor,
    is_interpolating_kernel: List[bool],
    kernel_support: List[Tuple[float, float]],
    is_zero_on_bounds: List[Tuple[bool, bool]],
    limit_direction: List[str],
    conv_tol: float,
    limit_tol: float,
    target_device: torch_device,
) -> Optional[
    Tuple[
        List[Optional[Tensor]],
        List[int],
        List[int],
        List[bool],
        List[Tuple[int, int]],
        List[Tuple[int, int]],
        List[int],
        List[int],
    ]
]:
    """Obtain parameters for convolutional sampling.

    Args:
        volume_spatial_shape: Shape of the spatial dimensions of the volume.
        grid_spatial_shape: Shape of the spatial dimensions of the grid.
        grid_affine_matrix: Affine matrix applied to the voxel grid.
        is_interpolating_kernel: Whether the kernel is interpolating in each spatial dimension.
        kernel_support: Support of the kernel in each spatial dimension.
        is_zero_on_bounds: Whether the kernel is zero on the bounds in each spatial dimension.
        limit_direction: Limit direction of the kernel in each spatial dimension.
        conv_tol: Maximum allowed difference in coordinates
            when using convolution-based sampling (the difference might be upper
            bounded when doing the decision).
        target_device: Device to use for the kernel coordinates.

    Returns:
        Tuple of kernel coordinates, and other convolution parameters. None if
        the sampling can not be done using convolution.
    """
    normalized_grid_parameters = normalize_sampling_grid(
        grid_spatial_shape=grid_spatial_shape,
        grid_affine_matrix=grid_affine_matrix,
    )
    if normalized_grid_parameters is None:
        return None
    (
        grid_spatial_shape,
        grid_affine_matrix,
        inverse_spatial_permutation,
        flipped_spatial_dims,
    ) = normalized_grid_parameters

    conv_sampling_parameters = calculate_convolutional_sampling_args(
        grid_spatial_shape=grid_spatial_shape,
        grid_affine_matrix=grid_affine_matrix,
        is_interpolating_kernel=is_interpolating_kernel,
        conv_tol=conv_tol,
    )
    if conv_sampling_parameters is None:
        return None
    (
        downsampling_factor_tensor,
        translation_tensor,
        slicing_mask_tensor,
        convolution_mask_tensor,
        transposed_convolution_mask_tensor,
    ) = conv_sampling_parameters
    downsampling_factor: List[float] = downsampling_factor_tensor.tolist()
    translation: List[float] = translation_tensor.tolist()
    slicing_mask: List[bool] = slicing_mask_tensor.tolist()
    convolution_mask: List[bool] = convolution_mask_tensor.tolist()
    transposed_convolution_mask: List[bool] = transposed_convolution_mask_tensor.tolist()

    pre_pads_or_crops: List[Tuple[int, int]] = []
    post_pads_or_crops: List[Tuple[int, int]] = []
    conv_paddings: List[int] = []
    conv_kernel_coordinates: List[Optional[Tensor]] = []
    for (
        use_slicing,
        use_convolution,
        use_transposed_convolution,
        dim_size_volume,
        dim_size_grid,
        dim_translation,
        dim_downsampling_factor,
        dim_kernel_support,
        dim_is_zero_on_bounds,
        dim_limit_direction,
    ) in zip(
        slicing_mask,
        convolution_mask,
        transposed_convolution_mask,
        volume_spatial_shape,
        grid_spatial_shape,
        translation,
        downsampling_factor,
        kernel_support,
        is_zero_on_bounds,
        limit_direction,
    ):
        kernel_min, kernel_max, inclusive_min, inclusive_max = (
            _infer_kernel_min_and_max_coordinates(
                kernel_support=dim_kernel_support,
                is_zero_on_bounds=dim_is_zero_on_bounds,
                limit_direction=dim_limit_direction,
                limit_tol=limit_tol,
            )
        )
        min_coordinate = dim_translation
        max_coordinate = dim_translation + dim_downsampling_factor * (dim_size_grid - 1)
        if use_convolution or use_transposed_convolution:
            pre_pad_or_crop_lower = _optionally_inclusive_floor(
                -kernel_min - dim_translation, inclusive=inclusive_min
            )
            pre_pad_or_crop_upper = _optionally_inclusive_floor(
                kernel_max
                + dim_translation
                + dim_downsampling_factor * (dim_size_grid - 1)
                - (dim_size_volume - 1),
                inclusive=inclusive_max,
            )
            if use_transposed_convolution:
                start_kernel_coordinate = (
                    1
                    - dim_translation
                    + _optionally_inclusive_floor(
                        (kernel_max - (1 - dim_translation)) / dim_downsampling_factor,
                        inclusive=inclusive_max,
                    )
                    * dim_downsampling_factor
                )
                end_kernel_coordinate = (
                    -dim_translation
                    - _optionally_inclusive_floor(
                        (-kernel_min - dim_translation) / dim_downsampling_factor,
                        inclusive=inclusive_min,
                    )
                    * dim_downsampling_factor
                )
                kernel_step_size = dim_downsampling_factor
            elif use_convolution:
                relative_coordinate = dim_translation - floor(dim_translation)
                start_kernel_coordinate = (
                    -_optionally_inclusive_floor(
                        -kernel_min - relative_coordinate, inclusive=inclusive_min
                    )
                    - relative_coordinate
                )
                end_kernel_coordinate = _optionally_inclusive_floor(
                    kernel_max - (1 - relative_coordinate), inclusive=inclusive_max
                ) + (1 - relative_coordinate)
                kernel_step_size = 1.0
            else:
                raise ValueError("Invalid convolution mask")
            kernel_coordinates = linspace(
                start_kernel_coordinate,
                end_kernel_coordinate,
                int(round(abs(end_kernel_coordinate - start_kernel_coordinate) / kernel_step_size))
                + 1,
                dtype=grid_affine_matrix.dtype,
                device=target_device,
            )
        elif use_slicing:
            kernel_coordinates = None
            start_kernel_coordinate = 0.0  # dummy value for torchscript compatibility
            end_kernel_coordinate = 0.0  # dummy value for torchscript compatibility
            pre_pad_or_crop_lower = -int(min_coordinate)
            pre_pad_or_crop_upper = int(max_coordinate) - (dim_size_volume - 1)
        else:
            raise ValueError("Invalid convolution mask")
        if use_transposed_convolution:
            post_pad_or_crop_lower = -int(
                round(
                    (min_coordinate + pre_pad_or_crop_lower + start_kernel_coordinate)
                    / dim_downsampling_factor
                )
            )
            post_pad_or_crop_upper = -int(
                round(
                    (
                        (dim_size_volume - 1)
                        + pre_pad_or_crop_upper
                        - end_kernel_coordinate
                        - max_coordinate
                    )
                    / dim_downsampling_factor
                )
            )
            assert post_pad_or_crop_lower <= 0 and post_pad_or_crop_upper <= 0
            conv_padding = -max(post_pad_or_crop_lower, post_pad_or_crop_upper)
            post_pad_or_crop_lower += conv_padding
            post_pad_or_crop_upper += conv_padding
        else:
            conv_padding = 0
            post_pad_or_crop_lower = 0
            post_pad_or_crop_upper = 0
        conv_kernel_coordinates.append(kernel_coordinates)
        conv_paddings.append(conv_padding)
        pre_pads_or_crops.append((pre_pad_or_crop_lower, pre_pad_or_crop_upper))
        post_pads_or_crops.append((post_pad_or_crop_lower, post_pad_or_crop_upper))
    conv_strides = (
        (
            downsampling_factor_tensor * (~transposed_convolution_mask_tensor)
            + (1 / downsampling_factor_tensor) * transposed_convolution_mask_tensor
        )
        .round()
        .long()
    )
    conv_strides_list: List[int] = conv_strides.tolist()
    return (
        conv_kernel_coordinates,
        conv_strides_list,
        conv_paddings,
        transposed_convolution_mask,
        pre_pads_or_crops,
        post_pads_or_crops,
        inverse_spatial_permutation,
        flipped_spatial_dims,
    )
