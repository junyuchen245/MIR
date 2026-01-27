"""Dense deformations utility functions."""

from importlib import import_module
from typing import List, Optional, Tuple

from torch import Tensor, tensor
from torch.jit import script
from torch.nn.functional import grid_sample

from ._second_order_differentiable_grid_sample.grid_sample import (
    grid_sample as second_order_differentiable_grid_sample,
)


def interpolate(
    volume: Tensor,
    grid: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "border",
    second_order_differentiable: bool = False,
) -> Tensor:
    """Interpolate in voxel coordinates.

    Args:
        volume: Tensor with shape
            (*batch_shape , *channels_shape, *spatial_shape).
        grid: Tensor with shape (*batch_shape, n_spatial_dims, *target_shape).
        mode: Interpolation mode. Default is "bilinear".
        padding_mode: Padding mode. Default is "border".
        second_order_differentiable: Whether to use second order differentiable interpolation.

    Returns:
        Tensor with shape (*broadcasted_batch_shape, *channels_shape, *target_shape).
    """
    if second_order_differentiable:
        return _interpolate_second_order_differentiable(volume, grid, mode, padding_mode)
    return _interpolate(volume, grid, mode, padding_mode)


@script
def _convert_voxel_to_normalized_coordinates(
    coordinates: Tensor, volume_shape: Optional[List[int]]
) -> Tensor:
    channel_dim = 0 if coordinates.ndim == 1 else 1
    n_spatial_dims = 0 if coordinates.ndim <= 2 else coordinates.ndim - 2
    n_dims = coordinates.size(channel_dim)
    inferred_volume_shape = coordinates.shape[-n_dims:] if volume_shape is None else volume_shape
    add_spatial_dims_view = (-1,) + n_spatial_dims * (1,)
    volume_shape_tensor = (
        tensor(
            inferred_volume_shape,
            dtype=coordinates.dtype,
        )
        .view(add_spatial_dims_view)
        .to(device=coordinates.device, non_blocking=coordinates.device.type != "cpu")
    )
    return coordinates / (volume_shape_tensor - 1) * 2 - 1


def _broadcast_batch_size(tensor_1: Tensor, tensor_2: Tensor) -> Tuple[Tensor, Tensor]:
    batch_size = max(tensor_1.size(0), tensor_2.size(0))
    if tensor_1.size(0) == 1 and batch_size != 1:
        tensor_1 = tensor_1[0].expand((batch_size,) + tensor_1.shape[1:])
    elif tensor_2.size(0) == 1 and batch_size != 1:
        tensor_2 = tensor_2[0].expand((batch_size,) + tensor_2.shape[1:])
    elif tensor_1.size(0) != tensor_2.size(0) and batch_size != 1:
        raise ValueError("Can not broadcast batch size")
    return tensor_1, tensor_2


def _match_grid_shape_to_dims(grid: Tensor) -> Tensor:
    batch_size = grid.size(0)
    n_dims = grid.size(1)
    grid_shape = grid.shape[2:]
    dim_matched_grid_shape = (
        (1,) * max(0, n_dims - grid.ndim + 1) + grid_shape[: n_dims - 1] + (-1,)
    )
    return grid.view(
        (
            batch_size,
            n_dims,
        )
        + dim_matched_grid_shape
    )


@script
def _handle_interpolation_inputs(
    volume: Tensor,
    grid: Tensor,
) -> Tuple[Tensor, Tensor, List[int], List[int]]:
    if grid.ndim == 1:
        grid = grid[None]
    n_dims = grid.size(1)
    if volume.ndim < n_dims + 2:
        raise ValueError(
            "Volume must have batch dimension, at least one channel dimension, "
            "and spatial dimensions"
        )
    channel_shape = volume.shape[1:-n_dims]
    volume_spatial_shape = volume.shape[-n_dims:]
    target_shape = grid.shape[2:]
    dim_matched_grid = _match_grid_shape_to_dims(grid)
    normalized_grid: Tensor = _convert_voxel_to_normalized_coordinates(
        dim_matched_grid, list(volume_spatial_shape)
    )
    simplified_volume = volume.view((volume.size(0), -1) + volume_spatial_shape)
    permuted_volume = simplified_volume.permute(
        [0, 1] + list(range(simplified_volume.ndim - 1, 2 - 1, -1))
    )
    permuted_grid = normalized_grid.moveaxis(1, -1)
    permuted_volume, permuted_grid = _broadcast_batch_size(permuted_volume, permuted_grid)
    return permuted_volume, permuted_grid, list(channel_shape), list(target_shape)


@script
def _interpolate(
    volume: Tensor,
    grid: Tensor,
    mode: str,
    padding_mode: str,
) -> Tensor:
    permuted_volume, permuted_grid, channel_shape, target_shape = _handle_interpolation_inputs(
        volume, grid
    )
    return grid_sample(
        input=permuted_volume,
        grid=permuted_grid,
        align_corners=True,
        mode=mode,
        padding_mode=padding_mode,
    ).view([-1] + channel_shape + target_shape)


def _interpolate_second_order_differentiable(
    volume: Tensor, grid: Tensor, mode: str, padding_mode: str
) -> Tensor:
    permuted_volume, permuted_grid, channel_shape, target_shape = _handle_interpolation_inputs(
        volume, grid
    )
    return second_order_differentiable_grid_sample(
        input=permuted_volume,
        grid=permuted_grid,
        align_corners=True,
        mode=mode,
        padding_mode=padding_mode,
    ).view([-1] + channel_shape + target_shape)
