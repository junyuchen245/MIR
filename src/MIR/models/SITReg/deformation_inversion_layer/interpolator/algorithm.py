"""Core interpolation algorithms"""

from typing import List, Optional, Tuple

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import linspace, meshgrid, stack, tensor
from torch.jit import script
from torch.nn.functional import grid_sample


@script
def _move_channels_last(tensor_to_modify: Tensor, num_channel_dims: int = 1) -> Tensor:
    if tensor_to_modify.ndim == num_channel_dims:
        return tensor_to_modify
    return tensor_to_modify.permute(
        [0]
        + list(range(num_channel_dims + 1, tensor_to_modify.ndim))
        + list(range(1, num_channel_dims + 1))
    )


@script
def _index_by_channel_dims(n_total_dims: int, channel_dim_index: int, n_channel_dims: int) -> int:
    if n_total_dims < n_channel_dims:
        raise RuntimeError("Number of channel dimensions do not match")
    if n_total_dims == n_channel_dims:
        return channel_dim_index
    return channel_dim_index + 1


@script
def _num_spatial_dims(n_total_dims: int, n_channel_dims: int) -> int:
    if n_total_dims < n_channel_dims:
        raise RuntimeError("Number of channel dimensions do not match")
    if n_total_dims <= n_channel_dims + 1:
        return 0
    return n_total_dims - n_channel_dims - 1


@script
def _convert_voxel_to_normalized_coordinates(
    coordinates: Tensor, volume_shape: Optional[List[int]] = None
) -> Tensor:
    channel_dim = _index_by_channel_dims(coordinates.ndim, channel_dim_index=0, n_channel_dims=1)
    n_spatial_dims = _num_spatial_dims(n_total_dims=coordinates.ndim, n_channel_dims=1)
    n_dims = coordinates.size(channel_dim)
    inferred_volume_shape = coordinates.shape[-n_dims:] if volume_shape is None else volume_shape
    add_spatial_dims_view = (-1,) + n_spatial_dims * (1,)
    volume_shape_tensor = tensor(
        inferred_volume_shape, dtype=coordinates.dtype, device=coordinates.device
    ).view(add_spatial_dims_view)
    coordinate_grid_start = tensor(-1.0, dtype=coordinates.dtype, device=coordinates.device).view(
        add_spatial_dims_view
    )
    coordinate_grid_end = tensor(1.0, dtype=coordinates.dtype, device=coordinates.device).view(
        add_spatial_dims_view
    )
    output = (
        coordinates / (volume_shape_tensor - 1) * (coordinate_grid_end - coordinate_grid_start)
        + coordinate_grid_start
    )
    return output


@script
def _broadcast_batch_size(tensor_1: Tensor, tensor_2: Tensor) -> Tuple[Tensor, Tensor]:
    batch_size = max(tensor_1.size(0), tensor_2.size(0))
    if tensor_1.size(0) == 1 and batch_size != 1:
        tensor_1 = tensor_1[0].expand((batch_size,) + tensor_1.shape[1:])
    elif tensor_2.size(0) == 1 and batch_size != 1:
        tensor_2 = tensor_2[0].expand((batch_size,) + tensor_2.shape[1:])
    elif tensor_1.size(0) != tensor_2.size(0) and batch_size != 1:
        raise ValueError("Can not broadcast batch size")
    return tensor_1, tensor_2


@script
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
def interpolate(
    volume: Tensor, grid: Tensor, mode: str = "bilinear", padding_mode: str = "border"
) -> Tensor:
    """Interpolate in voxel coordinates

    Args:
        volume: Interpolated volume with shape
            (batch_size, [channel_1, ..., channel_n, ]dim_1, ..., dim_{n_dims})
        grid: Grid defining interpolation locations with shape (batch_size, n_dims, *target_shape)
        mode: Interpolation mode
        padding_mode: Padding mode defining extrapolation behaviour

    Returns:
        Volume interpolated at grid locations with shape
            (batch_size, channel_1, ..., channel_n, *target_shape)
    """
    if grid.ndim == 1:
        grid = grid[None]
    n_dims = grid.size(1)
    channel_shape = volume.shape[1:-n_dims]
    volume_shape = volume.shape[-n_dims:]
    target_shape = grid.shape[2:]
    dim_matched_grid = _match_grid_shape_to_dims(grid)
    normalized_grid = _convert_voxel_to_normalized_coordinates(dim_matched_grid, list(volume_shape))
    simplified_volume = volume.view((volume.size(0), -1) + volume_shape)
    permuted_volume = simplified_volume.permute(
        [0, 1] + list(range(simplified_volume.ndim - 1, 2 - 1, -1))
    )
    permuted_grid = _move_channels_last(normalized_grid, 1)
    permuted_volume, permuted_grid = _broadcast_batch_size(permuted_volume, permuted_grid)
    return grid_sample(
        input=permuted_volume,
        grid=permuted_grid,
        align_corners=True,
        mode=mode,
        padding_mode=padding_mode,
    ).view((-1,) + channel_shape + target_shape)


@script
def generate_voxel_coordinate_grid(
    shape: List[int], device: torch_device, dtype: Optional[torch_dtype] = None
) -> Tensor:
    """Generate voxel coordinate grid

    Args:
        shape: Shape of the grid
        device: Device of the grid
        dtype: Data type of the grid

    Returns:
        Voxel coordinate grid with shape (1, len(shape), dim_1, ..., dim_{len(shape)})
    """
    axes = [
        linspace(
            start=0,
            end=int(dim_size) - 1,
            steps=int(dim_size),
            device=device,
            dtype=dtype,
        )
        for dim_size in shape
    ]
    coordinates = stack(meshgrid(axes, indexing="ij"), dim=0)
    return coordinates[None]
