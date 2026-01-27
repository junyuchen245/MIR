"""Utility functions for common tasks."""

from itertools import repeat
from typing import Iterable, Optional, Sequence, Tuple, TypeVar, Union

from torch import Tensor, broadcast_shapes
from torch.nn.functional import pad

T = TypeVar("T")


def get_batch_dims(n_total_dims: int, n_channel_dims: int = 1) -> Tuple[int, ...]:
    """Obtain indices for batch dimensions.

    See `composable_mapping` for special handling of shapes in the codebase.

    Args:
        n_total_dims: Number of total dimensions.
        n_channel_dims: Number of channel dimensions.

    Returns:
        Indices for batch dimensions.
    """
    first_channel_dim = get_channel_dims(n_total_dims, n_channel_dims)[0]
    return tuple(range(first_channel_dim))


def get_spatial_dims(
    n_total_dims: int,
    n_channel_dims: int = 1,
) -> Tuple[int, ...]:
    """Obtain indices for spatial dimensions.

    See `composable_mapping` for special handling of shapes in the codebase.

    Args:
        n_total_dims: Number of total dimensions.
        n_channel_dims: Number of channel dimensions.

    Returns:
        Indices for spatial dimensions.
    """
    last_channel_dim = get_channel_dims(n_total_dims, n_channel_dims)[-1]
    return tuple(range(last_channel_dim + 1, n_total_dims))


def get_channel_dims(n_total_dims: int, n_channel_dims: int = 1) -> Tuple[int, ...]:
    """Obtain indices for channel dimensions.

    See `composable_mapping` for special handling of shapes in the codebase.

    Args:
        n_total_dims: Number of total dimensions.
        n_channel_dims: Number of channel dimensions.

    Returns:
        Indices for channel dimensions.
    """
    if n_total_dims < 1:
        raise RuntimeError("Invalid number of total dimensions")
    if n_total_dims < n_channel_dims:
        raise RuntimeError(
            "Number of channel dimensions can not be larger than total number of dimensions"
        )
    if n_channel_dims < 1:
        raise RuntimeError("Invalid number of channel dimensions")
    if n_total_dims == n_channel_dims:
        return tuple(range(n_total_dims))
    return tuple(range(1, n_channel_dims + 1))


def get_batch_shape(shape: Sequence[int], n_channel_dims: int = 1) -> Tuple[int, ...]:
    """Obtain shape of the batch dimensions.

    See `composable_mapping` for special handling of shapes in the codebase.

    Args:
        shape: Shape of the tensor.
        n_channel_dims: Number of channel dimensions.

    Returns:
        Shape of the batch dimensions.
    """
    first_channel_dim = get_channel_dims(len(shape), n_channel_dims)[0]
    return tuple(shape[:first_channel_dim])


def get_spatial_shape(
    shape: Sequence[int],
    n_channel_dims: int = 1,
) -> Tuple[int, ...]:
    """Obtain shape of the spatial dimensions.

    See `composable_mapping` for special handling of shapes in the codebase.

    Args:
        shape: Shape of the tensor.
        n_channel_dims: Number of channel dimensions.

    Returns:
        Shape of the spatial dimensions.
    """
    last_channel_dim = get_channel_dims(len(shape), n_channel_dims)[-1]
    return tuple(shape[last_channel_dim + 1 :])


def get_channels_shape(shape: Sequence[int], n_channel_dims: int = 1) -> Tuple[int, ...]:
    """Obtain shape of the channel dimensions.

    See `composable_mapping` for special handling of shapes in the codebase.

    Args:
        shape: Shape of the tensor.
        n_channel_dims: Number of channel dimensions.

    Returns:
        Shape of the channel dimensions.
    """
    channel_dims = get_channel_dims(len(shape), n_channel_dims)
    n_channel_dims = len(channel_dims)
    first_channel_dim = channel_dims[0]
    return tuple(shape[first_channel_dim : first_channel_dim + n_channel_dims])


def has_spatial_dims(
    shape: Sequence[int],
    n_channel_dims: int = 1,
) -> bool:
    """Has the shape spatial dimensions.

    See `composable_mapping` for special handling of shapes in the codebase.

    Args:
        shape: Shape of the tensor.
        n_channel_dims: Number of channel dimensions.

    Returns:
        Whether the shape has spatial dimensions.
    """
    return bool(get_spatial_shape(shape, n_channel_dims))


def reduce_channels_shape_to_ones(
    shape: Sequence[int],
    n_channel_dims: int = 1,
) -> Tuple[int, ...]:
    """Modify channel shape to ones.

    See `composable_mapping` for special handling of shapes in the codebase.

    E.g. (3, 5, 4, 4) with n_channel_dims = 2 returns
    (3, 1, 1, 4).

    Args:
        shape: Shape of the tensor.
        n_channel_dims: Number of channel dimensions.

    Returns:
        Shape with channel dimensions reduced to ones.
    """
    batch_shape, channel_shape, spatial_shape = split_shape(shape, n_channel_dims)
    return batch_shape + tuple(1 for _ in channel_shape) + spatial_shape


def get_n_channel_dims(n_total_dims: int, n_spatial_dims: int) -> int:
    """Obtain number of channel dimensions based on number of spatial
    dimensions, if possible.

    See `composable_mapping` for special handling of shapes in the codebase.

    Args:
        n_total_dims: Number of total dimensions.
        n_spatial_dims: Number of spatial dimensions.

    Returns:
        Number of channel dimensions.
    """
    if n_spatial_dims < 0:
        raise RuntimeError("Invalid number of spatial dimensions")
    if n_spatial_dims == 0:
        if n_total_dims == 1:
            return 1
        raise RuntimeError("Can not infer number of channel dimensions")
    n_channel_dims = n_total_dims - n_spatial_dims - 1
    if n_channel_dims < 1:
        raise RuntimeError("Invalid n_total_dims for given n_spatial_dims")
    return n_channel_dims


def move_channels_first(
    tensor: Tensor,
    n_channel_dims: int = 1,
) -> Tensor:
    """Move channel dimensions back from being the last dimensions.

    See `composable_mapping` for special handling of shapes in the codebase.

    Args:
        tensor: Tensor with shape
            (*batch_shape, *spatial_shape, *channels_shape).
        n_channel_dims: Number of channel dimensions.

    Returns:
        Tensor with shape
        (*batch_shape, *channels_shape, *spatial_shape)
    """
    channel_dims = get_channel_dims(tensor.ndim, n_channel_dims)
    return tensor.moveaxis(
        tuple(range(-len(channel_dims), 0)),
        channel_dims,
    )


def move_channels_last(tensor: Tensor, n_channel_dims: int = 1) -> Tensor:
    """Move channel dimensions last

    See `composable_mapping` for special handling of shapes in the codebase.

    Args:
        tensor: Tensor with shape
            (*batch_shape, *channels_shape, *spatial_shape)
        n_channel_dims: Number of channel dimensions.

    Returns:
        Tensor with shape
        (*batch_shape, *spatial_shape, *channels_shape)
    """
    channel_dims = get_channel_dims(tensor.ndim, n_channel_dims)
    return tensor.moveaxis(
        channel_dims,
        list(range(-len(channel_dims), 0)),
    )


def _n_dims_to_iterable(n_channel_dims: Union[int, Iterable[int]]) -> Iterable[int]:
    if isinstance(n_channel_dims, int):
        return repeat(n_channel_dims)
    return n_channel_dims


def broadcast_shapes_in_parts_splitted(
    *shapes: Sequence[int],
    n_channel_dims: Union[int, Iterable[int]] = 1,
    broadcast_batch: bool = True,
    broadcast_channels: bool = True,
    broadcast_spatial: bool = True,
) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
    """Broadcast batch dimension, channel dimensions and spatial dimensions separately.

    See `composable_mapping` for special handling of shapes in the codebase.

    Args:
        shapes: Shapes to broadcast.
        n_channel_dims: Number of channel dims for each shape, if integer is
            given, same number will be used for all shapes.
        broadcast_batch: Whether to broadcast batch dimension.
        broadcast_channels: Whether to broadcast channel dimensions.
        broadcast_spatial: Whether to broadcast spatial dimensions.

    Returns:
        Tuple of broadcasted batch shape, broadcasted channel shape and
        broadcasted spatial shape. If the part of the shape is not broadcasted,
        None is returned.
    """
    channel_dims_iterable = _n_dims_to_iterable(n_channel_dims)
    splitted_shapes = [
        split_shape(shape, individual_n_channel_dims)
        for shape, individual_n_channel_dims in zip(shapes, channel_dims_iterable)
    ]
    if broadcast_batch:
        broadcasted_batch_shape: Optional[Tuple[int, ...]] = broadcast_shapes(
            *(shape[0] for shape in splitted_shapes)
        )
    else:
        broadcasted_batch_shape = None
    if broadcast_channels:
        broadcasted_channel_shape: Optional[Tuple[int, ...]] = broadcast_shapes(
            *(shape[1] for shape in splitted_shapes)
        )
    else:
        broadcasted_channel_shape = None
    if broadcast_spatial:
        broadcasted_spatial_shape: Optional[Tuple[int, ...]] = broadcast_shapes(
            *(shape[2] for shape in splitted_shapes)
        )
    else:
        broadcasted_spatial_shape = None
    return broadcasted_batch_shape, broadcasted_channel_shape, broadcasted_spatial_shape


def _select_optional_shapes(
    *shapes: Optional[Sequence[int]],
    n_channel_dims: Union[int, Iterable[int]] = 1,
) -> Tuple[Sequence[Sequence[int]], Sequence[int]]:
    channel_dims_iterable = _n_dims_to_iterable(n_channel_dims)
    not_optional_shapes_and_n_channel_dims = [
        (shape, individual_n_channel_dims)
        for shape, individual_n_channel_dims in zip(shapes, channel_dims_iterable)
        if shape is not None
    ]
    not_optional_shapes = [shape for shape, _ in not_optional_shapes_and_n_channel_dims]
    n_channel_dims = [
        individual_n_channel_dims
        for _, individual_n_channel_dims in not_optional_shapes_and_n_channel_dims
    ]
    return not_optional_shapes, n_channel_dims


def broadcast_optional_shapes_in_parts_splitted(
    *shapes: Optional[Sequence[int]],
    n_channel_dims: Union[int, Iterable[int]] = 1,
    broadcast_batch: bool = True,
    broadcast_channels: bool = True,
    broadcast_spatial: bool = True,
) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
    """Broadcast batch dimension, channel dimensions and spatial dimensions
    separately for optional shapes.

    See `composable_mapping` for special handling of shapes in the codebase.

    Args:
        shapes: Optional shapes to broadcast. If None, the value is ignored.
        n_channel_dims: Number of channel dims for each shape, if integer is
            given, same number will be used for all shapes.
        broadcast_batch: Whether to broadcast batch dimension.
        broadcast_channels: Whether to broadcast channel dimensions.
        broadcast_spatial: Whether to broadcast spatial dimensions.

    Returns:
        Tuple of broadcasted batch shape, broadcasted channel shape and
        broadcasted spatial shape. If the part of the shape is not broadcasted,
        None is returned.
    """
    not_optional_shapes, n_channel_dims = _select_optional_shapes(
        *shapes, n_channel_dims=n_channel_dims
    )
    if not not_optional_shapes:
        return None, None, None
    return broadcast_shapes_in_parts_splitted(
        *not_optional_shapes,
        n_channel_dims=n_channel_dims,
        broadcast_batch=broadcast_batch,
        broadcast_channels=broadcast_channels,
        broadcast_spatial=broadcast_spatial,
    )


def broadcast_shapes_in_parts(
    *shapes: Sequence[int],
    n_channel_dims: Union[int, Iterable[int]] = 1,
    broadcast_batch: bool = True,
    broadcast_channels: bool = True,
    broadcast_spatial: bool = True,
) -> Sequence[Tuple[int, ...]]:
    """Broadcast batch dimension, channel dimensions and spatial dimensions
    separately while returning the broadcasted full shapes.

    See `composable_mapping` for special handling of shapes in the codebase.

    Args:
        shapes: Shapes to broadcast.
        n_channel_dims: Number of channel dims for each shape, if integer is
            given, same number will be used for all shapes.
        broadcast_batch: Whether to broadcast batch dimension.
        broadcast_channels: Whether to broadcast channel dimensions.
        broadcast_spatial: Whether to broadcast spatial dimensions.

    Returns:
        Broadcasted shapes.
    """
    batch_shape, channel_shape, spatial_shape = broadcast_shapes_in_parts_splitted(
        *shapes,
        n_channel_dims=n_channel_dims,
        broadcast_batch=broadcast_batch,
        broadcast_channels=broadcast_channels,
        broadcast_spatial=broadcast_spatial,
    )
    output_shapes = []
    for shape, individual_n_channel_dims in zip(shapes, _n_dims_to_iterable(n_channel_dims)):
        target_batch_shape, target_channel_shape, target_spatial_shape = split_shape(
            shape, individual_n_channel_dims
        )
        if batch_shape is not None:
            target_batch_shape = batch_shape
        if channel_shape is not None:
            target_channel_shape = channel_shape
        if spatial_shape is not None:
            target_spatial_shape = spatial_shape
        output_shapes.append(target_batch_shape + target_channel_shape + target_spatial_shape)
    return output_shapes


def broadcast_shapes_in_parts_to_single_shape(
    *shapes: Sequence[int],
    n_channel_dims: Union[int, Iterable[int]] = 1,
    broadcast_batch: bool = True,
    broadcast_channels: bool = True,
    broadcast_spatial: bool = True,
) -> Tuple[int, ...]:
    """Broadcasts batch dimension, channel dimensions and spatial dimensions
    separately while returning a single broadcasted shape.

    See `composable_mapping` for special handling of shapes in the codebase.

    Args:
        shapes: Shapes to broadcast.
        n_channel_dims: Number of channel dims for each shape, if integer is
            given, same number will be used for all shapes.
        broadcast_batch: Whether to broadcast batch dimension.
        broadcast_channels: Whether to broadcast channel dimensions.
        broadcast_spatial: Whether to broadcast spatial dimensions.

    Returns:
        Broadcasted shape.

    Raises:
        RuntimeError: If the shapes do not broadcast to the same shape.
    """
    broadcasted_shapes = broadcast_shapes_in_parts(
        *shapes,
        n_channel_dims=n_channel_dims,
        broadcast_batch=broadcast_batch,
        broadcast_channels=broadcast_channels,
        broadcast_spatial=broadcast_spatial,
    )
    if len(set(broadcasted_shapes)) != 1:
        raise RuntimeError("Shapes do not broadcast to the same shape")
    return broadcasted_shapes[0]


def broadcast_optional_shapes_in_parts_to_single_shape(
    *shapes: Optional[Sequence[int]],
    n_channel_dims: Union[int, Iterable[int]] = 1,
    broadcast_batch: bool = True,
    broadcast_channels: bool = True,
    broadcast_spatial: bool = True,
) -> Tuple[int, ...]:
    """Broadcasts batch dimension, channel dimensions and spatial dimensions of
    optional shapes separately while returning a single broadcasted shape.

    See `composable_mapping` for special handling of shapes in the codebase.

    Args:
        shapes: Optional shapes to broadcast.
        n_channel_dims: Number of channel dims for each shape, if integer is
            given, same number will be used for all shapes.
        broadcast_batch: Whether to broadcast batch dimension.
        broadcast_channels: Whether to broadcast channel dimensions.
        broadcast_spatial: Whether to broadcast spatial dimensions.

    Returns:
        Broadcasted shape.

    Raises:
        RuntimeError: If the shapes do not broadcast to the same shape or if
            no shapes are provided.
    """
    not_optional_shapes, n_channel_dims = _select_optional_shapes(
        *shapes, n_channel_dims=n_channel_dims
    )
    broadcasted_shapes = broadcast_shapes_in_parts(
        *not_optional_shapes,
        n_channel_dims=n_channel_dims,
        broadcast_batch=broadcast_batch,
        broadcast_channels=broadcast_channels,
        broadcast_spatial=broadcast_spatial,
    )
    if len(set(broadcasted_shapes)) != 1:
        raise RuntimeError("Shapes do not broadcast to the same shape")
    return broadcasted_shapes[0]


def broadcast_to_in_parts(
    tensor: Tensor,
    batch_shape: Optional[Sequence[int]] = None,
    channels_shape: Optional[Sequence[int]] = None,
    spatial_shape: Optional[Sequence[int]] = None,
    n_channel_dims: int = 1,
) -> Tensor:
    """Broadcasts tensor to given batch, channel and spatial shapes separately.

    See `composable_mapping` for special handling of shapes in the codebase.

    Args:
        tensor: Tensor to broadcast.
        batch_shape: Shape of the batch dimensions to broadcast to.
        channels_shape: Shape of the channel dimensions to broadcast to.
        spatial_shape: Shape of the spatial dimensions to broadcast to.

    Returns:
        Broadcasted tensor.
    """
    initial_batch_shape, initial_channels_shape, initial_spatial_shape = split_shape(
        tensor.shape, n_channel_dims=n_channel_dims
    )
    if batch_shape is not None:
        if len(batch_shape) < len(initial_batch_shape):
            raise RuntimeError("Cannot broadcast to smaller batch shape")
        initial_batch_shape = (1,) * (
            len(batch_shape) - len(initial_batch_shape)
        ) + initial_batch_shape
        tensor = tensor.reshape(
            initial_batch_shape + initial_channels_shape + initial_spatial_shape
        )
    else:
        batch_shape = initial_batch_shape
    if channels_shape is not None:
        if len(channels_shape) < len(initial_channels_shape):
            raise RuntimeError("Cannot broadcast to smaller channel shape")
        initial_channels_shape = (1,) * (
            len(channels_shape) - len(initial_channels_shape)
        ) + initial_channels_shape
        tensor = tensor.reshape(
            initial_batch_shape + initial_channels_shape + initial_spatial_shape
        )
    else:
        channels_shape = initial_channels_shape
    if spatial_shape is not None:
        if len(spatial_shape) < len(initial_spatial_shape):
            raise RuntimeError("Cannot broadcast to smaller spatial shape")
        initial_spatial_shape = (1,) * (
            len(spatial_shape) - len(initial_spatial_shape)
        ) + initial_spatial_shape
        tensor = tensor.reshape(
            initial_batch_shape + initial_channels_shape + initial_spatial_shape
        )
    else:
        spatial_shape = initial_spatial_shape
    return tensor.broadcast_to(tuple(batch_shape) + tuple(channels_shape) + tuple(spatial_shape))


def broadcast_tensors_in_parts(
    *tensors: Tensor,
    broadcast_batch: bool = True,
    broadcast_channels: bool = True,
    broadcast_spatial: bool = True,
    n_channel_dims: Union[int, Iterable[int]] = 1,
) -> Tuple[Tensor, ...]:
    """Broadcast multiple tensors separately for batch, channel and spatial
    dimensions.

    See `composable_mapping` for special handling of shapes in the codebase.

    Args:
        tensors: Tensors to broadcast.
        broadcast_batch: Whether to broadcast batch dimension.
        broadcast_channels: Whether to broadcast channel dimensions.
        broadcast_spatial: Whether to broadcast spatial dimensions.
        n_channel_dims: Number of channel dims for each tensor, if integer is
            given, same number will be used for all tensors.

    Returns:
        Broadcasted tensors.
    """
    shapes = [tensor.shape for tensor in tensors]
    broadcasted_batch_shape, broadcasted_channel_shape, broadcasted_spatial_shape = (
        broadcast_shapes_in_parts_splitted(
            *shapes,
            n_channel_dims=n_channel_dims,
            broadcast_batch=broadcast_batch,
            broadcast_channels=broadcast_channels,
            broadcast_spatial=broadcast_spatial,
        )
    )
    return tuple(
        broadcast_to_in_parts(
            tensor,
            batch_shape=broadcasted_batch_shape,
            channels_shape=broadcasted_channel_shape,
            spatial_shape=broadcasted_spatial_shape,
            n_channel_dims=individual_n_channel_dims,
        )
        for tensor, individual_n_channel_dims in zip(
            tensors,
            _n_dims_to_iterable(n_channel_dims),
        )
    )


def broadcast_optional_tensors_in_parts(
    *tensors: Optional[Tensor],
    broadcast_batch: bool = True,
    broadcast_channels: bool = True,
    broadcast_spatial: bool = True,
    n_channel_dims: Union[int, Iterable[int]] = 1,
) -> Tuple[Optional[Tensor], ...]:
    """Broadcast multiple optional tensors separately for batch, channel and
    spatial dimensions.

    See `composable_mapping` for special handling of shapes in the codebase.

    Args:
        tensors: Optional tensors to broadcast.
        broadcast_batch: Whether to broadcast batch dimension.
        broadcast_channels: Whether to broadcast channel dimensions.
        broadcast_spatial: Whether to broadcast spatial dimensions.
        n_channel_dims: Number of channel dims for each tensor, if integer is
            given, same number will be used for all tensors.

    Returns:
        Broadcasted tensors with the same None positions as the input tensors.
    """
    shapes = [None if tensor is None else tensor.shape for tensor in tensors]
    broadcasted_batch_shape, broadcasted_channel_shape, broadcasted_spatial_shape = (
        broadcast_optional_shapes_in_parts_splitted(
            *shapes,
            n_channel_dims=n_channel_dims,
            broadcast_batch=broadcast_batch,
            broadcast_channels=broadcast_channels,
            broadcast_spatial=broadcast_spatial,
        )
    )
    return tuple(
        (
            None
            if tensor is None
            else broadcast_to_in_parts(
                tensor,
                batch_shape=broadcasted_batch_shape,
                channels_shape=broadcasted_channel_shape,
                spatial_shape=broadcasted_spatial_shape,
                n_channel_dims=individual_n_channel_dims,
            )
        )
        for tensor, individual_n_channel_dims in zip(
            tensors,
            _n_dims_to_iterable(n_channel_dims),
        )
    )


def split_shape(
    shape: Sequence[int], n_channel_dims: int = 1
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """Splits shape into batch, channel and spatial shapes.

    See `composable_mapping` for special handling of shapes in the codebase.

    Args:
        shape: Shape of the tensor.
        n_channel_dims: Number of channel dimensions.

    Returns:
        Tuple of batch shape, channel shape and spatial shape.
    """
    channel_dims = get_channel_dims(len(shape), n_channel_dims)
    n_channel_dims = len(channel_dims)
    first_channel_dim = channel_dims[0]
    return (
        tuple(shape[:first_channel_dim]),
        tuple(shape[first_channel_dim : first_channel_dim + n_channel_dims]),
        tuple(shape[first_channel_dim + n_channel_dims :]),
    )


def combine_optional_masks(
    *masks: Optional[Tensor],
    n_channel_dims: Union[int, Iterable[int]] = 1,
) -> Optional[Tensor]:
    """Combine optional masks.

    Args:
        masks: Masks to combine.
        n_channel_dims: Number of channel dims for each mask, if integer is
            given, same number will be used for all masks.

    Returns:
        Combined mask.
    """
    broadcasted_masks = broadcast_tensors_in_parts(
        *(mask for mask in masks if mask is not None),
        n_channel_dims=n_channel_dims,
    )
    combined_mask: Optional[Tensor] = None
    for mask in broadcasted_masks:
        if mask is not None:
            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = combined_mask & mask
    return combined_mask


def are_broadcastable(shape_1: Sequence[int], shape_2: Sequence[int]) -> bool:
    """Check if two shapes are broadcastable.

    Args:
        shape_1: First shape.
        shape_2: Second shape.

    Returns:
        Whether the shapes are broadcastable.
    """
    if any(
        dim_size_1 != dim_size_2 and dim_size_1 != 1 and dim_size_2 != 1
        for dim_size_1, dim_size_2 in zip(shape_1[::-1], shape_2[::-1])
    ):
        return False
    return True


def is_broadcastable_to(source_shape: Sequence[int], target_shape: Sequence[int]) -> bool:
    """Check if source shape is broadcastable to target shape.

    Args:
        source_shape: Source shape.
        target_shape: Target shape.

    Returns:
        Whether the source shape is broadcastable to the target shape.
    """
    if len(target_shape) < len(source_shape) or any(
        dim_size_1 not in {dim_size_2, 1}
        for dim_size_1, dim_size_2 in zip(source_shape[::-1], target_shape[::-1])
    ):
        return False
    return True


def crop_and_then_pad_spatial(
    tensor: Tensor,
    pads_or_crops: Sequence[Tuple[int, int]],
    mode: str = "constant",
    value: Optional[float] = 0.0,
    n_channel_dims: int = 1,
) -> Tensor:
    """Crop or pad spatial dimensions starting from the first spatial dimension.

    Applies cropping first and then padding. Negative values in pads_or_crops
    indicate cropping.

    Args:
        tensor: Tensor to pad or crop.
        pads_or_crops: Pads or crops to apply. Each element is a tuple of
            padding or cropping for the start and end of the spatial dimension.
            Negative values indicate cropping.
        mode: Padding mode.
        value: Value to pad with.
        n_channel_dims: Number of channel dimensions.

    Returns:
        Padded or cropped tensor.
    """
    batch_shape, channels_shape, spatial_shape = split_shape(tensor.shape, n_channel_dims)
    n_spatial_dims = len(spatial_shape)
    if len(pads_or_crops) != n_spatial_dims:
        raise ValueError("Number of paddings must match number of spatial dimensions")
    if all(padding == (0, 0) for padding in pads_or_crops):
        return tensor
    crops = []
    for (padding_start, padding_end), spatial_dim_size in zip(pads_or_crops, spatial_shape):
        if not _is_dim_croppable_first(spatial_dim_size, padding_start, padding_end, mode):
            raise ValueError("Invalid crops and pads provided")
        if padding_start >= 0:
            crop_start = None
        else:
            crop_start = min(-padding_start, spatial_dim_size)
        if padding_end >= 0:
            crop_end = None
        else:
            crop_end = -min(-padding_end, spatial_dim_size)
        crops.append(slice(crop_start, crop_end))
    torch_paddings = []
    for (padding_start, padding_end), spatial_dim_size in reversed(
        list(zip(pads_or_crops, spatial_shape))
    ):
        output_dim_size = spatial_dim_size + padding_start + padding_end
        if padding_start >= 0:
            torch_paddings.append(min(padding_start, output_dim_size))
        else:
            torch_paddings.append(0)
        if padding_end >= 0:
            torch_paddings.append(min(padding_end, output_dim_size))
        else:
            torch_paddings.append(0)
    tensor = tensor.view(batch_shape + (-1,) + spatial_shape)
    cropped_and_padded = pad(
        tensor[(...,) + tuple(crops)],
        torch_paddings,
        mode=mode,
        value=value,
    )
    cropped_and_padded_spatial_shape = get_spatial_shape(cropped_and_padded.shape, n_channel_dims=1)
    return cropped_and_padded.view(batch_shape + channels_shape + cropped_and_padded_spatial_shape)


def is_croppable_first(
    spatial_shape: Sequence[int],
    pads_or_crops: Sequence[Tuple[int, int]],
    mode: str = "constant",
) -> bool:
    """Checks whether the pads or crops can be applied by applying the cropping first.

    Args:
        spatial_shape: Shape of the spatial dimensions.
        pads_or_crops: Pads or crops to apply. Each element is a tuple of
            padding or cropping for the start and end of the spatial dimension.
            Negative values indicate cropping.
        mode: Padding mode.

    Returns:
        Whether the pads or crops can be applied by applying the cropping first.
    """
    if mode == "constant":
        return True
    n_spatial_dims = len(spatial_shape)
    if len(pads_or_crops) != n_spatial_dims:
        raise ValueError("Number of paddings must match number of spatial dimensions")
    for (padding_start, padding_end), spatial_dim_size in zip(pads_or_crops, spatial_shape):
        if not _is_dim_croppable_first(spatial_dim_size, padding_start, padding_end, mode):
            return False
    return True


def _is_dim_croppable_first(
    spatial_dim_size: int,
    padding_start: int,
    padding_end: int,
    mode: str,
) -> bool:
    if mode == "constant":
        return True
    if spatial_dim_size + padding_start + padding_end < 0:
        return False
    if padding_start <= 0 and padding_end <= 0:
        return True
    if padding_start >= 0 and padding_end >= 0:
        if mode in ("reflect", "circular") and (
            padding_start >= spatial_dim_size or padding_end >= spatial_dim_size
        ):
            return False
        return True
    pad_width = max(padding_start, padding_end)
    crop_width = -min(padding_start, padding_end)
    remaining_width = spatial_dim_size - crop_width
    if mode == "replicate" and remaining_width <= 0:
        return False
    if mode in ("reflect", "circular") and pad_width >= remaining_width:
        return False
    return True


def includes_padding(
    pads_or_crops: Sequence[Tuple[int, int]],
) -> bool:
    """Checks if pads or crops include padding (positive values).

    Args:
        pads_or_crops: Pads or crops to apply. Each element is a tuple of
            padding or cropping for the start and end of the spatial dimension.
            Negative values indicate cropping.

    Returns:
        Whether pads or crops include padding.
    """
    return any(padding_start > 0 or padding_end > 0 for padding_start, padding_end in pads_or_crops)
