"""Utilities for mappable tensors."""

from typing import Optional

from torch import Tensor, cat, stack

from MIR.models.SITReg.composable_mapping.util import combine_optional_masks, get_channel_dims

from .mappable_tensor import MappableTensor, mappable


def concatenate_mappable_tensors(
    *mappable_tensors: MappableTensor, channel_index: int = 0
) -> "MappableTensor":
    """Concatenate mappable tensors along the channel dimension

    Args:
        mappable_tensors: Mappable tensors to concatenate
        channel_index: Index of the channel dimension starting from the first
            channel dimension over which to concatenate.

    Returns:
        Concatenated masked tensor.
    """
    if not all(
        mappable_tensor.n_channel_dims == mappable_tensors[0].n_channel_dims
        for mappable_tensor in mappable_tensors[1:]
    ):
        raise ValueError("Lengths of channel shapes of masked tensors must be the same")
    n_channel_dims = len(mappable_tensors[0].channels_shape)
    concatenation_dim = get_channel_dims(
        n_total_dims=len(mappable_tensors[0].shape),
        n_channel_dims=n_channel_dims,
    )[channel_index]
    values = cat(
        [mappable_tensor.generate_values() for mappable_tensor in mappable_tensors],
        dim=concatenation_dim,
    )
    mask: Optional[Tensor] = None
    for mappable_tensor in mappable_tensors:
        update_mask = mappable_tensor.generate_mask(generate_missing_mask=False, cast_mask=False)
        mask = combine_optional_masks(mask, update_mask, n_channel_dims=n_channel_dims)
    return mappable(
        values=values,
        mask=mask,
        n_channel_dims=n_channel_dims,
    )


def stack_mappable_tensors(
    *mappable_tensors: MappableTensor, channel_index: int = 0
) -> "MappableTensor":
    """Stack mappable tensors along the channel dimension.

    Args:
        mappable_tensors: Mappable tensors to concatenate
        channel_index: Index of the channel dimension over which to stack
            starting from the first channel dimension over which to stack.

    Returns:
        Stacked masked tensor.
    """
    if not all(
        mappable_tensor.n_channel_dims == mappable_tensors[0].n_channel_dims
        for mappable_tensor in mappable_tensors[1:]
    ):
        raise ValueError("Lengths of channel shapes of masked tensors must be the same")
    n_channel_dims = len(mappable_tensors[0].channels_shape)
    channel_dims = get_channel_dims(
        n_total_dims=len(mappable_tensors[0].shape),
        n_channel_dims=n_channel_dims,
    )
    stacking_dim = (channel_dims + (channel_dims[-1] + 1,))[channel_index]
    values = stack(
        [mappable_tensor.generate_values() for mappable_tensor in mappable_tensors],
        dim=stacking_dim,
    )
    mask: Optional[Tensor] = None
    for mappable_tensor in mappable_tensors:
        update_mask = mappable_tensor.generate_mask(generate_missing_mask=False, cast_mask=False)
        if update_mask is not None:
            update_mask = update_mask.unsqueeze(dim=stacking_dim)
            mask = combine_optional_masks(mask, update_mask, n_channel_dims=n_channel_dims + 1)
    return mappable(
        values=values,
        mask=mask,
        n_channel_dims=n_channel_dims + 1,
    )
