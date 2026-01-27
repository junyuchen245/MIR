"""Utility functions for composable mappings."""

from typing import Mapping, TypeVar, cast

from torch import Tensor

from MIR.models.SITReg.composable_mapping.mappable_tensor import (
    MappableTensor,
    concatenate_mappable_tensors,
    stack_mappable_tensors,
)
from MIR.models.SITReg.composable_mapping.tensor_like import TensorLike

from .composable_mapping import ComposableMapping, GridComposableMapping

ComposableMappingT = TypeVar("ComposableMappingT", bound="ComposableMapping")


def stack_mappings(*mappings: ComposableMappingT, channel_index: int = 0) -> ComposableMappingT:
    """Stack mappings along a channel dimension.

    Args:
        mappings: Mappings to stack.
        channel_index: Channel index along which to stack.

    Returns:
        A mapping with the output being the outputs of the input mappings
        stacked along the channel dimension.
    """
    stacked: ComposableMapping = _Stack(*mappings, channel_index=channel_index)
    for mapping in mappings:
        if isinstance(mapping, GridComposableMapping):
            stacked = stacked.assign_coordinates(mapping)
            break
    return cast(ComposableMappingT, stacked)


def concatenate_mappings(
    *mappings: ComposableMappingT, channel_index: int = 0
) -> ComposableMappingT:
    """Concatenate mappings along a channel dimension.

    Args:
        mappings: Mappings to concatenate.
        channel_index: Channel index along which to concatenate.

    Returns:
        A mapping with the output being the outputs of the input mappings
        concatenated along the channel dimension.
    """
    concatenated: ComposableMapping = _Concatenate(*mappings, channel_index=channel_index)
    for mapping in mappings:
        if isinstance(mapping, GridComposableMapping):
            concatenated = concatenated.assign_coordinates(mapping)
            break
    return cast(ComposableMappingT, concatenated)


class _Stack(ComposableMapping):
    """Stacked mappings."""

    def __init__(self, *mappings: ComposableMapping, channel_index: int) -> None:
        super().__init__()
        self._mappings = mappings
        self._channel_index = channel_index

    @property
    def default_resampling_data_format(self) -> None:
        return None

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, TensorLike]
    ) -> "_Stack":
        return _Stack(
            *(cast(ComposableMapping, children[f"mapping_{i}"]) for i in range(len(children))),
            channel_index=self._channel_index,
        )

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, TensorLike]:
        children = {}
        for i, mapping in enumerate(self._mappings):
            children[f"mapping_{i}"] = mapping
        return children

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        return stack_mappable_tensors(
            *(mapping(masked_coordinates) for mapping in self._mappings),
            channel_index=self._channel_index,
        )

    def invert(self, **arguments) -> "ComposableMapping":
        raise NotImplementedError("Inversion of stacked mappings is not implemented")

    def __repr__(self) -> str:
        return f"_Stack(mappings={self._mappings}, " f"channel_index={self._channel_index})"


class _Concatenate(ComposableMapping):
    """Concatenated mappings."""

    def __init__(self, *mappings: ComposableMapping, channel_index: int) -> None:
        super().__init__()
        self._mappings = mappings
        self._channel_index = channel_index

    @property
    def default_resampling_data_format(self) -> None:
        return None

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, TensorLike]
    ) -> "_Concatenate":
        return _Concatenate(
            *(cast(ComposableMapping, children[f"mapping_{i}"]) for i in range(len(children))),
            channel_index=self._channel_index,
        )

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, TensorLike]:
        children = {}
        for i, mapping in enumerate(self._mappings):
            children[f"mapping_{i}"] = mapping
        return children

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        return concatenate_mappable_tensors(
            *(mapping(masked_coordinates) for mapping in self._mappings),
            channel_index=self._channel_index,
        )

    def invert(self, **arguments) -> "ComposableMapping":
        raise NotImplementedError("Inversion of stacked mappings is not implemented")

    def __repr__(self) -> str:
        return f"_Concatenate(mappings={self._mappings}, " f"channel_index={self._channel_index})"
