"""Returns constant value for all spatial locations"""

from typing import Mapping

from torch import Tensor

from .composable_mapping import ComposableMapping
from .mappable_tensor import MappableTensor, mappable
from .tensor_like import TensorLike
from .util import broadcast_to_in_parts, has_spatial_dims


class Constant(ComposableMapping):
    """Mapping which returns constant value for all spatial locations.

    Arguments.
        value: Tensor with shape (*batch_shape, *channels_shape)
        n_channel_dims: Number of channel dimensions.
    """

    def __init__(
        self,
        value: Tensor,
        n_channel_dims: int = 1,
    ) -> None:
        self._value = value
        self._n_channel_dims = n_channel_dims
        if has_spatial_dims(value.shape, n_channel_dims=n_channel_dims):
            raise ValueError("Constant should not have spatial dimensions.")

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {"value": self._value}

    def _get_children(self) -> Mapping[str, TensorLike]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, TensorLike]
    ) -> "Constant":
        return Constant(value=tensors["value"], n_channel_dims=self._n_channel_dims)

    def __call__(self, coordinates: MappableTensor) -> MappableTensor:
        broadcasted_values = broadcast_to_in_parts(
            self._value,
            batch_shape=coordinates.batch_shape,
            spatial_shape=coordinates.spatial_shape,
            n_channel_dims=self._n_channel_dims,
        )
        return mappable(
            broadcasted_values,
            mask=coordinates.generate_mask(generate_missing_mask=False, cast_mask=False),
            n_channel_dims=self._n_channel_dims,
        )

    def invert(self, **arguments):
        raise NotImplementedError("Constant mapping is not invertible")

    def __repr__(self) -> str:
        return f"Constant(value={self._value})"
