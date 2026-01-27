"""Composable mappings for modifying masks of the input tensors."""

from typing import Mapping, Sequence

from torch import Tensor
from torch import all as torch_all
from torch import ge, gt, le, lt, tensor

from MIR.models.SITReg.composable_mapping.util import move_channels_first, move_channels_last

from .composable_mapping import ComposableMapping
from .mappable_tensor import MappableTensor
from .tensor_like import TensorLike


class RectangleMask(ComposableMapping):
    """Modify mask of the input based on bounds

    Arguments.
        min_values: Minimum values for the mask over each dimension.
        max_values: Maximum values for the mask over each dimension.
        inclusive_min: Whether the minimum values are inclusive.
        inclusive_max: Whether the maximum values are inclusive
    """

    def __init__(
        self,
        min_values: Sequence[float],
        max_values: Sequence[float],
        inclusive_min: bool = True,
        inclusive_max: bool = True,
    ) -> None:
        self._min_values = min_values
        self._max_values = max_values
        self._inclusive_min = inclusive_min
        self._inclusive_max = inclusive_max

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, TensorLike]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, TensorLike]
    ) -> "RectangleMask":
        return RectangleMask(min_values=self._min_values, max_values=self._max_values)

    def __call__(self, coordinates: MappableTensor) -> MappableTensor:
        values = coordinates.generate_values()
        update_mask = self._generate_mask_based_on_bounds(
            coordinates=values,
            n_channel_dims=coordinates.n_channel_dims,
            min_values=self._min_values,
            max_values=self._max_values,
            inclusive_min=self._inclusive_min,
            inclusive_max=self._inclusive_max,
        )
        return coordinates.mask_and(update_mask)

    @staticmethod
    def _generate_mask_based_on_bounds(
        coordinates: Tensor,
        min_values: Sequence[float],
        max_values: Sequence[float],
        n_channel_dims: int = 1,
        inclusive_min: bool = True,
        inclusive_max: bool = True,
    ) -> Tensor:
        coordinates = move_channels_last(coordinates.detach(), n_channel_dims=n_channel_dims)
        non_blocking = coordinates.device.type != "cpu"
        min_values_tensor = tensor(min_values, dtype=coordinates.dtype).to(
            device=coordinates.device, non_blocking=non_blocking
        )
        max_values_tensor = tensor(max_values, dtype=coordinates.dtype).to(
            device=coordinates.device, non_blocking=non_blocking
        )
        normalized_coordinates = (coordinates - min_values_tensor) / (
            max_values_tensor - min_values_tensor
        )
        min_operator = ge if inclusive_min else gt
        max_operator = le if inclusive_max else lt
        fov_mask = min_operator(normalized_coordinates, 0) & max_operator(normalized_coordinates, 1)
        fov_mask = move_channels_first(
            torch_all(fov_mask, dim=-1, keepdim=True), n_channel_dims=n_channel_dims
        )
        return fov_mask

    def invert(self, **arguments):
        raise NotImplementedError("Rectangle mask is not invertible")

    def detach(self) -> "RectangleMask":
        return self

    def __repr__(self) -> str:
        return f"RectangleMask(min_values={self._min_values}, max_values={self._max_values})"


class ClearMask(ComposableMapping):
    """Clear mask of the input"""

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, TensorLike]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, TensorLike]
    ) -> "ClearMask":
        return ClearMask()

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        return masked_coordinates.clear_mask()

    def invert(self, **arguments):
        raise NotImplementedError("Mask clearing is not invertible")

    def detach(self) -> "ClearMask":
        return self

    def __repr__(self) -> str:
        return "ClearMask()"
