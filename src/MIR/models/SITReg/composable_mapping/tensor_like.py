"""Abstraction for PyTorch tensor like objects."""

from abc import ABC, abstractmethod
from typing import Mapping, Optional, TypeVar

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype

TensorLikeT = TypeVar("TensorLikeT", bound="TensorLike")


class TensorLike(ABC):
    """Interface for classes having tensor like properties.

    Usually contains wrapped tensors with device, dtype, and detachment
    functionalities corresponding directly to the wrapped tensors.
    """

    def _get_tensors(self) -> Mapping[str, Tensor]:
        """Obtain list of wrapped tensors"""
        return {}

    def _get_children(self) -> Mapping[str, "TensorLike"]:
        """Obtain list of tensor like objects contained within the current object"""
        return {}

    @abstractmethod
    def _modified_copy(
        self: TensorLikeT,
        tensors: Mapping[str, Tensor],
        children: Mapping[str, "TensorLike"],
    ) -> TensorLikeT:
        """Create a modified copy of the object with new tensors and children.

        Args:
            tensors: New tensors to be wrapped.
            children: New children to be wrapped.

        Returns:
            Modified copy of the object.
        """

    @property
    def dtype(
        self,
    ) -> torch_dtype:
        """PyTorch data type."""
        tensors = self._get_tensors()
        if not tensors:
            return next(iter(self._get_children().values())).dtype
        return next(iter(tensors.values())).dtype

    @property
    def device(
        self,
    ) -> torch_device:
        """PyTorch data device."""
        tensors = self._get_tensors()
        if not tensors:
            return next(iter(self._get_children().values())).device
        return next(iter(tensors.values())).device

    def cast(
        self: TensorLikeT,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
        non_blocking: bool = False,
    ) -> TensorLikeT:
        """Cast to given data type and device.

        We will not use method name "to" as it would create conflict with
        torch.nn.Module.to method which does casting in-place.

        Args:
            dtype: Data type to cast to.
            device: Device to cast to.
            non_blocking: Whether to perform the operation asynchronously (if
                possible).

        Returns:
            New tensor like object with the casted tensor(s).
        """
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        modified_tensors = {
            key: tensor.to(dtype=dtype, device=device, non_blocking=non_blocking)
            for key, tensor in self._get_tensors().items()
        }
        modified_children = {
            key: child.cast(dtype=dtype, device=device, non_blocking=non_blocking)
            for key, child in self._get_children().items()
        }
        return self._modified_copy(modified_tensors, modified_children)

    def detach(self: TensorLikeT) -> TensorLikeT:
        """Detach the wrapped tensors from computational graph."""
        modified_tensors = {key: tensor.detach() for key, tensor in self._get_tensors().items()}
        modified_children = {key: child.detach() for key, child in self._get_children().items()}
        return self._modified_copy(modified_tensors, modified_children)
