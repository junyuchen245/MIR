"""Affine transformations on PyTorch tensors."""

from abc import abstractmethod
from typing import Mapping, Optional, Sequence, Tuple, Union, cast, overload

from torch import Tensor, broadcast_shapes
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import ones, zeros
import torch
def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

from MIR.models.SITReg.composable_mapping.interface import Number
from MIR.models.SITReg.composable_mapping.tensor_like import TensorLike
from MIR.models.SITReg.composable_mapping.util import (
    are_broadcastable,
    broadcast_shapes_in_parts,
    broadcast_to_in_parts,
    get_batch_shape,
    get_channel_dims,
    get_channels_shape,
    get_spatial_shape,
    is_broadcastable_to,
    split_shape,
)

from .diagonal_matrix import (
    DiagonalAffineMatrixDefinition,
    add_diagonal_affine_matrices,
    compose_diagonal_affine_matrices,
    invert_diagonal_affine_matrix,
    is_identity_diagonal_affine_matrix,
    is_zero_diagonal_affine_matrix,
    negate_diagonal_affine_matrix,
    transform_values_with_diagonal_affine_matrix,
)
from .matrix import (
    add_affine_matrices,
    clear_translation_from_affine_matrix,
    compose_affine_matrices,
    embed_matrix,
    invert_matrix,
    is_identity_matrix,
    is_zero_matrix,
    negate_affine_matrix,
    transform_values_with_affine_matrix,
)


class IAffineTransformation(TensorLike):
    """Affine transformation"""

    @abstractmethod
    def __matmul__(
        self, affine_transformation: "IAffineTransformation"
    ) -> "IAffineTransformation": ...

    @abstractmethod
    def __add__(
        self, affine_transformation: "IAffineTransformation"
    ) -> "IAffineTransformation": ...

    @abstractmethod
    def __sub__(
        self, affine_transformation: "IAffineTransformation"
    ) -> "IAffineTransformation": ...

    @abstractmethod
    def __neg__(self) -> "IAffineTransformation": ...

    @abstractmethod
    def invert(self) -> "IAffineTransformation":
        """Invert the transformation"""

    @abstractmethod
    def as_matrix(
        self,
    ) -> Tensor:
        """Obtain the transformation as the corresponding affine transformation
        matrix"""

    @abstractmethod
    def as_host_matrix(self) -> Optional[Tensor]:
        """Obtain the transformation as the corresonding affine transformation
        matrix on host (cpu), if available"""

    @abstractmethod
    def as_diagonal(
        self,
    ) -> Optional[DiagonalAffineMatrixDefinition]:
        """Obtain the transformation as corresponding diagonal affine
        transformation matrix, if available"""

    @abstractmethod
    def as_host_diagonal(
        self,
    ) -> Optional[DiagonalAffineMatrixDefinition]:
        """Obtain the mapping as corresponding diagonal affine transformation
        matrix on host (cpu), if available"""

    @abstractmethod
    def __call__(self, values: Tensor, n_channel_dims: int = 1) -> Tensor:
        """Evaluate the transformation at values

        Args:
            values: Tensor with shape
                ([*batch_shape, *channels_shape[:-1], ]n_input_dims[, *spatial_shape])
            n_channel_dims: Number of channel dimensions in the values tensor

        Returns:
            Tensor with shape
                ([*batch_shape, *channels_shape[:-1], ]n_output_dims[, *spatial_shape])
        """

    @abstractmethod
    def get_output_shape(
        self, input_shape: Sequence[int], n_channel_dims: int = 1
    ) -> Tuple[int, ...]:
        """Shape of the output tensor given the input shape

        Args:
            input_shape: Shape of the input tensor in form
                ([*batch_shape, *channels_shape[:-1], ]n_input_dims[, *spatial_shape])
            n_channel_dims: Number of channel dimensions in the input tensor

        Returns:
            Shape of the output tensor in form
                ([*batch_shape, *channels_shape[:-1], ]n_output_dims[, *spatial_shape])

        Raises:
            RuntimeError if the transformation is not compatible with the input shape
        """

    @abstractmethod
    def is_zero(self) -> Optional[bool]:
        """Return whether the transformation corresponds to zero matrix

        Returns:
            None if the check cannot be done on CPU, otherwise bool indicating
            whether the transformation is zero matrix.
        """

    @abstractmethod
    def is_identity(self) -> Optional[bool]:
        """Return whether the transformation corresponds to identity matrix

        Returns:
            None if the check cannot be done on CPU, otherwise bool indicating
            whether the transformation is identity matrix.
        """

    @property
    @abstractmethod
    def shape(self) -> Sequence[int]:
        """Shape of the transformation when represented as an affine
        transformation matrix

        The spape is in form
            ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape])
        """

    @property
    @abstractmethod
    def batch_shape(self) -> Tuple[int, ...]:
        """Shape of the batch dimensions"""

    @property
    @abstractmethod
    def channels_shape(self) -> Tuple[int, ...]:
        """Shape of the channel dimensions"""

    @property
    @abstractmethod
    def spatial_shape(self) -> Tuple[int, ...]:
        """Shape of the spatial dimensions"""

    @abstractmethod
    def is_composable(self, affine_transformation: "IAffineTransformation") -> bool:
        """Is the transformation composable with the other transformation

        Args:
            affine_transformation: Transformation to compose with

        Returns:
            Whether the transformation is composable with the other transformation
        """

    @abstractmethod
    def is_addable(self, affine_transformation: "IAffineTransformation") -> bool:
        """Is the transformation addable with the other transformation

        Args:
            affine_transformation: Transformation to add with

        Returns:
            Whether the transformation is addable with the other transformation"""

    @abstractmethod
    def broadcast_to_n_output_channels(
        self,
        n_output_channels: int,
    ) -> "IAffineTransformation":
        """Modify the transformation to output n_output_channels channels such that the output
        would equal broadcasting the original output to have n_output_channels channels.

        Args:
            n_output_channels: Number of output channels

        Returns:
            Affine transformation with n_output_channels output channels
        """

    @abstractmethod
    def clear_translation(self) -> "IAffineTransformation":
        """Clear translation from the transformation"""


class IHostAffineTransformation(IAffineTransformation):
    """Affine transformation for which the matrix is stored on host (cpu) and
    the matrix on target device is created only when needed.

    Allows to do control flow decisions on host based on the transformation
    without having to do CPU-GPU synchronization.
    """

    @overload
    def __matmul__(
        self, affine_transformation: "IHostAffineTransformation"
    ) -> "IHostAffineTransformation": ...

    @overload
    def __matmul__(
        self, affine_transformation: "IAffineTransformation"
    ) -> "IAffineTransformation": ...

    @abstractmethod
    def __matmul__(
        self, affine_transformation: "IAffineTransformation"
    ) -> "IAffineTransformation": ...

    @overload
    def __add__(
        self, affine_transformation: "IHostAffineTransformation"
    ) -> "IHostAffineTransformation": ...

    @overload
    def __add__(
        self, affine_transformation: "IAffineTransformation"
    ) -> "IAffineTransformation": ...

    @abstractmethod
    def __add__(
        self, affine_transformation: "IAffineTransformation"
    ) -> "IAffineTransformation": ...

    @overload
    def __sub__(
        self, affine_transformation: "IHostAffineTransformation"
    ) -> "IHostAffineTransformation": ...

    @overload
    def __sub__(
        self, affine_transformation: "IAffineTransformation"
    ) -> "IAffineTransformation": ...

    @abstractmethod
    def __sub__(
        self, affine_transformation: "IAffineTransformation"
    ) -> "IAffineTransformation": ...

    @abstractmethod
    def invert(self) -> "IHostAffineTransformation":
        pass

    @abstractmethod
    def as_host_matrix(self) -> Tensor:
        """Obtain the transformation as the corresonding affine transformation
        matrix on host (cpu)"""

    @abstractmethod
    def clear_translation(self) -> "IHostAffineTransformation":
        pass


class BaseAffineTransformation(IAffineTransformation):
    """Base affine transformation implementation"""

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        return get_batch_shape(self.shape, n_channel_dims=2)

    @property
    def channels_shape(self) -> Tuple[int, ...]:
        return get_channels_shape(self.shape, n_channel_dims=2)

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        return get_spatial_shape(self.shape, n_channel_dims=2)

    def is_composable(self, affine_transformation: "IAffineTransformation") -> bool:
        return (
            are_broadcastable(affine_transformation.batch_shape, self.batch_shape)
            and are_broadcastable(affine_transformation.spatial_shape, self.spatial_shape)
            and self.channels_shape[1] == affine_transformation.channels_shape[0]
        )

    def is_addable(self, affine_transformation: "IAffineTransformation") -> bool:
        return (
            are_broadcastable(affine_transformation.batch_shape, self.batch_shape)
            and are_broadcastable(affine_transformation.spatial_shape, self.spatial_shape)
            and self.channels_shape == affine_transformation.channels_shape
        )

    def broadcast_to_n_output_channels(
        self,
        n_output_channels: int,
    ) -> IAffineTransformation:
        n_previous_output_channels = self.channels_shape[0] - 1
        if not is_broadcastable_to((n_previous_output_channels,), (n_output_channels,)):
            raise RuntimeError("Cannot broadcast to the given number of output channels")
        target_n_output_channels = broadcast_shapes(
            (n_previous_output_channels,), (n_output_channels,)
        )[0]
        if target_n_output_channels == n_output_channels:
            return self
        assert n_previous_output_channels == 1
        return (
            HostAffineTransformation(
                transformation_matrix_on_host=embed_matrix(
                    ones(
                        target_n_output_channels,
                        1,
                        dtype=self.dtype,
                        device=torch_device("cpu"),
                    ),
                    target_shape=(n_output_channels + 1, 2),
                ),
                device=self.device,
            )
            @ self
        )

    def get_output_shape(
        self, input_shape: Sequence[int], n_channel_dims: int = 1
    ) -> Tuple[int, ...]:
        try:
            broadcasted_input_shape, _broadcasted_matrix_shape = broadcast_shapes_in_parts(
                input_shape,
                self.shape,
                n_channel_dims=(n_channel_dims, 2),
                broadcast_channels=False,
            )
        except RuntimeError as broadcasting_error:
            raise RuntimeError(
                "Input shape does not match the transformation matrix"
            ) from broadcasting_error
        matrix_channel_dims = get_channel_dims(len(self.shape), n_channel_dims=2)
        n_matrix_input_channels = self.shape[matrix_channel_dims[1]] - 1
        last_input_channel_index = get_channel_dims(
            len(input_shape), n_channel_dims=n_channel_dims
        )[-1]
        if input_shape[last_input_channel_index] != n_matrix_input_channels:
            raise RuntimeError("Input shape does not match the transformation matrix")
        n_output_channels = self.shape[matrix_channel_dims[0]] - 1
        modified_input_shape = list(broadcasted_input_shape)
        modified_input_shape[last_input_channel_index] = n_output_channels
        return tuple(modified_input_shape)

    @overload
    def __matmul__(
        self, affine_transformation: IHostAffineTransformation
    ) -> IHostAffineTransformation: ...

    @overload
    def __matmul__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation: ...

    def __matmul__(self, affine_transformation: IAffineTransformation) -> "IAffineTransformation":
        if not isinstance(affine_transformation, IAffineTransformation):
            return NotImplemented
        if self.is_identity():
            if not self.is_composable(affine_transformation):
                raise RuntimeError("Transformation is not composable with the other transformation")
            return affine_transformation
        if affine_transformation.is_identity():
            if not affine_transformation.is_composable(self):
                raise RuntimeError("Transformation is not composable with the other transformation")
            return self
        self_host_matrix = self.as_host_matrix()
        target_host_matrix = affine_transformation.as_host_matrix()
        if self_host_matrix is not None and target_host_matrix is not None:
            return HostAffineTransformation(
                transformation_matrix_on_host=compose_affine_matrices(
                    self_host_matrix, target_host_matrix
                ),
                device=self.device,
            )
        return AffineTransformation(
            compose_affine_matrices(
                self.as_matrix(),
                affine_transformation.as_matrix(),
            )
        )

    @overload
    def __add__(
        self, affine_transformation: IHostAffineTransformation
    ) -> IHostAffineTransformation: ...

    @overload
    def __add__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation: ...

    def __add__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation:
        if not isinstance(affine_transformation, IAffineTransformation):
            return NotImplemented
        self_host_matrix = self.as_host_matrix()
        target_host_matrix = affine_transformation.as_host_matrix()
        if self_host_matrix is not None and target_host_matrix is not None:
            return HostAffineTransformation(
                transformation_matrix_on_host=add_affine_matrices(
                    self_host_matrix, target_host_matrix
                ),
                device=self.device,
            )
        return AffineTransformation(
            add_affine_matrices(self.as_matrix(), affine_transformation.as_matrix())
        )

    @overload
    def __sub__(
        self, affine_transformation: IHostAffineTransformation
    ) -> IHostAffineTransformation: ...

    @overload
    def __sub__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation: ...

    def __sub__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation:
        if not isinstance(affine_transformation, IAffineTransformation):
            return NotImplemented
        return self.__add__(-affine_transformation)

    def is_zero(self) -> Optional[bool]:
        host_matrix = self.as_host_matrix()
        if host_matrix is not None:
            return is_zero_matrix(host_matrix)
        return None

    def is_identity(self) -> Optional[bool]:
        host_matrix = self.as_host_matrix()
        if host_matrix is not None:
            return is_identity_matrix(host_matrix)
        return None

    def as_diagonal(self) -> Optional[DiagonalAffineMatrixDefinition]:
        return None

    def as_host_diagonal(self) -> Optional[DiagonalAffineMatrixDefinition]:
        return None


class AffineTransformation(BaseAffineTransformation):
    """Generic affine transformation

    Arguments:
        transformation_matrix: Tensor with shape
            ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape])
    """

    def __init__(self, transformation_matrix: Tensor) -> None:
        self._transformation_matrix = transformation_matrix

    @property
    def shape(self) -> Sequence[int]:
        return self._transformation_matrix.shape

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {"transformation_matrix": self._transformation_matrix}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, TensorLike]
    ) -> "AffineTransformation":
        return AffineTransformation(tensors["transformation_matrix"])

    def __call__(self, values: Tensor, n_channel_dims: int = 1) -> Tensor:
        return transform_values_with_affine_matrix(
            self.as_matrix(),
            values,
            n_channel_dims=n_channel_dims,
        )

    def as_matrix(
        self,
    ) -> Tensor:
        return self._transformation_matrix

    def as_host_matrix(self) -> Optional[Tensor]:
        return None

    def __neg__(self) -> "AffineTransformation":
        return AffineTransformation(negate_affine_matrix(self.as_matrix()))

    def invert(self) -> "AffineTransformation":
        return AffineTransformation(invert_matrix(self.as_matrix()))

    def __repr__(self) -> str:
        return f"AffineTransformation(transformation_matrix={self._transformation_matrix})"

    def clear_translation(self) -> "AffineTransformation":
        return AffineTransformation(clear_translation_from_affine_matrix(self.as_matrix()))


class HostAffineTransformation(AffineTransformation, IHostAffineTransformation):
    """Affine transformation with the matrix stored on host (cpu) and the matrix
    on target device created only when needed.

    Allows to do control flow decisions on host based on the transformation
    without having to do CPU-GPU synchronization.

    Arguments:
        transformation_matrix_on_host: Transformation matrix on cpu, tensor with shape
            ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape]).
        device: Device to use for the transformation matrix produced by as_matrix method.
    """

    def __init__(
        self,
        transformation_matrix_on_host: Tensor,
        device: Optional[torch_device] = None,
    ) -> None:
        if transformation_matrix_on_host.device.type != "cpu":
            raise ValueError("Please give the matrix on CPU")
        if transformation_matrix_on_host.requires_grad:
            raise ValueError("The implementation assumes a detached transformation matrix.")
        super().__init__(transformation_matrix=transformation_matrix_on_host)
        self._device = get_default_device() if device is None else device

    def __call__(self, values: Tensor, n_channel_dims: int = 1) -> Tensor:
        if (
            is_identity_matrix(self._transformation_matrix)
            and self.get_output_shape(values.shape) == values.shape
        ):
            return values
        if self.is_zero():
            return zeros(
                1,
                device=self._device,
                dtype=values.dtype,
            ).expand(self.get_output_shape(values.shape, n_channel_dims))
        return super().__call__(values, n_channel_dims)

    def as_matrix(
        self,
    ) -> Tensor:
        matrix = (
            super()
            .as_matrix()
            .to(device=self._device, non_blocking=self._device != torch_device("cpu"))
        )
        return matrix

    def as_host_matrix(self) -> Tensor:
        return super().as_matrix()

    def detach(self) -> "HostAffineTransformation":
        return self

    @property
    def device(
        self,
    ) -> torch_device:
        return self._device

    def cast(
        self,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
        non_blocking: bool = False,
    ) -> "HostAffineTransformation":
        return HostAffineTransformation(
            transformation_matrix_on_host=self._transformation_matrix.to(
                dtype=self._transformation_matrix.dtype if dtype is None else dtype,
                non_blocking=non_blocking,
            ),
            device=self._device if device is None else device,
        )

    def __repr__(self) -> str:
        return (
            f"HostAffineTransformation("
            f"transformation_matrix_on_host={self._transformation_matrix}, "
            f"device={self._device})"
        )

    def __neg__(self) -> "HostAffineTransformation":
        return HostAffineTransformation(
            transformation_matrix_on_host=negate_affine_matrix(self.as_host_matrix()),
            device=self.device,
        )

    def invert(self) -> "HostAffineTransformation":
        return HostAffineTransformation(
            transformation_matrix_on_host=invert_matrix(self.as_host_matrix()),
            device=self.device,
        )

    def clear_translation(self) -> "HostAffineTransformation":
        return HostAffineTransformation(
            transformation_matrix_on_host=clear_translation_from_affine_matrix(
                self.as_host_matrix()
            ),
            device=self.device,
        )


class BaseDiagonalAffineTransformation(BaseAffineTransformation):
    """Base diagonal affine transformation"""

    @overload
    def __matmul__(
        self, affine_transformation: IHostAffineTransformation
    ) -> IHostAffineTransformation: ...

    @overload
    def __matmul__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation: ...

    def __matmul__(self, affine_transformation: IAffineTransformation) -> "IAffineTransformation":
        if not isinstance(affine_transformation, IAffineTransformation):
            return NotImplemented
        self_host_diagonal = self.as_host_diagonal()
        target_host_diagonal = affine_transformation.as_host_diagonal()
        if self_host_diagonal is not None and target_host_diagonal is not None:
            return HostDiagonalAffineTransformation.from_definition(
                compose_diagonal_affine_matrices(
                    self_host_diagonal,
                    target_host_diagonal,
                ),
                device=self.device,
            )

        self_diagonal = self.as_diagonal()
        target_diagonal = affine_transformation.as_diagonal()
        if self_diagonal is not None and target_diagonal is not None:
            return DiagonalAffineTransformation.from_definition(
                compose_diagonal_affine_matrices(
                    self_diagonal,
                    target_diagonal,
                )
            )
        return super().__matmul__(affine_transformation)

    @overload
    def __add__(
        self, affine_transformation: IHostAffineTransformation
    ) -> IHostAffineTransformation: ...

    @overload
    def __add__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation: ...

    def __add__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation:
        if not isinstance(affine_transformation, IAffineTransformation):
            return NotImplemented
        if isinstance(affine_transformation, BaseDiagonalAffineTransformation):
            self_host_diagonal = self.as_host_diagonal()
            target_host_diagonal = affine_transformation.as_host_diagonal()
            if self_host_diagonal is not None and target_host_diagonal is not None:
                return HostDiagonalAffineTransformation.from_definition(
                    add_diagonal_affine_matrices(self_host_diagonal, target_host_diagonal),
                    device=self.device,
                )
            self_diagonal = self.as_diagonal()
            target_diagonal = affine_transformation.as_diagonal()
            if self_diagonal is not None and target_diagonal is not None:
                return DiagonalAffineTransformation.from_definition(
                    add_diagonal_affine_matrices(self_diagonal, target_diagonal)
                )
        return super().__add__(affine_transformation)

    @abstractmethod
    def as_diagonal(self) -> DiagonalAffineMatrixDefinition:
        """Return the diagonal and translation tensors"""

    @abstractmethod
    def as_host_diagonal(self) -> Optional[DiagonalAffineMatrixDefinition]:
        """Return the diagonal and translation tensors detached on cpu, if available"""


class DiagonalAffineTransformation(BaseDiagonalAffineTransformation):
    """Affine transformation representable as a diagonal affine transformation matrix

    Arguments:
        diagonal: Tensor with shape ([*batch_shape, ]diagonal_length[, *spatial_shape]),
            if None, corresponds to all ones. Can also be given as a sequence of numbers
            or a single number.
        translation: Tensor with shape ([*batch_shape, ]n_output_dims[, *spatial_shape]),
            if None, corresponds to all zeros. Can also be given as a sequence of numbers
            or a single number.
        matrix_shape: Shape of the target affine transformation matrix
            ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape])
        dtype: Data type of the transformation matrix, needed only if no
            diagonal or translation is given.
        device: Device of the transformation matrix, needed only if no
            diagonal or translation is given.
    """

    def __init__(
        self,
        diagonal: Optional[Union[Tensor, Sequence[Number], Number]] = None,
        translation: Optional[Union[Tensor, Sequence[Number], Number]] = None,
        matrix_shape: Optional[Sequence[int]] = None,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> None:
        self._matrix_definition = DiagonalAffineMatrixDefinition(
            diagonal=diagonal,
            translation=translation,
            matrix_shape=matrix_shape,
            dtype=dtype,
            device=device,
        )

    @classmethod
    def from_definition(
        cls, matrix_definition: DiagonalAffineMatrixDefinition
    ) -> "DiagonalAffineTransformation":
        """Create diagonal affine transformation from definition"""
        instance = cls.__new__(cls)
        instance._matrix_definition = matrix_definition
        return instance

    def _get_children(self) -> Mapping[str, TensorLike]:
        return {"matrix_definition": self._matrix_definition}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, TensorLike]
    ) -> "DiagonalAffineTransformation":
        return self.from_definition(
            cast(DiagonalAffineMatrixDefinition, children["matrix_definition"])
        )

    @property
    def shape(self) -> Sequence[int]:
        return self._matrix_definition.shape

    def __call__(self, values: Tensor, n_channel_dims: int = 1) -> Tensor:
        return transform_values_with_diagonal_affine_matrix(
            self.as_diagonal(),
            values,
            n_channel_dims=n_channel_dims,
        )

    def as_matrix(
        self,
    ) -> Tensor:
        return self._matrix_definition.as_matrix()

    def as_host_matrix(self) -> Optional[Tensor]:
        return None

    def as_diagonal(
        self,
    ) -> DiagonalAffineMatrixDefinition:
        return self._matrix_definition

    def as_host_diagonal(self) -> Optional[DiagonalAffineMatrixDefinition]:
        return None

    def __neg__(self) -> "DiagonalAffineTransformation":
        return self.from_definition(negate_diagonal_affine_matrix(self._matrix_definition))

    def invert(self) -> "DiagonalAffineTransformation":
        return self.from_definition(invert_diagonal_affine_matrix(self._matrix_definition))

    def __repr__(self) -> str:
        return f"DiagonalAffineTransformation(definition={self._matrix_definition})"

    def is_zero(self) -> Optional[bool]:
        return None

    def clear_translation(self) -> IAffineTransformation:
        return self.from_definition(self._matrix_definition.clear_translation())


class HostDiagonalAffineTransformation(DiagonalAffineTransformation, IHostAffineTransformation):
    """Affine transformation representable as a diagonal affine transformation matrix
    with the matrix stored on host (cpu) and the matrix on target device created
    only when needed.

    Allows to do control flow decisions on host based on the transformation
    without having to do CPU-GPU synchronization.

    Arguments:
        diagonal: Tensor with shape ([*batch_shape, ]diagonal_length[, *spatial_shape]),
            if None, corresponds to all ones. Can also be given as a sequence of numbers
            or a single number.
        translation: Tensor with shape ([*batch_shape, ]n_output_dims[, *spatial_shape]),
            if None, corresponds to all zeros. Can also be given as a sequence of numbers
            or a single number.
        matrix_shape: Shape of the target affine transformation matrix
            ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape])
        dtype: Data type of the transformation matrix, needed only if no
            diagonal or translation is given.
        device: Device of the transformation matrix generated by as_matrix and as_diagonal
            methods.
    """

    def __init__(
        self,
        diagonal: Optional[Union[Tensor, Sequence[Number], Number]] = None,
        translation: Optional[Union[Tensor, Sequence[Number], Number]] = None,
        matrix_shape: Optional[Sequence[int]] = None,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> None:
        super().__init__(
            diagonal=diagonal,
            translation=translation,
            matrix_shape=matrix_shape,
            dtype=dtype,
            device=torch_device("cpu"),
        )
        self._target_device = get_default_device() if device is None else device

    @classmethod
    def from_definition(
        cls,
        matrix_definition: DiagonalAffineMatrixDefinition,
        device: Optional[torch_device] = None,
    ) -> "HostDiagonalAffineTransformation":
        """Create diagonal affine transformation from definition"""
        instance = cls.__new__(cls)
        instance._matrix_definition = matrix_definition
        instance._target_device = get_default_device() if device is None else device
        return instance

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, TensorLike]
    ) -> "HostDiagonalAffineTransformation":
        return HostDiagonalAffineTransformation.from_definition(
            cast(DiagonalAffineMatrixDefinition, children["matrix_definition"]), self.device
        )

    def __call__(self, values: Tensor, n_channel_dims: int = 1) -> Tensor:
        if is_identity_diagonal_affine_matrix(self._matrix_definition):
            batch_shape, channels_shape, spatial_shape = split_shape(
                self.get_output_shape(values.shape, n_channel_dims), n_channel_dims=n_channel_dims
            )
            return broadcast_to_in_parts(
                values,
                batch_shape=batch_shape,
                channels_shape=channels_shape,
                spatial_shape=spatial_shape,
                n_channel_dims=n_channel_dims,
            )
        if self.is_zero():
            return zeros(
                self.get_output_shape(values.shape, n_channel_dims),
                device=self._target_device,
                dtype=values.dtype,
            )
        return super().__call__(values, n_channel_dims)

    def as_matrix(
        self,
    ) -> Tensor:
        matrix = super().as_matrix().to(device=self.device, non_blocking=self.device.type != "cpu")
        return matrix

    def as_host_matrix(self) -> Tensor:
        return super().as_matrix()

    def as_diagonal(self) -> DiagonalAffineMatrixDefinition:
        matrix_definition = super().as_diagonal()
        if self._target_device.type != "cpu":
            matrix_definition = matrix_definition.cast(
                device=self._target_device, non_blocking=self.device.type != "cpu"
            )
        return matrix_definition

    def as_host_diagonal(self) -> DiagonalAffineMatrixDefinition:
        return super().as_diagonal()

    def detach(self) -> "HostDiagonalAffineTransformation":
        return self

    @property
    def device(
        self,
    ) -> torch_device:
        return self._target_device

    def cast(
        self,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
        non_blocking: bool = False,
    ) -> "HostDiagonalAffineTransformation":
        dtype = self.dtype if dtype is None else dtype
        device = self.device if device is None else device
        return HostDiagonalAffineTransformation.from_definition(
            self._matrix_definition.cast(dtype=dtype, non_blocking=non_blocking), device=device
        )

    def __repr__(self) -> str:
        return f"HostDiagonalAffineTransformation(definition={self._matrix_definition})"

    def __neg__(self) -> "HostDiagonalAffineTransformation":
        return HostDiagonalAffineTransformation.from_definition(
            negate_diagonal_affine_matrix(self._matrix_definition), device=self.device
        )

    def invert(self) -> "HostDiagonalAffineTransformation":
        return HostDiagonalAffineTransformation.from_definition(
            invert_diagonal_affine_matrix(self._matrix_definition), device=self.device
        )

    def is_identity(self) -> bool:
        host_diagonal_matrix = self.as_host_diagonal()
        return is_identity_diagonal_affine_matrix(host_diagonal_matrix)

    def is_zero(self) -> bool:
        host_diagonal_matrix = self.as_host_diagonal()
        return is_zero_diagonal_affine_matrix(host_diagonal_matrix)

    def clear_translation(self) -> "HostDiagonalAffineTransformation":
        return self.from_definition(self._matrix_definition.clear_translation(), device=self.device)


class IdentityAffineTransformation(HostDiagonalAffineTransformation):
    """Identity affine transformation

    Arguments:
        n_dims: Number of spatial dimensions
        dtype: Data type of the transformation matrix
        device: Device of the transformation matrix
    """

    def __init__(
        self,
        n_dims: int,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> None:
        super().__init__(
            diagonal=None,
            translation=None,
            matrix_shape=(n_dims + 1, n_dims + 1),
            dtype=dtype,
            device=device,
        )
        self._n_dims = n_dims

    def __repr__(self) -> str:
        return (
            f"IdentityAffineTransformation("
            f"n_dims={self._n_dims}, "
            f"dtype={self.dtype}, "
            f"device={self.device})"
        )
