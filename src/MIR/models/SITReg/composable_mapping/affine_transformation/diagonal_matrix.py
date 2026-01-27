"""Core matrix operations for diagonal affine transformations."""

from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

from torch import Tensor, allclose, broadcast_shapes, cat
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import get_default_dtype, ones, tensor, zeros
import torch
from MIR.models.SITReg.composable_mapping.interface import Number
from MIR.models.SITReg.composable_mapping.tensor_like import TensorLike
from MIR.models.SITReg.composable_mapping.util import (
    broadcast_optional_shapes_in_parts_splitted,
    broadcast_shapes_in_parts_to_single_shape,
    broadcast_tensors_in_parts,
    broadcast_to_in_parts,
    get_channel_dims,
    get_channels_shape,
    split_shape,
)

from .matrix import (
    IDENTITY_MATRIX_TOLERANCE,
    ZERO_MATRIX_TOLERANCE,
    embed_matrix,
    generate_scale_matrix,
)

def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiagonalAffineMatrixDefinition(TensorLike):
    """Definition of a diagonal affine matrix

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
        self._dtype, self._device = self._handle_dtype_and_device(
            diagonal, translation, dtype, device
        )
        if isinstance(diagonal, (float, int)):
            diagonal = ones(1, dtype=self._dtype, device=self._device) * diagonal
        elif not isinstance(diagonal, Tensor) and diagonal is not None:
            diagonal = tensor(diagonal, dtype=self._dtype, device=torch_device("cpu"))
            if self._device.type != "cpu":
                diagonal = diagonal.to(device=self._device, non_blocking=True)
        if isinstance(translation, (float, int)):
            translation = ones(1, dtype=self._dtype, device=self._device) * translation
        elif not isinstance(translation, Tensor) and translation is not None:
            translation = tensor(translation, dtype=self._dtype, device=torch_device("cpu"))
            if self._device.type != "cpu":
                translation = translation.to(device=self._device, non_blocking=True)
        if diagonal is not None and diagonal.ndim == 0:
            diagonal = diagonal.unsqueeze(0)
        if translation is not None and translation.ndim == 0:
            translation = translation.unsqueeze(0)
        batch_shape, channels_shape, spatial_shape = self._infer_diagonal_matrix_shape_in_parts(
            diagonal, translation, matrix_shape
        )
        diagonal_length = min(channels_shape[0] - 1, channels_shape[1] - 1)
        if diagonal is not None:
            diagonal = broadcast_to_in_parts(
                diagonal,
                channels_shape=(diagonal_length,),
            )
        if translation is not None:
            translation = broadcast_to_in_parts(
                translation,
                channels_shape=(channels_shape[0] - 1,),
            )
        self._broadcasted_shape = batch_shape + channels_shape + spatial_shape
        self._matrix_shape = matrix_shape
        self._diagonal = diagonal
        self._translation = translation

    def _get_children(self) -> Mapping[str, TensorLike]:
        return {}

    def _get_tensors(self) -> Mapping[str, Tensor]:
        tensors: Dict[str, Tensor] = {}
        if self._diagonal is not None:
            tensors["diagonal"] = self._diagonal
        if self._translation is not None:
            tensors["translation"] = self._translation
        return tensors

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, TensorLike]
    ) -> "DiagonalAffineMatrixDefinition":
        diagonal = tensors.get("diagonal", self._diagonal)
        translation = tensors.get("translation", self._translation)
        return DiagonalAffineMatrixDefinition(
            diagonal,
            translation,
            matrix_shape=self._matrix_shape,
            dtype=self._dtype,
            device=self._device,
        )

    def cast(
        self,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
        non_blocking: bool = False,
    ) -> "DiagonalAffineMatrixDefinition":
        return DiagonalAffineMatrixDefinition(
            diagonal=(
                None
                if self._diagonal is None
                else self._diagonal.to(dtype=dtype, device=device, non_blocking=non_blocking)
            ),
            translation=(
                None
                if self._translation is None
                else self._translation.to(dtype=dtype, device=device, non_blocking=non_blocking)
            ),
            matrix_shape=self._matrix_shape,
            dtype=self._dtype if dtype is None else dtype,
            device=self._device if device is None else device,
        )

    def __repr__(self) -> str:
        return (
            f"DiagonalAffineMatrixDefinition(diagonal={self._diagonal}, "
            f"translation={self._translation}, matrix_shape={self._matrix_shape}, "
            f"dtype={self._dtype}, device={self._device})"
        )

    @property
    def dtype(self) -> torch_dtype:
        """PyTorch data type"""
        return self._dtype

    @property
    def device(self) -> torch_device:
        """PyTorch device"""
        return self._device

    @property
    def diagonal(self) -> Optional[Tensor]:
        """Diagonal of the matrix

        Batch and spatial dimensions are not broadcasted to the final matrix shape.
        """
        return self._diagonal

    @property
    def translation(self) -> Optional[Tensor]:
        """Translation of the matrix

        Batch and spatial dimensions are not broadcasted to the final matrix shape.
        """
        return self._translation

    @property
    def shape(self) -> Sequence[int]:
        """Matrix shape"""
        return self._broadcasted_shape

    def as_matrix(self) -> Tensor:
        """Generate the diagonal affine matrix"""
        batch_shape, channels_shape, spatial_shape = split_shape(self.shape, n_channel_dims=2)
        matrix = generate_scale_matrix(self.generate_diagonal())
        matrix = broadcast_to_in_parts(
            matrix,
            batch_shape=batch_shape,
            spatial_shape=spatial_shape,
            n_channel_dims=2,
        )
        if self._translation is not None:
            translation = broadcast_to_in_parts(
                self._translation,
                batch_shape=batch_shape,
                spatial_shape=spatial_shape,
            )
            matrix = embed_matrix(matrix, (channels_shape[0] - 1, channels_shape[1] - 1))
            matrix_channel_dims = get_channel_dims(matrix.ndim, n_channel_dims=2)
            matrix = cat(
                (matrix, translation.unsqueeze(matrix_channel_dims[1])),
                dim=matrix_channel_dims[1],
            )
        matrix = embed_matrix(matrix, channels_shape)
        return matrix

    def generate_diagonal(self) -> Tensor:
        """Generate the diagonal"""
        batch_shape, channels_shape, spatial_shape = split_shape(self.shape, n_channel_dims=2)
        if self._diagonal is None:
            diagonal_length = min(channels_shape[0] - 1, channels_shape[1] - 1)
            diagonal = ones(
                diagonal_length,
                dtype=self._dtype,
                device=self._device,
            )
        else:
            diagonal = self._diagonal
        return broadcast_to_in_parts(
            diagonal,
            batch_shape=batch_shape,
            spatial_shape=spatial_shape,
            n_channel_dims=1,
        )

    def generate_translation(self) -> Tensor:
        """Generate the translation"""
        batch_shape, channels_shape, spatial_shape = split_shape(self.shape, n_channel_dims=2)
        if self._translation is None:
            translation = ones(
                channels_shape[0] - 1,
                dtype=self._dtype,
                device=self._device,
            )
        else:
            translation = self._translation
        return broadcast_to_in_parts(
            translation,
            batch_shape=batch_shape,
            spatial_shape=spatial_shape,
            n_channel_dims=1,
        )

    @staticmethod
    def _infer_diagonal_matrix_shape_in_parts(
        diagonal: Optional[Tensor],
        translation: Optional[Tensor],
        matrix_shape: Optional[Sequence[int]],
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        """Get the shape of a diagonal matrix in parts"""
        if matrix_shape is None:
            if diagonal is not None and translation is not None:
                channels_shape: Tuple[int, ...] = (
                    get_channels_shape(translation.shape, n_channel_dims=1)[0] + 1,
                    get_channels_shape(diagonal.shape, n_channel_dims=1)[0] + 1,
                )
            elif diagonal is not None:
                channels_shape = (
                    get_channels_shape(diagonal.shape, n_channel_dims=1)[0] + 1,
                    get_channels_shape(diagonal.shape, n_channel_dims=1)[0] + 1,
                )
            elif translation is not None:
                channels_shape = (
                    get_channels_shape(translation.shape, n_channel_dims=1)[0] + 1,
                    get_channels_shape(translation.shape, n_channel_dims=1)[0] + 1,
                )
            else:
                raise ValueError(
                    "At least one of diagonal, translation, and matrix_shape must be given."
                )
        else:
            channels_shape = get_channels_shape(matrix_shape, n_channel_dims=2)
        if diagonal is not None:
            diagonal_length = min(channels_shape[0] - 1, channels_shape[1] - 1)
            diagonal = broadcast_to_in_parts(
                diagonal,
                channels_shape=(diagonal_length,),
            )
        if translation is not None:
            translation = broadcast_to_in_parts(
                translation,
                channels_shape=(channels_shape[0] - 1,),
            )
        batch_size, _, spatial_shape = broadcast_optional_shapes_in_parts_splitted(
            diagonal.shape if diagonal is not None else None,
            translation.shape if translation is not None else None,
            matrix_shape,
            n_channel_dims=(1, 1, 2),
            broadcast_channels=False,
        )
        assert batch_size is not None and spatial_shape is not None
        return batch_size, channels_shape, spatial_shape

    @staticmethod
    def _handle_dtype_and_device(
        diagonal: Optional[Union[Tensor, Sequence[Number], Number]],
        translation: Optional[Union[Tensor, Sequence[Number], Number]],
        dtype: Optional[torch_dtype],
        device: Optional[torch_device],
    ) -> Tuple[torch_dtype, torch_device]:
        if diagonal is not None and isinstance(diagonal, Tensor):
            if device is not None and device != diagonal.device:
                raise ValueError(
                    "Device mismatch, note that the device is needed only if no diagonal or "
                    "translation is given as tensor."
                )
            if dtype is not None and dtype != diagonal.dtype:
                raise ValueError(
                    "Dtype mismatch, note that the dtype is needed only if no diagonal or "
                    "translation is given as tensor."
                )
            dtype = diagonal.dtype
            device = diagonal.device
            if translation is not None and isinstance(translation, Tensor):
                if translation.device != device:
                    raise ValueError("Device mismatch between diagonal and translation.")
                if translation.dtype != dtype:
                    raise ValueError("Dtype mismatch between diagonal and translation.")
        elif translation is not None and isinstance(translation, Tensor):
            if device is not None and device != translation.device:
                raise ValueError(
                    "Device mismatch, note that the device is needed only if no diagonal or "
                    "translation is given."
                )
            if dtype is not None and dtype != translation.dtype:
                raise ValueError(
                    "Dtype mismatch, note that the dtype is needed only if no diagonal or "
                    "translation is given."
                )
            dtype = translation.dtype
            device = translation.device
        dtype = get_default_dtype() if dtype is None else dtype
        device = get_default_device() if device is None else device
        return dtype, device

    def clear_translation(self) -> "DiagonalAffineMatrixDefinition":
        """Clear the translation"""
        return DiagonalAffineMatrixDefinition(
            diagonal=self._diagonal,
            translation=None,
            matrix_shape=self._matrix_shape,
            dtype=self._dtype,
            device=self._device,
        )


def transform_values_with_diagonal_affine_matrix(
    matrix_definition: DiagonalAffineMatrixDefinition,
    values: Tensor,
    n_channel_dims: int = 1,
) -> Tensor:
    """Transform values with a diagonal affine matrix

    Args:
        matrix_definition: Definition of the diagonal affine matrix.
        values: Values to transform.
        n_channel_dims: Number of channel dimensions in the values tensor.

    Returns:
        Values transformed with the diagonal affine matrix.
    """
    affine_channels_shape = get_channels_shape(matrix_definition.shape, n_channel_dims=2)
    n_input_dims, n_output_dims = (affine_channels_shape[1] - 1, affine_channels_shape[0] - 1)
    if get_channels_shape(values.shape, n_channel_dims=n_channel_dims)[-1] != n_input_dims:
        raise ValueError("The diagonal matrix does not match the number of dimensions.")
    diagonal_length = min(affine_channels_shape[0] - 1, affine_channels_shape[1] - 1)
    values_channel_dim = get_channel_dims(values.ndim, n_channel_dims=n_channel_dims)[-1]
    if diagonal_length < n_input_dims:
        values = values.narrow(values_channel_dim, 0, diagonal_length)
    if matrix_definition.diagonal is not None:
        diagonal, values = broadcast_tensors_in_parts(
            matrix_definition.generate_diagonal(), values, n_channel_dims=(1, n_channel_dims)
        )
        values = values * diagonal

    if diagonal_length < n_output_dims:
        values = _pad_dimension_with_zeros(values, n_output_dims, values_channel_dim)
    if matrix_definition.translation is not None:
        translation, values = broadcast_tensors_in_parts(
            matrix_definition.generate_translation(), values, n_channel_dims=(1, n_channel_dims)
        )
        values = values + translation
    return values


def invert_diagonal_affine_matrix(
    matrix_definition: DiagonalAffineMatrixDefinition,
) -> DiagonalAffineMatrixDefinition:
    """Invert a diagonal affine matrix

    Args:
        matrix_definition: Definition of the diagonal affine matrix.

    Returns:
        Definition of the inverted diagonal affine matrix.
    """
    affine_channels_shape = get_channels_shape(matrix_definition.shape, n_channel_dims=2)
    if affine_channels_shape[0] != affine_channels_shape[1]:
        raise ValueError("The diagonal matrix must be square.")
    if matrix_definition.diagonal is None:
        diagonal: Optional[Tensor] = None
    else:
        diagonal = 1 / matrix_definition.diagonal
    if matrix_definition.translation is None:
        translation: Optional[Tensor] = None
    else:
        translation = -matrix_definition.translation
        if diagonal is not None:
            diagonal, translation = broadcast_tensors_in_parts(
                diagonal, translation, n_channel_dims=1
            )
            translation = translation * diagonal
    return DiagonalAffineMatrixDefinition(
        diagonal,
        translation,
        matrix_definition.shape,
        dtype=matrix_definition.dtype,
        device=matrix_definition.device,
    )


def negate_diagonal_affine_matrix(
    matrix_definition: DiagonalAffineMatrixDefinition,
) -> DiagonalAffineMatrixDefinition:
    """Negate diagonal affine matrix.

    Args:
        matrix_definition: Definition of the diagonal affine matrix.

    Returns:
        Definition of the negated diagonal affine matrix.
    """
    affine_channels_shape = get_channels_shape(matrix_definition.shape, n_channel_dims=2)
    if matrix_definition.translation is not None:
        translation: Optional[Tensor] = -matrix_definition.translation
    else:
        translation = None
    if matrix_definition.diagonal is not None:
        diagonal = -matrix_definition.diagonal
    else:
        diagonal_length = min(affine_channels_shape[0] - 1, affine_channels_shape[1] - 1)
        diagonal = -ones(
            diagonal_length, dtype=matrix_definition.dtype, device=matrix_definition.device
        )
    return DiagonalAffineMatrixDefinition(
        diagonal,
        translation,
        matrix_shape=matrix_definition.shape,
        dtype=matrix_definition.dtype,
        device=matrix_definition.device,
    )


def is_identity_diagonal_affine_matrix(
    matrix_definition: DiagonalAffineMatrixDefinition,
) -> bool:
    """Is the diagonal matrix an identity matrix.

    Args:
        matrix_definition: Definition of the diagonal affine matrix.

    Returns:
        Whether the diagonal matrix is an identity matrix.
    """
    if matrix_definition.diagonal is not None:
        if not allclose(
            matrix_definition.diagonal,
            ones(1, dtype=matrix_definition.dtype, device=matrix_definition.device),
            atol=IDENTITY_MATRIX_TOLERANCE,
        ):
            return False
    if matrix_definition.translation is not None:
        if not allclose(
            matrix_definition.translation,
            zeros(1, dtype=matrix_definition.dtype, device=matrix_definition.device),
            atol=IDENTITY_MATRIX_TOLERANCE,
        ):
            return False
    return True


def is_zero_diagonal_affine_matrix(
    matrix_definition: DiagonalAffineMatrixDefinition,
) -> bool:
    """Is the diagonal matrix a zero matrix.

    Args:
        matrix_definition: Definition of the diagonal affine matrix.

    Returns:
        Whether the diagonal matrix is a zero matrix.
    """
    if matrix_definition.diagonal is None:
        return False
    else:
        if not allclose(
            matrix_definition.diagonal,
            zeros(
                1, dtype=matrix_definition.diagonal.dtype, device=matrix_definition.diagonal.device
            ),
            atol=ZERO_MATRIX_TOLERANCE,
        ):
            return False
    if matrix_definition.translation is not None:
        if not allclose(
            matrix_definition.translation,
            zeros(
                1,
                dtype=matrix_definition.translation.dtype,
                device=matrix_definition.translation.device,
            ),
            atol=ZERO_MATRIX_TOLERANCE,
        ):
            return False
    return True


def compose_diagonal_affine_matrices(
    matrix_definition_1: DiagonalAffineMatrixDefinition,
    matrix_definition_2: DiagonalAffineMatrixDefinition,
) -> DiagonalAffineMatrixDefinition:
    """Compose two diagonal affine matrices.

    Args:
        matrix_definition_1: Definition of the first diagonal affine matrix.
        matrix_definition_2: Definition of the second diagonal affine matrix.

    Returns:
        Definition of the composed diagonal affine matrix.
    """
    batch_shape_1, channels_shape_1, spatial_shape_1 = split_shape(
        matrix_definition_1.shape, n_channel_dims=2
    )
    batch_shape_2, channels_shape_2, spatial_shape_2 = split_shape(
        matrix_definition_2.shape, n_channel_dims=2
    )
    diagonal_length_1 = min(channels_shape_1[0] - 1, channels_shape_1[1] - 1)
    diagonal_length_2 = min(channels_shape_2[0] - 1, channels_shape_2[1] - 1)
    shared_diagonal_length = min(diagonal_length_1, diagonal_length_2)
    batch_shape = broadcast_shapes(batch_shape_1, batch_shape_2)
    spatial_shape = broadcast_shapes(spatial_shape_1, spatial_shape_2)
    if channels_shape_1[1] != channels_shape_2[0]:
        raise ValueError("The matrices are not compatible.")
    channels_shape = (channels_shape_1[0], channels_shape_2[1])
    diagonal_length = min(channels_shape[0] - 1, channels_shape[1] - 1)

    diagonal_1 = matrix_definition_1.diagonal
    diagonal_2 = matrix_definition_2.diagonal
    if diagonal_1 is not None and shared_diagonal_length < diagonal_length_1:
        channel_dim = get_channel_dims(diagonal_1.ndim, n_channel_dims=1)[0]
        diagonal_1 = diagonal_1.narrow(channel_dim, 0, diagonal_length)
    if diagonal_2 is not None and shared_diagonal_length < diagonal_length_2:
        channel_dim = get_channel_dims(diagonal_2.ndim, n_channel_dims=1)[0]
        diagonal_2 = diagonal_2.narrow(channel_dim, 0, diagonal_length)

    if diagonal_1 is not None and diagonal_2 is not None:
        diagonal_1, diagonal_2 = broadcast_tensors_in_parts(
            diagonal_1, diagonal_2, n_channel_dims=1
        )
        diagonal: Optional[Tensor] = diagonal_1 * diagonal_2
    elif diagonal_1 is not None:
        diagonal = diagonal_1
    elif diagonal_2 is not None:
        diagonal = diagonal_2
    else:
        diagonal = None
    if diagonal is not None:
        channel_dim = get_channel_dims(diagonal.ndim, n_channel_dims=1)[0]
        diagonal = _pad_dimension_with_zeros(diagonal, diagonal_length, channel_dim)

    if matrix_definition_2.translation is not None:
        channel_dim = get_channel_dims(matrix_definition_2.translation.ndim, n_channel_dims=1)[0]
        translation_2 = matrix_definition_2.translation.narrow(channel_dim, 0, diagonal_length_1)
        if diagonal_1 is not None:
            diagonal_1, translation_2 = broadcast_tensors_in_parts(
                diagonal_1, translation_2, n_channel_dims=1
            )
            translation_2 = translation_2 * diagonal_1
        channel_dim = get_channel_dims(translation_2.ndim, n_channel_dims=1)[0]
        translation_2 = _pad_dimension_with_zeros(translation_2, channels_shape[0] - 1, channel_dim)
    else:
        translation_2 = None

    if matrix_definition_1.translation is not None and translation_2 is not None:
        translation_1, translation_2 = broadcast_tensors_in_parts(
            matrix_definition_1.translation, translation_2, n_channel_dims=1
        )
        translation: Optional[Tensor] = translation_2 + translation_1
    elif matrix_definition_1.translation is not None:
        translation = matrix_definition_1.translation
    elif translation_2 is not None:
        translation = translation_2
    else:
        translation = None

    return DiagonalAffineMatrixDefinition(
        diagonal,
        translation,
        matrix_shape=batch_shape + channels_shape + spatial_shape,
        dtype=matrix_definition_1.dtype,
        device=matrix_definition_1.device,
    )


def add_diagonal_affine_matrices(
    matrix_definition_1: DiagonalAffineMatrixDefinition,
    matrix_definition_2: DiagonalAffineMatrixDefinition,
) -> DiagonalAffineMatrixDefinition:
    """Add two diagonal affine matrices.

    Args:
        matrix_definition_1: Definition of the first diagonal affine matrix.
        matrix_definition_2: Definition of the second diagonal affine matrix.

    Returns:
        Definition of the sum of the two diagonal affine matrices.
    """
    matrix_shape = broadcast_shapes_in_parts_to_single_shape(
        matrix_definition_1.shape, matrix_definition_2.shape, n_channel_dims=2
    )
    if get_channels_shape(matrix_definition_1.shape, n_channel_dims=2) != get_channels_shape(
        matrix_definition_2.shape, n_channel_dims=2
    ):
        raise ValueError("The matrices are not compatible.")

    if matrix_definition_1.diagonal is None and matrix_definition_2.diagonal is None:
        diagonal: Optional[Tensor] = None
    else:
        diagonal_1, diagonal_2 = broadcast_tensors_in_parts(
            matrix_definition_1.generate_diagonal(),
            matrix_definition_2.generate_diagonal(),
            n_channel_dims=1,
        )
        diagonal = diagonal_1 + diagonal_2

    if matrix_definition_1.translation is not None and matrix_definition_2.translation is not None:
        translation_1, translation_2 = broadcast_tensors_in_parts(
            matrix_definition_1.translation, matrix_definition_2.translation, n_channel_dims=1
        )
        translation: Optional[Tensor] = translation_2 + translation_1
    elif matrix_definition_1.translation is not None:
        translation = matrix_definition_1.translation
    elif matrix_definition_2.translation is not None:
        translation = matrix_definition_2.translation
    else:
        translation = None

    return DiagonalAffineMatrixDefinition(
        diagonal,
        translation,
        matrix_shape=matrix_shape,
        dtype=matrix_definition_1.dtype,
        device=matrix_definition_1.device,
    )


def _pad_dimension_with_zeros(item: Tensor, new_size: int, dim: int) -> Tensor:
    current_size = item.size(dim)
    return cat(
        (
            item,
            zeros(1, device=item.device, dtype=item.dtype).expand(
                *item.shape[:dim],
                new_size - current_size,
                *item.shape[dim + 1 :],
            ),
        ),
        dim=dim,
    )
