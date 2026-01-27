"""Affine transformation acting on mappable tensors."""

from typing import Mapping, Optional, Sequence, cast

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype

from .affine_transformation import (
    AffineTransformation,
    DiagonalAffineTransformation,
    IAffineTransformation,
    IdentityAffineTransformation,
)
from .composable_mapping import ComposableMapping
from .mappable_tensor import MappableTensor
from .sampler.interface import DataFormat
from .tensor_like import TensorLike


class Affine(ComposableMapping):
    """Affine mapping.

    This class represents an affine transformation applicable to mappable
    tensors. In essence, it is a wrapper around an affine transformation object
    which acts on PyTorch tensors. Since it is a composable mapping, it can be
    composed with other composable mappings.

    Recommended way to create an affine transformation is to use the factory
    functions provided in this module or as class methods of this class:
    `affine`, `diagonal_affine`, `Affine.identity`,
    `Affine.from_matrix`, `Affine.from_diagonal_and_translation`.

    Args:
        transformation: The wrapped affine transformation acting on PyTorch tensors.
    """

    def __init__(self, transformation: IAffineTransformation) -> None:
        self.transformation: IAffineTransformation = transformation
        """The wrapped affine transformation acting on PyTorch tensors."""

    @classmethod
    def from_matrix(cls, matrix: Tensor) -> "Affine":
        """Create affine mapping from an affine transformation matrix.

        Args:
            matrix: Affine transformation matrix with shape
                ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape]).

        Returns:
            Affine mapping.

        """
        return cls(AffineTransformation(matrix))

    @classmethod
    def from_diagonal_and_translation(
        cls,
        diagonal: Optional[Tensor] = None,
        translation: Optional[Tensor] = None,
        matrix_shape: Optional[Sequence[int]] = None,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "Affine":
        """Create affine mapping from diagonal and translation

        Args:
            diagonal: Diagonal of the affine transformation matrix with shape
                ([*batch_shape, ]diagonal_length[, *spatial_shape]).
            translation: Translation of the affine transformation matrix
                with shape ([*batch_shape, ]n_output_dims[, *spatial_shape]).
            matrix_shape: Shape of the affine transformation matrix in format
                ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape]).
            dtype: Data type of the affine transformation matrix, needed only if diagonal
                and translation are not provided.
            device: Device of the affine transformation matrix, needed only if diagonal
                and translation are not provided.

        Returns:
            Affine mapping.
        """
        return cls(
            DiagonalAffineTransformation(
                diagonal=diagonal,
                translation=translation,
                matrix_shape=matrix_shape,
                dtype=dtype,
                device=device,
            )
        )

    @classmethod
    def identity(
        cls, n_dims: int, dtype: Optional[torch_dtype] = None, device: Optional[torch_device] = None
    ) -> "Affine":
        """Create identity affine mapping.

        Args:
            n_dims: Number of input and output dimensions of the identity affine mapping.
            dtype: Data type of the identity affine mapping.
            device: Device of the identity affine mapping.

        Returns:
            Identity affine mapping.
        """
        return cls(IdentityAffineTransformation(n_dims, dtype=dtype, device=device))

    @property
    def default_resampling_data_format(self) -> DataFormat:
        return DataFormat.voxel_displacements()

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        return masked_coordinates.transform(self.transformation)

    def _get_children(self) -> Mapping[str, TensorLike]:
        return {"transformation": self.transformation}

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, TensorLike]
    ) -> "Affine":
        return Affine(cast(IAffineTransformation, children["transformation"]))

    def invert(self, **arguments) -> ComposableMapping:
        return Affine(self.transformation.invert())

    def __repr__(self) -> str:
        return f"Affine(transformation={self.transformation})"


def affine(matrix: Tensor) -> Affine:
    """Create affine mapping from an affine transformation matrix.

    See: `Affine.from_matrix.`
    """
    return Affine.from_matrix(matrix)


def diagonal_affine(
    diagonal: Optional[Tensor] = None,
    translation: Optional[Tensor] = None,
    matrix_shape: Optional[Sequence[int]] = None,
    dtype: Optional[torch_dtype] = None,
    device: Optional[torch_device] = None,
) -> Affine:
    """Create affine mapping from diagonal and translation.

    See: `Affine.from_diagonal_and_translation`.
    """
    return Affine.from_diagonal_and_translation(
        diagonal=diagonal,
        translation=translation,
        matrix_shape=matrix_shape,
        dtype=dtype,
        device=device,
    )
