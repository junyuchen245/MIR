"""Defines a voxel grid with given shape transformed by given affine
transformation."""

from typing import Mapping, Optional, Sequence, Tuple, cast

from torch import Tensor, arange, broadcast_shapes, cat
from torch import device as torch_device
from torch import diag
from torch import dtype as torch_dtype
from torch import meshgrid, stack, tensor, zeros

from MIR.models.SITReg.composable_mapping.affine_transformation import (
    HostAffineTransformation,
    HostDiagonalAffineTransformation,
    IAffineTransformation,
)
from MIR.models.SITReg.composable_mapping.affine_transformation.matrix import (
    convert_to_homogenous_coordinates,
)
from MIR.models.SITReg.composable_mapping.tensor_like import TensorLike
from MIR.models.SITReg.composable_mapping.util import (
    are_broadcastable,
    get_batch_shape,
    get_channels_shape,
    has_spatial_dims,
    is_broadcastable_to,
)


class GridDefinition(TensorLike):
    """Defines a voxel grid transformed by an affine transformation.

    Arguments:
        spatial_shape: Spatial shape of the voxel grid.
        affine_transformation: Affine transformation to apply to the grid.
            The transformation should not differ over spatial dimensions.
    """

    def __init__(
        self,
        spatial_shape: Sequence[int],
        affine_transformation: IAffineTransformation,
    ) -> None:
        self._spatial_shape = spatial_shape
        if has_spatial_dims(affine_transformation.shape, n_channel_dims=2):
            raise ValueError(
                "Affine transformation for grid may not differ over spatial dimensions"
            )
        self._affine_transformation = affine_transformation

    def _get_children(self) -> Mapping[str, TensorLike]:
        return {"affine_transformation": self._affine_transformation}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, TensorLike]
    ) -> "GridDefinition":
        return GridDefinition(
            spatial_shape=self._spatial_shape,
            affine_transformation=cast(IAffineTransformation, children["affine_transformation"]),
        )

    def generate(self) -> Tensor:
        """Generate the grid as a tensor"""
        if self._affine_transformation.is_zero():
            return zeros(1, dtype=self.dtype, device=self.device).expand(self.shape)
        voxel_grid = self._generate_voxel_coordinate_grid(
            spatial_shape=self._spatial_shape, device=self.device, dtype=self.dtype
        )
        return self._affine_transformation(voxel_grid, n_channel_dims=1)

    @property
    def affine_transformation(self) -> IAffineTransformation:
        """Affine transformation applied to the voxel grid"""
        return self._affine_transformation

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        """Shape of the spatial dimensions"""
        return tuple(self._spatial_shape)

    @property
    def channels_shape(self) -> Tuple[int, ...]:
        """Shape of the channel dimensions"""
        return get_channels_shape(self.shape)

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """Shape of the batch dimensions"""
        return get_batch_shape(self.shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the grid"""
        return self._affine_transformation.get_output_shape(
            (1, len(self._spatial_shape)) + tuple(self._spatial_shape)
        )

    def is_zero(self) -> Optional[bool]:
        """Is the grid all zeros

        Returns:
            None if the check cannot be done on CPU, otherwise bool indicating
            whether the grid is all zeros.
        """
        return self._affine_transformation.is_zero()

    def is_transformable(self, affine_transformation: IAffineTransformation) -> bool:
        """Is the grid transformable by the given affine transformation"""
        if has_spatial_dims(affine_transformation.shape):
            return False
        if not are_broadcastable(
            affine_transformation.batch_shape, self._affine_transformation.batch_shape
        ):
            return False
        return (
            affine_transformation.channels_shape[1] == self._affine_transformation.channels_shape[0]
        )

    def broadcast_to_n_channels(self, n_channels: int) -> "GridDefinition":
        """Broadcast the grid to have the given number of channels.

        Args:
            n_channels: Number of channels to broadcast to.

        Returns:
            GridDefinition with the given number of channels.
        """
        return GridDefinition(
            spatial_shape=self.spatial_shape,
            affine_transformation=self.affine_transformation.broadcast_to_n_output_channels(
                n_channels
            ),
        )

    def broadcast_to_spatial_shape(self, spatial_shape: Sequence[int]) -> "GridDefinition":
        """Broadcast the grid to have the given spatial shape.

        Args:
            spatial_shape: Spatial shape to broadcast to.

        Returns:
            GridDefinition with the given spatial shape.
        """
        if not is_broadcastable_to(self.spatial_shape, spatial_shape):
            raise RuntimeError("Can not broadcast to given spatial shape")
        target_spatial_shape = broadcast_shapes(self.spatial_shape, spatial_shape)
        if target_spatial_shape == self.spatial_shape:
            return self
        assert len(target_spatial_shape) >= len(self.spatial_shape)
        embedding_diagonal = (tensor(self.spatial_shape, device=torch_device("cpu")) != 1).to(
            self.dtype
        )
        if len(target_spatial_shape) > len(self.spatial_shape):
            embedding_transformation: IAffineTransformation = HostAffineTransformation(
                cat(
                    (
                        zeros(
                            (
                                len(self.spatial_shape) + 1,
                                len(spatial_shape) - len(self.spatial_shape),
                            ),
                            dtype=self.dtype,
                            device=torch_device("cpu"),
                        ),
                        diag(convert_to_homogenous_coordinates(embedding_diagonal)),
                    ),
                    dim=1,
                ),
                device=self.device,
            )
        else:
            embedding_transformation = HostDiagonalAffineTransformation(
                embedding_diagonal, device=self.device
            )
        return GridDefinition(
            spatial_shape=spatial_shape,
            affine_transformation=self.affine_transformation @ embedding_transformation,
        )

    def __add__(self, other: "GridDefinition") -> "GridDefinition":
        broadcasted_n_channel_dims = broadcast_shapes(self.channels_shape, other.channels_shape)[-1]
        broadcasted_spatial_shapes = broadcast_shapes(self.spatial_shape, other.spatial_shape)
        self_broadcasted = self.broadcast_to_n_channels(
            broadcasted_n_channel_dims
        ).broadcast_to_spatial_shape(broadcasted_spatial_shapes)
        other_broadcasted = other.broadcast_to_n_channels(
            broadcasted_n_channel_dims
        ).broadcast_to_spatial_shape(broadcasted_spatial_shapes)
        return GridDefinition(
            spatial_shape=broadcasted_spatial_shapes,
            affine_transformation=self_broadcasted.affine_transformation
            + other_broadcasted.affine_transformation,
        )

    def transform(self, affine_transformation: IAffineTransformation) -> "GridDefinition":
        """Transform the grid by the given affine transformation.

        Args:
            affine_transformation: Affine transformation to apply to the grid.

        Returns:
            GridDefinition with the given affine transformation applied.
        """
        return GridDefinition(
            spatial_shape=self.spatial_shape,
            affine_transformation=affine_transformation @ self.affine_transformation,
        )

    def __neg__(self) -> "GridDefinition":
        return GridDefinition(
            spatial_shape=self.spatial_shape,
            affine_transformation=-self.affine_transformation,
        )

    def __sub__(self, other: "GridDefinition") -> "GridDefinition":
        return self + (-other)

    def __repr__(self) -> str:
        return (
            f"GridDefinition(spatial_shape={self.spatial_shape}, "
            f"affine_transformation={self.affine_transformation})"
        )

    @staticmethod
    def _generate_voxel_coordinate_grid(
        spatial_shape: Sequence[int],
        device: Optional[torch_device] = None,
        dtype: Optional[torch_dtype] = None,
    ) -> Tensor:
        axes = [
            arange(start=0, end=dim_size, device=device, dtype=dtype) for dim_size in spatial_shape
        ]
        coordinates = stack(meshgrid(axes, indexing="ij"), dim=0)
        return coordinates[None]
