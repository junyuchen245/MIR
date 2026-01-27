"""Coordinate system defining locations on voxel grid in world coordinates."""

from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union, cast, overload

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import empty, tensor
from torch.nn import Module

from MIR.models.SITReg.composable_mapping.affine_mapping import Affine
from MIR.models.SITReg.composable_mapping.affine_transformation import (
    HostAffineTransformation,
    HostDiagonalAffineTransformation,
    IHostAffineTransformation,
)
from MIR.models.SITReg.composable_mapping.composable_mapping import ICoordinateSystemContainer
from MIR.models.SITReg.composable_mapping.interface import Number
from MIR.models.SITReg.composable_mapping.mappable_tensor import MappableTensor, voxel_grid
from MIR.models.SITReg.composable_mapping.tensor_like import TensorLike
from MIR.models.SITReg.composable_mapping.util import (
    broadcast_shapes_in_parts_splitted,
    broadcast_tensors_in_parts,
    broadcast_to_in_parts,
    get_channel_dims,
    get_channels_shape,
    get_spatial_shape,
    move_channels_last,
)

from .reformatting_reference import Center, ReformattingReference
from .reformatting_spatial_shape import (
    OriginalFOV,
    OriginalShape,
    ReformattingSpatialShape,
)


class CoordinateSystem(Module, ICoordinateSystemContainer, TensorLike):
    """Represents coordinate system between voxel and world coordinates on a regular grid

    Recommended way to create a coordinate system is to use the factory
    methods provided as class method of this class:
    `CoordinateSystem.centered_normalized`,
    `CoordinateSystem.centered`, `CoordinateSystem.voxel`,
    `CoordinateSystem.torch_grid_sample`,
    `CoordinateSystem.from_affine_matrix`,
    `CoordinateSystem.from_diagonal_affine_matrix`.

    Arguments:
        spatial_shape: Spatial shape of the grid
        to_voxel_coordinates: Affine transformation from world to voxel
            coordinates. This should be the inverse of from_voxel_coordinates
            and can be omitted if from_voxel_coordinates is given. This should
            be a IHostAffineTransformation meaning that the transformation is
            stored on the host (cpu) until needed on a target device. That is,
            since the transformation may be used in control flow decisions, and
            we want to avoid synchronizing the target device with the host.
        from_voxel_coordinates: Affine transformation from voxel to world
            coordinates. This should be the inverse of to_voxel_coordinates
            and can be omitted if to_voxel_coordinates is given. This should
            be a IHostAffineTransformation meaning that the transformation is
            stored on the host (cpu) until needed on a target device. That is,
            since the transformation may be used in control flow decisions, and
            we want to avoid synchronizing the target device with the host.
    """

    def __init__(
        self,
        spatial_shape: Sequence[int],
        to_voxel_coordinates: Optional[IHostAffineTransformation] = None,
        from_voxel_coordinates: Optional[IHostAffineTransformation] = None,
    ) -> None:
        super().__init__()
        self._spatial_shape = tuple(spatial_shape)
        if from_voxel_coordinates is None:
            if to_voxel_coordinates is None:
                raise ValueError(
                    "Either from_voxel_coordinates or to_voxel_coordinates should be given"
                )
            from_voxel_coordinates = to_voxel_coordinates.invert()
        elif to_voxel_coordinates is None:
            to_voxel_coordinates = from_voxel_coordinates.invert()
        self._coordinate_transformations_which_should_not_be_accessed_directly = {
            "from_voxel_coordinates": from_voxel_coordinates,
            "to_voxel_coordinates": to_voxel_coordinates,
        }
        if (
            get_channels_shape(from_voxel_coordinates.shape, n_channel_dims=2)[1]
            != len(spatial_shape) + 1
            or get_spatial_shape(from_voxel_coordinates.shape, n_channel_dims=2) != tuple()
        ):
            raise ValueError("Invalid affine transformation for a coordinate system")
        if (
            get_channels_shape(to_voxel_coordinates.shape, n_channel_dims=2)[1]
            != len(spatial_shape) + 1
            or get_spatial_shape(to_voxel_coordinates.shape, n_channel_dims=2) != tuple()
        ):
            raise ValueError("Invalid affine transformation for a coordinate system")
        self._voxel_grid_which_should_not_be_accessed_directly = voxel_grid(
            spatial_shape=self._spatial_shape,
            dtype=from_voxel_coordinates.dtype,
            device=from_voxel_coordinates.device,
        )
        self._grid_which_should_not_be_accessed_directly = (
            self._voxel_grid_which_should_not_be_accessed_directly.transform(from_voxel_coordinates)
        )
        # Trick to make torch.nn.Module type conversion work automatically, we
        # use the empty indicator tensor to infer the device and dtype of the
        # coordinate system.
        self.register_buffer(
            "_indicator",
            empty(0, device=to_voxel_coordinates.device, dtype=to_voxel_coordinates.dtype),
        )

    @classmethod
    def centered_normalized(
        cls,
        spatial_shape: Sequence[int],
        voxel_size: Union[Sequence[Number], Number, Tensor] = 1.0,
        fov_convention: str = "full_voxels",
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "CoordinateSystem":
        """Create a centered normalized coordinate system.

        The coordinate system is normalized such that the grid just fits into the
        cube from -1 to 1 in each dimension. The grid is centered in the cube.

        Args:
            spatial_shape: Spatial shape of the grid.
            voxel_size: Relative voxel size of the grid.
            fov_convention: Convention for defining the field of view, either "full_voxels"
                or "voxel_centers". If voxels are seens as cubes with the value at the
                center, the convention "full voxels" includes the full cubes in the field
                of view, while the convention "voxel_centers" includes only the centers.
                The latter results in a field of view that is one voxel smaller in each
                dimension. Similar to the align_corners option in
                torch.nn.functional.grid_sample.
            dtype: Data type of the created coordinate system.
            device: Device of the created coordinate system.

        Returns:
            Centered normalized coordinate system.
        """
        centered = cls.centered(spatial_shape, voxel_size, dtype=dtype, device=device)
        voxel_size = centered.grid_spacing_cpu()
        shape_tensor = voxel_size.new_tensor(spatial_shape)
        if fov_convention == "voxel_centers":
            shape_tensor -= 1
        elif fov_convention != "full_voxels":
            raise ValueError(f"Unknown fov convention {fov_convention}")
        fov_sizes = shape_tensor * voxel_size
        max_fov_size_index = fov_sizes.argmax()
        relative_voxel_sizes = voxel_size / voxel_size[max_fov_size_index]
        target_voxel_size = 2 / shape_tensor[max_fov_size_index] * relative_voxel_sizes
        return centered.reformat(
            voxel_size=target_voxel_size,
            reference=Center(),
            spatial_shape=OriginalShape(),
        )

    @classmethod
    def centered(
        cls,
        spatial_shape: Sequence[int],
        voxel_size: Union[Sequence[Number], Number, Tensor] = 1.0,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "CoordinateSystem":
        """Create centered coordinate system with a given voxel size.

        Args:
            spatial_shape: Spatial shape of the grid.
            voxel_size: Voxel size of the grid.
            dtype: Data type of the created coordinate system.
            device: Device of the created coordinate system.

        Returns:
            Centered coordinate system.
        """
        return cls.voxel(
            spatial_shape, voxel_size=voxel_size, dtype=dtype, device=device
        ).translate_voxel([-(dim_size - 1) / 2 for dim_size in spatial_shape])

    @classmethod
    def voxel(
        cls,
        spatial_shape: Sequence[int],
        voxel_size: Optional[Union[Tensor, Sequence[Number], Number]] = None,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "CoordinateSystem":
        """Create coordinate system corresponding to the voxel coordinate
        potentially scaled by the voxel size.

        Args:
            spatial_shape: Spatial shape of the grid.
            voxel_size: Voxel size of the grid.
            dtype: Data type of the created coordinate system.
            device: Device of the created coordinate system.

        Returns:
            Voxel coordinate system.
        """
        return cls.from_diagonal_affine_matrix(
            spatial_shape=spatial_shape,
            diagonal=voxel_size,
            dtype=dtype,
            device=device,
        )

    @classmethod
    def torch_grid_sample(
        cls,
        spatial_shape: Sequence[int],
        fov_convention: str = "full_voxels",
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "CoordinateSystem":
        """Create coordinate system corresponding to the one used by the
        torch.nn.functional.grid_sample.

        The coordinate system is normalized along each dimensions to the range
        from -1 to 1.

        Args:
            spatial_shape: Spatial shape of the grid.
            fov_convention: Convention for defining the field of view, either "full_voxels"
                or "voxel_centers". If voxels are seens as cubes with the value at the
                center, the convention "full voxels" includes the full cubes in the field
                of view, while the convention "voxel_centers" includes only the centers.
                The latter results in a field of view that is one voxel smaller in each
                dimension. Similar to the align_corners option in
                torch.nn.functional.grid_sample.
            dtype: Data type of the created coordinate system.
            device: Device of the created coordinate system.

        Returns:
            Coordinate system corresponding to the one used by the
            torch.nn.functional.grid_sample.
        """
        centered = cls.centered(spatial_shape, dtype=dtype, device=device)
        shape_tensor = tensor(spatial_shape, dtype=centered.dtype, device=torch_device("cpu"))
        if fov_convention == "voxel_centers":
            shape_tensor -= 1
        elif fov_convention != "full_voxels":
            raise ValueError(f"Unknown fov convention {fov_convention}.")
        fov_sizes = shape_tensor
        target_voxel_size = 2 / fov_sizes
        return centered.reformat(
            voxel_size=target_voxel_size,
            reference=Center(),
            spatial_shape=OriginalShape(),
        )

    @classmethod
    def from_affine_matrix(
        cls,
        spatial_shape: Sequence[int],
        affine_matrix: Tensor,
        device: Optional[torch_device] = None,
    ) -> "CoordinateSystem":
        """Create coordinate system from an affine matrix

        Args:
            spatial_shape: Spatial shape of the grid
            affine_matrix: Affine matrix describing the transformation
                from voxel to world coordinates. Tensor with shape
                ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1).
                The affine matrix tensor should be provided on the host (cpu).
                That is, since the transformation may be used in control flow
                decisions, and we want to avoid synchronizing the target device
                with the host. The matrix is moved to the target device
                asynchronously, if needed.
            device: Device of the created coordinate system.

        Returns:
            Coordinate system with the given affine matrix.
        """
        cls._ensure_tensor_on_host(affine_matrix)
        return CoordinateSystem(
            from_voxel_coordinates=HostAffineTransformation(
                affine_matrix,
                device=device,
            ),
            spatial_shape=spatial_shape,
        )

    @classmethod
    def from_diagonal_affine_matrix(
        cls,
        spatial_shape: Sequence[int],
        diagonal: Optional[Union[Tensor, Sequence[Number], Number]] = None,
        translation: Optional[Union[Tensor, Sequence[Number], Number]] = None,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "CoordinateSystem":
        """Create coordinate system from diagonal affine matrix

        The diagonal and the translation should be provided on the host (cpu).
        That is, since the transformation may be used in control flow decisions,
        and we want to avoid synchronizing the target device with the host. The
        matrix is moved to the target device asynchronously, if needed.

        Args:
            spatial_shape: Spatial shape of the created grid
            diagonal: Diagonal of the affine transformation matrix describing
                the transformation from voxel to world coordinates. Tensor
                with shape ([*batch_shape, ]diagonal_length[, *spatial_shape]).
            translation: Translation of the affine transformation describing
                the transformation from voxel to world coordinates. Tensor with
                shape ([*batch_shape, ]n_output_dims[, *spatial_shape]).
            dtype: Data type of the created coordinate system.
            device: Device of the created coordinate system.

        Returns:
            Coordinate system with the given diagonal affine matrix.
        """
        cls._ensure_tensor_on_host(diagonal)
        cls._ensure_tensor_on_host(translation)
        n_dims = len(spatial_shape)
        return CoordinateSystem(
            from_voxel_coordinates=HostDiagonalAffineTransformation(
                diagonal=diagonal,
                translation=translation,
                matrix_shape=(n_dims + 1, n_dims + 1),
                dtype=dtype,
                device=device,
            ),
            spatial_shape=spatial_shape,
        )

    @property
    def coordinate_system(self) -> "CoordinateSystem":
        return self

    def forward(self) -> None:
        """Dummy forward pass to make the coordinate system a torch.nn.Module"""

    @property
    def dtype(self) -> torch_dtype:
        return self.get_buffer("_indicator").dtype

    @property
    def device(self) -> torch_device:
        return self.get_buffer("_indicator").device

    @property
    def from_voxel_coordinates(self) -> Affine:
        """Affine mapping from voxel to world coordinates"""
        return Affine(self._from_voxel_coordinates)

    @property
    def to_voxel_coordinates(self) -> Affine:
        """Affine mapping from world to voxel coordinates"""
        return Affine(self._to_voxel_coordinates)

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        """Spatial shape of the coordinate system grid"""
        return self._spatial_shape

    @property
    def grid(self) -> MappableTensor:
        """Grid in the world coordinates"""
        return self._grid

    @property
    def voxel_grid(self) -> MappableTensor:
        """Grid in the voxel coordinates"""
        return self._voxel_grid

    @property
    def _grid(self) -> MappableTensor:
        if (
            self._grid_which_should_not_be_accessed_directly.device != self.device
            or self._grid_which_should_not_be_accessed_directly.dtype != self.dtype
        ):
            self._grid_which_should_not_be_accessed_directly = (
                self._grid_which_should_not_be_accessed_directly.cast(
                    dtype=self.dtype, device=self.device
                )
            )
        return self._grid_which_should_not_be_accessed_directly

    @property
    def _voxel_grid(self) -> MappableTensor:
        if (
            self._voxel_grid_which_should_not_be_accessed_directly.device != self.device
            or self._voxel_grid_which_should_not_be_accessed_directly.dtype != self.dtype
        ):
            self._voxel_grid_which_should_not_be_accessed_directly = (
                self._voxel_grid_which_should_not_be_accessed_directly.cast(
                    dtype=self.dtype, device=self.device
                )
            )
        return self._voxel_grid_which_should_not_be_accessed_directly

    @staticmethod
    def _calculate_voxel_size(affine_matrix: Tensor) -> Tensor:
        channel_dims = get_channel_dims(affine_matrix.ndim, n_channel_dims=2)
        row_dim = channel_dims[0]
        col_dim = channel_dims[1]
        matrix = affine_matrix.narrow(row_dim, 0, affine_matrix.size(row_dim) - 1)
        matrix = matrix.narrow(col_dim, 0, affine_matrix.size(col_dim) - 1)
        return matrix.square().sum(dim=row_dim).sqrt()

    def grid_spacing_cpu(self) -> Tensor:
        """Obtain grid spacing as a CPU tensor"""
        diagonal_on_host = self._from_voxel_coordinates.as_host_diagonal()
        if diagonal_on_host is None:
            return self._calculate_voxel_size(self._from_voxel_coordinates.as_host_matrix())
        return diagonal_on_host.generate_diagonal().abs()

    def grid_spacing(self) -> Tensor:
        """Obtain grid spacing on the target device"""
        diagonal = self._from_voxel_coordinates.as_diagonal()
        if diagonal is None:
            return self._calculate_voxel_size(self._from_voxel_coordinates.as_matrix())
        return diagonal.generate_diagonal().abs()

    def __repr__(self) -> str:
        return (
            "CoordinateSystem("
            f"spatial_shape={self._spatial_shape}, "
            f"to_voxel_coordinates={self._to_voxel_coordinates}, "
            f"from_voxel_coordinates={self.from_voxel_coordinates})"
        )

    def transform_voxel(self, affine_matrix: Tensor) -> "CoordinateSystem":
        """Transform the coordinates with an affine matrix in the
        voxel coordinates before applying the current affine transformation

        Args:
            affine_matrix: Affine matrix describing the transformation
                in the voxel coordinates. Tensor with shape
                ([*batch_shape, ]n_output_dims + 1, n_spatial_dims + 1).
                The affine matrix tensor should be provided on the host (cpu).
                That is, since the transformation may be used in control flow
                decisions, and we want to avoid synchronizing the target device
                with the host. The matrix is moved to the target device
                asynchronously, if needed.

        Returns:
            Coordinate system with a modified affine transformation from voxel
            to world coordinates.
        """
        return CoordinateSystem(
            from_voxel_coordinates=self._from_voxel_coordinates
            @ HostAffineTransformation(affine_matrix, device=self.device),
            spatial_shape=self._spatial_shape,
        )

    def transform_voxel_with_diagonal_matrix(
        self,
        diagonal: Optional[Union[Tensor, Sequence[Number], Number]] = None,
        translation: Optional[Union[Tensor, Sequence[Number], Number]] = None,
    ) -> "CoordinateSystem":
        """Transform the coordinates with a diagonal affine matrix in the
        voxel coordinates before applying the current affine transformation.

        The diagonal and the translation should be provided on the host (cpu).
        That is, since the transformation may be used in control flow decisions,
        and we want to avoid synchronizing the target device with the host. The
        matrix is moved to the target device asynchronously, if needed.

        Args:
            diagonal: Diagonal of the affine transformation matrix with shape
                ([*batch_shape, ]n_spatial_dims). Can be also
                given as a number or sequence of numbers. If None, corresponds
                to the diagonal of ones.
            translation: Translation of the affine transformation matrix
                with shape ([*batch_shape, ]n_spatial_dims).
                Can be also given as a number or sequence of numbers. If None,
                corresponds to the zero translation.

        Returns:
            Coordinate system with a modified affine transformation from voxel
            to world coordinates.
        """
        n_spatial_dims = len(self.spatial_shape)
        return CoordinateSystem(
            from_voxel_coordinates=self._from_voxel_coordinates
            @ HostDiagonalAffineTransformation(
                diagonal=diagonal,
                translation=translation,
                matrix_shape=(n_spatial_dims + 1, n_spatial_dims + 1),
                dtype=self.dtype,
                device=self.device,
            ),
            spatial_shape=self._spatial_shape,
        )

    def transform_world(self, affine_matrix: Tensor) -> "CoordinateSystem":
        """Transform the coordinates with an affine matrix in the
        world coordinates after applying the current affine transformation

        Args:
            affine_matrix: Affine matrix describing the transformation
                in the world coordinates. Tensor with shape
                ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1).
                The affine matrix tensor should be provided on the host (cpu).
                That is, since the transformation may be used in control flow
                decisions, and we want to avoid synchronizing the target device
                with the host. The matrix is moved to the target device
                asynchronously, if needed.

        Returns:
            Coordinate system with a modified affine transformation from voxel
            to world coordinates.
        """
        return CoordinateSystem(
            from_voxel_coordinates=HostAffineTransformation(affine_matrix, device=self.device)
            @ self._from_voxel_coordinates,
            spatial_shape=self._spatial_shape,
        )

    def __rmatmul__(self, affine_matrix: Tensor) -> "CoordinateSystem":
        return self.transform_world(affine_matrix)

    def transform_world_with_diagonal_matrix(
        self,
        diagonal: Optional[Union[Tensor, Sequence[Number], Number]] = None,
        translation: Optional[Union[Tensor, Sequence[Number], Number]] = None,
        n_output_dims: Optional[int] = None,
    ) -> "CoordinateSystem":
        """Transform the coordinates with a diagonal affine matrix in the
        world coordinates after applying the current affine transformation

        The diagonal and the translation should be provided on the host (cpu).
        That is, since the transformation may be used in control flow decisions,
        and we want to avoid synchronizing the target device with the host. The
        matrix is moved to the target device asynchronously, if needed.

        Args:
            diagonal: Diagonal of the affine transformation matrix with shape
                ([*batch_shape, ]diagonal_length). Can be also
                given as a number or sequence of numbers. If None, corresponds
                to the diagonal of ones.
            translation: Translation of the affine transformation matrix
                with shape ([*batch_shape, ]n_output_dims).
                Can be also given as a number or sequence of numbers. If None,
                corresponds to the zero translation.
            n_output_dims: Number of output dimensions of the transformation.

        Returns:
            Coordinate system with a modified affine transformation from voxel
            to world coordinates.
        """
        n_input_dims = self._from_voxel_coordinates.channels_shape[0] - 1
        if n_output_dims is None:
            n_output_dims = n_input_dims
        return CoordinateSystem(
            from_voxel_coordinates=HostDiagonalAffineTransformation(
                diagonal=diagonal,
                translation=translation,
                matrix_shape=(n_output_dims + 1, n_input_dims + 1),
                dtype=self.dtype,
                device=self.device,
            )
            @ self._from_voxel_coordinates,
            spatial_shape=self._spatial_shape,
        )

    def translate_voxel(
        self, translation: Union[Sequence[Number], Number, Tensor]
    ) -> "CoordinateSystem":
        """Translate the coordinates in the voxel coordinates before applying
        the current affine transformation

        Args:
            translation: Translation with shape ([*batch_shape, ]n_spatial_dims).
                Can be also given as a number or sequence of numbers.

        Returns:
            Coordinate system with translated coordinates.
        """
        return self.transform_voxel_with_diagonal_matrix(translation=translation)

    def translate_world(
        self, translation: Union[Sequence[Number], Number, Tensor]
    ) -> "CoordinateSystem":
        """Translate the coordinates in the world coordinates after applying
        the current affine transformation

        Args:
            translation: Translation with shape ([*batch_shape, ]n_output_dims).
                Can be also given as a number or sequence of numbers.
        """
        return self.transform_world_with_diagonal_matrix(translation=translation)

    def __add__(self, other: Union[Sequence[Number], Number, Tensor]) -> "CoordinateSystem":
        return self.translate_world(other)

    def __sub__(self, other: Union[Sequence[Number], Number, Tensor]) -> "CoordinateSystem":
        if not isinstance(other, (float, int, Tensor)):
            return self.translate_world([-value for value in other])
        return self.translate_world(-other)

    def __neg__(self) -> "CoordinateSystem":
        return self.multiply_world(-1)

    def multiply_voxel(self, factor: Union[Sequence[Number], Number, Tensor]) -> "CoordinateSystem":
        """Multiply the coordinates in the voxel coordinates (before applying
        the current affine transformation)

        Args:
            factor: Factor to multiply the grid spacing. A tensor with shape
                ([*batch_shape, ]n_spatial_dims). Can be also given as a number
                or sequence of numbers.

        Returns:
            Coordinate system with a modified affine transformation from voxel
            to world coordinates.
        """
        return self.transform_voxel_with_diagonal_matrix(diagonal=factor)

    def multiply_world(self, factor: Union[Sequence[Number], Number, Tensor]) -> "CoordinateSystem":
        """Multiply the coordinates in the world coordinates after applying
        the current affine transformation.

        Args:
            factor: Factor to multiply each dimension of the grid. A tensor with
                shape ([*batch_shape, ]n_output_dims). Can be also given as a number
                or sequence of numbers.

        Returns:
            Coordinate system with a modified affine transformation from voxel
            to world coordinates.
        """
        return self.transform_world_with_diagonal_matrix(diagonal=factor)

    def __mul__(self, other: Union[Sequence[Number], Number, Tensor]) -> "CoordinateSystem":
        return self.multiply_world(other)

    @overload
    def reformat(
        self,
        *,
        downsampling_factor: Optional[Union[Sequence[Number], Number, Tensor]] = None,
        upsampling_factor: Optional[Union[Sequence[Number], Number, Tensor]] = None,
        spatial_shape: Union[
            Sequence[Union[ReformattingSpatialShape, int]],
            ReformattingSpatialShape,
            int,
            Tensor,
        ] = OriginalFOV(fitting_method="round", fov_convention="full_voxels"),
        reference: Union[
            Sequence[Union[ReformattingReference, Number]],
            ReformattingReference,
            Number,
        ] = Center(),
        target_reference: Optional[
            Union[
                Sequence[Union[ReformattingReference, Number]],
                ReformattingReference,
                Number,
            ]
        ] = None,
    ) -> "CoordinateSystem": ...

    @overload
    def reformat(
        self,
        *,
        voxel_size: Optional[Union[Sequence[Union[float, int]], float, int, Tensor]] = None,
        spatial_shape: Union[
            Sequence[Union[ReformattingSpatialShape, int]],
            ReformattingSpatialShape,
            int,
            Tensor,
        ] = OriginalFOV(fitting_method="round", fov_convention="full_voxels"),
        reference: Union[
            Sequence[Union[ReformattingReference, Number]],
            ReformattingReference,
            Number,
        ] = Center(),
        target_reference: Optional[
            Union[
                Sequence[Union[ReformattingReference, Number]],
                ReformattingReference,
                Number,
            ]
        ] = None,
    ) -> "CoordinateSystem": ...

    def reformat(
        self,
        *,
        downsampling_factor: Optional[Union[Sequence[Number], Number, Tensor]] = None,
        upsampling_factor: Optional[Union[Sequence[Number], Number, Tensor]] = None,
        voxel_size: Optional[Union[Sequence[Number], Number, Tensor]] = None,
        spatial_shape: Union[
            Sequence[Union[ReformattingSpatialShape, int]],
            ReformattingSpatialShape,
            int,
            Tensor,
        ] = OriginalFOV(fitting_method="round", fov_convention="full_voxels"),
        reference: Union[
            Sequence[Union[ReformattingReference, Number]],
            ReformattingReference,
            Number,
        ] = Center(),
        target_reference: Optional[
            Union[
                Sequence[Union[ReformattingReference, Number]],
                ReformattingReference,
                Number,
            ]
        ] = None,
    ) -> "CoordinateSystem":
        """Reformat the coordinate system.

        All tensors should be provided on the host (cpu). That is, since the
        transformation may be used in control flow decisions, and we want to
        avoid synchronizing the target device with the host. The matrix is moved
        to the target device asynchronously, if needed.

        Args:
            downsampling_factor: Factor to downsample the grid voxel size. A tensor
                with shape ([*batch_shape, ]n_spatial_dims). Can be also given as a
                number or sequence of numbers.
            upsampling_factor: Factor to upsample the grid voxel size. A tensor with
                shape ([*batch_shape, ]n_spatial_dims). Can be also given as a number
                or sequence of numbers.
            voxel_size: Voxel size of the reformatted grid. A tensor with shape
                ([*batch_shape, ]n_spatial_dims). Can be also given as a number or
                sequence of numbers.
            spatial_shape: Defines spatial_shape of the target grid, either
                given separately for each dimension or as a single value in
                which case the same value is used for all the dimensions.
                Defaults to OriginalFOV("round", "full_voxels").
            reference: Defines the point in the original voxel
                coordinates which will be aligned with the target reference in
                the reformatted coordinates. Either given separately for each
                dimension or as a single value in which case the same value is
                used for all the dimensions. Defaults to Center().
            target_reference: Defaults to reference. Defines the point in the
                reformatted voxel coordinates which will be aligned with the
                source reference in the original coordinates. Either given
                separately for each dimension or as a single value in which case
                the same value is used for all the dimensions.

        Returns:
            Reformatted coordinate system.
        """
        original_voxel_size = self.grid_spacing_cpu()
        downsampling_factor = self._as_downsampling_factor(
            original_voxel_size, downsampling_factor, upsampling_factor, voxel_size
        )
        target_shape = self._get_target_shape(downsampling_factor, spatial_shape)
        source_reference_in_voxel_coordinates = original_voxel_size.new_tensor(
            self._get_reference_in_voxel_coordinates(reference, self._spatial_shape)
        )
        if target_reference is None:
            target_reference = reference
        target_reference_in_voxel_coordinates = original_voxel_size.new_tensor(
            self._get_reference_in_voxel_coordinates(target_reference, target_shape)
        )
        source_translation = HostDiagonalAffineTransformation(
            translation=source_reference_in_voxel_coordinates, device=self.device
        )
        target_translation = HostDiagonalAffineTransformation(
            translation=-target_reference_in_voxel_coordinates, device=self.device
        )
        downsampling_translation = HostDiagonalAffineTransformation(
            downsampling_factor, device=self.device
        )
        reformatted_transformation = (
            self._from_voxel_coordinates
            @ source_translation
            @ downsampling_translation
            @ target_translation
        )

        return CoordinateSystem(
            from_voxel_coordinates=reformatted_transformation,
            spatial_shape=target_shape,
        )

    def _as_downsampling_factor(
        self,
        original_voxel_size: Tensor,
        downsampling_factor: Optional[Union[Sequence[Number], Number, Tensor]] = None,
        upsampling_factor: Optional[Union[Sequence[Number], Number, Tensor]] = None,
        voxel_size: Optional[Union[Sequence[Number], Number, Tensor]] = None,
    ) -> Tensor:
        if voxel_size is not None:
            if not isinstance(voxel_size, Tensor):
                voxel_size = original_voxel_size.new_tensor(voxel_size)
            if voxel_size.ndim == 0:
                voxel_size = voxel_size.unsqueeze(0)
            voxel_size, original_voxel_size = broadcast_tensors_in_parts(
                voxel_size, original_voxel_size
            )
            if get_spatial_shape(voxel_size.shape, n_channel_dims=1) != tuple():
                raise ValueError(
                    "Downsampling factors or upsampling factors should not have spatial dimensions"
                )
            return voxel_size / original_voxel_size
        if downsampling_factor is not None:
            if not isinstance(downsampling_factor, Tensor):
                downsampling_factor = original_voxel_size.new_tensor(downsampling_factor)
            if downsampling_factor.ndim == 0:
                downsampling_factor = downsampling_factor.unsqueeze(0)
            processed_downsampling_factor = downsampling_factor
        else:
            processed_downsampling_factor = original_voxel_size.new_ones(1)
        if upsampling_factor is not None:
            if not isinstance(upsampling_factor, Tensor):
                upsampling_factor = original_voxel_size.new_tensor(upsampling_factor)
            if upsampling_factor.ndim == 0:
                upsampling_factor = upsampling_factor.unsqueeze(0)
            processed_upsampling_factor = 1 / upsampling_factor
        else:
            processed_upsampling_factor = original_voxel_size.new_ones(1)
        _batch_shape, channels_shape, spatial_shape = broadcast_shapes_in_parts_splitted(
            processed_downsampling_factor.shape,
            processed_upsampling_factor.shape,
            original_voxel_size.shape,
        )
        if spatial_shape != tuple():
            raise ValueError(
                "Downsampling factors or upsampling factors should not have spatial dimensions"
            )
        processed_downsampling_factor = broadcast_to_in_parts(
            processed_downsampling_factor,
            channels_shape=channels_shape,
        )
        processed_upsampling_factor = broadcast_to_in_parts(
            processed_upsampling_factor,
            channels_shape=channels_shape,
        )
        return processed_downsampling_factor * processed_upsampling_factor

    def _get_reference_in_voxel_coordinates(
        self,
        reference: Union[
            Sequence[Union[ReformattingReference, Number]],
            ReformattingReference,
            Number,
        ],
        spatial_shape: Sequence[int],
    ) -> List[float]:
        if isinstance(reference, (ReformattingReference, float, int)):
            reference = [reference] * len(spatial_shape)
        voxel_coordinate_reference: List[float] = []
        for dim_reference, dim_size in zip(reference, spatial_shape):
            if isinstance(dim_reference, (float, int)):
                voxel_coordinate_reference.append(dim_reference)
            else:
                voxel_coordinate_reference.append(
                    float(dim_reference.get_voxel_coordinate(dim_size))
                )
        return voxel_coordinate_reference

    def _get_target_shape(
        self,
        downsampling_factor: Tensor,
        spatial_shape: Union[
            Sequence[Union[ReformattingSpatialShape, int]], ReformattingSpatialShape, int, Tensor
        ],
    ) -> Sequence[int]:
        if isinstance(spatial_shape, Tensor):
            if spatial_shape.ndim not in (0, 1):
                raise ValueError("Invalid spatial_shape tensor for reformatting")
            if spatial_shape.is_floating_point() or spatial_shape.is_complex():
                raise ValueError("Shape tensor should be an integer tensor")
            spatial_shape = spatial_shape.expand((len(self._spatial_shape),))
            return spatial_shape.tolist()
        if isinstance(spatial_shape, int):
            return [spatial_shape] * len(self._spatial_shape)
        if isinstance(spatial_shape, ReformattingSpatialShape):
            spatial_shape = [spatial_shape] * len(self._spatial_shape)
        else:
            if len(spatial_shape) != len(self._spatial_shape):
                raise ValueError("Invalid spatial_shape for reformatting")
            if all(isinstance(dim_shape, int) for dim_shape in spatial_shape):
                return cast(Sequence[int], spatial_shape)
        downsampling_factor = move_channels_last(downsampling_factor).view(
            -1, get_channels_shape(downsampling_factor.shape, n_channel_dims=1)[0]
        )
        if downsampling_factor.ndim == 1:
            downsampling_factor = downsampling_factor.unsqueeze(0)
        output_shape: Optional[Sequence[int]] = None
        for batch_index in range(downsampling_factor.size(0)):
            target_shape = self._get_target_shape_for_batch(
                downsampling_factor[batch_index], spatial_shape
            )
            if output_shape is None:
                output_shape = target_shape
            else:
                if output_shape != target_shape:
                    raise ValueError(
                        "Inconsistent target shapes obtained with given spatial shape options, "
                        "consider defining an explicit target shape for reformatting."
                    )
        assert output_shape is not None
        return output_shape

    def _get_target_shape_for_batch(
        self,
        single_downsampling_factor: Tensor,
        spatial_shape: Sequence[Union[ReformattingSpatialShape, int]],
    ) -> Sequence[int]:
        target_shape = []
        for dim_original_shape, dim_shape, dim_downsampling_factor in zip(
            self._spatial_shape, spatial_shape, single_downsampling_factor
        ):
            if isinstance(dim_shape, ReformattingSpatialShape):
                target_shape.append(
                    dim_shape.target_size(dim_original_shape, dim_downsampling_factor.item())
                )
            elif isinstance(dim_shape, int):
                target_shape.append(dim_shape)
            else:
                raise ValueError(f"Invalid spatial shape for reformatting: {spatial_shape}")
        return target_shape

    @staticmethod
    def _ensure_tensor_on_host(item: Any) -> None:
        if isinstance(item, Tensor) and item.device.type != "cpu":
            raise ValueError(
                "Tensors contained by the coordinate system should be located on CPU. "
                "That is since the coordinate system can affect the control flow "
                "and we want to avoid CPU-GPU synchronizations. Tensors contained "
                "by the coordinate system are moved to the target device of the "
                "coordinate system when needed (which can be done asynchonously)."
            )

    @property
    def _from_voxel_coordinates(self) -> IHostAffineTransformation:
        from_voxel_coordinates = (
            self._coordinate_transformations_which_should_not_be_accessed_directly[
                "from_voxel_coordinates"
            ]
        )
        if (
            from_voxel_coordinates.device != self.device
            or from_voxel_coordinates.dtype != self.dtype
        ):
            self._coordinate_transformations_which_should_not_be_accessed_directly[
                "from_voxel_coordinates"
            ] = from_voxel_coordinates.cast(dtype=self.dtype, device=self.device)
        return self._coordinate_transformations_which_should_not_be_accessed_directly[
            "from_voxel_coordinates"
        ]

    @property
    def _to_voxel_coordinates(self) -> IHostAffineTransformation:
        to_voxel_coordinates = (
            self._coordinate_transformations_which_should_not_be_accessed_directly[
                "to_voxel_coordinates"
            ]
        )
        if to_voxel_coordinates.device != self.device or to_voxel_coordinates.dtype != self.dtype:
            self._coordinate_transformations_which_should_not_be_accessed_directly[
                "to_voxel_coordinates"
            ] = to_voxel_coordinates.cast(dtype=self.dtype, device=self.device)
        return self._coordinate_transformations_which_should_not_be_accessed_directly[
            "to_voxel_coordinates"
        ]

    def _get_children(self) -> Mapping[str, TensorLike]:
        return {
            "from_voxel_coordinates": self._from_voxel_coordinates,
            "to_voxel_coordinates": self._to_voxel_coordinates,
        }

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, TensorLike]
    ) -> "CoordinateSystem":
        if not isinstance(children["from_voxel_coordinates"], IHostAffineTransformation):
            raise ValueError("from_voxel_coordinates should be an affine transformation")
        if not isinstance(children["to_voxel_coordinates"], IHostAffineTransformation):
            raise ValueError("to_voxel_coordinates should be an affine transformation")
        return CoordinateSystem(
            spatial_shape=self._spatial_shape,
            from_voxel_coordinates=children["from_voxel_coordinates"],
            to_voxel_coordinates=children["to_voxel_coordinates"],
        )
