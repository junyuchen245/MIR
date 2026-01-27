"""Mappings composable with each other, and acting on mappable tensors."""

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    TypeVar,
    Union,
    cast,
    overload,
)

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import get_default_dtype
import torch
def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
from MIR.models.SITReg.composable_mapping.affine_transformation import IAffineTransformation
from MIR.models.SITReg.composable_mapping.interface import Number
from MIR.models.SITReg.composable_mapping.mappable_tensor import MappableTensor, mappable
from MIR.models.SITReg.composable_mapping.sampler import DataFormat, ISampler, get_sampler
from MIR.models.SITReg.composable_mapping.tensor_like import TensorLike

from .interface import ICoordinateSystemContainer

if TYPE_CHECKING:
    from MIR.models.SITReg.composable_mapping.coordinate_system import CoordinateSystem


def _assign_coordinates_if_available(
    target_mapping: "ComposableMapping", sources: Iterable[Any]
) -> "ComposableMapping":
    for source in sources:
        if isinstance(source, ICoordinateSystemContainer):
            return target_mapping.assign_coordinates(source)
    return target_mapping


@overload
def _bivariate_arithmetic_operator_template(
    mapping: "GridComposableMapping", other: Union["ComposableMapping", Number, MappableTensor]
) -> "GridComposableMapping": ...
@overload
def _bivariate_arithmetic_operator_template(
    mapping: "ComposableMapping", other: "GridComposableMapping"
) -> "GridComposableMapping": ...
@overload
def _bivariate_arithmetic_operator_template(
    mapping: "ComposableMapping", other: Union["ComposableMapping", Number, MappableTensor]
) -> "ComposableMapping": ...
def _bivariate_arithmetic_operator_template(  # type: ignore
    mapping: "ComposableMapping",  # pylint: disable=unused-argument
    other: Union["ComposableMapping", Number, MappableTensor],  # pylint: disable=unused-argument
) -> "ComposableMapping": ...


@overload
def _univariate_arithmetic_operator_template(
    mapping: "GridComposableMapping",
) -> "GridComposableMapping": ...
@overload
def _univariate_arithmetic_operator_template(
    mapping: "ComposableMapping",
) -> "ComposableMapping": ...
def _univariate_arithmetic_operator_template(  # type: ignore
    mapping: "ComposableMapping",  # pylint: disable=unused-argument
) -> "ComposableMapping": ...


T = TypeVar("T")


def _generate_bivariate_arithmetic_operator(
    operator: Callable[[MappableTensor, Any], MappableTensor],
    inverse_operator: Callable[[MappableTensor, Any], MappableTensor],
    _type_template: T,
) -> T:
    def _operator(
        mapping: "ComposableMapping",
        other: Union["ComposableMapping", Number, MappableTensor],
    ) -> "ComposableMapping":
        return _assign_coordinates_if_available(
            _BivariateArithmeticOperator(
                mapping, other, operator=operator, inverse_operator=inverse_operator
            ),
            [mapping, other],
        )

    return cast(T, _operator)


def _generate_univariate_arithmetic_operator(
    operator: Callable[[MappableTensor], MappableTensor],
    inverse_operator: Callable[[MappableTensor], MappableTensor],
    _type_template: T,
) -> T:
    def _operator(mapping: "ComposableMapping") -> "ComposableMapping":
        return _assign_coordinates_if_available(
            _UnivariateArithmeticOperator(
                mapping,
                operator=operator,
                inverse_operator=inverse_operator,
            ),
            [mapping],
        )

    return cast(T, _operator)


@overload
def _composition(
    self: "GridComposableMapping", right_mapping: "ComposableMapping"
) -> "GridComposableMapping": ...
@overload
def _composition(
    self: "ComposableMapping", right_mapping: "GridComposableMapping"
) -> "GridComposableMapping": ...
@overload
def _composition(
    self: "ComposableMapping", right_mapping: "ComposableMapping"
) -> "ComposableMapping": ...
def _composition(
    self: "ComposableMapping", right_mapping: "ComposableMapping"
) -> "ComposableMapping":
    return _assign_coordinates_if_available(
        _Composition(self, right_mapping), [self, right_mapping]
    )


@overload
def _set_default_resampling_data_format(
    self: "GridComposableMapping", data_format: Optional[DataFormat]
) -> "GridComposableMapping": ...
@overload
def _set_default_resampling_data_format(
    self: "ComposableMapping", data_format: Optional[DataFormat]
) -> "ComposableMapping": ...
def _set_default_resampling_data_format(
    self: "ComposableMapping", data_format: Optional[DataFormat]
) -> "ComposableMapping":
    return _assign_coordinates_if_available(
        _SetDefaultResamplingDataFormatDecorator(self, data_format), [self]
    )


class ComposableMapping(TensorLike, ABC):
    """Base class for mappings composable with each other, and acting on mappable
    tensors.

    In general a composable mapping is a callable object that takes coordinates
    as input and returns the mapping evaluated at these coordinates. Composable
    mappings are generally assumed to be independent over the batch and spatial
    dimensions of an input.

    As the name suggests, a composable mapping can be additionally composed with
    other composable mappings with the `__matmul__` operator (@). Composing does
    not apply resampling, an operation which can be executed separately using
    `resample_to` method (or `GridComposableMapping.resample` method for
    composable mappings with an assigned coordinate system).

    Basic arithmetic operations are also supported between two composable
    mappings or between a composable mapping and a number or a tensor, both of
    which return a new composable mapping.
    """

    @abstractmethod
    def __call__(self, coordinates: MappableTensor) -> MappableTensor:
        """Evaluate the mapping at coordinates.

        Args:
            coordinates: Coordinates to evaluate the mapping at with shape
                ([*batch_shape ,]n_dims[, *spatial_shape]).

        Returns:
            Mapping evaluated at the coordinates.

        @public
        """

    @abstractmethod
    def invert(self, **arguments) -> "ComposableMapping":
        """Invert the mapping.

        Args:
            arguments: Arguments for the inversion.

        Returns:
            The inverted mapping.
        """

    def sample_to(
        self,
        target: ICoordinateSystemContainer,
        data_format: DataFormat = DataFormat.world_coordinates(),
    ) -> MappableTensor:
        """Evaluate the mapping at the coordinates defined by the target.

        Args:
            target: Target coordinate system (or a container with a coordinate system)
                defining a grid to evaluate the mapping at.
            data_format: Data format of the output. Default data format is
                DataFormat.world_coordinates() corresponding to the values
                obtained by evaluating the mapping at the coordinates defined.

        Returns:
            Mappable tensor containing the values obtained by evaluating the
            mapping at the coordinates defined by the target.
        """
        sampled = self(target.coordinate_system.grid)
        if data_format.coordinate_type == "voxel":
            sampled = target.coordinate_system.to_voxel_coordinates(sampled)
        if data_format.representation == "displacements":
            grid = (
                target.coordinate_system.voxel_grid
                if data_format.coordinate_type == "voxel"
                else target.coordinate_system.grid
            )
            sampled = sampled - grid
        return sampled

    def resample_to(
        self,
        target: ICoordinateSystemContainer,
        data_format: Optional[DataFormat] = None,
        sampler: Optional["ISampler"] = None,
    ) -> "SamplableVolume":
        """Resample the mapping at the coordinates defined by the target.

        Args:
            target: Target coordinate system (or a container with a coordinate system)
                defining a grid to resample the mapping at.
            data_format: Data format used as an internal representation of the
                generated resampled mapping. Default data format depends on the
                mapping, but as a general rule is the same as the data format of
                the mapping being sampled, or the default data format of the
                left mapping in a composition or other operation. When no clear
                default data format is available, DataFormat.world_coordinates()
                is used. Default resampling data format can be set for a mapping
                using `set_default_resampling_data_format`.
            sampler: Sampler used by the generated resampled mapping. Note that
                this sampler is not used to resample the mapping, but to sample
                the generated resampled mapping. If None, the default sampler
                is used (see `default_sampler`).

        Returns:
            Resampled mapping.
        """
        data_format = self._get_resampling_data_format(data_format)
        return SamplableVolume(
            data=self.sample_to(
                target,
                data_format=data_format,
            ),
            coordinate_system=target.coordinate_system,
            data_format=data_format,
            sampler=sampler,
        )

    def assign_coordinates(
        self, coordinates: "ICoordinateSystemContainer"
    ) -> "GridComposableMapping":
        """Assign a coordinate system for the mapping.

        This only changes the coordinate system of the mapping, the mapping itself
        is not changed. The coordinate system contained by the mapping affects
        behaviour of some methods such as `GridComposableMapping.sample` and
        `GridComposableMapping.resample`.

        Args:
            coordinates: Coordinate system (or a container with a coordinate system)
                to assign for the mapping.

        Returns:
            Mapping with the given target coordinate system.
        """
        return _AssignCoordinatesDecorator(self, coordinates.coordinate_system)

    def as_affine_transformation(self) -> IAffineTransformation:
        """Obtain the mapping as an affine transformation on PyTorch tensors, if possible.

        Returns:
            Affine transformation on PyTorch tensors.

        Raises:
            NotAffineTransformationError: If the mapping is not an affine transformation on
                PyTorch tensors.
        """
        tracer = _AffineTracer()
        traced = self(tracer)
        if isinstance(traced, _AffineTracer):
            if traced.traced_affine is None:
                raise NotAffineTransformationError("Could not infer affine transformation.")
            return traced.traced_affine
        raise NotAffineTransformationError("Could not infer affine transformation.")

    def as_affine_matrix(self) -> Tensor:
        """Obtain the mapping as an affine matrix, if possible.

        Returns:
            Affine matrix with
            shape ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape]).
        """
        return self.as_affine_transformation().as_matrix()

    @property
    def default_resampling_data_format(self) -> Optional[DataFormat]:
        """Default data format to use in resampling operations for
        the mapping.

        If None, DataFormat.world_coordinates() will be used but the behaviour
        in operations with other mappings is different as the default data format
        of the other mapping will be used.
        """
        return None

    def set_default_resampling_data_format(
        self, data_format: Optional[DataFormat]
    ) -> "ComposableMapping":
        """Set the default data format to use in resampling operations for
        the mapping.

        Args:
            data_format: Default data format to use in resampling operations.

        Returns:
            Mapping with the default data format set.
        """
        return _SetDefaultResamplingDataFormatDecorator(self, data_format)

    def _get_resampling_data_format(self, data_format: Optional[DataFormat]) -> DataFormat:
        if data_format is not None:
            return data_format
        if self.default_resampling_data_format is None:
            return DataFormat.world_coordinates()
        return self.default_resampling_data_format

    __matmul__ = _composition
    """Compose with another mapping.
    
    Args:
        right_mapping: Mapping to compose with.
    
    Returns:
        Composed mapping.
    @public
    """
    __add__ = _generate_bivariate_arithmetic_operator(
        lambda x, y: x + y, lambda x, y: x - y, _bivariate_arithmetic_operator_template
    )
    __radd__ = __add__
    __sub__ = _generate_bivariate_arithmetic_operator(
        lambda x, y: x - y, lambda x, y: x + y, _bivariate_arithmetic_operator_template
    )
    __rsub__ = _generate_bivariate_arithmetic_operator(
        lambda x, y: y - x, lambda x, y: y - x, _bivariate_arithmetic_operator_template
    )
    __mul__ = _generate_bivariate_arithmetic_operator(
        lambda x, y: x * y, lambda x, y: x / y, _bivariate_arithmetic_operator_template
    )
    __rmul__ = __mul__
    __truediv__ = _generate_bivariate_arithmetic_operator(
        lambda x, y: x / y, lambda x, y: x * y, _bivariate_arithmetic_operator_template
    )
    __rtruediv__ = _generate_bivariate_arithmetic_operator(
        lambda x, y: y / x, lambda x, y: y / x, _bivariate_arithmetic_operator_template
    )
    __pow__ = _generate_bivariate_arithmetic_operator(
        lambda x, y: x**y, lambda x, y: x ** (1 / y), _bivariate_arithmetic_operator_template
    )
    __neg__ = _generate_univariate_arithmetic_operator(
        lambda x: -x, lambda x: -x, _univariate_arithmetic_operator_template
    )


class GridComposableMapping(ComposableMapping, ICoordinateSystemContainer, ABC):
    """Base class for composable mappings with an assigned coordinate system."""

    @abstractmethod
    def invert(self, **arguments) -> "GridComposableMapping":
        pass

    def sample(
        self,
        data_format: DataFormat = DataFormat.world_coordinates(),
    ) -> MappableTensor:
        """Evaluate the mapping at the coordinates contained by the mapping.

        Args:
            data_format: Data format of the output. Default data format is
                DataFormat.world_coordinates() corresponding to the values
                obtained by evaluating the mapping at the coordinates defined.

        Returns:
            Mappable tensor containing the values obtained by evaluating the
            mapping at the coordinates contained by the mapping.
        """
        return self.sample_to(self, data_format=data_format)

    def resample(
        self,
        data_format: Optional[DataFormat] = None,
        sampler: Optional[ISampler] = None,
    ) -> "SamplableVolume":
        """Resample the mapping at the coordinates contained by the mapping.

        Args:
            data_format: Data format used as an internal representation of the
                generated resampled mapping. Default data format depends on the
                mapping, but as a general rule is the same as the data format of
                the mapping being sampled, or the default data format of the
                left mapping in a composition or other operation. When no clear
                default data format is available, DataFormat.world_coordinates()
                is used. Default resampling data format can be set for a mapping
                using `set_default_resampling_data_format`.
            sampler: Sampler used by the generated resampled mapping. Note that
                this sampler is not used to resample the mapping, but to sample
                the generated resampled mapping. If None, the default sampler
                is used (see `default_sampler`).

        Returns:
            Resampled mapping.
        """
        return self.resample_to(
            self,
            data_format=data_format,
            sampler=sampler,
        )

    def set_default_resampling_data_format(
        self, data_format: Optional[DataFormat]
    ) -> "GridComposableMapping":
        return super().set_default_resampling_data_format(data_format).assign_coordinates(self)


class _AssignCoordinatesDecorator(GridComposableMapping):
    """Decorator for coupling a composable mapping with a coordinate system.

    Arguments:
        mapping: Composable mapping.
        coordinate_system: Coordinate system assigned to the mapping.
    """

    def __init__(self, mapping: ComposableMapping, coordinate_system: "CoordinateSystem") -> None:
        super().__init__()
        self._mapping = mapping
        self._coordinate_system = coordinate_system

    def _get_children(self) -> Mapping[str, TensorLike]:
        return {"mapping": self._mapping, "coordinate_system": self._coordinate_system}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, TensorLike]
    ) -> "_AssignCoordinatesDecorator":
        return _AssignCoordinatesDecorator(
            cast(ComposableMapping, children["mapping"]),
            cast("CoordinateSystem", children["coordinate_system"]),
        )

    def __call__(self, coordinates: MappableTensor) -> MappableTensor:
        return self._mapping(coordinates)

    def invert(self, **arguments) -> GridComposableMapping:
        return self._mapping.invert(**arguments).assign_coordinates(self._coordinate_system)

    @property
    def coordinate_system(self) -> "CoordinateSystem":
        return self._coordinate_system

    @property
    def default_resampling_data_format(self) -> Optional[DataFormat]:
        return self._mapping.default_resampling_data_format

    def __repr__(self) -> str:
        return (
            f"_AssignCoordinatesDecorator(mapping={self._mapping}, "
            f"coordinate_system={self._coordinate_system})"
        )


class _SetDefaultResamplingDataFormatDecorator(ComposableMapping):
    """Decorator for setting the default data format of a composable mapping.

    Arguments:
        mapping: Composable mapping.
        data_format: Default data format to use in sampling and resampling operations.
    """

    def __init__(self, mapping: ComposableMapping, data_format: Optional[DataFormat]) -> None:
        super().__init__()
        self._mapping = mapping
        self._data_format = data_format

    def _get_children(self) -> Mapping[str, TensorLike]:
        return {"mapping": self._mapping}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, TensorLike]
    ) -> "_SetDefaultResamplingDataFormatDecorator":
        return _SetDefaultResamplingDataFormatDecorator(
            cast(ComposableMapping, children["mapping"]),
            self._data_format,
        )

    def __call__(self, coordinates: MappableTensor) -> MappableTensor:
        return self._mapping(coordinates)

    def invert(self, **arguments) -> ComposableMapping:
        return self._mapping.invert(**arguments).set_default_resampling_data_format(
            self._data_format
        )

    @property
    def default_resampling_data_format(self) -> Optional[DataFormat]:
        return self._data_format

    def __repr__(self) -> str:
        return (
            f"_SetDefaultResamplingDataFormatDecorator("
            f"mapping={self._mapping}, "
            f"data_format={self._data_format})"
        )


class Identity(ComposableMapping):
    """Identity mapping.

    Arguments:
        dtype: Data type of the mapping. This has no effect except on
            the data type property of the mapping, but for consistency
            of the composable mapping interface, the data type is stored.
        device: Device of the tensor. This has no effect except on
            the device property of the mapping, but for consistency
            of the composable mapping interface, the device is stored.
    """

    def __init__(
        self, dtype: Optional[torch_dtype] = None, device: Optional[torch_device] = None
    ) -> None:
        super().__init__()
        self._dtype = get_default_dtype() if dtype is None else dtype
        self._device = get_default_device() if device is None else device

    @property
    def dtype(self) -> torch_dtype:
        return self._dtype

    @property
    def device(self) -> torch_device:
        return self._device

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, TensorLike]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, TensorLike]
    ) -> "Identity":
        return Identity()

    def __call__(self, coordinates: MappableTensor) -> MappableTensor:
        return coordinates

    def invert(self, **_inversion_parameters) -> "Identity":
        return Identity()

    def detach(self) -> "Identity":
        return self

    def __repr__(self) -> str:
        return "Identity()"


class _Composition(ComposableMapping):
    """Composition of two mappings."""

    def __init__(self, left_mapping: ComposableMapping, right_mapping: ComposableMapping) -> None:
        super().__init__()
        self._left_mapping = left_mapping
        self._right_mapping = right_mapping

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, TensorLike]
    ) -> "_Composition":
        return _Composition(
            cast(ComposableMapping, children["left_mapping"]),
            cast(ComposableMapping, children["right_mapping"]),
        )

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, TensorLike]:
        return {"left_mapping": self._left_mapping, "right_mapping": self._right_mapping}

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        return self._left_mapping(self._right_mapping(masked_coordinates))

    def invert(self, **arguments) -> "ComposableMapping":
        return _Composition(
            self._right_mapping.invert(**arguments),
            self._left_mapping.invert(**arguments),
        )

    @property
    def default_resampling_data_format(self) -> Optional[DataFormat]:
        if self._left_mapping.default_resampling_data_format is not None:
            return self._left_mapping.default_resampling_data_format
        return self._right_mapping.default_resampling_data_format

    def __repr__(self) -> str:
        return (
            f"_Composition(left_mapping={self._left_mapping}, right_mapping={self._right_mapping})"
        )


class _BivariateArithmeticOperator(ComposableMapping):
    def __init__(
        self,
        mapping: ComposableMapping,
        other: Union[ComposableMapping, MappableTensor, Number, Tensor],
        operator: Callable[[MappableTensor, Any], MappableTensor],
        inverse_operator: Callable[[MappableTensor, Any], MappableTensor],
    ) -> None:
        super().__init__()
        self._mapping = mapping
        self._other = other
        self._operator = operator
        self._inverse_operator = inverse_operator

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, TensorLike]
    ) -> "_BivariateArithmeticOperator":
        return _BivariateArithmeticOperator(
            self._mapping,
            cast(Union[ComposableMapping, MappableTensor, Number, Tensor], tensors["other"]),
            self._operator,
            self._inverse_operator,
        )

    def _get_tensors(self) -> Mapping[str, Tensor]:
        tensors = {}
        if isinstance(self._other, Tensor):
            tensors["other"] = self._other
        return tensors

    def _get_children(self) -> Mapping[str, TensorLike]:
        children: Dict[str, TensorLike] = {"mapping": self._mapping}
        if isinstance(self._other, (MappableTensor, ComposableMapping)):
            children["other"] = self._other
        return children

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        if isinstance(self._other, ComposableMapping):
            return self._operator(
                self._mapping(masked_coordinates), self._other(masked_coordinates)
            )
        return self._operator(self._mapping(masked_coordinates), self._other)

    def invert(self, **arguments) -> "ComposableMapping":
        if isinstance(self._other, ComposableMapping):
            raise ValueError("Operation is not invertible")
        return _Composition(
            self._mapping.invert(**arguments),
            _BivariateArithmeticOperator(
                Identity(), self._other, self._inverse_operator, self._operator
            ),
        )

    @property
    def default_resampling_data_format(self) -> Optional[DataFormat]:
        if self._mapping.default_resampling_data_format is not None:
            return self._mapping.default_resampling_data_format
        if isinstance(self._other, ComposableMapping):
            return self._other.default_resampling_data_format
        return None

    def __repr__(self) -> str:
        return (
            f"_BivariateArithmeticOperator(mapping={self._mapping}, "
            "other={self._other}, operator={self._operator}, "
            "inverse_operator={self._inverse_operator})"
        )


class _UnivariateArithmeticOperator(ComposableMapping):
    def __init__(
        self,
        mapping: ComposableMapping,
        operator: Callable[[MappableTensor], MappableTensor],
        inverse_operator: Callable[[MappableTensor], MappableTensor],
    ) -> None:
        super().__init__()
        self._mapping = mapping
        self._operator = operator
        self._inverse_operator = inverse_operator

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, TensorLike]
    ) -> "_UnivariateArithmeticOperator":
        return _UnivariateArithmeticOperator(
            cast(ComposableMapping, self._mapping),
            self._operator,
            self._inverse_operator,
        )

    def _get_children(self) -> Mapping[str, TensorLike]:
        return {"mapping": self._mapping}

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        return self._operator(self._mapping(masked_coordinates))

    def invert(self, **arguments) -> "ComposableMapping":
        return _Composition(
            self._mapping.invert(**arguments),
            _UnivariateArithmeticOperator(Identity(), self._inverse_operator, self._operator),
        )

    @property
    def default_resampling_data_format(self) -> Optional[DataFormat]:
        return self._mapping.default_resampling_data_format

    def __repr__(self) -> str:
        return (
            f"_UnivariateArithmeticOperator(mapping={self._mapping}, "
            "operator={self._operator}, "
            "inverse_operator={self._inverse_operator})"
        )


class SamplableVolume(GridComposableMapping):
    """Mapping defined based on a regular grid of values and a sampler turning the
    grid values into a continuously defined mapping.

    The easiest way to create a samplable volume is to use the factory
    function provided in this module or the class method of this class:
    `samplable_volume`, `SamplableVolume.from_tensor`.

    Arguments:
        data: Regular grid of values, with shape
            (*batch_shape, *channels_shape, *spatial_shape).
        coordinate_system: Coordinate system describing transformation from the
            voxel coordinates on the data grid to the world coordinates.
        data_format: Data format of the grid values.
        sampler: Sampler turning the grid values into a continuously defined mapping
            over spatial coordinates.
    """

    def __init__(
        self,
        data: MappableTensor,
        coordinate_system: "CoordinateSystem",
        data_format: DataFormat = DataFormat.world_coordinates(),
        sampler: Optional[ISampler] = None,
    ) -> None:
        super().__init__()
        if coordinate_system.spatial_shape != data.spatial_shape:
            raise ValueError(
                "Coordinate system spatial shape must match the data spatial shape. "
                f"Coordinate system spatial shape: {coordinate_system.spatial_shape}, "
                f"data spatial shape: {data.spatial_shape}."
            )
        self._data = data
        self._coordinate_system = coordinate_system
        self._data_format = data_format
        self._sampler = get_sampler(sampler)

    @classmethod
    def from_tensor(
        cls,
        data: Tensor,
        coordinate_system: "CoordinateSystem",
        mask: Optional[Tensor] = None,
        data_format: DataFormat = DataFormat.world_coordinates(),
        sampler: Optional[ISampler] = None,
        n_channel_dims: int = 1,
    ) -> "SamplableVolume":
        """Create a samplable volume from a tensor.

        Args:
            data: Regular grid of values, with shape
                (*batch_shape, *channels_shape, *spatial_shape).
            coordinate_system: Coordinate system describing transformation from the
                voxel coordinates on the data grid to the world coordinates.
            mask: Mask for the data,
                with shape (*batch_shape, *(1,) * n_channel_dims, *spatial_shape).
            data_format: Data format of the grid values.
            sampler: Sampler turning the grid values into a continuously defined mapping
                over spatial coordinates.
            n_channel_dims: Number of channel dimensions.

        Returns:
            Samplable volume.
        """
        return SamplableVolume(
            data=mappable(data, mask, n_channel_dims=n_channel_dims),
            coordinate_system=coordinate_system,
            data_format=data_format,
            sampler=sampler,
        )

    def modify_sampler(self, sampler: ISampler) -> "SamplableVolume":
        """Modify the sampler of the volume.

        Args:
            sampler: New sampler.

        Returns:
            Samplable volume with the new sampler.
        """
        return SamplableVolume(
            data=self._data,
            coordinate_system=self._coordinate_system,
            data_format=self._data_format,
            sampler=sampler,
        )

    def clear_mask(self) -> "SamplableVolume":
        """Clear data mask."""
        return SamplableVolume(
            data=self._data.clear_mask(),
            coordinate_system=self._coordinate_system,
            data_format=self._data_format,
            sampler=self._sampler,
        )

    @property
    def coordinate_system(self) -> "CoordinateSystem":
        return self._coordinate_system

    @property
    def default_resampling_data_format(self) -> DataFormat:
        return self._data_format

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, TensorLike]:
        return {
            "data": self._data,
            "coordinate_system": self._coordinate_system,
        }

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, TensorLike]
    ) -> "SamplableVolume":
        return SamplableVolume(
            data=cast(MappableTensor, children["data"]),
            coordinate_system=cast("CoordinateSystem", children["coordinate_system"]),
            data_format=self._data_format,
            sampler=self._sampler,
        )

    def __call__(self, coordinates: MappableTensor) -> MappableTensor:
        voxel_coordinates = self._coordinate_system.to_voxel_coordinates(coordinates)
        sampled = self._sampler(self._data, voxel_coordinates)
        if self._data_format.representation == "displacements":
            if self._data_format.coordinate_type == "voxel":
                sampled = voxel_coordinates + sampled
            elif self._data_format.coordinate_type == "world":
                sampled = coordinates + sampled
        if self._data_format.coordinate_type == "voxel":
            sampled = self._coordinate_system.from_voxel_coordinates(sampled)
        return sampled

    def invert(self, **arguments) -> "SamplableVolume":
        return SamplableVolume(
            data=self._data,
            coordinate_system=self._coordinate_system,
            data_format=self._data_format,
            sampler=self._sampler.inverse(self._coordinate_system, self._data_format, arguments),
        )

    def __repr__(self) -> str:
        return (
            f"SamplableVolume(data={self._data}, "
            f"coordinate_system={self._coordinate_system}, "
            f"data_format={self._data_format}, "
            f"sampler={self._sampler})"
        )


def samplable_volume(
    data: Tensor,
    coordinate_system: "CoordinateSystem",
    mask: Optional[Tensor] = None,
    data_format: DataFormat = DataFormat.world_coordinates(),
    sampler: Optional[ISampler] = None,
    n_channel_dims: int = 1,
) -> SamplableVolume:
    """Create a samplable volume from a tensor.

    See: `SamplableVolume.from_tensor`.
    """
    return SamplableVolume.from_tensor(
        data=data,
        coordinate_system=coordinate_system,
        mask=mask,
        data_format=data_format,
        sampler=sampler,
        n_channel_dims=n_channel_dims,
    )


class NotAffineTransformationError(Exception):
    """Error raised when trying to represent a non-affine composable mapping as
    an affine transformation."""


class _AffineTracer(MappableTensor):
    # pylint: disable=super-init-not-called
    def __init__(self, affine_transformation: Optional[IAffineTransformation] = None) -> None:
        self.traced_affine: Optional[IAffineTransformation] = affine_transformation

    def transform(self, affine_transformation: IAffineTransformation) -> MappableTensor:
        if self.traced_affine is not None:
            traced_affine = affine_transformation @ self.traced_affine
        else:
            traced_affine = affine_transformation
        return _AffineTracer(traced_affine)

    def __getattribute__(self, name: str):
        if name not in ("transform", "traced_affine"):
            raise NotAffineTransformationError(
                "Could not infer affine transformation since an other operation "
                f"than applying affine transformation was applied ({name})"
            )
        return object.__getattribute__(self, name)
