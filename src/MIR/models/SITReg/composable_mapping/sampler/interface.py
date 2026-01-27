"""Interface for samplers."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

from torch import Tensor

from MIR.models.SITReg.composable_mapping.mappable_tensor.mappable_tensor import MappableTensor

if TYPE_CHECKING:
    from MIR.models.SITReg.composable_mapping.coordinate_system import CoordinateSystem


class LimitDirection:
    """Direction of a limit.

    Arguments:
        direction: Direction of the limit. Can be one of "left", "right", or
            "average".
    """

    def __init__(self, direction: str) -> None:
        if direction not in ["left", "right", "average"]:
            raise ValueError("Invalid direction")
        self.direction = direction

    @classmethod
    def left(cls) -> "LimitDirection":
        """Left limit direction."""
        return cls("left")

    @classmethod
    def right(cls) -> "LimitDirection":
        """Right limit direction."""
        return cls("right")

    @classmethod
    def average(cls) -> "LimitDirection":
        """Average of left and right limit directions."""
        return cls("average")

    def for_all_spatial_dims(self) -> Callable[[int], "LimitDirection"]:
        """Obtain callable with spatial dimension as input, and the
        limit direction as output.

        Useful for creating a callable that returns the same limit direction
        for all spatial dimensions.
        """
        return _SameLimitDirectionForAllSpatialDims(limit_direction=self)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, LimitDirection):
            return False
        return self.direction == value.direction

    def __repr__(self) -> str:
        return f"LimitDirection({self.direction})"


class _SameLimitDirectionForAllSpatialDims:
    def __init__(self, limit_direction: "LimitDirection") -> None:
        self.limit_direction = limit_direction

    def __call__(self, _: int) -> "LimitDirection":
        return self.limit_direction


class DataFormat:
    """Defines data format for sampled volumes.

    Arguments:
        representation: Representation of the data. Can be one of "coordinates"
            or "displacements".
        coordinate_type: Type of the coordinates. Can be one of "world" or
            "voxel".
    """

    def __init__(self, representation: str, coordinate_type: str) -> None:
        if representation not in ["coordinates", "displacements"]:
            raise ValueError("Invalid format")
        if coordinate_type not in ["world", "voxel"]:
            raise ValueError("Invalid coordinate type")
        self.representation = representation
        self.coordinate_type = coordinate_type

    @classmethod
    def voxel_displacements(cls) -> "DataFormat":
        """Voxel displacements data format."""
        return cls(representation="displacements", coordinate_type="voxel")

    @classmethod
    def world_displacements(cls) -> "DataFormat":
        """World displacements data format."""
        return cls(representation="displacements", coordinate_type="world")

    @classmethod
    def voxel_coordinates(cls) -> "DataFormat":
        """Voxel coordinates data format."""
        return cls(representation="coordinates", coordinate_type="voxel")

    @classmethod
    def world_coordinates(cls) -> "DataFormat":
        """World coordinates data format."""
        return cls(representation="coordinates", coordinate_type="world")

    def __repr__(self) -> str:
        return (
            f"DataFormat(representation={self.representation}, "
            f"coordinate_type={self.coordinate_type})"
        )


class ISampler(ABC):
    """Samples values on regular grid in voxel coordinates."""

    @abstractmethod
    def __call__(self, volume: MappableTensor, coordinates: MappableTensor) -> MappableTensor:
        """Sample the volume at coordinates.

        Args:
            volume: Volume to be sampled over spatial dimensions.
            coordinates: Coordinates in voxel coordinates.

        Returns:
            Sampled volume.
        """

    @abstractmethod
    def derivative(
        self,
        spatial_dim: int,
    ) -> "ISampler":
        """Obtain sampler for sampling derivatives corresponding to the current sampler.

        Args:
            spatial_dim: Spatial dimension along which to compute the derivative.

        Returns:
            Sampler for sampling derivatives.
        """

    @abstractmethod
    def inverse(
        self,
        coordinate_system: "CoordinateSystem",
        data_format: DataFormat,
        arguments: Optional[Mapping[str, Any]] = None,
    ) -> "ISampler":
        """Obtain sampler for sampling inverse values corresponding to the
        current sampler, if available.

        Args:
            coordinate_system: Coordinate system of the mapping.
            data_format: Data format of the sampled volume.
            arguments: Additional arguments for the inverse.

        Returns:
            Sampler for sampling inverse values.
        """

    @abstractmethod
    def sample_values(
        self,
        volume: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        """Sample values at spatial locations.

        Args:
            volume: Volume to sample over spatial dimensions
                with shape (*batch_shape, *channels_shape, *spatial_shape)
            coordinates: Coordinates in voxel coordinates with shape
                (*batch_shape, n_spatial_dims, *target_shape).

        Returns:
            Sampled values with shape (*batch_shape, *channels_shape, *target_shape).
        """

    @abstractmethod
    def sample_mask(
        self,
        mask: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        """Sample mask at spatial locations.

        Args:
            mask: Mask to sample over spatial dimensions
                with shape (*batch_shape, *(1,) * n_channel_dims, *spatial_shape).
            coordinates: Coordinates in voxel coordinates with shape
                (*batch_shape, n_spatial_dims, *target_shape).
        """
