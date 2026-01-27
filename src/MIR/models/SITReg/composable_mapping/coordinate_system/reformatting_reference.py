"""Reformatting reference point aligned between original and reformatted
coordinates."""

# pylint:disable=too-many-lines
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Union

from MIR.models.SITReg.composable_mapping.interface import Number


def _define_bivariate_reference_operator(operator: Callable[[Number, Number], Number]) -> Tuple[
    Callable[
        ["ReformattingReference", Union["ReformattingReference", Number]],
        "ReformattingReference",
    ],
    Callable[
        ["ReformattingReference", Union["ReformattingReference", Number]],
        "ReformattingReference",
    ],
]:
    def _left_operator(
        self: Union["ReformattingReference", Number],
        other: Union["ReformattingReference", Number],
    ) -> "ReformattingReference":
        if isinstance(self, (float, int)):
            self = _ConstantReference(self)
        if isinstance(other, (float, int)):
            other = _ConstantReference(other)
        if isinstance(other, ReformattingReference):
            return _BivariateReferenceOperator(self, other, operator)
        return NotImplemented

    def _right_operator(
        self: "ReformattingReference",
        other: Union["ReformattingReference", Number],
    ) -> "ReformattingReference":
        return _left_operator(self=other, other=self)

    return _left_operator, _right_operator


def _define_univariate_reference_operator(
    operator: Callable[[Number], Number]
) -> Callable[["ReformattingReference"], "ReformattingReference"]:
    def _operator(self: "ReformattingReference") -> "ReformattingReference":
        return _UnivariateReferenceOperator(self, operator)

    return _operator


class ReformattingReference(ABC):
    """Reference point which will be aligned between the
    original and the reformatted coordinates.

    Allows to define the reference point for reformatting based on the
    spatial size of the grid.

    For ease of use, the class implements the basic arithmetic operators
    to allow for easy manipulation of the reference point.
    """

    @abstractmethod
    def get_voxel_coordinate(self, size: int) -> Number:
        """Get a voxel coordinate corresponding to the reference position with
        a given dimension size.

        Args:
            size: Size of the dimension

        Returns:
            Voxel coordinate corresponding to the reference position along the dimension.
        """

    __add__, __radd__ = _define_bivariate_reference_operator(lambda x, y: x + y)
    __sub__, __rsub__ = _define_bivariate_reference_operator(lambda x, y: x - y)
    __mul__, __rmul__ = _define_bivariate_reference_operator(lambda x, y: x * y)
    __floordiv__, __rfloordiv__ = _define_bivariate_reference_operator(lambda x, y: x // y)
    __truediv__, __rtruediv__ = _define_bivariate_reference_operator(lambda x, y: x / y)
    __pow__, __rpow__ = _define_bivariate_reference_operator(lambda x, y: x**y)
    __abs__ = _define_univariate_reference_operator(abs)
    __neg__ = _define_univariate_reference_operator(lambda x: -x)


class _BivariateReferenceOperator(ReformattingReference):
    def __init__(
        self,
        shape_1: ReformattingReference,
        shape_2: ReformattingReference,
        operator: Callable[[Number, Number], Number],
    ) -> None:
        self._shape_1 = shape_1
        self._shape_2 = shape_2
        self._operator = operator

    def get_voxel_coordinate(self, size: int) -> Number:
        return self._operator(
            self._shape_1.get_voxel_coordinate(size),
            self._shape_2.get_voxel_coordinate(size),
        )


class _UnivariateReferenceOperator(ReformattingReference):
    def __init__(
        self,
        shape: ReformattingReference,
        operator: Callable[[Number], Number],
    ) -> None:
        self._shape = shape
        self._operator = operator

    def get_voxel_coordinate(self, size: int) -> Number:
        return self._operator(self._shape.get_voxel_coordinate(size))


class _ConstantReference(ReformattingReference):
    def __init__(self, point: Number) -> None:
        self._point = point

    def get_voxel_coordinate(self, size: int) -> Number:
        return self._point


class Start(ReformattingReference):
    """Reference point at the start of the dimension.

    Arguments:
        fov_convention: Convention for defining the field of view, either "full_voxels"
            or "voxel_centers". If voxels are seens as cubes with the value at the
            center, the convention "full voxels" includes the full cubes in the field
            of view, while the convention "voxel_centers" includes only the centers.
            The latter results in a field of view that is one voxel smaller in each
            dimension. Similar to the align_corners option in
            torch.nn.functional.grid_sample
    """

    def __init__(
        self,
        fov_convention: str = "full_voxels",
    ) -> None:
        self._fov_convention = fov_convention

    def get_voxel_coordinate(self, size: int) -> float:
        if self._fov_convention == "full_voxels":
            return -0.5
        if self._fov_convention == "voxel_centers":
            return 0
        raise ValueError(f"Unknown fov convention: {self._fov_convention}")


class End(ReformattingReference):
    """Reference point at the end of the dimension.

    Arguments:
        fov_convention: Convention for defining the field of view, either "full_voxels"
            or "voxel_centers". If voxels are seens as cubes with the value at the
            center, the convention "full voxels" includes the full cubes in the field
            of view, while the convention "voxel_centers" includes only the centers.
            The latter results in a field of view that is one voxel smaller in each
            dimension. Similar to the align_corners option in
            torch.nn.functional.grid_sample
    """

    def __init__(
        self,
        fov_convention: str = "full_voxels",
    ) -> None:
        self._fov_convention = fov_convention

    def get_voxel_coordinate(self, size: int) -> float:
        if self._fov_convention == "full_voxels":
            return size - 0.5
        if self._fov_convention == "voxel_centers":
            return size - 1
        raise ValueError(f"Unknown fov convention: {self._fov_convention}")


class Center(ReformattingReference):
    """Reference point at the center of the dimension."""

    def get_voxel_coordinate(self, size: int) -> float:
        return (size - 1) / 2
