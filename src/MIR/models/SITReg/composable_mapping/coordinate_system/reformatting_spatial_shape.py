"""Target spatial size of reformatting."""

from abc import ABC, abstractmethod
from typing import Callable, Tuple, Union

from MIR.models.SITReg.composable_mapping.interface import Number


def _define_bivariate_shape_operator(operator: Callable[[int, int], int]) -> Tuple[
    Callable[
        ["ReformattingSpatialShape", Union["ReformattingSpatialShape", int]],
        "ReformattingSpatialShape",
    ],
    Callable[
        ["ReformattingSpatialShape", Union["ReformattingSpatialShape", int]],
        "ReformattingSpatialShape",
    ],
]:
    def _left_operator(
        self: Union["ReformattingSpatialShape", int],
        other: Union["ReformattingSpatialShape", int],
    ) -> "ReformattingSpatialShape":
        if isinstance(self, int):
            self = _ConstantShape(self)
        if isinstance(other, int):
            other = _ConstantShape(other)
        if isinstance(other, ReformattingSpatialShape):
            return _BivariateShapeOperator(self, other, operator)
        return NotImplemented

    def _right_operator(
        self: "ReformattingSpatialShape",
        other: Union["ReformattingSpatialShape", int],
    ) -> "ReformattingSpatialShape":
        return _left_operator(self=other, other=self)

    return _left_operator, _right_operator


def _define_univariate_shape_operator(
    operator: Callable[[int], int]
) -> Callable[["ReformattingSpatialShape"], "ReformattingSpatialShape"]:
    def _operator(self: "ReformattingSpatialShape") -> "ReformattingSpatialShape":
        return _UnivariateShapeOperator(self, operator)

    return _operator


class ReformattingSpatialShape(ABC):
    """Instances of this class can be used as target spatial shapes for reformatting
    coordinate systems.

    Allows to define how the spatial shape of the grid should be reformatted based on
    the original spatial shape and the downsampling factor.

    For ease of use, the class implements the basic arithmetic operators to allow for
    easy manipulation of the spatial shape.
    """

    @abstractmethod
    def target_size(self, original_size: int, downsampling_factor: float) -> int:
        """Return a target size given an original size and a downsampling factor.

        Args:
            original_size: Original size of the dimension.
            downsampling_factor: Downsampling factor of the dimension during reformatting.

        Returns:
            Target size of the dimension.
        """

    __add__, __radd__ = _define_bivariate_shape_operator(lambda x, y: x + y)
    __sub__, __rsub__ = _define_bivariate_shape_operator(lambda x, y: x - y)
    __mul__, __rmul__ = _define_bivariate_shape_operator(lambda x, y: x * y)
    __floordiv__, __rfloordiv__ = _define_bivariate_shape_operator(lambda x, y: x // y)
    __pow__, __rpow__ = _define_bivariate_shape_operator(lambda x, y: x**y)
    __abs__ = _define_univariate_shape_operator(abs)
    __neg__ = _define_univariate_shape_operator(lambda x: -x)


class _BivariateShapeOperator(ReformattingSpatialShape):
    def __init__(
        self,
        shape_1: ReformattingSpatialShape,
        shape_2: ReformattingSpatialShape,
        operator: Callable[[int, int], int],
    ) -> None:
        self._shape_1 = shape_1
        self._shape_2 = shape_2
        self._operator = operator

    def target_size(self, original_size: int, downsampling_factor: float) -> int:
        return self._operator(
            self._shape_1.target_size(original_size, downsampling_factor),
            self._shape_2.target_size(original_size, downsampling_factor),
        )


class _UnivariateShapeOperator(ReformattingSpatialShape):
    def __init__(
        self,
        shape: ReformattingSpatialShape,
        operator: Callable[[int], int],
    ) -> None:
        self._shape = shape
        self._operator = operator

    def target_size(self, original_size: int, downsampling_factor: float) -> int:
        return self._operator(self._shape.target_size(original_size, downsampling_factor))


class _ConstantShape(ReformattingSpatialShape):
    def __init__(self, size: int) -> None:
        self._size = size

    def target_size(self, original_size: int, downsampling_factor: float) -> int:
        return self._size


class OriginalShape(ReformattingSpatialShape):
    """Spatial shape of the original grid."""

    def target_size(self, original_size: int, downsampling_factor: float) -> int:
        return original_size


def _ceildiv(denominator: Number, numerator: Number) -> Number:
    return -(denominator // -numerator)


class OriginalFOV(ReformattingSpatialShape):
    """Spatial shape such that the size of the field of view is the same as with
    the original grid.

    E.g. if the original grid has a size of 10 and the downsampling factor is 2,
    the target size will be 5 (with fov_convention == "full_voxels").

    Arguments:
        fitting_method: Method for fitting the field of view size, either "round",
            "floor", or "ceil".
        fov_convention: Convention for defining the field of view, either "full_voxels"
            or "voxel_centers". If voxels are seens as cubes with the value at the
            center, the convention "full voxels" includes the full cubes in the field
            of view, while the convention "voxel_centers" includes only the centers.
            The latter results in a field of view that is one voxel smaller in each
            dimension. Similar to the align_corners option in
            torch.nn.functional.grid_sample
    """

    DIVISION_FUNCTIONS = {
        "round": lambda x, y: int(round(x / y)),
        "floor": lambda x, y: int(x // y),
        "ceil": lambda x, y: int(_ceildiv(x, y)),
    }

    def __init__(self, fitting_method: str = "round", fov_convention: str = "full_voxels") -> None:
        if fitting_method not in ("round", "floor", "ceil"):
            raise ValueError(f"Unknown fitting method ({fitting_method})")
        if fov_convention not in ("full_voxels", "voxel_centers"):
            raise ValueError(f"Unknown fov convention ({fov_convention}).")
        self._division_function = self.DIVISION_FUNCTIONS[fitting_method]
        self._fov_convention = fov_convention

    def target_size(self, original_size: int, downsampling_factor: float) -> int:
        if self._fov_convention == "full_voxels":
            return self._division_function(original_size, downsampling_factor)
        elif self._fov_convention == "voxel_centers":
            return self._division_function(original_size - 1, downsampling_factor) + 1
        raise ValueError(f"Unknown fov convention ({self._fov_convention}).")
