"""Interpolation class wrappers"""

from torch import Tensor

from ..interface import Interpolator
from .algorithm import interpolate


class LinearInterpolator(Interpolator):
    """Linear interpolation in voxel coordinates"""

    def __init__(self, padding_mode: str = "border") -> None:
        self._padding_mode = padding_mode

    def __call__(self, volume: Tensor, coordinates: Tensor) -> Tensor:
        return interpolate(
            volume=volume,
            grid=coordinates,
            mode="bilinear",
            padding_mode=self._padding_mode,
        )


class NearestInterpolator(Interpolator):
    """Nearest neighbour interpolation in voxel coordinates"""

    def __init__(self, padding_mode: str = "border") -> None:
        self._padding_mode = padding_mode

    def __call__(self, volume: Tensor, coordinates: Tensor) -> Tensor:
        return interpolate(
            volume=volume,
            grid=coordinates,
            mode="nearest",
            padding_mode=self._padding_mode,
        )


class BicubicInterpolator(Interpolator):
    """Bicubic interpolation in voxel coordinates"""

    def __init__(self, padding_mode: str = "border") -> None:
        self._padding_mode = padding_mode

    def __call__(self, volume: Tensor, coordinates: Tensor) -> Tensor:
        return interpolate(
            volume=volume,
            grid=coordinates,
            mode="bicubic",
            padding_mode=self._padding_mode,
        )
