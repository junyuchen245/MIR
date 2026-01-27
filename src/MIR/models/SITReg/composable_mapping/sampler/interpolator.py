"""Interpolating samplers."""

from typing import Callable

import torch.nn
from torch import Tensor
from torch import bool as torch_bool
from torch import cat
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import linspace, ones, ones_like, stack, zeros

from MIR.models.SITReg.composable_mapping.util import (
    get_batch_shape,
    get_channels_shape,
    get_n_channel_dims,
    get_spatial_shape,
    split_shape,
)

from .interpolate import interpolate
from .separable_sampler import PiecewiseKernelDefinition, SeparableSampler


class LinearKernel(PiecewiseKernelDefinition):
    """Kernel for linear interpolation."""

    def is_interpolating_kernel(self, spatial_dim: int) -> bool:
        return True

    def edge_continuity_schedule(self, spatial_dim: int, device: torch_device) -> Tensor:
        return stack(
            [
                ones(3, device=device, dtype=torch_bool),  # Original
                zeros(3, device=device, dtype=torch_bool),  # First derivative
                ones(3, device=device, dtype=torch_bool),  # Second derivative and beyond
            ],
            dim=0,
        )

    def piece_edges(self, spatial_dim: int, dtype: torch_dtype, device: torch_device) -> Tensor:
        return linspace(-1, 1, 3, dtype=dtype, device=device)

    def evaluate(self, spatial_dim: int, coordinates: Tensor) -> Tensor:
        return stack(
            [
                1 + coordinates[0, :],
                1 - coordinates[1, :],
            ],
            dim=0,
        )


class LinearInterpolator(SeparableSampler):
    """Linear interpolation in voxel coordinates

    Arguments:
        second_order_differentiable: If `True`, the interpolation is differentiable up to the second order.
            if `False`, the interpolation is differentiable up to the first order.
        **kwargs: See `.separable_sampler.SeparableSampler`.
    """

    def __init__(self, second_order_differentiable: bool = False, **kwargs) -> None:
        super().__init__(kernel=LinearKernel(), **kwargs)
        self._second_order_differentiable = second_order_differentiable

    def sample_values(
        self,
        volume: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        return interpolate(
            volume,
            coordinates,
            mode="bilinear",
            padding_mode=self._extrapolation_mode,
            second_order_differentiable=self._second_order_differentiable,
        )

    def sample_mask(
        self,
        mask: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        interpolated_mask = interpolate(
            volume=mask.to(coordinates.dtype),
            grid=coordinates,
            mode="bilinear",
            padding_mode="zeros",
        )
        return interpolated_mask >= 1 - self._mask_tol


class NearestKernel(PiecewiseKernelDefinition):
    """Kernel for nearest neighbour interpolation."""

    def is_interpolating_kernel(self, spatial_dim: int) -> bool:
        return True

    def edge_continuity_schedule(self, spatial_dim: int, device: torch_device) -> Tensor:
        return stack(
            [
                zeros(2, device=device, dtype=torch_bool),  # Original
                ones(2, device=device, dtype=torch_bool),  # First derivative and beyond
            ],
            dim=0,
        )

    def piece_edges(self, spatial_dim: int, dtype: torch_dtype, device: torch_device) -> Tensor:
        return linspace(-0.5, 0.5, 2, dtype=dtype, device=device)

    def evaluate(self, spatial_dim: int, coordinates: Tensor) -> Tensor:
        return ones_like(coordinates)


class NearestInterpolator(SeparableSampler):
    """Nearest neighbour interpolation in voxel coordinates.

    For the arguments see `.separable_sampler.SeparableSampler`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            kernel=NearestKernel(),
            **kwargs,
        )

    def sample_values(
        self,
        volume: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        return interpolate(
            volume,
            coordinates,
            mode="nearest",
            padding_mode=self._extrapolation_mode,
        )

    def sample_mask(
        self,
        mask: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        return interpolate(
            volume=mask.to(coordinates.dtype),
            grid=coordinates,
            mode="nearest",
            padding_mode="zeros",
        ).to(mask.dtype)


class BicubicKernel(PiecewiseKernelDefinition):
    """Kernel for bicubic interpolation."""

    def is_interpolating_kernel(self, spatial_dim: int) -> bool:
        return True

    def edge_continuity_schedule(self, spatial_dim: int, device: torch_device) -> Tensor:
        return cat(
            [
                ones((2, 5), device=device, dtype=torch_bool),  # Original and first derivative
                cat(  # Second derivative
                    (
                        zeros((1, 1), device=device, dtype=torch_bool),
                        ones((1, 3), device=device, dtype=torch_bool),
                        zeros((1, 1), device=device, dtype=torch_bool),
                    ),
                    dim=1,
                ),
                zeros((1, 5), device=device, dtype=torch_bool),  # Third derivative
                ones((1, 5), device=device, dtype=torch_bool),  # Fourth derivative and beyond
            ],
            dim=0,
        )

    def piece_edges(self, spatial_dim: int, dtype: torch_dtype, device: torch_device) -> Tensor:
        return linspace(-2.0, 2.0, 5, dtype=dtype, device=device)

    def evaluate(self, spatial_dim: int, coordinates: Tensor) -> Tensor:
        alpha = -0.75
        return stack(
            [
                alpha
                * (
                    -coordinates[0, :] ** 3 - 5 * coordinates[0, :] ** 2 - 8 * coordinates[0, :] - 4
                ),
                -(alpha + 2) * coordinates[1, :] ** 3 - (alpha + 3) * coordinates[1, :] ** 2 + 1,
                (alpha + 2) * coordinates[2, :] ** 3 - (alpha + 3) * coordinates[2, :] ** 2 + 1,
                alpha
                * (coordinates[3, :] ** 3 - 5 * coordinates[3, :] ** 2 + 8 * coordinates[3, :] - 4),
            ],
            dim=0,
        )


class BicubicInterpolator(SeparableSampler):
    """Bicubic interpolation in voxel coordinates.

    For the arguments see `.separable_sampler.SeparableSampler`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            kernel=BicubicKernel(),
            **kwargs,
        )

    def sample_values(
        self,
        volume: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        return interpolate(
            volume,
            coordinates,
            mode="bicubic",
            padding_mode=self._extrapolation_mode,
        )

    def sample_mask(
        self,
        mask: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        n_spatial_dims = get_channels_shape(coordinates.shape, n_channel_dims=1)[0]
        n_channel_dims = get_n_channel_dims(mask.ndim, n_spatial_dims)
        batch_shape, channels_shape, spatial_shape = split_shape(
            mask.shape, n_channel_dims=n_channel_dims
        )
        mask = mask.view(batch_shape + (1,) + spatial_shape).to(coordinates.dtype)
        mask = _avg_pool_nd_function(n_spatial_dims)(mask, kernel_size=3, stride=1, padding=1) >= 1
        interpolated_mask = interpolate(
            volume=mask.to(coordinates.dtype),
            grid=coordinates,
            mode="bilinear",
            padding_mode="zeros",
        )
        return (
            interpolated_mask.view(
                get_batch_shape(interpolated_mask.shape, n_channel_dims=1)
                + channels_shape
                + get_spatial_shape(interpolated_mask.shape, n_channel_dims=1)
            )
            >= 1 - self._mask_tol
        )


def _avg_pool_nd_function(n_dims: int) -> Callable[..., Tensor]:
    return getattr(torch.nn.functional, f"avg_pool{n_dims}d")
