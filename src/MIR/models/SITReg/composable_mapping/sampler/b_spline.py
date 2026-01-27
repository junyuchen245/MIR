"""B-spline samplers"""

from torch import Tensor
from torch import bool as torch_bool
from torch import cat
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import linspace, ones, stack, zeros

from .separable_sampler import PiecewiseKernelDefinition, SeparableSampler


class CubicSplineKernel(PiecewiseKernelDefinition):
    """Kernel for Cubic splines."""

    def is_interpolating_kernel(self, spatial_dim: int) -> bool:
        return False

    def edge_continuity_schedule(self, spatial_dim: int, device: torch_device) -> Tensor:
        return cat(
            [
                ones(  # Original, and first and second derivative
                    (3, 5), device=device, dtype=torch_bool
                ),
                zeros((1, 5), device=device, dtype=torch_bool),  # Third derivative
                ones((1, 5), device=device, dtype=torch_bool),  # Fourth derivative and beyond
            ],
            dim=0,
        )

    def piece_edges(self, spatial_dim: int, dtype: torch_dtype, device: torch_device) -> Tensor:
        return linspace(-2.0, 2.0, 5, dtype=dtype, device=device)

    def evaluate(self, spatial_dim: int, coordinates: Tensor) -> Tensor:
        return stack(
            [
                -((-coordinates[0, :] - 2) ** 3) / 6,
                2 / 3 + (-0.5 * coordinates[1, :] - 1) * coordinates[1, :] ** 2,
                2 / 3 + (0.5 * coordinates[2, :] - 1) * coordinates[2, :] ** 2,
                -((coordinates[3, :] - 2) ** 3) / 6,
            ],
            dim=0,
        )


class CubicSplineSampler(SeparableSampler):
    """Sampling based on regularly spaced cubic spline control points in voxel
    coordinates

    Arguments:
        prefilter: Whether to prefilter the volume before sampling. Currently
            not implemented.

    For the other arguments see `.separable_sampler.SeparableSampler`.
    """

    def __init__(
        self,
        prefilter: bool = False,
        **kwargs,
    ) -> None:
        if prefilter:
            raise NotImplementedError(
                "Prefiltering is currently not implemented. "
                "Contact the developers if you would want it included."
            )
        super().__init__(
            kernel=CubicSplineKernel(),
            **kwargs,
        )
