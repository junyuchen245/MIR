"""Grid sample implementation that allows for second order differentiation."""

import logging

from torch import Tensor

from .cuda_gridsample import grid_sample_2d as cuda_grid_sample_2d
from .cuda_gridsample import grid_sample_3d as cuda_grid_sample_3d
from .naive_gridsample import grid_sample_2d as naive_grid_sample_2d
from .naive_gridsample import grid_sample_3d as naive_grid_sample_3d

logger = logging.getLogger(__name__)


def grid_sample(
    input: Tensor,
    grid: Tensor,
    align_corners: bool = True,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> Tensor:
    """Grid sample with second order gradients."""
    if input.device.type == "cuda":
        if mode != "bilinear":
            raise ValueError("Only bilinear interpolation supports second order gradients")
        if padding_mode == "reflection":
            raise ValueError("Reflection padding is not supported for second order gradients")
        if grid.shape[-1] == 2:
            return cuda_grid_sample_2d(
                input, grid, align_corners=align_corners, padding_mode=padding_mode
            )
        if grid.shape[-1] == 3:
            return cuda_grid_sample_3d(
                input, grid, align_corners=align_corners, padding_mode=padding_mode
            )
        raise ValueError("Only 2D and 3D grids are supported")
    if mode != "bilinear":
        raise ValueError("Only bilinear interpolation supports second order gradients")
    if padding_mode != "border":
        raise ValueError("Only border padding is supported for second order gradients on CPU.")
    if not align_corners:
        raise ValueError("Only align_corners=True is supported for second order gradients on CPU.")
    logger.warning(
        "Using naive grid sample implementation for second order gradients "
        "on CPU. Consider using the CUDA implementation for better performance."
    )
    if grid.shape[-1] == 2:
        return naive_grid_sample_2d(input, grid)
    if grid.shape[-1] == 3:
        return naive_grid_sample_3d(input, grid)
    raise ValueError("Only 2D and 3D grids are supported")
