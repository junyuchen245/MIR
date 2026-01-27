"""Library for handling spatial coordinate transformations in PyTorch.

Developed originally for medical imaging, this library provides a set of classes
and functions for handling spatial coordinate transformations.

The most powerful feature of this library is the ability to easily compose
transformations lazily and resample them to different coordinate systems as well
as sampler classes for sampling volumes defined on regular grids such that the
optimal method (either slicing operation, convolution, or torch.grid_sample) is
used based on the sampling locations.

The main idea was to develop a library that allows handling of the coordinate
mappings as if they were mathematical functions, without losing much performance
compared to more manual implementation.

In this codebase tensor shapes are seen as consisting of three parts (in the
following order): batch dimension, channel dimensions and spatial dimensions.
The split is defined by providing the number of channel dimensions with the
assumption that there is at most one batch dimension. If the number of
dimensions equals the number of channel dimensions, the batch dimension is
assumed to be empty. Codebase implements custom broadcasting operations which
apply normal broadcasting separately to each split. This makes many operations
easier as one does not have to worry about unsqueezing correct number of
dimensions to apply the same operation e.g. over all spatial and batch
dimensions.
"""

from . import affine_transformation, util
from .affine_mapping import Affine, affine, diagonal_affine
from MIR.models.SITReg.composable_mapping.composable_mapping import (
    ComposableMapping,
    GridComposableMapping,
    ICoordinateSystemContainer,
    Identity,
    SamplableVolume,
    concatenate_mappings,
    samplable_volume,
    stack_mappings,
)
from .constant import Constant
from .coordinate_system import (
    Center,
    CoordinateSystem,
    End,
    OriginalFOV,
    OriginalShape,
    ReformattingReference,
    ReformattingSpatialShape,
    Start,
)
from .derivative import (
    estimate_coordinate_mapping_spatial_derivatives,
    estimate_spatial_jacobian_matrices,
)
from .factory import from_file
from .interface import Number
from .mappable_tensor import (
    MappableTensor,
    concatenate_mappable_tensors,
    mappable,
    stack_mappable_tensors,
    voxel_grid,
)
from .mask import ClearMask, RectangleMask
from .sampler import (
    BicubicInterpolator,
    CubicSplineSampler,
    DataFormat,
    ISampler,
    LimitDirection,
    LinearInterpolator,
    NearestInterpolator,
    PiecewiseKernelDefinition,
    ScalingAndSquaring,
    SeparableSampler,
    clear_default_sampler,
    clear_sampling_cache,
    default_sampler,
    get_default_sampler,
    get_sampler,
    no_sampling_cache,
    sampling_cache,
    set_default_sampler,
)
from .visualization import (
    GridVisualizationArguments,
    ImageVisualizationArguments,
    visualize_as_deformed_grid,
    visualize_as_image,
    visualize_grid,
    visualize_image,
    visualize_to_as_deformed_grid,
    visualize_to_as_image,
)

__all__ = [
    "Affine",
    "SeparableSampler",
    "BicubicInterpolator",
    "ClearMask",
    "ComposableMapping",
    "Constant",
    "CoordinateSystem",
    "CubicSplineSampler",
    "DataFormat",
    "GridComposableMapping",
    "GridVisualizationArguments",
    "Identity",
    "ICoordinateSystemContainer",
    "ISampler",
    "ImageVisualizationArguments",
    "LimitDirection",
    "LinearInterpolator",
    "MappableTensor",
    "NearestInterpolator",
    "Number",
    "OriginalFOV",
    "OriginalShape",
    "PiecewiseKernelDefinition",
    "ReformattingReference",
    "ReformattingSpatialShape",
    "RectangleMask",
    "SamplableVolume",
    "ScalingAndSquaring",
    "Start",
    "Center",
    "End",
    "affine",
    "affine_transformation",
    "clear_sampling_cache",
    "clear_default_sampler",
    "concatenate_mappings",
    "concatenate_mappable_tensors",
    "default_sampler",
    "diagonal_affine",
    "estimate_coordinate_mapping_spatial_derivatives",
    "estimate_spatial_jacobian_matrices",
    "from_file",
    "get_default_sampler",
    "get_sampler",
    "mappable",
    "no_sampling_cache",
    "sampling_cache",
    "samplable_volume",
    "set_default_sampler",
    "stack_mappings",
    "stack_mappable_tensors",
    "util",
    "voxel_grid",
    "visualize_as_deformed_grid",
    "visualize_as_image",
    "visualize_grid",
    "visualize_image",
    "visualize_to_as_deformed_grid",
    "visualize_to_as_image",
]
