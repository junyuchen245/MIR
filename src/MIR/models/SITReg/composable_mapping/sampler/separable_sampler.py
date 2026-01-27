"""Sampling with separable kernels"""

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import ones, zeros_like
from torch.autograd.functional import vjp
from torch.nn import functional as torch_functional

from MIR.models.SITReg.composable_mapping.mappable_tensor import MappableTensor, mappable
from MIR.models.SITReg.composable_mapping.util import (
    combine_optional_masks,
    crop_and_then_pad_spatial,
    get_spatial_dims,
    includes_padding,
    is_croppable_first,
    split_shape,
)

from .convolution_sampling import (
    apply_flips_and_permutation_to_volume,
    calculate_convolutional_sampling_parameters,
)
from .interface import DataFormat, ISampler, LimitDirection
from .inverse import FixedPointInverseSampler
from .sampling_cache import get_cached_sampling_parameters

if TYPE_CHECKING:
    from MIR.models.SITReg.composable_mapping.coordinate_system import CoordinateSystem


_ConvParametersType = Optional[
    Tuple[
        Sequence[Sequence[int]],
        Sequence[Optional[Tensor]],
        Sequence[bool],
        Sequence[int],
        Sequence[int],
        Sequence[Tuple[int, int]],
        Sequence[Tuple[int, int]],
        List[int],
        List[int],
    ]
]
from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import ones, zeros_like
from torch.autograd.functional import vjp

from .interface import ISampler, LimitDirection


class PiecewiseKernelDefinition(ABC):
    """Piecewise kernel definition"""

    @abstractmethod
    def is_interpolating_kernel(self, spatial_dim: int) -> bool:
        """Is the kernel is interpolating.

        Args:
            spatial_dim: Spatial dimension for which to obtain the information.

        Returns:
            Whether the kernel is interpolating over the specified spatial dimension.
        """

    @abstractmethod
    def edge_continuity_schedule(self, spatial_dim: int, device: torch_device) -> Tensor:
        """Whether the edges are continuous between the pieces for the kernel, and
        its derivatives.

        Args:
            spatial_dim: Spatial dimension for which to obtain the information.
            device: Device for the edge continuity Tensor.

        Returns:
            Boolean Tensor with shape (any_size, n_edges) indicating whether the
            edges are continuous between the pieces for the kernel, and its
            derivatives. The first item in the second dimension is used for the
            kernel itself, and the remaining items are used for the derivatives.
            For derivatives after the last element, the last element is used.
        """

    def edge_continuity(self, spatial_dim: int, device: torch_device) -> Tensor:
        """Whether the edges are continuous between the pieces

        Args:
            spatial_dim: Spatial dimension for which to obtain the information.
            device: Device for the edge continuity Tensor.

        Returns:
            Whether the edges are continuous between the pieces for the kernel.
        """
        return self.edge_continuity_schedule(spatial_dim, device)[0]

    @abstractmethod
    def piece_edges(self, spatial_dim: int, dtype: torch_dtype, device: torch_device) -> Tensor:
        """Defines the edge points of the piecewise smooth kernel.

        Args:
            spatial_dim: Spatial dimension for which to obtain the edge points.
            dtype: Data type for the edge points.
            device: Device for the edge points.

        Returns:
            Edge points of the piecewise smooth kernel, Tensor with shape
            (n_edges,), and a boolean tensor indicating whether the edge is
            continuous.
        """

    @abstractmethod
    def evaluate(self, spatial_dim: int, coordinates: Tensor) -> Tensor:
        """Evaluate the interpolation kernel for the given coordinates.

        Function values produced need to be valid only for coordinates within
        the range of each piecewise smooth kernel. Each coordinate location
        for each piece should be computed independently.


        Args:
            spatial_dim: Spatial dimension for which to evaluate the kernel.
            coordinates: Coordinates with shape (n_pieces, n_coordinates)

        Returns:
            Interpolation kernel evaluated at the given coordinates for each of
            the piecewise smooth functions. Tensor with shape (n_pieces, n_coordinates).
        """

    def derivative(self, spatial_dim: int) -> "PiecewiseKernelDefinition":
        """Derivative of the piecewise kernel.

        Args:
            spatial_dim: Spatial dimension for which to obtain the derivative.

        Returns:
            Derivative of the piecewise kernel.
        """
        return PiecewiseKernelDerivative(self, spatial_dim)


class PiecewiseKernelDerivative(PiecewiseKernelDefinition):
    """Derivative of a piecewise kernel"""

    def __init__(self, kernel: PiecewiseKernelDefinition, spatial_dim: int):
        super().__init__()
        self._kernel = kernel
        self._spatial_dim = spatial_dim

    def is_interpolating_kernel(self, spatial_dim: int) -> bool:
        if spatial_dim == self._spatial_dim:
            return False
        return self._kernel.is_interpolating_kernel(spatial_dim)

    def edge_continuity_schedule(self, spatial_dim: int, device: torch_device) -> Tensor:
        schedule = self._kernel.edge_continuity_schedule(spatial_dim, device)
        if spatial_dim != self._spatial_dim or schedule.size(0) == 1:
            return schedule
        return self._kernel.edge_continuity_schedule(spatial_dim, device)[1:]

    def piece_edges(self, spatial_dim: int, dtype: torch_dtype, device: torch_device) -> Tensor:
        return self._kernel.piece_edges(spatial_dim, dtype, device)

    def evaluate(self, spatial_dim: int, coordinates: Tensor) -> Tensor:
        if spatial_dim == self._spatial_dim:
            return self._kernel_derivative(self._kernel.evaluate, spatial_dim, coordinates)
        return self._kernel.evaluate(spatial_dim, coordinates)

    def _kernel_derivative(
        self,
        kernel_function: Callable[[int, Tensor], Tensor],
        spatial_dim: int,
        coordinates: Tensor,
    ) -> Tensor:
        def partial_kernel_function(coordinates: Tensor) -> Tensor:
            return -kernel_function(spatial_dim, coordinates)

        _output, derivatives = vjp(
            partial_kernel_function,
            inputs=coordinates,
            v=ones(coordinates.shape, device=coordinates.device, dtype=coordinates.dtype),
            create_graph=coordinates.requires_grad,
        )
        return derivatives


class SeparableSampler(ISampler):
    """Sampler in voxel coordinates which can be implemented as a
    separable convolution

    Kernel is assumed to be defined as a piecewise smooth function. Note that
    the custom kernel allows only for convolution based sampling.

    Arguments:
        kernel: Piecewise kernel definition for convolution-based sampling.
        extrapolation_mode: Extrapolation mode for out-of-bound coordinates.
        mask_extrapolated_regions: Whether to mask extrapolated regions.
        conv_tol: Maximum allowed difference in coordinates
            for using convolution-based sampling (the difference might be upper
            bounded when doing the decision).
        mask_tol: Maximum allowed weight for masked regions in a
            sampled location to still consider it valid (non-masked).
        limit_tol: The setting allows for tolerance for being on the different side
            of discontinuous point than the limit direction for the limit to still
            be computed based on the other side. Needed for numerically stable use
            of limit directions. Additionally, if a kernel which goes to zero on
            bounds, the actual kernel bounds are shrunk by this tolerance to avoid
            having practically zero values at the bounds.
        limit_direction: Side of evaluation in discontinuous points (or in their
            neighborhood if limit_tol > 0).
    """

    def __init__(
        self,
        kernel: PiecewiseKernelDefinition,
        extrapolation_mode: str = "border",
        mask_extrapolated_regions: bool = True,
        conv_tol: float = 1e-3,
        mask_tol: float = 1e-3,
        limit_tol: float = 1e-3,
        limit_direction: Union[
            LimitDirection, Callable[[int], LimitDirection]
        ] = LimitDirection.left(),
    ) -> None:
        if extrapolation_mode not in ("zeros", "border", "reflection"):
            raise ValueError("Unknown extrapolation mode")
        self._extrapolation_mode = extrapolation_mode
        self._mask_extrapolated_regions = mask_extrapolated_regions
        self._conv_tol = conv_tol
        self._mask_tol = mask_tol
        self._limit_tol = limit_tol
        self._limit_direction = (
            limit_direction.for_all_spatial_dims()
            if isinstance(limit_direction, LimitDirection)
            else limit_direction
        )
        self._kernel = kernel

    def _evaluate_kernel(self, spatial_dim: int, coordinates: Tensor) -> Tensor:
        limit_direction = self._limit_direction(spatial_dim)
        output: Tensor = zeros_like(coordinates)
        edges = self._kernel.piece_edges(
            spatial_dim, dtype=coordinates.dtype, device=coordinates.device
        )
        edge_continuity = self._kernel.edge_continuity(spatial_dim, device=coordinates.device)
        piece_start = edges[:-1]
        piece_end = edges[1:]
        piece_start_continuity = edge_continuity[:-1]
        piece_end_continuity = edge_continuity[1:]
        start_tolerance = self._limit_tol * (~piece_start_continuity)
        end_tolerance = self._limit_tol * (~piece_end_continuity)
        if limit_direction == LimitDirection.left():
            coordinate_mask = (coordinates[None, :] >= (piece_start - start_tolerance)[:, None]) & (
                coordinates[None, :] < (piece_end - end_tolerance)[:, None]
            )
        elif limit_direction == LimitDirection.right():
            coordinate_mask = (coordinates[None, :] > (piece_start + start_tolerance)[:, None]) & (
                coordinates[None, :] <= (piece_end + end_tolerance)[:, None]
            )
        elif limit_direction == LimitDirection.average():
            coordinate_mask = (coordinates[None, :] > (piece_start + start_tolerance)[:, None]) & (
                coordinates[None, :] < (piece_end - end_tolerance)[:, None]
            )
        else:
            raise ValueError("Unknown limit direction")
        evaluated = self._kernel.evaluate(
            spatial_dim,
            coordinates[None, :].clamp(min=piece_start[:, None], max=piece_end[:, None]),
        )
        output = (evaluated * coordinate_mask).sum(dim=0)
        if limit_direction == LimitDirection.average():
            coordinate_mask_edge = (
                (coordinates[None, :] >= (piece_start - start_tolerance)[:, None])
                & (coordinates[None, :] <= (piece_start + start_tolerance)[:, None])
            ) | (
                (coordinates[None, :] >= (piece_end - end_tolerance)[:, None])
                & (coordinates[None, :] <= (piece_end + end_tolerance)[:, None])
            )
            output = output + (evaluated * coordinate_mask_edge / 2).sum(dim=0)
        return output

    def derivative(
        self,
        spatial_dim: int,
    ) -> "SeparableSampler":
        return SeparableSampler(
            kernel=self._kernel.derivative(spatial_dim),
            extrapolation_mode=self._extrapolation_mode,
            mask_extrapolated_regions=self._mask_extrapolated_regions,
            conv_tol=self._conv_tol,
            mask_tol=self._mask_tol,
            limit_tol=self._limit_tol,
            limit_direction=self._limit_direction,
        )

    def __call__(self, volume: MappableTensor, coordinates: MappableTensor) -> MappableTensor:
        if coordinates.n_channel_dims != 1:
            raise ValueError("Interpolation assumes single channel coordinates")
        if coordinates.channels_shape[0] != len(volume.spatial_shape):
            raise ValueError("Interpolation assumes same number of channels as spatial dims")
        interpolated = self._interpolate_conv(volume, coordinates)
        if interpolated is None:
            return self._interpolate_general(volume, coordinates)
        return interpolated

    @property
    def _padding_mode_and_value(self) -> Tuple[str, float]:
        return {
            "zeros": ("constant", 0.0),
            "border": ("replicate", 0.0),
            "reflection": ("reflect", 0.0),
        }[self._extrapolation_mode]

    @staticmethod
    def _build_joint_kernel(
        kernel_dims: Sequence[int], kernels_1d: Sequence[Optional[Tensor]]
    ) -> Optional[Tensor]:
        not_none_kernels = [kernel for kernel in kernels_1d if kernel is not None]
        if not not_none_kernels:
            return None
        generated_kernels_1d = [
            (
                ones(1, dtype=not_none_kernels[0].dtype, device=not_none_kernels[0].device)
                if kernel is None
                else kernel
            )
            for kernel in kernels_1d
        ]
        conv_kernel = generated_kernels_1d[kernel_dims[0]]
        for dim in kernel_dims[1:]:
            conv_kernel = conv_kernel[..., None] * generated_kernels_1d[dim]
        return conv_kernel

    def _obtain_conv_parameters(
        self,
        volume: MappableTensor,
        voxel_coordinates: MappableTensor,
    ) -> _ConvParametersType:
        if voxel_coordinates.displacements is not None:
            return None
        grid = voxel_coordinates.grid
        if grid is None:
            return None
        grid_affine_matrix = grid.affine_transformation.as_host_matrix()
        if grid_affine_matrix is None:
            return None
        kernel_support = []
        is_zero_on_bounds = []
        for spatial_dim in range(len(grid.spatial_shape)):
            piece_edges = self._kernel.piece_edges(
                spatial_dim, device=torch_device("cpu"), dtype=voxel_coordinates.dtype
            )
            edge_continuity = self._kernel.edge_continuity(
                spatial_dim,
                device=torch_device("cpu"),
            )
            kernel_support.append((piece_edges[0], piece_edges[-1]))
            is_zero_on_bounds.append((edge_continuity[0], edge_continuity[-1]))
        conv_parameters = calculate_convolutional_sampling_parameters(
            volume_spatial_shape=volume.spatial_shape,
            grid_spatial_shape=grid.spatial_shape,
            grid_affine_matrix=grid_affine_matrix,
            is_interpolating_kernel=[
                self._kernel.is_interpolating_kernel(spatial_dim)
                for spatial_dim in range(len(volume.spatial_shape))
            ],
            kernel_support=kernel_support,
            is_zero_on_bounds=is_zero_on_bounds,
            limit_direction=[
                self._limit_direction(spatial_dim).direction
                for spatial_dim in range(len(volume.spatial_shape))
            ],
            conv_tol=self._conv_tol,
            limit_tol=self._limit_tol,
            target_device=voxel_coordinates.device,
        )
        if conv_parameters is None:
            return None
        (
            conv_kernel_coordinates,
            conv_strides,
            conv_paddings,
            transposed_convolve,
            pre_pads_or_crops,
            post_pads_or_crops,
            inverse_spatial_permutation,
            flipped_spatial_dims,
        ) = conv_parameters
        conv_kernels = [
            (
                self._evaluate_kernel(
                    spatial_dim,
                    kernel_coordinates,
                )
                if kernel_coordinates is not None
                else None
            )
            for spatial_dim, kernel_coordinates in enumerate(conv_kernel_coordinates)
        ]
        padding_mode, _padding_value = self._padding_mode_and_value
        if not is_croppable_first(
            spatial_shape=volume.spatial_shape, pads_or_crops=pre_pads_or_crops, mode=padding_mode
        ):
            return None
        # Optimize to use either spatially separable convolutions or general
        # convolutions. This is currently a very simple heuristic that seems to
        # work somewhat well in practice.
        if all(
            conv_kernel is None or conv_kernel.size(0) <= 2 for conv_kernel in conv_kernels
        ) == 1 and not any(transposed_convolve):
            conv_kernel_dims = [list(range(len(volume.spatial_shape)))]
            conv_kernel_transposed = [transposed_convolve[0]]
        else:
            conv_kernel_dims = [[dim] for dim in range(len(volume.spatial_shape))]
            conv_kernel_transposed = transposed_convolve
        return (
            conv_kernel_dims,
            [
                self._build_joint_kernel(kernel_dims, conv_kernels)
                for kernel_dims in conv_kernel_dims
            ],
            conv_kernel_transposed,
            conv_strides,
            conv_paddings,
            pre_pads_or_crops,
            post_pads_or_crops,
            inverse_spatial_permutation,
            flipped_spatial_dims,
        )

    def _interpolate_conv(
        self,
        volume: MappableTensor,
        voxel_coordinates: MappableTensor,
    ) -> Optional[MappableTensor]:
        conv_parameters = get_cached_sampling_parameters(
            lambda: self._obtain_conv_parameters(volume, voxel_coordinates)
        )
        if conv_parameters is None:
            return None
        (
            conv_kernel_dims,
            conv_kernels,
            conv_kernel_transposed,
            conv_strides,
            conv_paddings,
            pre_pads_or_crops,
            post_pads_or_crops,
            inverse_spatial_permutation,
            flipped_spatial_dims,
        ) = conv_parameters
        padding_mode, padding_value = self._padding_mode_and_value

        values = volume.generate_values()
        interpolated_values = crop_and_then_pad_spatial(
            values,
            pads_or_crops=pre_pads_or_crops,
            mode=padding_mode,
            value=padding_value,
            n_channel_dims=volume.n_channel_dims,
        )
        interpolated_values = self._separable_conv(
            interpolated_values,
            kernels=conv_kernels,
            kernel_spatial_dims=conv_kernel_dims,
            kernel_transposed=conv_kernel_transposed,
            stride=conv_strides,
            padding=conv_paddings,
            n_channel_dims=volume.n_channel_dims,
        )
        interpolated_values = crop_and_then_pad_spatial(
            interpolated_values,
            pads_or_crops=post_pads_or_crops,
            mode=padding_mode,
            value=padding_value,
            n_channel_dims=volume.n_channel_dims,
        )
        interpolated_values = apply_flips_and_permutation_to_volume(
            interpolated_values,
            n_channel_dims=volume.n_channel_dims,
            spatial_permutation=inverse_spatial_permutation,
            flipped_spatial_dims=flipped_spatial_dims,
        )
        interpolated_mask: Optional[Tensor] = None
        if self._mask_extrapolated_regions:
            mask = volume.generate_mask(
                generate_missing_mask=includes_padding(pre_pads_or_crops),
                cast_mask=False,
            )
            if mask is not None:
                interpolated_mask = crop_and_then_pad_spatial(
                    mask,
                    pads_or_crops=pre_pads_or_crops,
                    mode="constant",
                    value=False,
                    n_channel_dims=volume.n_channel_dims,
                )
                interpolated_mask = (
                    self._separable_conv(
                        (~interpolated_mask).to(dtype=voxel_coordinates.dtype),
                        kernels=[
                            None if kernel is None else kernel.abs() for kernel in conv_kernels
                        ],
                        kernel_spatial_dims=conv_kernel_dims,
                        kernel_transposed=conv_kernel_transposed,
                        stride=conv_strides,
                        padding=conv_paddings,
                        n_channel_dims=volume.n_channel_dims,
                    )
                    <= self._mask_tol
                )
                interpolated_mask = crop_and_then_pad_spatial(
                    interpolated_mask,
                    pads_or_crops=post_pads_or_crops,
                    mode="constant",
                    value=False,
                    n_channel_dims=volume.n_channel_dims,
                )
                interpolated_mask = apply_flips_and_permutation_to_volume(
                    interpolated_mask,
                    n_channel_dims=volume.n_channel_dims,
                    spatial_permutation=inverse_spatial_permutation,
                    flipped_spatial_dims=flipped_spatial_dims,
                )
        return mappable(
            interpolated_values,
            combine_optional_masks(
                interpolated_mask,
                voxel_coordinates.generate_mask(generate_missing_mask=False, cast_mask=False),
                n_channel_dims=(volume.n_channel_dims, voxel_coordinates.n_channel_dims),
            ),
            n_channel_dims=volume.n_channel_dims,
        )

    def _separable_conv(
        self,
        volume: Tensor,
        kernels: Sequence[Optional[Tensor]],
        kernel_spatial_dims: Sequence[Sequence[int]],
        kernel_transposed: Sequence[bool],
        stride: Sequence[int],
        padding: Sequence[int],
        n_channel_dims: int,
    ) -> Tensor:
        n_spatial_dims = len(get_spatial_dims(volume.ndim, n_channel_dims))
        if n_spatial_dims != len(stride) or n_spatial_dims != len(padding):
            raise ValueError("Invalid number of strides, transposed, or paddings")
        if len(kernels) != len(kernel_spatial_dims) or len(kernels) != len(kernel_transposed):
            raise ValueError("Invalid number of kernels, kernel spatial dims, or kernel transposed")
        for spatial_dims, kernel, single_kernel_transposed in zip(
            kernel_spatial_dims, kernels, kernel_transposed
        ):
            if kernel is None or kernel.shape.numel() == 1 and not single_kernel_transposed:
                slicing_tuple: Tuple[slice, ...] = tuple()
                for dim in range(n_spatial_dims):
                    if dim in spatial_dims:
                        slicing_tuple += (slice(None, None, stride[dim]),)
                    else:
                        slicing_tuple += (slice(None),)
                volume = volume[(...,) + slicing_tuple]
                if kernel is not None:
                    volume = kernel * volume
            else:
                volume = self._conv_nd(
                    volume,
                    spatial_dims=spatial_dims,
                    kernel=kernel,
                    stride=[stride[dim] for dim in spatial_dims],
                    padding=[padding[dim] for dim in spatial_dims],
                    transposed=single_kernel_transposed,
                    n_channel_dims=n_channel_dims,
                )
        return volume

    @classmethod
    def _conv_nd(
        cls,
        volume: Tensor,
        spatial_dims: Sequence[int],
        kernel: Tensor,
        stride: Sequence[int],
        padding: Sequence[int],
        transposed: bool,
        n_channel_dims: int,
    ) -> Tensor:
        n_kernel_dims = kernel.ndim
        volume_spatial_dims = get_spatial_dims(volume.ndim, n_channel_dims)
        convolved_dims = [volume_spatial_dims[dim] for dim in spatial_dims]
        last_dims = list(range(-n_kernel_dims, 0))
        volume = volume.moveaxis(convolved_dims, last_dims)
        convolved_dims_excluded_shape = volume.shape[:-n_kernel_dims]
        volume = volume.reshape(-1, 1, *volume.shape[-n_kernel_dims:])
        conv_function = (
            cls._conv_nd_function(n_kernel_dims)
            if not transposed
            else cls._conv_transpose_nd_function(n_kernel_dims)
        )
        convolved = conv_function(  # pylint: disable=not-callable
            volume,
            kernel[None, None],
            bias=None,
            stride=stride,
            padding=padding,
        )
        convolved = convolved.reshape(
            *convolved_dims_excluded_shape, *convolved.shape[-n_kernel_dims:]
        )
        return convolved.moveaxis(last_dims, convolved_dims)

    @staticmethod
    def _conv_nd_function(n_dims: int) -> Callable[..., Tensor]:
        return getattr(torch_functional, f"conv{n_dims}d")

    @staticmethod
    def _conv_transpose_nd_function(n_dims: int) -> Callable[..., Tensor]:
        return getattr(torch_functional, f"conv_transpose{n_dims}d")

    def _interpolate_general(
        self, volume: MappableTensor, voxel_coordinates: MappableTensor
    ) -> MappableTensor:
        volume_values = volume.generate_values()
        coordinate_values, coordinate_mask = voxel_coordinates.generate(
            generate_missing_mask=False, cast_mask=False
        )
        interpolated_values = self.sample_values(volume_values, coordinate_values)
        if self._mask_extrapolated_regions:
            interpolated_mask: Optional[Tensor] = self.sample_mask(
                volume.generate_mask(generate_missing_mask=True, cast_mask=False),
                coordinate_values,
            )
        else:
            interpolated_mask = None
        return mappable(
            interpolated_values,
            combine_optional_masks(coordinate_mask, interpolated_mask),
            n_channel_dims=volume.n_channel_dims,
        )

    def sample_values(
        self,
        volume: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        raise NotImplementedError(
            "Sampling at arbitrary coordinates is not implemented for this sampler."
        )

    def sample_mask(
        self,
        mask: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        raise NotImplementedError(
            "Sampling at arbitrary coordinates is not implemented for this sampler."
        )

    def inverse(
        self,
        coordinate_system: "CoordinateSystem",
        data_format: DataFormat,
        arguments: Optional[Mapping[str, Any]] = None,
    ) -> ISampler:
        if data_format.coordinate_type == "voxel" and data_format.representation == "displacements":
            if arguments is None:
                arguments = {}
            fixed_point_inversion_arguments = arguments.get("fixed_point_inversion_arguments", {})
            return FixedPointInverseSampler(
                self,
                forward_solver=fixed_point_inversion_arguments.get("forward_solver"),
                backward_solver=fixed_point_inversion_arguments.get("backward_solver"),
                forward_dtype=fixed_point_inversion_arguments.get("forward_dtype"),
                backward_dtype=fixed_point_inversion_arguments.get("backward_dtype"),
                mask_extrapolated_regions=self._mask_extrapolated_regions,
            )
        raise ValueError(
            "Inverse sampler has been currently implemented only for voxel "
            "displacements data format."
        )
