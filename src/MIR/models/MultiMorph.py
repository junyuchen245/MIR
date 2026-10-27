"""MultiMorph group-wise registration model for on-demand atlas construction.

Abulnaga, S. M., Hoopes, A., Dey, N., Hoffmann, M., Fischl, B., Guttag, J., & Dalca, A. (2025).
MultiMorph: On-demand Atlas Construction.
In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 30906-30917).

Code retrieved from:
https://github.com/mabulnaga/multimorph

Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import MIR.models.registration_utils as utils
import torch.utils.checkpoint as checkpoint
import numpy as np
import random
from typing import List
from MIR import _amp


def make_epoch_batches(n_samples: int, min_bs: int, max_bs: int) -> List[np.ndarray]:
    """Create randomized batch index lists for one epoch.

    Indices are shuffled once with ``np.random.permutation``; each batch size is
    drawn uniformly from ``[min_bs, max_bs]`` with ``random.randint``. When at most
    ``max_bs`` samples remain, they all form the last batch (which can therefore be
    smaller than ``min_bs``).

    Args:
        n_samples: Total number of samples.
        min_bs: Minimum batch size (inclusive).
        max_bs: Maximum batch size (inclusive).

    Returns:
        List of numpy arrays of indices for each batch; together they cover every
        index in ``range(n_samples)`` exactly once.

    """
    # shuffle indices once per epoch (use numpy for permutation)
    inds = np.random.permutation(n_samples)
    batches = []
    i = 0
    while i < n_samples:
        remaining = n_samples - i
        if remaining <= max_bs:
            # take all remaining as the last batch
            bs = remaining
        else:
            bs = random.randint(min_bs, max_bs)
        batch_inds = inds[i : i + bs]
        batches.append(batch_inds)
        i += bs
    return batches


class ListBatchSampler(torch.utils.data.Sampler):
    """Yield precomputed lists of indices as batches.

    This allows constructing a DataLoader with arbitrary batch sizes per iteration
    while still supporting num_workers and automatic collation.

    Args:
        batches: Iterable of index sequences (e.g. the output of
            :func:`make_epoch_batches`); each index is converted to ``int``.
    """

    def __init__(self, batches):
        self.batches = [list(map(int, b)) for b in batches]

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)


class DeformationFieldComposer(nn.Module):
    """Compose group-wise displacement fields.

    Args:
        field_size: Spatial shape of the deformation fields, e.g. ``(H, W)`` for 2D
            or ``(H, W, D)`` for 3D.
        mode: Interpolation mode of the internal ``SpatialTransformer``.
    """

    def __init__(self, field_size, mode='bilinear'):
        """Initialize the composer with the shape of the deformation fields.

        Args:
            field_size (tuple): Shape of the deformation fields, e.g., ( H, W) for 2D
                                 or ( D, H, W) for 3D.
            mode (str): Interpolation mode of the internal ``SpatialTransformer``.

        """
        super().__init__()
        self.field_shape = field_size
        self.grid = self._create_grid(field_size)
        self.transformer = utils.SpatialTransformer(field_size, mode=mode)

    def _create_grid(self, field_size):
        """Create a grid for sampling based on the field shape.

        Also stores the grid as ``self.grid``.

        Args:
            field_size (tuple): Shape of the deformation fields.

        Returns:
            torch.Tensor: The grid for sampling, of shape (1, C, D, H, W) for 3D
                          or (1, C, H, W) for 2D.

        """
        # create sampling grid
        vectors = [torch.arange(0, s) for s in field_size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        self.grid = grid.type(torch.FloatTensor)

        return self.grid

    def collapse_group_dim(self, field: torch.Tensor) -> torch.Tensor:
        """Collapse the group dimension into the batch dimension.

        A 5-D tensor with 2 channels (``[B, G, 2, H, W]``) or a 6-D tensor with
        3 channels (``[B, G, 3, H, W, D]``) is reshaped to ``[B*G, C, ...]``; any
        other input is returned unchanged.

        Args:
            field: Deformation field of shape ``[B, G, C, *spatial]``.

        Returns:
            torch.Tensor: Field of shape ``[B*G, C, *spatial]``.
        """
        C = field.shape[2]
        len_dim = len(field.shape)

        if C == 2 and len_dim == 5:
            field = einops.rearrange(field, 'b g c h w -> (b g) c h w')
        elif C == 3 and len_dim == 6:
            field = einops.rearrange(field, 'b g c h w d -> (b g) c h w d')

        return field

    def expand_group_dim(self, field: torch.Tensor, group_size: int) -> torch.Tensor:
        """Split the batch dimension back into batch and group dimensions.

        Inverse of :meth:`collapse_group_dim`; only applied when the channel count
        is 2 or 3, otherwise the input is returned unchanged.

        Args:
            field: Field of shape ``[B*G, C, *spatial]``.
            group_size: Group size ``G``.

        Returns:
            torch.Tensor: Field of shape ``[B, G, C, *spatial]``.
        """
        n = group_size
        C = field.shape[1]
        if C == 2:
            field = einops.rearrange(field, '(b n) c h w -> b n c h w', n=n)
        elif C == 3:
            field = einops.rearrange(field, '(b n) c h w d -> b n c h w d', n=n)

        return field

    def compose(self, fields):
        """Compose the given deformation fields.

        Starting from ``fields[0]``, each subsequent field ``f`` is composed as
        ``u <- u + warp(f, u)``.

        Args:
            fields (list of torch.Tensor): List of displacement fields to compose, each of shape
                                           (B, G, 2, H, W) for 2D or (B, G, 3, H, W, D) for 3D.

        Returns:
            torch.Tensor: Composed deformation field of shape ``[B, G, C, *spatial]``.

        Raises:
            ValueError: If ``fields`` is empty.

        """
        if not fields:
            raise ValueError("No deformation fields to compose.")

        group_size = fields[0].shape[1]

        # collapse field group dim
        f_orig = self.collapse_group_dim(fields[0])

        composed_field = f_orig.clone()
        for _idx, field in enumerate(fields[1:]):
            field = self.collapse_group_dim(field)
            composed_field += self.transformer(field, composed_field)

        composed_field = self.expand_group_dim(composed_field, group_size)

        return composed_field

    def forward(self, fields):
        """Forward pass to compose the deformation fields.

        Args:
            fields (list of torch.Tensor): List of deformation fields to compose,
                each of shape ``[B, G, C, *spatial]``.

        Returns:
            torch.Tensor: Composed deformation field (see :meth:`compose`).

        """
        return self.compose(fields)


class Lambda(nn.Module):
    """Wrap a callable inside an `nn.Module`.

    Args:
        lambd: Callable applied to the input in :meth:`forward`.
    """

    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        """Apply the wrapped callable to input."""
        return self.lambd(x)


class SubtractMean(nn.Module):
    """Subtract mean along a specified dimension.

    Args:
        dim: Dimension along which the mean is computed and subtracted.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subtract mean along `self.dim` from input tensor."""
        return x - torch.mean(x, dim=self.dim, keepdim=True)


class VecInt(nn.Module):
    """Integrate a vector field via scaling and squaring.

    Args:
        inshape: Spatial shape of the vector field.
        nsteps: Number of squaring steps (must be >= 0); the field is first
            scaled by ``1 / 2**nsteps``.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2**self.nsteps)
        self.transformer = utils.SpatialTransformer(inshape)

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        """Integrate the vector field.

        Args:
            vec: Tensor of shape ``[B, C, *spatial]``.

        Returns:
            Integrated vector field tensor of the same shape.

        """
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """Resize and rescale a vector field transform.

    The field is resampled by ``1 / vel_resize`` and its values are multiplied by
    the same factor.

    Args:
        vel_resize: Resize factor; the spatial scale factor is ``1 / vel_resize``.
        ndims: Number of spatial dimensions (2 selects bilinear, 3 trilinear).
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        """Resize and rescale the transform.

        Args:
            x: Vector field tensor of shape ``[B, C, *spatial]``.

        Returns:
            Resized and rescaled vector field tensor (unchanged if the factor is 1).

        """
        if self.factor < 1:
            # resize first to save memory
            x = _amp.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = _amp.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


class MeanConv2d(nn.Module):
    """Perform a group mean convolution (see UniverSeg paper).

    inputs are [b, n, c, h, w] where
        b is the batch size
        n is the number of group entries
        c is the number of channels
        h is the height
        w is the width

    operation is:
        mean representation along group
        concat the mean representation with each group entry representation
        perform a convolution for each concated representation

    The idea is that this allows the entries to interact through the mean
    representation, while still performing individual convolutions.

    Args:
        in_channels: Number of input channels per group entry.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        padding: Convolution padding.
        do_activation: Whether to append a PReLU activation.
        do_batchnorm: Whether to append ``BatchNorm2d``.
        summary_stats: Group statistics concatenated to each entry; any of
            ``'mean'``, ``'max'``, ``'var'``, ``'min'``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        do_activation=True,
        do_batchnorm=True,
        summary_stats=['mean'],
    ):
        super().__init__()

        conv = nn.Conv2d(in_channels * (1 + len(summary_stats)), out_channels, kernel_size=kernel_size, padding=padding)
        lst = [conv]
        if do_batchnorm:
            lst.append(nn.BatchNorm2d(out_channels))
        if do_activation:
            lst.append(nn.PReLU())

        self.conv = nn.Sequential(*lst)
        self.summary_stats = summary_stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the group mean convolution.

        Args:
            x: Tensor of shape ``[B, N, C, H, W]``.

        Returns:
            torch.Tensor: Tensor of shape ``[B, N, out_channels, H', W']``.
        """
        # TODO, use pylot.util.shapechecker
        b, n, c, h, w = x.shape
        # n = x.shape[1]
        stats = torch.empty((b, n, 0, h, w), device=x.device)

        for stat in self.summary_stats:
            if stat == 'mean':
                # mean represetation along group
                meanx = torch.mean(x, dim=1, keepdim=False)  # [B, C, H, W]
                meanx = einops.repeat(meanx, 'b c h w -> b n c h w', n=n)  # too memory intense?
                stats = torch.cat([stats, meanx], dim=2)
            elif stat == 'max':
                maxx, _ = torch.max(x, dim=1, keepdim=False)  # [B, C, H, W]
                maxx = einops.repeat(maxx, 'b c h w -> b n c h w', n=n)  # too memory intense?
                stats = torch.cat([stats, maxx], dim=2)
            elif stat == 'var':
                varx = torch.var(x, dim=1, keepdim=False)  # [B, C, H, W]
                varx = einops.repeat(varx, 'b c h w -> b n c h w', n=n)  # too memory intense?
                stats = torch.cat([stats, varx], dim=2)
            elif stat == 'min':
                minx, _ = torch.min(x, dim=1, keepdim=False)  # [B, C, H, W]
                minx = einops.repeat(minx, 'b c h w -> b n c h w', n=n)  # too memory intense?
                stats = torch.cat([stats, minx], dim=2)

        # concat the mean representation with each group entry representation
        x = torch.cat([x, stats], dim=2)  # [b n 2c h w]

        # move group to batch dimension and do convolution
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.conv(x)
        x = einops.rearrange(x, '(b n) c h w -> b n c h w', n=n)

        return x


class MeanConv3d(nn.Module):
    """Perform a group mean convolution (see UniverSeg paper).

    inputs are [b, n, c, h, w, d] where
        b is the batch size
        n is the number of group entries
        c is the number of channels
        h is the height
        w is the width
        d is the depth

    operation is:
        mean representation along group
        concat the mean representation with each group entry representation
        perform a convolution for each concated representation

    The idea is that this allows the entries to interact through the mean
    representation, while still performing individual convolutions.

    Args:
        in_channels: Number of input channels per group entry.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        padding: Convolution padding.
        do_activation: Whether to append a PReLU activation.
        do_batchnorm: Whether to append ``BatchNorm3d``.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, do_activation=True, do_batchnorm=True):
        super().__init__()

        conv = nn.Conv3d(in_channels * 2, out_channels, kernel_size=kernel_size, padding=padding)
        lst = [conv]
        if do_batchnorm:
            lst.append(nn.BatchNorm3d(out_channels))
        if do_activation:
            lst.append(nn.PReLU())

        self.conv = nn.Sequential(*lst)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the group mean convolution.

        Args:
            x: Tensor of shape ``[B, N, C, H, W, D]``.

        Returns:
            torch.Tensor: Tensor of shape ``[B, N, out_channels, H', W', D']``.
        """
        # TODO, use pylot.util.shapechecker
        n = x.shape[1]

        # mean represetation along group
        meanx = torch.mean(x, dim=1, keepdim=False)  # [B, C, H, W D]
        meanx = einops.repeat(meanx, 'b c h w d -> b n c h w d', n=n)  # too memory intense?

        # concat the mean representation with each group entry representation
        x = torch.cat([x, meanx], dim=2)  # [b n 2c h w d]

        # move group to batch dimension and do convolution
        x = einops.rearrange(x, 'b n c h w d -> (b n) c h w d')
        x = self.conv(x)
        x = einops.rearrange(x, '(b n) c h w d -> b n c h w d', n=n)

        return x


class FastMeanConv3d(nn.Module):
    """Perform a group mean convolution (see UniverSeg paper).

    inputs are [b, n, c, h, w, d] where
        b is the batch size
        n is the number of group entries
        c is the number of channels
        h is the height
        w is the width
        d is the depth

    operation is:
        mean representation along group
        concat the mean representation with each group entry representation
        perform a convolution for each concated representation

    The idea is that this allows the entries to interact through the mean
    representation, while still performing individual convolutions.

    Instead of materializing the concatenation, the convolution weight is split
    into an entry part and a summary part; the summary part is applied once to the
    group statistic and added to every entry's output.

    Args:
        in_channels: Number of input channels per group entry.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        padding: Convolution padding.
        stride: Convolution stride.
        dilation: Convolution dilation.
        groups: Convolution groups.
        summary_stat: Group statistic, one of ``'mean'``, ``'max'``, ``'var'``;
            ``None`` falls back to ``'mean'``.
        bias: Whether the convolution has a bias.
        padding_mode: Padding mode; only ``'zeros'`` is supported.
        do_activation: Whether to apply a PReLU activation.
        do_instancenorm: Whether to apply ``InstanceNorm3d`` before the activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        summary_stat='mean',
        bias: bool = True,
        padding_mode: str = "zeros",
        do_activation: bool = True,
        do_instancenorm: bool = False,
    ):

        super().__init__()

        assert padding_mode == "zeros", "Only zero-padding is supported"

        self.in_channels = in_channels
        self.do_activation = do_activation
        self.do_instancenorm = do_instancenorm
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding

        conv = nn.Conv3d(
            in_channels * 2,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.activation = nn.PReLU() if self.do_activation else None
        self.instance_norm = nn.InstanceNorm3d(out_channels) if self.do_instancenorm else None

        self.weights = conv.weight
        self.bias = conv.bias

        if summary_stat is not None:
            assert summary_stat in ['max', 'mean', 'var'], "Only max, mean, var are supported"
        elif summary_stat is None:
            summary_stat = 'mean'

        self.summary_stat = summary_stat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the group mean convolution.

        The summary-statistic branch has shape ``[B, C_out, ...]`` and is added by
        broadcasting to the ``[B*N, C_out, ...]`` entry branch.

        Args:
            x: Tensor of shape ``[B, N, C, H, W, D]``.

        Returns:
            torch.Tensor: Tensor of shape ``[B, N, out_channels, H', W', D']``.
        """
        n = x.shape[1]

        weight_x = self.weights[:, : self.in_channels]
        weight_mean = self.weights[:, self.in_channels :]
        # mean represetation along group
        if self.summary_stat == 'mean':
            meanx = torch.mean(x, dim=1, keepdim=False)  # [B, C, H, W D]
        elif self.summary_stat == 'max':
            meanx, _ = torch.max(x, dim=1, keepdim=False)
        elif self.summary_stat == 'var':
            meanx = torch.var(x, dim=1, keepdim=False)

        x = einops.rearrange(x, 'b n c h w d -> (b n) c h w d')

        ox = F.conv3d(
            x,
            weight=weight_x,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        out_mean = F.conv3d(
            meanx,
            weight_mean,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        out = ox + out_mean.repeat_interleave(n, dim=0)  # [B, C, ...] -> [B*N, C, ...]

        if self.do_instancenorm:
            out = self.instance_norm(out)

        if self.do_activation:
            out = self.activation(out)

        out = einops.rearrange(out, '(b n) c h w d -> b n c h w d', n=n)

        return out


class FastMeanConv3dUp(nn.Module):
    """Perform a group mean convolution for the upsampling UNet.

    inputs are [b, n, c, h, w, d] where
        b is the batch size
        n is the number of group entries
        c is the number of channels
        h is the height
        w is the width
        d is the depth

    operation is:
        mean representation along group
        concat the mean representation with each group entry representation
        perform a convolution for each concated representation
        repeat this twice, once for the input and once for the skip connection input.

    The idea is that this allows the entries to interact through the mean
    representation, while still performing individual convolutions.

    Args:
        in_channels_skip: Channels of the skip-connection input ``x``.
        in_channels: Channels of the upsampled input ``y``.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        padding: Convolution padding.
        stride: Convolution stride.
        dilation: Convolution dilation.
        groups: Convolution groups.
        bias: Whether the convolution has a bias.
        summary_stat: Group statistic, one of ``'mean'``, ``'max'``, ``'var'``;
            ``None`` falls back to ``'mean'``.
        padding_mode: Padding mode; only ``'zeros'`` is supported.
        do_activation: Whether to apply a PReLU activation.
        do_instancenorm: Whether to apply ``InstanceNorm3d`` before the activation.
    """

    def __init__(
        self,
        in_channels_skip: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        summary_stat: str = 'mean',
        padding_mode: str = "zeros",
        do_activation: bool = True,
        do_instancenorm: bool = False,
    ):

        super().__init__()

        assert padding_mode == "zeros", "Only zero-padding is supported"

        self.in_channels = in_channels
        self.in_channels_skip = in_channels_skip
        self.do_activation = do_activation
        self.do_instancenorm = do_instancenorm
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding

        conv = nn.Conv3d(
            (in_channels_skip + in_channels) * 2,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.instance_norm = nn.InstanceNorm3d(out_channels) if self.do_instancenorm else None
        self.activation = nn.PReLU() if self.do_activation else None

        self.weights = conv.weight
        self.bias = conv.bias

        if summary_stat is not None:
            assert summary_stat in ['max', 'mean', 'var'], "Only max, mean, var are supported"
        elif summary_stat is None:
            summary_stat = 'mean'

        self.summary_stat = summary_stat

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply the group mean convolution to a skip input and an upsampled input.

        Args:
            x: Skip-connection tensor of shape ``[b, n, in_channels_skip, h, w, d]``.
            y: Upsampled tensor of shape ``[b, n, in_channels, h, w, d]``.

        Returns:
            torch.Tensor: Tensor of shape ``[b, n, out_channels, h', w', d']``.

        """
        n = x.shape[1]

        weight_x = self.weights[:, : self.in_channels_skip]
        weight_y = self.weights[:, self.in_channels_skip : self.in_channels + self.in_channels_skip]
        weight_mean_x = self.weights[
            :, self.in_channels + self.in_channels_skip : self.in_channels + self.in_channels_skip * 2
        ]
        weight_mean_y = self.weights[:, self.in_channels + self.in_channels_skip * 2 :]
        # mean represetation along group
        if self.summary_stat == 'mean':
            meanx = torch.mean(x, dim=1, keepdim=False)  # [B, C, H, W D]
            meany = torch.mean(y, dim=1, keepdim=False)  # [B, C, H, W D]
        elif self.summary_stat == 'max':
            meanx, _ = torch.max(x, dim=1, keepdim=False)
            meany, _ = torch.max(y, dim=1, keepdim=False)
        elif self.summary_stat == 'var':
            meanx = torch.var(x, dim=1, keepdim=False)
            meany = torch.var(y, dim=1, keepdim=False)

        x = einops.rearrange(x, 'b n c h w d -> (b n) c h w d')
        y = einops.rearrange(y, 'b n c h w d -> (b n) c h w d')

        ox = F.conv3d(
            x,
            weight=weight_x,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        out_mean_x = F.conv3d(
            meanx,
            weight_mean_x,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        # repeat
        oy = F.conv3d(
            y,
            weight=weight_y,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        out_mean_y = F.conv3d(
            meany,
            weight_mean_y,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        out = ox + out_mean_x.repeat_interleave(n, dim=0) + oy + out_mean_y.repeat_interleave(n, dim=0)

        if self.do_instancenorm:
            out = self.instance_norm(out)

        if self.do_activation:
            out = self.activation(out)

        out = einops.rearrange(out, '(b n) c h w d -> b n c h w d', n=n)

        return out


class GroupConv2d(nn.Module):
    """Perform a group convolution without communication.

    Used for the mean-conv ablation.

    inputs are [b, n, c, h, w] where
        b is the batch size
        n is the number of group entries
        c is the number of channels
        h is the height
        w is the width

    operation is:
        move the group dimension into the batch dimension
        perform the same convolution independently on every group entry

    Unlike :class:`MeanConv2d`, no group summary statistic is used, so entries
    do not interact.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        padding: Convolution padding.
        do_activation: Whether to append a PReLU activation.
        do_batchnorm: Whether to append ``BatchNorm2d``.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, do_activation=True, do_batchnorm=True):
        super().__init__()

        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        lst = [conv]
        if do_batchnorm:
            lst.append(nn.BatchNorm2d(out_channels))
        if do_activation:
            lst.append(nn.PReLU())

        self.conv = nn.Sequential(*lst)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution to each group entry.

        Args:
            x: Tensor of shape ``[B, N, C, H, W]``.

        Returns:
            torch.Tensor: Tensor of shape ``[B, N, out_channels, H', W']``.
        """
        # TODO, use pylot.util.shapechecker
        n = x.shape[1]

        # move group to batch dimension and do convolution
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.conv(x)
        x = einops.rearrange(x, '(b n) c h w -> b n c h w', n=n)

        return x


class GroupConv3d(nn.Module):
    """Perform a group convolution without communication.

    Used for the mean-conv ablation.

    inputs are [b, n, c, h, w, d] where
        b is the batch size
        n is the number of group entries
        c is the number of channels
        h is the height
        w is the width
        d is the depth

    operation is:
        move the group dimension into the batch dimension
        perform the same convolution independently on every group entry

    Unlike :class:`FastMeanConv3d`, no group summary statistic is used, so
    entries do not interact.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        padding: Convolution padding.
        stride: Convolution stride.
        do_activation: Whether to append a PReLU activation.
        do_instancenorm: Whether to append ``InstanceNorm3d``.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, padding, stride=1, do_activation=True, do_instancenorm=True
    ):
        super().__init__()

        conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        lst = [conv]
        if do_instancenorm:
            lst.append(nn.InstanceNorm3d(out_channels))
        if do_activation:
            lst.append(nn.PReLU())

        self.conv = nn.Sequential(*lst)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution to each group entry.

        Args:
            x: Tensor of shape ``[B, N, C, H, W, D]``.

        Returns:
            torch.Tensor: Tensor of shape ``[B, N, out_channels, H', W', D']``.
        """
        # TODO, use pylot.util.shapechecker
        n = x.shape[1]

        # move group to batch dimension and do convolution
        x = einops.rearrange(x, 'b n c h w d -> (b n) c h w d')
        x = self.conv(x)
        x = einops.rearrange(x, '(b n) c h w d -> b n c h w d', n=n)

        return x


class MaxPool2d(nn.Module):
    """2D max pooling applied to each group entry of a ``[B, N, C, H, W]`` tensor.

    Args:
        kernel_size: Pooling kernel size.
        stride: Pooling stride.
    """

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool each group entry.

        Args:
            x: Tensor of shape ``[B, N, C, H, W]``.

        Returns:
            torch.Tensor: Pooled tensor of shape ``[B, N, C, H', W']``.
        """
        n = x.shape[1]
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.pool(x)
        x = einops.rearrange(x, '(b n) c h w -> b n c h w', n=n)
        return x


class MaxPool3d(nn.Module):
    """3D max pooling applied to each group entry of a ``[B, N, C, H, W, D]`` tensor.

    Args:
        kernel_size: Pooling kernel size.
        stride: Pooling stride.
    """

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool each group entry.

        Args:
            x: Tensor of shape ``[B, N, C, H, W, D]``.

        Returns:
            torch.Tensor: Pooled tensor of shape ``[B, N, C, H', W', D']``.
        """
        n = x.shape[1]
        x = einops.rearrange(x, 'b n c h w d -> (b n) c h w d')
        x = self.pool(x)
        x = einops.rearrange(x, '(b n) c h w d -> b n c h w d', n=n)
        return x


class UpsamplingBilinear2d(nn.Module):
    """Bilinear upsampling applied to each group entry of a ``[B, N, C, H, W]`` tensor.

    Args:
        scale_factor: Spatial upsampling factor.
    """

    def __init__(self, scale_factor):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample each group entry.

        Args:
            x: Tensor of shape ``[B, N, C, H, W]``.

        Returns:
            torch.Tensor: Upsampled tensor of shape ``[B, N, C, H*s, W*s]``.
        """
        n = x.shape[1]
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.upsample(x)
        x = einops.rearrange(x, '(b n) c h w -> b n c h w', n=n)
        return x


class UpsamplingTrilinear3d(nn.Module):
    """Trilinear upsampling applied to each group entry of a ``[B, N, C, H, W, D]`` tensor.

    Args:
        scale_factor: Spatial upsampling factor.
    """

    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample each group entry.

        Args:
            x: Tensor of shape ``[B, N, C, H, W, D]``.

        Returns:
            torch.Tensor: Upsampled tensor of shape ``[B, N, C, H*s, W*s, D*s]``.
        """
        n = x.shape[1]
        x = einops.rearrange(x, 'b n c h w d -> (b n) c h w d')
        x = _amp.interpolate(x, scale_factor=self.scale_factor, mode='trilinear')
        x = einops.rearrange(x, '(b n) c h w d -> b n c h w d', n=n)
        return x


class Warp2d(nn.Module):
    """Warp each group entry of a 2D group tensor with its own displacement field.

    Args:
        vol_shape: Spatial shape ``(H, W)``.
        mode: Interpolation mode of the ``SpatialTransformer``.
    """

    def __init__(self, vol_shape, mode='bilinear'):
        super().__init__()
        self.st = utils.SpatialTransformer(vol_shape, mode=mode)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Warp the images.

        Args:
            x: Images of shape ``[B, N, C, H, W]``.
            w: Displacement fields of shape ``[B, N, 2, H, W]`` (voxel units).

        Returns:
            torch.Tensor: Warped images of shape ``[B, N, C, H, W]``.
        """
        n = x.shape[1]
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        w = einops.rearrange(w, 'b n c h w -> (b n) c h w')
        x = self.st(x, w)
        x = einops.rearrange(x, '(b n) c h w -> b n c h w', n=n)
        return x


class Warp3d(nn.Module):
    """Warp each group entry of a 3D group tensor with its own displacement field.

    Args:
        vol_shape: Spatial shape ``(H, W, D)``.
        mode: Interpolation mode of the ``SpatialTransformer``.
    """

    def __init__(self, vol_shape, mode='bilinear'):
        super().__init__()
        self.st = utils.SpatialTransformer(vol_shape, mode=mode)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Warp the images.

        Args:
            x: Images of shape ``[B, N, C, H, W, D]``.
            w: Displacement fields of shape ``[B, N, 3, H, W, D]`` (voxel units).

        Returns:
            torch.Tensor: Warped images of shape ``[B, N, C, H, W, D]``.
        """
        n = x.shape[1]
        x = einops.rearrange(x, 'b n c h w d -> (b n) c h w d')
        w = einops.rearrange(w, 'b n c h w d -> (b n) c h w d')
        x = self.st(x, w)
        x = einops.rearrange(x, '(b n) c h w d -> b n c h w d', n=n)
        return x


class VecIntGroup(nn.Module):
    """Vector integration with group dimension.

    Args:
        img_size: Spatial shape ``(H, W, D)`` of the velocity fields.
        nsteps: Number of scaling-and-squaring steps.
    """

    def __init__(self, img_size, nsteps=5):
        super().__init__()
        self.img_size = img_size
        self.integrate = utils.VecInt(inshape=[*self.img_size], nsteps=nsteps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate each group entry's velocity field.

        Args:
            x: Velocity fields of shape ``[B, N, 3, H, W, D]``.

        Returns:
            torch.Tensor: Displacement fields of shape ``[B, N, 3, H, W, D]``.
        """
        n = x.shape[1]
        x = einops.rearrange(x, 'b n c h w d -> (b n) c h w d')
        # self.velocity_field = x
        x = self.integrate(x)
        x = einops.rearrange(x, '(b n) c h w d -> b n c h w d', n=n)

        return x


class ComposeCentrality(nn.Module):
    """Compose group-wise fields with a shared central field.

    Args:
        field_size: Spatial shape of the deformation fields.
        dim: Dimension to average over (stored but not used by :meth:`forward`).
    """

    def __init__(self, field_size, dim=1):
        """Initialize the composer with the shape of the deformation fields.

        Args:
            field_size (tuple): Shape of the deformation fields, e.g., ( H, W) for 2D
                                 or ( D, H, W) for 3D.
            dim : dimension to average over

        """
        super().__init__()
        self.field_size = field_size
        self.dim = dim
        self.tx_composer = DeformationFieldComposer(field_size)

    def forward(self, field_1: torch.Tensor, field_2: torch.Tensor) -> torch.Tensor:
        """Compose each group entry's field with the central field.

        Args:
            field_1: Displacement fields of shape ``[B, G, 3, H, W, D]``.
            field_2: Central displacement field of shape ``[B, 1, 3, H, W, D]``;
                it is repeated ``G`` times along dim 1.

        Returns:
            torch.Tensor: Composed fields of shape ``[B, G, 3, H, W, D]``
            (``field_1`` followed by ``field_2``, see
            :meth:`DeformationFieldComposer.compose`).
        """
        B, G, C, H, W, D = field_1.shape
        central_warp = field_2.repeat(1, G, 1, 1, 1, 1)
        # central_warp = torch.mean(field_2, dim=self.dim, keepdim=True).repeat(1, G, 1, 1, 1, 1)

        return self.tx_composer([field_1, central_warp])


class GroupNet3D(nn.Module):
    """MultiMorph 3D group U-Net predicting per-entry diffeomorphic fields.

    Operates on a group of images ``[B, N, C, H, W, D]``. Group-aware
    convolutions (:class:`FastMeanConv3d`) let the entries interact; the predicted
    velocity fields are optionally mean-centred across the group, integrated by
    scaling and squaring, and (with ``do_half_res``) upsampled to full resolution.

    Args:
        in_channels: Input channels per group entry.
        out_channels: Output channels (3 for a 3D velocity field).
        features: Feature widths of the encoder levels.
        conv_kernel_size: Convolution kernel size.
        displacement_field_dim: Dimension used for mean subtraction (1 = group).
        do_mean_conv: Use group mean convolutions; otherwise use independent
            :class:`GroupConv3d` (ablation).
        diffeo_steps: Number of scaling-and-squaring steps.
        img_size: Full-resolution spatial shape ``(H, W, D)``.
        summary_stat: Group statistic for mean convolutions (``'mean'``,
            ``'max'`` or ``'var'``).
        do_instancenorm: Whether to use instance normalization.
        subtract_mean: Subtract the group mean of the velocity fields.
        do_half_res: Run the first convolution with stride 2 and integrate at
            half resolution, then upsample the displacement fields.
        output_inverse_field: Also return the field integrated from the negated
            velocity.
        checkpoint_model: Use gradient checkpointing for the convolution blocks.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=[64, 64, 64, 64],
        conv_kernel_size=3,
        displacement_field_dim=1,
        do_mean_conv=True,
        diffeo_steps=5,
        img_size=[64, 64, 64],
        summary_stat='mean',
        do_instancenorm=False,
        subtract_mean=True,
        do_half_res=True,
        output_inverse_field=False,
        checkpoint_model=False,
    ):
        super().__init__()
        self.subtract_mean = subtract_mean
        self.img_size = img_size
        self.do_mean_conv = do_mean_conv
        self.velocity_field = None
        self.do_half_res = do_half_res
        self.checkpoint_model = checkpoint_model
        self.output_inverse_field = output_inverse_field

        # summary statistic for group conv
        if summary_stat is not None and do_mean_conv is True:
            assert summary_stat in ['max', 'mean', 'var'], "Only max, mean, var are supported"
        elif summary_stat is None and do_mean_conv is True:
            summary_stat = 'mean'

        padding = (conv_kernel_size - 1) // 2

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = MaxPool3d(kernel_size=2, stride=2)

        # Down part of U-Net
        for idx, feat in enumerate(features):
            if self.do_half_res:
                stride = 2 if idx == 0 else 1
            else:
                stride = 1
            if self.do_mean_conv:
                self.downs.append(
                    FastMeanConv3d(
                        in_channels,
                        feat,
                        kernel_size=conv_kernel_size,
                        padding=padding,
                        summary_stat=summary_stat,
                        do_instancenorm=do_instancenorm,
                        stride=stride,
                    )
                )
            else:
                self.downs.append(
                    GroupConv3d(
                        in_channels,
                        feat,
                        kernel_size=conv_kernel_size,
                        padding=padding,
                        do_instancenorm=do_instancenorm,
                        stride=stride,
                    )
                )
            in_channels = feat

        # Up part of U-Net
        prev_layers = [features[-1]] + features[::-1]
        features_use = features if not self.do_half_res else features[1:]
        for idx, feat in enumerate(reversed(features_use)):
            self.ups.append(UpsamplingTrilinear3d(scale_factor=2))
            if self.do_mean_conv:
                self.ups.append(
                    FastMeanConv3dUp(
                        feat,
                        prev_layers[idx],
                        feat,
                        kernel_size=conv_kernel_size,
                        padding=padding,
                        summary_stat=summary_stat,
                        do_instancenorm=do_instancenorm,  # prev was feat * 2
                    )
                )
            else:
                self.ups.append(
                    GroupConv3d(
                        feat * 2, feat, kernel_size=conv_kernel_size, padding=padding, do_instancenorm=do_instancenorm
                    )
                )

        final_feature = features[0] if not self.do_half_res else features[1]
        if self.do_mean_conv:
            self.bottleneck = FastMeanConv3d(
                features[-1],
                features[-1],
                kernel_size=conv_kernel_size,
                padding=padding,
                summary_stat=summary_stat,
                do_instancenorm=do_instancenorm,
            )
            self.final_conv = FastMeanConv3d(
                final_feature,
                out_channels,
                kernel_size=1,
                padding=0,
                summary_stat=summary_stat,
                do_activation=False,
                do_instancenorm=False,
            )
        else:
            self.bottleneck = GroupConv3d(
                features[-1],
                features[-1],
                kernel_size=conv_kernel_size,
                padding=padding,
                do_instancenorm=do_instancenorm,
            )
            self.final_conv = GroupConv3d(
                final_feature, out_channels, kernel_size=1, padding=0, do_activation=False, do_instancenorm=False
            )

        self.subtract_mean_layer = SubtractMean(dim=displacement_field_dim)

        img_size_layer = [int(x / 2) for x in img_size] if self.do_half_res else img_size

        # self.subtract_mean_layer = ComposeCentrality(dim = displacement_field_dim, field_size = img_size_layer)
        self.integrate_layer = VecIntGroup(img_size=img_size_layer, nsteps=diffeo_steps)
        self.resize_layer = ResizeTransform(1 / 2, 3) if self.do_half_res else None

        # if self.do_half_res:
        #     self.integrate = layers.VecInt(inshape=[ int(x/2) for x in self.img_size ], nsteps=diffeo_steps)
        #     self.resize = layers.ResizeTransform(1/2, 3)
        # else:
        #     self.integrate = layers.VecInt(inshape=[ *self.img_size], nsteps=diffeo_steps)
        #     self.resize = None

    def forward(self, x: torch.Tensor):
        """Predict group-wise displacement fields.

        The pre-integration velocity field is stored in ``self.velocity_field``.

        Args:
            x: Group of images of shape ``[B, N, in_channels, H, W, D]``.

        Returns:
            Displacement fields of shape ``[B, N, 3, H, W, D]``; if
            ``output_inverse_field`` is True, a tuple ``(flow, inverse_flow)``
            where ``inverse_flow`` is integrated from the negated velocity.
        """
        skip_connections = []
        for idx, down in enumerate(self.downs):
            x = checkpoint.checkpoint(down, x) if self.checkpoint_model else down(x)
            if self.do_half_res:
                if idx > 0:
                    skip_connections.append(x)
                    x = self.pool(x)
            else:
                skip_connections.append(x)
                x = self.pool(x)

        # x = self.bottleneck(x)
        x = checkpoint.checkpoint(self.bottleneck, x) if self.checkpoint_model else self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            # x = self.ups[idx](x)
            x = checkpoint.checkpoint(self.ups[idx], x) if self.checkpoint_model else self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if self.do_mean_conv:
                x = self.ups[idx + 1](skip_connection, x)  # (concat_skip)
            else:
                concat_skip = torch.cat((skip_connection, x), dim=2)
                x = self.ups[idx + 1](concat_skip)

        # x = self.final_conv(x)
        x = checkpoint.checkpoint(self.final_conv, x) if self.checkpoint_model else self.final_conv(x)

        # mean subtraction layer
        if self.subtract_mean:
            x = self.subtract_mean_layer(x)

        # centrality composition layer. Old stuff with compositions.
        # if self.subtract_mean:
        #     #y = self.integrate_layer(-1*x)
        #     disp_field = self.integrate_layer(x)
        #     y = self.integrate_layer(torch.mean(-1*x, dim=1, keepdim=True))
        #     x = self.subtract_mean_layer(disp_field, y)
        # else:
        #     x = self.integrate_layer(x)

        self.velocity_field = x
        y = -1 * x if self.output_inverse_field else None

        # integrate
        x = checkpoint.checkpoint(self.integrate_layer, x) if self.checkpoint_model else self.integrate_layer(x)
        if self.output_inverse_field:
            y = self.integrate_layer(y)

        if self.do_half_res:
            # todo: make another layer that also includes the einops stuff.
            n = x.shape[1]
            x = einops.rearrange(x, 'b n c h w d -> (b n) c h w d')
            x = self.resize_layer(x)
            x = einops.rearrange(x, '(b n) c h w d -> b n c h w d', n=n)
            # later to do: make this einops stuff happen only once.
            if self.output_inverse_field:
                n = y.shape[1]
                y = einops.rearrange(y, 'b n c h w d -> (b n) c h w d')
                y = self.resize_layer(y)
                y = einops.rearrange(y, '(b n) c h w d -> b n c h w d', n=n)

        if self.output_inverse_field:
            return x, y
        return x


class GroupNet(nn.Module):
    """MultiMorph 2D group U-Net predicting per-entry displacement fields.

    Args:
        in_channels: Input channels per group entry.
        out_channels: Output channels (2 for a 2D field).
        features: Feature widths of the encoder levels.
        conv_kernel_size: Convolution kernel size.
        displacement_field_dim: Dimension used for mean subtraction (1 = group).
        do_mean_conv: Use :class:`MeanConv2d`; otherwise :class:`GroupConv2d`.
        do_diffeomorphism: Integrate the output with scaling and squaring.
        diffeo_steps: Number of scaling-and-squaring steps.
        img_size: Spatial shape ``(H, W)``.
        do_batchnorm: Whether to use batch normalization.
        subtract_mean: Subtract the group mean of the output fields.
        summary_stats: Group statistics used by :class:`MeanConv2d`.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=[64, 64, 64, 64],
        conv_kernel_size=3,
        displacement_field_dim=1,
        do_mean_conv=True,
        do_diffeomorphism=False,
        diffeo_steps=5,
        img_size=[64, 64],
        do_batchnorm=True,
        subtract_mean=True,
        summary_stats=['mean'],
    ):
        super().__init__()
        self.subtract_mean = subtract_mean
        self.do_diffeomorphism = do_diffeomorphism
        self.img_size = img_size
        self.do_mean_conv = do_mean_conv
        self.velocity_field = None

        padding = (conv_kernel_size - 1) // 2

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = MaxPool2d(kernel_size=2, stride=2)

        # Down part of U-Net
        for feat in features:
            if self.do_mean_conv:
                self.downs.append(
                    MeanConv2d(
                        in_channels,
                        feat,
                        kernel_size=conv_kernel_size,
                        padding=padding,
                        do_batchnorm=do_batchnorm,
                        summary_stats=summary_stats,
                    )
                )
            else:
                self.downs.append(
                    GroupConv2d(
                        in_channels, feat, kernel_size=conv_kernel_size, padding=padding, do_batchnorm=do_batchnorm
                    )
                )
            in_channels = feat

        # Up part of U-Net
        for feat in reversed(features):
            self.ups.append(UpsamplingBilinear2d(scale_factor=2))
            if self.do_mean_conv:
                self.ups.append(
                    MeanConv2d(
                        feat * 2,
                        feat,
                        kernel_size=conv_kernel_size,
                        padding=padding,
                        do_batchnorm=do_batchnorm,
                        summary_stats=summary_stats,
                    )
                )
            else:
                self.ups.append(
                    GroupConv2d(
                        feat * 2, feat, kernel_size=conv_kernel_size, padding=padding, do_batchnorm=do_batchnorm
                    )
                )

        if self.do_mean_conv:
            self.bottleneck = MeanConv2d(
                features[-1],
                features[-1],
                kernel_size=conv_kernel_size,
                padding=padding,
                do_batchnorm=do_batchnorm,
                summary_stats=summary_stats,
            )
            self.final_conv = MeanConv2d(
                features[0],
                out_channels,
                kernel_size=1,
                padding=0,
                do_activation=False,
                do_batchnorm=False,
                summary_stats=summary_stats,
            )
        else:
            self.bottleneck = GroupConv2d(
                features[-1], features[-1], kernel_size=conv_kernel_size, padding=padding, do_batchnorm=do_batchnorm
            )
            self.final_conv = GroupConv2d(
                features[0], out_channels, kernel_size=1, padding=0, do_activation=False, do_batchnorm=False
            )

        self.subtract_mean_layer = SubtractMean(dim=displacement_field_dim)

        if self.do_diffeomorphism:
            self.integrate = utils.VecInt(inshape=[*self.img_size], nsteps=diffeo_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict group-wise fields.

        If ``do_diffeomorphism`` is set, the output is integrated first (the
        integrated field is stored in ``self.velocity_field``) and the group mean
        is subtracted afterwards. Encoder inputs/outputs are stored in
        ``self.int_features`` and ``self.int_features_out``.

        Args:
            x: Group of images of shape ``[B, N, in_channels, H, W]``.

        Returns:
            torch.Tensor: Fields of shape ``[B, N, out_channels, H, W]``.
        """
        skip_connections = []
        self.int_features = []
        self.int_features_out = []
        for _idx, down in enumerate(self.downs):
            self.int_features.append(x)
            x = down(x)
            self.int_features_out.append(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            #             if x.shape != skip_connection.shape:
            #                 print('interpolating')
            #                 x = nn.functional.interpolate(x,
            #                                               size=skip_connection.shape[2:],
            #                                               mode='bilinear',
            #                                               align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=2)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)

        # diffeomorphism layer
        if self.do_diffeomorphism:
            n = x.shape[1]
            x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
            x = self.integrate(x)
            x = einops.rearrange(x, '(b n) c h w -> b n c h w', n=n)
            self.velocity_field = x
        # mean subtraction layer
        if self.subtract_mean:
            x = self.subtract_mean_layer(x)
        return x


class SimpleUNet(nn.Module):
    """Plain 2D U-Net baseline (no group dimension).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        features: Feature widths of the encoder levels.
        conv_kernel_size: Convolution kernel size.
        do_batchnorm: Unused; ``conv_block`` always adds batch normalization.
        do_diffeomorphism: Integrate the output with scaling and squaring.
        bidir: Also integrate the negated velocity field.
        diffeo_steps: Number of scaling-and-squaring steps.
        img_size: Spatial shape ``(H, W)``.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=[64, 64, 64, 64],
        conv_kernel_size=3,
        do_batchnorm=True,
        do_diffeomorphism=True,
        bidir=False,
        diffeo_steps=5,
        img_size=[64, 64],
    ):

        super().__init__()

        self.img_size = img_size
        self.do_diffeomorphism = do_diffeomorphism
        self.velocity_field = None
        self.bidir = bidir

        padding = (conv_kernel_size - 1) // 2

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act = nn.ReLU(inplace=True)

        # Down part of U-Net
        for feat in features:
            self.downs.append(self.conv_block(in_channels, feat, kernel_size=conv_kernel_size, padding=padding))
            in_channels = feat

        # Up part of U-Net
        for feat in reversed(features):
            self.ups.append(nn.UpsamplingBilinear2d(scale_factor=2))
            self.ups.append(self.conv_block(feat * 2, feat, kernel_size=conv_kernel_size, padding=padding))

        self.bottleneck = self.conv_block(features[-1], features[-1], kernel_size=conv_kernel_size, padding=padding)

        self.final_conv = self.conv_block(features[0], out_channels, kernel_size=1, padding=0)
        if self.do_diffeomorphism:
            self.integrate = utils.VecInt(inshape=[*self.img_size], nsteps=diffeo_steps)

    def conv_block(self, in_channels, out_channels, kernel_size, padding, do_activation=True, do_batchnorm=True):
        """Build a ``Conv2d`` (+ ``BatchNorm2d``) (+ ReLU) block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            padding: Convolution padding.
            do_activation: Whether to append the shared ReLU.
            do_batchnorm: Whether to append ``BatchNorm2d``.

        Returns:
            nn.Sequential: The block.
        """
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        lst = [conv]
        if do_batchnorm:
            lst.append(nn.BatchNorm2d(out_channels))
        if do_activation:
            lst.append(self.act)

        return nn.Sequential(*lst)

    def conv_block2(self, *args, **kwargs):
        """Build a ``Conv2d`` followed by the shared ReLU.

        Args:
            *args: Positional arguments forwarded to ``nn.Conv2d``.
            **kwargs: Keyword arguments forwarded to ``nn.Conv2d``.

        Returns:
            nn.Sequential: The block.
        """
        return nn.Sequential(nn.Conv2d(*args, **kwargs), self.act)

    def forward(self, x: torch.Tensor):
        """Run the U-Net.

        Note that the return expression ``x, y if self.bidir else x`` always
        yields a 2-tuple: ``(x, y)`` when ``bidir`` is True (``y`` is the field
        integrated from the negated velocity, or None without
        ``do_diffeomorphism``) and ``(x, x)`` otherwise.

        Args:
            x: Input tensor of shape ``[B, in_channels, H, W]``.

        Returns:
            tuple: Two tensors of shape ``[B, out_channels, H, W]`` (see above).
        """
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            #             if x.shape != skip_connection.shape:
            #                 print('interpolating')
            #                 x = nn.functional.interpolate(x,
            #                                               size=skip_connection.shape[2:],
            #                                               mode='bilinear',
            #                                               align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)
        y = None

        # diffeomorphism layer
        if self.do_diffeomorphism:
            self.velocity_field = x
            x = self.integrate(x)
            # inverse field
            y = self.integrate(-1 * self.velocity_field) if self.bidir else None

        return (x, y) if self.bidir else x
