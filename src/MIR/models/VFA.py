"""VFA models and utilities for deformable image registration."""

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import MIR.models.registration_utils as utils
from typing import List, Tuple
from MIR import _amp


def _candidates_by_slicing(x, dim, kernel=3):
    """ONNX-friendly equivalent of ``Tensor.unfold``-based patch extraction (stride 1).

    Returns ``[B, *spatial_out, kernel**dim, C]`` with the same patch ordering as the
    ``unfold`` path, built from explicit shifted slices so that exporters that cannot
    resolve ``aten::unfold`` sizes (e.g. torch.onnx) can trace it.
    """
    import itertools

    out_shape = [int(s) - kernel + 1 for s in x.shape[2:]]
    cands = []
    for off in itertools.product(range(kernel), repeat=dim):
        sl = (slice(None), slice(None)) + tuple(slice(o, o + n) for o, n in zip(off, out_shape))
        cands.append(x[sl])
    patches = torch.stack(cands, dim=-1)  # [B, C, *out, P]
    perm = [0] + list(range(2, 2 + dim)) + [2 + dim, 1]
    return patches.permute(*perm)


def grid_upsample(grid, mode='bilinear', align_corners=True, scale_factor=2):
    """Upsample a sampling grid by a scale factor.

    Absolute (voxel) coordinates are first rescaled from the input to the output grid size, assuming
    corner-aligned voxel centres, and then interpolated.

    Args:
        grid: Tensor of shape ``[B, C, *spatial]`` containing absolute coordinates.
        mode: Interpolation mode passed to `torch.nn.functional.interpolate`.
        align_corners: Whether to align corners during interpolation.
        scale_factor: Scale factor to upsample spatial dimensions.

    Returns:
        Tensor with the same shape as `grid` except spatial dimensions scaled.
    """
    if len(grid.shape[2:]) == 3 and mode == 'bilinear':
        mode = 'trilinear'

    in_dims = torch.tensor(grid.shape[2:]).to(grid.device)
    in_dims = in_dims.view(-1, len(in_dims), *[1 for x in range(len(in_dims))])
    out_dims = in_dims * scale_factor
    out = grid / (in_dims - 1) * (out_dims - 1)
    out = _amp.interpolate(
        out,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )
    return out


def grid_sampler(source, grid, mode='bilinear', align_corners=True, normalize=True, padding_mode='border'):
    """Sample `source` at coordinates defined by `grid`.

    Args:
        source: Tensor of shape ``[B, C, *spatial]``.
        grid: Tensor of shape ``[B, ndim, *spatial]`` with absolute coordinates.
        mode: Interpolation mode for sampling.
        align_corners: Whether to align corners in grid sampling.
        normalize: If True, convert `grid` to normalized [-1, 1] coordinates using the grid's own spatial size
            (so `source` and `grid` are expected to share the same spatial size).
        padding_mode: Padding mode for out-of-bound values.

    Returns:
        Warped tensor of shape ``[B, C, *grid_spatial]`` sampled from `source` at `grid` locations. Only 2D and 3D
        grids are supported.
    """
    out_dtype = source.dtype
    source, grid = _amp._cast_floats(source), _amp._cast_floats(grid)  # sample in float32 for fp16/bf16
    if normalize:
        normed_grid = normalize_grid(grid)
    else:
        normed_grid = grid

    if len(grid.shape[2:]) == 3:
        normed_grid = normed_grid.permute(0, 2, 3, 4, 1).flip(-1)
        warped = nnf.grid_sample(
            source,
            normed_grid,
            align_corners=align_corners,
            padding_mode=padding_mode,
            mode=mode,
        )
    elif len(grid.shape[2:]) == 2:
        normed_grid = normed_grid.permute(0, 2, 3, 1).flip(-1)
        warped = nnf.grid_sample(
            source,
            normed_grid,
            align_corners=align_corners,
            padding_mode=padding_mode,
            mode=mode,
        )
    return warped.to(out_dtype)


def normalize_grid(grid):
    """Normalize an absolute coordinate grid to [-1, 1] for grid sampling.

    Args:
        grid: Tensor of shape ``[B, ndim, *spatial]`` with absolute coordinates.

    Returns:
        Tensor of the same shape with normalized coordinates.
    """
    dims = torch.tensor(grid.shape[2:]).to(grid.device)
    dims = dims.view(-1, len(dims), *[1 for x in range(len(dims))])
    normed_grid = grid / (dims - 1) * 2 - 1  # convert to normalized space
    return normed_grid


def identity_grid_like(tensor, normalize, padding=0):
    """Return an identity grid matching a reference tensor.

    Args:
        tensor: Reference tensor of shape ``[B, C, *spatial]``.
        normalize: If True, return normalized [-1, 1] coordinates.
        padding: Int or sequence of per-dimension padding for the grid; the grid then spans
            ``[-pad, dim + pad)`` along each axis.

    Returns:
        Float32 tensor of shape ``[B, ndim, *(spatial + 2 * padding)]`` with identity coordinates, on the device
        of `tensor`. It is built under ``torch.inference_mode`` and returned as a clone.
    """
    with torch.inference_mode():
        dims = tensor.shape[2:]
        if isinstance(padding, int):
            pads = [padding for j in range(len(dims))]
        else:
            pads = padding
        vectors = [torch.arange(start=0 - pad, end=dim + pad) for (dim, pad) in zip(dims, pads)]

        try:
            grids = torch.meshgrid(vectors, indexing='ij')
        except TypeError:
            # compatible with old pytorch version
            grids = torch.meshgrid(vectors)
        grid = torch.stack(grids).unsqueeze(0).type(torch.float32)

        if normalize:
            in_dims = torch.tensor(dims).view(-1, len(dims), *[1 for x in range(len(dims))])
            grid = grid / (in_dims - 1) * 2 - 1

        grid = grid.to(tensor.device).repeat(tensor.shape[0], 1, *[1 for j in range(len(dims))])
    return grid.clone()


def grid_to_flow(grid: torch.Tensor) -> torch.Tensor:
    """Convert an absolute sampling grid (as produced by VFA) to a voxel displacement field.

    The result is the displacement field that VoxelMorph’s SpatialTransformer expects.

    Args:
        grid: Tensor of shape ``[B, ndim, *spatial_dims]`` with absolute coords.

    Returns:
        Tensor of the same shape containing displacements from identity.
    """
    # build an identity grid on the same device / dtype / size
    id_grid = identity_grid_like(grid, normalize=False)  # already channels‑first
    flow = grid - id_grid
    return flow


class VFASPR(nn.Module):
    """VFA model with spatially varying regularization (VFA-SPR).

    Args:
        configs: VFA configuration object (uses ``img_size``, ``in_channels``, ``downsamples``,
            ``start_channels``, ``max_channels``, ``matching_channels``, ``skip``, ``initialize``, ``int_steps``
            and ``affine``).
        device: Device to which the decoder's precomputed radial vectors ``R`` are moved.
        return_orginal: If True, return VFA-style composed grids and stats.
        return_all_flows: If True, return flows for all decoder levels.
        SVF: If True, integrate flow as stationary velocity field.
        SVF_steps: Number of scaling-and-squaring steps for integration.
        return_full: If True, also return warped images and inverse flow (only used when ``SVF`` is True; the
            spatial transformer is created on CUDA).

    Forward inputs:
        sample: Tuple `(mov, fix)` tensors of shape ``[B, 1, *spatial]``.

    Forward outputs:
        Depending on flags, returns flow(s), spatial weights, or a results dict (see :meth:`forward`).
    """

    def __init__(
        self, configs, device, return_orginal=False, return_all_flows=False, SVF=False, SVF_steps=7, return_full=False
    ):
        super().__init__()

        self.dim = len(configs.img_size)
        self.encoder = Encoder(
            dimension=self.dim,
            in_channels=configs.in_channels,
            downsamples=configs.downsamples,
            start_channels=configs.start_channels,
            max_channels=configs.max_channels,
        ).type(torch.float32)

        self.decoder = Decoder(
            dimension=self.dim,
            downsamples=configs.downsamples,
            matching_channels=configs.matching_channels,
            start_channels=configs.start_channels,
            max_channels=configs.max_channels,
            skip=configs.skip,
            initialize=configs.initialize,
            int_steps=configs.int_steps,
        ).type(torch.float32)
        self.configs = configs
        self.device = device
        self.decoder.R = self.decoder.R.to(device)
        self.return_orginal = return_orginal
        self.return_all_flows = return_all_flows
        self.SPRDecoder = SPRDecoder(self.dim, configs.start_channels)
        self.SVF = SVF
        if self.SVF:
            self.integrate = utils.VecInt(configs.img_size, SVF_steps)
        self.return_full = return_full
        if self.return_full:
            self.spatial_trans = utils.SpatialTransformer(configs.img_size).to(device)

    def forward(self, sample: Tuple[torch.Tensor, torch.Tensor]):
        """Run forward registration.

        Args:
            sample: Tuple `(mov, fix)` tensors, each of shape ``[B, C, *spatial]``.

        Returns:
            Output varies by flags (`return_orginal`, `return_all_flows`, `SVF`,
            `return_full`), checked in this order:

            - ``return_orginal``: a results dict built by ``self.generate_results`` (and
              ``self.generate_affine_results`` when ``configs.affine``); these methods are not defined on this
              class, so this path raises ``AttributeError`` unless a subclass provides them.
            - ``return_all_flows``: ``(composed_flows, spatial_wts)`` where ``composed_flows`` is a list of displacement
              fields, coarse to fine, one per decoder level.
            - ``SVF`` and ``return_full``: ``(def_sor, def_tar, pos_flow, neg_flow, spatial_wts)``: the moving and fixed
              images warped by ``pos_flow`` and ``neg_flow``, where ``pos_flow`` integrates the predicted flow
              and ``neg_flow = exp(-v)`` is its inverse.
            - ``SVF``: ``(pos_flow, spatial_wts)`` -- the predicted flow integrated as a stationary velocity field.
            - otherwise: ``(flow, spatial_wts)`` -- the displacement field ``[B, ndim, *spatial]`` (voxel units).

            ``spatial_wts`` is a ``[B, 1, *spatial]`` map of regularization weights in [1e-6, 1] predicted from
            the finest-level moving and fixed features.
        """
        mov, fix = sample
        F = self.encoder(fix)
        M = self.encoder(mov)
        spatial_wts = self.SPRDecoder(M[0], F[0])
        composed_grids = self.decoder(F, M)
        if self.return_orginal:
            if not hasattr(self, 'generate_results'):
                raise NotImplementedError(
                    "return_orginal=True needs generate_results(), which is not part of the MIR port of VFA; "
                    "construct the model with return_orginal=False"
                )
            results = self.generate_results(composed_grids[-1], sample)
            results.update(
                {
                    'composed_grids': composed_grids,
                    'beta': self.decoder.beta.clone(),
                }
            )

            if self.configs.affine:
                affine_results = self.generate_affine_results(
                    composed_grids[-1],
                    sample,
                )
                results.update(affine_results)

            return results
        if self.return_all_flows:
            composed_flows = []
            for i in range(len(composed_grids)):
                composed_flows.append(grid_to_flow(composed_grids[i]))
            return composed_flows, spatial_wts
        flow = grid_to_flow(composed_grids[-1])
        if self.SVF:
            pos_flow = self.integrate(flow)
            if self.return_full:
                neg_flow = self.integrate(-flow)  # inverse of exp(v) is exp(-v)
                def_sor = self.spatial_trans(mov, pos_flow)
                def_tar = self.spatial_trans(fix, neg_flow)
                return def_sor, def_tar, pos_flow, neg_flow, spatial_wts
            return pos_flow, spatial_wts
        return flow, spatial_wts


class VFA(nn.Module):
    """VFA model for image registration.

    Args:
        configs: VFA configuration object (uses ``img_size``, ``in_channels``, ``downsamples``,
            ``start_channels``, ``max_channels``, ``matching_channels``, ``skip``, ``initialize``, ``int_steps``
            and ``affine``).
        device: Device to which the decoder's precomputed radial vectors ``R`` are moved.
        return_orginal: If True, return VFA-style composed grids and stats.
        return_all_flows: If True, return flows for all decoder levels.
        SVF: If True, integrate flow as stationary velocity field.
        SVF_steps: Number of scaling-and-squaring steps for integration.
        return_full: If True, also return warped images and inverse flow (only used when ``SVF`` is True; the
            spatial transformer is created on CUDA).

    Forward inputs:
        sample: Tuple `(mov, fix)` tensors of shape ``[B, 1, *spatial]``.

    Forward outputs:
        Depending on flags, returns flow(s) or a results dict (see :meth:`forward`).
    """

    def __init__(
        self, configs, device, return_orginal=False, return_all_flows=False, SVF=False, SVF_steps=7, return_full=False
    ):
        super().__init__()

        self.dim = len(configs.img_size)
        self.encoder = Encoder(
            dimension=self.dim,
            in_channels=configs.in_channels,
            downsamples=configs.downsamples,
            start_channels=configs.start_channels,
            max_channels=configs.max_channels,
        ).type(torch.float32)

        self.decoder = Decoder(
            dimension=self.dim,
            downsamples=configs.downsamples,
            matching_channels=configs.matching_channels,
            start_channels=configs.start_channels,
            max_channels=configs.max_channels,
            skip=configs.skip,
            initialize=configs.initialize,
            int_steps=configs.int_steps,
        ).type(torch.float32)
        self.configs = configs
        self.device = device
        self.decoder.R = self.decoder.R.to(device)
        self.return_orginal = return_orginal
        self.return_all_flows = return_all_flows
        self.SVF = SVF
        if self.SVF:
            self.integrate = utils.VecInt(configs.img_size, SVF_steps)
        self.return_full = return_full
        if self.return_full:
            self.spatial_trans = utils.SpatialTransformer(configs.img_size).to(device)

    def forward(self, sample: Tuple[torch.Tensor, torch.Tensor]):
        """Run forward registration.

        Args:
            sample: Tuple `(mov, fix)` tensors, each of shape ``[B, C, *spatial]``.

        Returns:
            Output varies by flags (`return_orginal`, `return_all_flows`, `SVF`,
            `return_full`), checked in this order:

            - ``return_orginal``: a results dict built by ``self.generate_results`` (and
              ``self.generate_affine_results`` when ``configs.affine``); these methods are not defined on this
              class, so this path raises ``AttributeError`` unless a subclass provides them.
            - ``return_all_flows``: ``composed_flows`` where ``composed_flows`` is a list of displacement
              fields, coarse to fine, one per decoder level.
            - ``SVF`` and ``return_full``: ``(def_sor, def_tar, pos_flow, neg_flow)``: the moving and fixed
              images warped by ``pos_flow`` and ``neg_flow``, where ``pos_flow`` integrates the predicted flow
              and ``neg_flow = exp(-v)`` is its inverse.
            - ``SVF``: ``pos_flow`` -- the predicted flow integrated as a stationary velocity field.
            - otherwise: ``flow`` -- the displacement field ``[B, ndim, *spatial]`` (voxel units).
        """
        mov, fix = sample
        F = self.encoder(fix)
        M = self.encoder(mov)
        composed_grids = self.decoder(F, M)
        if self.return_orginal:
            if not hasattr(self, 'generate_results'):
                raise NotImplementedError(
                    "return_orginal=True needs generate_results(), which is not part of the MIR port of VFA; "
                    "construct the model with return_orginal=False"
                )
            results = self.generate_results(composed_grids[-1], sample)
            results.update(
                {
                    'composed_grids': composed_grids,
                    'beta': self.decoder.beta.clone(),
                }
            )

            if self.configs.affine:
                affine_results = self.generate_affine_results(
                    composed_grids[-1],
                    sample,
                )
                results.update(affine_results)

            return results
        if self.return_all_flows:
            composed_flows = []
            for i in range(len(composed_grids)):
                composed_flows.append(grid_to_flow(composed_grids[i]))
            return composed_flows
        flow = grid_to_flow(composed_grids[-1])
        if self.SVF:
            pos_flow = self.integrate(flow)
            if self.return_full:
                neg_flow = self.integrate(-flow)  # inverse of exp(v) is exp(-v)
                def_sor = self.spatial_trans(mov, pos_flow)
                def_tar = self.spatial_trans(fix, neg_flow)
                return def_sor, def_tar, pos_flow, neg_flow
            return pos_flow
        return flow


class SPRDecoder(nn.Module):
    """Spatially varying regularization head.

    Inputs:
        M_feat: Moving-image features.
        F_feat: Fixed-image features.

    Returns:
        Tensor of spatial weights in [1e-6, 1] with shape ``[B, 1, *spatial]``.

    Args:
        dimension (int): Spatial dimensionality (2 or 3).
        start_channels (int): Base channel count; the concatenated inputs must have ``4 * start_channels``
            channels.
    """

    def __init__(self, dimension, start_channels):
        super().__init__()
        self.dim = dimension
        DoubleConv = globals()[f'DoubleConv{self.dim}d']
        self.SPR_conv = DoubleConv(
            in_channels=start_channels * 4,
            mid_channels=start_channels * 2,
            out_channels=start_channels,
        )
        self.lambda_out = getattr(nn, f'Conv{self.dim}d')(start_channels, 1, kernel_size=3, padding=1)
        self.wts_act = nn.Sigmoid()
        self.eps = 1e-6

    def forward(self, M_feat: torch.Tensor, F_feat: torch.Tensor) -> torch.Tensor:
        """Predict spatially varying regularization weights.

        Args:
            M_feat: Moving-image features of shape ``[B, 2 * start_channels, *spatial]``.
            F_feat: Fixed-image features of shape ``[B, 2 * start_channels, *spatial]``.

        Returns:
            torch.Tensor: Sigmoid weights clamped to [1e-6, 1], shape ``[B, 1, *spatial]``.
        """
        x = torch.cat((M_feat, F_feat), dim=1)
        x = self.SPR_conv(x)
        lambda_out = self.lambda_out(x)
        spatial_wts = self.wts_act(lambda_out)
        spatial_wts = torch.clamp(spatial_wts, self.eps, 1.0)
        return spatial_wts


class Encoder(nn.Module):
    """VFA encoder network.

    Inputs:
        x: Tensor of shape ``[B, C, *spatial]``.

    Returns:
        List of multiscale feature maps ordered from fine to coarse.

    Args:
        dimension (int): Spatial dimensionality (2 or 3).
        in_channels (int): Number of input image channels.
        downsamples (int): Number of 2x downsampling levels.
        start_channels (int): Channel count of the first convolution; doubled at each level.
        max_channels (int): Upper bound on the channel count.
    """

    def __init__(self, dimension, in_channels, downsamples, start_channels, max_channels):
        super().__init__()
        self.dim = dimension
        self.downsamples = downsamples
        self.encoder_conv = nn.ModuleList()
        self.pooling = nn.ModuleList()
        self.decoder_conv = nn.ModuleList()
        self.upsampling = nn.ModuleList()

        Conv = getattr(nn, f'Conv{self.dim}d')
        Norm = getattr(nn, f'InstanceNorm{self.dim}d')
        Pooling = getattr(nn, f'AvgPool{self.dim}d')
        DoubleConv = globals()[f'DoubleConv{self.dim}d']
        Upsample = globals()[f'Upsample{self.dim}d']

        self.in_norm = Norm(in_channels, affine=True)
        self.in_conv = Conv(in_channels, start_channels, 3, padding=1)
        for i in range(self.downsamples):
            num_channels = start_channels * 2**i
            self.encoder_conv.append(
                DoubleConv(
                    in_channels=min(max_channels, num_channels),
                    mid_channels=min(max_channels, num_channels * 2),
                    out_channels=min(max_channels, num_channels * 2),
                )
            )
            self.pooling.append(Pooling(2, stride=2))
            self.upsampling.append(
                Upsample(
                    in_channels=min(max_channels, num_channels * 4),
                    out_channels=min(max_channels, num_channels * 2),
                )
            )
            self.decoder_conv.append(
                DoubleConv(
                    in_channels=min(max_channels, num_channels * 2) * 2,
                    mid_channels=min(max_channels, num_channels * 2),
                    out_channels=min(max_channels, num_channels * 2),
                )
            )

        # bottleneck
        num_channels = start_channels * 2**self.downsamples
        self.bottleneck_conv = DoubleConv(
            in_channels=min(max_channels, num_channels),
            mid_channels=min(max_channels, num_channels * 2),
            out_channels=min(max_channels, num_channels * 2),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Encode an image with a U-Net and return the decoder feature maps.

        Args:
            x: Tensor of shape ``[B, C, *spatial]``; each spatial size must be divisible by ``2**downsamples``.

        Returns:
            list[torch.Tensor]: ``downsamples + 1`` feature maps ordered from fine (input resolution) to coarse
            (bottleneck, ``spatial / 2**downsamples``).
        """
        x = self.in_norm(x)
        x = nnf.leaky_relu(self.in_conv(x))
        feature_maps = []
        for i in range(self.downsamples):
            feature_maps.append(self.encoder_conv[i](x))
            x = self.pooling[i](feature_maps[-1])

        out_feature_maps = [self.bottleneck_conv(x)]

        for i in range(self.downsamples - 1, -1, -1):
            x = self.upsampling[i](out_feature_maps[-1], feature_maps[i])
            out_feature_maps.append(self.decoder_conv[i](x))

        return out_feature_maps[::-1]


class Decoder(nn.Module):
    """VFA decoder network.

    Inputs:
        F: List of fixed-image feature maps.
        M: List of moving-image feature maps.

    Returns:
        List of composed grids at each scale.

    Args:
        dimension (int): Spatial dimensionality (2 or 3).
        downsamples (int): Number of downsampling levels of the encoder.
        matching_channels (int): Base channel count of the projected matching features; doubled at each level.
        start_channels (int): Base channel count of the encoder.
        max_channels (int): Upper bound on the channel count.
        skip (bool): If True, skip matching at the finest level and only upsample the previous grid.
        initialize (float): Initial value of the learnable displacement scale ``beta``.
        int_steps (int): Number of self-composition (scaling and squaring) steps applied to each local grid.
    """

    def __init__(
        self, dimension, downsamples, matching_channels, start_channels, max_channels, skip, initialize, int_steps
    ):
        super().__init__()
        self.dim = dimension
        self.downsamples = downsamples
        self.project = nn.ModuleList()
        self.skip = skip
        self.int_steps = int_steps

        self.attention = Attention()
        Conv = getattr(nn, f'Conv{self.dim}d')
        self.beta = nn.Parameter(torch.tensor([float(initialize)]))

        self.similarity = 'inner_product'
        self.temperature = None

        for i in range(self.downsamples + 1):
            num_channels = start_channels * 2**i
            self.project.append(
                Conv(
                    min(max_channels, num_channels * 2),
                    min(max_channels, matching_channels * 2**i),
                    kernel_size=3,
                    padding=1,
                )
            )

        # precompute radial vector field r and R
        r = identity_grid_like(torch.zeros(1, 1, *[3 for _ in range(self.dim)]), normalize=True)
        self.register_buffer('R', self.to_token(r).squeeze().detach(), persistent=False)

    def to_token(self, x):
        """Flatten spatial dimensions and move channels to the last axis.

        Args:
            x: Tensor of shape ``[B, C, *spatial]``.

        Returns:
            Tensor of shape [B, N, C] where N is the flattened spatial size.
        """
        x = x.flatten(start_dim=2)
        x = x.transpose(-1, -2)
        return x

    def forward(self, F: List[torch.Tensor], M: List[torch.Tensor]) -> List[torch.Tensor]:
        """Estimate coarse-to-fine sampling grids by local feature matching.

        At each level, fixed-image features (queries) are matched against the 3^ndim neighbourhood of the
        moving-image features warped by the current grid (keys); the attention-weighted radial vectors give a
        local displacement, scaled by ``beta / 2**int_steps``, integrated by self-composition and composed with
        the upsampled grid of the previous level.

        Args:
            F: Fixed-image feature maps from :class:`Encoder`, fine to coarse.
            M: Moving-image feature maps from :class:`Encoder`, fine to coarse.

        Returns:
            list[torch.Tensor]: Composed absolute sampling grids (voxel coordinates), ordered coarse to fine, each
            of shape ``[B, ndim, *spatial_i]``.
        """
        composed_grids = []
        for i in range(self.downsamples, -1, -1):
            if i != self.downsamples:
                composed_grids[-1] = grid_upsample(composed_grids[-1])

            if i == 0 and self.skip:
                identity_grid = identity_grid_like(composed_grids[-1], normalize=False)
                composed_grids.append(grid_sampler(composed_grids[-1], identity_grid))
            else:
                # Q from the fixed image feature
                f = self.project[i](F[i])
                if self.similarity == 'cosine':
                    f = nnf.normalize(f, dim=1)
                permute_order = [0] + list(range(2, 2 + self.dim)) + [1]
                Q = f.permute(*permute_order).unsqueeze(-2)

                # K from the moving image feature maps
                m = grid_sampler(M[i], composed_grids[-1]) if len(composed_grids) != 0 else M[i]
                pad_size = [1 for _ in range(self.dim * 2)]
                m = _amp.pad(m, pad=tuple(pad_size), mode='replicate')
                m = self.project[i](m)
                if self.similarity == 'cosine':
                    m = nnf.normalize(m, dim=1)
                K = self.get_candidate_from_tensor(m, self.dim)

                # feature matching and location retrieval
                local_disp = self.attention(Q, K, self.R, self.temperature)
                permute_order = [0, self.dim + 1] + list(range(1, 1 + self.dim))  # non-negative axes (ONNX)
                local_disp = local_disp.squeeze(-2).permute(*permute_order)
                identity_grid = identity_grid_like(local_disp, normalize=False)
                local_grid = local_disp * self.beta / 2**self.int_steps + identity_grid

                for _ in range(self.int_steps):
                    local_grid = grid_sampler(local_grid, local_grid)

                if i != self.downsamples:
                    composed_grids.append(grid_sampler(composed_grids[-1], local_grid))
                else:
                    composed_grids.append(local_grid.clone())

        return composed_grids

    def get_candidate_from_tensor(self, x, dim, kernel=3, stride=1):
        """Extract local patches into a token tensor.

        Args:
            x: Tensor of shape ``[B, C, *spatial]``.
            dim: Spatial dimensionality (2 or 3).
            kernel: Patch size.
            stride: Patch stride.

        Returns:
            Tensor of shape ``[B, *spatial_out, P, C]`` with patch tokens, where ``P = kernel**dim`` and
            ``spatial_out = (spatial - kernel) // stride + 1``. During ONNX export (or when ``export_friendly`` is
            set) with ``stride == 1``, an equivalent slicing-based implementation is used.
        """
        if stride == 1 and (torch.onnx.is_in_onnx_export() or getattr(self, 'export_friendly', False)):
            return _candidates_by_slicing(x, dim, kernel)
        if dim == 3:
            '''from tensor with [Batch x Feature x Height x Weight x Depth],
                    extract patches [Batch x Feature x HxWxD x Patch],
                    and reshape to [Batch x HxWxS x Patch x Feature]'''
            patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride).unfold(4, kernel, stride)
            patches = patches.flatten(start_dim=5)
            token = patches.permute(0, 2, 3, 4, 5, 1)
        elif dim == 2:
            '''From tensor with [Batch x Feature x Height x Weight],
                    extract patches [Batch x Feature x HxW x Patch],
                    and reshape to [Batch x HxW x Patch x Feature]'''
            patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
            patches = patches.flatten(start_dim=4)
            token = patches.permute(0, 2, 3, 4, 1)

        return token


class DoubleConv3d(nn.Module):
    """Two-layer 3D convolutional block with instance norm and LeakyReLU.

    Args:
        in_channels (int): Number of input channels.
        mid_channels (int): Number of channels after the first convolution.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, mid_channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(mid_channels, affine=True)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv -> instance norm -> LeakyReLU twice.

        Args:
            x: Tensor of shape ``[B, in_channels, *spatial]``.

        Returns:
            torch.Tensor: Tensor of shape ``[B, out_channels, *spatial]``.
        """
        x = nnf.leaky_relu(self.norm1(self.conv1(x)))
        x = nnf.leaky_relu(self.norm2(self.conv2(x)))
        return x


class DoubleConv2d(nn.Module):
    """Two-layer 2D convolutional block with instance norm and LeakyReLU.

    Args:
        in_channels (int): Number of input channels.
        mid_channels (int): Number of channels after the first convolution.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(mid_channels, affine=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv -> instance norm -> LeakyReLU twice.

        Args:
            x: Tensor of shape ``[B, in_channels, *spatial]``.

        Returns:
            torch.Tensor: Tensor of shape ``[B, out_channels, *spatial]``.
        """
        x = nnf.leaky_relu(self.norm1(self.conv1(x)))
        x = nnf.leaky_relu(self.norm2(self.conv2(x)))
        return x


class Attention(nn.Module):
    """Scaled dot-product attention used for feature matching."""

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, temperature):
        """Apply attention.

        Args:
            query: Query tensor of shape [B, ..., F].
            key: Key tensor of shape [B, ..., F].
            value: Value tensor of shape [B, ..., F].
            temperature: Optional scaling factor; defaults to ``sqrt(key.size(-1))`` when None.

        Returns:
            Tensor of attended values.
        """
        if temperature is None:
            temperature = key.size(-1) ** 0.5
        attention = torch.matmul(query, key.transpose(-1, -2)) / temperature
        attention = self.softmax(attention)
        x = torch.matmul(attention, value)
        return x


class Upsample3d(nn.Module):
    """3D upsampling block with convolution and skip concatenation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels after the convolution (before concatenation).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        """Upsample ``x`` by 2, apply conv -> instance norm -> LeakyReLU and concatenate the skip features.

        Args:
            x: Tensor of shape ``[B, in_channels, *spatial]``.
            feature_map: Skip features of shape ``[B, C_skip, *(2 * spatial)]``.

        Returns:
            torch.Tensor: Tensor of shape ``[B, out_channels + C_skip, *(2 * spatial)]``.
        """
        x = _amp.interpolate(
            x,
            size=None,
            scale_factor=2,
            mode='trilinear',
            align_corners=True,
        )
        x = nnf.leaky_relu(self.norm(self.conv(x)))
        x = torch.cat((x, feature_map), dim=1)
        return x


class Upsample2d(nn.Module):
    """2D upsampling block with convolution and skip concatenation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels after the convolution (before concatenation).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        """Upsample ``x`` by 2, apply conv -> instance norm -> LeakyReLU and concatenate the skip features.

        Args:
            x: Tensor of shape ``[B, in_channels, *spatial]``.
            feature_map: Skip features of shape ``[B, C_skip, *(2 * spatial)]``.

        Returns:
            torch.Tensor: Tensor of shape ``[B, out_channels + C_skip, *(2 * spatial)]``.
        """
        x = _amp.interpolate(
            x,
            size=None,
            scale_factor=2,
            mode='bilinear',
            align_corners=True,
        )
        x = nnf.leaky_relu(self.norm(self.conv(x)))
        x = torch.cat((x, feature_map), dim=1)
        return x
