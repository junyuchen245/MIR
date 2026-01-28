"""VFA models and utilities for deformable image registration."""

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import MIR.models.registration_utils as utils

def grid_upsample(grid, mode='bilinear', align_corners=True, scale_factor=2):
    """Upsample a sampling grid by a scale factor.

    Args:
        grid: Tensor of shape [B, C, *spatial] containing absolute coordinates.
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
    out = nnf.interpolate(
                            out,
                            scale_factor=scale_factor,
                            mode=mode,
                            align_corners=align_corners,
    )
    return out

def grid_sampler(source, grid, mode='bilinear', align_corners=True, normalize=True, padding_mode='border'):
    """Sample `source` at coordinates defined by `grid`.

    Args:
        source: Tensor of shape [B, C, *spatial].
        grid: Tensor of shape [B, ndim, *spatial] with absolute coordinates.
        mode: Interpolation mode for sampling.
        align_corners: Whether to align corners in grid sampling.
        normalize: If True, convert `grid` to normalized [-1, 1] coordinates.
        padding_mode: Padding mode for out-of-bound values.

    Returns:
        Warped tensor sampled from `source` at `grid` locations.
    """
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
    return warped

def normalize_grid(grid):
    """Normalize an absolute coordinate grid to [-1, 1] for grid sampling.

    Args:
        grid: Tensor of shape [B, ndim, *spatial] with absolute coordinates.

    Returns:
        Tensor of the same shape with normalized coordinates.
    """
    dims = torch.tensor(grid.shape[2:]).to(grid.device)
    dims = dims.view(-1, len(dims), *[1 for x in range(len(dims))])
    normed_grid = grid / (dims - 1) * 2 - 1 # convert to normalized space
    return normed_grid

def identity_grid_like(tensor, normalize, padding=0):
    """Return an identity grid matching a reference tensor.

    Args:
        tensor: Reference tensor of shape [B, C, *spatial].
        normalize: If True, return normalized [-1, 1] coordinates.
        padding: Int or sequence of per-dimension padding for the grid.

    Returns:
        Tensor of shape [B, ndim, *spatial] with identity coordinates.
    """
    with torch.inference_mode():
        dims = tensor.shape[2:]
        if isinstance(padding, int):
            pads = [padding for j in range(len(dims))]
        else:
            pads = padding
        vectors = [torch.arange(start=0-pad, end=dim+pad) for (dim,pad) in zip(dims, pads)]

        try:
            grids = torch.meshgrid(vectors, indexing='ij')
        except TypeError:
            # compatible with old pytorch version
            grids = torch.meshgrid(vectors)
        grid = torch.stack(grids).unsqueeze(0).type(torch.float32)

        if normalize:
            in_dims = torch.tensor(dims).view(-1, len(dims), *[1 for x in range(len(dims))])
            grid = grid / (in_dims - 1) * 2 - 1

        grid = grid.to(tensor.device).repeat(tensor.shape[0],1,*[1 for j in range(len(dims))])
    return grid.clone()

def grid_to_flow(grid: torch.Tensor):
    """
    Convert an absolute sampling grid (as produced by VFA) to a
    voxel‑displacement field that VoxelMorph’s SpatialTransformer expects.

    Args:
        grid: Tensor of shape [B, ndim, *spatial_dims] with absolute coords.

    Returns:
        Tensor of the same shape containing displacements from identity.
    """
    # build an identity grid on the same device / dtype / size
    id_grid = identity_grid_like(grid, normalize=False)  # already channels‑first
    flow    = grid - id_grid
    return flow

class VFASPR(nn.Module):
    """VFA model with spatially varying regularization (VFA-SPR).

    Args:
        configs: VFA configuration object.
        device: Device to run the model on.
        return_orginal: If True, return VFA-style composed grids and stats.
        return_all_flows: If True, return flows for all decoder levels.
        SVF: If True, integrate flow as stationary velocity field.
        SVF_steps: Number of scaling-and-squaring steps for integration.
        return_full: If True, also return warped images and inverse flow.

    Forward inputs:
        sample: Tuple `(mov, fix)` tensors of shape [B, 1, *spatial].

    Forward outputs:
        Depending on flags, returns flow(s), spatial weights, or a results dict.
    """
    def __init__(self, configs, device, return_orginal=False, return_all_flows=False, SVF=False, SVF_steps=7, return_full=False):
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
            self.spatial_trans = utils.SpatialTransformer(configs.img_size).cuda()

    def forward(self, sample):
        """Run forward registration.

        Args:
            sample: Tuple `(mov, fix)` tensors.

        Returns:
            Output varies by flags (`return_orginal`, `return_all_flows`, `SVF`,
            `return_full`). See class docstring for details.
        """
        mov, fix = sample
        F = self.encoder(fix)
        M = self.encoder(mov)
        spatial_wts = self.SPRDecoder(M[0], F[0])
        composed_grids = self.decoder(F, M)
        if self.return_orginal:
            results = self.generate_results(composed_grids[-1], sample)
            results.update({'composed_grids': composed_grids,
                            'beta':self.decoder.beta.clone(),})

            if self.configs.affine:
                affine_results = self.generate_affine_results(
                                                        composed_grids[-1],
                                                        sample,
                )
                results.update(affine_results)
        
            return results
        elif self.return_all_flows:
            composed_flows = []
            for i in range(len(composed_grids)):
                composed_flows.append(grid_to_flow(composed_grids[i]))
            return composed_flows, spatial_wts
        else:
            flow = grid_to_flow(composed_grids[-1])
            if self.SVF:
                pos_flow = self.integrate(flow)
                if self.return_full:
                    neg_flow = -pos_flow
                    neg_flow = self.integrate(neg_flow)
                    def_sor = self.spatial_trans(mov, pos_flow)
                    def_tar = self.spatial_trans(fix, neg_flow)
                    return def_sor, def_tar, pos_flow, neg_flow, spatial_wts
                return pos_flow, spatial_wts
            return flow, spatial_wts

class VFA(nn.Module):
    """VFA model for image registration.

    Args:
        configs: VFA configuration object.
        device: Device to run the model on.
        return_orginal: If True, return VFA-style composed grids and stats.
        return_all_flows: If True, return flows for all decoder levels.
        SVF: If True, integrate flow as stationary velocity field.
        SVF_steps: Number of scaling-and-squaring steps for integration.
        return_full: If True, also return warped images and inverse flow.

    Forward inputs:
        sample: Tuple `(mov, fix)` tensors of shape [B, 1, *spatial].

    Forward outputs:
        Depending on flags, returns flow(s) or a results dict.
    """
    def __init__(self, configs, device, return_orginal=False, return_all_flows=False, SVF=False, SVF_steps=7, return_full=False):
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
            self.spatial_trans = utils.SpatialTransformer(configs.img_size).cuda()

    def forward(self, sample):
        """Run forward registration.

        Args:
            sample: Tuple `(mov, fix)` tensors.

        Returns:
            Output varies by flags (`return_orginal`, `return_all_flows`, `SVF`,
            `return_full`). See class docstring for details.
        """
        mov, fix = sample
        F = self.encoder(fix)
        M = self.encoder(mov)
        composed_grids = self.decoder(F, M)
        if self.return_orginal:
            results = self.generate_results(composed_grids[-1], sample)
            results.update({'composed_grids': composed_grids,
                            'beta':self.decoder.beta.clone(),})

            if self.configs.affine:
                affine_results = self.generate_affine_results(
                                                        composed_grids[-1],
                                                        sample,
                )
                results.update(affine_results)
        
            return results
        elif self.return_all_flows:
            composed_flows = []
            for i in range(len(composed_grids)):
                composed_flows.append(grid_to_flow(composed_grids[i]))
            return composed_flows
        else:
            flow = grid_to_flow(composed_grids[-1])
            if self.SVF:
                pos_flow = self.integrate(flow)
                if self.return_full:
                    neg_flow = -pos_flow
                    neg_flow = self.integrate(neg_flow)
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
        Tensor of spatial weights in [0, 1] with shape [B, 1, *spatial].
    """
    def __init__(self, dimension, start_channels):
        super().__init__()
        self.dim = dimension
        DoubleConv = globals()[f'DoubleConv{self.dim}d']
        self.SPR_conv = DoubleConv(
                in_channels= start_channels * 4,
                mid_channels=start_channels * 2,
                out_channels=start_channels,
            )
        self.lambda_out = nn.Conv3d(start_channels, 1, kernel_size=3, padding=1)
        self.wts_act = nn.Sigmoid()
        self.eps = 1e-6
    def forward(self, M_feat, F_feat):
        x = torch.cat((M_feat, F_feat), dim=1)
        x = self.SPR_conv(x)
        lambda_out = self.lambda_out(x)
        spatial_wts = self.wts_act(lambda_out)
        spatial_wts = torch.clamp(spatial_wts, self.eps, 1.0)
        return spatial_wts
    
class Encoder(nn.Module):
    """VFA encoder network.

    Inputs:
        x: Tensor of shape [B, C, *spatial].

    Returns:
        List of multiscale feature maps ordered from fine to coarse.
    """
    def __init__(self, dimension, in_channels, downsamples, start_channels, max_channels):
        super().__init__()
        self.dim = dimension
        self.downsamples = downsamples
        self.encoder_conv = nn.ModuleList()
        self.pooling =  nn.ModuleList()
        self.decoder_conv = nn.ModuleList()
        self.upsampling =  nn.ModuleList()

        Conv = getattr(nn, f'Conv{self.dim}d')
        Norm = getattr(nn, f'InstanceNorm{self.dim}d')
        Pooling = getattr(nn, f'AvgPool{self.dim}d')
        DoubleConv = globals()[f'DoubleConv{self.dim}d']
        Upsample = globals()[f'Upsample{self.dim}d']

        self.in_norm = Norm(in_channels, affine=True)
        self.in_conv = Conv(in_channels, start_channels, 3, padding=1)
        for i in range(self.downsamples):
            num_channels = start_channels * 2 ** i
            self.encoder_conv.append(DoubleConv(
                in_channels=min(max_channels, num_channels),
                mid_channels=min(max_channels, num_channels*2),
                out_channels=min(max_channels, num_channels*2)
            ))
            self.pooling.append(Pooling(2, stride=2))
            self.upsampling.append(Upsample(
                in_channels=min(max_channels, num_channels*4),
                out_channels=min(max_channels, num_channels*2),
            ))
            self.decoder_conv.append(DoubleConv(
                in_channels=min(max_channels, num_channels*2) * 2,
                mid_channels=min(max_channels, num_channels*2),
                out_channels=min(max_channels, num_channels*2),
            ))

        # bottleneck
        num_channels = start_channels * 2 ** self.downsamples
        self.bottleneck_conv = DoubleConv(
            in_channels=min(max_channels, num_channels),
            mid_channels=min(max_channels, num_channels*2),
            out_channels=min(max_channels, num_channels*2),
        )

    def forward(self, x):
        x = self.in_norm(x)
        x = nnf.leaky_relu(self.in_conv(x))
        feature_maps = []
        for i in range(self.downsamples):
            feature_maps.append(self.encoder_conv[i](x))
            x = self.pooling[i](feature_maps[-1])

        out_feature_maps = [self.bottleneck_conv(x)]

        for i in range(self.downsamples-1, -1, -1):
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
    """
    def __init__(self, dimension, downsamples, matching_channels, start_channels, max_channels,
                 skip, initialize, int_steps):
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
            num_channels = start_channels * 2 ** i
            self.project.append(Conv(
                            min(max_channels, num_channels*2),
                            min(max_channels, matching_channels*2**i),
                            kernel_size=3,
                            padding=1,
            ))

        # precompute radial vector field r and R
        r = identity_grid_like(
                torch.zeros(1, 1, *[3 for _ in range(self.dim)]),
                normalize=True
        )
        self.R = self.to_token(r).squeeze().detach()

    def to_token(self, x):
        """Flatten spatial dimensions and move channels to the last axis.

        Args:
            x: Tensor of shape [B, C, *spatial].

        Returns:
            Tensor of shape [B, N, C] where N is the flattened spatial size.
        """
        x = x.flatten(start_dim=2)
        x = x.transpose(-1, -2)
        return x

    def forward(self, F, M):
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
                m = nnf.pad(m, pad=tuple(pad_size), mode='replicate')
                m = self.project[i](m)
                if self.similarity == 'cosine':
                    m = nnf.normalize(m, dim=1)
                K = self.get_candidate_from_tensor(m, self.dim)

                # feature matching and location retrieval
                local_disp = self.attention(Q, K, self.R, self.temperature)
                permute_order = [0, -1] + list(range(1, 1 + self.dim))
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
            x: Tensor of shape [B, C, *spatial].
            dim: Spatial dimensionality (2 or 3).
            kernel: Patch size.
            stride: Patch stride.

        Returns:
            Tensor of shape [B, *spatial, P, C] with patch tokens.
        """
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
    """Two-layer 3D convolutional block with instance norm and LeakyReLU."""
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, mid_channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(mid_channels, affine=True)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        x = nnf.leaky_relu(self.norm1(self.conv1(x)))
        x = nnf.leaky_relu(self.norm2(self.conv2(x)))
        return x

class DoubleConv2d(nn.Module):
    """Two-layer 2D convolutional block with instance norm and LeakyReLU."""
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(mid_channels, affine=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
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
            temperature: Optional scaling factor.

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
    """3D upsampling block with convolution and skip concatenation."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x, feature_map):
        x = nnf.interpolate(
                x,
                size=None,
                scale_factor=2,
                mode='trilinear',
                align_corners=True,
        )
        x = nnf.leaky_relu(self.norm(self.conv(x)))
        x = torch.cat((x, feature_map),dim=1)
        return x

class Upsample2d(nn.Module):
    """2D upsampling block with convolution and skip concatenation."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x, feature_map):
        x = nnf.interpolate(
                        x,
                        size=None,
                        scale_factor=2,
                        mode='bilinear',
                        align_corners=True,
        )
        x = nnf.leaky_relu(self.norm(self.conv(x)))
        x = torch.cat((x, feature_map),dim=1)
        return x