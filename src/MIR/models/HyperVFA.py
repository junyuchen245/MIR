"""HyperVFA models and utilities for hyperparameter-conditioned registration."""

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal

class HyperConv(nn.Module):
    """Hyper convolutional layer with weights predicted from conditioning."""

    def __init__(self, in_channels, out_channels, ndims=3, kernel_size=3, stride=1, padding=1, bias=True, normalize=False, nb_hyp_units=96):
        super().__init__()
        self.linear_conv = nn.Linear(nb_hyp_units, kernel_size**ndims*out_channels*in_channels, bias=False)
        self.linear_bias = nn.Linear(nb_hyp_units, out_channels, bias=False)
        if normalize:
            self.linear_conv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.linear_conv.weight.shape))
            self.linear_bias.weight = nn.Parameter(Normal(0, 1e-5).sample(self.linear_bias.weight.shape))
        self.stride = stride
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.padding = padding
        self.ks = kernel_size
        self.if_bias = bias

    def forward(self, x, hyp_feat):
        """Apply hyper-conditioned convolution.

        Args:
            x: Feature tensor.
            hyp_feat: Hyperparameter embedding.

        Returns:
            Tensor after applying predicted weights and bias.
        """
        kernel = self.linear_conv(hyp_feat).reshape([self.out_channels, self.in_channels, self.ks, self.ks, self.ks])
        out = nnf.conv3d(x, kernel, stride=self.stride, padding=self.padding)
        if self.if_bias:
            bias= self.linear_bias(hyp_feat).unsqueeze(2).unsqueeze(2).unsqueeze(2)
            out = out+bias
        return out

class HyperBlocks(nn.Module):
    """MLP that embeds hyperparameters for hypernetwork conditioning."""
    def __init__(self, nb_hyp_params=1, nb_hyp_layers=6, nb_hyp_units=96):
        super().__init__()
        self.fcs = nn.ModuleList()
        hyp_last = nb_hyp_params
        for _ in range(nb_hyp_layers):
            self.fcs.append(nn.Linear(hyp_last, nb_hyp_units))
            hyp_last = nb_hyp_units

    def forward(self, x):
        """Project hyperparameters into conditioning features."""
        for fc in self.fcs:
            x = fc(x)
            x = torch.relu(x)
        return x

class HyperLinear(nn.Module):
    """Linear layer with weights predicted from hyperparameters."""
    def __init__(self, in_features, out_features, nb_hyp_units=96, bias=True):
        super().__init__()
        self.linear_wts = nn.Linear(nb_hyp_units, in_features * out_features, bias=False)
        self.linear_bias = nn.Linear(nb_hyp_units, out_features, bias=False)
        self.in_features = in_features
        self.out_features = out_features
        self.if_bias = bias

    def forward(self, x, h):
        """Apply hyper-conditioned linear projection."""
        weight = self.linear_wts(h).reshape([self.out_features, self.in_features])
        output = torch.matmul(x, weight.t())
        if self.if_bias:
            bias = self.linear_bias(h).reshape([self.out_features])
            output = output+bias
        return output

def grid_upsample(grid, mode='bilinear', align_corners=True, scale_factor=2):
    """Upsample a sampling grid by a scale factor.

    Args:
        grid: Tensor of shape [B, C, *spatial] containing absolute coordinates.
        mode: Interpolation mode.
        align_corners: Whether to align corners during interpolation.
        scale_factor: Scale factor to upsample spatial dimensions.

    Returns:
        Tensor with upsampled spatial dimensions.
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
    """Normalize an absolute coordinate grid to [-1, 1]."""
    dims = torch.tensor(grid.shape[2:]).to(grid.device)
    dims = dims.view(-1, len(dims), *[1 for x in range(len(dims))])
    normed_grid = grid / (dims - 1) * 2 - 1 # convert to normalized space
    return normed_grid

def identity_grid_like(tensor, normalize, padding=0):
    """Return an identity grid matching a reference tensor."""
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
    """Convert an absolute sampling grid to a displacement field.

    Args:
        grid: Tensor of shape [B, ndim, *spatial_dims] with absolute coords.

    Returns:
        Tensor of the same shape containing displacements from identity.
    """
    # build an identity grid on the same device / dtype / size
    id_grid = identity_grid_like(grid, normalize=False)  # already channels‑first
    flow    = grid - id_grid
    return flow

class HyperVFASPR(nn.Module):
    """HyperVFA model with spatially varying regularization.

    Args:
        configs: VFA configuration object.
        device: Device to run the model on.
        return_orginal: If True, return composed grids and stats.
        return_all_flows: If True, return flows for all decoder levels.

    Forward inputs:
        sample: Tuple `(mov, fix)` tensors.
        hyper_val: Hyperparameter tensor.

    Forward outputs:
        Flow(s) and spatial weights depending on flags.
    """
    def __init__(self, configs, device, return_orginal=False, return_all_flows=False):
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
        self.hyper_model = HyperBlocks(nb_hyp_params=2)
        
    def forward(self, sample, hyper_val):
        """Run forward registration.

        Args:
            sample: Tuple `(mov, fix)` tensors.
            hyper_val: Hyperparameter tensor.

        Returns:
            Output varies by flags (`return_orginal`, `return_all_flows`).
        """
        hyper_feat = self.hyper_model(hyper_val)
        mov, fix = sample
        F = self.encoder(fix, hyper_feat)
        M = self.encoder(mov, hyper_feat)
        spatial_wts = self.SPRDecoder(M[0], F[0], hyper_feat)
        composed_grids = self.decoder(F, M, hyper_feat)
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
            return grid_to_flow(composed_grids[-1]), spatial_wts

class HyperVFA(nn.Module):
    """Hyperparameter-conditioned VFA model.

    Args:
        configs: VFA configuration object.
        device: Device to run the model on.
        return_orginal: If True, return composed grids and stats.
        return_all_flows: If True, return flows for all decoder levels.

    Forward inputs:
        sample: Tuple `(mov, fix)` tensors.
        hyper_val: Hyperparameter tensor.

    Forward outputs:
        Flow(s) depending on flags.
    """
    def __init__(self, configs, device, return_orginal=False, return_all_flows=False):
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
        self.hyper_model = HyperBlocks()
        
    def forward(self, sample, hyper_val):
        """Run forward registration.

        Args:
            sample: Tuple `(mov, fix)` tensors.
            hyper_val: Hyperparameter tensor.

        Returns:
            Output varies by flags (`return_orginal`, `return_all_flows`).
        """
        hyper_feat = self.hyper_model(hyper_val)
        mov, fix = sample
        F = self.encoder(fix, hyper_feat)
        M = self.encoder(mov, hyper_feat)
        composed_grids = self.decoder(F, M, hyper_feat)
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
            return grid_to_flow(composed_grids[-1])
        
class SPRDecoder(nn.Module):
    """Spatially varying regularization head.

    Inputs:
        M_feat: Moving-image features.
        F_feat: Fixed-image features.
        hyper_feat: Hyperparameter embedding.

    Returns:
        Tensor of spatial weights in [0, 1].
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
        self.lambda_out = HyperConv(start_channels, 1, kernel_size=3, padding=1)
        self.wts_act = nn.Sigmoid()
        self.eps = 1e-6
    def forward(self, M_feat, F_feat, hyper_feat):
        x = torch.cat((M_feat, F_feat), dim=1)
        x = self.SPR_conv(x, hyper_feat)
        lambda_out = self.lambda_out(x, hyper_feat)
        spatial_wts = self.wts_act(lambda_out)
        spatial_wts = torch.clamp(spatial_wts, self.eps, 1.0)
        return spatial_wts
    
class Encoder(nn.Module):
    """HyperVFA encoder network.

    Inputs:
        x: Tensor of shape [B, C, *spatial].
        hyper_feat: Hyperparameter embedding.

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
        self.in_conv = HyperConv(in_channels, start_channels, kernel_size=3, ndims=self.dim, padding=1)
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

    def forward(self, x, hyper_feat):
        """Run encoder forward pass."""
        x = self.in_norm(x)
        x = nnf.leaky_relu(self.in_conv(x, hyper_feat))
        feature_maps = []
        for i in range(self.downsamples):
            feature_maps.append(self.encoder_conv[i](x, hyper_feat))
            x = self.pooling[i](feature_maps[-1])

        out_feature_maps = [self.bottleneck_conv(x, hyper_feat)]

        for i in range(self.downsamples-1, -1, -1):
            x = self.upsampling[i](out_feature_maps[-1], feature_maps[i], hyper_feat)
            out_feature_maps.append(self.decoder_conv[i](x, hyper_feat))

        return out_feature_maps[::-1]

class Decoder(nn.Module):
    """HyperVFA decoder network.

    Inputs:
        F: List of fixed-image feature maps.
        M: List of moving-image feature maps.
        hyper_feat: Hyperparameter embedding.

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
        self.beta = nn.Parameter(torch.tensor([float(initialize)]))

        self.similarity = 'inner_product'
        self.temperature = None

        for i in range(self.downsamples + 1):
            num_channels = start_channels * 2 ** i
            self.project.append(HyperConv(
                            min(max_channels, num_channels*2),
                            min(max_channels, matching_channels*2**i),
                            ndims=self.dim,
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
        """Flatten spatial dimensions and move channels to the last axis."""
        x = x.flatten(start_dim=2)
        x = x.transpose(-1, -2)
        return x

    def forward(self, F, M, hyper_feat):
        """Run decoder forward pass."""
        composed_grids = []
        for i in range(self.downsamples, -1, -1):
            if i != self.downsamples:
                composed_grids[-1] = grid_upsample(composed_grids[-1])

            if i == 0 and self.skip:
                identity_grid = identity_grid_like(composed_grids[-1], normalize=False)
                composed_grids.append(grid_sampler(composed_grids[-1], identity_grid))
            else:
                # Q from the fixed image feature
                f = self.project[i](F[i], hyper_feat)
                if self.similarity == 'cosine':
                    f = nnf.normalize(f, dim=1)
                permute_order = [0] + list(range(2, 2 + self.dim)) + [1]
                Q = f.permute(*permute_order).unsqueeze(-2)

                # K from the moving image feature maps
                m = grid_sampler(M[i], composed_grids[-1]) if len(composed_grids) != 0 else M[i]
                pad_size = [1 for _ in range(self.dim * 2)]
                m = nnf.pad(m, pad=tuple(pad_size), mode='replicate')
                m = self.project[i](m, hyper_feat)
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
        """Extract local patches into a token tensor."""
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
    """Two-layer 3D hyper-conv block with instance norm and LeakyReLU."""
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = HyperConv(in_channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(mid_channels, affine=True)
        self.conv2 = HyperConv(mid_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x, hyper_feat):
        x = nnf.leaky_relu(self.norm1(self.conv1(x, hyper_feat)))
        x = nnf.leaky_relu(self.norm2(self.conv2(x, hyper_feat)))
        return x

class DoubleConv2d(nn.Module):
    """Two-layer 2D hyper-conv block with instance norm and LeakyReLU."""
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = HyperConv(in_channels, mid_channels, ndims=2, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(mid_channels, affine=True)
        self.conv2 = HyperConv(mid_channels, out_channels, ndims=2, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x, hyper_feat):
        x = nnf.leaky_relu(self.norm1(self.conv1(x, hyper_feat)))
        x = nnf.leaky_relu(self.norm2(self.conv2(x, hyper_feat)))
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
    """3D upsampling block with hyper-conv and skip concatenation."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = HyperConv(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x, feature_map, hyper_feat):
        x = nnf.interpolate(
                x,
                size=None,
                scale_factor=2,
                mode='trilinear',
                align_corners=True,
        )
        x = nnf.leaky_relu(self.norm(self.conv(x, hyper_feat)))
        x = torch.cat((x, feature_map),dim=1)
        return x

class Upsample2d(nn.Module):
    """2D upsampling block with hyper-conv and skip concatenation."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = HyperConv(in_channels, out_channels, ndims=2, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x, feature_map, hyper_feat):
        x = nnf.interpolate(
                        x,
                        size=None,
                        scale_factor=2,
                        mode='bilinear',
                        align_corners=True,
        )
        x = nnf.leaky_relu(self.norm(self.conv(x, hyper_feat)))
        x = torch.cat((x, feature_map),dim=1)
        return x