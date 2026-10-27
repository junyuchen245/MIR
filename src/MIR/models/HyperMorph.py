"""HyperMorph: hyperparameter-conditioned VoxelMorph.

Original code retrieved from:
https://github.com/voxelmorph/voxelmorph

Original paper:

Hoopes, Andrew, et al.
"Hypermorph: Amortized hyperparameter learning for image registration."
Information Processing in Medical Imaging: 27th International Conference,
IPMI 2021, Virtual Event, June 28–June 30, 2021, Proceedings 27.
Springer International Publishing, 2021.

Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., & Dalca, A. V. (2019).
VoxelMorph: a learning framework for deformable medical image registration.
IEEE transactions on medical imaging, 38(8), 1788-1800.

Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf

import numpy as np
from torch.distributions.normal import Normal
import MIR.models.registration_utils as utils
from typing import Tuple


class HyperBlocks(nn.Module):
    """MLP that embeds hyperparameters for hypernetwork conditioning.

    Args:
        nb_hyp_params: Number of input hyperparameters.
        nb_hyp_layers: Number of ``Linear`` + ReLU layers.
        nb_hyp_units: Width of every layer (embedding size).
    """

    def __init__(self, nb_hyp_params=1, nb_hyp_layers=6, nb_hyp_units=128):
        super().__init__()
        self.fcs = nn.ModuleList()
        hyp_last = nb_hyp_params
        for _ in range(nb_hyp_layers):
            self.fcs.append(nn.Linear(hyp_last, nb_hyp_units))
            hyp_last = nb_hyp_units

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project hyperparameters into conditioning features.

        Args:
            x: Hyperparameter values of shape ``[..., nb_hyp_params]``.

        Returns:
            torch.Tensor: Embedding of shape ``[..., nb_hyp_units]``.
        """
        for fc in self.fcs:
            x = fc(x)
            x = torch.relu(x)
        return x


class ConvBlock_(nn.Module):
    """Convolutional block with hyperparameter-conditioned bias.

    A regular bias-free convolution (kernel 3, padding 1) whose bias is predicted
    from the hyperparameter embedding, followed by LeakyReLU(0.2).

    Args:
        ndims: Number of spatial dimensions.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Convolution stride.
        nb_hyp_units: Size of the hyperparameter embedding.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1, nb_hyp_units=128):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1, bias=False)
        self.activation = nn.LeakyReLU(0.2)
        self.linear = nn.Linear(nb_hyp_units, out_channels)

    def forward(self, x: torch.Tensor, hyp_feat: torch.Tensor) -> torch.Tensor:
        """Apply convolution with hypernetwork bias.

        The bias is reshaped with three trailing singleton axes, i.e. a 3D input
        ``[B, C, H, W, D]`` is assumed.

        Args:
            x: Feature tensor.
            hyp_feat: Hyperparameter embedding of shape ``[B or 1, nb_hyp_units]``.

        Returns:
            Tensor after conditioning and activation.

        """
        out = self.main(x)
        bias = self.linear(hyp_feat).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        out = self.activation(out + bias)
        return out


class ConvBlock(nn.Module):
    """Convolutional block with hypernetwork-generated weights.

    Kernel (3x3x3, padding 1) and bias are predicted from the hyperparameter
    embedding, followed by LeakyReLU(0.2). The forward pass always uses
    ``conv3d``.

    Args:
        ndims: Number of spatial dimensions (used to size the weight generator).
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Convolution stride.
        nb_hyp_units: Size of the hyperparameter embedding.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1, nb_hyp_units=128):
        super().__init__()

        # self.linear = nn.Linear(nb_hyp_units, nb_hyp_units)
        self.activation = nn.LeakyReLU(0.2)
        self.linear_conv = nn.Linear(nb_hyp_units, 3**ndims * out_channels * in_channels, bias=False)
        self.linear_bias = nn.Linear(nb_hyp_units, out_channels, bias=False)
        # self.linear_conv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.linear_conv.weight.shape))
        # self.linear_bias.weight = nn.Parameter(Normal(0, 1e-5).sample(self.linear_bias.weight.shape))
        self.stride = stride
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor, hyp_feat: torch.Tensor) -> torch.Tensor:
        """Apply hypernetwork-generated convolution.

        Args:
            x: Feature tensor of shape ``[B, in_channels, H, W, D]``.
            hyp_feat: Hyperparameter embedding of shape ``[1, nb_hyp_units]``
                (one kernel is generated and shared across the batch).

        Returns:
            Tensor after conditioning and activation.

        """
        # hyp_feat = self.linear(hyp_feat)
        # hyp_feat = torch.relu(hyp_feat)
        kernel = self.linear_conv(hyp_feat).reshape([self.out_channels, self.in_channels, 3, 3, 3])
        # self.main.weight.data = kernel
        # out = self.main(x)
        out = nnf.conv3d(x, kernel, stride=self.stride, padding=1)
        bias = self.linear_bias(hyp_feat).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        out = self.activation(out + bias)
        return out


class CustomConv(nn.Module):
    """Convolution with weights predicted by a hypernetwork.

    Like :class:`ConvBlock` but without activation, and the weight generators
    are initialized from ``N(0, 1e-5)`` so the initial output is near zero.

    Args:
        ndims: Number of spatial dimensions (used to size the weight generator).
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Convolution stride.
        nb_hyp_units: Size of the hyperparameter embedding.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1, nb_hyp_units=128):
        super().__init__()

        # self.linear = nn.Linear(nb_hyp_units, nb_hyp_units)
        self.linear_conv = nn.Linear(nb_hyp_units, 3**ndims * out_channels * in_channels, bias=False)
        self.linear_bias = nn.Linear(nb_hyp_units, out_channels, bias=False)
        self.linear_conv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.linear_conv.weight.shape))
        self.linear_bias.weight = nn.Parameter(Normal(0, 1e-5).sample(self.linear_bias.weight.shape))
        self.stride = stride
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor, hyp_feat: torch.Tensor) -> torch.Tensor:
        """Apply hypernetwork-generated convolution.

        Args:
            x: Feature tensor of shape ``[B, in_channels, H, W, D]``.
            hyp_feat: Hyperparameter embedding of shape ``[1, nb_hyp_units]``.

        Returns:
            Tensor after applying predicted weights and bias.

        """
        # out = self.main(x)
        # hyp_feat = self.linear(hyp_feat)
        # hyp_feat = torch.relu(hyp_feat)
        kernel = self.linear_conv(hyp_feat).reshape([self.out_channels, self.in_channels, 3, 3, 3])
        out = nnf.conv3d(x, kernel, stride=self.stride, padding=1)
        bias = self.linear_bias(hyp_feat).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        out = out + bias
        return out


class Unet(nn.Module):
    """U-Net backbone with hyperparameter conditioning.

    All convolutions are hypernetwork-generated :class:`ConvBlock` layers.

    Args:
        inshape: Spatial input shape, e.g. (192, 192, 192).
        nb_features: Feature specification for encoder/decoder.
        nb_levels: Number of U-Net levels when `nb_features` is int.
        feat_mult: Per-level feature multiplier when `nb_features` is int.

    Inputs:
        x: Tensor of shape ``[B, 2, *spatial]`` with concatenated images.
        h: Hyperparameter embedding.

    Returns:
        Tensor of decoded features at full resolution.

    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        """Initialize the hyper-conditioned U-Net.

        Args:
            inshape: Spatial input shape, e.g. (192, 192, 192).
            nb_features: Feature specification for encoder/decoder: a pair
                ``(enc_nf, dec_nf)`` or an int. Defaults to
                ``((8, 32, 32, 32), (32, 32, 32, 32, 32, 8, 8))``.
            nb_levels: Number of U-Net levels when `nb_features` is int.
            feat_mult: Per-level feature multiplier when `nb_features` is int.

        Raises:
            ValueError: If `nb_features` is an int and `nb_levels` is None, or
                `nb_levels` is given while `nb_features` is not an int.

        """
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = ((8, 32, 32, 32), (32, 32, 32, 32, 32, 8, 8))

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[: len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf) :]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Run U-Net forward pass with hyperparameter conditioning.

        Args:
            x: Tensor of shape ``[B, 2, H, W, D]``; spatial sizes must be
                divisible by ``2**len(enc_nf)``.
            h: Hyperparameter embedding of shape ``[1, nb_hyp_units]``.

        Returns:
            torch.Tensor: Features of shape ``[B, dec_nf[-1], H, W, D]``.
        """
        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1], h))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x, h)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x, h)

        return x


class HyperVxmDense(nn.Module):
    """HyperMorph variant of VoxelMorph with hyperparameter conditioning.

    Args:
        configs: Config object with VoxelMorph hyperparameters (``img_size``,
            ``nb_unet_features``, ``nb_unet_levels``, ``unet_feat_mult``,
            ``use_probs``).
        gen_output: If True, also return the warped moving image.

    Inputs:
        input_imgs: Tuple `(mov, fix)` of tensors ``[B, 1, *spatial]``.
        hyp_val: Tensor of hyperparameter values of shape ``[1, 1]``.

    Returns:
        Flow tensor, or ``(warped, flow)`` if `gen_output` is True.

    """

    def __init__(self, configs, gen_output=False):
        """Initialize HyperMorph with config settings.

        Args:
            configs: Config object with VoxelMorph hyperparameters.
            gen_output: If True, also return the warped moving image.

        Raises:
            NotImplementedError: If ``configs.use_probs`` is True.

        """
        super().__init__()
        inshape = configs.img_size
        nb_unet_features = configs.nb_unet_features
        nb_unet_levels = configs.nb_unet_levels
        unet_feat_mult = configs.unet_feat_mult
        use_probs = configs.use_probs
        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        self.hyper_model = HyperBlocks()

        # configure core unet model
        self.unet_model = Unet(
            inshape, nb_features=nb_unet_features, nb_levels=nb_unet_levels, feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        self.flow = CustomConv(ndims, self.unet_model.dec_nf[-1], ndims)

        # init flow layer with small weights and bias
        # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')
        self.gen_output = gen_output
        if self.gen_output:
            # configure transformer
            self.spatial_trans = utils.SpatialTransformer(inshape)

    def forward(self, input_imgs: Tuple[torch.Tensor, torch.Tensor], hyp_val: torch.Tensor):
        """Run HyperMorph forward pass.

        Args:
            input_imgs: Tuple `(mov, fix)` tensors of shape ``[B, 1, H, W, D]``.
            hyp_val: Hyperparameter tensor of shape ``[1, 1]`` (embedded once and
                shared across the batch).

        Returns:
            Displacement field ``[B, 3, H, W, D]`` (voxel units, not integrated);
            if `gen_output` is True, the tuple ``(warped_mov, flow)``.

        """
        mov, fix = input_imgs
        x = torch.cat((mov, fix), dim=1)
        h = self.hyper_model(hyp_val)

        # concatenate inputs and propagate unet
        x = self.unet_model(x, h)

        # transform into flow field
        flow_field = self.flow(x, h)

        # resize flow for integration
        pos_flow = flow_field
        if self.gen_output:
            # warp image with flow field
            def_x = self.spatial_trans(mov, pos_flow)
            return def_x, pos_flow
        return pos_flow
