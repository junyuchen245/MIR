'''
LessNet

Original code retrieved from:
https://github.com/xi-jia/LessNet

Modified for MIR by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''

import torch
import torch.nn as nn
import MIR.models.registration_utils as utils


class LessNetUNet(nn.Module):
    """LessNet 3-D U-Net backbone from the original implementation.

    Inputs:
        x: Moving image tensor of shape [B, C, D, H, W].
        y: Fixed image tensor of shape [B, C, D, H, W].

    Returns:
        Channel-first displacement field [B, 3, D, H, W] in the normalized
        grid-sampling coordinates predicted by the upstream LessNet code.
    """

    def __init__(self, image_channels=1, n_classes=3, start_channel=8, return_features=False):
        """Initialize LessNet.

        Args:
            image_channels: Number of channels per input image.
            n_classes: Number of output displacement channels.
            start_channel: Upstream LessNet start channel hyperparameter.
            return_features: If True, also return the final decoder feature map.
        """
        super().__init__()
        self.image_channels = image_channels
        self.in_channel = image_channels * 6
        self.n_classes = n_classes
        self.start_channel = start_channel
        self.return_features = return_features

        bias_opt = True

        self.eninput = self.encoder(self.in_channel, self.start_channel * 4, bias=bias_opt)

        self.dc1 = self.encoder(self.start_channel * 4, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc2 = self.encoder(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.dc3 = self.encoder(self.start_channel * 3 + 6 * image_channels, self.start_channel * 3, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc4 = self.encoder(self.start_channel * 3, self.start_channel * 3, kernel_size=3, stride=1, bias=bias_opt)
        self.dc5 = self.encoder(self.start_channel * 2 + 6 * image_channels, self.start_channel * 2, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc6 = self.encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc7 = self.encoder(self.start_channel * 1 + 2 * image_channels, self.start_channel * 1, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc8 = self.encoder(self.start_channel * 1, self.start_channel * 1, kernel_size=3, stride=1, bias=bias_opt)
        self.dc9 = self.outputs(self.start_channel * 1, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)

        self.up2 = self.decoder(self.start_channel * 4, self.start_channel * 3)
        self.up3 = self.decoder(self.start_channel * 3, self.start_channel * 2)
        self.up4 = self.decoder(self.start_channel * 2, self.start_channel * 1)

        self.maxp_1_2 = nn.MaxPool3d(2, stride=2)
        self.maxp_1_4 = nn.MaxPool3d(4, stride=4)
        self.maxp_1_8 = nn.MaxPool3d(8, stride=8)
        self.avgp_1_2 = nn.AvgPool3d(2, stride=2)
        self.avgp_1_4 = nn.AvgPool3d(4, stride=4)
        self.avgp_1_8 = nn.AvgPool3d(8, stride=8)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, batchnorm=False):
        """Create an upstream LessNet encoder block."""
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True):
        """Create an upstream LessNet decoder block."""
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.LeakyReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        """Create the upstream LessNet output head."""
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def _pooled_inputs(self, x, y, pool_scale):
        """Return average, max, and min pooled moving/fixed images."""
        avg_pool = getattr(self, 'avgp_1_%d' % pool_scale)
        max_pool = getattr(self, 'maxp_1_%d' % pool_scale)
        return (
            avg_pool(x),
            avg_pool(y),
            max_pool(x),
            max_pool(y),
            -max_pool(-x),
            -max_pool(-y),
        )

    def forward(self, x, y):
        """Run the original LessNet forward pass."""
        x_in = torch.cat(self._pooled_inputs(x, y, 8), 1)
        d0 = self.eninput(x_in)

        d0 = self.dc1(d0)
        d0 = self.dc2(d0)

        d1 = self.up2(d0)
        d1 = torch.cat((d1, *self._pooled_inputs(x, y, 4)), 1)

        d1 = self.dc3(d1)
        d1 = self.dc4(d1)
        d2 = self.up3(d1)

        d2 = torch.cat((d2, *self._pooled_inputs(x, y, 2)), 1)

        d2 = self.dc5(d2)
        d2 = self.dc6(d2)
        d3 = self.up4(d2)

        d3 = torch.cat((d3, x, y), 1)

        d3 = self.dc7(d3)
        d3 = self.dc8(d3)

        f_xy = self.dc9(d3)

        if self.return_features:
            return f_xy, d3
        return f_xy


class _LessNetBase(nn.Module):
    """Shared MIR wrapper utilities for LessNet variants."""

    def __init__(self, configs=None, gen_output=False, return_features=False):
        super().__init__()
        if configs is None:
            from MIR.models.configs_LessNet import get_LessNet_default_config
            configs = get_LessNet_default_config()

        self.configs = configs
        self.gen_output = gen_output
        self.SVF = bool(getattr(configs, 'SVF', False))
        self.n_dims = int(getattr(configs, 'n_dims', 3))
        if self.n_dims != 3:
            raise ValueError('LessNet upstream implementation is 3-D only.')

        self.image_channels = int(getattr(configs, 'image_channels', getattr(configs, 'in_channels', 1)))
        self.out_channels = int(getattr(configs, 'out_channels', 3))
        self.start_channel = int(getattr(configs, 'start_channel', 8))
        self.unet = LessNetUNet(
            image_channels=self.image_channels,
            n_classes=self.out_channels,
            start_channel=self.start_channel,
            return_features=return_features,
        )

        self.SVF_steps = int(getattr(configs, 'SVF_steps', getattr(configs, 'time_step', 7)))
        self._transformers = nn.ModuleDict()
        self._integrators = nn.ModuleDict()

    @staticmethod
    def _shape_key(shape):
        """Create a stable ModuleDict key for a spatial shape."""
        return 'x'.join(str(int(dim)) for dim in shape)

    @staticmethod
    def _normalized_to_voxel_flow(flow):
        """Convert LessNet normalized-grid displacement to MIR voxel units."""
        scale = torch.as_tensor(
            [(dim - 1) / 2.0 for dim in flow.shape[2:]],
            device=flow.device,
            dtype=flow.dtype,
        ).view(1, flow.shape[1], *([1] * (flow.dim() - 2)))
        return flow * scale

    def _get_transformer(self, shape, device):
        """Return a cached MIR spatial transformer for a shape."""
        key = self._shape_key(shape)
        if key not in self._transformers:
            self._transformers[key] = utils.SpatialTransformer(shape)
        return self._transformers[key].to(device)

    def _get_integrator(self, shape, device):
        """Return a cached MIR vector-field integrator for a shape."""
        key = self._shape_key(shape)
        if key not in self._integrators:
            self._integrators[key] = utils.VecInt(shape, self.SVF_steps)
        return self._integrators[key].to(device)

    def _integrate(self, flow):
        """Integrate a flow when SVF mode is enabled."""
        if self.SVF:
            flow = self._get_integrator(flow.shape[2:], flow.device)(flow)
        return flow

    def _warp(self, mov, flow):
        """Warp a moving image with a MIR voxel-space flow."""
        return self._get_transformer(flow.shape[2:], flow.device)(mov, flow)


class LessNet(_LessNetBase):
    """MIR-callable LessNet registration model.

    Inputs:
        input_imgs: Tuple `(mov, fix)` of tensors [B, C, D, H, W].

    Returns:
        MIR-convention voxel displacement field [B, 3, D, H, W], or
        `(warped, flow)` when `gen_output` is True.
    """

    def __init__(self, configs=None, gen_output=False, SVF=None, SVF_steps=None):
        """Initialize MIR LessNet.

        Args:
            configs: Config object with LessNet hyperparameters.
            gen_output: If True, also return the warped moving image.
            SVF: Optional override for stationary velocity field integration.
            SVF_steps: Optional override for scaling-and-squaring steps.
        """
        super().__init__(configs=configs, gen_output=gen_output, return_features=False)
        if SVF is not None:
            self.SVF = bool(SVF)
        if SVF_steps is not None:
            self.SVF_steps = int(SVF_steps)

    def forward(self, input_imgs):
        """Run LessNet registration.

        Args:
            input_imgs: Tuple `(mov, fix)` tensors.

        Returns:
            Flow tensor, and optionally warped moving image.
        """
        mov, fix = input_imgs
        flow = self._normalized_to_voxel_flow(self.unet(mov, fix))
        flow = self._integrate(flow)

        if self.gen_output:
            warped = self._warp(mov, flow)
            return warped, flow
        return flow


class LessNetSPR(_LessNetBase):
    """LessNet with spatially varying regularization weights.

    Forward outputs:
        `(flow, x_weight)` when `SVF=False`, or
        `(pos_flow, neg_flow, x_weight)` when `SVF=True`, mirroring
        `TransMorphTVFSPR`.
    """

    def __init__(self, configs=None, SVF=True, SVF_steps=None, gen_output=False):
        """Initialize LessNetSPR.

        Args:
            configs: Config object with LessNet hyperparameters.
            SVF: Whether to integrate a stationary velocity field.
            SVF_steps: Optional override for scaling-and-squaring steps.
            gen_output: If True and `SVF=False`, also return warped moving image.
        """
        super().__init__(configs=configs, gen_output=gen_output, return_features=True)
        self.SVF = bool(SVF)
        if SVF_steps is not None:
            self.SVF_steps = int(SVF_steps)

        self.sigma_0 = nn.Sequential(
            nn.Conv3d(self.start_channel, self.start_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        self.sigma_1 = nn.Sequential(
            nn.Conv3d(self.start_channel, self.start_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        self.sigma_2 = nn.Conv3d(self.start_channel, 1, kernel_size=3, padding=1)
        self.wts_act = nn.Sigmoid()
        self.eps = 1e-6

    def forward(self, input_imgs):
        """Run LessNetSPR registration."""
        mov, fix = input_imgs
        flow, features = self.unet(mov, fix)
        flow = self._normalized_to_voxel_flow(flow)

        x_weight = self.sigma_0(features)
        x_weight = self.sigma_1(x_weight)
        x_weight = self.sigma_2(x_weight)
        x_weight = self.wts_act(x_weight)
        x_weight = torch.clamp(x_weight, self.eps, 1.0)

        if self.SVF:
            pos_flow = self._get_integrator(flow.shape[2:], flow.device)(flow)
            pos_flow = torch.clamp(pos_flow, -100, 100)
            neg_flow = self._get_integrator(flow.shape[2:], flow.device)(-flow)
            neg_flow = torch.clamp(neg_flow, -100, 100)
            return pos_flow, neg_flow, x_weight

        if self.gen_output:
            warped = self._warp(mov, flow)
            return warped, flow, x_weight
        return flow, x_weight
