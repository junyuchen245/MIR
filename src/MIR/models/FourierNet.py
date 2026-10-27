"""Fourier-Net and Fourier-Net+ registration networks wrapped for MIR.

The networks predict a band-limited displacement field at low resolution; it is
brought to full resolution by zero-padding its centred Fourier spectrum.

Original code retrieved from:
https://github.com/xi-jia/Fourier-Net

Modified for MIR by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
"""

import torch
import torch.nn as nn
import MIR.models.registration_utils as utils
from typing import Tuple


def _center_crop_frequency(x, crop_shape):
    """Center crop a Fourier-shifted tensor over spatial dimensions."""
    slices = []
    for size, crop in zip(x.shape[-3:], crop_shape):
        start = (size - crop) // 2
        slices.append(slice(start, start + crop))
    return x[..., slices[0], slices[1], slices[2]]


def _center_embed_frequency(x, out_shape):
    """Center embed a Fourier-shifted tensor in a larger zero spectrum."""
    out = x.new_zeros(*x.shape[:-3], *out_shape)
    slices = []
    for size, target in zip(x.shape[-3:], out_shape):
        start = (target - size) // 2
        slices.append(slice(start, start + size))
    out[..., slices[0], slices[1], slices[2]] = x
    return out


def _low_spatial_to_full_spatial(low_spatial, out_shape):
    """Convert a low-resolution spatial field to full resolution via Fourier padding (computed in float32)."""
    with torch.autocast(device_type=low_spatial.device.type, enabled=False):
        spectrum = torch.fft.fftshift(torch.fft.fftn(low_spatial.float(), dim=(-3, -2, -1)), dim=(-3, -2, -1))
        spectrum = _center_embed_frequency(spectrum, out_shape)
        out = torch.real(torch.fft.ifftn(torch.fft.ifftshift(spectrum, dim=(-3, -2, -1)), dim=(-3, -2, -1)))
    return out.to(low_spatial.dtype)


def _full_spatial_to_low_spatial(x, low_shape):
    """Extract a low-frequency spatial image by center cropping Fourier coefficients (computed in float32)."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        spectrum = torch.fft.fftshift(torch.fft.fftn(x.float(), dim=(-3, -2, -1)), dim=(-3, -2, -1))
        spectrum = _center_crop_frequency(spectrum, low_shape)
        out = torch.real(torch.fft.ifftn(torch.fft.ifftshift(spectrum, dim=(-3, -2, -1)), dim=(-3, -2, -1)))
    return out.to(x.dtype)


class FourierNetBackbone(nn.Module):
    """3-D Fourier-Net backbone from the original implementation.

    A 4-level strided encoder followed by a 2-level decoder, so the predicted
    field has 1/4 of the input resolution along each axis.

    Args:
        in_channel: Number of concatenated moving/fixed input channels.
        n_classes: Number of output displacement channels.
        start_channel: Upstream start channel hyperparameter.
        return_features: If True, also return the last decoder feature map.
    """

    def __init__(self, in_channel=2, n_classes=3, start_channel=8, return_features=False):
        """Initialize the upstream 3-D Fourier-Net backbone."""
        super().__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        self.return_features = return_features

        bias_opt = True

        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)
        self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)
        self.ec3 = self.encoder(self.start_channel * 2, self.start_channel * 2, bias=bias_opt)
        self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt)
        self.ec5 = self.encoder(self.start_channel * 4, self.start_channel * 4, bias=bias_opt)
        self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt)
        self.ec7 = self.encoder(self.start_channel * 8, self.start_channel * 8, bias=bias_opt)
        self.ec8 = self.encoder(self.start_channel * 8, self.start_channel * 16, stride=2, bias=bias_opt)
        self.ec9 = self.encoder(self.start_channel * 16, self.start_channel * 8, bias=bias_opt)

        self.r_dc1 = self.encoder(
            self.start_channel * 8 + self.start_channel * 8,
            self.start_channel * 8,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )
        self.r_dc2 = self.encoder(
            self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt
        )
        self.r_dc3 = self.encoder(
            self.start_channel * 4 + self.start_channel * 4,
            self.start_channel * 4,
            kernel_size=3,
            stride=1,
            bias=bias_opt,
        )
        self.r_dc4 = self.encoder(
            self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt
        )
        self.rr_dc9 = self.outputs(
            self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.r_up1 = self.decoder(self.start_channel * 8, self.start_channel * 8)
        self.r_up2 = self.decoder(self.start_channel * 4, self.start_channel * 4)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, batchnorm=False):
        """Create an upstream Fourier-Net encoder block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            padding: Convolution padding.
            bias: Whether the convolution has a bias.
            batchnorm: Insert ``BatchNorm3d`` before the activation.

        Returns:
            nn.Sequential: ``Conv3d`` (+ ``BatchNorm3d``) + ``PReLU``.
        """
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.PReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias), nn.PReLU()
            )
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True):
        """Create an upstream Fourier-Net decoder block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Transposed-convolution kernel size.
            stride: Transposed-convolution stride (2 doubles the resolution).
            padding: Transposed-convolution padding.
            output_padding: Transposed-convolution output padding.
            bias: Whether the convolution has a bias.

        Returns:
            nn.Sequential: ``ConvTranspose3d`` + ``PReLU``.
        """
        layer = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=bias,
            ),
            nn.PReLU(),
        )
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, batchnorm=False):
        """Create the upstream Fourier-Net output head.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            padding: Convolution padding.
            bias: Whether the convolution has a bias.
            batchnorm: Use ``Conv3d`` + ``BatchNorm3d`` + ``Tanh`` instead of a
                plain linear ``Conv3d``.

        Returns:
            nn.Sequential: The output head.
        """
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
            )
        return layer

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Run the original 3-D Fourier-Net backbone.

        Args:
            x: Moving image ``[B, C, D, H, W]``; sizes must be divisible by 16.
            y: Fixed image ``[B, C, D, H, W]``.

        Returns:
            Low-resolution field ``[B, n_classes, D/4, H/4, W/4]``; if
            ``return_features`` is True, ``(flow, features)`` with features
            ``[B, 2*start_channel, D/4, H/4, W/4]``.
        """
        x_in = torch.cat((x, y), 1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        e4 = self.ec8(e3)
        e4 = self.ec9(e4)

        r_d0 = torch.cat((self.r_up1(e4), e3), 1)
        r_d0 = self.r_dc1(r_d0)
        r_d0 = self.r_dc2(r_d0)

        r_d1 = torch.cat((self.r_up2(r_d0), e2), 1)
        r_d1 = self.r_dc3(r_d1)
        r_d1 = self.r_dc4(r_d1)

        flow = self.rr_dc9(r_d1)
        if self.return_features:
            return flow, r_d1
        return flow


class FourierNetPlusCore(nn.Module):
    """Low-resolution core network used by 3-D Fourier-Net+.

    A 3-level strided encoder followed by a 2-level decoder, so the predicted
    field has 1/2 of the (already low-resolution) input size along each axis.

    Args:
        in_channel: Number of concatenated moving/fixed input channels.
        n_classes: Number of output displacement channels.
        start_channel: Upstream start channel hyperparameter.
        return_features: If True, also return the last decoder feature map.
    """

    def __init__(self, in_channel=2, n_classes=3, start_channel=8, return_features=False):
        """Initialize the upstream Fourier-Net+ low-resolution core."""
        super().__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        self.return_features = return_features

        bias_opt = True

        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)
        self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)
        self.ec3 = self.encoder(self.start_channel * 2, self.start_channel * 2, bias=bias_opt)
        self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt)
        self.ec5 = self.encoder(self.start_channel * 4, self.start_channel * 4, bias=bias_opt)
        self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt)
        self.ec7 = self.encoder(self.start_channel * 8, self.start_channel * 4, bias=bias_opt)

        self.dc1 = self.encoder(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.dc2 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc3 = self.encoder(self.start_channel * 2, self.start_channel * 1, kernel_size=3, stride=1, bias=bias_opt)
        self.dc9 = self.outputs(self.start_channel * 1, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)

        self.up1 = self.decoder(self.start_channel * 4, self.start_channel * 4)
        self.up2 = self.decoder(self.start_channel * 2, self.start_channel * 2)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, batchnorm=False):
        """Create an upstream Fourier-Net+ encoder block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            padding: Convolution padding.
            bias: Whether the convolution has a bias.
            batchnorm: Insert ``BatchNorm3d`` before the activation.

        Returns:
            nn.Sequential: ``Conv3d`` (+ ``BatchNorm3d``) + ``PReLU``.
        """
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.PReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias), nn.PReLU()
            )
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True):
        """Create an upstream Fourier-Net+ decoder block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Transposed-convolution kernel size.
            stride: Transposed-convolution stride (2 doubles the resolution).
            padding: Transposed-convolution padding.
            output_padding: Transposed-convolution output padding.
            bias: Whether the convolution has a bias.

        Returns:
            nn.Sequential: ``ConvTranspose3d`` + ``PReLU``.
        """
        layer = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=bias,
            ),
            nn.PReLU(),
        )
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, batchnorm=False):
        """Create the upstream Fourier-Net+ output head.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            padding: Convolution padding.
            bias: Whether the convolution has a bias.
            batchnorm: Use ``Conv3d`` + ``BatchNorm3d`` + ``Tanh`` instead of a
                plain linear ``Conv3d``.

        Returns:
            nn.Sequential: The output head.
        """
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
            )
        return layer

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Run the original Fourier-Net+ low-resolution core.

        Args:
            x: Low-resolution moving image ``[B, C, d, h, w]``; sizes must be
                divisible by 8.
            y: Low-resolution fixed image ``[B, C, d, h, w]``.

        Returns:
            Field ``[B, n_classes, d/2, h/2, w/2]``; if ``return_features`` is
            True, ``(flow, features)`` with features ``[B, start_channel, d/2, h/2, w/2]``.
        """
        x_in = torch.cat((x, y), 1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        d0 = self.up1(e3)
        d0 = self.dc1(d0)
        d0 = self.dc2(d0)

        d1 = self.up2(d0)
        d1 = self.dc3(d1)

        flow = self.dc9(d1)
        if self.return_features:
            return flow, d1
        return flow


class FourierNetPlusBackbone(nn.Module):
    """3-D Fourier-Net+ backbone with upstream low-frequency preprocessing.

    Both images are reduced to ``low_img_size`` by cropping their centred
    Fourier spectra, :class:`FourierNetPlusCore` predicts a field, and the
    field is zero-padded in the Fourier domain back to ``img_size``.

    Args:
        in_channel: Number of concatenated moving/fixed input channels.
        n_classes: Number of output displacement channels.
        start_channel: Upstream start channel hyperparameter.
        img_size: Full-resolution image size.
        low_img_size: Center-cropped low-frequency image size (defaults to
            half of ``img_size``).
        return_features: If True, also return the upsampled decoder features.
    """

    def __init__(
        self,
        in_channel=2,
        n_classes=3,
        start_channel=8,
        img_size=(160, 192, 224),
        low_img_size=None,
        return_features=False,
    ):
        """Initialize Fourier-Net+.

        Args:
            in_channel: Number of concatenated moving/fixed input channels.
            n_classes: Number of output displacement channels.
            start_channel: Upstream start channel hyperparameter.
            img_size: Full-resolution image size.
            low_img_size: Center-cropped low-frequency image size (defaults to
                half of ``img_size``).
            return_features: If True, also return the upsampled decoder features.

        """
        super().__init__()
        self.return_features = return_features
        self.img_size = tuple(img_size)
        if low_img_size is None:
            low_img_size = tuple(dim // 2 for dim in self.img_size)
        self.low_img_size = tuple(low_img_size)
        self.net1 = FourierNetPlusCore(in_channel, n_classes, start_channel, return_features=return_features)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Run Fourier-Net+ and return full-resolution normalized displacement.

        Args:
            x: Moving image ``[B, C, *img_size]``.
            y: Fixed image ``[B, C, *img_size]``.

        Returns:
            Normalized-grid displacement ``[B, n_classes, *img_size]``; if
            ``return_features`` is True, ``(flow, features)`` with the decoder
            features also brought to ``img_size``.
        """
        x_low = _full_spatial_to_low_spatial(x, self.low_img_size)
        y_low = _full_spatial_to_low_spatial(y, self.low_img_size)
        if self.return_features:
            low_flow, low_features = self.net1(x_low, y_low)
            flow = _low_spatial_to_full_spatial(low_flow, self.img_size)
            features = _low_spatial_to_full_spatial(low_features, self.img_size)
            return flow, features

        low_flow = self.net1(x_low, y_low)
        return _low_spatial_to_full_spatial(low_flow, self.img_size)


class _FourierNetBase(nn.Module):
    """Shared MIR wrapper utilities for Fourier-Net variants."""

    def __init__(self, configs=None, gen_output=False):
        super().__init__()
        if configs is None:
            from MIR.models.configs_FourierNet import get_FourierNet_default_config

            configs = get_FourierNet_default_config()

        self.configs = configs
        self.gen_output = gen_output
        self.SVF = bool(getattr(configs, 'SVF', False))
        self.SVF_steps = int(getattr(configs, 'SVF_steps', 7))
        self.img_size = tuple(getattr(configs, 'img_size', (160, 192, 224)))
        self.n_dims = int(getattr(configs, 'n_dims', 3))
        if self.n_dims != 3:
            raise ValueError('Fourier-Net MIR wrappers currently support the upstream 3-D models only.')

        self._transformers = nn.ModuleDict()
        self._integrators = nn.ModuleDict()

    @staticmethod
    def _shape_key(shape):
        """Create a stable ModuleDict key for a spatial shape."""
        return 'x'.join(str(int(dim)) for dim in shape)

    @staticmethod
    def _normalized_to_voxel_flow(flow):
        """Convert upstream normalized-grid displacement to MIR voxel units."""
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

    def _finish_flow(self, mov, normalized_flow):
        """Convert, optionally integrate, and optionally warp."""
        flow = self._normalized_to_voxel_flow(normalized_flow)
        if self.SVF:
            flow = self._get_integrator(flow.shape[2:], flow.device)(flow)

        if self.gen_output:
            warped = self._get_transformer(flow.shape[2:], flow.device)(mov, flow)
            return warped, flow
        return flow


class FourierNet(_FourierNetBase):
    """MIR-callable 3-D Fourier-Net registration model.

    Args:
        configs: Config object with Fourier-Net hyperparameters (defaults to
            ``get_FourierNet_default_config()``).
        gen_output: If True, also return the warped moving image.
        SVF: Optional override for stationary velocity field integration.
        SVF_steps: Optional override for scaling-and-squaring steps.
    """

    def __init__(self, configs=None, gen_output=False, SVF=None, SVF_steps=None):
        """Initialize FourierNet.

        Args:
            configs: Config object with Fourier-Net hyperparameters.
            gen_output: If True, also return the warped moving image.
            SVF: Optional override for stationary velocity field integration.
            SVF_steps: Optional override for scaling-and-squaring steps.

        """
        super().__init__(configs=configs, gen_output=gen_output)
        if SVF is not None:
            self.SVF = bool(SVF)
        if SVF_steps is not None:
            self.SVF_steps = int(SVF_steps)

        image_channels = int(getattr(self.configs, 'image_channels', getattr(self.configs, 'in_channels', 1)))
        out_channels = int(getattr(self.configs, 'out_channels', 3))
        start_channel = int(getattr(self.configs, 'start_channel', 8))
        self.backbone = FourierNetBackbone(
            in_channel=image_channels * 2,
            n_classes=out_channels,
            start_channel=start_channel,
        )

    def forward(self, input_imgs: Tuple[torch.Tensor, torch.Tensor]):
        """Run FourierNet registration.

        Args:
            input_imgs: Tuple `(mov, fix)` tensors of shape ``[B, C, D, H, W]``.

        Returns:
            Voxel displacement field ``[B, 3, D, H, W]`` (integrated if ``SVF``);
            if `gen_output` is True, the tuple ``(warped, flow)``.
        """
        mov, fix = input_imgs
        low_flow = self.backbone(mov, fix)
        normalized_flow = _low_spatial_to_full_spatial(low_flow, mov.shape[2:])
        return self._finish_flow(mov, normalized_flow)


class FourierNetPlus(_FourierNetBase):
    """MIR-callable 3-D Fourier-Net+ registration model.

    Inputs must have the spatial size ``configs.img_size``.

    Args:
        configs: Config object with Fourier-Net+ hyperparameters (defaults to
            ``get_FourierNetPlus_default_config()``).
        gen_output: If True, also return the warped moving image.
        SVF: Optional override for stationary velocity field integration.
        SVF_steps: Optional override for scaling-and-squaring steps.
    """

    def __init__(self, configs=None, gen_output=False, SVF=None, SVF_steps=None):
        """Initialize FourierNetPlus."""
        if configs is None:
            from MIR.models.configs_FourierNet import get_FourierNetPlus_default_config

            configs = get_FourierNetPlus_default_config()
        super().__init__(configs=configs, gen_output=gen_output)
        if SVF is not None:
            self.SVF = bool(SVF)
        if SVF_steps is not None:
            self.SVF_steps = int(SVF_steps)

        image_channels = int(getattr(self.configs, 'image_channels', getattr(self.configs, 'in_channels', 1)))
        out_channels = int(getattr(self.configs, 'out_channels', 3))
        start_channel = int(getattr(self.configs, 'start_channel', 8))
        low_img_size = getattr(self.configs, 'low_img_size', None)
        self.backbone = FourierNetPlusBackbone(
            in_channel=image_channels * 2,
            n_classes=out_channels,
            start_channel=start_channel,
            img_size=self.img_size,
            low_img_size=low_img_size,
        )

    def forward(self, input_imgs: Tuple[torch.Tensor, torch.Tensor]):
        """Run FourierNet+ registration.

        Args:
            input_imgs: Tuple `(mov, fix)` tensors of shape ``[B, C, *img_size]``.

        Returns:
            Voxel displacement field ``[B, 3, *img_size]`` (integrated if
            ``SVF``); if `gen_output` is True, the tuple ``(warped, flow)``.
        """
        mov, fix = input_imgs
        normalized_flow = self.backbone(mov, fix)
        return self._finish_flow(mov, normalized_flow)


class FourierNetSPR(_FourierNetBase):
    """MIR-callable 3-D Fourier-Net with spatially varying regularization weights.

    Args:
        configs: Config object with Fourier-Net hyperparameters (defaults to
            ``get_FourierNet_default_config()``).
        SVF: Whether to integrate a stationary velocity field.
        SVF_steps: Optional override for scaling-and-squaring steps.
        gen_output: If True and `SVF=False`, also return the warped moving image.

    Forward outputs:
        `(flow, x_weight)` when `SVF=False`, or
        `(pos_flow, neg_flow, x_weight)` when `SVF=True`, mirroring
        `TransMorphTVFSPR`. With `SVF=False` and `gen_output=True` the output is
        `(warped, flow, x_weight)`.
    """

    def __init__(self, configs=None, SVF=True, SVF_steps=None, gen_output=False):
        """Initialize FourierNetSPR."""
        super().__init__(configs=configs, gen_output=gen_output)
        self.SVF = bool(SVF)
        if SVF_steps is not None:
            self.SVF_steps = int(SVF_steps)

        image_channels = int(getattr(self.configs, 'image_channels', getattr(self.configs, 'in_channels', 1)))
        out_channels = int(getattr(self.configs, 'out_channels', 3))
        start_channel = int(getattr(self.configs, 'start_channel', 8))
        self.backbone = FourierNetBackbone(
            in_channel=image_channels * 2,
            n_classes=out_channels,
            start_channel=start_channel,
            return_features=True,
        )
        weight_channels = start_channel * 2
        self.sigma_0 = nn.Sequential(
            nn.Conv3d(weight_channels, weight_channels, kernel_size=3, padding=1),
            nn.PReLU(),
        )
        self.sigma_1 = nn.Sequential(
            nn.Conv3d(weight_channels, weight_channels, kernel_size=3, padding=1),
            nn.PReLU(),
        )
        self.sigma_2 = nn.Conv3d(weight_channels, 1, kernel_size=3, padding=1)
        self.wts_act = nn.Sigmoid()
        self.eps = 1e-6

    def _weight_from_features(self, features):
        """Predict a bounded SPR weight map from decoder features."""
        x_weight = self.sigma_0(features)
        x_weight = self.sigma_1(x_weight)
        x_weight = self.sigma_2(x_weight)
        x_weight = self.wts_act(x_weight)
        return torch.clamp(x_weight, self.eps, 1.0)

    def forward(self, input_imgs: Tuple[torch.Tensor, torch.Tensor]):
        """Run FourierNetSPR registration.

        Args:
            input_imgs: Tuple `(mov, fix)` tensors of shape ``[B, C, D, H, W]``.

        Returns:
            See the class docstring. Displacements are in voxel units (SVF
            outputs clamped to ``[-100, 100]``); ``x_weight`` has shape
            ``[B, 1, D, H, W]`` with values in ``[1e-6, 1]``.
        """
        mov, fix = input_imgs
        low_flow, low_features = self.backbone(mov, fix)
        normalized_flow = _low_spatial_to_full_spatial(low_flow, mov.shape[2:])
        features = _low_spatial_to_full_spatial(low_features, mov.shape[2:])
        flow = self._normalized_to_voxel_flow(normalized_flow)
        x_weight = self._weight_from_features(features)

        if self.SVF:
            pos_flow = self._get_integrator(flow.shape[2:], flow.device)(flow)
            pos_flow = torch.clamp(pos_flow, -100, 100)
            neg_flow = self._get_integrator(flow.shape[2:], flow.device)(-flow)
            neg_flow = torch.clamp(neg_flow, -100, 100)
            return pos_flow, neg_flow, x_weight

        if self.gen_output:
            warped = self._get_transformer(flow.shape[2:], flow.device)(mov, flow)
            return warped, flow, x_weight
        return flow, x_weight


class FourierNetPlusSPR(_FourierNetBase):
    """MIR-callable 3-D Fourier-Net+ with spatially varying regularization weights.

    Args:
        configs: Config object with Fourier-Net+ hyperparameters (defaults to
            ``get_FourierNetPlus_default_config()``).
        SVF: Whether to integrate a stationary velocity field.
        SVF_steps: Optional override for scaling-and-squaring steps.
        gen_output: If True and `SVF=False`, also return the warped moving image.

    Forward outputs:
        `(flow, x_weight)` when `SVF=False`, or
        `(pos_flow, neg_flow, x_weight)` when `SVF=True`, mirroring
        `TransMorphTVFSPR`. With `SVF=False` and `gen_output=True` the output is
        `(warped, flow, x_weight)`.
    """

    def __init__(self, configs=None, SVF=True, SVF_steps=None, gen_output=False):
        """Initialize FourierNetPlusSPR."""
        if configs is None:
            from MIR.models.configs_FourierNet import get_FourierNetPlus_default_config

            configs = get_FourierNetPlus_default_config()
        super().__init__(configs=configs, gen_output=gen_output)
        self.SVF = bool(SVF)
        if SVF_steps is not None:
            self.SVF_steps = int(SVF_steps)

        image_channels = int(getattr(self.configs, 'image_channels', getattr(self.configs, 'in_channels', 1)))
        out_channels = int(getattr(self.configs, 'out_channels', 3))
        start_channel = int(getattr(self.configs, 'start_channel', 8))
        low_img_size = getattr(self.configs, 'low_img_size', None)
        self.backbone = FourierNetPlusBackbone(
            in_channel=image_channels * 2,
            n_classes=out_channels,
            start_channel=start_channel,
            img_size=self.img_size,
            low_img_size=low_img_size,
            return_features=True,
        )
        weight_channels = start_channel
        self.sigma_0 = nn.Sequential(
            nn.Conv3d(weight_channels, weight_channels, kernel_size=3, padding=1),
            nn.PReLU(),
        )
        self.sigma_1 = nn.Sequential(
            nn.Conv3d(weight_channels, weight_channels, kernel_size=3, padding=1),
            nn.PReLU(),
        )
        self.sigma_2 = nn.Conv3d(weight_channels, 1, kernel_size=3, padding=1)
        self.wts_act = nn.Sigmoid()
        self.eps = 1e-6

    def _weight_from_features(self, features):
        """Predict a bounded SPR weight map from decoder features."""
        x_weight = self.sigma_0(features)
        x_weight = self.sigma_1(x_weight)
        x_weight = self.sigma_2(x_weight)
        x_weight = self.wts_act(x_weight)
        return torch.clamp(x_weight, self.eps, 1.0)

    def forward(self, input_imgs: Tuple[torch.Tensor, torch.Tensor]):
        """Run FourierNetPlusSPR registration.

        Args:
            input_imgs: Tuple `(mov, fix)` tensors of shape ``[B, C, *img_size]``.

        Returns:
            See the class docstring. Displacements are in voxel units (SVF
            outputs clamped to ``[-100, 100]``); ``x_weight`` has shape
            ``[B, 1, *img_size]`` with values in ``[1e-6, 1]``.
        """
        mov, fix = input_imgs
        normalized_flow, features = self.backbone(mov, fix)
        flow = self._normalized_to_voxel_flow(normalized_flow)
        x_weight = self._weight_from_features(features)

        if self.SVF:
            pos_flow = self._get_integrator(flow.shape[2:], flow.device)(flow)
            pos_flow = torch.clamp(pos_flow, -100, 100)
            neg_flow = self._get_integrator(flow.shape[2:], flow.device)(-flow)
            neg_flow = torch.clamp(neg_flow, -100, 100)
            return pos_flow, neg_flow, x_weight

        if self.gen_output:
            warped = self._get_transformer(flow.shape[2:], flow.device)(mov, flow)
            return warped, flow, x_weight
        return flow, x_weight
