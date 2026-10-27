"""Self-supervised learning heads for registration models."""

import torch
import torch.nn as nn
import MIR.models.registration_utils as utils
from typing import Tuple


class SSL_blocks(nn.Module):
    """Decoder block for SSL heads.

    Inputs:
        x: Tensor of shape [B, C, D, H, W].

    Returns:
        Upsampled tensor with `out_dim` channels.

    Conv-InstanceNorm-LeakyReLU, trilinear upsampling by ``scale_fac``, then another conv-InstanceNorm-LeakyReLU.

    Args:
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        scale_fac (int | float): Spatial upsampling factor.
    """

    def __init__(self, dim, out_dim, scale_fac):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_dim),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=scale_fac, mode="trilinear", align_corners=False),
            nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply decoder block.

        Args:
            x: Tensor of shape [B, dim, D, H, W].

        Returns:
            torch.Tensor: Tensor of shape [B, out_dim, D*s, H*s, W*s] with ``s = scale_fac``.
        """
        x = self.block(x)
        return x


class flow_blocks(nn.Module):
    """Predict mean (and optionally log-variance) flow fields.

    The mean convolution is initialized with N(0, 1e-5) weights and zero bias; the log-variance convolution with
    N(0, 1e-10) weights and bias -10.

    Args:
        dim (int): Number of input channels.
        if_KL (bool): If True, also predict a log-variance field (for a KL / variational objective). Default: True.
    """

    def __init__(self, dim, if_KL=True):
        super().__init__()
        self.if_KL = if_KL
        self.flow_mean = nn.Conv3d(dim, 3, kernel_size=3, stride=1, padding=1)
        self.flow_mean.weight.data.normal_(0.0, 1e-5)
        self.flow_mean.bias.data = nn.Parameter(torch.zeros(self.flow_mean.bias.shape))
        if self.if_KL:
            self.flow_sigma = nn.Conv3d(dim, 3, kernel_size=3, stride=1, padding=1)
            self.flow_sigma.weight.data.normal_(0.0, 1e-10)
            self.flow_sigma.bias.data = torch.Tensor([-10] * 3)

    def forward(self, x: torch.Tensor):
        """Compute flow statistics from features.

        Args:
            x: Features of shape [B, dim, D, H, W].

        Returns:
            ``(mean, log_var)`` of shape [B, 3, D, H, W] each when ``if_KL`` is True, otherwise ``mean`` only.
        """
        mean = self.flow_mean(x)
        if self.if_KL:
            sigma = self.flow_sigma(x)
            return mean, sigma
        return mean


class SSLHeadNLvl(nn.Module):
    """Self-supervised learning head with multiple levels.

    Args:
        encoder: Encoder model.
        img_size: Image size (of the final flow and of the spatial transformer).
        num_lvls: Number of levels.
        channels: Number of channels per level, ordered like the (possibly reversed) encoder outputs.
        if_upsamp: Whether to upsample predicted flows (by 2, with magnitudes scaled by 2).
        encoder_output_type: Encoder output type ('single' or 'multi'); 'multi' sums each ``(mov, fix)`` feature
            pair.
        encoder_input_type: Encoder input type ('single': concatenated images; 'multi': the ``(mov, fix)`` tuple;
            'separate': each image encoded on its own, which requires ``encoder_output_type='multi'``).
        swap_encoder_order: Whether to reverse the order of the encoder outputs.
        gen_output: Whether to generate deformed output.

    Forward inputs:
        inputs: Tuple `(mov, fix)` tensors.

    Forward outputs:
        If `gen_output`, returns `(warped, flow, stats)`; otherwise `(flow, stats)`.
    """

    def __init__(
        self,
        encoder,
        img_size=(128, 128, 128),
        num_lvls=3,
        channels=(96 * 4, 96 * 2, 96),
        if_upsamp=True,
        encoder_output_type='single',
        encoder_input_type='single',
        swap_encoder_order=True,
        gen_output=True,
    ):
        super().__init__()
        self.num_lvls = num_lvls
        self.encoder = encoder
        self.decoder_blocks = nn.ModuleList()
        self.flow_heads = nn.ModuleList()
        for i in range(num_lvls):
            self.decoder_blocks.append(SSL_blocks(channels[i], 16, 2 ** (num_lvls - i)))
            self.flow_heads.append(flow_blocks(16))
        self.flow_block_final = flow_blocks(16)
        self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.if_upsamp = if_upsamp
        self.spatial_trans = utils.SpatialTransformer(img_size)
        self.encoder_output_type = encoder_output_type
        self.encoder_input_type = encoder_input_type
        self.swap_encoder_order = swap_encoder_order
        self.gen_output = gen_output

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]):
        """Run SSL head forward pass.

        Each level's features are decoded to a common resolution and mapped to a flow mean and log-variance; the
        decoded features are summed for a final flow prediction. In training mode, flows are sampled with the
        reparameterization trick. The noise is drawn on the CPU and moved to the input's device.

        Args:
            inputs: Tuple `(mov, fix)` of images, each of shape [B, 1, D, H, W].

        Returns:
            ``(warped, flow_final, stats)`` if ``gen_output`` else ``(flow_final, stats)``. ``flow_final`` is the
            final displacement field [B, 3, *img_size], ``warped`` is ``mov`` warped by it, and ``stats`` is a list
            of ``(mean, log_var)`` pairs, one per level followed by the final prediction.
        """
        mov, fix = inputs
        x_cat = torch.cat((mov, fix), dim=1)
        if self.encoder_input_type == 'multi':
            x_out = self.encoder((mov, fix))
        elif self.encoder_input_type == 'separate':
            x_mov = self.encoder(mov)
            x_fix = self.encoder(fix)
            x_out = [[x_mov[i], x_fix[i]] for i in range(len(x_mov))]
        else:
            x_out = self.encoder(x_cat)
        if self.swap_encoder_order:
            x_out = x_out[::-1]
        if self.encoder_output_type == 'multi':
            x_out = [x_out[i][0] + x_out[i][1] for i in range(len(x_out))]
        stats = []
        flows = []
        x_sum = 0
        for i in range(self.num_lvls):
            x = self.decoder_blocks[i](x_out[i])
            x_sum = x_sum + x
            mean, std = self.flow_heads[i](x)
            stats.append((mean, std))
            noise = torch.randn(mean.shape).to(mean.device)
            if self.training:
                mean = mean + torch.exp(std / 2.0) * noise
            if self.if_upsamp:
                flow = self.up_sample(mean) * 2
            else:
                flow = mean
            flows.append(flow)
        mean_final, std_final = self.flow_block_final(x_sum)
        stats.append((mean_final, std_final))
        noise = torch.randn(mean_final.shape).to(mean_final.device)
        if self.training:
            mean_final = mean_final + torch.exp(std_final / 2.0) * noise
        if self.if_upsamp:
            flow_final = self.up_sample(mean_final) * 2
        else:
            flow_final = mean_final
        flows.append(flow_final)
        if self.gen_output:
            out = self.spatial_trans(mov, flow_final)
            return out, flow_final, stats
        return flow_final, stats


class SSLHead1Lvl(nn.Module):
    """Self-supervised learning head with one level.

    Args:
        encoder: Encoder model.
        img_size: Image size (of the flow and of the spatial transformer).
        num_lvls: Number of levels (stored but not used).
        channels: Number of channels per level; only ``channels[-1]`` (for the last encoder output after optional
            reversal) is used.
        if_upsamp: Whether to upsample predicted flows (by 2, with magnitudes scaled by 2).
        encoder_output_type: Encoder output type ('single' or 'multi'). With 'multi', the encoder receives the
            ``(mov, fix)`` tuple and each output pair is summed; otherwise it receives the concatenated images.
        swap_encoder_order: Whether to reverse the order of the encoder outputs.
        gen_output: Whether to generate deformed output.

    Forward inputs:
        inputs: Tuple `(mov, fix)` tensors.

    Forward outputs:
        If `gen_output`, returns `(warped, flow)`; otherwise `flow`.
    """

    def __init__(
        self,
        encoder,
        img_size=(128, 128, 128),
        num_lvls=3,
        channels=(96 * 4, 96 * 2, 96),
        if_upsamp=True,
        encoder_output_type='single',
        swap_encoder_order=True,
        gen_output=True,
    ):
        super().__init__()
        self.num_lvls = num_lvls
        self.encoder = encoder
        self.decoder_blocks = nn.ModuleList()
        self.flow_heads = nn.ModuleList()
        self.decoder_blocks.append(SSL_blocks(channels[-1], 16, 2))
        self.flow_heads.append(flow_blocks(16, if_KL=False))
        self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.if_upsamp = if_upsamp
        self.encoder_output_type = encoder_output_type
        self.spatial_trans = utils.SpatialTransformer(img_size)
        self.swap_encoder_order = swap_encoder_order
        self.gen_output = gen_output

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]):
        """Run SSL head forward pass.

        Args:
            inputs: Tuple `(mov, fix)` of images, each of shape [B, 1, D, H, W].

        Returns:
            ``(warped, flow)`` if ``gen_output`` else ``flow``, where ``flow`` is a deterministic displacement
            field [B, 3, *img_size] predicted from the last encoder output and ``warped`` is ``mov`` warped by it.
        """
        mov, fix = inputs
        x_cat = torch.cat((mov, fix), dim=1)
        if self.encoder_output_type == 'multi':
            x_out = self.encoder((mov, fix))
        else:
            x_out = self.encoder(x_cat)
        if self.swap_encoder_order:
            x_out = x_out[::-1]
        if self.encoder_output_type == 'multi':
            x_out = [x_out[i][0] + x_out[i][1] for i in range(len(x_out))]
        x_sum = 0
        i = -1
        x = self.decoder_blocks[i](x_out[i])
        x_sum = x_sum + x
        mean = self.flow_heads[i](x)
        if self.training:
            flow = mean
        if self.if_upsamp:
            flow = self.up_sample(mean) * 2
        else:
            flow = mean
        if self.gen_output:
            out = self.spatial_trans(mov, flow)
            return out, flow
        return flow
