"""Segmentation heads and decoder blocks for volumetric models."""

import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import List, Sequence


class SCSEBlock3D(nn.Module):
    """Concurrent spatial and channel squeeze-and-excite block.

    Inputs:
        x: Tensor of shape [B, C, D, H, W].

    Returns:
        Tensor of the same shape with attention gating applied (``x * cSE(x) + x * sSE(x)``).

    Args:
        channels (int): Number of channels.
        reduction (int): Channel reduction ratio of the channel-excitation branch. Default: 16.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # channel-wise excitation
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        # spatial excitation
        self.sSE = nn.Sequential(nn.Conv3d(channels, 1, kernel_size=1, bias=False), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial and channel excitation.

        Args:
            x: Tensor of shape [B, C, D, H, W].

        Returns:
            torch.Tensor: Gated tensor of shape [B, C, D, H, W].
        """
        # channel gate
        xc = self.cSE(x)
        # spatial gate
        xs = self.sSE(x)
        # combine
        return x * xc + x * xs


class ASPP3D(nn.Module):
    """3D atrous spatial pyramid pooling without global pooling.

    Inputs:
        x: Tensor of shape [B, C, D, H, W].

    Returns:
        Tensor with ASPP features.

    Each dilation has a 3x3x3 conv-InstanceNorm-GELU branch; the concatenated branches are fused by a 1x1x1
    conv-InstanceNorm-GELU projection with Dropout3d(0.1).

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of channels of each branch and of the output.
        dilations (Sequence[int]): Dilation rates of the branches. Default: (1, 6, 12).
    """

    def __init__(self, in_ch: int, out_ch: int, dilations: Sequence[int] = (1, 6, 12)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=d, dilation=d, bias=False),
                    nn.InstanceNorm3d(out_ch, affine=True),
                    nn.GELU(),
                )
                for d in dilations
            ]
        )
        # fuse them
        self.project = nn.Sequential(
            nn.Conv3d(len(dilations) * out_ch, out_ch, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.GELU(),
            nn.Dropout3d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ASPP branches and projection.

        Args:
            x: Tensor of shape [B, in_ch, D, H, W].

        Returns:
            torch.Tensor: Tensor of shape [B, out_ch, D, H, W].
        """
        res = [b(x) for b in self.branches]
        x = torch.cat(res, dim=1)
        return self.project(x)


class AdvancedDecoder3D(nn.Module):
    """Advanced 3D decoder for multi-resolution encoder features.

    Args:
        encoder_channels: List of channel counts for each feature level, ordered like ``x_feats``.
        aspp_out: Number of channels after ASPP.
        num_classes: Number of output classes.

    Inputs:
        x_feats: List of feature maps ordered from high to low resolution (``x_feats[0]`` is the finest level and
            ``x_feats[-1]`` the deepest, which is fed to the ASPP bottleneck).

    Returns:
        Logits tensor of shape [B, num_classes, D0, H0, W0], at the resolution of ``x_feats[0]``.
    """

    def __init__(self, encoder_channels: list, aspp_out: int = 128, num_classes: int = 133):
        super().__init__()
        self.num_levels = len(encoder_channels)
        # bottleneck ASPP on the deepest feature
        self.aspp = ASPP3D(encoder_channels[-1], aspp_out, dilations=(1, 6, 12))

        # decoder blocks: one per skip level
        self.blocks = nn.ModuleList()
        in_ch = aspp_out
        # for each skip (from deepest-1 down to highest-res)
        for skip_ch in reversed(encoder_channels[:-1]):
            out_ch = skip_ch
            block = nn.Sequential(
                # fuse upsample + skip
                nn.Conv3d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.GELU(),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.GELU(),
                SCSEBlock3D(out_ch),
            )
            self.blocks.append(block)
            in_ch = out_ch

        # final classifier → num_classes
        self.classifier = nn.Conv3d(in_ch, num_classes, kernel_size=1)

    def forward(self, x_feats: List[torch.Tensor]) -> torch.Tensor:
        """Run decoder forward pass.

        Args:
            x_feats: Feature maps ordered from high to low resolution; level ``i`` has ``encoder_channels[i]``
                channels.

        Returns:
            torch.Tensor: Logits of shape [B, num_classes, D0, H0, W0] at the resolution of ``x_feats[0]``.
        """
        # start from deepest feature
        x = x_feats[-1]
        x = self.aspp(x)

        # progressively upsample and fuse with skips
        for block, skip in zip(self.blocks, reversed(x_feats[:-1])):
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return self.classifier(x)
