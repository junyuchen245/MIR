"""Synthesis heads for image reconstruction from features."""

import torch.nn as nn
import torch
from typing import Optional


class SynthesisHead3D(nn.Module):
    """Simple synthesis head for 3D feature maps.

    Two 3x3x3 conv-norm-activation blocks followed by a 1x1x1 projection and a sigmoid.

    Args:
        in_channels: Number of input channels.
        mid_channels: Number of intermediate channels. Defaults to ``max(in_channels // 2, 16)``.
        out_channels: Number of output channels.
        norm: Normalization type ('instance' or 'batch'); any value other than 'instance' selects batch norm.
        activation: Activation function ('leaky_relu' or 'relu'); any value other than 'leaky_relu' selects ReLU.

    Inputs:
        x: Tensor of shape [B, C, D, H, W].

    Returns:
        Tensor of shape [B, out_channels, D, H, W] with values in (0, 1).
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: Optional[int] = None,
        out_channels: int = 1,
        norm: str = "instance",  # or "batch"
        activation: str = "leaky_relu",  # or "relu"
    ):
        super().__init__()
        mid = mid_channels or max(in_channels // 2, 16)
        Norm = nn.InstanceNorm3d if norm == "instance" else nn.BatchNorm3d
        Act = nn.LeakyReLU if activation == "leaky_relu" else nn.ReLU

        self.net = nn.Sequential(
            # 3×3×3 conv block to mix channels
            nn.Conv3d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            Norm(mid, affine=True),
            Act(inplace=True),
            # another 3×3×3 block
            nn.Conv3d(mid, mid, kernel_size=3, padding=1, bias=False),
            Norm(mid, affine=True),
            Act(inplace=True),
            # final projection to 1 channel
            nn.Conv3d(mid, out_channels, kernel_size=1),
            nn.Sigmoid(),  # or nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply synthesis head.

        Args:
            x: Tensor of shape [B, in_channels, D, H, W].

        Returns:
            torch.Tensor: Sigmoid output of shape [B, out_channels, D, H, W].
        """
        return self.net(x)


class ResBlock3D(nn.Module):
    """Residual block with 3D convolution.

    Two 3x3x3 convolutions with normalization and LeakyReLU(0.2); the input is added before the final activation.

    Inputs:
        x: Tensor of shape [B, C, D, H, W].

    Returns:
        Tensor of the same shape.

    Args:
        channels (int): Number of input and output channels.
        norm (str): 'instance' for InstanceNorm3d or 'group' for GroupNorm with ``channels // 8`` groups.
            Default: 'instance'.
    """

    def __init__(self, channels: int, norm: str = "instance"):
        super().__init__()
        Norm = nn.InstanceNorm3d if norm == "instance" else nn.GroupNorm
        ng = channels // 8 if norm == "group" else 1

        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = Norm(ng, channels) if norm == "group" else Norm(channels, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = Norm(ng, channels) if norm == "group" else Norm(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block.

        Args:
            x: Tensor of shape [B, C, D, H, W].

        Returns:
            torch.Tensor: Tensor of shape [B, C, D, H, W].
        """
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class SEBlock3D(nn.Module):
    """Squeeze-and-excitation block for 3D features.

    Args:
        channels (int): Number of channels.
        reduction (int): Channel reduction ratio of the bottleneck. Default: 16.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise excitation.

        Args:
            x: Tensor of shape [B, C, H, W, D].

        Returns:
            torch.Tensor: ``x`` rescaled per channel by sigmoid gates, same shape as ``x``.
        """
        # x: (B, C, H, W, D)
        b, c, *_ = x.shape
        # squeeze
        y = x.view(b, c, -1).mean(-1)  # (B, C)
        y = self.act(self.fc1(y))
        y = self.sig(self.fc2(y))  # (B, C)
        return x * y.view(b, c, 1, 1, 1)  # scale


class SynthesisHead3DAdvanced(nn.Module):
    """Synthesis head with residual and SE blocks for 3D features.

    Args:
        in_channels: Number of input channels.
        mid_channels: Number of intermediate channels. Defaults to ``max(in_channels // 2, 32)``.
        num_res_blocks: Number of residual blocks (each followed by an SE block).
        norm: Normalization type ('instance' or 'group').

    Inputs:
        x: Tensor of shape [B, C, D, H, W].

    Returns:
        Tensor of shape [B, 1, D, H, W] with values in (0, 1).
    """

    def __init__(
        self, in_channels: int, mid_channels: Optional[int] = None, num_res_blocks: int = 4, norm: str = "instance"
    ):
        super().__init__()
        mid = mid_channels or max(in_channels // 2, 32)
        self.init_norm = nn.InstanceNorm3d(in_channels, affine=True)
        # initial projection
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, mid, kernel_size=1, bias=False),
            nn.InstanceNorm3d(mid, affine=True) if norm == "instance" else nn.GroupNorm(mid // 8, mid),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # a stack of Residual + SE blocks
        blocks = []
        for _ in range(num_res_blocks):
            blocks += [ResBlock3D(mid, norm=norm), SEBlock3D(mid)]
        self.blocks = nn.Sequential(*blocks)

        # final output
        self.out_conv = nn.Conv3d(mid, 1, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply advanced synthesis head.

        Args:
            x: Tensor of shape [B, in_channels, D, H, W].

        Returns:
            torch.Tensor: Sigmoid output of shape [B, 1, D, H, W].
        """
        x = self.init_norm(x)
        x = self.proj(x)
        x = self.blocks(x)
        x = self.out_conv(x)
        return self.out_act(x)


class ECA3D(nn.Module):
    """Efficient channel attention for 3D feature maps.

    Global average pooling followed by a 1D convolution across channels and a sigmoid gate.

    Args:
        channels (int): Number of channels.
        k_size (int): Kernel size of the 1D channel convolution. Default: 3.
    """

    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        # 1D conv on channel‐wise pooled features
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention.

        Args:
            x: Tensor of shape [B, C, H, W, D].

        Returns:
            torch.Tensor: ``x`` reweighted per channel, same shape as ``x``.
        """
        # x: (B, C, H, W, D)
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)  # (B,1,C)
        y = self.conv(y).view(b, c)  # (B,C)
        y = self.sigmoid(y).view(b, c, 1, 1, 1)  # (B,C,1,1,1)
        return x * y  # reweight channels


class InvertedResidual3D(nn.Module):
    """MobileNet-style inverted residual block for 3D features.

    Pointwise expansion, 3x3x3 depthwise convolution and pointwise projection, each followed by normalization
    (ReLU after the first two). A residual connection is used when ``stride == 1`` and ``in_ch == out_ch``.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        stride (int): Stride of the depthwise convolution. Default: 1.
        expand_ratio (int): Channel expansion ratio of the hidden layer. Default: 4.
        norm_layer (type): Normalization layer class (called with ``affine=True``). Default: nn.InstanceNorm3d.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand_ratio: int = 4, norm_layer=nn.InstanceNorm3d):
        super().__init__()
        hidden_ch = in_ch * expand_ratio
        self.use_res_connect = stride == 1 and in_ch == out_ch

        layers = []
        # pointwise expand
        layers += [
            nn.Conv3d(in_ch, hidden_ch, 1, bias=False),
            norm_layer(hidden_ch, affine=True),
            nn.ReLU(inplace=True),
        ]
        # depthwise spatial conv
        layers += [
            nn.Conv3d(hidden_ch, hidden_ch, 3, stride, padding=1, groups=hidden_ch, bias=False),
            norm_layer(hidden_ch, affine=True),
            nn.ReLU(inplace=True),
        ]
        # pointwise project
        layers += [nn.Conv3d(hidden_ch, out_ch, 1, bias=False), norm_layer(out_ch, affine=True)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverted residual block.

        Args:
            x: Tensor of shape [B, in_ch, H, W, D].

        Returns:
            torch.Tensor: Tensor of shape [B, out_ch, H', W', D'] (spatial size reduced by ``stride``).
        """
        out = self.conv(x)
        if self.use_res_connect:
            return x + out
        return out


class EfficientAdvancedSynthHead3D(nn.Module):
    """Memory-efficient advanced synthesis head using inverted residuals + ECA attention.

    Args:
        in_channels (int): Number of input channels.
        mid_channels (int, optional): Number of intermediate channels. Defaults to ``max(in_channels // 2, 32)``.
        num_blocks (int): Number of inverted residual blocks. Default: 3.
        expansion (int): Expansion ratio of the inverted residual blocks. Default: 4.
    """

    def __init__(self, in_channels: int, mid_channels: Optional[int] = None, num_blocks: int = 3, expansion: int = 4):
        super().__init__()
        mid = mid_channels or max(in_channels // 2, 32)
        # initial projection
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, mid, kernel_size=1, bias=False),
            nn.InstanceNorm3d(mid, affine=True),
            nn.ReLU(inplace=True),
        )
        # inverted residual stack
        blocks = []
        for _i in range(num_blocks):
            blocks.append(
                InvertedResidual3D(
                    in_ch=mid, out_ch=mid, stride=1, expand_ratio=expansion, norm_layer=nn.InstanceNorm3d
                )
            )
        self.blocks = nn.Sequential(*blocks)
        # efficient channel attention
        self.eca = ECA3D(mid, k_size=3)
        # final 1×1×1 conv to single channel
        self.head = nn.Sequential(nn.Conv3d(mid, 1, kernel_size=1, bias=True), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Synthesize a single-channel image from features.

        Args:
            x: Tensor of shape (B, C, H, W, D).

        Returns:
            torch.Tensor: Sigmoid output of shape (B, 1, H, W, D).
        """
        x = self.proj(x)  # B×mid×H×W×D
        x = self.blocks(x)  # deeper features, still B×mid×H×W×D
        x = self.eca(x)  # channel reweighting
        return self.head(x)  # B×1×H×W×D


class ConvNeXtBlock3D(nn.Module):
    """ConvNeXt-style 3D block: depthwise 7x7x7 convolution, LayerNorm and a channel MLP with a residual.

    Args:
        dim (int): Number of channels.
        mlp_ratio (int): Hidden-width ratio of the channel MLP. Default: 4.
        drop_path (float): Drop rate applied to the residual branch; implemented with ``nn.Dropout`` (element-wise
            dropout, not per-sample stochastic depth). Default: 0.0.
    """

    def __init__(self, dim: int, mlp_ratio: int = 4, drop_path: float = 0.0):
        super().__init__()
        hidden_dim = dim * mlp_ratio
        # depthwise 7×7×7 conv (local spatial mixing)
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        # LayerNorm over channels: we need to permute to (B,HWD,C)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # two-layer MLP on channels with GELU
        self.pwconv1 = nn.Linear(dim, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim, bias=True)
        # optional stochastic depth (can omit for low memory)
        self.drop_path = nn.Identity() if drop_path == 0.0 else nn.Dropout(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the ConvNeXt block.

        Args:
            x: Tensor of shape (B, C, H, W, D).

        Returns:
            torch.Tensor: Tensor of shape (B, C, H, W, D).
        """
        shortcut = x
        # local mixing
        x = self.dwconv(x)  # (B, C, H, W, D)
        # channel MLP: LN expects last dim = features
        B, C, H, W, D = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # → (B, H, W, D, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3)  # → (B, C, H, W, D)
        # residual
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXtSynthHead3D(nn.Module):
    """Synthesis head built from ConvNeXt-style 3D blocks.

    Args:
        in_channels (int): Number of input channels.
        mid_channels (int, optional): Number of intermediate channels. Defaults to ``max(in_channels // 2, 32)``.
        num_blocks (int): Number of ConvNeXt blocks. Default: 3.
        mlp_ratio (int): Hidden-width ratio of the block MLPs. Default: 4.
    """

    def __init__(self, in_channels: int, mid_channels: Optional[int] = None, num_blocks: int = 3, mlp_ratio: int = 4):
        super().__init__()
        mid = mid_channels or max(in_channels // 2, 32)
        # initial 1×1×1 conv to project features
        self.proj = nn.Conv3d(in_channels, mid, kernel_size=1, bias=True)
        # stack of ConvNeXt blocks
        self.blocks = nn.Sequential(*[ConvNeXtBlock3D(mid, mlp_ratio=mlp_ratio) for _ in range(num_blocks)])
        # final 1×1×1 conv to map to single channel
        self.head = nn.Sequential(nn.Conv3d(mid, 1, kernel_size=1, bias=True), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Synthesize a single-channel image from features.

        Args:
            x: Tensor of shape (B, C, H, W, D).

        Returns:
            torch.Tensor: Sigmoid output of shape (B, 1, H, W, D).
        """
        x = self.proj(x)  # project to mid-channels
        x = self.blocks(x)  # residual depthwise+MLP mixing
        return self.head(x)  # single-channel output
