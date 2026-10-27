"""Hyper-TransMorph: TransMorph conditioned on a regularization hyperparameter.

Convolution and linear weights are predicted from an embedding of the
hyperparameter value (hypernetwork conditioning), so a single trained network
covers a range of regularization strengths.

Chen, J., Du, Y., He, Y., Segars, W. P., Li, Y., & Frey, E. C. (2021).
TransMorph: Transformer for unsupervised medical image registration.
arXiv preprint arXiv:2111.10480.
Swin-Transformer code retrieved from:
https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation
Original paper:
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021).
Swin transformer: Hierarchical vision transformer using shifted windows.
arXiv preprint arXiv:2103.14030.
Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import numpy as np
import MIR.models.configs_TransMorph as configs
import MIR.models.registration_utils as utils
from typing import List, Optional, Tuple


class CustomConv(nn.Module):
    """Hyper-conditioned 3D convolution with weights predicted from parameters.

    The kernel (``[out_channels, in_channels, k, k, k]``) and bias are produced by
    bias-free linear maps of the hyperparameter embedding, so one embedding
    vector (shared across the batch) is expected per forward pass.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        ndims: Number of spatial dimensions used to size the weight generator;
            the forward pass always applies ``conv3d``.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        bias: Whether to add a predicted bias.
        normalize: Initialize the weight generators from ``N(0, 1e-5)``.
        nb_hyp_units: Size of the hyperparameter embedding.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        ndims=3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        normalize=False,
        nb_hyp_units=96,
    ):
        super().__init__()
        self.linear_conv = nn.Linear(nb_hyp_units, kernel_size**ndims * out_channels * in_channels, bias=False)
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

    def forward(self, x: torch.Tensor, hyp_feat: torch.Tensor) -> torch.Tensor:
        """Apply hyper-conditioned convolution.

        Args:
            x: Feature tensor of shape ``[B, in_channels, H, W, D]``.
            hyp_feat: Hyperparameter embedding of shape ``[1, nb_hyp_units]``.

        Returns:
            Tensor of shape ``[B, out_channels, H', W', D']`` after applying the
            predicted weights and bias.

        """
        kernel = self.linear_conv(hyp_feat).reshape([self.out_channels, self.in_channels, self.ks, self.ks, self.ks])
        out = nnf.conv3d(x, kernel, stride=self.stride, padding=self.padding)
        if self.if_bias:
            bias = self.linear_bias(hyp_feat).unsqueeze(2).unsqueeze(2).unsqueeze(2)
            out = out + bias
        return out


class HyperBlocks(nn.Module):
    """MLP that embeds hyperparameters for hypernetwork conditioning.

    Args:
        nb_hyp_params: Number of input hyperparameters.
        nb_hyp_layers: Number of ``Linear`` + ReLU layers.
        nb_hyp_units: Width of every layer (embedding size).
    """

    def __init__(self, nb_hyp_params=1, nb_hyp_layers=6, nb_hyp_units=96):
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


class HyperLinear(nn.Module):
    """Linear layer with weights predicted from hyperparameters.

    Args:
        in_features: Input feature size.
        out_features: Output feature size.
        nb_hyp_units: Size of the hyperparameter embedding.
        bias: Whether to add a predicted bias.
    """

    def __init__(self, in_features, out_features, nb_hyp_units=96, bias=True):
        super().__init__()
        self.linear_wts = nn.Linear(nb_hyp_units, in_features * out_features, bias=False)
        self.linear_bias = nn.Linear(nb_hyp_units, out_features, bias=False)
        self.in_features = in_features
        self.out_features = out_features
        self.if_bias = bias

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Apply hyper-conditioned linear projection.

        Args:
            x: Input of shape ``[..., in_features]``.
            h: Hyperparameter embedding with ``nb_hyp_units`` elements
                (e.g. ``[1, nb_hyp_units]``).

        Returns:
            torch.Tensor: Output of shape ``[..., out_features]``.
        """
        weight = self.linear_wts(h).reshape([self.out_features, self.in_features])
        output = torch.matmul(x, weight.t())
        if self.if_bias:
            bias = self.linear_bias(h).reshape([self.out_features])
            output = output + bias
        return output


class Mlp(nn.Module):
    """Feed-forward MLP block used inside transformer blocks.

    Both projections are :class:`HyperLinear` layers.

    Args:
        in_features: Input feature size.
        hidden_features: Hidden size (defaults to ``in_features``).
        out_features: Output size (defaults to ``in_features``).
        act_layer: Activation class.
        drop: Dropout rate.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = HyperLinear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = HyperLinear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, hyper: torch.Tensor) -> torch.Tensor:
        """Apply MLP to input features conditioned on hyperparameters.

        Args:
            x: Input of shape ``[..., in_features]``.
            hyper: Hyperparameter embedding.

        Returns:
            torch.Tensor: Output of shape ``[..., out_features]``.
        """
        x = self.fc1(x, hyper)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x, hyper)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size) -> torch.Tensor:
    """Partition a feature map into non-overlapping 3D windows.

    Args:
        x: (B, H, W, L, C)
        window_size (tuple[int]): window size per spatial axis
    Returns:
        windows: (num_windows*B, window_size[0], window_size[1], window_size[2], C)

    """
    B, H, W, L, C = x.shape
    x = x.view(
        B,
        H // window_size[0],
        window_size[0],
        W // window_size[1],
        window_size[1],
        L // window_size[2],
        window_size[2],
        C,
    )

    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse(windows: torch.Tensor, window_size, H: int, W: int, L: int) -> torch.Tensor:
    """Merge 3D windows back into a feature map (inverse of :func:`window_partition`).

    Args:
        windows: (num_windows*B, window_size[0], window_size[1], window_size[2], C)
        window_size (tuple[int]): Window size per spatial axis
        H (int): Height of image
        W (int): Width of image
        L (int): Length of image
    Returns:
        x: (B, H, W, L, C)

    """
    B = int(windows.shape[0] / (H * W * L / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(
        B,
        H // window_size[0],
        W // window_size[1],
        L // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, L, -1)  #
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.

    It supports both of shifted and non-shifted window. The qkv and output
    projections are :class:`HyperLinear` layers.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The 3D window size (Wh, Ww, Wt).
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        rpe (bool, optional): If True, add relative position bias. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0

    """

    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, rpe=True, attn_drop=0.0, proj_drop=0.0
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1 * 2*Wt-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))  # 3, Wh, Ww, Wt
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wt
        self.rpe = rpe
        if self.rpe:
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wt, Wh*Ww*Wt
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wt, Wh*Ww*Wt, 3
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wt, Wh*Ww*Wt
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = HyperLinear(dim, dim * 3, bias=qkv_bias)  # nn.Linear
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = HyperLinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, hyper=None):
        """Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) with N = Wh*Ww*Wt, or None
            hyper: Hyperparameter embedding for the hyper-conditioned projections.

        Returns:
            torch.Tensor: Output features with shape (num_windows*B, N, C).

        """
        B_, N, C = x.shape  # (num_windows*B, Wh*Ww*Wt, C)
        qkv = self.qkv(x, hyper).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                -1,
            )  # Wh*Ww*Wt,Wh*Ww*Wt,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wt, Wh*Ww*Wt
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x, hyper)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        rpe (bool, optional): If True, use relative position bias. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(7, 7, 7),
        shift_size=(0, 0, 0),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        rpe=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= min(self.shift_size) < min(self.window_size), (
            f"shift_size must in 0-window_size, shift_sz: {self.shift_size}, win_size: {self.window_size}"
        )

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            rpe=rpe,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None
        self.T = None

    def forward(self, x: torch.Tensor, mask_matrix: torch.Tensor, hyper: torch.Tensor) -> torch.Tensor:
        """Apply (shifted-)window attention and the MLP.

        The spatial size is read from ``self.H``, ``self.W`` and ``self.T``, which
        :class:`BasicLayer` sets before calling the block.

        Args:
            x: Input tokens of shape ``(B, H*W*T, C)``.
            mask_matrix: Attention mask for shifted windows (used only when the
                shift size is non-zero).
            hyper: Hyperparameter embedding.

        Returns:
            torch.Tensor: Output tokens of shape ``(B, H*W*T, C)``.
        """
        H, W, T = self.H, self.W, self.T
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, T, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] - T % self.window_size[2]) % self.window_size[2]
        x = nnf.pad(x, (0, 0, pad_f, pad_h, pad_t, pad_b, pad_l, pad_r))
        _, Hp, Wp, Tp, _ = x.shape

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3)
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size[0] * self.window_size[1] * self.window_size[2], C
        )  # nW*B, window_size*window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask, hyper=hyper)  # nW*B, window_size*window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, Tp)  # B H' W' L' C

        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3)
            )
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_h > 0:
            x = x[:, :H, :W, :T, :].contiguous()

        x = x.view(B, H * W * T, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), hyper))

        return x


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Uses a plain (not hyper-conditioned) ``nn.Linear`` reduction.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        reduce_factor (int, optional): Output channels are ``(8 // reduce_factor) * dim``. Default: 2

    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, (8 // reduce_factor) * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int, T: int, hyper: torch.Tensor) -> torch.Tensor:
        """Merge 2x2x2 neighbouring patches.

        Args:
            x: Input tokens of shape ``(B, H*W*T, C)``.
            H: Height of the token grid (must be even).
            W: Width of the token grid (must be even).
            T: Depth of the token grid (must be even).
            hyper: Hyperparameter embedding (unused).

        Returns:
            torch.Tensor: Tokens of shape ``(B, H/2*W/2*T/2, (8 // reduce_factor) * C)``.
        """
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, T, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x5 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x6 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 T/2 8*C
        x = x.view(B, -1, 8 * C)  # B H/2*W/2*T/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        rpe (bool, optional): If True, use relative position bias. Default: True
        pat_merg_rf (int, optional): Reduce factor passed to the downsample layer. Default: 2

    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=(7, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        rpe=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        pat_merg_rf=2,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf
        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0)
                    if (i % 2 == 0)
                    else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    rpe=rpe,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, reduce_factor=self.pat_merg_rf)
        else:
            self.downsample = None

    def forward(self, x, H, W, T, hyper):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, H*W*T, C).
            H: Height of the input feature.
            W: Width of the input feature.
            T: Depth of the input feature.
            hyper: Hyperparameter embedding.

        Returns:
            tuple: ``(x, H, W, T, x_down, Wh, Ww, Wt)`` where ``x`` is the stage
            output and ``x_down`` the downsampled tokens of size
            ``(Wh, Ww, Wt) = ((H+1)//2, (W+1)//2, (T+1)//2)``; without a downsample
            layer, ``x_down`` is ``x`` and the sizes are unchanged.

        """
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        Tp = int(np.ceil(T / self.window_size[2])) * self.window_size[2]
        img_mask = torch.zeros((1, Hp, Wp, Tp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (
            slice(0, -self.window_size[0]),
            slice(-self.window_size[0], -self.shift_size[0]),
            slice(-self.shift_size[0], None),
        )
        w_slices = (
            slice(0, -self.window_size[1]),
            slice(-self.window_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None),
        )
        t_slices = (
            slice(0, -self.window_size[2]),
            slice(-self.window_size[2], -self.shift_size[2]),
            slice(-self.shift_size[2], None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    img_mask[:, h, w, t, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(attn_mask == 0, 0.0)

        for blk in self.blocks:
            blk.H, blk.W, blk.T = H, W, T
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask, hyper)
            else:
                x = blk(x, attn_mask, hyper)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T, hyper)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x, H, W, T, x_down, Wh, Ww, Wt
        return x, H, W, T, x, H, W, T


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    The projection is a hyper-conditioned :class:`CustomConv`.

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = CustomConv(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor, hyper: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x: Input of shape ``(B, in_chans, H, W, T)``; zero-padded to a
                multiple of the patch size.
            hyper: Hyperparameter embedding.

        Returns:
            torch.Tensor: Patch embeddings of shape ``(B, embed_dim, Wh, Ww, Wt)``.
        """
        # padding
        _, _, H, W, T = x.size()
        if T % self.patch_size[2] != 0:
            x = nnf.pad(x, (0, self.patch_size[2] - T % self.patch_size[2]))
        if W % self.patch_size[1] != 0:
            x = nnf.pad(x, (0, 0, 0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x, hyper)  # B C Wh Ww Wt
        if self.norm is not None:
            Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x


class SinusoidalPositionEmbedding(nn.Module):
    """1D sinusoidal position embedding over the token axis.

    Returns a fixed sinusoidal embedding (interleaved sin/cos); it is not a
    rotary embedding and is not applied to q/k.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the embedding for a token sequence.

        Args:
            x: Tokens of shape ``(B, N, C)``; only the shape is used (``C`` must
                be even).

        Returns:
            torch.Tensor: Embedding of shape ``(1, N, C)``.
        """
        batch_sz, n_patches, hidden = x.shape
        position_ids = torch.arange(0, n_patches).float().to(x.device)
        indices = torch.arange(0, hidden // 2).float().to(x.device)
        indices = torch.pow(10000.0, -2 * indices / hidden)
        embeddings = torch.einsum('b,d->bd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (1, n_patches, hidden))
        return embeddings


class SinPositionalEncoding3D(nn.Module):
    """3D sinusoidal positional encoding for channels-first feature maps.

    Args:
        channels: Number of channels of the tensor the encoding is added to.
    """

    def __init__(self, channels):
        """:param channels: The last dimension of the tensor you want to apply pos emb to."""
        super().__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        # self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute the positional encoding.

        Args:
            tensor: A 5d tensor of size (batch_size, ch, x, y, z); only its shape,
                device and dtype are used.

        Returns:
            torch.Tensor: Positional Encoding Matrix of size (batch_size, ch, x, y, z).

        Raises:
            RuntimeError: If the input is not 5d.
        """
        tensor = tensor.permute(0, 2, 3, 4, 1)
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(tensor.type())
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z
        emb = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return emb.permute(0, 4, 1, 2, 3)


class SwinTransformer(nn.Module):
    r"""Hyper-conditioned 3D Swin Transformer encoder.

    A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
    https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int | tuple(int)): Input image size used to size the absolute
            position embedding. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: (7, 7, 7)
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.2
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        spe (bool): If True (and ``ape`` is False), add a sinusoidal 3D positional
            encoding (created on CUDA). Default: False
        rpe (bool): If True, use relative position bias in window attention. Default: True
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        out_indices (Sequence[int]): Output from which stages. Default: (0, 1, 2, 3)
        frozen_stages (int): Stages to be frozen (-1 means not freezing any parameters). Default: -1
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pat_merg_rf (int): Reduce factor of the patch merging layers. Default: 2
    """

    def __init__(
        self,
        pretrain_img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(7, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        ape=False,
        spe=False,
        rpe=True,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        use_checkpoint=False,
        pat_merg_rf=4,
    ):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [
                pretrain_img_size[0] // patch_size[0],
                pretrain_img_size[1] // patch_size[1],
                pretrain_img_size[2] // patch_size[2],
            ]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2])
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        elif self.spe:
            self.pos_embd = SinPositionalEncoding3D(embed_dim).cuda()
            # self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                rpe=rpe,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                pat_merg_rf=pat_merg_rf,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.

        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x: torch.Tensor, hyper: torch.Tensor) -> List[torch.Tensor]:
        """Forward function.

        Args:
            x: Input of shape ``(B, in_chans, H, W, T)``.
            hyper: Hyperparameter embedding.

        Returns:
            list[torch.Tensor]: Feature maps of the stages in ``out_indices``, each of
            shape ``(B, C_i, H_i, W_i, T_i)``.
        """
        x = self.patch_embed(x, hyper)

        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt, hyper)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                outs.append(out)
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed.

        Args:
            mode: Whether to set training mode.
        """
        super().train(mode)
        self._freeze_stages()
        return self


class Conv3dReLU(nn.Module):
    """Hyper-conditioned convolution, LeakyReLU, then normalization.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        padding: Convolution padding.
        stride: Convolution stride.
        use_batchnorm: Use ``BatchNorm3d`` if True, otherwise ``InstanceNorm3d``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        super().__init__()
        self.conv = CustomConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            self.nm = nn.InstanceNorm3d(out_channels)
        else:
            self.nm = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor, hyper: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: Input of shape ``(B, in_channels, H, W, D)``.
            hyper: Hyperparameter embedding.

        Returns:
            torch.Tensor: Output of shape ``(B, out_channels, H', W', D')``.
        """
        x = self.conv(x, hyper)
        x = self.relu(x)
        x = self.nm(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block: 2x trilinear upsampling, optional skip concat, two convs.

    Args:
        in_channels: Channels of the input to be upsampled.
        out_channels: Output channels.
        skip_channels: Channels of the skip connection (0 for none).
        use_batchnorm: Normalization choice passed to :class:`Conv3dReLU`.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels=0,
        use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None, hyper: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Upsample, concatenate the skip connection and convolve.

        Args:
            x: Input of shape ``(B, in_channels, H, W, D)``.
            skip: Optional skip tensor of shape ``(B, skip_channels, 2H, 2W, 2D)``.
            hyper: Hyperparameter embedding.

        Returns:
            torch.Tensor: Output of shape ``(B, out_channels, 2H, 2W, 2D)``.
        """
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x, hyper)
        x = self.conv2(x, hyper)
        return x


class RegistrationHead(nn.Module):
    """Hyper-conditioned convolution producing a displacement field.

    The weight generators are initialized with small values (``normalize=True``).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (3 for a 3D field).
        kernel_size: Convolution kernel size.
        upsampling: Unused.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        super().__init__()
        self.conv3d = CustomConv(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, normalize=True
        )

    def forward(self, x: torch.Tensor, hyper: torch.Tensor) -> torch.Tensor:
        """Predict the field.

        Args:
            x: Features of shape ``(B, in_channels, H, W, D)``.
            hyper: Hyperparameter embedding.

        Returns:
            torch.Tensor: Field of shape ``(B, out_channels, H, W, D)``.
        """
        x = self.conv3d(x, hyper)
        return x


class HyperTransMorphTVF(nn.Module):
    """Hyper-conditioned TransMorph with time-varying (progressive) flow estimation.

    A hyper-conditioned Swin encoder/decoder produces features; ``time_steps``
    hyper-conditioned heads then predict incremental flows that are composed
    (or added) while the moving image is re-warped at each step.

    Args:
        config: TransMorph configuration (e.g. from :data:`CONFIGS`).
        time_steps: Number of progressive flow-estimation steps.
        SVF: If True, treat the composed flow as a stationary velocity field and
            integrate it in both directions.
        SVF_steps: Number of scaling-and-squaring steps.
        composition: ``'composition'`` to compose incremental flows, any other
            value for the additive update ``flow_prev + warp(flow, flow)``.
    """

    def __init__(self, config, time_steps=7, SVF=True, SVF_steps=7, composition='composition'):
        """Multi-resolution TransMorph."""
        super().__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.time_steps = time_steps
        self.img_size = config.img_size
        self.composition = composition
        self.transformer = SwinTransformer(
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            depths=config.depths,
            num_heads=config.num_heads,
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            ape=config.ape,
            spe=config.spe,
            rpe=config.rpe,
            patch_norm=config.patch_norm,
            use_checkpoint=config.use_checkpoint,
            out_indices=config.out_indices,
            pat_merg_rf=config.pat_merg_rf,
        )
        self.up0 = DecoderBlock(
            embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if if_transskip else 0, use_batchnorm=False
        )
        self.up1 = DecoderBlock(
            embed_dim * 2, embed_dim, skip_channels=embed_dim if if_transskip else 0, use_batchnorm=False
        )  # 384, 20, 20, 64
        self.up2 = DecoderBlock(
            embed_dim, embed_dim // 2, skip_channels=embed_dim // 2 if if_transskip else 0, use_batchnorm=False
        )  # 384, 20, 20, 64
        self.c1 = Conv3dReLU(2, embed_dim // 2, 3, 1)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.reg_heads = nn.ModuleList()
        self.up3s = nn.ModuleList()
        self.cs = nn.ModuleList()
        for _t in range(self.time_steps):
            self.cs.append(Conv3dReLU(2, embed_dim // 2, 3, 1))
            self.reg_heads.append(
                RegistrationHead(
                    in_channels=config.reg_head_chan,
                    out_channels=3,
                    kernel_size=3,
                )
            )
            self.up3s.append(
                DecoderBlock(
                    embed_dim // 2,
                    config.reg_head_chan,
                    skip_channels=embed_dim // 2 if if_convskip else 0,
                    use_batchnorm=False,
                )
            )
        self.spatial_trans_ = utils.SpatialTransformer(config.img_size)
        self.spatial_trans = utils.SpatialTransformer(
            (config.img_size[0] * 2, config.img_size[1] * 2, config.img_size[2] * 2)
        )
        self.upsamp = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.SVF = SVF
        if self.SVF:
            self.vec_int = utils.VecInt(config.img_size, SVF_steps)

        self.hyper_model = HyperBlocks()

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor], hyp_val: torch.Tensor):
        """Register a moving image to a fixed image.

        Args:
            inputs: Tuple ``(mov, fix)`` of tensors with shape ``(B, 1, *config.img_size)``.
            hyp_val: Hyperparameter value(s) of shape ``(1, 1)``, embedded by
                :class:`HyperBlocks` and shared across the batch.

        Returns:
            ``(pos_flow, neg_flow)`` if ``SVF`` is True (forward and inverse
            displacement fields, each clamped to ``[-100, 100]``), otherwise the
            composed displacement field ``flow``. All fields have shape
            ``(B, 3, *config.img_size)``.
        """
        hyper = self.hyper_model(hyp_val)
        mov, fix = inputs
        x_cat = torch.cat((mov, fix), dim=1)
        x_s1 = self.avg_pool(x_cat)
        out_feats = self.transformer(x_cat, hyper)  # (B, n_patch, hidden)
        if self.if_convskip:
            f3 = self.c1(x_s1, hyper)
        else:
            f3 = None
        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
        else:
            f1 = None
            f2 = None

        x = self.up0(out_feats[-1], f1, hyper=hyper)
        x = self.up1(x, f2, hyper=hyper)
        xx = self.up2(x, f3, hyper=hyper)
        def_x = mov.clone()
        flow_previous = torch.zeros((mov.shape[0], 3, self.img_size[0], self.img_size[1], self.img_size[2])).to(
            mov.device
        )
        flows = []

        # flow integration
        for t in range(self.time_steps):
            f_out = self.cs[t](torch.cat((def_x, fix), dim=1), hyper)
            x = self.up3s[t](xx, f_out, hyper=hyper)
            flow = self.reg_heads[t](x, hyper)
            flows.append(flow)
            if self.composition == 'composition':
                flow_new = flow_previous + self.spatial_trans_(flow, flow_previous)
            else:
                # essentially addition
                flow_new = flow_previous + self.spatial_trans_(flow, flow)
            def_x = self.spatial_trans_(mov, flow_new)
            flow_previous = flow_new
        flow = flow_new

        if self.SVF:
            pos_flow = self.vec_int(flow)
            pos_flow = torch.clamp(pos_flow, -100, 100)
            neg_flow = -flow
            neg_flow = self.vec_int(neg_flow)
            neg_flow = torch.clamp(neg_flow, -100, 100)
            return pos_flow, neg_flow
        return flow


class HyperTransMorphTVFSPR(nn.Module):
    """Hyper-conditioned TransMorph TVF with spatially-varying regularization.

    Args:
        config: Configuration object containing model parameters
        SVF: Boolean indicating whether to use SVF (Time Stationary Velocity Field) integration
        time_steps: Number of time steps for progressive registration
        SVF_steps: Number of steps for SVF integration
        composition: Type of composition for flow integration ('composition' or 'addition')
    """

    def __init__(self, config, SVF=True, time_steps=12, SVF_steps=7, composition='composition'):
        super().__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.time_steps = time_steps
        self.img_size = config.img_size
        self.composition = composition
        self.transformer = SwinTransformer(
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            depths=config.depths,
            num_heads=config.num_heads,
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            ape=config.ape,
            spe=config.spe,
            rpe=config.rpe,
            patch_norm=config.patch_norm,
            use_checkpoint=config.use_checkpoint,
            out_indices=config.out_indices,
            pat_merg_rf=config.pat_merg_rf,
        )
        self.up0 = DecoderBlock(
            embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if if_transskip else 0, use_batchnorm=False
        )
        self.up1 = DecoderBlock(
            embed_dim * 2, embed_dim, skip_channels=embed_dim if if_transskip else 0, use_batchnorm=False
        )  # 384, 20, 20, 64
        self.up2 = DecoderBlock(
            embed_dim, embed_dim // 2, skip_channels=embed_dim // 2 if if_transskip else 0, use_batchnorm=False
        )  # 384, 20, 20, 64
        self.c1 = Conv3dReLU(2, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.reg_heads = nn.ModuleList()
        self.up3s = nn.ModuleList()
        self.cs = nn.ModuleList()
        for _t in range(self.time_steps):
            self.cs.append(Conv3dReLU(2, embed_dim // 2, 3, 1, use_batchnorm=False))
            self.reg_heads.append(
                RegistrationHead(
                    in_channels=config.reg_head_chan,
                    out_channels=3,
                    kernel_size=3,
                )
            )
            self.up3s.append(
                DecoderBlock(
                    embed_dim // 2,
                    config.reg_head_chan,
                    skip_channels=embed_dim // 2 if if_convskip else 0,
                    use_batchnorm=False,
                )
            )
        self.spatial_trans_half = utils.SpatialTransformer(config.img_size)
        self.spatial_trans = utils.SpatialTransformer(
            (config.img_size[0] * 2, config.img_size[1] * 2, config.img_size[2] * 2)
        )
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.SVF = SVF
        if self.SVF:
            self.integrate = utils.VecInt(config.img_size, SVF_steps)
        self.sigma_0 = Conv3dReLU(config.reg_head_chan, embed_dim // 2, 3, 3 // 2, use_batchnorm=False)
        self.sigma_1 = Conv3dReLU(embed_dim // 2, embed_dim // 2, 3, 3 // 2, use_batchnorm=False)
        self.sigma_2 = nn.Conv3d(embed_dim // 2, 1, kernel_size=3, padding=1)
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.wts_act = nn.Sigmoid()
        self.eps = 1e-6
        self.hyper_model = HyperBlocks()

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor], hyper_val: torch.Tensor):
        """Forward pass for the TransMorphTVFSPR model.

        Args:
            inputs: Tuple of moving and fixed images (mov, fix), each of shape
                ``(B, 1, *config.img_size)``.
            hyper_val: Hyperparameter value(s) of shape ``(1, 1)``, embedded by
                :class:`HyperBlocks` and shared across the batch.

        Returns:
            If ``SVF`` is True, ``(pos_flow, neg_flow, x_weight)`` with the forward
            and inverse displacement fields (clamped to ``[-100, 100]``);
            otherwise ``(flow, x_weight)``. Fields have shape
            ``(B, 3, *config.img_size)``; ``x_weight`` is the spatial
            regularization weight map of shape ``(B, 1, *config.img_size)`` with
            values in ``[1e-6, 1]``.

        """
        hyper = self.hyper_model(hyper_val)
        mov, fix = inputs
        x_cat = torch.cat((mov, fix), dim=1)
        x_s1 = self.avg_pool(x_cat)
        out_feats = self.transformer(x_cat, hyper)
        if self.if_convskip:
            f3 = self.c1(x_s1, hyper)
        else:
            f3 = None
        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]

        else:
            f1 = None
            f2 = None

        f0 = out_feats[-1]
        x = self.up0(f0, f1, hyper=hyper)
        x = self.up1(x, f2, hyper=hyper)
        xx = self.up2(x, f3, hyper=hyper)
        def_x = mov.clone()
        flow_previous = torch.zeros((mov.shape[0], 3, self.img_size[0], self.img_size[1], self.img_size[2])).to(
            mov.device
        )
        flows = []

        # flow integration
        xs = 0
        for t in range(self.time_steps):
            f_out = self.cs[t](torch.cat((def_x, fix), dim=1), hyper)
            x = self.up3s[t](xx, f_out, hyper=hyper)
            xs += x
            flow = self.reg_heads[t](x, hyper)
            flows.append(flow)
            if self.composition == 'composition':
                flow_new = flow_previous + self.spatial_trans_half(flow, flow_previous)
            else:
                # essentially addition
                flow_new = flow_previous + self.spatial_trans_half(flow, flow)
            def_x = self.spatial_trans_half(mov, flow_new)
            flow_previous = flow_new
        flow = flow_new

        x_weight = self.sigma_0(xs, hyper)  # hyper-conditioned blocks (previously called without it)
        x_weight = self.sigma_1(x_weight, hyper)
        x_weight = self.sigma_2(x_weight)
        x_weight = self.wts_act(x_weight)
        x_weight = torch.clamp(x_weight, self.eps, 1.0)

        if self.SVF:
            pos_flow = self.integrate(flow)
            pos_flow = torch.clamp(pos_flow, -100, 100)
            neg_flow = -flow
            neg_flow = self.integrate(neg_flow)
            neg_flow = torch.clamp(neg_flow, -100, 100)
            return pos_flow, neg_flow, x_weight
        return flow, x_weight


CONFIGS = {
    'TransMorph-3-LVL': configs.get_3DTransMorph3Lvl_config(),
    'TransMorph-3-LVL-DWin': configs.get_3DTransMorphDWin3Lvl_config(),
    'TransMorph': configs.get_3DTransMorph_config(),
}
