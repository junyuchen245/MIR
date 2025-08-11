import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import numpy as np
import einops

class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w d -> b h w d c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w d c -> b c h w d')

class Offset_block0(nn.Module):
    def __init__(self, in_channels, num_heads, kernel_size=3):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, num_heads, kernel_size=kernel_size, padding=kernel_size // 2, groups=num_heads, bias=False)
        self.LN = LayerNormProxy(num_heads)
        self.act = nn.GELU()
        self.offsetx = nn.Conv3d(num_heads, num_heads, kernel_size=1, bias=False)
        self.offsety = nn.Conv3d(num_heads, num_heads, kernel_size=1, bias=False)
        self.offsetz = nn.Conv3d(num_heads, num_heads, kernel_size=1, bias=False)
    def forward(self, x):
        x = self.conv3d(x)
        x = self.LN(x)
        x = self.act(x)
        dx = self.offsetx(x).unsqueeze(2)
        dy = self.offsety(x).unsqueeze(2)
        dz = self.offsetz(x).unsqueeze(2)
        x = torch.cat((dx, dy, dz), dim=2)
        return x

class Offset_block(nn.Module):
    def __init__(self, in_channels, num_heads, kernel_size=3):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(in_channels, in_channels//2, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.conv3d_1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.conv3d_1.weight.shape))

        self.conv3d_2 = nn.Conv3d(in_channels//2, in_channels//4, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.conv3d_2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.conv3d_2.weight.shape))

        self.conv3d_3 = nn.Conv3d(in_channels // 4, 3*num_heads, kernel_size=1, bias=False)
        self.conv3d_3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.conv3d_3.weight.shape))

        self.bn_1 = nn.BatchNorm3d(in_channels // 2)
        self.bn_2 = nn.BatchNorm3d(in_channels // 4)
        self.relu_1 = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.conv3d_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
        x = self.conv3d_3(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class XMlp(nn.Module):
    def __init__(self, in_size, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, reduction=4, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        kernel_size = 3
        self.avg_pool_1 = nn.AdaptiveAvgPool1d(1)
        self.avg_pool_2 = nn.AdaptiveAvgPool1d(1)
        self.se_conv_1 = nn.Sequential(
            nn.Conv3d(in_features*2, in_features//4, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_features//4, in_features//16, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_features//16, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.se_fc_2 = nn.Sequential(
            nn.Linear(in_features*2, in_features // reduction, bias=False),
            act_layer(),
            nn.Linear(in_features // reduction, in_features, bias=False),
            nn.Sigmoid()
        )
        self.H, self.W, self.T = in_size

    def forward(self, x, y):
        N, D, C = y.shape
        y_conv = torch.reshape(y.permute(0, 2, 1), (N, C, self.H, self.W, self.T))
        x_conv = torch.reshape(x.permute(0, 2, 1), (N, C, self.H, self.W, self.T))
        y_se = self.se_conv_1(torch.cat((x_conv, y_conv), dim=1)).view(N, 1, D).permute(0, 2, 1)
        x = x * y_se
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x_se = self.avg_pool_2(torch.cat((x, y), dim=2).permute(0, 2, 1)).view(N, 2*C)
        x_se = self.se_fc_2(x_se).view(N, 1, C)
        x = x * x_se
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, L, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    B, H, W, L, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], L // window_size[2], window_size[2], C)

    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows

def deform_window_partition(x, window_size):
    """
    Args:
        x: (Head_size, B, H, W, L, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    B, Head, H, W, L, C = x.shape
    x = x.view(B, Head, H // window_size[0], window_size[0], W // window_size[1], window_size[1], L // window_size[2], window_size[2], C)

    windows = x.permute(0, 1, 2, 4, 6, 3, 5, 7, 8).contiguous().view(B, Head, -1, window_size[0], window_size[1], window_size[2], C)
    return windows

def window_reverse(windows, window_size, H, W, L):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        L (int): Length of image
    Returns:
        x: (B, H, W, L, C)
    """
    B = int(windows.shape[0] / (H * W * L / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, H // window_size[0], W // window_size[1], L // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, L, -1)#
    return x

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, rpe=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*Wt-1, nH

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

        self.m_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.m_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.f_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.f_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.m_proj = nn.Linear(dim*2, dim)
        self.f_proj = nn.Linear(dim*2, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww, Wt*Ww) or None
        """
        mov, fix, dmov, dfix = x
        B_, N, C = mov.shape
        mov_kv = self.m_kv(mov).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        fix_kv = self.f_kv(fix).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        dmov_q = self.m_q(dmov).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        dfix_q = self.f_q(dfix).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        dmov_Q, mov_K, mov_V = dmov_q[0], mov_kv[0], mov_kv[1]  # make torchscript happy (cannot use tensor as tuple)
        dfix_Q, fix_K, fix_V = dfix_q[0], fix_kv[0], fix_kv[1]  # make torchscript happy (cannot use tensor as tuple)

        dmov_Q = dmov_Q * self.scale
        dfix_Q = dfix_Q * self.scale
        mov_attn = (dfix_Q @ mov_K.transpose(-2, -1))
        fix_attn = (dmov_Q @ fix_K.transpose(-2, -1))

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wt,Wh*Ww*Wt,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wt, Wh*Ww*Wt
            mov_attn = mov_attn + relative_position_bias.unsqueeze(0)
            fix_attn = fix_attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mov_attn = mov_attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            mov_attn = mov_attn.view(-1, self.num_heads, N, N)
            mov_attn = self.softmax(mov_attn)
            fix_attn = fix_attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            fix_attn = fix_attn.view(-1, self.num_heads, N, N)
            fix_attn = self.softmax(fix_attn)
        else:
            mov_attn = self.softmax(mov_attn)
            fix_attn = self.softmax(fix_attn)

        mov_attn = self.attn_drop(mov_attn)
        fix_attn = self.attn_drop(fix_attn)
        mov = (mov_attn @ mov_V).transpose(1, 2).reshape(B_, N, C)
        mov_f = (fix_attn.transpose(2, 3) @ mov_V).transpose(1, 2).reshape(B_, N, C)
        mov = torch.cat((mov, mov_f), dim=2)
        mov = self.m_proj(mov)
        mov = self.proj_drop(mov)

        fix = (fix_attn @ fix_V).transpose(1, 2).reshape(B_, N, C)
        fix_m = (mov_attn.transpose(2, 3) @ fix_V).transpose(1, 2).reshape(B_, N, C)
        fix = torch.cat((fix, fix_m), dim=2)
        fix = self.f_proj(fix)
        fix = self.proj_drop(fix)

        return mov, fix

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, rpe=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, img_size=(160, 160, 160), dwin_size=3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= min(self.shift_size) < min(self.window_size), "shift_size must in 0-window_size, shift_sz: {}, win_size: {}".format(self.shift_size, self.window_size)

        self.m_norm1 = norm_layer(dim)
        self.f_norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, rpe=rpe, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.m_norm2 = norm_layer(dim)
        self.f_norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.m_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop) #XMlp(in_size=img_size,
        self.f_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)#XMlp(in_size=img_size,
        
        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] - img_size[0] % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - img_size[1] % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] - img_size[2] % self.window_size[2]) % self.window_size[2]
        grid_shape = (img_size[0]+pad_l+pad_r, img_size[1]+pad_t+pad_b, img_size[2]+pad_f+pad_h)
        vectors = [torch.arange(0, s) for s in grid_shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        #for i in range(len(img_size)):
        #    grid[:, i, ...] = 2 * (grid[:, i, ...] / (img_size[i] - 1) - 0.5)
        sample_grid1 = grid.cuda()
        sample_grid1.requires_grad = False
        sample_grid1 = sample_grid1.permute(0, 2, 3, 4, 1)
        sample_grid1 = sample_grid1[..., [2, 1, 0]]
        self.grid = window_partition(sample_grid1, self.window_size)
        self.grid.requires_grad = False

        nW, wW, wH, wD, gC = self.grid.shape
        self.nW = nW
        self.wW = wW
        self.wH = wH
        self.wD = wD
        self.grid = self.grid.view(1, nW, -1, 3)
        self.grid = torch.unsqueeze(self.grid, 0)

        self.offset_block_1 = Offset_block(self.dim*2, num_heads, dwin_size)
        self.offset_block_2 = Offset_block(self.dim*2, num_heads, dwin_size)

        self.H = None
        self.W = None
        self.T = None


    def forward(self, x, mask_matrix):
        mov, fix = x
        H, W, T = self.H, self.W, self.T
        B, L, C = mov.shape
        assert L == H * W * T, "input feature has wrong size"

        mov_shortcut = mov
        fix_shortcut = fix

        mov = self.m_norm1(mov)
        mov = mov.view(B, H, W, T, C)

        fix = self.f_norm1(fix)
        fix = fix.view(B, H, W, T, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] - T % self.window_size[2]) % self.window_size[2]
        mov = nnf.pad(mov, (0, 0, pad_f, pad_h, pad_t, pad_b, pad_l, pad_r))
        _, Hp, Wp, Tp, _ = mov.shape
        fix = nnf.pad(fix, (0, 0, pad_f, pad_h, pad_t, pad_b, pad_l, pad_r))
        _, Hp, Wp, Tp, _ = fix.shape

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_mov = torch.roll(mov, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            shifted_fix = torch.roll(fix, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_mov = mov
            shifted_fix = fix
            attn_mask = None

        offset_range = torch.tensor([Hp, Wp, Tp], device='cuda').reshape(1, 1, 3, 1, 1, 1) * 0.5
        off_mov = shifted_mov.permute(0, 4, 1, 2, 3)
        off_fix = shifted_fix.permute(0, 4, 1, 2, 3)
        offset_mov_org = self.offset_block_1(torch.cat((off_mov, off_fix), dim=1))
        offset_mov_org = offset_mov_org.tanh().reshape(B, self.num_heads, 3, Hp, Wp, Tp).mul(offset_range)
        offset_mov = deform_window_partition(offset_mov_org.permute(0, 1, 3, 4, 5, 2), self.window_size)
        offset_mov = torch.reshape(offset_mov, (B, self.num_heads, self.nW, self.wW * self.wH * self.wD, 3))
        offset_mov = offset_mov[..., [2, 1, 0]]
        offset_mov = torch.unsqueeze(offset_mov, 2)

        offset_fix_org = self.offset_block_2(torch.cat((off_fix, off_mov), dim=1))
        offset_fix_org = offset_fix_org.tanh().reshape(B, self.num_heads, 3, Hp, Wp, Tp).mul(offset_range)
        offset_fix = deform_window_partition(offset_fix_org.permute(0, 1, 3, 4, 5, 2), self.window_size)
        offset_fix = torch.reshape(offset_fix, (B, self.num_heads, self.nW, self.wW * self.wH * self.wD, 3))
        offset_fix = offset_fix[..., [2, 1, 0]]
        offset_fix = torch.unsqueeze(offset_fix, 2)

        dmov = torch.clone(shifted_mov)
        dfix = torch.clone(shifted_fix)

        # deformable window for moving image features
        offset_mov_ = offset_mov.repeat(1, 1, C // self.num_heads, 1, 1, 1).view(B, C, self.nW, self.wW * self.wH * self.wD, 3)
        offset_mov_ = offset_mov_.view(B * C, self.nW, self.wW * self.wH * self.wD, 3)
        offset_mov_ = torch.unsqueeze(offset_mov_, 1)
        offset_fix_ = offset_fix.repeat(1, 1, C // self.num_heads, 1, 1, 1).view(B, C, self.nW, self.wW * self.wH * self.wD, 3)
        offset_fix_ = offset_fix_.view(B * C, self.nW, self.wW * self.wH * self.wD, 3)
        offset_fix_ = torch.unsqueeze(offset_fix_, 1)
        mov_locs = self.grid.repeat(B*C, 1, 1, 1, 1) + offset_mov_
        fix_locs = self.grid.repeat(B*C, 1, 1, 1, 1) + offset_fix_
        # need to normalize grid values to [-1, 1] for resampler
        shape = [Tp, Hp, Wp]
        for i in range(len(shape)):
            mov_locs[..., i] = 2 * (mov_locs[..., i] / (shape[i] - 1) - 0.5)
            fix_locs[..., i] = 2 * (fix_locs[..., i] / (shape[i] - 1) - 0.5)

        dmov = dmov.permute(0, 4, 1, 2, 3)
        dmov = torch.reshape(dmov, (B * C, 1, Hp, Wp, Tp))
        dmov_windows = nnf.grid_sample(dmov, mov_locs)
        dmov_windows = torch.reshape(dmov_windows, (B, C, self.nW, self.window_size[0], self.window_size[1], self.window_size[2])).permute(0, 2, 3, 4, 5, 1)
        dmov_windows = torch.reshape(dmov_windows, (-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C))

        # deformable window for moving image features
        dfix = dfix.permute(0, 4, 1, 2, 3)
        dfix = torch.reshape(dfix, (B * C, 1, Hp, Wp, Tp))
        dfix_windows = nnf.grid_sample(dfix, fix_locs)
        dfix_windows = torch.reshape(dfix_windows, (B, C, self.nW, self.window_size[0], self.window_size[1], self.window_size[2])).permute(0, 2, 3, 4, 5, 1)
        dfix_windows = torch.reshape(dfix_windows, (-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C))

        # partition windows
        mov_windows = window_partition(shifted_mov, self.window_size)  # nW*B, window_size, window_size, C
        mov_windows = mov_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # nW*B, window_size**3, C

        fix_windows = window_partition(shifted_fix, self.window_size)  # nW*B, window_size, window_size, C
        fix_windows = fix_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # nW*B, window_size**3, C

        # Deformable Cross W-MSA/SW-MSA
        mov_attn_windows, fix_attn_windows = self.attn((mov_windows, fix_windows, dmov_windows, dfix_windows), mask=attn_mask)  # nW*B, window_size**3, C

        # merge windows
        mov_attn_windows = mov_attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_mov = window_reverse(mov_attn_windows, self.window_size, Hp, Wp, Tp)  # B H' W' L' C

        fix_attn_windows = fix_attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_fix = window_reverse(fix_attn_windows, self.window_size, Hp, Wp, Tp)  # B H' W' L' C

        # reverse cyclic shift
        if min(self.shift_size) > 0:
            mov = torch.roll(shifted_mov, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
            fix = torch.roll(shifted_fix, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            mov, fix = shifted_mov, shifted_fix

        if pad_r > 0 or pad_b > 0 or pad_h > 0:
            mov = mov[:, :H, :W, :T, :].contiguous()
            fix = fix[:, :H, :W, :T, :].contiguous()
            
        mov = mov.view(B, H * W * T, C)
        fix = fix.view(B, H * W * T, C)

        # FFN
        mov = mov_shortcut + self.drop_path(mov)
        fix = fix_shortcut + self.drop_path(fix)
        mov_norm = self.m_norm2(mov)
        fix_norm = self.f_norm2(fix)
        mov = mov + self.drop_path(self.m_mlp(mov_norm))#, fix_norm))
        fix = fix + self.drop_path(self.f_mlp(fix_norm))#, mov_norm))
        return mov, fix

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, (8//reduce_factor) * dim, bias=False)
        self.norm = norm_layer(8 * dim)


    def forward(self, x, H, W, T):
        """
        x: B, H*W*T, C
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
    """ A basic Swin Transformer layer for one stage.
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
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 rpe=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 img_size=(160, 160, 160),
                 dwin_size=3):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                rpe=rpe,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                img_size=img_size,
                dwin_size=dwin_size)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, reduce_factor=self.pat_merg_rf)
        else:
            self.downsample = None

    def forward(self, x, H, W, T):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W*T, C).
            H, W, T: Spatial resolution of the input feature.
        """
        mov, fix = x
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        Tp = int(np.ceil(T / self.window_size[2])) * self.window_size[2]
        img_mask = torch.zeros((1, Hp, Wp, Tp, 1), device=mov.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        t_slices = (slice(0, -self.window_size[2]),
                    slice(-self.window_size[2], -self.shift_size[2]),
                    slice(-self.shift_size[2], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    img_mask[:, h, w, t, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        x = (mov, fix)
        for blk in self.blocks:
            blk.H, blk.W, blk.T = H, W, T
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        mov, fix = x
        if self.downsample is not None:
            mov_down = self.downsample(mov, H, W, T)
            fix_down = self.downsample(fix, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return (mov, fix), H, W, T, (mov_down, fix_down), Wh, Ww, Wt
        else:
            return (mov, fix), H, W, T, (mov, fix), H, W, T

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
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

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W, T = x.size()
        if T % self.patch_size[2] != 0:
            x = nnf.pad(x, (0, self.patch_size[2] - T % self.patch_size[2]))
        if W % self.patch_size[1] != 0:
            x = nnf.pad(x, (0, 0, 0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww Wt
        if self.norm is not None:
            Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x

class SinusoidalPositionEmbedding(nn.Module):
    '''
    Rotary Position Embedding
    '''
    def __init__(self,):
        super(SinusoidalPositionEmbedding, self).__init__()

    def forward(self, x):
        batch_sz, n_patches, hidden = x.shape
        position_ids = torch.arange(0, n_patches).float().cuda()
        indices = torch.arange(0, hidden//2).float().cuda()
        indices = torch.pow(10000.0, -2 * indices / hidden)
        embeddings = torch.einsum('b,d->bd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (1, n_patches, hidden))
        return embeddings

class SinPositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(SinPositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels/6)*2)
        if channels % 2:
            channels += 1
        self.channels = channels
        self.inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        #self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
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
        emb = torch.zeros((x,y,z,self.channels*3),device=tensor.device).type(tensor.type())
        emb[:,:,:,:self.channels] = emb_x
        emb[:,:,:,self.channels:2*self.channels] = emb_y
        emb[:,:,:,2*self.channels:] = emb_z
        emb = emb[None,:,:,:,:orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return emb.permute(0, 4, 1, 2, 3)

class DefSwinTransformerV2(nn.Module):
    r""" Deformable Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 rpe=True,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 img_size=(160, 160, 160),
                 dwin_size=(3, 3, 3)):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        print('Depths: {}'.format(depths))
        print('DWin kernel size: {}'.format(dwin_size))
        print('window_size: {}'.format(window_size))
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=1, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1], pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinPositionalEncoding3D(embed_dim).cuda()
            #self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                rpe=rpe,
                                qk_scale=qk_scale,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint,
                               pat_merg_rf=pat_merg_rf,
                               img_size=(img_size[0]//patch_size//2**i_layer, img_size[1]//patch_size//2**i_layer, img_size[2]//patch_size//2**i_layer),
                               dwin_size=dwin_size[i_layer])
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'm_norm{i_layer}'
            self.add_module(layer_name, layer)
            layer = norm_layer(num_features[i_layer])
            layer_name = f'f_norm{i_layer}'
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
            for i in range(0, self.frozen_stages - 1):
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
                trunc_normal_(m.weight, std=.02)
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

    def forward(self, x):
        mov, fix = x
        """Forward function."""
        mov = self.patch_embed(mov)
        fix = self.patch_embed(fix)
        
        Wh, Ww, Wt = mov.size(2), mov.size(3), mov.size(4)

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            mov = (mov + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
            fix = (fix + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            mov = (mov + self.pos_embd(mov)).flatten(2).transpose(1, 2)
            fix = (fix + self.pos_embd(mov)).flatten(2).transpose(1, 2)
        else:
            mov = mov.flatten(2).transpose(1, 2)
            fix = fix.flatten(2).transpose(1, 2)
        mov = self.pos_drop(mov)
        fix = self.pos_drop(fix)
        
        
        x = (mov, fix)
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            if i in self.out_indices:
                m_norm_layer = getattr(self, f'm_norm{i}')
                f_norm_layer = getattr(self, f'f_norm{i}')
                mov_out, fix_out = x_out
                mov_out = m_norm_layer(mov_out)
                fix_out = f_norm_layer(fix_out)
                mov_out = mov_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                fix_out = fix_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                out = (mov_out, fix_out)#torch.cat(x_out, dim=4)
                outs.append(out)
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(DefSwinTransformerV2, self).train(mode)
        self._freeze_stages()