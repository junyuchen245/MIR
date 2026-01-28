"""Normalized cross-correlation (NCC) loss implementations."""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import torch.nn as nn



def gaussian(window_size, sigma):
    """Create a 1D Gaussian kernel.

    Args:
        window_size: Kernel size.
        sigma: Standard deviation.

    Returns:
        1D normalized Gaussian kernel tensor.
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """Create a 2D Gaussian window tensor.

    Args:
        window_size: Kernel size.
        channel: Number of channels.

    Returns:
        2D window tensor of shape (C,1,H,W).
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def create_window_3D(window_size, channel):
    """Create a 3D Gaussian window tensor.

    Args:
        window_size: Kernel size.
        channel: Number of channels.

    Returns:
        3D window tensor of shape (C,1,H,W,D).
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

class NCC_gauss(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss via Gaussian
    """

    def __init__(self, win=9):
        super(NCC_gauss, self).__init__()
        self.win = [win]*3
        self.filt = self.create_window_3D(win, 1).to("cuda")

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window_3D(self, window_size, channel):
        _1D_window = gaussian(window_size, 2.).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t())
        _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                      window_size).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
        #print(_2D_window)
        return window

    def forward(self, y_true, y_pred):
        """Compute Gaussian-windowed NCC loss.

        Args:
            y_true: Fixed image tensor (B, 1, ...).
            y_pred: Moving image tensor (B, 1, ...).

        Returns:
            Scalar NCC loss (negative mean NCC).
        """

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # compute filters
        pad_no = math.floor(self.win[0] / 2)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        mu1 = conv_fn(Ii, self.filt, padding=pad_no)
        mu2 = conv_fn(Ji, self.filt, padding=pad_no)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = conv_fn(Ii * Ii, self.filt, padding=pad_no) - mu1_sq
        sigma2_sq = conv_fn(Ji * Ji, self.filt, padding=pad_no) - mu2_sq
        sigma12 = conv_fn(Ii * Ji, self.filt, padding=pad_no) - mu1_mu2

        cc = (sigma12 * sigma12)/(sigma1_sq * sigma2_sq + 1e-5)
        return -torch.mean(cc)

class NCC(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(NCC, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):
        """Compute NCC loss over a local window.

        Args:
            y_true: Fixed image tensor (B, 1, ...).
            y_pred: Moving image tensor (B, 1, ...).

        Returns:
            Scalar NCC loss (negative mean NCC).
        """
        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else [self.win] * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")/float(np.prod(win))

        pad_no = win[0] // 2

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        mu1 = conv_fn(Ii, sum_filt, padding=padding, stride=stride)
        mu2 = conv_fn(Ji, sum_filt, padding=padding, stride=stride)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = conv_fn(Ii * Ii, sum_filt, padding=padding, stride=stride) - mu1_sq
        sigma2_sq = conv_fn(Ji * Ji, sum_filt, padding=padding, stride=stride) - mu2_sq
        sigma12 = conv_fn(Ii * Ji, sum_filt, padding=padding, stride=stride) - mu1_mu2

        cc = (sigma12 * sigma12) / (sigma1_sq * sigma2_sq + 1e-5)
        return - torch.mean(cc)


class NCC_mok(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=9, eps=1e-5):
        super(NCC_mok, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)

class NCC_mok2(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=9, eps=1e-5, channel=1):
        super(NCC_mok2, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win
        self.channel = channel

    def forward(self, I, J):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((self.channel, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2), groups=self.channel)
        J_sum = conv_fn(J, weight, padding=int(win_size/2), groups=self.channel)
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2), groups=self.channel)
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2), groups=self.channel)
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2), groups=self.channel)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)
        # print(I_var.min(), I_var.max())
        # print(cc.min(), cc.max())

        # return negative cc.
        return -1.0 * torch.mean(cc)

class NCC_vxm(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None, ndims=3):
        super(NCC_vxm, self).__init__()
        self.win = win

        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        self.win = [9] * ndims if self.win is None else self.win
        self.win_size = torch.tensor(np.prod(self.win)).float().cuda()

        # compute filters
        self.sum_filt = torch.ones([1, 1, *self.win]).to("cuda").float()
        self.sum_filt.requires_grad = False
        pad_no = math.floor(self.win[0] / 2)
        if ndims == 1:
            self.stride = (1)
            self.padding = (pad_no)
        elif ndims == 2:
            self.stride = (1, 1)
            self.padding = (pad_no, pad_no)
        else:
            self.stride = (1, 1, 1)
            self.padding = (pad_no, pad_no, pad_no)
        self.ndims = ndims

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % self.ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, self.sum_filt, stride=self.stride, padding=self.padding)
        J_sum = conv_fn(Ji, self.sum_filt, stride=self.stride, padding=self.padding)
        I2_sum = conv_fn(I2, self.sum_filt, stride=self.stride, padding=self.padding)
        J2_sum = conv_fn(J2, self.sum_filt, stride=self.stride, padding=self.padding)
        IJ_sum = conv_fn(IJ, self.sum_filt, stride=self.stride, padding=self.padding)


        u_I = I_sum / self.win_size
        u_J = J_sum / self.win_size
        #print(u_I.max(), J2_sum.max())
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * self.win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * self.win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * self.win_size

        cc = (cross * cross) / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
    
class SingleScaleNCC(nn.Module):
    def __init__(self, window_size, **kwargs):
        super().__init__()
        if isinstance(window_size, int):
            self.window_size = [window_size]
        else:
            self.window_size = window_size

    def forward(self, pred, target):
        """ LNCC loss
            modified based on https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/TransMorph/losses.py
        """
        Ii = target
        Ji = pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = self.window_size * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(pred.device) / np.prod(win)

        pad_no = win[0] // 2

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        mu1 = conv_fn(Ii, sum_filt, padding=padding, stride=stride)
        mu2 = conv_fn(Ji, sum_filt, padding=padding, stride=stride)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = conv_fn(Ii * Ii, sum_filt, padding=padding, stride=stride) - mu1_sq
        sigma2_sq = conv_fn(Ji * Ji, sum_filt, padding=padding, stride=stride) - mu2_sq
        sigma12 = conv_fn(Ii * Ji, sum_filt, padding=padding, stride=stride) - mu1_mu2

        eps = torch.finfo(sigma12.dtype).eps
        cc = (sigma12 * sigma12) / torch.clamp(sigma1_sq * sigma2_sq, min=eps)
        return - torch.mean(cc)

class NCC_vfa(torch.nn.Module):
    """
    Multi-scale NCC from C2FViT: https://github.com/cwmok/C2FViT
    suitable for FP16
    """
    def  __init__(self, window_size=9, scale=1, kernel=3, half_resolution=0):
        super().__init__()
        self.num_scales = scale
        self.kernel = kernel
        self.half_resolution = half_resolution

        self.similarity_metric = []
        for i in range(self.num_scales):
            self.similarity_metric.append(
                        SingleScaleNCC(window_size-(i*2))
            )

    def forward(self, I, J):
        dim = I.dim() - 2
        if self.half_resolution:
            kwargs = {'scale_factor':0.5, 'align_corners':True}
            if dim == 2:
                I = F.interpolate(I, mode='bilinear', **kwargs)
                J = F.interpolate(J, mode='bilinear', **kwargs)
            elif dim == 3:
                I = F.interpolate(I, mode='trilinear', **kwargs)
                J = F.interpolate(J, mode='trilinear', **kwargs)

        if dim == 2:
            pooling_fn = F.avg_pool2d
        elif dim == 3:
            pooling_fn = F.avg_pool3d

        total_NCC = []
        for i in range(self.num_scales):
            current_NCC = self.similarity_metric[i](I, J)
            total_NCC.append(current_NCC / self.num_scales)

            I = pooling_fn(I, kernel_size=self.kernel, stride=2, padding=self.kernel//2, count_include_pad=False)
            J = pooling_fn(J, kernel_size=self.kernel, stride=2, padding=self.kernel//2, count_include_pad=False)

        return sum(total_NCC)
    
class FastNCC(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss by XiJia
    https://github.com/xi-jia/FastLNCC
    
    # For PyTorch versions > 2.0, if there are numerical differences, please add the following code.
    torch.backends.cudnn.allow_tf32 = False
    """

    def __init__(self, win=9):
        super().__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [self.win] * ndims

        # compute filters
        # sum_filt = torch.ones([1, 1, *win]).to("cuda")
        sum_filt = torch.ones([5, 1, *win]).to(Ii.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji
        
        all_five = torch.cat((Ii, Ji, I2, J2, IJ),dim=1)
        all_five_conv = conv_fn(all_five, sum_filt, stride=stride, padding=padding, groups=5)
        I_sum, J_sum, I2_sum, J2_sum, IJ_sum = torch.split(all_five_conv, 1, dim=1)
        
        # I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        # J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        # I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        # J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        # IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        # compute cross correlation
        # win_size = np.prod(win)
        # u_I = I_sum / win_size
        # u_J = J_sum / win_size

        # cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        # I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        # J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size


        # compute cross correlation
        win_size = np.prod(self.win)

        cross = IJ_sum - J_sum/win_size*I_sum
        I_var = I2_sum - I_sum/win_size*I_sum
        J_var = J2_sum - J_sum/win_size*J_sum

        
        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
    
class NCC_fp16(nn.Module):
    """
    Local normalized cross‑correlation loss for 1‑, 2‑ or 3‑D inputs.

    Parameters
    ----------
    win      : int
        Side length of the cubic averaging window.  Default: 9.
    squared  : bool
        • False  → classic NCC   ( σ_xy / √(σ_x σ_y) )  
        • True   → squared NCC   ( σ_xy² / (σ_x σ_y) )
        Default: False.
    eps      : float
        Small constant to avoid divide‑by‑zero.  Default: 1e‑5.
    """

    def __init__(self, win: int = 9, squared: bool = True, eps: float = 1e-5):
        super().__init__()
        self.win = win
        self.squared = squared
        self.eps = eps

        # averaged window filter is cached lazily on first forward pass
        self.register_buffer("_filt", torch.Tensor())
        self._filt_device_dtype = None

    # ------------------------------------------------------------------
    # internal helper
    # ------------------------------------------------------------------
    def _get_filter(self, x: torch.Tensor):
        """Return the averaging filter on the input’s device / dtype (cached)."""
        ndims = x.dim() - 2
        win = [self.win] * ndims
        dev_dtype = (x.device, x.dtype)

        if self._filt.numel() == 0 or self._filt_device_dtype != dev_dtype:
            kernel = torch.ones(1, 1, *win, device=x.device, dtype=x.dtype)
            kernel /= float(np.prod(win))
            self._filt = kernel
            self._filt_device_dtype = dev_dtype

        # repeat across channels for grouped conv
        C = x.shape[1]
        return self._filt.repeat(C, 1, *([1] * ndims))

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        assert y_true.shape == y_pred.shape, "Inputs must have the same shape"
        ndims = y_true.dim() - 2
        assert ndims in (1, 2, 3), f"Only 1‑, 2‑ or 3‑D inputs supported, got {ndims}‑D"

        filt = self._get_filter(y_true)
        conv = getattr(F, f"conv{ndims}d")

        padding = (self.win // 2,) * ndims
        stride = (1,) * ndims
        groups = y_true.shape[1]

        mu_x = conv(y_true, filt, stride=stride, padding=padding, groups=groups)
        mu_y = conv(y_pred, filt, stride=stride, padding=padding, groups=groups)

        sigma_x  = conv(y_true * y_true, filt, stride=stride,
                        padding=padding, groups=groups) - mu_x.pow(2)
        sigma_y  = conv(y_pred * y_pred, filt, stride=stride,
                        padding=padding, groups=groups) - mu_y.pow(2)
        sigma_xy = conv(y_true * y_pred, filt, stride=stride,
                        padding=padding, groups=groups) - mu_x * mu_y

        if self.squared:
            ncc = sigma_xy.pow(2) / (sigma_x * sigma_y + self.eps)
        else:
            ncc = sigma_xy / torch.sqrt(sigma_x * sigma_y + self.eps)

        return -ncc.mean()