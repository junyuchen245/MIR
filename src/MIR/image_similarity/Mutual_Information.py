"""Mutual information losses for image registration."""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import torch.nn as nn

class MutualInformation(torch.nn.Module):
    """
    Mutual Information
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        #print(sigma)

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        """Compute mutual information between two images.

        Args:
            y_true: Fixed image tensor (B, 1, ...).
            y_pred: Moving image tensor (B, 1, ...).

        Returns:
            Scalar mutual information.
        """
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.reshape(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def forward(self, y_true, y_pred):
        """Return negative mutual information as a loss.

        Args:
            y_true: Fixed image tensor (B, 1, ...).
            y_pred: Moving image tensor (B, 1, ...).

        Returns:
            Scalar loss.
        """
        return -self.mi(y_true, y_pred)

class localMutualInformation(torch.nn.Module):
    """
    Local Mutual Information for non-overlapping patches
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32, patch_size=5):
        super(localMutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers
        self.patch_size = patch_size

    def local_mi(self, y_true, y_pred):
        """Compute local mutual information over patches.

        Args:
            y_true: Fixed image tensor (B, 1, ...).
            y_pred: Moving image tensor (B, 1, ...).

        Returns:
            Scalar local mutual information.
        """
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """Making image paddings"""
        if len(list(y_pred.size())[2:]) == 3:
            ndim = 3
            x, y, z = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            z_r = -z % self.patch_size
            padding = (z_r // 2, z_r - z_r // 2, y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        elif len(list(y_pred.size())[2:]) == 2:
            ndim = 2
            x, y = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            padding = (y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        else:
            raise Exception('Supports 2D and 3D but not {}'.format(list(y_pred.size())))
        y_true = F.pad(y_true, padding, "constant", 0)
        y_pred = F.pad(y_pred, padding, "constant", 0)

        """Reshaping images into non-overlapping patches"""
        if ndim == 3:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 3, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 3, 1))
        else:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 3, 5)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 2, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 3, 5)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 2, 1))

        """Compute MI"""
        I_a_patch = torch.exp(- self.preterm * torch.square(y_true_patch - vbc))
        I_a_patch = I_a_patch / torch.sum(I_a_patch, dim=-1, keepdim=True)

        I_b_patch = torch.exp(- self.preterm * torch.square(y_pred_patch - vbc))
        I_b_patch = I_b_patch / torch.sum(I_b_patch, dim=-1, keepdim=True)

        pab = torch.bmm(I_a_patch.permute(0, 2, 1), I_b_patch)
        pab = pab / self.patch_size ** ndim
        pa = torch.mean(I_a_patch, dim=1, keepdim=True)
        pb = torch.mean(I_b_patch, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()

    def forward(self, y_true, y_pred):
        """Return negative local mutual information as a loss.

        Args:
            y_true: Fixed image tensor (B, 1, ...).
            y_pred: Moving image tensor (B, 1, ...).

        Returns:
            Scalar loss.
        """
        return -self.local_mi(y_true, y_pred)


class MattesMutualInformation(torch.nn.Module):
    """Mattes mutual information with Parzen windowing (differentiable).

    Uses a cubic B-spline Parzen window for the fixed image and a linear
    (first-order) Parzen window for the moving image, following the common
    Mattes MI formulation.
    """

    def __init__(self, num_bin: int = 32, minval: float = 0.0, maxval: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.num_bins = int(num_bin)
        self.minval = float(minval)
        self.maxval = float(maxval)
        self.eps = float(eps)

        bin_centers = torch.linspace(self.minval, self.maxval, self.num_bins)
        self.register_buffer("bin_centers", bin_centers)
        self.bin_width = (self.maxval - self.minval) / (self.num_bins - 1)

    @staticmethod
    def _bspline_cubic(u: torch.Tensor) -> torch.Tensor:
        """Cubic B-spline kernel for |u|."""
        au = torch.abs(u)
        w = torch.zeros_like(au)
        mask1 = au < 1
        mask2 = (au >= 1) & (au < 2)
        w[mask1] = (4 - 6 * au[mask1] ** 2 + 3 * au[mask1] ** 3) / 6.0
        w[mask2] = ((2 - au[mask2]) ** 3) / 6.0
        return w

    @staticmethod
    def _linear(u: torch.Tensor) -> torch.Tensor:
        """Linear (first-order) kernel for |u|."""
        au = torch.abs(u)
        return torch.clamp(1.0 - au, min=0.0)

    def _parzen_weights(self, x: torch.Tensor, kernel: str) -> torch.Tensor:
        """Compute Parzen window weights for input samples.

        Args:
            x: Tensor (B, N).
            kernel: "bspline" or "linear".

        Returns:
            Weights tensor (B, N, num_bins).
        """
        x = torch.clamp(x, self.minval, self.maxval)
        u = (x.unsqueeze(-1) - self.bin_centers.to(x.device)) / self.bin_width
        if kernel == "bspline":
            return self._bspline_cubic(u)
        if kernel == "linear":
            return self._linear(u)
        raise ValueError("kernel must be 'bspline' or 'linear'.")

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Return negative Mattes mutual information as a loss.

        Args:
            y_true: Fixed image tensor (B, 1, ...).
            y_pred: Moving image tensor (B, 1, ...).

        Returns:
            Scalar loss.
        """
        y_true = y_true.reshape(y_true.shape[0], -1)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)

        w_fix = self._parzen_weights(y_true, kernel="bspline")
        w_mov = self._parzen_weights(y_pred, kernel="linear")

        nb_voxels = y_true.shape[1]
        pxy = torch.bmm(w_fix.permute(0, 2, 1), w_mov) / (nb_voxels + self.eps)
        px = torch.mean(w_fix, dim=1, keepdim=True)
        py = torch.mean(w_mov, dim=1, keepdim=True)

        pxpy = torch.bmm(px.permute(0, 2, 1), py) + self.eps
        mi = torch.sum(pxy * torch.log((pxy + self.eps) / pxpy), dim=(1, 2))
        return -mi.mean()


class NormalizedMutualInformation(MattesMutualInformation):
    """Normalized mutual information (NMI) using Mattes-style Parzen windows.

    NMI = (H(X) + H(Y)) / H(X, Y)
    """

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_true = y_true.reshape(y_true.shape[0], -1)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)

        w_fix = self._parzen_weights(y_true, kernel="bspline")
        w_mov = self._parzen_weights(y_pred, kernel="linear")

        nb_voxels = y_true.shape[1]
        pxy = torch.bmm(w_fix.permute(0, 2, 1), w_mov) / (nb_voxels + self.eps)
        px = torch.mean(w_fix, dim=1) + self.eps  # (B, num_bins)
        py = torch.mean(w_mov, dim=1) + self.eps

        hx = -torch.sum(px * torch.log(px), dim=1)
        hy = -torch.sum(py * torch.log(py), dim=1)
        hxy = -torch.sum(pxy * torch.log(pxy + self.eps), dim=(1, 2))

        nmi = (hx + hy) / (hxy + self.eps)
        return -nmi.mean()