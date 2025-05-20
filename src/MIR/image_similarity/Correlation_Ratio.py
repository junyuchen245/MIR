import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import torch.nn as nn

class CorrRatio(torch.nn.Module):
    """
    Correlation Ratio based on Parzen window
    Implemented by Junyu Chen, jchen245@jhmi.edu
    TODO: Under testing

    The Correlation Ratio as a New Similarity Measure for Multimodal Image Registration
    by Roche et al. 1998
    https://link.springer.com/chapter/10.1007/BFb0056301
    """

    def __init__(self, bins=32, sigma_ratio=1):
        super(CorrRatio, self).__init__()
        self.num_bins = bins
        bin_centers = np.linspace(0, 1, num=bins)
        self.vol_bin_centers = Variable(torch.linspace(0, 1, bins), requires_grad=False).cuda().view(1, 1, bins, 1)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 2 / (2 * sigma ** 2)

    def gaussian_kernel(self, diff, preterm):
        return torch.exp(- preterm * torch.square(diff))#torch.exp(-0.5 * (diff ** 2) / (sigma ** 2))

    def correlation_ratio(self, X, Y):
        B, C, H, W, D = Y.shape
        y_flat = Y.reshape(B, C, -1)  # Flatten spatial dimensions
        x_flat = X.reshape(B, C, -1)

        bins = self.vol_bin_centers

        # Calculate distances from each pixel to each bin
        y_expanded = y_flat.unsqueeze(2)  # [B, C, 1, H*W*D]
        diff = y_expanded - bins  # Broadcasted subtraction

        # Apply Parzen window approximation
        weights = self.gaussian_kernel(diff, preterm=self.preterm)
        weights_norm = weights / (torch.sum(weights, dim=-1, keepdim=True)+1e-5)
        # Compute weighted mean intensity in y_pred for each bin
        x_flat_expanded = x_flat.unsqueeze(2)  # Shape: [B, C, 1, H*W*D]
        mean_intensities = torch.sum(weights_norm * x_flat_expanded, dim=3)  # conditional mean, [B, C, bin]
        bin_counts = torch.sum(weights, dim=3)
        # mean_intensities = weighted_sums / (bin_counts + 1e-8)  # Add epsilon to avoid division by zero

        # Compute total mean of y_pred
        total_mean = torch.mean(x_flat, dim=2, keepdim=True) # [B, C, 1]

        # Between-group variance
        between_group_variance = torch.sum(bin_counts * (mean_intensities - total_mean) ** 2, dim=2) / (torch.sum(
            bin_counts, dim=2)+1e-5)

        # Total variance
        total_variance = torch.var(x_flat, dim=2)

        # Correlation ratio
        eta_square = between_group_variance / (total_variance + 1e-5)

        return eta_square.mean()/3

    def forward(self, y_true, y_pred):
        CR = self.correlation_ratio(y_true, y_pred) + self.correlation_ratio(y_pred, y_true)
        return -CR/2

class LocalCorrRatio(torch.nn.Module):
    """
    Localized Correlation Ratio based on Parzen window
    Implemented by Junyu Chen, jchen245@jhmi.edu
    TODO: Under testing

    The Correlation Ratio as a New Similarity Measure for Multimodal Image Registration
    by Roche et al. 1998
    https://link.springer.com/chapter/10.1007/BFb0056301
    """

    def __init__(self, bins=32, sigma_ratio=1, win=9):
        super(LocalCorrRatio, self).__init__()
        self.num_bins = bins
        bin_centers = np.linspace(0, 1, num=bins)
        self.vol_bin_centers = Variable(torch.linspace(0, 1, bins), requires_grad=False).cuda().view(1, 1, bins, 1)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 2 / (2 * sigma ** 2)
        self.win = win

    def gaussian_kernel(self, diff, preterm):
        return torch.exp(- preterm * torch.square(diff))

    def correlation_ratio(self, X, Y):
        B, C, H, W, D = Y.shape

        h_r = -H % self.win
        w_r = -W % self.win
        d_r = -D % self.win
        padding = (d_r // 2, d_r - d_r // 2, w_r // 2, w_r - w_r // 2, h_r // 2, h_r - h_r // 2, 0, 0, 0, 0)
        X = F.pad(X, padding, "constant", 0)
        Y = F.pad(Y, padding, "constant", 0)

        B, C, H, W, D = Y.shape
        num_patch = (H // self.win) * (W // self.win) * (D // self.win)
        x_patch = torch.reshape(X, (B, C, H // self.win, self.win, W // self.win, self.win, D // self.win, self.win))
        x_flat = x_patch.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B*num_patch, C, self.win ** 3)

        y_patch = torch.reshape(Y, (B, C, H // self.win, self.win, W // self.win, self.win, D // self.win, self.win))
        y_flat = y_patch.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B * num_patch, C, self.win ** 3)

        bins = self.vol_bin_centers

        # Calculate distances from each pixel to each bin
        y_expanded = y_flat.unsqueeze(2)  # [B*num_patch, C, 1, win**3]
        diff = y_expanded - bins  # Broadcasted subtraction

        # Apply Parzen window approximation
        weights = self.gaussian_kernel(diff, preterm=self.preterm)
        weights_norm = weights / (torch.sum(weights, dim=-1, keepdim=True)+1e-5)
        # Compute weighted mean intensity in y_pred for each bin
        x_flat_expanded = x_flat.unsqueeze(2)  # Shape: [B*num_patch, C, 1, win**3]
        mean_intensities = torch.sum(weights_norm * x_flat_expanded, dim=3)  # conditional mean, [B*num_patch, C, bin]
        bin_counts = torch.sum(weights, dim=3)
        # mean_intensities = weighted_sums / (bin_counts + 1e-8)  # Add epsilon to avoid division by zero

        # Compute total mean of y_pred
        total_mean = torch.mean(x_flat, dim=2, keepdim=True)  # [B*num_patch, C, 1]

        # Between-group variance
        between_group_variance = torch.sum(bin_counts * (mean_intensities - total_mean) ** 2, dim=2) / torch.sum(
            bin_counts, dim=2)

        # Total variance
        total_variance = torch.var(x_flat, dim=2)

        # Correlation ratio
        eta_square = between_group_variance / (total_variance + 1e-5)

        return eta_square.mean() / 3

    def forward(self, y_true, y_pred):
        CR = self.correlation_ratio(y_true, y_pred) + self.correlation_ratio(y_pred, y_true) #make it symmetric

        shift_size = self.win//2
        y_true = torch.roll(y_true, shifts=(-shift_size, -shift_size, -shift_size), dims=(2, 3, 4))
        y_pred = torch.roll(y_pred, shifts=(-shift_size, -shift_size, -shift_size), dims=(2, 3, 4))

        CR_shifted = self.correlation_ratio(y_true, y_pred) + self.correlation_ratio(y_pred, y_true)
        return -CR/4 - CR_shifted/4