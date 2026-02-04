"""Normalized gradient field (NGF) loss for image registration."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizedGradientFieldLoss(nn.Module):
    """Normalized gradient field loss (differentiable).

    This loss is robust to intensity differences by comparing normalized image gradients.
    Inputs are assumed to be normalized to [0, 1].
    """

    def __init__(self, eps: float = 1e-6, eta: float = 0.01, smooth_window: int = 3):
        super().__init__()
        self.eps = float(eps)
        self.eta = float(eta)
        self.smooth_window = int(smooth_window)

    def _smooth(self, img: torch.Tensor) -> torch.Tensor:
        if self.smooth_window <= 1:
            return img
        k = self.smooth_window
        if img.dim() == 4:
            return F.avg_pool2d(img, kernel_size=k, stride=1, padding=k // 2)
        return F.avg_pool3d(img, kernel_size=k, stride=1, padding=k // 2)

    def _gradients_2d(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = img.shape[1]
        device = img.device
        dtype = img.dtype

        kx = torch.tensor([-0.5, 0.0, 0.5], device=device, dtype=dtype).view(1, 1, 1, 3)
        ky = torch.tensor([-0.5, 0.0, 0.5], device=device, dtype=dtype).view(1, 1, 3, 1)

        kx = kx.repeat(c, 1, 1, 1)
        ky = ky.repeat(c, 1, 1, 1)

        gx = F.conv2d(img, kx, padding=(0, 1), groups=c)
        gy = F.conv2d(img, ky, padding=(1, 0), groups=c)
        return gx, gy

    def _gradients_3d(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = img.shape[1]
        device = img.device
        dtype = img.dtype

        kz = torch.tensor([-0.5, 0.0, 0.5], device=device, dtype=dtype).view(1, 1, 3, 1, 1)
        ky = torch.tensor([-0.5, 0.0, 0.5], device=device, dtype=dtype).view(1, 1, 1, 3, 1)
        kx = torch.tensor([-0.5, 0.0, 0.5], device=device, dtype=dtype).view(1, 1, 1, 1, 3)

        kz = kz.repeat(c, 1, 1, 1, 1)
        ky = ky.repeat(c, 1, 1, 1, 1)
        kx = kx.repeat(c, 1, 1, 1, 1)

        gz = F.conv3d(img, kz, padding=(1, 0, 0), groups=c)
        gy = F.conv3d(img, ky, padding=(0, 1, 0), groups=c)
        gx = F.conv3d(img, kx, padding=(0, 0, 1), groups=c)
        return gx, gy, gz

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        if y_true.dim() != y_pred.dim():
            raise ValueError("y_true and y_pred must have the same number of dimensions.")
        if y_true.dim() not in {4, 5}:
            raise ValueError("Expected 4D (B,C,H,W) or 5D (B,C,D,H,W) tensors.")

        y_true = self._smooth(y_true)
        y_pred = self._smooth(y_pred)

        if y_true.dim() == 4:
            gx_t, gy_t = self._gradients_2d(y_true)
            gx_p, gy_p = self._gradients_2d(y_pred)

            dot = gx_t * gx_p + gy_t * gy_p
            norm_t = gx_t.pow(2) + gy_t.pow(2)
            norm_p = gx_p.pow(2) + gy_p.pow(2)
        else:
            gx_t, gy_t, gz_t = self._gradients_3d(y_true)
            gx_p, gy_p, gz_p = self._gradients_3d(y_pred)

            dot = gx_t * gx_p + gy_t * gy_p + gz_t * gz_p
            norm_t = gx_t.pow(2) + gy_t.pow(2) + gz_t.pow(2)
            norm_p = gx_p.pow(2) + gy_p.pow(2) + gz_p.pow(2)

        ngf = (dot.pow(2) + self.eps) / ((norm_t + self.eta ** 2) * (norm_p + self.eta ** 2) + self.eps)
        return 1.0 - ngf.mean()
