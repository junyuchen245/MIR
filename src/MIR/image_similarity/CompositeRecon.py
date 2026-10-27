"""Composite reconstruction loss for image synthesis."""

import torch
import torch.nn as nn


class CompRecon(nn.Module):
    """Composite reconstruction loss for image synthesis without GANs.

    The loss is ``charb_weight * L_charb + grad_weight * L_grad + ff_weight * L_ff`` where

    - ``L_charb``: Charbonnier (smooth L1) loss ``mean(sqrt(d^2 + eps^2) - eps)`` to preserve edges,
    - ``L_grad``: gradient-difference loss (mean absolute difference of absolute forward differences along the
      three spatial axes, averaged over axes) to align edges,
    - ``L_ff``: focal-frequency loss ``mean(|F(pred) - F(target)|^(alpha + beta))`` on the 3D FFT over the last
      three dims, to boost high-frequency detail.

    All terms use mean reduction; inputs must be 5D (``[B, C, H, W, D]``).

    Args:
        charb_weight (float): Weight for Charbonnier loss.
        grad_weight (float): Weight for gradient difference loss.
        ff_weight (float): Weight for focal frequency loss.
        eps (float): Charbonnier smoothing constant.
        alpha (float): Exponent of the focal-frequency weight ``|diff|^alpha``.
        beta (float): Exponent applied to the spectrum difference magnitude ``|diff|^beta``.
    """

    def __init__(
        self,
        charb_weight: float = 100.0,
        grad_weight: float = 10.0,
        ff_weight: float = 1.0,
        eps: float = 1e-3,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.charb_weight = charb_weight
        self.grad_weight = grad_weight
        self.ff_weight = ff_weight
        self.eps = eps
        self.alpha = alpha
        self.beta = beta

    def _charbonnier(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Charbonnier (smooth L1) loss.

        Args:
            pred: Predicted image tensor.
            target: Target image tensor.

        Returns:
            Scalar loss.

        """
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps**2) - self.eps)

    def _grad_diff(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute gradient-difference loss.

        Args:
            pred: Predicted image tensor (B, C, H, W, D).
            target: Target image tensor (B, C, H, W, D).

        Returns:
            Scalar loss.

        """
        # assume pred/target shape (B, C, H, W, D) or (B, 1, H, W, D)
        dx_pred = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
        dx_tgt = torch.abs(target[:, :, 1:, :, :] - target[:, :, :-1, :, :])
        dx = torch.mean(torch.abs(dx_pred - dx_tgt))

        dy_pred = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
        dy_tgt = torch.abs(target[:, :, :, 1:, :] - target[:, :, :, :-1, :])
        dy = torch.mean(torch.abs(dy_pred - dy_tgt))

        dz_pred = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
        dz_tgt = torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])
        dz = torch.mean(torch.abs(dz_pred - dz_tgt))

        return (dx + dy + dz) / 3.0

    def _focal_freq(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal-frequency loss.

        Args:
            pred: Predicted image tensor.
            target: Target image tensor.

        Returns:
            Scalar loss.

        """
        # FFT over the last three dims
        Fp = torch.fft.fftn(pred, dim=(-3, -2, -1))
        Ft = torch.fft.fftn(target, dim=(-3, -2, -1))
        diff = Fp - Ft
        mag = torch.abs(diff)
        w = mag.pow(self.alpha)
        return torch.mean(w * mag.pow(self.beta))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute composite reconstruction loss.

        Args:
            pred: Predicted (synthesised) image tensor ``[B, C, H, W, D]``.
            target: Target image tensor with the same shape.

        Returns:
            Scalar weighted loss to minimise.

        """
        l_char = self._charbonnier(pred, target)
        l_grad = self._grad_diff(pred, target)
        l_ff = self._focal_freq(pred, target)

        return self.charb_weight * l_char + self.grad_weight * l_grad + self.ff_weight * l_ff
