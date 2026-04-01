"""Local regularizers for deformation regularization.

This module contains regularization terms used for spatially varying
deformation regularization.
"""

from __future__ import annotations

import torch


def _apply_penalty(tensor: torch.Tensor, penalty: str) -> torch.Tensor:
    """Apply the configured penalty while preserving legacy behavior.

    Any value other than ``"l2"`` behaves like ``"l1"``, matching the
    previous implementation.
    """
    if penalty == 'l2':
        return tensor.pow(2)
    return tensor

class logBeta(torch.nn.Module):
    """Negative log-likelihood term for Beta prior on weights."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.beta = 1.

    def forward(self, weights, alpha):
        """Compute Beta prior regularization.

        Args:
            weights: Tensor of weights.
            alpha: Beta distribution alpha parameter.

        Returns:
            Scalar regularization loss.
        """
        lambdas = torch.clamp(weights, self.eps, 1.0)
        log_beta = torch.log(lambdas)
        return (1. - alpha) * log_beta.mean()

class logGaussian(torch.nn.Module):
    """Gaussian prior regularizer for weights."""

    def __init__(self, gaus_bond=5., eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gaus_bond = gaus_bond

    def forward(self, weights, inv_sigma2):
        """Compute Gaussian prior regularization.

        Args:
            weights: Tensor of weights.
            inv_sigma2: Inverse variance scalar/tensor.

        Returns:
            Scalar regularization loss.
        """
        weights = torch.clamp(weights, self.eps, self.gaus_bond)
        return inv_sigma2 * torch.mean((weights - 1.) ** 2)

class LocalGrad3d(torch.nn.Module):
    """Local 3D gradient loss."""

    def __init__(self, penalty='l1', loss_mult=None):
        super().__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, weight):
        """Compute weighted 3D gradient regularization.

        Args:
            y_pred: Predicted displacement/field tensor (B, C, H, W, D).
            weight: Spatial weight tensor (B, 1, H, W, D).

        Returns:
            Scalar weighted gradient penalty.
        """
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        dy = _apply_penalty(dy, self.penalty)
        dx = _apply_penalty(dx, self.penalty)
        dz = _apply_penalty(dz, self.penalty)

        dx_loss = torch.mean(dx * weight[:, :, :, 1:, :])
        dy_loss = torch.mean(dy * weight[:, :, 1:, :, :])
        dz_loss = torch.mean(dz * weight[:, :, :, :, 1:])
        grad = (dx_loss + dy_loss + dz_loss) / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad
