"""Local regularizers for deformation regularization.

This module contains regularization terms used for spatially varying
deformation regularization.
"""

from __future__ import annotations

from typing import Optional, Union

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
    """Negative log-likelihood term for Beta prior on weights.

    Implements ``(1 - alpha) * mean(log(w))``, the negative log-density of a ``Beta(alpha, 1)`` prior up to a
    constant, with ``w`` clamped to ``[eps, 1]``.

    Args:
        eps: Lower clamp for the weights before taking the logarithm.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.beta = 1.0

    def forward(self, weights: torch.Tensor, alpha: Union[float, torch.Tensor]) -> torch.Tensor:
        """Compute Beta prior regularization.

        Args:
            weights: Tensor of weights (any shape), expected in (0, 1].
            alpha: Beta distribution alpha parameter.

        Returns:
            Scalar regularization loss (mean over all elements).

        """
        lambdas = torch.clamp(weights, self.eps, 1.0)
        log_beta = torch.log(lambdas)
        return (1.0 - alpha) * log_beta.mean()


class logGaussian(torch.nn.Module):
    """Gaussian prior regularizer for weights.

    Implements ``inv_sigma2 * mean((w - 1)^2)`` with ``w`` clamped to ``[eps, gaus_bond]``, i.e. a Gaussian prior
    centred at 1.

    Args:
        gaus_bond: Upper clamp for the weights.
        eps: Lower clamp for the weights.
    """

    def __init__(self, gaus_bond: float = 5.0, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gaus_bond = gaus_bond

    def forward(self, weights: torch.Tensor, inv_sigma2: Union[float, torch.Tensor]) -> torch.Tensor:
        """Compute Gaussian prior regularization.

        Args:
            weights: Tensor of weights (any shape).
            inv_sigma2: Inverse variance scalar/tensor.

        Returns:
            Scalar regularization loss (a tensor-valued ``inv_sigma2`` broadcasts against the scalar mean).

        """
        weights = torch.clamp(weights, self.eps, self.gaus_bond)
        return inv_sigma2 * torch.mean((weights - 1.0) ** 2)


class LocalGrad3d(torch.nn.Module):
    """Local 3D gradient loss (spatially weighted diffusion regulariser).

    Forward differences of the field along each spatial axis are penalised (``|d|`` or ``d^2``), multiplied by the
    spatial weight map (cropped to match the difference), averaged over all elements per axis, and the three axis
    means are averaged.

    Args:
        penalty: ``'l2'`` squares the differences; any other value uses absolute differences (``'l1'``).
        loss_mult: Optional multiplier applied to the final loss.
    """

    def __init__(self, penalty: str = 'l1', loss_mult: Optional[float] = None):
        super().__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
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


class LocalGrad2d(torch.nn.Module):
    """Local 2D gradient loss (spatially weighted), the 2D counterpart of ``LocalGrad3d``.

    Args:
        penalty: ``'l2'`` squares the differences; any other value uses absolute differences (``'l1'``).
        loss_mult: Optional multiplier applied to the final loss.
    """

    def __init__(self, penalty: str = 'l1', loss_mult: Optional[float] = None):
        super().__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Compute weighted 2D gradient regularization.

        Args:
            y_pred: Predicted displacement/field tensor (B, C, H, W).
            weight: Spatial weight tensor (B, 1, H, W).

        Returns:
            Scalar weighted gradient penalty.

        """
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        dy = _apply_penalty(dy, self.penalty)
        dx = _apply_penalty(dx, self.penalty)

        dx_loss = torch.mean(dx * weight[:, :, :, 1:])
        dy_loss = torch.mean(dy * weight[:, :, 1:, :])
        grad = (dx_loss + dy_loss) / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad
