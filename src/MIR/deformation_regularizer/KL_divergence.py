"""KL divergence losses for Gaussian parameter fields."""

import torch
import torch.nn as nn
from MIR import _amp


class KL_divergence(nn.Module):
    """KL divergence between factorized (voxel- and component-wise independent) Gaussian fields.

    Computes ``KL(P || Q)`` for univariate normals at every element and averages over all elements. The
    constructor takes no arguments.
    """

    def __init__(
        self,
    ):
        super().__init__()

    @_amp.fp32_forward
    def forward(self, P, Q) -> torch.Tensor:
        """Compute KL divergence between two Gaussian fields.

        ``exp(log_sigma)`` is used as the standard deviation (``scale``) of ``torch.distributions.Normal``.

        Args:
            P: Tuple (mean_p, log_sigma_p) of tensors shaped (B, d, H, W, D).
            Q: Tuple (mean_q, log_sigma_q) of tensors with the same shape.

        Returns:
            Scalar ``KL(P || Q)`` averaged over all batch, component and voxel elements.

        """
        mean_p, log_sigma_p = P
        mean_q, log_sigma_q = Q
        b, d, _, _, _ = mean_p.shape
        mean_p = mean_p.flatten().unsqueeze(1)
        log_sigma_p = log_sigma_p.flatten().unsqueeze(1)

        mean_q = mean_q.flatten().unsqueeze(1)
        log_sigma_q = log_sigma_q.flatten().unsqueeze(1)

        q_var = torch.exp(log_sigma_q)
        p_var = torch.exp(log_sigma_p)

        p = torch.distributions.Normal(mean_p, p_var)
        q = torch.distributions.Normal(mean_q, q_var)

        kl_loss = torch.distributions.kl_divergence(p, q).mean()
        return kl_loss


class MultiVariateKL_divergence(nn.Module):
    """KL divergence between multivariate Gaussian fields with diagonal covariance.

    Each voxel holds a ``d``-dimensional Gaussian; ``KL(P || Q)`` is computed per voxel and averaged over batch and
    voxels. The constructor takes no arguments.
    """

    def __init__(
        self,
    ):
        super().__init__()

    @_amp.fp32_forward
    def forward(self, P, Q) -> torch.Tensor:
        """Compute multivariate KL divergence between Gaussian fields.

        ``log_sigma`` is a log standard deviation, as in :class:`KL_divergence`: the covariance is diagonal with
        variances ``exp(2 * log_sigma)``.

        Args:
            P: Tuple (mean_p, log_sigma_p) of tensors shaped (B, d, H, W, D).
            Q: Tuple (mean_q, log_sigma_q) of tensors with the same shape.

        Returns:
            Scalar ``KL(P || Q)`` averaged over batch and voxels.

        """
        mean_p, log_sigma_p = P
        mean_q, log_sigma_q = Q
        b, d, _, _, _ = mean_p.shape
        mean_p = mean_p.view(b, d, -1)
        mean_p = mean_p.permute(0, 2, 1).view(-1, d)
        log_sigma_p = log_sigma_p.view(b, d, -1)
        log_sigma_p = log_sigma_p.permute(0, 2, 1).view(-1, d)

        mean_q = mean_q.view(b, d, -1)
        mean_q = mean_q.permute(0, 2, 1).view(-1, d)
        log_sigma_q = log_sigma_q.view(b, d, -1)
        log_sigma_q = log_sigma_q.permute(0, 2, 1).view(-1, d)

        q_var = torch.diag_embed(torch.exp(2 * log_sigma_q))  # log_sigma is a log standard deviation
        p_var = torch.diag_embed(torch.exp(2 * log_sigma_p))

        p = torch.distributions.MultivariateNormal(mean_p, p_var)
        q = torch.distributions.MultivariateNormal(mean_q, q_var)

        kl_loss = torch.distributions.kl_divergence(p, q).mean()
        return kl_loss
