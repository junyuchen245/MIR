"""KL divergence losses for Gaussian parameter fields."""

import torch
import torch.nn as nn

class KL_divergence(nn.Module):
    """KL divergence between factorized Gaussian fields."""
    def __init__(self,):
        super().__init__()
    def forward(self, P, Q):
        """Compute KL divergence between two Gaussian fields.

        Args:
            P: Tuple (mean_p, log_sigma_p) tensors.
            Q: Tuple (mean_q, log_sigma_q) tensors.

        Returns:
            Scalar KL divergence.
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
    """KL divergence between multivariate Gaussian fields."""
    def __init__(self,):
        super().__init__()
    def forward(self, P, Q):
        """Compute multivariate KL divergence between Gaussian fields.

        Args:
            P: Tuple (mean_p, log_sigma_p) tensors.
            Q: Tuple (mean_q, log_sigma_q) tensors.

        Returns:
            Scalar KL divergence.
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

        q_var = torch.diag_embed(torch.exp(log_sigma_q))
        p_var = torch.diag_embed(torch.exp(log_sigma_p))

        p = torch.distributions.MultivariateNormal(mean_p, p_var)
        q = torch.distributions.MultivariateNormal(mean_q, q_var)

        kl_loss = torch.distributions.kl_divergence(p, q).mean()
        return kl_loss