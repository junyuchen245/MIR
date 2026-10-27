"""MIND-SSC descriptor and loss for image similarity."""

from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


class MIND_loss(torch.nn.Module):
    """MIND-SSC (modality independent neighbourhood descriptor, self-similarity context) loss.

    Computes 12-channel MIND-SSC descriptors of both images (Heinrich et al., MICCAI 2013) and returns the mean squared
    difference between them. Only 3D (5D tensor) single-channel inputs are supported.

    Args:
        win: Stored as ``self.win`` but not used by the loss (kept for API compatibility).
    """

    def __init__(self, win: Optional[int] = None):
        super().__init__()
        self.win = win

    def pdist_squared(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pairwise squared Euclidean distances between the columns of ``x``.

        NaNs are set to 0 and negative values (round-off) are clamped to 0.

        Args:
            x: Tensor (B, D, N) holding N points of dimension D.

        Returns:
            Pairwise squared distances (B, N, N).

        """
        xx = (x**2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img: torch.Tensor, radius: int = 2, dilation: int = 2) -> torch.Tensor:
        """Compute the MIND-SSC descriptor.

        Args:
            img: Image tensor (B, 1, H, W, D).
            radius: Patch radius; patch SSDs are averaged over a ``2 * radius + 1`` cube.
            dilation: Dilation of the six-neighbourhood sampling pattern.

        Returns:
            MIND-SSC descriptor tensor (B, 12, H, W, D) with values in ``(0, 1]``.

        """
        # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

        # kernel size
        kernel_size = radius * 2 + 1

        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.Tensor([[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1]]).long()

        # squared distances
        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = (x > y).view(-1) & (dist == 2).view(-1)

        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3, device=img.device)
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3, device=img.device)
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        rpad1 = nn.ReplicationPad3d(dilation)
        rpad2 = nn.ReplicationPad3d(radius)

        # compute patch-ssd
        ssd = F.avg_pool3d(
            rpad2(
                (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation))
                ** 2
            ),
            kernel_size,
            stride=1,
        )

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
        mind = mind / mind_var
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

        return mind

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute MIND-SSC L2 loss between two images.

        Note the argument order ``(y_pred, y_true)``; the loss is symmetric, so the order does not change the value.
        Descriptors use the default ``radius=2`` and ``dilation=2``.

        Args:
            y_pred: Moving (warped) image tensor (B, 1, H, W, D).
            y_true: Fixed image tensor (B, 1, H, W, D).

        Returns:
            Scalar loss: mean squared descriptor difference over all batch, channel and voxel elements.

        """
        return torch.mean((self.MINDSSC(y_pred) - self.MINDSSC(y_true)) ** 2)
