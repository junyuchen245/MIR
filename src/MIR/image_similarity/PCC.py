"""Pearson correlation coefficient (PCC) loss."""

import torch
from MIR import _amp


class PCC(torch.nn.Module):
    """Global Pearson correlation coefficient (PCC) loss.

    Computes the PCC over all channels and voxels of each sample and returns ``1 - PCC`` averaged over the batch.
    The reductions are hard-coded over dims ``[1, 2, 3, 4]``, so inputs must be 5D (``[B, C, H, W, D]``).
    The constructor takes no arguments.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def pcc(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Compute the Pearson correlation coefficient.

        The PCC is computed per sample over dims ``[1, 2, 3, 4]`` and then averaged over the batch. No epsilon is
        added to the denominator, so a constant image yields NaN/inf.

        Args:
            y_true: Fixed image tensor ``[B, 1, H, W, D]``.
            y_pred: Moving (warped) image tensor ``[B, 1, H, W, D]``.

        Returns:
            Scalar tensor with the batch-mean PCC (a similarity in ``[-1, 1]``).

        """
        A_bar = torch.mean(y_pred, dim=[1, 2, 3, 4], keepdim=True)
        B_bar = torch.mean(y_true, dim=[1, 2, 3, 4], keepdim=True)
        top = torch.mean((y_pred - A_bar) * (y_true - B_bar), dim=[1, 2, 3, 4], keepdim=True)
        bottom = torch.sqrt(
            torch.mean((y_pred - A_bar) ** 2, dim=[1, 2, 3, 4], keepdim=True)
            * torch.mean((y_true - B_bar) ** 2, dim=[1, 2, 3, 4], keepdim=True)
        )
        return torch.mean(top / bottom)

    @_amp.fp32_forward
    def forward(self, I: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """Return ``1 - PCC`` as a loss to minimise.

        Args:
            I: Fixed image tensor ``[B, 1, H, W, D]`` (passed as ``y_true`` to :meth:`pcc`).
            J: Moving (warped) image tensor ``[B, 1, H, W, D]`` (passed as ``y_pred``).

        Returns:
            Scalar loss tensor ``1 - mean_b(PCC_b)``.

        """
        return 1 - self.pcc(I, J)
