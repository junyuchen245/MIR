"""Segmentation overlap losses for multi-class outputs."""

from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn


class DiceSegLoss(nn.Module):
    """Soft multi-class Dice loss computed from logits and integer labels.

    Softmax probabilities are compared with the one-hot encoded target. Intersection and union are summed over the
    batch and spatial dims (i.e. one Dice score per class for the whole batch); the loss is ``1 - mean_c(Dice_c)``.
    Only 3D (5D tensor) inputs are supported.

    Args:
        num_classes: Number of classes ``C``; target labels must lie in ``[0, C-1]``.
        ignore_index: Optional label value whose voxels are excluded from the Dice computation. Because the target
            is one-hot encoded with ``num_classes`` classes, this value must still lie in ``[0, C-1]``.
        eps: Smoothing constant added to numerator and denominator.
    """

    def __init__(self, num_classes: int, ignore_index: Optional[int] = None, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the soft Dice loss.

        Args:
            logits: Raw class scores ``[B, C, H, W, D]`` (softmax is applied internally).
            target: Integer labels ``[B, H, W, D]`` or ``[B, 1, H, W, D]`` with values in ``[0, C-1]`` (int64, as
                required by ``F.one_hot``).

        Returns:
            Scalar loss ``1 - mean_c(Dice_c)``.
        """
        # convert (B,1,ph,pw,pd) → (B,ph,pw,pd) if needed
        if target.dim() == logits.dim():
            target = target[:, 0]

        # softmax → (B,C,ph,pw,pd)
        probs = F.softmax(logits, dim=1)

        # one-hot encode target → (B,ph,pw,pd,C) → (B,C,ph,pw,pd)
        safe_target = target if self.ignore_index is None else target.masked_fill(target == self.ignore_index, 0)
        one_hot = F.one_hot(safe_target, self.num_classes).permute(0, 4, 1, 2, 3).float()  # ignored voxels masked below

        # mask out ignore_index voxels
        if self.ignore_index is not None:
            valid = (target != self.ignore_index).unsqueeze(1)  # (B,1,ph,pw,pd)
            probs = probs * valid
            one_hot = one_hot * valid

        # compute per-class dice
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * one_hot, dim=dims)
        union = torch.sum(probs + one_hot, dim=dims)
        dice_score = (2 * intersection + self.eps) / (union + self.eps)

        # average classes and return 1 − dice
        return 1 - dice_score.mean()


# 2) Combined cross‐entropy + Dice
class SegmentationLoss(nn.Module):
    """Weighted sum of cross-entropy and soft Dice loss for multi-class segmentation.

    Returns ``weight_ce * CE + weight_dice * DiceSegLoss``. The Dice term uses the default ``eps`` of
    :class:`DiceSegLoss`.

    Args:
        num_classes: Number of classes ``C``.
        weight_ce: Weight of the cross-entropy term (``nn.CrossEntropyLoss``, mean reduction).
        weight_dice: Weight of the Dice term.
        ignore_index: Optional label ignored by both terms.
    """

    def __init__(
        self, num_classes: int, weight_ce: float = 1.0, weight_dice: float = 1.0, ignore_index: Optional[int] = None
    ):
        super().__init__()
        if ignore_index is not None:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            self.ce = nn.CrossEntropyLoss()
        self.dice = DiceSegLoss(num_classes, ignore_index)
        self.w_ce = weight_ce
        self.w_dice = weight_dice

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the combined cross-entropy + Dice loss.

        Args:
            logits: Raw class scores ``[B, C, H, W, D]``.
            target: Integer labels ``[B, H, W, D]`` with values in ``[0, C-1]``. (``DiceSegLoss`` also accepts
                ``[B, 1, H, W, D]``, but ``nn.CrossEntropyLoss`` requires the channel-free shape.)

        Returns:
            Scalar loss tensor.
        """
        loss_ce = self.ce(logits, target)
        loss_dice = self.dice(logits, target)
        return self.w_ce * loss_ce + self.w_dice * loss_dice
