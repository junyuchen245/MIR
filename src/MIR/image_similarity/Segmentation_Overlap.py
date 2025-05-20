import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import torch.nn as nn

class DiceSegLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = None, eps: float = 1e-6):
        super().__init__()
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.eps           = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, ph, pw, pd)  raw scores
        target: (B, ph, pw, pd)     integer labels in [0..C-1]
        """
        # convert (B,1,ph,pw,pd) → (B,ph,pw,pd) if needed
        if target.dim() == logits.dim():
            target = target[:,0]

        # softmax → (B,C,ph,pw,pd)
        probs = F.softmax(logits, dim=1)

        # one-hot encode target → (B,ph,pw,pd,C) → (B,C,ph,pw,pd)
        one_hot = F.one_hot(target, self.num_classes) \
                    .permute(0,4,1,2,3).float()

        # mask out ignore_index voxels
        if self.ignore_index is not None:
            valid = (target != self.ignore_index).unsqueeze(1)  # (B,1,ph,pw,pd)
            probs   = probs   * valid
            one_hot = one_hot * valid

        # compute per-class dice
        dims = (0,2,3,4)
        intersection = torch.sum(probs * one_hot, dim=dims)
        union        = torch.sum(probs + one_hot,   dim=dims)
        dice_score   = (2 * intersection + self.eps) / (union + self.eps)

        # average classes and return 1 − dice
        return 1 - dice_score.mean()


# 2) Combined cross‐entropy + Dice
class SegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        ignore_index: int = None
    ):
        super().__init__()
        if ignore_index is not None:
            self.ce   = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceSegLoss(num_classes, ignore_index)
        self.w_ce   = weight_ce
        self.w_dice = weight_dice

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, ph, pw, pd)
        target: (B, ph, pw, pd)  ints in [0..C-1]
        """
        loss_ce   = self.ce(logits, target)
        loss_dice = self.dice(logits, target)
        return self.w_ce * loss_ce + self.w_dice * loss_dice