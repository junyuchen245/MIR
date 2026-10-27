"""Dice loss and helper utilities for segmentation evaluation."""

import random
from typing import Callable, Optional

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentations.

    Dice is computed per sample and per channel over the three spatial dims, then ``1 - mean(Dice)`` over batch and
    channels is returned (background channel included). No smoothing term is added to the numerator; ``1e-5`` is
    added to the denominator.

    Args:
        num_class: Number of classes used for one-hot encoding ``y_true`` (only used when ``one_hot=True``).
        one_hot: If True, ``y_true`` is an integer label map that is one-hot encoded internally; if False, ``y_true``
            must already have the same shape as ``y_pred``.
    """

    def __init__(self, num_class: int = 36, one_hot: bool = True):
        super().__init__()
        self.num_class = num_class
        self.one_hot = one_hot

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            y_pred: Predicted probabilities or soft masks (B, C, H, W, D).
            y_true: Integer labels (B, 1, H, W, D) or (B, H, W, D) (int64) when ``one_hot=True``; otherwise a
                (B, C, H, W, D) one-hot/soft tensor.

        Returns:
            Scalar Dice loss ``1 - mean(Dice)`` to minimise.

        """
        # y_pred = torch.round(y_pred)
        # y_pred = nn.functional.one_hot(torch.round(y_pred).long(), num_classes=7)
        # y_pred = torch.squeeze(y_pred, 1)
        # y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        if self.one_hot:
            y_true = nn.functional.one_hot(y_true, num_classes=self.num_class)
            y_true = torch.squeeze(y_true, 1)
            y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
        # y_true = nn.functional.one_hot(y_true, num_classes=self.num_class)
        # y_true = torch.squeeze(y_true, 1)
        # y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
        intersection = y_pred * y_true
        intersection = intersection.sum(dim=[2, 3, 4])
        union = torch.pow(y_pred, 1).sum(dim=[2, 3, 4]) + torch.pow(y_true, 1).sum(dim=[2, 3, 4])
        dsc = (2.0 * intersection) / (union + 1e-5)
        dsc = 1 - torch.mean(dsc)
        return dsc


class SparseDiceLoss(nn.Module):
    """Sampled multi-class Dice over integer labels without dense full one-hot.

    Module wrapper around :func:`sparse_dice_from_int_labels`; a new random subset of classes is drawn at every call.

    Args:
        num_classes: Total number of label classes.
        K: Number of classes sampled per call.
        include_bg: If True, class 0 may be sampled.
        present_only: If True, sample only from classes present in the target label map.
        dice_loss: Dice module applied to the K-channel masks; defaults to ``DiceLoss(num_class=num_classes,
            one_hot=False)``.
    """

    def __init__(
        self,
        num_classes: int,
        K: int = 16,
        include_bg: bool = False,
        present_only: bool = True,
        dice_loss: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.K = int(K)
        self.include_bg = bool(include_bg)
        self.present_only = bool(present_only)
        self.dice_loss = dice_loss if dice_loss is not None else DiceLoss(num_class=num_classes, one_hot=False)

    def forward(
        self,
        src_lbl_int: torch.Tensor,
        tgt_lbl_int: torch.Tensor,
        flow_src2tgt: torch.Tensor,
        warp_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Compute the sampled Dice loss between warped source labels and target labels.

        Args:
            src_lbl_int: Source (moving) integer label map (B, 1, H, W, D) or (B, H, W, D).
            tgt_lbl_int: Target (fixed) integer label map with the same layout.
            flow_src2tgt: Flow that warps source to target space, e.g. (B, 3, H, W, D).
            warp_fn: Callable ``warp_fn(tensor, flow)`` returning the warped tensor (e.g. a spatial transformer).

        Returns:
            Scalar Dice loss over the sampled classes.
        """
        return sparse_dice_from_int_labels(
            src_lbl_int=src_lbl_int,
            tgt_lbl_int=tgt_lbl_int,
            flow_src2tgt=flow_src2tgt,
            warp_fn=warp_fn,
            num_classes=self.num_classes,
            K=self.K,
            include_bg=self.include_bg,
            present_only=self.present_only,
            dice_loss=self.dice_loss,
        )


@torch.no_grad()
def sample_label_indices(
    num_classes: int, K: int, device: torch.device, present_only: bool, y_true_int: torch.Tensor, include_bg: bool
) -> torch.Tensor:
    """Sample K class indices to evaluate.

    Sampling uses Python's global ``random`` module (without replacement). If the candidate pool has fewer than K
    classes, the whole pool is returned (in pool order) padded with classes from ``range(num_classes)`` that are not in
    the pool, so the result may contain fewer than K indices.

    Args:
        num_classes: Total number of classes.
        K: Number of class indices to sample.
        device: Torch device for the returned tensor.
        present_only: If True, sample from labels present in y_true (falls back to all classes if none are present).
        y_true_int: Integer label map (B, 1, D, H, W) or (B, D, H, W).
        include_bg: If False, excludes background class 0.

    Returns:
        Long tensor of sampled class indices (at most K entries).

    Raises:
        ValueError: If ``K <= 0`` or no valid class is available.

    """
    if K <= 0:
        raise ValueError("K must be positive.")

    all_classes = list(range(num_classes))
    if not include_bg and 0 in all_classes:
        all_classes.remove(0)

    if present_only:
        lbl = y_true_int if y_true_int.dim() == 4 else y_true_int.squeeze(1)
        present = torch.unique(lbl).tolist()
        if not include_bg and 0 in present:
            present.remove(0)
        pool = present if present else list(all_classes)
    else:
        pool = list(all_classes)

    if not pool:
        raise ValueError("No valid label classes available for sparse Dice sampling.")

    # sample without replacement; if pool < K, pad arbitrarily
    if len(pool) >= K:
        idx = random.sample(pool, K)
    else:
        extra = [c for c in all_classes if c not in pool]
        idx = pool + extra[: max(0, K - len(pool))]
    return torch.tensor(idx, device=device, dtype=torch.long)


def build_k_hot_from_int(y_int: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Build a K-hot tensor for selected class indices.

    Args:
        y_int: Integer label map (B, 1, D, H, W) or (B, D, H, W).
        idx: Class indices (K,).

    Returns:
        K-hot tensor (B, K, D, H, W) with values in {0,1}.

    Raises:
        ValueError: If ``y_int`` is neither (B, 1, D, H, W) nor (B, D, H, W).

    """
    if y_int.dim() == 5 and y_int.size(1) == 1:
        lbl = y_int.squeeze(1).long()  # [B,D,H,W]
    elif y_int.dim() == 4:
        lbl = y_int.long()  # [B,D,H,W]
    else:
        raise ValueError("y_int must be [B,1,D,H,W] or [B,D,H,W].")
    return (lbl.unsqueeze(1) == idx.view(1, -1, 1, 1, 1)).float()  # [B,K,D,H,W]


def sparse_dice_from_int_labels(
    src_lbl_int: torch.Tensor,
    tgt_lbl_int: torch.Tensor,
    flow_src2tgt: torch.Tensor,
    warp_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],  # e.g., model.spatial_trans(tensor, flow)
    num_classes: int,
    K: int = 16,
    include_bg: bool = False,
    present_only: bool = True,
    dice_loss: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Compute Dice on K sampled classes without full one-hot.

    Classes are sampled from the target labels (see :func:`sample_label_indices`). The source K-hot masks are built
    before warping so the loss is differentiable w.r.t. the flow; the warped masks are clamped to [0, 1] and the
    target masks are detached.

    Args:
        src_lbl_int: Source integer label map [B,1,D,H,W] or [B,D,H,W].
        tgt_lbl_int: Target integer label map [B,1,D,H,W] or [B,D,H,W].
        flow_src2tgt: Flow that warps source -> target space.
        warp_fn: Callable ``warp_fn(tensor[B,K,D,H,W], flow)`` returning the warped tensor.
        num_classes: Total number of label classes.
        K: Number of classes to sample.
        include_bg: If True, class 0 may be sampled.
        present_only: If True, sample only from classes present in ``tgt_lbl_int``.
        dice_loss: Dice module applied to (warped source, target) K-channel masks; defaults to
            ``DiceLoss(one_hot=False)``.

    Returns:
        Scalar loss.
    """
    device = src_lbl_int.device
    idx = sample_label_indices(num_classes, K, device, present_only, tgt_lbl_int, include_bg)
    # Build K binary masks BEFORE warp for differentiability w.r.t. flow
    src_K = build_k_hot_from_int(src_lbl_int, idx)  # [B,K,D,H,W]
    tgt_K = build_k_hot_from_int(tgt_lbl_int, idx).detach()  # target can be treated as constant

    # Warp source masks with trilinear grid_sample inside warp_fn
    src_K_warped = warp_fn(src_K, flow_src2tgt).clamp_(0, 1)  # [B,K,D,H,W], soft masks in [0,1]

    if dice_loss is None:
        dice_loss = DiceLoss(one_hot=False)  # we already have K channels
    return dice_loss(src_K_warped, tgt_K)
