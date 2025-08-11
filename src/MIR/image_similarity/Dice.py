'''
Dice loss function
Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''

import torch
import torch.nn as nn
import random

class DiceLoss(nn.Module):
    """Dice loss"""
    def __init__(self, num_class=36, one_hot=True):
        super().__init__()
        self.num_class = num_class
        self.one_hot = one_hot

    def forward(self, y_pred, y_true):
        #y_pred = torch.round(y_pred)
        #y_pred = nn.functional.one_hot(torch.round(y_pred).long(), num_classes=7)
        #y_pred = torch.squeeze(y_pred, 1)
        #y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        if self.one_hot:
            y_true = nn.functional.one_hot(y_true, num_classes=self.num_class)
            y_true = torch.squeeze(y_true, 1)
            y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
        #y_true = nn.functional.one_hot(y_true, num_classes=self.num_class)
        #y_true = torch.squeeze(y_true, 1)
        #y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
        intersection = y_pred * y_true
        intersection = intersection.sum(dim=[2, 3, 4])
        union = torch.pow(y_pred, 1).sum(dim=[2, 3, 4]) + torch.pow(y_true, 1).sum(dim=[2, 3, 4])
        dsc = (2.*intersection) / (union + 1e-5)
        dsc = (1-torch.mean(dsc))
        return dsc

@torch.no_grad()
def sample_label_indices(num_classes: int,
                         K: int,
                         device: torch.device,
                         present_only: bool,
                         y_true_int: torch.Tensor,
                         include_bg: bool) -> torch.Tensor:
    """
    Return K class indices to evaluate. No grad needed for indices.
    y_true_int: [B,1,D,H,W] or [B,D,H,W]
    """
    if present_only:
        lbl = y_true_int if y_true_int.dim() == 4 else y_true_int.squeeze(1)
        present = torch.unique(lbl).tolist()
        if not include_bg and 0 in present:
            present.remove(0)
        pool = present if present else list(range(num_classes))
    else:
        pool = list(range(num_classes))
        if not include_bg and 0 in pool:
            pool.remove(0)

    # sample without replacement; if pool < K, pad arbitrarily
    if len(pool) >= K:
        idx = random.sample(pool, K)
    else:
        extra = [c for c in range(num_classes) if (include_bg or c != 0) and c not in pool]
        idx = pool + extra[:max(0, K - len(pool))]
    return torch.tensor(idx, device=device, dtype=torch.long)


def build_k_hot_from_int(y_int: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Vectorized K-hot from integer label map.
    y_int: [B,1,D,H,W] or [B,D,H,W] ints
    idx:   [K] class ids
    Returns: [B,K,D,H,W] float in {0,1}
    """
    if y_int.dim() == 5 and y_int.size(1) == 1:
        lbl = y_int.squeeze(1).long()   # [B,D,H,W]
    elif y_int.dim() == 4:
        lbl = y_int.long()              # [B,D,H,W]
    else:
        raise ValueError("y_int must be [B,1,D,H,W] or [B,D,H,W].")
    return (lbl.unsqueeze(1) == idx.view(1, -1, 1, 1, 1)).float()  # [B,K,D,H,W]


def sparse_dice_from_int_labels(src_lbl_int: torch.Tensor,
                                tgt_lbl_int: torch.Tensor,
                                flow_src2tgt: torch.Tensor,
                                warp_fn,            # e.g., model.spatial_trans(tensor, flow)
                                num_classes: int,
                                K: int = 16,
                                include_bg: bool = False,
                                present_only: bool = True,
                                dice_loss: nn.Module = None) -> torch.Tensor:
    """
    Compute Dice on K sampled classes without full one-hot.
    src_lbl_int, tgt_lbl_int: integer maps [B,1,D,H,W] or [B,D,H,W]
    flow_src2tgt: flow that warps source -> target space
    warp_fn: callable(tensor[B,K,D,H,W], flow) -> warped tensor
    Returns scalar loss.
    """
    device = src_lbl_int.device
    idx = sample_label_indices(num_classes, K, device, present_only, tgt_lbl_int, include_bg)
    # Build K binary masks BEFORE warp for differentiability w.r.t. flow
    src_K = build_k_hot_from_int(src_lbl_int, idx)     # [B,K,D,H,W]
    tgt_K = build_k_hot_from_int(tgt_lbl_int, idx).detach()  # target can be treated as constant

    # Warp source masks with trilinear grid_sample inside warp_fn
    src_K_warped = warp_fn(src_K, flow_src2tgt).clamp_(0, 1)  # [B,K,D,H,W], soft masks in [0,1]

    if dice_loss is None:
        dice_loss = DiceLoss(one_hot=False)  # we already have K channels
    return dice_loss(src_K_warped, tgt_K)