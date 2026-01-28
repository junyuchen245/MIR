"""Accuracy and overlap metrics for MIR evaluations."""

import math, random
import numpy as np
import torch.nn.functional as F
import torch, sys
from torch import nn

def dice_val_VOI(y_pred, y_true, num_clus=4, eval_labels=None):
    """Compute mean Dice over a set of labels of interest.

    Args:
        y_pred: Predicted label tensor (B, 1, H, W, D).
        y_true: Ground-truth label tensor (B, 1, H, W, D).
        num_clus: Number of classes (used when eval_labels is None).
        eval_labels: Optional list/array of label IDs to evaluate.

    Returns:
        Mean Dice score across the selected labels.
    """
    if eval_labels is not None:
        VOI_lbls = eval_labels
    else:
        VOI_lbls = np.arange(1, num_clus)
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    DSCs = np.zeros((len(VOI_lbls), 1))
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i
        true_i = true == i
        intersection = pred_i.astype(np.float32) * true_i.astype(np.float32)
        intersection = np.sum(intersection)
        union = np.sum(pred_i) + np.sum(true_i)
        dsc = (2.*intersection) / (union + 1e-5)
        DSCs[idx] =dsc
        idx += 1
    return np.mean(DSCs)

def dice_val_substruct(y_pred, y_true, std_idx, num_classes=46):
    """Compute per-class Dice scores and return as a CSV line string.

    Args:
        y_pred: Predicted label tensor (B, 1, H, W, D).
        y_true: Ground-truth label tensor (B, 1, H, W, D).
        std_idx: Case identifier used in the output line.
        num_classes: Total number of classes.

    Returns:
        CSV line string with per-class Dice values.
    """
    with torch.no_grad():
        y_pred = nn.functional.one_hot(y_pred, num_classes=num_classes)
        y_pred = torch.squeeze(y_pred, 1)
        y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        y_true = nn.functional.one_hot(y_true, num_classes=num_classes)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    line = 'p_{}'.format(std_idx)
    for i in range(num_classes):
        pred_clus = y_pred[0, i, ...]
        true_clus = y_true[0, i, ...]
        intersection = pred_clus * true_clus
        intersection = intersection.sum()
        union = pred_clus.sum() + true_clus.sum()
        dsc = (2.*intersection) / (union + 1e-5)
        line = line+','+str(dsc)
    return line

def dice_val_all(y_pred, y_true, num_clus):
    """Compute mean Dice across all classes.

    Args:
        y_pred: Predicted label tensor (B, 1, H, W, D).
        y_true: Ground-truth label tensor (B, 1, H, W, D).
        num_clus: Number of classes.

    Returns:
        Scalar mean Dice across classes.
    """
    y_pred = nn.functional.one_hot(y_pred, num_classes=num_clus)
    y_pred = torch.squeeze(y_pred, 1)
    y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_clus)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2.*intersection) / (union + 1e-5)
    return torch.mean(torch.mean(dsc, dim=1))