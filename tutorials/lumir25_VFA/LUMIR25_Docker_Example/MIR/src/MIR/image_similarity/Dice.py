'''
Dice loss function
Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''

import torch
import torch.nn as nn

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