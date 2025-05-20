'''
Pearson Correlation Coefficient (PCC) loss function
Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import torch.nn as nn

class PCC(torch.nn.Module):
    def __init__(self,):
        super(PCC, self).__init__()

    def pcc(self, y_true, y_pred):
        A_bar = torch.mean(y_pred, dim=[1, 2, 3, 4], keepdim=True)
        B_bar = torch.mean(y_true, dim=[1, 2, 3, 4], keepdim=True)
        top = torch.mean((y_pred - A_bar) * (y_true - B_bar), dim=[1, 2, 3, 4], keepdim=True)
        bottom = torch.sqrt(torch.mean((y_pred - A_bar) ** 2, dim=[1, 2, 3, 4], keepdim=True) * torch.mean((y_true - B_bar) ** 2, dim=[1, 2, 3, 4], keepdim=True))
        return torch.mean(top/bottom)

    def forward(self, I, J):
        return (1-self.pcc(I,J))