'''
Global regularizers for deformation regularization. Mainly for Spatially Varying Regularization.
Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''

import torch

class logBeta(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(logBeta, self).__init__()
        self.eps = eps
        self.beta = 1.

    def forward(self, weights, alpha):
        lambdas = torch.clamp(weights, self.eps, 1.0)
        #beta = torch.log(lambdas**(alpha-1)*(1-lambdas)**(self.beta-1))
        beta = torch.log(lambdas)
        return (1.-alpha)*beta.mean()

class logGaussian(torch.nn.Module):
    def __init__(self, gaus_bond=5., eps=1e-6):
        super(logGaussian, self).__init__()
        self.eps = eps
        self.gaus_bond = gaus_bond

    def forward(self, weights, inv_sigma2):
        weights = torch.clamp(weights, self.eps, self.gaus_bond)
        return inv_sigma2*torch.mean((weights-1.)**2)

class LocalGrad3d(torch.nn.Module):
    """
    Local 3D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(LocalGrad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, weight):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy.pow(2)
            dx = dx.pow(2)
            dz = dz.pow(2)
        d = torch.mean(dx*weight[:, :, :, 1:, :])+torch.mean(dy*weight[:, :, 1:, :, :])+torch.mean(dz*weight[:, :, :, :, 1:])
        grad = d / 3.0
        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad
