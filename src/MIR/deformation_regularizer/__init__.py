"""Regularization terms for deformation fields and velocity models."""
from .GlobalRegularizers import Grad2D, Grad3d, Grad3DiTV, DisplacementRegularizer, GradICON3d, GradICONExact3d
from .LocalRegularizers import logBeta, logGaussian, LocalGrad3d
from .KL_divergence import KL_divergence, MultiVariateKL_divergence

__all__ = [
    'Grad2D',
    'Grad3d',
    'Grad3DiTV',
    'DisplacementRegularizer',
    'logBeta',
    'logGaussian',
    'LocalGrad3d',
    'KL_divergence',
    'MultiVariateKL_divergence',
    'GradICON3d',
    'GradICONExact3d',
]
