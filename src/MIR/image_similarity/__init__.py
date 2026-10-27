"""Image similarity losses and metrics used in MIR registration."""

from .Dice import DiceLoss, SparseDiceLoss, sample_label_indices, build_k_hot_from_int, sparse_dice_from_int_labels
from .MIND_SSC import MIND_loss
from .Mutual_Information import (
    MutualInformation,
    localMutualInformation,
    MattesMutualInformation,
    NormalizedMutualInformation,
)
from .NCC import NCC_gauss, NCC, NCC_mok, NCC_mok2, NCC_vxm, NCC_vfa, FastNCC, NCC_fp16
from .PCC import PCC

# NCC.py and SSIM.py define identically named window helpers; the package-level names are the SSIM ones
# (as before). The NCC versions remain available from MIR.image_similarity.NCC.
from .SSIM import gaussian, create_window, create_window_3D, SSIM2D, SSIM3D, ssim, ssim3D
from .Correlation_Ratio import LocalCorrRatio, CorrRatio
from .CompositeRecon import CompRecon
from .Segmentation_Overlap import SegmentationLoss, DiceSegLoss
from .NGF import NormalizedGradientFieldLoss

__all__ = [
    'DiceLoss',
    'SparseDiceLoss',
    'sample_label_indices',
    'build_k_hot_from_int',
    'sparse_dice_from_int_labels',
    'MIND_loss',
    'MutualInformation',
    'localMutualInformation',
    'MattesMutualInformation',
    'NormalizedMutualInformation',
    'gaussian',
    'create_window',
    'create_window_3D',
    'NCC_gauss',
    'NCC',
    'NCC_mok',
    'NCC_mok2',
    'NCC_vxm',
    'NCC_vfa',
    'NCC_fp16',
    'FastNCC',
    'PCC',
    'gaussian',
    'create_window',
    'create_window_3D',
    'SSIM2D',
    'SSIM3D',
    'ssim',
    'ssim3D',
    'LocalCorrRatio',
    'CorrRatio',
    'CompRecon',
    'SegmentationLoss',
    'DiceSegLoss',
    'NormalizedGradientFieldLoss',
]
