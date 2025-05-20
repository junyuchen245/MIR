"""Auto-generated on 2025-05-01 – edit as needed."""
from .data_augmentation import flip_aug, affine_aug, RandomMultiContrastRemap, fill_holes_torch
from . import accuracy_measures
from . import deformation_regularizer
from . import image_similarity
from . import models
from . import utils
from . import random_image_generation
from . import intensity_normalization

__all__ = [
    'flip_aug',
    'affine_aug',
    'accuracy_measures',
    'deformation_regularizer',
    'image_similarity',
    'models',
    'utils',
    'random_image_generation',
    'intensity_normalization',
    'RandomMultiContrastRemap',
    'fill_holes_torch',
]

# ───────────────────────────────────────
# PyTorch‑version check
# ───────────────────────────────────────
import re
import torch
import warnings

_ver = torch.__version__        # e.g. '2.1.0+cu121', '1.13.1a0', …

m = re.match(r"(\d+)\.(\d+)", _ver)
if m:
    major, minor = map(int, m.groups())
    if (major, minor) >= (2, 0):
        warnings.warn(                         # or `print(...)` if you prefer
            "PyTorch ≥ 2.0 detected.  If you notice numerical differences, set\n"
            "torch.backends.cudnn.allow_tf32 = False\n"
            "before running your model.\n"
            "Your torch version is {}\n".format(_ver),
            UserWarning,
            stacklevel=2,
        )
else:
    warnings.warn(f"Could not parse PyTorch version string: '{_ver}'")