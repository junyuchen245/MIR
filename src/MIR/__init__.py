"""Top-level MIR package for medical image registration and utilities."""

try:  # single source of truth: the version in pyproject.toml, read from the installed distribution
    from importlib.metadata import PackageNotFoundError, version as _dist_version

    __version__ = _dist_version("MIR")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0+unknown"

from .data_augmentation import flip_aug, affine_aug, RandomMultiContrastRemap, fill_holes_torch
from . import accuracy_measures
from . import deformation_regularizer
from . import image_similarity
from . import models
from . import utils
from . import random_image_generation
from . import intensity_normalization
from . import statistical_analysis
from . import l2r_io
from .pretrained_wts import ModelWeights, DatasetJSONs, ValEvalModules
from . import label_reference

__all__ = [
    '__version__',
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
    'ModelWeights',
    'DatasetJSONs',
    'ValEvalModules',
    'fill_holes_torch',
    'label_reference',
    'statistical_analysis',
    'l2r_io',
]

# ───────────────────────────────────────
# PyTorch‑version check
# ───────────────────────────────────────
import re
import torch
import warnings

_ver = str(
    getattr(torch, "__version__", "")
)  # e.g. '2.1.0+cu121', '1.13.1a0', … (str() keeps doc builds with a mocked torch working)

m = re.match(r"(\d+)\.(\d+)", _ver)
if m:
    major, minor = map(int, m.groups())
    if (major, minor) >= (2, 0):
        warnings.warn(  # or `print(...)` if you prefer
            "PyTorch ≥ 2.0 detected.  If you notice numerical differences, set\n"
            "torch.backends.cudnn.allow_tf32 = False\n"
            "before running your model.\n"
            f"Your torch version is {_ver}\n",
            UserWarning,
            stacklevel=2,
        )
else:
    warnings.warn(f"Could not parse PyTorch version string: '{_ver}'", stacklevel=2)
