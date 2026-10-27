"""Training utilities (logging, metrics, and patch sampling)."""

from typing import List, Sequence, Tuple, Union

import numpy as np
import torch.nn.functional as F
import torch
import sys


class Logger:
    """File-like object that writes to stdout and appends to a log file (typically assigned to ``sys.stdout``).

    Args:
        save_dir: Directory path for the log file, including the trailing separator; the file
            ``save_dir + "logfile.log"`` is opened in append mode.
    """

    def __init__(self, save_dir: str):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message: str) -> None:
        """Write a message to both stdout and the log file."""
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        """No-op flush for compatibility with file-like interfaces."""


class AverageMeter:
    """Computes and stores the current value, the weighted running average and the std of all values seen."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset stored statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update running statistics.

        ``avg`` is the ``n``-weighted mean; ``std`` is the unweighted ``np.std`` of all values passed so far.

        Args:
            val: New value.
            n: Weight/count for the value.

        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)


def pad_image(img: torch.Tensor, target_size: Sequence[int]) -> torch.Tensor:
    """Zero-pad a 3D tensor at the end of each spatial axis up to a target size.

    Axes that are already at least as large as the target are left unchanged (no cropping).

    Args:
        img: Input tensor (B, C, H, W, D).
        target_size: Target spatial size (H, W, D).

    Returns:
        Padded tensor.

    """
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    slcs_to_pad = max(target_size[2] - img.shape[4], 0)
    padded_img = F.pad(img, (0, slcs_to_pad, 0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img


def normalize_01(
    x: Union[torch.Tensor, np.ndarray], method: str = 'minmax', percentile: float = 0.05
) -> Union[torch.Tensor, np.ndarray]:
    """Normalize a tensor/array to the range [0, 1].

    With ``'percentile'`` the values at the ``percentile`` and ``1 - percentile`` quantiles are mapped to 0 and 1;
    the result is clipped to [0, 1] in both modes. Inputs of other types are returned unchanged.

    Args:
        x: Torch tensor or NumPy array.
        method: 'minmax' or 'percentile'.
        percentile: Lower quantile as a fraction in [0, 0.5) for percentile normalization.

    Returns:
        Normalized tensor/array.

    Raises:
        ValueError: If ``method`` is unknown.

    """
    if isinstance(x, torch.Tensor):
        if method == 'minmax':
            x = x - x.min()
            x = x / (x.max() - x.min())
        elif method == 'percentile':
            p_min = torch.quantile(x, percentile)
            p_max = torch.quantile(x, 1 - percentile)
            x = x - p_min
            x = x / (p_max - p_min)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        x = torch.clamp(x, 0, 1)

    elif isinstance(x, np.ndarray):
        if method == 'minmax':
            x = x - np.min(x)
            x = x / (np.max(x) - np.min(x))
        elif method == 'percentile':
            p_min = np.percentile(x, percentile * 100)
            p_max = np.percentile(x, (1 - percentile) * 100)
            x = x - p_min
            x = x / (p_max - p_min)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        x = np.clip(x, 0, 1)

    return x


class RandomPatchSampler3D:
    """Crop the same random 3D patch from a feature volume and its target.

    The patch location is drawn once per call (shared across the batch) with ``torch.randint``.

    Args:
        patch_size: Patch size ``(ph, pw, pd)``.
    """

    def __init__(self, patch_size: Sequence[int]):
        self.ph, self.pw, self.pd = patch_size

    def __call__(self, feat: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Crop a random patch.

        Args:
            feat: Feature tensor ``(B, C, H, W, D)``.
            target: Target tensor ``(B, H, W, D)`` or ``(B, 1, H, W, D)``.

        Returns:
            Tuple ``(feat_patch, target_patch)`` of shapes ``(B, C, ph, pw, pd)`` and ``(B, ph, pw, pd)`` or
            ``(B, 1, ph, pw, pd)`` (views of the inputs).

        """
        B, C, H, W, D = feat.shape
        ph, pw, pd = self.ph, self.pw, self.pd

        # pick random start indices so patch fits inside
        sh = torch.randint(0, H - ph + 1, (1,)).item()
        sw = torch.randint(0, W - pw + 1, (1,)).item()
        sd = torch.randint(0, D - pd + 1, (1,)).item()

        # slice out the patch
        feat_patch = feat[:, :, sh : sh + ph, sw : sw + pw, sd : sd + pd]

        if target.ndim == 5:
            target_patch = target[:, :, sh : sh + ph, sw : sw + pw, sd : sd + pd]
        else:
            target_patch = target[:, sh : sh + ph, sw : sw + pw, sd : sd + pd]

        return feat_patch, target_patch


class MultiResPatchSampler3D:
    """Crop corresponding random patches from a multi-resolution feature pyramid and the target.

    Args:
        patch_size: Patch size ``(ph, pw, pd)`` at full resolution; must be divisible by ``2**i`` for all
            ``i = 0..N`` so that lower-resolution patch sizes are integral.
    """

    def __init__(self, patch_size: Sequence[int]):
        self.ph, self.pw, self.pd = patch_size

    def __call__(self, feats: Sequence[torch.Tensor], target: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Crop a random full-resolution patch and the matching patch at every level.

        Args:
            feats: List of ``N + 1`` tensors with ``feats[i].shape == (B, C_i, H/(2**i), W/(2**i), D/(2**i))``.
            target: Target tensor ``(B, H, W, D)`` or ``(B, 1, H, W, D)`` at full resolution.

        Returns:
            Tuple ``(feat_patches, target_patch)``: list of tensors ``(B, C_i, ph_i, pw_i, pd_i)`` and the target
            patch ``(B, ph, pw, pd)`` or ``(B, 1, ph, pw, pd)``.

        """
        H0, W0, D0 = feats[0].shape[2:]
        ph, pw, pd = self.ph, self.pw, self.pd

        # pick a random fully‐contained patch at full res
        sh = torch.randint(0, H0 - ph + 1, (1,)).item()
        sw = torch.randint(0, W0 - pw + 1, (1,)).item()
        sd = torch.randint(0, D0 - pd + 1, (1,)).item()

        feat_patches = []
        for _i, f in enumerate(feats):
            # downsampling factor
            fH, fW, fD = f.shape[2:]
            factor_h = H0 // fH
            factor_w = W0 // fW
            factor_d = D0 // fD

            # corresponding patch in this feature map
            ph_i = ph // factor_h
            pw_i = pw // factor_w
            pd_i = pd // factor_d
            sh_i = sh // factor_h
            sw_i = sw // factor_w
            sd_i = sd // factor_d

            feat_patches.append(f[:, :, sh_i : sh_i + ph_i, sw_i : sw_i + pw_i, sd_i : sd_i + pd_i])

        # crop target
        if target.ndim == 5:  # (B,1,H,W,D)
            tgt = target[:, :, sh : sh + ph, sw : sw + pw, sd : sd + pd]
        else:  # (B,H,W,D)
            tgt = target[:, sh : sh + ph, sw : sw + pw, sd : sd + pd]

        return feat_patches, tgt
