"""Training utilities (logging, metrics, and patch sampling)."""

import math, random, pickle
import numpy as np
import torch.nn.functional as F
import torch, sys
from torch import nn

class Logger(object):
    def __init__(self, save_dir):
        """Initialize a stdout logger that also writes to a file.

        Args:
            save_dir: Directory path for log file.
        """
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        """Write a message to both stdout and the log file."""
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """No-op flush for compatibility with file-like interfaces."""
        pass
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset stored statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        """Update running statistics.

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

def pad_image(img, target_size):
    """Pad a 3D tensor to a target size.

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

def normalize_01(x, method='minmax', percentile=0.05):
    """Normalize a tensor/array to the range [0, 1].

    Args:
        x: Torch tensor or NumPy array.
        method: 'minmax' or 'percentile'.
        percentile: Percentile for percentile normalization.

    Returns:
        Normalized tensor/array.
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
            raise ValueError("Unknown normalization method: {}".format(method))
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
            raise ValueError("Unknown normalization method: {}".format(method))
        x = np.clip(x, 0, 1)
    
    return x

class RandomPatchSampler3D:
    def __init__(self, patch_size):
        """
        patch_size: tuple of (ph, pw, pd)
        """
        self.ph, self.pw, self.pd = patch_size

    def __call__(self, feat: torch.Tensor, target: torch.Tensor):
        """
        feat:   (B, C,  H,  W,  D)
        target: (B,    H,  W,  D)   or   (B, 1, H, W, D)
        
                Returns:
          feat_patch:   (B, C, ph, pw, pd)
          target_patch: (B,   ph, pw, pd)  or  (B, 1, ph, pw, pd)
        """
        B, C, H, W, D = feat.shape
        ph, pw, pd    = self.ph, self.pw, self.pd

        # pick random start indices so patch fits inside
        sh = torch.randint(0, H - ph + 1, (1,)).item()
        sw = torch.randint(0, W - pw + 1, (1,)).item()
        sd = torch.randint(0, D - pd + 1, (1,)).item()

        # slice out the patch
        feat_patch = feat[:, :, sh:sh+ph, sw:sw+pw, sd:sd+pd]
        
        if target.ndim == 5:
            target_patch = target[:, :, sh:sh+ph, sw:sw+pw, sd:sd+pd]
        else:
            target_patch = target[:,    sh:sh+ph, sw:sw+pw, sd:sd+pd]

        return feat_patch, target_patch
    
class MultiResPatchSampler3D:
    def __init__(self, patch_size):
        """
        patch_size: tuple (ph, pw, pd) at full resolution, must be divisible by 2**i
                    for all i=0..N so that lower‐res patch sizes are integral.
        """
        self.ph, self.pw, self.pd = patch_size

    def __call__(self, feats, target: torch.Tensor):
        """
        Args:
          feats : list of length N+1 of tensors
                  feats[i].shape == (B, C_i, H/(2**i), W/(2**i), D/(2**i))
          target: (B, H, W, D)     or   (B,1,H,W,D)

                Returns:
          feat_patches  : list of tensors [ (B,C_i,ph_i,pw_i,pd_i) ... ]
          target_patch  : (B,ph,pw,pd) or (B,1,ph,pw,pd)
        """
        B = feats[0].shape[0]
        H0, W0, D0 = feats[0].shape[2:]
        ph, pw, pd = self.ph, self.pw, self.pd

        # pick a random fully‐contained patch at full res
        sh = torch.randint(0, H0 - ph + 1, (1,)).item()
        sw = torch.randint(0, W0 - pw + 1, (1,)).item()
        sd = torch.randint(0, D0 - pd + 1, (1,)).item()

        feat_patches = []
        for i, f in enumerate(feats):
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

            feat_patches.append(
                f[:, :,
                  sh_i : sh_i + ph_i,
                  sw_i : sw_i + pw_i,
                  sd_i : sd_i + pd_i]
            )

        # crop target
        if target.ndim == 5:  # (B,1,H,W,D)
            tgt = target[:, :,
                         sh:sh+ph,
                         sw:sw+pw,
                         sd:sd+pd]
        else:                 # (B,H,W,D)
            tgt = target[:,
                         sh:sh+ph,
                         sw:sw+pw,
                         sd:sd+pd]

        return feat_patches, tgt
