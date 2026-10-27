"""Miscellaneous utility helpers for MIR."""

import pickle
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
import zipfile
import os
from scipy.ndimage import zoom
import numpy as np

HERE = Path(__file__).resolve().parent
text_path = HERE / "FreeSurfer_label_info.txt"


def pkload(fname: Union[str, os.PathLike]) -> Any:
    """Load a pickled object from disk.

    Args:
        fname: Path to pickle file.

    Returns:
        Loaded object.

    """
    with open(fname, 'rb') as f:
        return pickle.load(f)


def savepkl(data: Any, path: Union[str, os.PathLike]) -> None:
    """Save an object to a pickle file.

    Args:
        data: Object to serialize.
        path: Output pickle file path.

    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def write2csv(line: str, name: str) -> None:
    """Append a line to a CSV file.

    Args:
        line: Text to write (a newline is appended).
        name: File name without extension; ``'.csv'`` is appended.

    """
    with open(name + '.csv', 'a') as file:
        file.write(line)
        file.write('\n')


def process_label() -> Dict[int, str]:
    """Build the lookup from contiguous label index to FreeSurfer structure name.

    The 46 FreeSurfer label IDs used by MIR are mapped, in order, to indices 0..45 and their names are read from
    ``FreeSurfer_label_info.txt`` next to this module.

    Returns:
        Dict mapping contiguous label index to structure name.

    """
    # process labeling information for FreeSurfer
    seg_table = [
        0,
        2,
        3,
        4,
        5,
        7,
        8,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        24,
        26,
        28,
        30,
        31,
        41,
        42,
        43,
        44,
        46,
        47,
        49,
        50,
        51,
        52,
        53,
        54,
        58,
        60,
        62,
        63,
        72,
        77,
        80,
        85,
        251,
        252,
        253,
        254,
        255,
    ]

    file1 = open(text_path)
    Lines = file1.readlines()
    dict = {}
    seg_i = 0
    seg_look_up = []
    for seg_label in seg_table:
        for line in Lines:
            line = re.sub(' +', ' ', line).split(' ')
            try:
                int(line[0])
            except (ValueError, IndexError):
                continue
            if int(line[0]) == seg_label:
                seg_look_up.append([seg_i, int(line[0]), line[1]])
                dict[seg_i] = line[1]
        seg_i += 1
    return dict


def SLANT_label_reassign(label_map: np.ndarray) -> np.ndarray:
    """Reassign SLANT label IDs to contiguous indices.

    Each SLANT label value in the internal lookup table is replaced by its position in the table; values not in the
    table are left unchanged.

    Args:
        label_map: Label map array.

    Returns:
        Reassigned label map (a copy with the dtype of ``label_map``).

    """
    # process labeling information for SLANT
    label_lookup = [
        0.0,
        4.0,
        11.0,
        23.0,
        30.0,
        31.0,
        32.0,
        35.0,
        36.0,
        37.0,
        38.0,
        39.0,
        40.0,
        41.0,
        44.0,
        45.0,
        47.0,
        48.0,
        49.0,
        50.0,
        51.0,
        52.0,
        55.0,
        56.0,
        57.0,
        58.0,
        59.0,
        60.0,
        61.0,
        62.0,
        71.0,
        72.0,
        73.0,
        75.0,
        76.0,
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,
        105.0,
        106.0,
        107.0,
        108.0,
        109.0,
        112.0,
        113.0,
        114.0,
        115.0,
        116.0,
        117.0,
        118.0,
        119.0,
        120.0,
        121.0,
        122.0,
        123.0,
        124.0,
        125.0,
        128.0,
        129.0,
        132.0,
        133.0,
        134.0,
        135.0,
        136.0,
        137.0,
        138.0,
        139.0,
        140.0,
        141.0,
        142.0,
        143.0,
        144.0,
        145.0,
        146.0,
        147.0,
        148.0,
        149.0,
        150.0,
        151.0,
        152.0,
        153.0,
        154.0,
        155.0,
        156.0,
        157.0,
        160.0,
        161.0,
        162.0,
        163.0,
        164.0,
        165.0,
        166.0,
        167.0,
        168.0,
        169.0,
        170.0,
        171.0,
        172.0,
        173.0,
        174.0,
        175.0,
        176.0,
        177.0,
        178.0,
        179.0,
        180.0,
        181.0,
        182.0,
        183.0,
        184.0,
        185.0,
        186.0,
        187.0,
        190.0,
        191.0,
        192.0,
        193.0,
        194.0,
        195.0,
        196.0,
        197.0,
        198.0,
        199.0,
        200.0,
        201.0,
        202.0,
        203.0,
        204.0,
        205.0,
        206.0,
        207.0,
    ]
    label = label_map.copy()
    ref = label_map.copy()
    for i in range(len(label_lookup)):
        label[ref == label_lookup[i]] = i
    return label


class CenterCropPad3D(nn.Module):
    """Crop or pad a 3-D medical image tensor so the spatial size becomes exactly ``target_size = (X, Y, Z)``.

    Input shape ``(B, C, H, W, D)``, output shape ``(B, C, X, Y, Z)``. Cropping and padding are done symmetrically
    around the centre (an odd excess is cropped from / padded on the end side).

    Args:
        target_size: Target spatial size ``(X, Y, Z)``.
        padding_mode: Padding mode passed to ``torch.nn.functional.pad``.
        padding_value: Default fill value for constant padding (overridden by the ``padding_value`` argument of
            :meth:`forward`).

    Raises:
        ValueError: If ``target_size`` does not have three entries.
    """

    def __init__(
        self,
        target_size: Sequence[int],
        padding_mode: str = "constant",
        padding_value: float = 0.0,
    ):
        super().__init__()
        if len(target_size) != 3:
            raise ValueError("`target_size` must be a 3‑tuple (X, Y, Z).")
        self.tgt = target_size
        self.padding_value = padding_value
        self.pad_mode = padding_mode

    @staticmethod
    def _get_slices(in_len: int, out_len: int) -> Tuple[int, int, int, int]:
        """Return crop indices and padding for one dimension.

        Returns:
            Tuple ``(start, end, pad_left, pad_right)`` such that cropping/padding is symmetric.
        """
        if in_len >= out_len:  # need crop
            excess = in_len - out_len
            start = excess // 2
            end = start + out_len
            pad_l = pad_r = 0
        else:  # need pad
            deficit = out_len - in_len
            pad_l = deficit // 2
            pad_r = deficit - pad_l
            start, end = 0, in_len
        return start, end, pad_l, pad_r

    def forward(self, x: torch.Tensor, padding_value: Optional[float] = None) -> torch.Tensor:
        """Center-crop and/or pad ``x`` to the target size.

        Args:
            x: Input tensor ``(B, C, H, W, D)``.
            padding_value: Fill value for constant padding; ``None`` uses the constructor's ``padding_value``.

        Returns:
            Tensor ``(B, C, X, Y, Z)``.

        Raises:
            ValueError: If ``x`` is not 5-D.

        """
        if padding_value is None:
            padding_value = self.padding_value
        if x.ndim != 5:
            raise ValueError("Input must have shape (B,C,H,W,D)")

        _, _, H, W, D = x.shape
        tgt_H, tgt_W, tgt_D = self.tgt

        # ---------- cropping ----------
        sl_H = slice(*self._get_slices(H, tgt_H)[:2])
        sl_W = slice(*self._get_slices(W, tgt_W)[:2])
        sl_D = slice(*self._get_slices(D, tgt_D)[:2])
        x = x[:, :, sl_H, sl_W, sl_D]

        # ---------- padding ------------
        _, _, Hc, Wc, Dc = x.shape
        _, _, pad_H_l, pad_H_r = self._get_slices(Hc, tgt_H)
        _, _, pad_W_l, pad_W_r = self._get_slices(Wc, tgt_W)
        _, _, pad_D_l, pad_D_r = self._get_slices(Dc, tgt_D)

        # torch.nn.functional.pad uses (D‑right, D‑left, W‑right, W‑left, H‑right, H‑left)
        pad_tuple = (
            pad_D_l,
            pad_D_r,
            pad_W_l,
            pad_W_r,
            pad_H_l,
            pad_H_r,
        )
        if any(pad_tuple):
            x = F.pad(x, pad_tuple, mode=self.pad_mode, value=padding_value)

        return x


def create_zip(zip_filename: Union[str, os.PathLike], source_dir: Union[str, os.PathLike]) -> None:
    """Create a deflate-compressed zip file from a directory.

    Files are stored recursively with paths relative to ``source_dir``.

    Args:
        zip_filename: The name of the zip file to create (overwritten if it exists).
        source_dir: The directory to zip.

    """
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate the relative path to store in the zip file
                relative_path = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, relative_path)


def load_partial_weights(
    model: nn.Module,
    checkpoint_path: Union[str, os.PathLike],
    weights_key: str = 'state_dict',
    strict: bool = False,
) -> None:
    """Load weights from a checkpoint into a model, skipping unmatched layers.

    The checkpoint is loaded on the CPU, a ``'module.'`` (DataParallel) prefix is stripped from its keys and only
    entries whose key exists in the model with the same shape are loaded. The number of loaded entries is printed.

    Args:
        model (torch.nn.Module): Your model with updated architecture.
        checkpoint_path (str): Path to the .pth or .pt file.
        weights_key (str): Key of the state dict inside the checkpoint dict; if absent, the checkpoint dict itself
            is used as the state dict.
        strict (bool): If True, raise when a model key is missing from the checkpoint (shape mismatches are still
            skipped). If False, loads what it can.

    Raises:
        ValueError: If the checkpoint is not a dict.
        KeyError: If ``strict`` is True and a model key is missing from the checkpoint.

    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle if checkpoint was saved using model.state_dict() or model directly
    if isinstance(checkpoint, dict) and weights_key in checkpoint:
        checkpoint_state_dict = checkpoint[weights_key]
    elif isinstance(checkpoint, dict):
        checkpoint_state_dict = checkpoint
    else:
        raise ValueError("Checkpoint format not recognized")

    # Remove 'module.' prefix if the model was wrapped in DataParallel
    new_ckpt_state_dict = {}
    for k, v in checkpoint_state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_ckpt_state_dict[new_key] = v

    model_state_dict = model.state_dict()
    loadable_state_dict = {}

    # Only load layers that exist in both and have matching shapes
    for k in model_state_dict:
        if k in new_ckpt_state_dict and model_state_dict[k].shape == new_ckpt_state_dict[k].shape:
            loadable_state_dict[k] = new_ckpt_state_dict[k]
        elif strict and k not in new_ckpt_state_dict:
            raise KeyError(f"Key '{k}' missing in checkpoint.")

    model.load_state_dict(loadable_state_dict, strict=False)

    print(f"Loaded {len(loadable_state_dict)} / {len(model_state_dict)} layers from checkpoint.")


def zoom_img(img: np.ndarray, pixel_dims: Tuple[Sequence[float], Sequence[float]], order: int = 3) -> np.ndarray:
    """Resample a 3D image from its voxel spacing to a target spacing with ``scipy.ndimage.zoom``.

    Args:
        img: Input 3D image array.
        pixel_dims: Pair ``(img_pixdim, target_pixdim)`` of per-axis voxel spacings; the zoom factor is
            ``img_pixdim / target_pixdim``.
        order: Spline interpolation order.

    Returns:
        Resampled image array.

    """
    img_pixdim, tar_pix = pixel_dims
    ratio = np.array(img_pixdim) / np.array(tar_pix)
    img = zoom(img, ratio, order=order)
    return img
