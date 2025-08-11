import pickle
import re
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import zipfile
import os

HERE = Path(__file__).resolve().parent
text_path = HERE / "FreeSurfer_label_info.txt"

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def savepkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        
def write2csv(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')
        
def process_label():
    #process labeling information for FreeSurfer
    seg_table = [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
                          28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
                          63, 72, 77, 80, 85, 251, 252, 253, 254, 255]


    file1 = open(text_path, 'r')
    Lines = file1.readlines()
    dict = {}
    seg_i = 0
    seg_look_up = []
    for seg_label in seg_table:
        for line in Lines:
            line = re.sub(' +', ' ',line).split(' ')
            try:
                int(line[0])
            except:
                continue
            if int(line[0]) == seg_label:
                seg_look_up.append([seg_i, int(line[0]), line[1]])
                dict[seg_i] = line[1]
        seg_i += 1
    return dict

def SLANT_label_reassign(label_map):
    #process labeling information for SLANT
    label_lookup = [0.,  4., 11., 23., 30., 31., 32., 35., 36., 37., 38., 39., 40., 41.,
            44.,  45.,  47.,  48.,  49.,  50.,  51.,  52.,  55.,  56.,  57.,  58.,  59., 60.,
            61.,  62.,  71.,  72.,  73.,  75.,  76., 100., 101., 102., 103., 104., 105., 106.,
            107., 108., 109., 112., 113., 114., 115., 116., 117., 118., 119., 120., 121., 122.,
            123., 124., 125., 128., 129., 132., 133., 134., 135., 136., 137., 138., 139., 140.,
            141., 142., 143., 144., 145., 146., 147., 148., 149., 150., 151., 152., 153., 154.,
            155., 156., 157., 160., 161., 162., 163., 164., 165., 166., 167., 168., 169., 170.,
            171., 172., 173., 174., 175., 176., 177., 178., 179., 180., 181., 182., 183., 184.,
            185., 186., 187., 190., 191., 192., 193., 194., 195., 196., 197., 198., 199., 200.,
            201., 202., 203., 204., 205., 206., 207.,]
    label = label_map.copy()
    ref = label_map.copy()
    for i in range(len(label_lookup)):
        label[ref == label_lookup[i]] = i
    return label

class CenterCropPad3D(nn.Module):
    """
    Crop or pad a 3‑D medical image tensor so the spatial size becomes
    exactly `target_size = (X, Y, Z)`.

    Input  shape:  (B, C, H, W, D)
    Output shape:  (B, C, X, Y, Z)

    Cropping and padding are done symmetrically around the centre.
    """

    def __init__(self,
        target_size,
        padding_mode: str = "constant",
        padding_value: float = 0.0,
    ):
        super().__init__()
        if len(target_size) != 3:
            raise ValueError("`target_size` must be a 3‑tuple (X, Y, Z).")
        self.tgt = target_size
        self.pad_mode = padding_mode

    @staticmethod
    def _get_slices(in_len: int, out_len: int):
        """
        Return (start, end) indices and required padding (left, right)
        for one dimension so that cropping/padding is symmetric.
        """
        if in_len >= out_len:                          # need crop
            excess = in_len - out_len
            start = excess // 2
            end   = start + out_len
            pad_l = pad_r = 0
        else:                                          # need pad
            deficit = out_len - in_len
            pad_l = deficit // 2
            pad_r = deficit - pad_l
            start, end = 0, in_len
        return start, end, pad_l, pad_r

    def forward(self, x: torch.Tensor, padding_value=0) -> torch.Tensor:
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
    
def create_zip(zip_filename, source_dir):
    """Creates a zip file from a directory.

    Args:
        zip_filename: The name of the zip file to create.
        source_dir: The directory to zip.
    """
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate the relative path to store in the zip file
                relative_path = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, relative_path)
                

def load_partial_weights(model, checkpoint_path, weights_key='state_dict', strict=False):
    """
    Load weights from a checkpoint into a model, skipping unmatched layers.

    Args:
        model (torch.nn.Module): Your model with updated architecture.
        checkpoint_path (str): Path to the .pth or .pt file.
        strict (bool): If True, behaves like standard strict loading. If False, loads what it can.
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