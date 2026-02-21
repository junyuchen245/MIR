import glob
import os
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader
import nibabel as nib

from MIR.utils.deedsbcv_binary import get_deedsbcv_executable
from MIR.models import SpatialTransformer
from MIR.accuracy_measures import dice_val_VOI
from data import datasets, trans


ATLAS_DIR = "/scratch2/jchen/DATA/IXI/atlas.pkl"
VAL_DIR = "/scratch2/jchen/DATA/IXI/Val/"
OUTPUT_DIR = "IXI_deedsBCV_val_outputs"
VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]


def load_flow(flow_prefix: str) -> torch.Tensor:
    disp_ux = nib.load(flow_prefix + "_ux.nii.gz").get_fdata()[None, ...]
    disp_vx = nib.load(flow_prefix + "_vx.nii.gz").get_fdata()[None, ...]
    disp_wx = nib.load(flow_prefix + "_wx.nii.gz").get_fdata()[None, ...]
    disp_arr = np.concatenate([disp_vx, disp_ux, disp_wx], axis=0)
    disp_tensor = torch.from_numpy(disp_arr).float().unsqueeze(0)
    return disp_tensor


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    deeds_exe = get_deedsbcv_executable("deedsBCV")

    test_composed = trans.Compose([
        trans.Seg_norm(),
        trans.NumpyType((np.float32, np.int16)),
    ])
    val_set = datasets.IXIBrainInferDataset(
        glob.glob(os.path.join(VAL_DIR, "*.pkl")),
        ATLAS_DIR,
        transforms=test_composed,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    spatial_trans = SpatialTransformer((160, 192, 224), mode="nearest").cuda()

    dsc_vals = []
    for i, data in enumerate(val_loader):
        x, y, x_seg, y_seg = [t.cuda() for t in data]
        x_path = os.path.join(OUTPUT_DIR, "moving.nii.gz")
        y_path = os.path.join(OUTPUT_DIR, "fixed.nii.gz")
        out_prefix = os.path.join(OUTPUT_DIR, "dense_disp")

        nib.Nifti1Image(x[0, 0].detach().cpu().numpy(), np.eye(4)).to_filename(x_path)
        nib.Nifti1Image(y[0, 0].detach().cpu().numpy(), np.eye(4)).to_filename(y_path)

        cmd = f"{deeds_exe} -F {y_path} -M {x_path} -O {out_prefix} -G 6x5x4x3x2 -L 6x5x4x3x2 -Q 5x4x3x2x1"
        os.system(cmd)

        flow = load_flow(out_prefix).cuda()
        def_out = spatial_trans(x_seg.float(), flow)
        dsc = dice_val_VOI(def_out.long(), y_seg.long(), eval_labels=VOI_lbls)
        dsc_vals.append(dsc.item())
        print(f"Case {i + 1}/{len(val_loader)} DSC: {dsc.item():.4f}")

    if dsc_vals:
        print(f"Validation DSC mean: {np.mean(dsc_vals):.4f} ± {np.std(dsc_vals):.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
