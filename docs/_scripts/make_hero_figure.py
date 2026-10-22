"""Generate the landing-page figure of the documentation (docs/_static/hero_registration.png).

A LUMIR validation pair (Learn2Reg 2024, CC BY 4.0) registered with the TransMorph-TVF model trained in
tutorials/lumir25_TransMorph: moving, fixed, warped moving, deformed grid and log Jacobian determinant on
the mid axial slice. The background is transparent so the figure reads in both light and dark mode.

Usage (from the repository root, needs the LUMIR data and the tutorial checkpoint):
    python3.8 docs/_scripts/make_hero_figure.py [--ckpt PATH] [--slice K]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from MIR.models import TransMorphTVF, SpatialTransformer
from MIR.models.lddmm import jacobian_fd, det3x3
from MIR.utils import mk_grid_img
import MIR.models.configs_TransMorph as configs_TransMorph

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = "/scratch2/jchen/DATA/LUMIR/LUMIR25/"
PAIR = ("imagesVal/LUMIRMRI_3454_0000.nii.gz", "imagesVal/LUMIRMRI_3455_0000.nii.gz")
CKPT = os.path.join(ROOT, "tutorials/lumir25_TransMorph/experiments/TransMorph_with_aug_LUMIR25_ncc_1_diffusion_1/dsc0.7253.pth.tar")
OUT = os.path.join(ROOT, "docs/_static/hero_registration.png")
H, W, D = 160, 224, 192
LABEL = "#8a96a3"  # mid grey: legible on light and dark backgrounds


def load(p):
    return torch.from_numpy(nib.load(DATA + p).get_fdata()[None, None] / 255.).float().cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--slice", type=int, default=None, help="index along the last axis (default: middle)")
    args = ap.parse_args()

    config = configs_TransMorph.get_3DTransMorph3Lvl_config()
    config.img_size = (H // 2, W // 2, D // 2)
    config.window_size = (H // 64, W // 64, D // 64)
    config.out_chan = 3
    model = TransMorphTVF(config, time_steps=7).cuda().eval()
    sd = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    model.load_state_dict({k: v for k, v in sd.items() if not k.endswith(".grid")})  # grids are non-persistent buffers now

    mov, fix = load(PAIR[0]), load(PAIR[1])
    st = SpatialTransformer((H, W, D)).cuda()
    with torch.no_grad():
        flow = model((F.avg_pool3d(mov, 2), F.avg_pool3d(fix, 2)))
        flow = F.interpolate(flow, size=(H, W, D), mode="trilinear", align_corners=False) * 2
        warped = st(mov, flow)
        grid = st(mk_grid_img(6, 1, (H, W, D), dim=2).cuda().float(), flow)
        logdet = torch.log(det3x3(jacobian_fd(flow)).clamp_min(1e-3))

    k = args.slice if args.slice is not None else D // 2

    def sl(t):
        return np.rot90(t[0, 0, :, :, k].cpu().numpy() if t.dim() == 5 else t[0, :, :, k].cpu().numpy())

    panels = [("moving", sl(mov), "gray", (0, 1)), ("fixed", sl(fix), "gray", (0, 1)),
              ("warped moving", sl(warped), "gray", (0, 1)), ("deformation", sl(grid), "gray", (0, 1)),
              ("log Jacobian", sl(logdet), "RdBu_r", (-1.2, 1.2))]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.9), dpi=170)
    fig.patch.set_alpha(0)
    body = sl(fix) > 0.02
    for ax, (name, img, cmap, (lo, hi)) in zip(axes, panels):
        if name == "log Jacobian":
            img = np.where(body, img, np.nan)
        ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="bilinear")
        ax.set_title(name, color=LABEL, fontsize=21, pad=8)
        ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.01, wspace=0.03)
    fig.savefig(OUT, transparent=True)
    print("wrote", OUT, "| folding (det<=0) in brain:",
          f"{float((det3x3(jacobian_fd(flow)) <= 0).float().mean()) * 100:.3f}%")


if __name__ == "__main__":
    main()
