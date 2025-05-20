import math, random, pickle
import numpy as np
import torch.nn.functional as F
import torch, sys
from torch import nn

def get_patch_starts(dim_size, patch_size, stride):
    starts = list(range(0, dim_size - patch_size + 1, stride))
    if not starts or starts[-1] + patch_size < dim_size:
        starts.append(dim_size - patch_size)
    return starts

def sliding_window_inference(
    feat: torch.Tensor,
    head: torch.nn.Module,
    patch_size: tuple,
    overlap: float = 0.5,
    num_classes: int = 133,
    mode: str = 'argmax'   # we'll focus on improved 'argmax'
) -> torch.Tensor:
    B, C, H, W, D = feat.shape
    ph, pw, pd = patch_size

    # strides
    sh = max(int(ph * (1 - overlap)), 1)
    sw = max(int(pw * (1 - overlap)), 1)
    sd = max(int(pd * (1 - overlap)), 1)

    xs = get_patch_starts(H, ph, sh)
    ys = get_patch_starts(W, pw, sw)
    zs = get_patch_starts(D, pd, sd)

    # output label map, init to ignore (-1)
    seg_pred = feat.new_full((B, H, W, D), -1, dtype=torch.long)

    head.eval()
    with torch.no_grad():
        for x0 in xs:
            # dynamic margins along H
            mh = ph - sh
            left_h  = mh // 2
            right_h = mh - left_h
            if x0 == 0:           left_h  = 0
            if x0 + ph == H:      right_h = 0
            ch = ph - left_h - right_h

            for y0 in ys:
                # dynamic margins along W
                mw = pw - sw
                left_w  = mw // 2
                right_w = mw - left_w
                if y0 == 0:           left_w  = 0
                if y0 + pw == W:      right_w = 0
                cw = pw - left_w - right_w

                for z0 in zs:
                    # dynamic margins along D
                    md = pd - sd
                    left_d  = md // 2
                    right_d = md - left_d
                    if z0 == 0:           left_d  = 0
                    if z0 + pd == D:      right_d = 0
                    cd = pd - left_d - right_d

                    # crop patch
                    patch = feat[
                        :,
                        :,
                        x0 : x0 + ph,
                        y0 : y0 + pw,
                        z0 : z0 + pd
                    ]  # (B,C,ph,pw,pd)

                    logits = head(patch)                 # (B,num_classes,ph,pw,pd)
                    patch_lbl = torch.argmax(logits, dim=1)  # (B,ph,pw,pd)

                    # define where to write in both
                    sx0 = x0 + left_h; ex0 = sx0 + ch
                    sy0 = y0 + left_w; ey0 = sy0 + cw
                    sz0 = z0 + left_d; ez0 = sz0 + cd

                    lx0 = left_h;            lex = left_h + ch
                    ly0 = left_w;            ley = left_w + cw
                    lz0 = left_d;            lez = left_d + cd

                    seg_pred[
                        :, sx0:ex0,
                           sy0:ey0,
                           sz0:ez0
                    ] = patch_lbl[
                        :, lx0:lex,
                            ly0:ley,
                            lz0:lez
                    ]

    return seg_pred.unsqueeze(1)

def sliding_window_inference_multires(
    x_feats,
    decoder: torch.nn.Module,
    patch_size: tuple,
    overlap: float = 0.5,
    num_classes: int = 133,
    mode: str = 'argmax'
) -> torch.Tensor:
    """
    x_feats     : list of (B, C_i, H_i, W_i, D_i), with H_i = H0/2**i etc.
    decoder     : nn.Module taking list of feature‐patches → (B,C,ph,pw,pd)
    patch_size  : (ph, pw, pd) at full resolution H0×W0×D0
    overlap     : fraction in [0..1)
    num_classes : number of output classes
    
    returns:
    --------
    seg_pred    : (B, H0, W0, D0) LongTensor of labels
    """
    B = x_feats[0].shape[0]
    H0, W0, D0 = x_feats[0].shape[2:]
    ph, pw, pd = patch_size

    # compute full‐res stride
    sh = max(int(ph * (1 - overlap)), 1)
    sw = max(int(pw * (1 - overlap)), 1)
    sd = max(int(pd * (1 - overlap)), 1)

    xs = get_patch_starts(H0, ph, sh)
    ys = get_patch_starts(W0, pw, sw)
    zs = get_patch_starts(D0, pd, sd)

    # output buffers
    device = x_feats[0].device
    logits_sum = x_feats[0].new_zeros((B, num_classes, H0, W0, D0))
    count_map  = x_feats[0].new_zeros((B, 1, H0, W0, D0))

    # precompute downsample factors for each level
    factors = [H0 // f.shape[2] for f in x_feats]

    decoder.eval()
    with torch.no_grad():
        for x0 in xs:
            for y0 in ys:
                for z0 in zs:
                    # extract multi‐res patches
                    patch_feats = []
                    for f, factor in zip(x_feats, factors):
                        ph_i = ph // factor
                        pw_i = pw // factor
                        pd_i = pd // factor
                        sh_i = x0 // factor
                        sw_i = y0 // factor
                        sd_i = z0 // factor
                        patch_feats.append(
                            f[:, :,
                              sh_i:sh_i+ph_i,
                              sw_i:sw_i+pw_i,
                              sd_i:sd_i+pd_i]
                        )
                    # run head
                    logits_patch = decoder(patch_feats)  
                    # accumulate at full res
                    logits_sum[
                        :,
                        :,
                        x0 : x0+ph,
                        y0 : y0+pw,
                        z0 : z0+pd
                    ] += logits_patch
                    count_map[
                        :,
                        0,
                        x0 : x0+ph,
                        y0 : y0+pw,
                        z0 : z0+pd
                    ] += 1

    # average logits and global argmax
    avg_logits = logits_sum / count_map
    seg_pred   = torch.argmax(avg_logits, dim=1)  # (B,H0,W0,D0)
    return seg_pred.unsqueeze(1)