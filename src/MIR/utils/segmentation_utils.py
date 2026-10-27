"""Segmentation inference helpers (sliding-window)."""

from typing import List, Sequence

import torch


def get_patch_starts(dim_size: int, patch_size: int, stride: int) -> List[int]:
    """Compute patch start indices for sliding-window inference.

    Starts are ``0, stride, 2*stride, ...``; a final start ``dim_size - patch_size`` is appended if needed so the
    last patch ends exactly at ``dim_size``.

    Args:
        dim_size: Size of the dimension.
        patch_size: Patch size along the dimension.
        stride: Stride between patches.

    Returns:
        List of start indices.

    """
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
    mode: str = 'argmax',  # we'll focus on improved 'argmax'
) -> torch.Tensor:
    """Run sliding-window inference on a full-resolution feature volume.

    ``head`` is put in eval mode and run under ``torch.no_grad()`` on every patch. The per-patch argmax labels are
    written back using only the central part of each patch (half of the overlap margin is trimmed on each interior
    side), so neighbouring patches do not overwrite each other's centres.

    Args:
        feat: Feature tensor (B, C, H, W, D).
        head: Segmentation head that outputs logits (B, num_classes, ph, pw, pd).
        patch_size: Patch size (ph, pw, pd).
        overlap: Overlap ratio between patches.
        num_classes: Number of output classes (not used).
        mode: Output mode (not used; argmax labels are always returned).

    Returns:
        Segmentation prediction (B, 1, H, W, D), dtype ``torch.long``.

    """
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
            left_h = mh // 2
            right_h = mh - left_h
            if x0 == 0:
                left_h = 0
            if x0 + ph == H:
                right_h = 0
            ch = ph - left_h - right_h

            for y0 in ys:
                # dynamic margins along W
                mw = pw - sw
                left_w = mw // 2
                right_w = mw - left_w
                if y0 == 0:
                    left_w = 0
                if y0 + pw == W:
                    right_w = 0
                cw = pw - left_w - right_w

                for z0 in zs:
                    # dynamic margins along D
                    md = pd - sd
                    left_d = md // 2
                    right_d = md - left_d
                    if z0 == 0:
                        left_d = 0
                    if z0 + pd == D:
                        right_d = 0
                    cd = pd - left_d - right_d

                    # crop patch
                    patch = feat[:, :, x0 : x0 + ph, y0 : y0 + pw, z0 : z0 + pd]  # (B,C,ph,pw,pd)

                    logits = head(patch)  # (B,num_classes,ph,pw,pd)
                    patch_lbl = torch.argmax(logits, dim=1)  # (B,ph,pw,pd)

                    # define where to write in both
                    sx0 = x0 + left_h
                    ex0 = sx0 + ch
                    sy0 = y0 + left_w
                    ey0 = sy0 + cw
                    sz0 = z0 + left_d
                    ez0 = sz0 + cd

                    lx0 = left_h
                    lex = left_h + ch
                    ly0 = left_w
                    ley = left_w + cw
                    lz0 = left_d
                    lez = left_d + cd

                    seg_pred[:, sx0:ex0, sy0:ey0, sz0:ez0] = patch_lbl[:, lx0:lex, ly0:ley, lz0:lez]

    return seg_pred.unsqueeze(1)


def sliding_window_inference_multires(
    x_feats: Sequence[torch.Tensor],
    decoder: torch.nn.Module,
    patch_size: tuple,
    overlap: float = 0.5,
    num_classes: int = 133,
    mode: str = 'argmax',
) -> torch.Tensor:
    """Run multiresolution sliding-window inference.

    For each full-resolution window the matching window of every level (downsample factor ``H0 // H_i``) is cut out
    and the list is passed to ``decoder``. Full-resolution logits are summed over overlapping windows, averaged and
    reduced with argmax. Runs in eval mode under ``torch.no_grad()``.

    Args:
        x_feats: List of feature tensors (B, C_i, H_i, W_i, D_i) at multiple resolutions, finest first.
        decoder: Decoder module mapping a list of patches to logits (B, num_classes, ph, pw, pd).
        patch_size: Full-resolution patch size (ph, pw, pd).
        overlap: Overlap ratio between patches.
        num_classes: Number of output classes (size of the logit accumulator).
        mode: Output mode (not used; argmax labels are always returned).

    Returns:
        Segmentation prediction (B, 1, H0, W0, D0), dtype ``torch.long``.

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
    logits_sum = x_feats[0].new_zeros((B, num_classes, H0, W0, D0))
    count_map = x_feats[0].new_zeros((B, 1, H0, W0, D0))

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
                        patch_feats.append(f[:, :, sh_i : sh_i + ph_i, sw_i : sw_i + pw_i, sd_i : sd_i + pd_i])
                    # run head
                    logits_patch = decoder(patch_feats)
                    # accumulate at full res
                    logits_sum[:, :, x0 : x0 + ph, y0 : y0 + pw, z0 : z0 + pd] += logits_patch
                    count_map[:, 0, x0 : x0 + ph, y0 : y0 + pw, z0 : z0 + pd] += 1

    # average logits and global argmax
    avg_logits = logits_sum / count_map
    seg_pred = torch.argmax(avg_logits, dim=1)  # (B,H0,W0,D0)
    return seg_pred.unsqueeze(1)
