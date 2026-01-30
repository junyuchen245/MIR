"""
ConvexAdam wrapper using VFA encoder multiscale features. (Based on the initial evaluation, this is not as good as expected.)

This module reuses the ConvexAdam optimization on VFA feature maps by running
coarse-to-fine registration across encoder scales. The feature extraction is
performed under no_grad by default for efficiency.
"""

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from MIR.models.VFA import Encoder, VFA
from MIR.models.convexAdam.convex_adam_features import convex_adam_feats
import MIR.models.configs_VFA as configs_VFA
import MIR.models.convexAdam.configs_ConvexAdam_MIND as configs_ConvexAdam


def _as_tensor(x: Union[np.ndarray, torch.Tensor], device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.from_numpy(x).to(device)


def _ensure_batched(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 4:
        return x.unsqueeze(0)
    return x


def _reduce_channels(feat: torch.Tensor, target_ch: int) -> torch.Tensor:
    if feat.shape[1] <= target_ch:
        return feat
    chunks = torch.chunk(feat, target_ch, dim=1)
    reduced = [c.mean(dim=1, keepdim=True) for c in chunks]
    return torch.cat(reduced, dim=1)


def _rescale_disp(disp: torch.Tensor, new_shape: Sequence[int]) -> torch.Tensor:
    old_shape = disp.shape[2:]
    disp_rs = F.interpolate(disp, size=new_shape, mode="trilinear", align_corners=False)
    scale = torch.tensor(
        [(new_shape[0] - 1) / (old_shape[0] - 1),
         (new_shape[1] - 1) / (old_shape[1] - 1),
         (new_shape[2] - 1) / (old_shape[2] - 1)],
        device=disp.device,
        dtype=disp.dtype,
    ).view(1, 3, 1, 1, 1)
    return disp_rs * scale


def _sorted_scales(features: List[torch.Tensor]) -> List[int]:
    sizes = [(i, int(np.prod(f.shape[2:]))) for i, f in enumerate(features)]
    sizes = sorted(sizes, key=lambda x: x[1])
    return [i for i, _ in sizes]


_ENCODER_CACHE = {}


def _get_cached_encoder(vfa_config, vfa_weights_path, vfa_weights_key, device):
    cache_key = (vfa_weights_path, vfa_weights_key, str(device))
    if cache_key in _ENCODER_CACHE:
        return _ENCODER_CACHE[cache_key]
    vfa_model = VFA(vfa_config, device=str(device)).to(device)
    checkpoint = torch.load(vfa_weights_path, map_location=device)
    state_dict = checkpoint.get(vfa_weights_key, checkpoint) if vfa_weights_key else checkpoint
    vfa_model.load_state_dict(state_dict, strict=False)
    encoder = vfa_model.encoder
    _ENCODER_CACHE[cache_key] = encoder
    return encoder


def convex_adam_vfa(
    img_fixed: Union[np.ndarray, torch.Tensor],
    img_moving: Union[np.ndarray, torch.Tensor],
    convex_config=None,
    vfa_config=None,
    feature_scales: Optional[Iterable[int]] = None,
    max_feat_channels: int = 32,
    use_no_grad: bool = True,
    vfa_weights_path: Optional[str] = None,
    vfa_weights_key: Optional[str] = None,
    vfa_encoder: Optional[torch.nn.Module] = None,
    initial_disp: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> torch.Tensor:
    """
    Register images using ConvexAdam on VFA encoder multiscale features.

    Args:
        img_fixed: Fixed image tensor/array.
        img_moving: Moving image tensor/array.
        convex_config: ConvexAdam config (defaults to MIND brain config).
        vfa_config: VFA config (defaults to VFA default config).
        feature_scales: Iterable of encoder scale indices to use (coarse->fine).
        max_feat_channels: Channel cap per scale for efficiency.
        use_no_grad: If True, compute VFA features under no_grad.
        vfa_weights_path: Optional path to a VFA checkpoint that contains both encoder and decoder.
        vfa_weights_key: Optional key to read from a checkpoint dict.
        vfa_encoder: Optional preloaded VFA encoder to reuse across calls.
        initial_disp: Optional initial displacement (B, 3, H, W, D).
        device: Torch device.

    Returns:
        Displacement field tensor (B, 3, H, W, D).
    """
    if convex_config is None:
        convex_config = configs_ConvexAdam.get_ConvexAdam_MIND_brain_default_config()
    if vfa_config is None:
        vfa_config = configs_VFA.get_VFA_default_config()

    x_fix = _ensure_batched(_as_tensor(img_fixed, device)).float()
    x_mov = _ensure_batched(_as_tensor(img_moving, device)).float()

    if not hasattr(vfa_config, "img_size"):
        vfa_config.img_size = x_fix.shape[2:]
    if vfa_encoder is not None:
        encoder = vfa_encoder.to(device)
    elif vfa_weights_path is not None:
        encoder = _get_cached_encoder(vfa_config, vfa_weights_path, vfa_weights_key, device)
    else:
        encoder = Encoder(
            dimension=len(vfa_config.img_size),
            in_channels=vfa_config.in_channels,
            downsamples=vfa_config.downsamples,
            start_channels=vfa_config.start_channels,
            max_channels=vfa_config.max_channels,
        ).to(device)

    if use_no_grad:
        with torch.no_grad():
            feats_fix = encoder(x_fix)
            feats_mov = encoder(x_mov)
    else:
        feats_fix = encoder(x_fix)
        feats_mov = encoder(x_mov)

    if feature_scales is None:
        scale_order = _sorted_scales(feats_fix)
        feature_scales = scale_order[:2] if len(scale_order) > 1 else scale_order

    disp = initial_disp
    for idx in feature_scales:
        f_fix = _reduce_channels(feats_fix[idx], max_feat_channels)
        f_mov = _reduce_channels(feats_mov[idx], max_feat_channels)
        if disp is not None:
            disp = _rescale_disp(disp, f_fix.shape[2:])
        disp = convex_adam_feats(
            img_fixed=f_fix,
            img_moving=f_mov,
            initial_disp=disp,
            lambda_weight=convex_config.lambda_weight,
            grid_sp=convex_config.grid_sp,
            disp_hw=convex_config.disp_hw,
            selected_niter=convex_config.selected_niter,
            selected_smooth=convex_config.selected_smooth,
            grid_sp_adam=convex_config.grid_sp_adam,
            ic=convex_config.ic,
            verbose=convex_config.verbose,
            save_disp=False,
        )

    disp = _rescale_disp(disp, x_fix.shape[2:])
    return disp
