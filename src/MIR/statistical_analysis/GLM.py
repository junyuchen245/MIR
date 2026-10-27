"""Voxel-wise general linear model and PET quantification helpers (torch)."""

from typing import List, Optional, Tuple, Union

import torch


def GLM(
    X: torch.Tensor, Y: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Perform voxel-wise General Linear Model (GLM) fitting by ordinary least squares.

    ``betas = pinv(X^T X) X^T Y``; the residual variance uses ``df = N - rank(X)`` (at least 1) and the t-statistic
    of coefficient ``j`` is ``beta_j / sqrt(sigma2 * pinv(X^T X)[j, j])``.

    Args:
        X: (N, P) design matrix.
        Y: (N, V) data matrix.
        mask: (H, W, D) boolean mask to select voxels (optional); if given, ``V`` must equal ``H*W*D`` and only
            masked columns of ``Y`` are fitted.

    Returns:
        Tuple ``(betas, t_maps)``: ``betas`` (P, V') estimated coefficients and ``t_maps``, a list of P tensors (V',)
        with the t-statistic of each coefficient (``V'`` = number of fitted voxels).

    """
    if mask is not None:
        mask_vec = mask.reshape(-1).to(Y.device)  # (H*W*D,)
        Y = Y[:, mask_vec.flatten()]
    # Proceed to compute GLM
    XtX_inv = torch.linalg.pinv(X.T @ X)
    betas = XtX_inv @ X.T @ Y
    resid = Y - X @ betas
    rank = torch.linalg.matrix_rank(X)
    df = int(X.shape[0] - rank.item())
    sigma2 = (resid.pow(2).sum(0) / max(df, 1)).clamp_min(1e-12)

    def t_for_col(j):
        se = torch.sqrt(sigma2 * XtX_inv[j, j])
        return betas[j] / se

    t_maps = [t_for_col(j) for j in range(betas.shape[0])]
    return betas, t_maps


def winsorize(Y: torch.Tensor, p_low: float = 1.0, p_high: float = 99.0) -> torch.Tensor:
    """Winsorize each column of Y at the given percentiles.

    Args:
        Y: (N, V) tensor.
        p_low: Lower percentile (0-100).
        p_high: Upper percentile (0-100).

    Returns:
        Tensor (N, V) with each column clamped to its ``[p_low, p_high]`` percentile range.

    """
    lo = torch.quantile(Y, p_low / 100.0, dim=0, keepdim=True)
    hi = torch.quantile(Y, p_high / 100.0, dim=0, keepdim=True)
    return torch.clamp(Y, lo, hi)


def janmahasatian_lbm(
    height_m: Union[float, torch.Tensor], weight_kg: Union[float, torch.Tensor], sex_is_male: bool = True
) -> Union[float, torch.Tensor]:
    """Compute lean body mass with the Janmahasatian formula (Clin Pharmacokinet 2005).

    Args:
        height_m: Height in meters.
        weight_kg: Weight in kg.
        sex_is_male: Use the male formula ``9270 W / (6680 + 216 BMI)``; otherwise the female formula
            ``9270 W / (8780 + 244 BMI)``.

    Returns:
        Lean Body Mass (LBM) in kg.

    """
    BMI = weight_kg / (height_m * height_m + 1e-8)
    if sex_is_male:
        # LBM_male = 9270 * W / (6680 + 216 * BMI)
        return 9270.0 * weight_kg / (6680.0 + 216.0 * BMI + 1e-8)
    # LBM_female = 9270 * W / (8780 + 244 * BMI)
    return 9270.0 * weight_kg / (8780.0 + 244.0 * BMI + 1e-8)


def robust_bp_ref(vol: torch.Tensor, mask: torch.Tensor, q: float = 0.5) -> torch.Tensor:
    """Blood-pool reference value: a quantile (default median) of the SUV within an aorta mask.

    Args:
        vol: (H, W, D) tensor.
        mask: (H, W, D) tensor; voxels with ``mask > 0`` are used.
        q: Quantile to compute (default 0.5 for median).

    Returns:
        Scalar tensor with the q-th quantile of vol within mask.

    """
    vals = vol[mask > 0]
    return torch.quantile(vals, q)
