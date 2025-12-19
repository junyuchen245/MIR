import torch

def GLM(X, Y, mask=None):
    '''
    Perform General Linear Model (GLM) fitting.
    X: (N, P) design matrix
    Y: (N, V) data matrix
    mask: (H, W, D) boolean mask to select voxels (optional)
    Returns:
        betas: (P, V) estimated coefficients
        t_for_col: function to compute t-statistics for a given column
    '''
    if mask is not None:
        mask_vec = mask.reshape(-1).to(Y.device)      # (H*W*D,)
        Y = Y[:, mask_vec.flatten()]
    # Proceed to compute GLM
    XtX_inv = torch.linalg.pinv(X.T @ X)
    betas = XtX_inv @ X.T @ Y
    resid = Y - X @ betas
    rank = torch.linalg.matrix_rank(X)
    df = int(X.shape[0] - rank.item())
    sigma2 = (resid.pow(2).sum(0) / max(df,1)).clamp_min(1e-12)

    def t_for_col(j):
        se = torch.sqrt(sigma2 * XtX_inv[j,j])
        return betas[j] / se
    t_maps = [t_for_col(j) for j in range(betas.shape[0])]
    return betas, t_maps

def winsorize(Y, p_low=1.0, p_high=99.0):
    '''
    Winsorize each column of Y at the given percentiles.
    Y: (N, V) tensor
    '''
    lo = torch.quantile(Y, p_low/100.0, dim=0, keepdim=True)
    hi = torch.quantile(Y, p_high/100.0, dim=0, keepdim=True)
    return torch.clamp(Y, lo, hi)

def janmahasatian_lbm(height_m, weight_kg, sex_is_male=True):
    """
    Janmahasatian et al. Clin Pharmacokinet 2005.
    height in meters, weight in kg.
    Returns Lean Body Mass (LBM) in kg.
    """
    BMI = weight_kg / (height_m*height_m + 1e-8)
    if sex_is_male:
        # LBM_male = 9270 * W / (6680 + 216 * BMI)
        return 9270.0 * weight_kg / (6680.0 + 216.0 * BMI + 1e-8)
    else:
        # LBM_female = 9270 * W / (8780 + 244 * BMI)
        return 9270.0 * weight_kg / (8780.0 + 244.0 * BMI + 1e-8)

def robust_bp_ref(vol, mask, q=0.5):
    """Median blood-pool SUV within aorta mask.
    vol: (H, W, D) tensor
    mask: (H, W, D) boolean tensor
    q: quantile to compute (default 0.5 for median)
    Returns: the q-th quantile of vol within mask.
    """
    vals = vol[mask>0]
    return torch.quantile(vals, q)
