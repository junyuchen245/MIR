import numpy as np
import torch
import torch.nn.functional as F
import math
import nibabel as nib


def remap_totalseg_lbl(lbl):
    # New grouping table based on updated TotalSegmentator labels,
    # but matching the *same grouped outputs* as the old version.
    grouping_table = [
        [1],                           # 1 spleen
        [2],                           # 2 kidney_right
        [3],                           # 3 kidney_left
        [4],                           # 4 gallbladder
        [5],                           # 5 liver
        [6],                           # 6 stomach
        [52],                          # 7 aorta
        [63],                          # 8 inferior_vena_cava
        [64],                          # 9 portal_vein_and_splenic_vein
        [7],                           # 10 pancreas
        [8],                           # 11 adrenal_gland_right
        [9],                           # 12 adrenal_gland_left
        [10, 11],                      # 13 lung upper+lower lobe left
        [12, 13, 14],                  # 14 lung upper/middle/lower right
        [27, 28, 29, 30, 31, 32, 33,   # 15 vertebrae L5 to C1
         34, 35, 36, 37, 38, 39, 40,
         41, 42, 43, 44, 45, 46, 47,
         48, 49, 50, 26, 25],          # Include S1 and sacrum (old grouping had sacrum separately, but L+T+C+S grouped)
        [15],                          # 16 esophagus
        [16],                          # 17 trachea
        [51],                          # 18 heart (old version separated myocardium/atrium/ventricle but grouped into one label)
        [61],                          # 19 heart_atrium_left (moved to same group as myocardium in old grouping)
        [46],                          # 20 heart_ventricle_left -> (merged)
        [47],                          # 21 heart_atrium_right -> (merged)
        [48],                          # 22 heart_ventricle_right -> (merged)
        [53],                          # 23 pulmonary vein (old had pulmonary artery, but analogous major pulmonary vessel group)
        [18],                          # 24 small_bowel
        [19],                          # 25 duodenum
        [20],                          # 26 colon
        [92, 93, 94, 95, 96, 97, 98,   # 27 left ribs 1-12
         99, 100, 101, 102, 103,
         104, 105, 106, 107, 108,      # right ribs 1-12
         109, 110, 111, 112, 113,
         114, 115],
        [69, 71, 73],                  # 28 humerus_left, scapula_left, clavicula_left
        [70, 72, 74],                  # 29 humerus_right, scapula_right, clavicula_right
        [75, 77],                      # 30 femur_left, hip_left
        [76, 78],                      # 31 femur_right, hip_right
        [25],                          # 32 sacrum (old had sacrum alone too, but here also part of vertebrae group above if desired)
        [80, 82, 84],                  # 33 gluteus_maximus_left, gluteus_medius_left, gluteus_minimus_left
        [81, 83, 85],                  # 34 gluteus_maximus_right, gluteus_medius_right, gluteus_minimus_right
        [86],                          # 35 autochthon_left
        [87],                          # 36 autochthon_right
        [88],                          # 37 iliopsoas_left
        [89],                          # 38 iliopsoas_right
        [21]                           # 39 urinary_bladder
    ]

    label_out = np.zeros_like(lbl)
    for idx, group in enumerate(grouping_table):
        for seg_i in group:
            label_out[lbl == seg_i] = idx + 1
    return label_out

def remap_suv_lbl(lbl, include_lesions=False):
    grouping_table = [
        [101],                           # 1 Liver
        [201],                           # 2 Kidneys
        #[301],                           # Bone
        #[401],                           # Lymph nodes
        [501],                           # 3 Spleen
        [601],                           # 4 Ureters
        [701],                           # 5 Bladder
        [801],                           # 6 Bowel
        [901],                           # 7 Prostate
        [1001],                          # 8 Sinus
        [1101],                          # 9 Paraglottic Space
        [1201],                          # 10 Lacrimal Gland
        [1301],                          # 11 Submandibular Gland
        [1401],                          # 12 Parotid Gland
        [1501],                          # 13 Sublingual Gland
        #[1601],                          # Blood Vessels
        #[1801],                           # Genito-Uninary Tract
        [102, 103, 104, 202, 203, 204, 302, 303, 304, 402, 403, 404, 502, 503, 504,
         602, 603, 604, 702, 703, 704, 802, 803, 804, 902, 903, 904, 1002, 1003, 1004,
         1102, 1103, 1104, 1202, 1203, 1204, 1302, 1303, 1304, 1402, 1403, 1404, 1502, 1503, 1504,
         1602, 1603, 1604, 1802, 1803, 1804]  # 14 Lesions
    ]
    if not include_lesions:
        grouping_table = grouping_table[:-1]
    label_out = np.zeros_like(lbl)
    for idx, group in enumerate(grouping_table):
        for seg_i in group:
            label_out[lbl == seg_i] = idx + 1
    return label_out

class RunningStats3D:
    def __init__(self, shape, device='cpu', dtype=torch.float32):
        self.count = torch.zeros(shape, dtype=torch.int32, device=device)
        self.mean  = torch.zeros(shape, dtype=dtype, device=device)
        self.M2    = torch.zeros(shape, dtype=dtype, device=device)

    @torch.no_grad()
    def update(self, x, mask=None):
        """
        x:    torch.Tensor [H,W,D] on CPU
        mask: bool tensor [H,W,D] on CPU (True where valid). If None, all valid.
        """
        # Only update finite voxels
        finite = torch.isfinite(x)
        if mask is None:
            m = finite
        else:
            m = finite & mask

        if not m.any():
            return

        inc = m.to(torch.int32)
        # Use full-field formulas but only write where m==True
        n_prev = self.count.to(torch.float32)
        n_new  = n_prev + inc.to(torch.float32)

        delta  = torch.where(m, x - self.mean, torch.zeros_like(self.mean))
        mean_new = self.mean + delta / torch.clamp(n_new, min=1.0)
        delta2 = torch.where(m, x - mean_new, torch.zeros_like(self.mean))
        M2_new = self.M2 + delta * delta2

        # Commit updates only where mask is true
        self.mean = torch.where(m, mean_new, self.mean)
        self.M2   = torch.where(m, M2_new,   self.M2)
        self.count += inc

    @torch.no_grad()
    def finalize_std(self, eps=1e-6):
        # unbiased variance with protection for small counts
        denom = torch.clamp(self.count.to(torch.float32) - 1.0, min=1.0)
        var = self.M2 / denom
        std = torch.sqrt(torch.clamp(var, min=0.0) + eps)
        return self.mean, std, self.count

@torch.no_grad()
def zscore_for_patient(case, mean_map, std_map):
    """
    case: a batch dict from your dataset for a single patient
    Returns z-score volume in atlas space as torch.Tensor [H,W,D] on CPU
    """

    def_ct_hu, def_suv_ph = case

    body_mask = (def_suv_ph[0,0] > 0).cuda()
    suv_cpu   = def_suv_ph[0,0].cuda()
    mask_cpu  = body_mask.cuda()

    #mean_map = nib.load(mean_path).get_fdata().astype(np.float32)
    #std_map  = nib.load(std_path).get_fdata().astype(np.float32)
    mean_t = torch.from_numpy(mean_map).cuda()
    std_t  = torch.from_numpy(std_map).cuda()

    eps = 1e-6
    z = torch.zeros_like(suv_cpu)
    denom = torch.clamp(std_t, min=eps)
    z[mask_cpu] = (suv_cpu[mask_cpu] - mean_t[mask_cpu]) / denom[mask_cpu]
    z[~mask_cpu] = 0.0
    return z  # [H,W,D] on CPU

from scipy.stats import t as t_dist, norm  # needs SciPy installed

@torch.no_grad()
def zscore_for_patient2(
    case,
    mean_map,
    std_map,
    *,
    occ_map=None,       # optional [H,W,D] in [0,1]
    occ_thr=0.7,        # occupancy threshold
    erode_iters=1,      # morphological erosion iters
    eps=1e-6,
    mode="gaussian",    # "gaussian" or "student_t"
    nu=3.0,             # df for Student-t
    outside_value=0.0,  # fill value outside mask (use np.nan if preferred)
    to_cpu=True,
):
    """
    case: tuple (def_ct_hu, def_suv_ph), each [1,1,H,W,D] already warped to atlas space
    mean_map, std_map: numpy arrays or torch tensors shaped [H,W,D]
    Returns:
        dict with keys:
          - "z": Gaussian z-map (if mode="gaussian") or Gaussian-equivalent z (if "student_t")
          - "t": t-map (only when mode="student_t")
          - "p_two": two-sided p-map (only when mode="student_t")
          - "mask": final boolean mask [H,W,D]
    """
    def_ct_hu, def_suv_ph = case
    device = def_suv_ph.device

    # Body mask from HU > -200 (atlas space)
    body = (def_suv_ph[0,0] > 0).cuda()

    # Combine with occupancy if provided
    mask = body.clone()
    if occ_map is not None:
        occ_t = torch.as_tensor(occ_map, device=device, dtype=torch.float32)
        mask &= (occ_t >= float(occ_thr))

    # Optional 1-voxel erosion to avoid boundary artifacts
    if erode_iters and erode_iters > 0:
        m = (~mask).float().unsqueeze(0).unsqueeze(0)
        for _ in range(int(erode_iters)):
            m = torch.nn.functional.max_pool3d(m, kernel_size=3, stride=1, padding=1)
        mask = ~(m.squeeze(0).squeeze(0) > 0)

    # Inputs on device
    x   = def_suv_ph[0, 0]
    mu  = torch.as_tensor(mean_map, device=device, dtype=torch.float32)
    sd  = torch.as_tensor(std_map,  device=device, dtype=torch.float32).clamp_min(eps)

    # Exclude invalid params
    finite = torch.isfinite(mu) & torch.isfinite(sd) & (sd > 0)
    mask &= finite

    # Allocate outputs
    z = torch.full_like(x, fill_value=float(outside_value))
    out = {"mask": mask if not to_cpu else mask.detach().cpu()}

    if mode.lower() == "gaussian":
        z[mask] = (x[mask] - mu[mask]) / sd[mask]
        out["z"] = z.detach().cpu() if to_cpu else z
        return out

    # ---- Student-t path (SciPy for p and Gaussian-equivalent z) ----
    t_map = torch.zeros_like(x)
    t_map[mask] = (x[mask] - mu[mask]) / sd[mask]

    # Two-sided p-values on CPU via SciPy (only over valid voxels)
    t_np = t_map[mask].detach().cpu().numpy()
    p_np = 2.0 * (1.0 - t_dist.cdf(np.abs(t_np), df=float(nu)))

    # Gaussian-equivalent z: sign(t) * Phi^{-1}(1 - p/2)
    z_abs_np = norm.ppf(1.0 - p_np / 2.0)
    z_map = torch.full_like(x, fill_value=float(outside_value))
    z_map[mask] = torch.from_numpy(z_abs_np).to(x.dtype).to(device) * torch.sign(t_map[mask])

    p_map = torch.ones_like(x, dtype=torch.float32)
    p_map[mask] = torch.from_numpy(p_np).to(torch.float32).to(device)

    out["t"]     = t_map.detach().cpu() if to_cpu else t_map
    out["z"]     = z_map.detach().cpu() if to_cpu else z_map
    out["p_two"] = p_map.detach().cpu() if to_cpu else p_map
    return out

def gaussian_kernel3d(sigma_vox, size_vox):
    ax = [torch.arange(-(s//2), s//2+1, dtype=torch.float32) for s in size_vox]
    zz, yy, xx = torch.meshgrid(ax[0], ax[1], ax[2], indexing='ij')
    g = torch.exp(-(xx**2+yy**2+zz**2)/(2*sigma_vox**2))
    g = g / g.sum()
    return g

# ---- helper: smooth inside mask to avoid edge bleed ----
def smooth_inside_mask(vol, m_float, sigma=0.85):
    """
    Gaussian smoothing inside mask with NaN-awareness (normalized convolution).
    - vol: torch.Tensor [H,W,D], may contain NaNs.
    - m_float: torch.Tensor [H,W,D], mask in {0,1} (float).
    - sigma: Gaussian sigma in voxels. Kernel size is set to ~6*sigma+1.
    Returns a float volume with NaNs locally averaged (if neighbors exist) and
    no bleed outside mask.
    """
    # Pick kernel size based on sigma (at least 5 and odd)
    rad = max(2, int(math.ceil(3.0 * float(sigma))))
    ksz = 2 * rad + 1
    k = gaussian_kernel3d(float(sigma), (ksz, ksz, ksz)).to(vol.device)[None, None, ...]
    # Build finite-data weight to avoid NaN propagation
    finite = torch.isfinite(vol)
    w = (finite & (m_float > 0)).float()
    v = torch.where(finite, vol, torch.zeros_like(vol))
    num = F.conv3d((v * w)[None, None, ...], k, padding="same")
    den = F.conv3d(w[None, None, ...], k, padding="same").clamp_min(1e-6)
    out = (num / den)[0, 0]
    return out

# ---- helper: morphological closing to fill small holes in a binary mask ----
def morph_close3d(mask_bool: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """
    Binary morphological closing (dilation then erosion) on a 3D boolean mask.
    Uses max-pooling to implement dilation and erosion efficiently on GPU.
    radius: number of voxels for the structuring element (kernel size = 2*radius+1).
    Returns bool mask of the same shape/device.
    """
    if radius is None or radius <= 0:
        return mask_bool
    k = int(2 * radius + 1)
    m = mask_bool.float()[None, None, ...]
    # Dilation
    dil = F.max_pool3d(m, kernel_size=k, stride=1, padding=radius)
    # Erosion via dilation of the complement: erode(A) = not dilate(not A)
    er = 1.0 - F.max_pool3d(1.0 - dil, kernel_size=k, stride=1, padding=radius)
    return (er[0, 0] > 0.5)

def save_masked_map(vec_masked, name, affine, shape, mask_bool3d):
    """
    vec_masked: tensor of shape (Vmask,)
    mask_bool3d: bool numpy or torch array of shape (H,W,D) with True inside body
    """
    H, W, D = shape
    out = torch.zeros(H*W*D, dtype=vec_masked.dtype)
    mv = torch.as_tensor(mask_bool3d).reshape(-1).bool()
    out[mv] = vec_masked.detach().cpu()
    vol = out.reshape(H, W, D)
    nib.save(nib.Nifti1Image(vol.numpy(), affine), name)
    
# ---------- Build baseline design ----------

def to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t

def safe_median_torch(x):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.median(x)

# ---- helper: nan-robust stats for older PyTorch ----
def torch_nanmean(x: torch.Tensor, dim=None, keepdim=False):
    mask = torch.isfinite(x)
    safe = torch.where(mask, x, torch.zeros(1, dtype=x.dtype, device=x.device))
    if dim is None:
        count = mask.sum().clamp_min(1)
        return safe.sum() / count
    count = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1)
    return safe.sum(dim=dim, keepdim=keepdim) / count

def torch_nanstd(x: torch.Tensor, dim=None, keepdim=False):
    if dim is None:
        m = torch_nanmean(x)
        mask = torch.isfinite(x)
        diff2 = torch.where(mask, (x - m)**2, torch.zeros_like(x))
        count = mask.sum().clamp_min(1)
        var = diff2.sum() / count
        return torch.sqrt(var.clamp_min(0))
    m = torch_nanmean(x, dim=dim, keepdim=True)
    mask = torch.isfinite(x)
    diff2 = torch.where(mask, (x - m)**2, torch.zeros_like(x))
    count = mask.sum(dim=dim, keepdim=True).clamp_min(1)
    var = diff2.sum(dim=dim, keepdim=True) / count
    if not keepdim:
        var = var.squeeze(dim)
    return torch.sqrt(var.clamp_min(0))

def zscore_inplace(X, cont_cols):
    if X.numel() == 0:
        return X, torch.zeros(len(cont_cols)), torch.ones(len(cont_cols))
    # use nan-robust helpers
    means = torch_nanmean(X[:, cont_cols], dim=0)
    stds  = torch_nanstd(X[:, cont_cols],  dim=0).clamp_min(1e-6)
    Xz = X.clone()
    for j, c in enumerate(cont_cols):
        col = Xz[:, c]
        col = torch.where(torch.isfinite(col), col, means[j])
        Xz[:, c] = (col - means[j]) / stds[j]
    return Xz, means, stds

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[0], grid_step):
        grid_img[j+line_thickness-1, :, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def make_affine_from_pixdim(pixdim):
    # Create a 4x4 affine with spacing along the diagonal
    affine = np.eye(4)
    affine[0, 0] = pixdim[0]
    affine[1, 1] = pixdim[1]
    affine[2, 2] = pixdim[2]
    return affine

# ---------- Save maps ----------
def save_vector_as_vol(vec, name, affine, shape, mv_bool):
    H, W, D = shape
    out = np.zeros(H*W*D, dtype=np.float32)
    out[mv_bool.reshape(-1)] = vec.astype(np.float32)
    vol = out.reshape(H, W, D)
    nib.save(nib.Nifti1Image(vol, affine), name)
    
def save_vol(vol, name, affine):
    nib.save(nib.Nifti1Image(vol, affine), name)