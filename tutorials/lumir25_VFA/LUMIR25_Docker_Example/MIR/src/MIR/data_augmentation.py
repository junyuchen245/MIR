import math, random
import numpy as np
import torch.nn.functional as F
import torch
from torch import Tensor
import matplotlib.pyplot as plt
def flip_aug(im_1, im_2, im_label_1=None, im_label_2=None):
    '''Flip the image along random axis'''
    with torch.no_grad():
        x_buffer = np.random.choice([True, False])
        y_buffer = np.random.choice([True, False])
        z_buffer = np.random.choice([True, False])
        if x_buffer:
            im_1 = torch.flip(im_1, dims=[2,])
            im_2 = torch.flip(im_2, dims=[2,])
            if im_label_1 is not None and im_label_2 is not None:
                im_label_1 = torch.flip(im_label_1, dims=[2, ])
                im_label_2 = torch.flip(im_label_2, dims=[2, ])
        if y_buffer:
            im_1 = torch.flip(im_1, dims=[3,])
            im_2 = torch.flip(im_2, dims=[3,])
            if im_label_1 is not None and im_label_2 is not None:
                im_label_1 = torch.flip(im_label_1, dims=[3, ])
                im_label_2 = torch.flip(im_label_2, dims=[3, ])
        if z_buffer:
            im_1 = torch.flip(im_1, dims=[4,])
            im_2 = torch.flip(im_2, dims=[4,])
            if im_label_1 is not None and im_label_2 is not None:
                im_label_1 = torch.flip(im_label_1, dims=[4, ])
                im_label_2 = torch.flip(im_label_2, dims=[4, ])
        if im_label_1 is not None and im_label_2 is not None:
            return im_1, im_2, im_label_1, im_label_2
        else:
            return im_1, im_2

def affine_aug(im, im_label=None, seed=10, angle_range=10, trans_range=0.1, scale_range=0.1):
    '''
    Random affine transformation
    '''
    with torch.no_grad():
        if seed is not None:
            random.seed(seed)
        angle_range = angle_range
        trans_range = trans_range
        scale_range = scale_range

        angle_xyz = (random.uniform(-angle_range, angle_range) * math.pi / 180,
                     random.uniform(-angle_range, angle_range) * math.pi / 180,
                     random.uniform(-angle_range, angle_range) * math.pi / 180)
        scale_xyz = (random.uniform(-scale_range, scale_range), random.uniform(-scale_range, scale_range),
                     random.uniform(-scale_range, scale_range))
        trans_xyz = (random.uniform(-trans_range, trans_range), random.uniform(-trans_range, trans_range),
                     random.uniform(-trans_range, trans_range))

        rotation_x = torch.tensor([
            [1., 0, 0, 0],
            [0, math.cos(angle_xyz[0]), -math.sin(angle_xyz[0]), 0],
            [0, math.sin(angle_xyz[0]), math.cos(angle_xyz[0]), 0],
            [0, 0, 0, 1.]
        ], requires_grad=False).unsqueeze(0).cuda()

        rotation_y = torch.tensor([
            [math.cos(angle_xyz[1]), 0, math.sin(angle_xyz[1]), 0],
            [0, 1., 0, 0],
            [-math.sin(angle_xyz[1]), 0, math.cos(angle_xyz[1]), 0],
            [0, 0, 0, 1.]
        ], requires_grad=False).unsqueeze(0).cuda()

        rotation_z = torch.tensor([
            [math.cos(angle_xyz[2]), -math.sin(angle_xyz[2]), 0, 0],
            [math.sin(angle_xyz[2]), math.cos(angle_xyz[2]), 0, 0],
            [0, 0, 1., 0],
            [0, 0, 0, 1.]
        ], requires_grad=False).unsqueeze(0).cuda()

        trans_shear_xyz = torch.tensor([
            [1. + scale_xyz[0], 0, 0, trans_xyz[0]],
            [0, 1. + scale_xyz[1], 0, trans_xyz[1]],
            [0, 0, 1. + scale_xyz[2], trans_xyz[2]],
            [0, 0, 0, 1]
        ], requires_grad=False).unsqueeze(0).cuda()

        theta_final = torch.matmul(rotation_x, rotation_y)
        theta_final = torch.matmul(theta_final, rotation_z)
        theta_final = torch.matmul(theta_final, trans_shear_xyz)

        output_disp_e0_v = F.affine_grid(theta_final[:, 0:3, :], im.shape, align_corners=False)

        im = F.grid_sample(im, output_disp_e0_v, mode='bilinear', padding_mode="border", align_corners=False)

        if im_label is not None:
            im_label = F.grid_sample(im_label, output_disp_e0_v, mode='nearest', padding_mode="border",
                                     align_corners=False)
            return im, im_label
        else:
            return im


# --------------------------------------------------------------------------
# helper 1 :   1‑D three‑class Gaussian mixture (EM, 20 iters)
# --------------------------------------------------------------------------
def gmm_soft_masks(brain_vox, iters: int = 40):
    """
    brain_vox : flat 1‑D tensor of brain intensities in [0,1]
    returns   : soft posterior masks (..., 3) ordered CSF, GM, WM
    """
    x = brain_vox[:, None]                       # shape (N,1)

    # --- k‑means init --------------------------------------------------
    centers = torch.linspace(torch.quantile(x, 0.05), torch.quantile(x, 0.95), 3, device=x.device)[:, None]  # (3,1)
    lbl = torch.argmin(torch.cdist(x, centers), dim=1)      # (N,)
    pi  = torch.stack([(lbl == k).float().mean() for k in range(3)])
    mu  = torch.stack([x[lbl == k].mean() for k in range(3)])
    var = torch.stack([x[lbl == k].var()  for k in range(3)]).clamp_min(1e-4)

    N = x.shape[0]
    for _ in range(iters):
        # E‑step --------------------------------------------------------
        pdf = torch.exp(-0.5 * (x - mu) ** 2 / var) / torch.sqrt(2 * torch.pi * var)
        pdf = (pi[:, None] * pdf.T).clamp_min(1e-12)   # (3, N)
        post = pdf / pdf.sum(0, keepdim=True)                   # (K,N)

        # M‑step --------------------------------------------------------
        Nk  = post.sum(1)
        pi  = Nk / N
        mu  = (post * x.T).sum(1) / Nk
        var = (post * (x.T - mu[:, None]) ** 2).sum(1) / Nk
        var = var.clamp_min(1e-4)

    order = torch.argsort(mu)                                   # low→high
    return post[order].T.reshape(brain_vox.shape + (-1,))       # (N,3)

def fcm_soft_masks(
    brain_vox: torch.Tensor,
    n_clusters: int = 3,
    m: float = 2.,
    iters: int = 30,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Fuzzy C-Means on 1-D intensities, k-means init, correct broadcasting.

    brain_vox : Tensor, shape (N,) of [0,1] intensities
    n_clusters: number of tissue classes (3→CSF/GM/WM)
    m         : fuzziness exponent (>1, larger→softer)
    iters     : EM updates
    eps       : small constant to avoid zeros
    Returns U of shape (N, n_clusters) with soft memberships.
    """

    # 1) Prepare data
    x = brain_vox.flatten().to(brain_vox.device)               # (N,)
    N = x.shape[0]
    x = x.unsqueeze(1)                                         # (N,1)

    # 2) Initialize centers by simple k-means on [5%, 95%] quantiles
    qs = torch.quantile(x.flatten(), torch.linspace(0.05, 0.95, n_clusters, device=x.device))
    centers = qs.unsqueeze(0)                                  # (1,K)

    # 3) Initial membership from distance ratios
    #    dist[i,j] = |x[i]-centers[j]|  -> broadcast (N,1)-(1,K)->(N,K)
    dist = torch.abs(x - centers).clamp_min(eps)               # (N,K)
    exponent = 2.0 / (m - 1.0)
    #    ratio[i,j,k] = dist[i,j] / dist[i,k]  -> (N,K,K)
    ratio = dist.unsqueeze(2) / dist.unsqueeze(1)              # (N,K,K)
    denom = (ratio**exponent).sum(dim=2)                        # (N,K)
    U = 1.0 / denom                                             # (N,K)

    # 4) EM iterations
    for _ in range(iters):
        Um = U**m                                             # (N,K)
        # update centers: c[j] = Σ_i Um[i,j]*x[i] / Σ_i Um[i,j]
        centers = (Um * x).sum(dim=0, keepdim=True) / Um.sum(dim=0, keepdim=True).clamp_min(eps)  # (1,K)

        # recompute U
        dist  = torch.abs(x - centers).clamp_min(eps)         # (N,K)
        ratio = dist.unsqueeze(2) / dist.unsqueeze(1)         # (N,K,K)
        denom = (ratio**exponent).sum(dim=2)                   # (N,K)
        U     = 1.0 / denom                                   # (N,K)

    # 5) Sort columns by ascending center value
    centers_flat = centers.flatten()                         # (K,)
    order = torch.argsort(centers_flat)
    U = U[:, order]                                          # (N,K) with cols [CSF, GM, WM]

    return U

def mask_edge(mask: torch.BoolTensor, kernel_size: int = 3) -> torch.BoolTensor:
    """
    Compute the one-voxel edge of a binary mask by exact morphological erosion:
      edge = mask AND NOT(eroded_mask)
    
    Uses max_pool3d on the inverted mask for precise erosion.
    
    Args:
        mask: (D, H, W) bool tensor
        kernel_size: odd int, size of the cubic structuring element
    
    Returns:
        edge: (D, H, W) bool tensor where True indicates the mask boundary
    """
    if mask.dtype != torch.bool:
        raise ValueError("mask must be a BoolTensor")
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel_size must be an odd positive integer")

    # add batch+channel dims for pooling: shape (1,1,D,H,W)
    m = mask.float().unsqueeze(0).unsqueeze(0)

    # invert mask: background=1, foreground=0
    inv = 1.0 - m

    # perform dilation on inverted mask -> erosion on original
    pad = kernel_size // 2
    dilated = F.max_pool3d(inv, kernel_size=kernel_size, stride=1, padding=pad)

    # eroded mask: True where all neighbors in the window were 1 in original
    eroded = (dilated < 1.0).squeeze(0).squeeze(0)  # (D,H,W) bool

    # edge = original mask minus its eroded version
    edge = mask & ~eroded
    return edge[None, None]


def fill_holes_torch(mask: torch.BoolTensor, closing_size: int = 3) -> torch.BoolTensor:
    """
    Remove small gaps and fill internal holes in a 3D brain mask.

    Steps:
    1. Closing (dilation followed by erosion) with a cubic kernel of `closing_size`
    2. Invert the closed mask to get 'holes + background'
    3. Flood-fill from the border (repeated dilation masked by inverted mask)
    4. The remaining True in inverted mask are true holes; add them back to closed mask.

    mask: (D,H,W) bool tensor
    closing_size: odd int, kernel size of the closing operation
    returns: bool tensor (D,H,W) with holes filled
    """

    # 1) morphological closing
    m = mask.float()
    k = closing_size
    pad = k//2
    # 1a) dilation
    dil = F.max_pool3d(m, kernel_size=k, stride=1, padding=pad)
    # 1b) erosion via dilation of the inverse
    inv = 1.0 - dil
    er_inv = F.max_pool3d(inv, kernel_size=k, stride=1, padding=pad)
    closed = (1.0 - (er_inv>0).float())[0,0].bool()  # back to (D,H,W)

    # 2) invert to get potential holes + true background
    inv_closed = ~closed

    # 3) prepare a “seed” for flood-fill: only the border of inv_closed
    outside = torch.zeros_like(inv_closed)
    # set all six faces:
    outside[0,...]   = inv_closed[0,...]
    outside[-1,...]  = inv_closed[-1,...]
    outside[:,0,:]   = inv_closed[:,0,:]
    outside[:,-1,:]  = inv_closed[:,-1,:]
    outside[:,:,0]   = inv_closed[:,:,0]
    outside[:,:,-1]  = inv_closed[:,:,-1]

    # 4) morphological reconstruction (flood-fill) of outside through inv_closed
    prev = torch.zeros_like(outside)
    curr = outside.clone()
    while True:
        # dilate curr by one voxel and mask by inv_closed
        d = F.max_pool3d(curr.float()[None,None], 3, 1, 1)[0,0].bool()
        d = d & inv_closed
        if torch.equal(d, curr):
            break
        curr = d

    # 5) holes are those parts of inv_closed *not* reached by flood
    holes = inv_closed & ~curr

    # 6) final mask = closed OR holes
    filled = closed | holes
    return filled[None, None]


class SimulatedWidthThickness(torch.nn.Module):
    """
    Simulate coarse resolution along the W (width) dimension by:
      1) Averaging adjacent groups of columns ('thickness' factor)
      2) Upsampling back to original width via trilinear interpolation

    Args:
        min_factor: Minimum integer factor of width down-sampling (e.g., 2)
        max_factor: Maximum integer factor of width down-sampling (e.g., 5)
        p: Probability of applying the augmentation
    """
    def __init__(self, min_factor=2, max_factor=5, p=0.5):
        super().__init__()
        if min_factor < 1 or max_factor < min_factor:
            raise ValueError("Invalid factor range")
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, D, H, W)
        Returns:
            Tensor of same shape with simulated coarse width resolution
        """
        if not self.training or random.random() > self.p:
            return x

        B, C, D, H, W = x.shape
        # Choose a random down-sampling factor
        factor = random.randint(self.min_factor, self.max_factor)

        # Compute padding so width is divisible by factor
        pad_w = (factor - (W % factor)) % factor
        # F.pad expects (Wl, Wr, Hl, Hr, Dl, Dr)
        x_padded = F.pad(x, (0, pad_w, 0, 0, 0, 0))
        W_padded = W + pad_w

        # Pool across the width dimension only
        pooled = F.avg_pool3d(
            x_padded,
            kernel_size=(1, 1, factor),
            stride=(1, 1, factor),
            padding=0
        )  # shape -> (B, C, D, H, W_padded/factor)

        # Upsample back to original width
        x_coarse = F.interpolate(
            pooled,
            size=(D, H, W),
            mode='trilinear',
            align_corners=False
        )
        return x_coarse

def random_contrast_lut(
    img: torch.Tensor,
    num_bins: int    = 128,
    sigma_bins: float= 128//10,
    chunk_size: int  = 64,
    use_half: bool   = True
) -> torch.Tensor:
    """
    Memory-efficient extreme-contrast LUT for [0,1] images.
    Works on 2D ([B,C,H,W]) or 3D ([B,C,D,H,W]) fully in PyTorch.
    """

    device = img.device
    dtype  = img.dtype
    idx_dtype = torch.float16 if use_half else torch.float32

    # 1) build & smooth LUT
    lut = torch.rand(num_bins, device=device)
    
    k = int(6 * sigma_bins + 1) | 1
    xs = torch.arange(k, device=device) - (k//2)
    gauss = torch.exp(-0.5 * (xs / sigma_bins)**2)
    gauss /= gauss.sum()
    lut = lut.view(1,1,-1)
    gauss = gauss.view(1,1,-1)
    lut_smooth = F.conv1d(F.pad(lut, (k//2,)*2), gauss).view(-1)
    # after you compute lut_smooth (before you apply it) do:
    lut_smooth = (lut_smooth - lut_smooth.min()) \
                / (lut_smooth.max() - lut_smooth.min())
    # enforce endpoints
    lut_smooth[0]           = 0.0
    lut_smooth[num_bins-1]  = 1.0
    if use_half:
        lut_smooth = lut_smooth.half()
    
    out = torch.empty_like(img)
    
    # 2) process in chunks along the H-axis
    H = img.shape[-2]
    for i0 in range(0, H, chunk_size):
        i1 = min(i0 + chunk_size, H)
        patch = img[..., i0:i1, :]               # [..., H_chunk, W]
        idx   = (patch * (num_bins - 1)).to(idx_dtype)
        lo    = idx.floor().long().clamp(0, num_bins-2)
        hi    = lo + 1
        w     = (idx - lo.to(idx_dtype))         # same shape as patch

        v_lo  = lut_smooth[lo]                   # same shape
        v_hi  = lut_smooth[hi]

        # now all three tensors are exactly the same shape
        mapped = v_lo + (v_hi - v_lo) * w

        out[..., i0:i1, :] = mapped.to(dtype)

    return out

class RandomMultiContrastRemap(torch.nn.Module):
    """
    Augment a skull‑stripped T1‑w brain MRI so that the same anatomy can resemble
    T2, FLAIR, T2*, or CT.  Background stays zero; brain voxels are re‑normalized
    to [0,1] at the end.

    Args:
        p: Probability of applying the augmentation
        noise_std: Standard deviation of Gaussian noise added to the image
        downsample_p: Probability of applying simulated coarse width resolution
    """

    def __init__(self, p=0.9, noise_std=0.03, downsample_p=0.5, recipe=["t2", "flair", "t2star", "pd", "ct", "random_remap", "random_lut"]):
        super().__init__()
        self.p = p
        self.noise_std = noise_std
        self.slice_down = SimulatedWidthThickness(4, 8, p=downsample_p)
        self.recipe_list = recipe
    # ──────────────────────────────────────────────────────────────────
    # Helper utilities
    # ──────────────────────────────────────────────────────────────────
    def _gamma_map(self, x, g):                 # monotonic γ curve
        return x.pow(g)

    def _invert(self, x):
        return 1.0 - x

    def _piecewise_linear(self, x, knots, values):
        """
        Safe piece‑wise linear mapping that never indexes out of bounds.
        x, knots, values : 1‑D (knots and values same length, strictly increasing knots)
        """
        idx = torch.searchsorted(knots, x)                 # ∈ [0, len(knots)]
        idx = torch.clamp(idx, max=len(knots) - 1)         # now ∈ [0, len‑1]

        idx0 = torch.clamp(idx - 1, min=0)                 # left neighbour
        x0, x1 = knots[idx0], knots[idx]
        y0, y1 = values[idx0], values[idx]

        t = (x - x0) / (x1 - x0 + 1e-6)                    # linear weight
        return torch.lerp(y0, y1, t)

    def _poly_bias(self, shape, order, device, strength):
        z, y, x = [torch.linspace(-1,1,s,device=device) for s in shape]
        zz, yy, xx = torch.meshgrid(z,y,x, indexing="ij")
        grid = torch.stack([xx,yy,zz])
        field = torch.zeros_like(grid[0])
        for o in range(1, order+1):
            field += torch.randn(1, device=device) * (grid[0]**o + grid[1]**o + grid[2]**o)
        field = 1.0 + strength * field / field.abs().max()
        return field                                # multiplicative bias field

    # ──────────────────────────────────────────────────────────────────
    def forward(self, img_, mask):
        mask = fill_holes_torch(mask, closing_size=5)
        img = img_.clone()  # avoid in‑place on caller’s tensor
        if random.random() > self.p:
            return img * mask.float()               # make sure background is 0

        single = img.dim() == 4
        if single:
            img, mask = img.unsqueeze(0), mask.unsqueeze(0)

        img = img.clone()
        B,C,D,H,W = img.shape
        device = img.device
        mask_bool = mask.bool()

        # ---------------------------------------------------------------- Recipe pool
        recipe = random.choice(self.recipe_list)

        if recipe == "t2":                          # bright CSF, dark WM
            m = mask[0,0].bool()
            vox = img[0,0][m]
            fuzzy_fac = random.uniform(0.4, 0.6)
            post = gmm_soft_masks(vox)**fuzzy_fac
            rngs = [(4,5),(0.3,0.35),(0.15,0.25)]
            targets = [random.uniform(*r) for r in rngs]
            img[0,0][m] = (post * vox[:,None] * torch.tensor(targets,device=vox.device)).sum(1)
            vmin, vmax = torch.quantile(img, 0.05), torch.quantile(img, 0.95)
            img = (img - vmin) / (vmax - vmin) 
            img = torch.clamp(img, 0, 1)
            
        elif recipe == "flair":                     # CSF bright but suppress fat‑like WM
            m = mask[0,0].bool()
            vox = img[0,0][m]
            fuzzy_fac = random.uniform(0.4, 0.6)
            post = gmm_soft_masks(vox)**fuzzy_fac
            rngs = [(0.1,0.1),(0.3,0.3),(0.1,0.1)]
            targets = [random.uniform(*r) for r in rngs]
            img[0,0][m] = (post * vox[:,None] * torch.tensor(targets,device=vox.device)).sum(1)
            vmin, vmax = torch.quantile(img, 0.05), torch.quantile(img, 0.95)
            img = (img - vmin) / (vmax - vmin) 
            img = torch.clamp(img, 0, 1)

        elif recipe == "t2star":                    # susceptibility darkening
            m = mask[0,0].bool()
            vox = img[0,0][m]
            fuzzy_fac = random.uniform(0.4, 0.6)
            post = gmm_soft_masks(vox)**fuzzy_fac
            rngs = [(3,4),(0.7,0.9),(0.3,0.4)]
            targets = [random.uniform(*r) for r in rngs]
            img[0,0][m] = (post * vox[:,None] * torch.tensor(targets,device=vox.device)).sum(1)
            vmin, vmax = torch.quantile(img, 0.05), torch.quantile(img, 0.95)
            img = (img - vmin) / (vmax - vmin) 
            img = torch.clamp(img, 0, 1)

        elif recipe == "ct":                       # linear ramp, WM≈0.3–0.4, GM higher
            m = mask[0,0].bool()
            vox = img[0,0][m]
            fuzzy_fac = random.uniform(0.4, 0.6)
            post = gmm_soft_masks(vox)**fuzzy_fac
            rngs = [(0.06,0.1),(0.06,0.08),(0.025,0.04)]
            targets = [random.uniform(*r) for r in rngs]
            img[0,0][m] = (post * vox[:,None] * torch.tensor(targets,device=vox.device)).sum(1)
            edge = mask_edge(mask[0,0], kernel_size=3)
            thickness = 1#random.randint(1, 2)
            k = 2*thickness + 1
            e = edge.float()
            e = F.max_pool3d(e, kernel_size=k, stride=1, padding=thickness) > 0
            edge = e.squeeze(0).squeeze(0)
            
            channel = img[0,0]
            channel[edge] = 0.1+channel[edge]
            img[0,0] = channel
            vmin, vmax = torch.quantile(img, 0.05), torch.quantile(img, 0.95)
                
            img = (img - vmin) / (vmax - vmin) 
            img = torch.clamp(img, 0, 1)
        
        elif recipe == "pd":                     # proton density
            img[mask_bool] = self._invert(img[mask_bool])          # CSF → bright
            g = random.uniform(1.1, 1.6)                   # flatten GM/WM contrast
            img[mask_bool] = self._gamma_map(img[mask_bool], g)
            knots = torch.tensor([0.0, 0.3, 0.8, 1.0], device=device)
            values = torch.tensor([0.0, 0.35, 0.85, 1.0], device=device)
            img[mask_bool] = self._piecewise_linear(img[mask_bool], knots, values)
            vmin, vmax = torch.quantile(img, 0.05), torch.quantile(img, 0.95)
                
            img = (img - vmin) / (vmax - vmin) 
            img = torch.clamp(img, 0, 1)
            
        elif recipe == "random_remap":             # catch‑all monotonic curve
            m = mask[0,0].bool()
            vox = img[0,0][m]
            fuzzy_fac = random.uniform(0.4, 0.6)
            post = gmm_soft_masks(vox)**fuzzy_fac
            rngs = [(0.01,1),(0.01,1),(0.01,1)]
            targets = [random.uniform(*r) for r in rngs]
            img[0,0][m] = (post * vox[:,None] * torch.tensor(targets,device=vox.device)).sum(1)
            vmin, vmax = torch.quantile(img, 0.05), torch.quantile(img, 0.95)
            img = (img - vmin) / (vmax - vmin) 
            img = torch.clamp(img, 0, 1)
            if random.random() > 0.5:
                img[mask_bool] = self._invert(img[mask_bool])
        
        elif recipe == "random_lut":                 # random LUT
            vmin, vmax = img.min(), img.max()
            img = (img - vmin) / (vmax - vmin) 
            img = torch.clamp(img, 0, 1)
            img = random_contrast_lut(img, num_bins=256)
            vmin, vmax = img.min(), img.max()
            img = (img - vmin) / (vmax - vmin) 
            img = torch.clamp(img, 0, 1)

        if self.noise_std > 0:
            sigma = random.uniform(0, self.noise_std)
            img[mask_bool] += sigma * torch.randn_like(img)[mask_bool]

        # ---------------------------------------------------------------- Final normalize to [0,1], background = 0
        vmin, vmax = torch.quantile(img, 0.01), torch.quantile(img, 0.99)
        img = (img - vmin) / (vmax - vmin) 
        img = torch.clamp(img, 0, 1)
        img[~mask_bool] = 0
        img = torch.nan_to_num(img)
        if single: img = img.squeeze(0)
        if recipe == "t2star" or recipe == "flair":
            img = self.slice_down(img)
        return img
        