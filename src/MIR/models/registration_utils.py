"""Model-level registration utilities (transformers and geometry)."""

import torch
import torch.nn as nn
import torch.nn.functional as nnf

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    Args:
        size: spatial size of the input tensor
        mode: interpolation mode, 'bilinear' or 'nearest'
    """

    def __init__(self, size, mode='bilinear'):
        """Initialize a spatial transformer.

        Args:
            size: Spatial size tuple (H, W[, D]).
            mode: Interpolation mode ('bilinear' or 'nearest').
        """
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        """Warp a source tensor with a displacement field.

        Args:
            src: Source tensor (B, C, ...).
            flow: Displacement field (B, ndim, ...).

        Returns:
            Warped tensor.
        """
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=False, mode=self.mode)
    
class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    Args:
        inshape: shape of the input tensor
        nsteps: number of integration steps
    """

    def __init__(self, inshape, nsteps):
        """Initialize vector field integrator.

        Args:
            inshape: Spatial shape of the field.
            nsteps: Number of scaling-and-squaring steps.
        """
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        """Integrate a vector field via scaling and squaring.

        Args:
            vec: Velocity field tensor (B, ndim, ...).

        Returns:
            Integrated displacement field.
        """
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class AffineTransformer(nn.Module):
    """
    3-D Affine Transformer
    Args:
        mode: interpolation mode, 'bilinear' or 'nearest'
    """

    def __init__(self, mode='bilinear'):
        """Initialize affine transformer.

        Args:
            mode: Interpolation mode ('bilinear' or 'nearest').
        """
        super().__init__()
        self.mode = mode

    def apply_affine(self, src, mat):
        """Apply an affine matrix to a source volume.

        Args:
            src: Source tensor (B, C, H, W, D).
            mat: Affine matrix (B, 3, 4).

        Returns:
            Warped tensor.
        """
        grid = nnf.affine_grid(mat, [src.shape[0], 3, src.shape[2], src.shape[3], src.shape[4]], align_corners=False)
        return nnf.grid_sample(src, grid, align_corners=False, mode=self.mode)

    def forward(self, src, affine, scale, translate, shear):
        """Apply composed affine parameters to a volume.

        Args:
            src: Source tensor (B, C, H, W, D).
            affine: Rotation parameters (B, 3).
            scale: Scale parameters (B, 3).
            translate: Translation parameters (B, 3).
            shear: Shear parameters (B, 6).

        Returns:
            Tuple of (warped, affine_matrix, inverse_affine_matrix).
        """

        theta_x = affine[:, 0]
        theta_y = affine[:, 1]
        theta_z = affine[:, 2]
        scale_x = scale[:, 0]
        scale_y = scale[:, 1]
        scale_z = scale[:, 2]
        trans_x = translate[:, 0]
        trans_y = translate[:, 1]
        trans_z = translate[:, 2]
        shear_xy = shear[:, 0]
        shear_xz = shear[:, 1]
        shear_yx = shear[:, 2]
        shear_yz = shear[:, 3]
        shear_zx = shear[:, 4]
        shear_zy = shear[:, 5]

        rot_mat_x = torch.stack([torch.stack([torch.ones_like(theta_x), torch.zeros_like(theta_x), torch.zeros_like(theta_x)], dim=1), torch.stack([torch.zeros_like(theta_x), torch.cos(theta_x), -torch.sin(theta_x)], dim=1), torch.stack([torch.zeros_like(theta_x), torch.sin(theta_x), torch.cos(theta_x)], dim=1)], dim=2).cuda()
        rot_mat_y = torch.stack([torch.stack([torch.cos(theta_y), torch.zeros_like(theta_y), torch.sin(theta_y)], dim=1), torch.stack([torch.zeros_like(theta_y), torch.ones_like(theta_x), torch.zeros_like(theta_x)], dim=1), torch.stack([-torch.sin(theta_y), torch.zeros_like(theta_y), torch.cos(theta_y)], dim=1)], dim=2).cuda()
        rot_mat_z = torch.stack([torch.stack([torch.cos(theta_z), -torch.sin(theta_z), torch.zeros_like(theta_y)], dim=1), torch.stack([torch.sin(theta_z), torch.cos(theta_z), torch.zeros_like(theta_y)], dim=1), torch.stack([torch.zeros_like(theta_y), torch.zeros_like(theta_y), torch.ones_like(theta_x)], dim=1)], dim=2).cuda()
        scale_mat = torch.stack(
            [torch.stack([scale_x, torch.zeros_like(theta_z), torch.zeros_like(theta_y)], dim=1),
             torch.stack([torch.zeros_like(theta_z), scale_y, torch.zeros_like(theta_y)], dim=1),
             torch.stack([torch.zeros_like(theta_y), torch.zeros_like(theta_y), scale_z], dim=1)], dim=2).cuda()
        shear_mat = torch.stack(
            [torch.stack([torch.ones_like(theta_x), torch.tan(shear_xy), torch.tan(shear_xz)], dim=1),
             torch.stack([torch.tan(shear_yx), torch.ones_like(theta_x), torch.tan(shear_yz)], dim=1),
             torch.stack([torch.tan(shear_zx), torch.tan(shear_zy), torch.ones_like(theta_x)], dim=1)], dim=2).cuda()
        trans = torch.stack([trans_x, trans_y, trans_z], dim=1).unsqueeze(dim=2)
        mat = torch.bmm(shear_mat, torch.bmm(scale_mat, torch.bmm(rot_mat_z, torch.matmul(rot_mat_y, rot_mat_x))))
        inv_mat = torch.inverse(mat)
        mat = torch.cat([mat, trans], dim=-1)
        inv_trans = torch.bmm(-inv_mat, trans)
        inv_mat = torch.cat([inv_mat, inv_trans], dim=-1)
        grid = nnf.affine_grid(mat, [src.shape[0], 3, src.shape[2], src.shape[3], src.shape[4]], align_corners=False)
        return nnf.grid_sample(src, grid, align_corners=False, mode=self.mode), mat, inv_mat

def _to_tensor(arr, device):
    """Convert array to float tensor on a device.

    Args:
        arr: NumPy array or tensor.
        device: Target torch device.

    Returns:
        Tensor on device.
    """
    if isinstance(arr, torch.Tensor):
        return arr.to(device).float()
    return torch.from_numpy(arr).to(device).float()
 
def jacobian_determinant(disp):
    """
    Compute the Jacobian determinant of a displacement field.
    disp: (B, C, *vol_shape) tensor with C = spatial dims (2 or 3)
    Returns detJ: (B, 1, *vol_shape)
    """
    dims = disp.shape[1]
    g = torch.gradient if hasattr(torch, "gradient") else _torch_gradient
 
    # ∂u_i/∂x_j for all i,j
    grads = [g(disp[:, i, ...], spacing=1.0)[j]               # d u_i / d x_j
             for i in range(dims) for j in range(dims)]
    J = torch.stack(grads, dim=0).reshape(dims, dims, *disp.shape[2:])
 
    if dims == 2:
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    else:  # 3‑D
        a = J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1])
        b = J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0])
        c = J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0])
        det = a - b + c
 
    return det.unsqueeze(1)   # (B,1,*)
 
 
def _torch_gradient(x, spacing=1):
    """Fallback torch gradient for older PyTorch versions.

    Args:
        x: Tensor to differentiate.
        spacing: Grid spacing.

    Returns:
        List of gradients per spatial dimension.
    """
    # minimal torch‑gradient fallback for < 2.1
    grads = []
    for dim in range(x.ndim - 2):
        d = dim + 2
        pad = [0, 0] * (x.ndim - d - 1) + [1, 1]
        g = nnf.pad(x, pad, mode='replicate')  # Neumann
        fwd = g.movedim(d, -1)[..., 2:]
        bwd = g.movedim(d, -1)[..., :-2]
        grads.append((fwd - bwd) / (2 * spacing))
    return grads
 
def fit_warp_to_svf(
    warp_t,
    nb_steps: int = 7,
    iters: int = 500,
    min_delta: float = 1e-5,
    lr: float = 0.1,
    objective: str = "mse",  # "mse" | "l1"
    init: str = "warp",          # "warp" | "jacwt"
    output_type: str = "disp",  # "svf" | "disp"
    verbose: bool = True,
    device: str = 'cpu',):
    """
    Fit a stationary‑velocity field v so that exp(v) ≈ given displacement field.
    Parameters mirror the original TF implementation.
    `warp` shape: (*vol_shape, ndims)  (numpy array or torch tensor)
    Returns: v as a numpy array of same shape.
    """
    device = warp_t.device
 
    vol_shape, ndims = warp_t.shape[2:], warp_t.shape[1]
 
    # -- optimisation variable ------------------
    vel = nn.Parameter(torch.zeros_like(warp_t, device=device))
 
    # init velocity
    if init == "warp":
        vel.data.copy_(warp_t)
    elif init == "jacwt":
        jac = jacobian_determinant(warp_t)                # (1,1,*)
        jac3 = jac.repeat(1, ndims, *[1] * len(vol_shape))
        vel.data.copy_(1.02 * warp_t + (1.0 - jac3) * 0.05)
 
    # integrator (re‑use your VecInt)
    integrator = VecInt(vol_shape, nb_steps).to(device)
 
    opt = torch.optim.Adam([vel], lr=lr)
    last_loss = None
    if objective == "mse":
        criterion = nnf.mse_loss
    elif objective == "l1":
        criterion = nnf.l1_loss
    for epoch in range(iters):
        opt.zero_grad(set_to_none=True)
        disp_pred = integrator(vel)                        # exp(v)
        loss = criterion(disp_pred, warp_t)
        loss.backward()
        opt.step()
 
        # early stop
        if last_loss is not None and abs(last_loss - loss.item()) < min_delta:
            if verbose:
                print(f"Converged @ {epoch:4d}  loss={loss.item():.4e}")
            break
        last_loss = loss.item()
 
        if verbose and epoch % 20 == 0:
            print(f"[{epoch:03d}]  loss={loss.item():.4e}")
    if output_type == "disp":
        return vel
    elif output_type == "svf":
        return disp_pred
    else:
        raise ValueError(f"unknown output_type {output_type}, expected 'disp' or 'svf'")
 
 
# -----------------------  inverse via velocity  --------------------------- #
 
def invert_warp_via_velocity(
    warp,
    nb_steps: int = 5,
    iters: int = 100,
    **kwargs,):
    """
    Approximate inverse by: fit v, then integrate ‑v.
    Returns a displacement field (numpy) of same shape.
    """
    vel = fit_warp_to_svf(warp, nb_steps, iters, **kwargs)
 
    # integrate ‑v
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inv_disp = VecInt(vel.shape[2:], nb_steps).to(device)(-vel)
    return inv_disp

def ensure_svf(field, field_type, nb_steps=5, iters=100, **fit_kw):
    """
    field: numpy or torch, shape (*vol, ndims)
    field_type: 'svf'  (already velocity)  |  'disp'  (plain displacement)
    Returns v as numpy, shape (*vol, ndims)
    """
    if field_type == 'svf':
        return field                     # nothing to do
    elif field_type == 'disp':
        # fit_warp_to_svf defined earlier
        return fit_warp_to_svf(field, nb_steps=nb_steps, iters=iters, **fit_kw)
    else:
        raise ValueError(f"unknown field_type {field_type}")
    
def ensemble_average(
    fields,                   # list of np arrays, each (*vol, ndims)
    field_types,              # parallel list, 'svf' or 'disp'
    weights=None,             # list of floats or None  (defaults to equal)
    nb_steps=5,
    iters=100,
    device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Ensemble average of displacement fields or SVFs.
    Args:
        fields: list of displacement fields, each (*vol, ndims)
        field_types: parallel list, 'svf' or 'disp'
        weights: list of floats or None (defaults to equal)
        nb_steps: number of integration steps for VecInt
        iters: number of iterations for fit_warp_to_svf
    Returns:
        disp_bar: numpy displacement field (*vol, ndims) = exp(Σ w_i v_i)
    
    # assume you already have three predictions:
    svf_A   = np.load("model_A_velocity.npy")    # (*vol, 3)
    disp_B  = np.load("model_B_disp.npy")        # (*vol, 3)
    disp_C  = np.load("model_C_disp.npy")
    
    fields      = [svf_A, disp_B, disp_C]
    field_types = ['svf', 'disp', 'disp']
    # e.g. let Dice on a small val‑set decide the weights
    weights     = [0.45, 0.35, 0.20]
    
    disp_mean = ensemble_average(
        fields,
        field_types,
        weights,
        nb_steps=7,          # same nsteps you used during training/inference
        iters=150,           # iterations for fit_warp_to_svf on the disp fields
    )
    
    # now warp any tensor (image, mask, logits) with the consensus deformation
    vol_shape = disp_mean.shape[:-1]
    transform = SpatialTransformer(vol_shape).cuda()         # if GPU used above
    moving_img_t = torch.from_numpy(moving_img)[None, None].float().cuda()
    disp_t = torch.from_numpy(disp_mean).permute(3,0,1,2)[None].float().cuda()
    warped = transform(moving_img_t, disp_t)                 # (1,1,*vol)
    """
    assert len(fields) == len(field_types), "length mismatch"
    N = len(fields)
    if weights is None:
        weights = [1.0 / N] * N
    else:
        wsum = sum(weights)
        weights = [w / wsum for w in weights]            # normalize to 1
 
    # 1. make sure everything is an SVF
    v_list = [
        ensure_svf(f, t, nb_steps=nb_steps, iters=iters)  # returns (*vol, ndims)
        for f, t in zip(fields, field_types)
    ]
 
    # 2. stack tensors (B,C,*) for averaging
    v_stack = torch.stack([
        v[0]
        for v in v_list
    ], dim=0).float().to(device)                         # (N,C,*)
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    v_bar = torch.sum(w[:, None, None, None, None] * v_stack, dim=0, keepdim=True)
    # v_bar shape (1,C,*)
 
    # 3. exponentiate once → displacement
    integrator = VecInt(v_bar.shape[2:], nb_steps).to(device)
    disp_bar = integrator(v_bar).squeeze(0)              # (C,*)
 
    return disp_bar