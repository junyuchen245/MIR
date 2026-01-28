"""Visualization helpers for grids and colormaps."""

import numpy as np
import torch
import matplotlib.pyplot as plt

def mk_grid_img(grid_step=8, line_thickness=1, grid_sz=(160, 192, 224), dim=0):
    """Create a grid image tensor for visualization.

    Args:
        grid_step: Spacing between grid lines.
        line_thickness: Grid line thickness.
        grid_sz: Grid size (H, W, D).
        dim: Axis along which to draw the grid.

    Returns:
        Grid tensor with shape (1, 1, H, W, D).
    """
    grid_img = np.zeros(grid_sz)
    if dim==0:
        for j in range(0, grid_img.shape[1], grid_step):
            grid_img[:, j+line_thickness-1, :] = 1
        for i in range(0, grid_img.shape[2], grid_step):
            grid_img[:, :, i+line_thickness-1] = 1
    elif dim==1:
        for j in range(0, grid_img.shape[0], grid_step):
            grid_img[j+line_thickness-1, :, :] = 1
        for i in range(0, grid_img.shape[2], grid_step):
            grid_img[:, :, i+line_thickness-1] = 1
    elif dim==2:
        for j in range(0, grid_img.shape[0], grid_step):
            grid_img[j+line_thickness-1, :, :] = 1
        for i in range(0, grid_img.shape[1], grid_step):
            grid_img[:, i+line_thickness-1, :] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img)
    return grid_img

def get_cmap(n, name='nipy_spectral'):
    """Return a matplotlib colormap with n distinct colors.

    Args:
        n: Number of discrete colors.
        name: Matplotlib colormap name.

    Returns:
        Colormap instance.
    """
    return plt.cm.get_cmap(name, n)

def pca_reduce_channels_cpu(x: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Reduce channel dimension via PCA on CPU and project back to device.

        Args:
                x: Feature maps (B, C, H, W, D).
                k: Number of principal components to keep.

        Returns:
                Channel-reduced tensor (B, k, H, W, D).
        """
    B, C, H, W, D = x.shape
    assert k <= C, "k must be <= C"
    device = x.device
    dtype  = x.dtype

    # allocate output
    y = torch.empty((B, k, H, W, D), device=device, dtype=dtype)

    for b in range(B):
        # 1) flatten spatial dims
        X = x[b].reshape(C, -1)            # (C, N)
        mu = X.mean(dim=1, keepdim=True)   # (C, 1)
        Xc = X - mu                        # center

        # 2) move to CPU for SVD
        Xc_cpu = Xc.cpu()
        U_cpu, S_cpu, Vh_cpu = torch.linalg.svd(Xc_cpu, full_matrices=False)

        # 3) pick top‐k directions and bring them back to GPU
        U_k = U_cpu[:, :k].to(device)      # (C, k)

        # 4) project the original (centered) data on GPU
        Z = U_k.t() @ Xc                   # (k, N)
        y[b] = Z.reshape(k, H, W, D)

    return y