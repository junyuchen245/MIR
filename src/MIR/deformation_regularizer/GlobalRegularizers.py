'''
Global regularizers for deformation regularization.
Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''

import torch
import MIR.utils.registration_utils as reg_utils
import torch.nn.functional as nnf

class Grad2D(torch.nn.Module):
    """
    2D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad2D, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class Grad3d(torch.nn.Module):
    """
    3D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true=None):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class Grad3DiTV(torch.nn.Module):
    """
    3D gradient Isotropic TV loss.
    """

    def __init__(self):
        super(Grad3DiTV, self).__init__()
        a = 1

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, :-1, 1:, 1:])
        dx = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, 1:, :-1, 1:])
        dz = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, 1:, 1:, :-1])
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz
        d = torch.mean(torch.sqrt(dx+dy+dz+1e-6))
        grad = d / 3.0
        return grad

class DisplacementRegularizer(torch.nn.Module):
    def __init__(self, energy_type):
        super().__init__()
        self.energy_type = energy_type

    def gradient_dx(self, fv): return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

    def gradient_dy(self, fv): return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

    def gradient_dz(self, fv): return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

    def gradient_txyz(self, Txyz, fn):
        return torch.stack([fn(Txyz[:,i,...]) for i in [0, 1, 2]], dim=1)

    def compute_gradient_norm(self, displacement, flag_l1=False):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        if flag_l1:
            norms = torch.abs(dTdx) + torch.abs(dTdy) + torch.abs(dTdz)
        else:
            norms = dTdx**2 + dTdy**2 + dTdz**2
        return torch.mean(norms)/3.0

    def compute_bending_energy(self, displacement):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        dTdxx = self.gradient_txyz(dTdx, self.gradient_dx)
        dTdyy = self.gradient_txyz(dTdy, self.gradient_dy)
        dTdzz = self.gradient_txyz(dTdz, self.gradient_dz)
        dTdxy = self.gradient_txyz(dTdx, self.gradient_dy)
        dTdyz = self.gradient_txyz(dTdy, self.gradient_dz)
        dTdxz = self.gradient_txyz(dTdx, self.gradient_dz)
        return torch.mean(dTdxx**2 + dTdyy**2 + dTdzz**2 + 2*dTdxy**2 + 2*dTdxz**2 + 2*dTdyz**2)

    def forward(self, disp, _):
        if self.energy_type == 'bending':
            energy = self.compute_bending_energy(disp)
        elif self.energy_type == 'gradient-l2':
            energy = self.compute_gradient_norm(disp)
        elif self.energy_type == 'gradient-l1':
            energy = self.compute_gradient_norm(disp, flag_l1=True)
        else:
            raise Exception('Not recognised local regulariser!')
        return energy
    
class GradICON3d(torch.nn.Module):
    """
    Gradient‑ICON loss for 3‑D displacement fields.
    Penalises the Frobenius‑norm of the Jacobian of the
    composition Φ^{AB}∘Φ^{BA} (forward ◦ inverse).
    """

    def __init__(self, flow_shape, penalty='l2', loss_mult=None, both_dirs=False, device='cpu'):
        """
        Args
        ----
        stn        : instance of SpatialTransformer (warps tensors by displacements)
        penalty    : 'l1' or 'l2'
        loss_mult  : optional scalar multiplier
        both_dirs  : if True also penalise the reverse composition
                     Φ^{BA}∘Φ^{AB} and average the two losses
        """
        super().__init__()
        if penalty not in ('l1', 'l2'):
            raise ValueError("penalty must be 'l1' or 'l2'")
        self.stn        = reg_utils.SpatialTransformer(flow_shape).to(device)
        self.penalty    = penalty
        self.loss_mult  = loss_mult
        self.both_dirs  = both_dirs

    @staticmethod
    def _grad3d(disp, p):
        """finite‑difference gradient loss for one displacement"""
        dy = torch.abs(disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :])
        dx = torch.abs(disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :])
        dz = torch.abs(disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1])

        if p == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
        return (dx.mean() + dy.mean() + dz.mean()) / 3.0

    def forward(self, flow_fwd, flow_inv):
        """
        Returns
        -------
        loss : scalar GradICON penalty
        """
        # Φ^{AB}∘Φ^{BA} − Id   (displacement form)
        comp_f = flow_inv + self.stn(flow_fwd, flow_inv)
        loss   = self._grad3d(comp_f, self.penalty)

        if self.both_dirs:
            # Φ^{BA}∘Φ^{AB} − Id
            comp_b = flow_fwd + self.stn(flow_inv, flow_fwd)
            loss   = 0.5 * (loss + self._grad3d(comp_b, self.penalty))

        if self.loss_mult is not None:
            loss *= self.loss_mult
        return loss

class GradICONExact3d(torch.nn.Module):
    """
    Paper‑faithful Gradient‑ICON for 3‑D flows
    """

    def __init__(self, vol_shape, penalty='l2',
                 both_dirs=False, device='cpu'):
        super().__init__()
        if penalty not in ('l1', 'l2'):
            raise ValueError("penalty must be 'l1' or 'l2'")
        self.D, self.H, self.W = vol_shape
        self.penalty   = penalty
        self.both_dirs = both_dirs
        self.device    = device

        # unchanged voxelmorph ST‑N
        self.stn = reg_utils.SpatialTransformer(vol_shape).to(device)

        # Δx = 1e‑3 in unit‑cube coords  →  voxel step
        self.dx_vox = torch.tensor([s - 1 for s in vol_shape],
                                   dtype=torch.float32, device=device) * 1e-3

        # full identity grid in voxel coords (D,H,W,3)
        z = torch.linspace(0, self.D - 1, self.D, device=device)
        y = torch.linspace(0, self.H - 1, self.H, device=device)
        x = torch.linspace(0, self.W - 1, self.W, device=device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        self.grid_full = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)  # (N,3)
        self.Nsub = self.grid_full.size(0) // 8                         # vox/2³

        self.register_buffer('eye3', torch.eye(3))

    # ---------------------------------------------------------------- utilities
    @staticmethod
    def _fro(diff, p):
        if p == 'l1':
            return diff.abs().sum((-2, -1))       # ||·||_F  (L¹)
        else:
            return diff.pow(2).sum((-2, -1))      # ||·||_F²

    # ---------------------------------------------------------------- helpers
    def _compose_disp(self, flow_fwd, flow_inv):
        """
        Returns displacement of Φ_AB∘Φ_BA on the voxel grid (B,3,D,H,W)
        """
        return flow_inv + self.stn(flow_fwd, flow_inv)    # exactly as in your code

    def _jacobian_samples(self, disp, pts_vox):
        """
        disp       : (B,3,D,H,W) displacement field in *voxel* units
        pts_vox    : (B,N,3) random sample points in *voxel* coords
        returns    : (B,N,3,3) finite‑difference Jacobian at those points
        """
        B, N, _ = pts_vox.shape

        # helper to sample disp at arbitrary pts via grid_sample
        def sample(f, p):
            p_norm = p.clone()                            # [0,1]³ → [-1,1]³
            p_norm[..., 0] = 2 * (p[..., 0] / (self.W - 1) - 0.5)
            p_norm[..., 1] = 2 * (p[..., 1] / (self.H - 1) - 0.5)
            p_norm[..., 2] = 2 * (p[..., 2] / (self.D - 1) - 0.5)
            g = p_norm.view(B, N, 1, 1, 3)
            v = nnf.grid_sample(f, g, align_corners=False,
                              mode='bilinear', padding_mode='border')
            return v.view(B, 3, N).permute(0, 2, 1)       # B×N×3

        # Φ(x) = x + disp(x)
        x      = pts_vox                                   # B×N×3
        phi_x  = x + sample(disp, x)

        grads = []
        # finite differences: (Φ(x+Δx e_i) - Φ(x)) / Δx
        for axis, dx in enumerate(self.dx_vox):
            x_shift    = x.clone()
            x_shift[..., axis] += dx
            phi_shift  = x_shift + sample(disp, x_shift)
            grads.append((phi_shift - phi_x) / dx)

        return torch.stack(grads, dim=-1)                  # B×N×3×3

    # ---------------------------------------------------------------- forward
    def forward(self, flow_fwd, flow_inv):
        """
        flow_fwd, flow_inv : (B,3,D,H,W) voxel‑unit displacements
        """
        B = flow_fwd.size(0)

        # uniform random sub‑sample of voxel centres
        idx  = torch.randperm(self.grid_full.size(0), device=self.device)[:self.Nsub]
        pts0 = self.grid_full[idx].unsqueeze(0).repeat(B, 1, 1)  # B×Ns×3

        # Φ_AB∘Φ_BA displacement field
        comp_disp = self._compose_disp(flow_fwd, flow_inv)

        # Jacobian of composition at sampled points
        J  = self._jacobian_samples(comp_disp, pts0)
        diff = J - self.eye3.to(J.device)                                # ∇Φ − I
        loss = self._fro(diff, self.penalty).mean()

        if self.both_dirs:                                  # reverse term
            comp_disp_b = self._compose_disp(flow_inv, flow_fwd)
            J_b  = self._jacobian_samples(comp_disp_b, pts0)
            diff_b = J_b - self.eye3
            loss   = 0.5 * (loss + self._fro(diff_b, self.penalty).mean())

        return loss