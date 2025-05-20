import torch
import torch.nn as nn

class CompRecon(nn.Module):
    """
    Composite loss for image synthesis without GANs:
      - Charbonnier (smooth L1) to preserve edges
      - Gradient-difference to align edges
      - Focal-frequency to boost high-frequency detail
    Args:
        charb_weight (float): Weight for Charbonnier loss.
        grad_weight (float): Weight for gradient difference loss.
        ff_weight (float): Weight for focal frequency loss.
        eps (float): Small constant to avoid division by zero.
        alpha (float): Exponent for focal frequency loss.
        beta (float): Exponent for focal frequency loss.
    """
    def __init__(
        self,
        charb_weight: float = 100.0,
        grad_weight: float = 10.0,
        ff_weight: float   = 1.0,
        eps: float         = 1e-3,
        alpha: float       = 1.0,
        beta: float        = 1.0
    ):
        super().__init__()
        self.charb_weight = charb_weight
        self.grad_weight  = grad_weight
        self.ff_weight    = ff_weight
        self.eps          = eps
        self.alpha        = alpha
        self.beta         = beta

    def _charbonnier(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff*diff + self.eps**2) - self.eps)

    def _grad_diff(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # assume pred/target shape (B, C, H, W, D) or (B, 1, H, W, D)
        dx_pred = torch.abs(pred[:,:,1:,:,:] - pred[:,:,:-1,:,:])
        dx_tgt  = torch.abs(target[:,:,1:,:,:] - target[:,:,:-1,:,:])
        dx = torch.mean(torch.abs(dx_pred - dx_tgt))

        dy_pred = torch.abs(pred[:,:,:,1:,:] - pred[:,:,:,:-1,:])
        dy_tgt  = torch.abs(target[:,:,:,1:,:] - target[:,:,:,:-1,:])
        dy = torch.mean(torch.abs(dy_pred - dy_tgt))

        dz_pred = torch.abs(pred[:,:,:,:,1:] - pred[:,:,:,:,:-1])
        dz_tgt  = torch.abs(target[:,:,:,:,1:] - target[:,:,:,:,:-1])
        dz = torch.mean(torch.abs(dz_pred - dz_tgt))

        return (dx + dy + dz) / 3.0

    def _focal_freq(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # FFT over the last three dims
        Fp = torch.fft.fftn(pred,   dim=(-3, -2, -1))
        Ft = torch.fft.fftn(target, dim=(-3, -2, -1))
        diff = Fp - Ft
        mag  = torch.abs(diff)
        w    = mag.pow(self.alpha)
        return torch.mean(w * mag.pow(self.beta))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_char = self._charbonnier(pred, target)
        l_grad = self._grad_diff(pred, target)
        l_ff   = self._focal_freq(pred, target)

        return (
            self.charb_weight * l_char
          + self.grad_weight  * l_grad
          + self.ff_weight    * l_ff
        )