"""Structural Similarity Index (SSIM) utilities.

All SSIM functions use a Gaussian window with ``sigma = 1.5`` and the constants ``C1 = 0.01^2`` and ``C2 = 0.03^2``,
which assume intensities in [0, 1].
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from MIR import _amp


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """Create a 1D Gaussian kernel for SSIM windows.

    Args:
        window_size: Number of taps; the kernel is centred at ``window_size // 2``.
        sigma: Standard deviation in voxels.

    Returns:
        Float tensor (window_size,) normalised to sum to 1.
    """
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    """Create a 2D SSIM window tensor.

    Args:
        window_size: Window edge length.
        channel: Number of channels (one depthwise filter per channel).

    Returns:
        Gaussian window (channel, 1, window_size, window_size) with ``sigma = 1.5``, on the CPU.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def create_window_3D(window_size: int, channel: int) -> torch.Tensor:
    """Create a 3D SSIM window tensor.

    Args:
        window_size: Window edge length.
        channel: Number of channels (one depthwise filter per channel).

    Returns:
        Gaussian window (channel, 1, window_size, window_size, window_size) with ``sigma = 1.5``, on the CPU.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = (
        _1D_window.mm(_2D_window.reshape(1, -1))
        .reshape(window_size, window_size, window_size)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """Compute 2D SSIM for a pair of images.

    Args:
        img1: Tensor (B, C, H, W).
        img2: Tensor (B, C, H, W).
        window: SSIM window tensor.
        window_size: Window size.
        channel: Number of channels.
        size_average: If True, returns scalar mean.

    Returns:
        SSIM value: scalar mean if ``size_average``, else per-image (B,) mean over C, H, W.

    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    """Compute 3D SSIM for a pair of volumes.

    Args:
        img1: Tensor (B, C, H, W, D).
        img2: Tensor (B, C, H, W, D).
        window: SSIM window tensor.
        window_size: Window size.
        channel: Number of channels.
        size_average: If True, returns scalar mean.

    Returns:
        SSIM value: scalar mean if ``size_average``, else the mean over dims C, H, W, which has shape (B, D) (the last
        spatial dim is not reduced).

    """
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)


class SSIM2D(torch.nn.Module):
    """2D structural similarity (SSIM) between two images.

    Returns the SSIM itself (a similarity, higher is better), not a loss. The Gaussian window is cached and rebuilt
    when the channel count or tensor type of the input changes.

    Args:
        window_size: Gaussian window edge length.
        size_average: If True, return the scalar mean SSIM; otherwise a per-image (B,) SSIM.
    """

    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    @_amp.fp32_forward
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute 2D SSIM between two images.

        Args:
            img1: Tensor (B, C, H, W).
            img2: Tensor (B, C, H, W).

        Returns:
            SSIM score (scalar, or (B,) if ``size_average=False``).

        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIM3D(torch.nn.Module):
    """3D SSIM loss between two volumes.

    Unlike :class:`SSIM2D`, this module returns ``1 - SSIM`` (a loss to minimise). The Gaussian window is cached and
    rebuilt when the channel count or tensor type of the input changes.

    Args:
        window_size: Gaussian window edge length.
        size_average: If True, use the scalar mean SSIM; otherwise the unreduced output of the per-image path
            (see :func:`_ssim_3D`).
    """

    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    @_amp.fp32_forward
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute the 3D SSIM loss between two volumes.

        Args:
            img1: Tensor (B, C, H, W, D).
            img2: Tensor (B, C, H, W, D).

        Returns:
            ``1 - SSIM`` (scalar if ``size_average=True``).

        """
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 - _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    """Compute 2D SSIM between two images (functional form of :class:`SSIM2D`).

    Args:
        img1: Tensor (B, C, H, W).
        img2: Tensor (B, C, H, W).
        window_size: Gaussian window edge length.
        size_average: If True, return the scalar mean; otherwise a per-image (B,) SSIM.

    Returns:
        SSIM similarity (higher is better).
    """
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim3D(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    """Compute 3D SSIM between two volumes.

    Unlike :class:`SSIM3D`, this returns the SSIM similarity itself, not ``1 - SSIM``.

    Args:
        img1: Tensor (B, C, H, W, D).
        img2: Tensor (B, C, H, W, D).
        window_size: Gaussian window edge length.
        size_average: If True, return the scalar mean; otherwise the unreduced output of :func:`_ssim_3D`.

    Returns:
        SSIM similarity (higher is better).
    """
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, window_size, channel, size_average)
