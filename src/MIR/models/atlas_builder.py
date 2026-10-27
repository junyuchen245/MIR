"""Deformable template (atlas) construction: a template-creation wrapper and a running-mean flow stream."""

from typing import Any, Sequence, Tuple

import torch
import torch.nn as nn

from .registration_utils import SpatialTransformer

try:
    from .SITReg.composable_mapping import DataFormat

    _HAS_SITREG = True
except Exception:  # pragma: no cover - optional dependency
    DataFormat = None
    _HAS_SITREG = False


class TemplateCreation(nn.Module):
    """Deformable template network that registers a learnable atlas and an image in both directions.

    Args:
        reg_model: Registration model. With ``use_sitreg=False`` it is called as ``reg_model(inputs)`` and must
            return ``(def_atlas, def_image, pos_flow, neg_flow)``; with ``use_sitreg=True`` it is a SITReg model
            called as ``reg_model(atlas, image, mappings_for_levels=((0, False),))``.
        img_size: Spatial image size ``(H, W, D)``.
        mean_cap: Count cap of the running mean in :class:`MeanStream`.
        use_sitreg: Use SITReg-style mapping outputs when True; VFA, TransMorph, VoxelMorph-style otherwise.
        mode: SpatialTransformer interpolation mode (SITReg path).

    Raises:
        ImportError: If ``use_sitreg`` is True but SITReg cannot be imported.
    """

    def __init__(
        self,
        reg_model: nn.Module,
        img_size: Sequence[int],
        mean_cap: float = 100,
        use_sitreg: bool = False,
        mode: str = 'bilinear',
    ):
        super().__init__()
        self.reg_model = reg_model
        self.mean_stream = MeanStream(mean_cap, img_size)
        self.use_sitreg = use_sitreg
        self.spatial_trans = SpatialTransformer(img_size, mode=mode) if use_sitreg else None
        if self.use_sitreg and not _HAS_SITREG:
            raise ImportError("SITReg is not available; cannot use SITReg mode.")

    def forward(self, inputs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Register atlas and image and update the running mean of the atlas-to-image flow.

        Args:
            inputs: Model input. For the SITReg path a pair ``(atlas, image)`` of tensors ``[B, C, H, W, D]``;
                otherwise whatever ``reg_model`` expects.

        Returns:
            Tuple ``(def_atlas, def_image, pos_flow, neg_flow, mean_stream)``: atlas warped to the image, image warped
            to the atlas, forward and inverse voxel displacement fields ``[B, 3, H, W, D]``, and the output of
            :class:`MeanStream` for ``pos_flow``.

        """
        if self.use_sitreg:
            atlas, image = inputs
            mapping_pair = self.reg_model(atlas, image, mappings_for_levels=((0, False),))[0]
            pos_flow = mapping_pair.forward_mapping.sample(DataFormat.voxel_displacements()).generate_values()
            neg_flow = mapping_pair.inverse_mapping.sample(DataFormat.voxel_displacements()).generate_values()
            def_atlas = self.spatial_trans(atlas, pos_flow)
            def_image = self.spatial_trans(image, neg_flow)
        else:
            def_atlas, def_image, pos_flow, neg_flow = self.reg_model(inputs)
        mean_stream = self.mean_stream(pos_flow)
        return def_atlas, def_image, pos_flow, neg_flow, mean_stream


class MeanStream(nn.Module):
    """Running mean of 3-channel flow fields for the deformable template network.

    The mean ``[1, 3, *in_shape]`` and the sample count are stored as non-trainable ``nn.Parameter`` objects so that
    they are saved in the state dict.

    Args:
        cap: Count cap; once the count reaches ``cap`` new batches are weighted by ``B / cap``.
        in_shape: Spatial shape ``(H, W, D)`` of the flow fields.
    """

    def __init__(self, cap: float = 100.0, in_shape: Sequence[int] = (160, 160, 256)):
        super().__init__()
        self.cap = float(cap)
        weights = torch.zeros((1, 3, in_shape[0], in_shape[1], in_shape[2]))
        self.mean = nn.Parameter(weights)
        self.mean.requires_grad = False
        self.count = nn.Parameter(torch.zeros(1))
        self.count.requires_grad = False

    def mean_update(
        self, pre_mean: torch.Tensor, pre_count: torch.Tensor, x: torch.Tensor, pre_cap: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the capped running-mean update for a batch.

        Args:
            pre_mean: Previous mean ``[1, C, H, W, L]``.
            pre_count: Previous sample count, shape ``[1]``.
            x: Batch ``[B, C, H, W, L]``.
            pre_cap: Count cap; the update weight is ``B / min(pre_count + B, pre_cap)``.

        Returns:
            Tuple ``(new_mean, new_count)``.

        """
        pre_cap = torch.tensor(pre_cap)
        B, C, H, W, L = x.shape
        # compute this batch stats
        this_sum = torch.sum(x, dim=0, keepdim=True)

        # increase count and compute weights
        new_count = pre_count + B
        alpha = B / torch.minimum(new_count, pre_cap)

        # compute new mean. Note that once we reach self.cap (e.g. 1000),
        # the 'previous mean' matters less
        new_mean = pre_mean * (1 - alpha) + (this_sum / B) * alpha
        return (new_mean, new_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the (count-scaled) running mean, updating it first in training mode.

        In training mode the stored mean and count are replaced by the updated values. The returned mean is
        broadcast to the shape of ``x`` and multiplied by ``min(1, count / cap)``.

        Args:
            x: Flow batch ``[B, 3, H, W, D]``.

        Returns:
            Tensor with the shape of ``x``.

        """
        z = torch.ones_like(x)

        if self.training is False:
            return torch.minimum(torch.tensor(1.0), self.count / self.cap) * (z * self.mean)

        # get new mean and count
        new_mean, new_count = self.mean_update(self.mean, self.count, x, self.cap)
        stored_mean, stored_count = new_mean.detach(), new_count.detach()
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            # DistributedDataParallel: every rank applies the update of the global batch, so the running mean (the
            # template statistics) stays identical across ranks. The returned, differentiable value is unchanged.
            batch_sum = torch.sum(x.detach(), dim=0, keepdim=True)
            batch_n = torch.tensor([float(x.shape[0])], device=x.device, dtype=batch_sum.dtype)
            torch.distributed.all_reduce(batch_sum)
            torch.distributed.all_reduce(batch_n)
            stored_count = self.count.detach() + batch_n
            alpha = batch_n / torch.minimum(stored_count, torch.tensor(float(self.cap), device=x.device))
            stored_mean = self.mean.detach() * (1 - alpha) + (batch_sum / batch_n) * alpha
        # update op
        self.count = nn.Parameter(stored_count, requires_grad=False)
        self.mean = nn.Parameter(stored_mean, requires_grad=False)

        # the first few 1000 should not matter that much towards this cost
        return torch.minimum(torch.tensor(1.0), new_count / self.cap) * (z * new_mean)
