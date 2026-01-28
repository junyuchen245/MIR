import torch
import torch.nn as nn
import torch.nn.functional as F

from .registration_utils import SpatialTransformer

try:
    from .SITReg.composable_mapping import DataFormat
    _HAS_SITREG = True
except Exception:  # pragma: no cover - optional dependency
    DataFormat = None
    _HAS_SITREG = False


class TemplateCreation(nn.Module):
    '''
    Deformable Template Network
    Args:
        reg_model: Registration model
        img_size: Image size
        mean_cap: Mean cap for the MeanStream
        use_sitreg: Use SITReg-style mapping outputs when True; VFA, TransMorph, VoxelMorph-style otherwise
        mode: SpatialTransformer interpolation mode (SITReg path)
    '''
    def __init__(self, reg_model, img_size, mean_cap=100, use_sitreg=False, mode='bilinear'):
        super().__init__()
        self.reg_model = reg_model
        self.mean_stream = MeanStream(mean_cap, img_size)
        self.use_sitreg = use_sitreg
        self.spatial_trans = SpatialTransformer(img_size, mode=mode) if use_sitreg else None
        if self.use_sitreg and not _HAS_SITREG:
            raise ImportError("SITReg is not available; cannot use SITReg mode.")

    def forward(self, inputs):
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
    '''
    Mean stream for the Deformable Template Network
    Args:
        cap: Cap for the mean
        in_shape: Input shape
    '''
    def __init__(self, cap=100.0, in_shape=(160, 160, 256)):
        super().__init__()
        self.cap = float(cap)
        weights = torch.zeros((1, 3, in_shape[0], in_shape[1], in_shape[2]))
        self.mean = nn.Parameter(weights)
        self.mean.requires_grad = False
        self.count = nn.Parameter(torch.zeros((1)))
        self.count.requires_grad = False

    def mean_update(self, pre_mean, pre_count, x, pre_cap=0.):
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

    def forward(self, x):
        z = torch.ones_like(x)

        if self.training is False:
            return torch.minimum(torch.tensor(1.), self.count / self.cap) * (z * self.mean)

        # get new mean and count
        new_mean, new_count = self.mean_update(self.mean, self.count, x, self.cap)
        # update op
        self.count = nn.Parameter(new_count)
        self.mean = nn.Parameter(new_mean)

        # the first few 1000 should not matter that much towards this cost
        return torch.minimum(torch.tensor(1.), new_count / self.cap) * (z * new_mean)


