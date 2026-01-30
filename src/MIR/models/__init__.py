"""Model zoo and configuration helpers for MIR registration pipelines."""
from .Deformable_Swin_Transformer import DefSwinTransformer
from .Deformable_Swin_Transformer_v2 import DefSwinTransformerV2
from .Swin_Transformer import SwinTransformer
from .TransMorph import Conv3dReLU, DecoderBlock, RegistrationHead, TransMorphAffine, TransMorph, TransMorphTVF, TransMorphTVFSPR, CONFIGS
from .configs_TransMorph import get_3DTransMorphDWin3Lvl_config, get_3DTransMorph3Lvl_config, get_3DTransMorph_config, get_3DTransMorphNoRelativePosEmbd_config, get_3DTransMorphSin_config, get_3DTransMorphLrn_config, get_3DTransMorphNoConvSkip_config, get_3DTransMorphNoTransSkip_config, get_3DTransMorphNoSkip_config, get_3DTransMorphLarge_config, get_3DTransMorphSmall_config, get_3DTransMorphTiny_config, get_3DTransMorphRelativePosEmbdSimple_config
from .registration_utils import SpatialTransformer, VecInt, AffineTransformer, ensemble_average, invert_warp_via_velocity, fit_warp_to_svf, fit_warp_to_svf_fast
from .Selfsupervised_Learning_Heads import SSLHeadNLvl, SSLHead1Lvl
from .VFA import VFA, Decoder, DoubleConv3d, grid_to_flow, VFASPR
from .MultiMorph import GroupNet3D, Warp3d, make_epoch_batches, ListBatchSampler
from .HyperVFA import HyperVFA, HyperVFASPR
from .TransVFA import TransVFA
from .atlas_builder import TemplateCreation, MeanStream
from .configs_VFA import get_VFA_default_config
from .Synthesis_Heads import SynthesisHead3DAdvanced, SynthesisHead3D, EfficientAdvancedSynthHead3D, ConvNeXtSynthHead3D
from .Segmentation_Heads import AdvancedDecoder3D
from .convexAdam import convex_adam_MIND, convex_adam_MIND_SPR, convex_adam_seg_features, convex_adam_features, convex_adam_vfa, get_ConvexAdam_MIND_brain_default_config
try:
    from . import fireants
except Exception:  # pragma: no cover - optional dependency
    fireants = None
from .VoxelMorph import VxmDense
from .HyperMorph import HyperVxmDense
from .HyperTransMorph import HyperTransMorphTVF, HyperTransMorphTVFSPR
from .configs_VoxelMorph import get_VXM_1_config, get_VXM_default_config, get_VXM_BJ_config
try:
    from .SITReg import SITReg, MappingPair, EncoderFeatureExtractor, ReLUFactory, GroupNormalizerFactory
except Exception:  # pragma: no cover - optional dependency
    SITReg = None
    MappingPair = None
    EncoderFeatureExtractor = None
    ReLUFactory = None
    GroupNormalizerFactory = None
__all__ = [
    'convex_adam_MIND',
    'convex_adam_MIND_SPR',
    'convex_adam_seg_features',
    'convex_adam_features',
    'convex_adam_vfa',
    'get_ConvexAdam_MIND_brain_default_config',
    'DefSwinTransformer',
    'DefSwinTransformerV2',
    'SwinTransformer',
    'Conv3dReLU',
    'DecoderBlock',
    'RegistrationHead',
    'TransMorphAffine',
    'TransMorph',
    'TransMorphTVF',
    'TransMorphTVFSPR',
    'VxmDense',
    'HyperVxmDense',
    'HyperTransMorphTVF',
    'HyperTransMorphTVFSPR',
    'GroupNet3D',
    'Warp3d',
    'make_epoch_batches',
    'ListBatchSampler',
    *([
        'SITReg',
        'MappingPair',
        'EncoderFeatureExtractor',
        'ReLUFactory',
        'GroupNormalizerFactory',
    ] if SITReg is not None else []),
    'CONFIGS',
    'get_3DTransMorphDWin3Lvl_config',
    'get_3DTransMorph3Lvl_config',
    'get_3DTransMorph_config',
    'get_3DTransMorphNoRelativePosEmbd_config',
    'get_3DTransMorphSin_config',
    'get_3DTransMorphLrn_config',
    'get_3DTransMorphNoConvSkip_config',
    'get_3DTransMorphNoTransSkip_config',
    'get_3DTransMorphNoSkip_config',
    'get_3DTransMorphLarge_config',
    'get_3DTransMorphSmall_config',
    'get_3DTransMorphTiny_config',
    'get_3DTransMorphRelativePosEmbdSimple_config',
    'get_VXM_1_config',
    'get_VXM_default_config',
    'get_VXM_BJ_config',
    'SpatialTransformer',
    'VecInt',
    'SSLHeadNLvl',
    'SSLHead1Lvl',
    'AffineTransformer',
    'VFA',
    'VFASPR',
    'HyperVFA',
    'HyperVFASPR',
    'Decoder',
    'DoubleConv3d',
    'grid_to_flow',
    'TransVFA',
    'get_VFA_default_config',
    'ensemble_average',
    'invert_warp_via_velocity',
    'fit_warp_to_svf',
    'fit_warp_to_svf_fast',
    'TemplateCreation',
    'MeanStream',
    'SynthesisHead3DAdvanced',
    'SynthesisHead3D',
    'EfficientAdvancedSynthHead3D',
    'ConvNeXtSynthHead3D',
    'AdvancedDecoder3D',
    *(['fireants'] if fireants is not None else []),
]
