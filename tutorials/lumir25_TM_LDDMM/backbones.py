"""Backbone factory for the LUMIR25 LDDMM tutorial.

``build_model(args, H, W, D)`` returns ``(model, scale_factor)`` where ``model`` is either the
dedicated :class:`TransMorphLDDMM` or an :class:`LDDMMWrapper` around another MIR backbone, and
``scale_factor`` is the down-sampling applied to the images before they enter the network
(``2`` = the network runs at half resolution).  Every backbone is built exactly as in its own
``tutorials/lumir25_*`` training script, with its SVF integration switched off because the LDDMM
flow is integrated by the wrapper.
"""
import os
import gdown
import torch
from MIR import ModelWeights
from MIR.models import (TransMorphLDDMM, LDDMMWrapper, load_backbone_weights, TransMorph, TransMorphTVFSPR, VFA,
                        VFASPR, VxmDense, LessNet, FourierNetPlus)
import MIR.models.configs_TransMorph as configs_TransMorph
import MIR.models.configs_VFA as configs_VFA
import MIR.models.configs_VoxelMorph as configs_VoxelMorph
import MIR.models.configs_LessNet as configs_LessNet
import MIR.models.configs_FourierNet as configs_FourierNet

BACKBONES = ['transmorph_lddmm', 'transmorph', 'transmorph_spr', 'vfa', 'vfa_spr', 'voxelmorph', 'lessnet',
             'fouriernet', 'sitreg']
SPR_BACKBONES = ('transmorph_spr', 'vfa_spr')  # return (field, spr_weight) themselves
# network resolution used by the corresponding lumir25 tutorial (image size / scale_factor)
DEFAULT_SCALE = {'transmorph_lddmm': 2, 'transmorph': 2, 'transmorph_spr': 2, 'vfa': 1, 'vfa_spr': 1,
                 'voxelmorph': 1, 'lessnet': 1, 'fouriernet': 1, 'sitreg': 1}


def _lddmm_kwargs(args):
    kw = dict(param=args.param, time_steps=args.time_steps, kernel=args.kernel, alpha=args.alpha,
              gamma=args.gamma, power=args.power, sigma=args.sigma, int_scale=args.int_scale)
    sigmas = getattr(args, 'sigmas', None)
    if sigmas:  # spatially varying multi-Gaussian kernel (region-specific LDDMM)
        kw.update(sigmas=list(sigmas), weight_smooth_sigma=getattr(args, 'weight_smooth_sigma', 1.0))
    spr = getattr(args, 'spr', 'none')
    if spr and spr != 'none':
        kw['spr'] = spr
    return kw


def _spr_source(args):
    source = getattr(args, 'spr_source', 'auto')
    if source == 'auto':
        return 'backbone' if args.backbone in SPR_BACKBONES else 'head'
    return source


def _transmorph_config(h, w, d, H, W, D):
    config = configs_TransMorph.get_3DTransMorph3Lvl_config()
    config.img_size = (h, w, d)
    config.window_size = (H // 64, W // 64, D // 64)
    config.out_chan = 3
    return config


def _build_sitreg(shape):
    from MIR.models import EncoderFeatureExtractor, SITReg
    from MIR.models.SITReg import ReLUFactory, GroupNormalizerFactory
    from MIR.models.SITReg.deformation_inversion_layer.fixed_point_iteration import (
        AndersonSolver, AndersonSolverArguments, MaxElementWiseAbsStopCriterion, RelativeL2ErrorStopCriterion)
    feature_extractor = EncoderFeatureExtractor(
        n_input_channels=1,
        activation_factory=ReLUFactory(),
        n_features_per_resolution=[12, 16, 32, 64, 128, 128],
        n_convolutions_per_resolution=[2, 2, 2, 2, 2, 2],
        input_shape=shape,
        normalizer_factory=GroupNormalizerFactory(2),
    )
    return SITReg(
        feature_extractor=feature_extractor,
        n_transformation_convolutions_per_resolution=[2, 2, 2, 2, 2, 2],
        n_transformation_features_per_resolution=[12, 64, 128, 256, 256, 256],
        max_control_point_multiplier=0.99,
        affine_transformation_type=None,
        input_voxel_size=(1.0, 1.0, 1.0),
        input_shape=shape,
        transformation_downsampling_factor=(1.0, 1.0, 1.0),
        forward_fixed_point_solver=AndersonSolver(
            MaxElementWiseAbsStopCriterion(min_iterations=2, max_iterations=50, threshold=1e-2),
            AndersonSolverArguments(memory_length=4)),
        backward_fixed_point_solver=AndersonSolver(
            RelativeL2ErrorStopCriterion(min_iterations=2, max_iterations=50, threshold=1e-2),
            AndersonSolverArguments(memory_length=4)),
        activation_factory=ReLUFactory(),
        normalizer_factory=GroupNormalizerFactory(4),
    )


def build_model(args, H, W, D):
    """Build the LDDMM model for ``args.backbone``. Returns ``(model, scale_factor)``."""
    name = args.backbone
    if name not in BACKBONES:
        raise ValueError(f'Unknown backbone {name}; choose from {BACKBONES}')
    s = args.scale_factor if args.scale_factor > 0 else DEFAULT_SCALE[name]
    h, w, d = H // s, W // s, D // s
    kw = _lddmm_kwargs(args)
    adapter = None
    if name == 'transmorph_lddmm':
        return TransMorphLDDMM(_transmorph_config(h, w, d, H, W, D), **kw), s
    if name in SPR_BACKBONES and 'spr' not in kw:
        raise ValueError(f'{name} is an SPR backbone; choose --spr kernel or --spr energy')
    if 'spr' in kw:
        kw['spr_source'] = _spr_source(args)
    device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'
    if name == 'transmorph':
        backbone = TransMorph(_transmorph_config(h, w, d, H, W, D), SVF=False)
    elif name == 'transmorph_spr':
        backbone = TransMorphTVFSPR(_transmorph_config(h, w, d, H, W, D), SVF=False, time_steps=7)
    elif name == 'vfa':
        config = configs_VFA.get_VFA_default_config()
        config.img_size = (h, w, d)
        backbone = VFA(config, device=device)
    elif name == 'vfa_spr':
        config = configs_VFA.get_VFA_default_config()
        config.img_size = (h, w, d)
        backbone = VFASPR(config, device=device, SVF=False)
    elif name == 'voxelmorph':
        config = configs_VoxelMorph.get_VXM_default_config()
        config.img_size = (h, w, d)
        backbone = VxmDense(config)
    elif name == 'lessnet':
        config = configs_LessNet.get_LessNet_default_config()
        config.img_size = (h, w, d)
        backbone = LessNet(config, SVF=False)
    elif name == 'fouriernet':
        config = configs_FourierNet.get_FourierNetPlus_default_config()
        config.img_size = (h, w, d)
        config.low_img_size = (h // 2, w // 2, d // 2)
        backbone = FourierNetPlus(config, SVF=False)
    elif name == 'sitreg':
        backbone = _build_sitreg((h, w, d))
        adapter = 'sitreg'
    return LDDMMWrapper(backbone, adapter=adapter, **kw), s


def init_backbone(model, args, pretrained_dir='pretrained_wts/'):
    """Initialize the backbone from ``args.init_from`` ('lumir24', a checkpoint path, or '')."""
    if not args.init_from:
        return
    target = model.backbone if isinstance(model, LDDMMWrapper) else model
    if args.init_from == 'lumir24':
        if args.backbone in ('transmorph_lddmm', 'transmorph', 'transmorph_spr'):
            entry, fname = ModelWeights['TransMorphTVF-LUMIR24-MonoModal'], 'TransMorphTVF_LUMIR24.pth.tar'
        elif args.backbone in ('vfa', 'vfa_spr'):
            entry, fname = ModelWeights['VFA-LUMIR24-MonoModal'], 'VFA_LUMIR24.pth'
        else:
            print(f'No LUMIR24 pretrained weights registered for backbone {args.backbone}; random init.')
            return
        os.makedirs(pretrained_dir, exist_ok=True)
        wts_path = pretrained_dir + fname
        if not os.path.isfile(wts_path):
            gdown.download(f"https://drive.google.com/uc?id={entry['wts']}", wts_path, quiet=False)
        ckpt = torch.load(wts_path, map_location='cpu')
        state_dict = ckpt[entry['wts_key']] if entry['wts_key'] in ckpt else ckpt
    else:
        ckpt = torch.load(args.init_from, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt.get('model_state_dict', ckpt)
        if isinstance(model, LDDMMWrapper) and any(k.startswith('backbone.') for k in state_dict):
            state_dict = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
    load_backbone_weights(target, state_dict)
