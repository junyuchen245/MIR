import pytest
import torch

from MIR.models import (
    VFA,
    VxmDense,
)
from MIR.models import convex_adam_MIND
from MIR.models import configs_TransMorph as configs_TransMorph
from MIR.models import configs_VFA as configs_VFA
import MIR.models.convexAdam.configs_ConvexAdam_MIND as configs_ConvexAdam


def test_vfa_init_cpu():
    config = configs_VFA.get_VFA_default_config()
    config.img_size = (16, 16, 16)
    model = VFA(config, device="cpu")
    assert model is not None


def test_transmorph_init_cpu():
    from MIR.models import TransMorph

    config = configs_TransMorph.get_3DTransMorph3Lvl_config()
    config.img_size = (16, 16, 16)
    config.window_size = (2, 2, 2)
    config.out_chan = 3
    model = TransMorph(config, SVF=False)
    assert model is not None


def test_voxelmorph_init_cpu():
    config = type("cfg", (), {})()
    config.img_size = (16, 16, 16)
    config.nb_unet_features = ((8, 16), (16, 8))
    config.nb_unet_levels = None
    config.unet_feat_mult = 1
    config.use_probs = False
    model = VxmDense(config, gen_output=False)
    assert model is not None


def test_convexadam_init_cpu():
    config = configs_ConvexAdam.get_ConvexAdam_MIND_brain_default_config()
    assert config is not None
    assert callable(convex_adam_MIND)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SITReg requires CUDA in current setup")
def test_sitreg_init_cuda():
    from MIR.models import SITReg, EncoderFeatureExtractor
    from MIR.models.SITReg import ReLUFactory, GroupNormalizerFactory
    from MIR.models.SITReg.deformation_inversion_layer.fixed_point_iteration import (
        AndersonSolver,
        AndersonSolverArguments,
        MaxElementWiseAbsStopCriterion,
        RelativeL2ErrorStopCriterion,
    )

    input_shape = (16, 16, 16)
    feature_extractor = EncoderFeatureExtractor(
        n_input_channels=1,
        activation_factory=ReLUFactory(),
        n_features_per_resolution=[8, 16, 32],
        n_convolutions_per_resolution=[1, 1, 1],
        input_shape=input_shape,
        normalizer_factory=GroupNormalizerFactory(2),
    ).cuda()

    solver_fwd = AndersonSolver(
        MaxElementWiseAbsStopCriterion(min_iterations=1, max_iterations=5, threshold=1e-2),
        AndersonSolverArguments(memory_length=2),
    )
    solver_bwd = AndersonSolver(
        RelativeL2ErrorStopCriterion(min_iterations=1, max_iterations=5, threshold=1e-2),
        AndersonSolverArguments(memory_length=2),
    )

    model = SITReg(
        feature_extractor=feature_extractor,
        n_transformation_convolutions_per_resolution=[1, 1, 1],
        n_transformation_features_per_resolution=[8, 16, 32],
        max_control_point_multiplier=0.99,
        affine_transformation_type=None,
        input_voxel_size=(1.0, 1.0, 1.0),
        input_shape=input_shape,
        transformation_downsampling_factor=(1.0, 1.0, 1.0),
        forward_fixed_point_solver=solver_fwd,
        backward_fixed_point_solver=solver_bwd,
        activation_factory=ReLUFactory(),
        normalizer_factory=GroupNormalizerFactory(4),
    ).cuda()

    assert model is not None
