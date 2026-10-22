import torch
import pytest

from MIR.models.TransMorphLDDMM import (FourierKernel, LDDMMIntegrator, TransMorphLDDMM,
                                        jacobian_fd, det3x3, load_backbone_weights)
from MIR.models.registration_utils import SpatialTransformer
import MIR.models.configs_TransMorph as configs_TransMorph

SIZE = (24, 32, 28)


def _smooth_momentum(kernel, v0_max, seed=0):
    torch.manual_seed(seed)
    m0 = kernel(torch.randn(1, 3, *SIZE))
    return m0 / kernel(m0).abs().max() * v0_max


@pytest.mark.parametrize('kernel_kwargs', [dict(kernel='helmholtz', alpha=2.0, gamma=1.0, power=2),
                                           dict(kernel='gaussian', sigma=[1.5, 3.0])])
def test_kernel_is_symmetric_positive_with_unit_dc_gain(kernel_kwargs):
    kernel = FourierKernel(**kernel_kwargs)
    ir = kernel.impulse_response(SIZE)
    c = tuple(s // 2 for s in SIZE)
    assert ir[c] == ir.max()
    assert torch.allclose(ir[c[0] + 2, c[1], c[2]], ir[c[0] - 2, c[1], c[2]], atol=1e-6)
    assert torch.allclose(ir.sum(), torch.tensor(kernel.k_hat.flatten()[0].item()), atol=1e-4)
    p = torch.randn(2, 3, *SIZE)
    assert kernel.inner(p, kernel(p)) > 0


def test_jacobian_of_linear_field():
    A = torch.tensor([[0.1, 0.02, 0.0], [0.0, -0.05, 0.03], [0.01, 0.0, 0.2]])
    grid = torch.stack(torch.meshgrid([torch.arange(s, dtype=torch.float32) for s in SIZE], indexing='ij'))
    disp = torch.einsum('ij,jdhw->idhw', A, grid)[None]
    jac = jacobian_fd(disp)[0, :, :, 2:-2, 2:-2, 2:-2]
    expected = (torch.eye(3) + A).view(3, 3, 1, 1, 1).expand_as(jac)
    assert torch.allclose(jac, expected, atol=1e-4)
    det = det3x3(jacobian_fd(disp))[0, 2:-2, 2:-2, 2:-2]
    assert torch.allclose(det, torch.linalg.det(torch.eye(3) + A).expand_as(det), atol=1e-4)


def test_shoot_zero_momentum_is_identity():
    kernel = FourierKernel()
    integ = LDDMMIntegrator(SIZE, 5, kernel)
    out = integ.shoot(torch.zeros(1, 3, *SIZE))
    assert torch.count_nonzero(out['flow']) == 0
    assert torch.count_nonzero(out['inv_flow']) == 0
    assert out['energy'] == 0


def test_shoot_energy_conserved_and_inverse_consistent():
    kernel = FourierKernel(alpha=2.0, gamma=1.0, power=2)
    integ = LDDMMIntegrator(SIZE, 8, kernel)
    m0 = _smooth_momentum(kernel, v0_max=3.0)
    out = integ.shoot(m0)
    energies = torch.stack(out['energies'])
    assert ((energies / energies[0]) - 1).abs().max() < 0.15  # kinetic energy ~constant along the geodesic
    u, w = out['flow'], out['inv_flow']
    st = SpatialTransformer(SIZE)
    residual = (u + st(w, u)).norm(dim=1)[0, 2:-2, 2:-2, 2:-2]
    assert residual.mean() < 0.1 * u.norm(dim=1).mean()
    det = det3x3(jacobian_fd(u))
    assert det.min() > 0  # diffeomorphic


def test_shoot_is_differentiable():
    kernel = FourierKernel()
    integ = LDDMMIntegrator(SIZE, 4, kernel)
    m0 = _smooth_momentum(kernel, v0_max=2.0).requires_grad_(True)
    out = integ.shoot(m0)
    (out['flow'].abs().mean() + out['energy']).backward()
    assert m0.grad is not None and torch.isfinite(m0.grad).all() and m0.grad.abs().sum() > 0


def test_relax_matches_shoot_for_single_step():
    kernel = FourierKernel()
    m0 = _smooth_momentum(kernel, v0_max=2.0)
    integ = LDDMMIntegrator(SIZE, 1, kernel)
    out_s = integ.shoot(m0)
    out_r = integ.relax(lambda t, u: m0)
    assert torch.allclose(out_s['flow'], out_r['flow'], atol=1e-6)
    assert torch.allclose(out_s['energy'], out_r['energy'], atol=1e-5)


def _tiny_config():
    config = configs_TransMorph.get_3DTransMorph3Lvl_config()
    config.img_size = (32, 32, 32)
    config.window_size = (2, 2, 2)
    config.embed_dim = 16
    config.depths = (1, 1, 1)
    config.num_heads = (2, 2, 2)
    config.reg_head_chan = 4
    return config


@pytest.mark.parametrize('kwargs', [dict(param='shooting', time_steps=3),
                                    dict(param='shooting', time_steps=2, int_scale=2),
                                    dict(param='tvf', time_steps=3)])
def test_model_forward_backward(kwargs):
    torch.manual_seed(0)
    model = TransMorphLDDMM(_tiny_config(), **kwargs)
    mov = torch.rand(1, 1, 32, 32, 32)
    fix = torch.rand(1, 1, 32, 32, 32)
    out = model((mov, fix), return_dict=True)
    s = kwargs.get('int_scale', 1)
    assert out['flow'].shape == (1, 3, 32 * s, 32 * s, 32 * s)
    assert out['inv_flow'].shape == out['flow'].shape
    assert torch.isfinite(out['flow']).all()
    assert len(out['energies']) == kwargs['time_steps']
    (out['flow'].abs().mean() + out['energy']).backward()
    head = model.reg_head if kwargs['param'] == 'shooting' else model.reg_heads[0]
    assert head[0].weight.grad is not None
    assert model(( mov, fix)).shape == out['flow'].shape


def test_tvf_rejects_int_scale():
    with pytest.raises(ValueError):
        TransMorphLDDMM(_tiny_config(), param='tvf', int_scale=2)


def test_load_backbone_weights_counts():
    torch.manual_seed(0)
    src = TransMorphLDDMM(_tiny_config(), param='tvf', time_steps=2)
    dst = TransMorphLDDMM(_tiny_config(), param='shooting', time_steps=2)
    n_loaded, n_total = load_backbone_weights(dst, src.state_dict(), verbose=False)
    assert 0 < n_loaded < n_total
    assert torch.equal(dst.up0.conv1[0].weight, src.up0.conv1[0].weight)


# --------------------------------------------------------------------------- #
# Model-agnostic wrapper
# --------------------------------------------------------------------------- #
from MIR.models import LDDMMWrapper, VxmDense, LessNet
import MIR.models.configs_VoxelMorph as configs_VoxelMorph
import MIR.models.configs_LessNet as configs_LessNet


def _tiny_vxm():
    config = configs_VoxelMorph.get_VXM_default_config()
    config.img_size = (32, 32, 32)
    config.nb_unet_features = ((4, 8), (8, 8, 4, 4))
    return VxmDense(config)


@pytest.mark.parametrize('kwargs', [dict(param='shooting', time_steps=3),
                                    dict(param='shooting', time_steps=2, int_scale=2),
                                    dict(param='tvf', time_steps=3)])
def test_wrapper_voxelmorph(kwargs):
    torch.manual_seed(0)
    model = LDDMMWrapper(_tiny_vxm(), **kwargs)
    mov, fix = torch.rand(1, 1, 32, 32, 32), torch.rand(1, 1, 32, 32, 32)
    out = model((mov, fix), return_dict=True)
    s = kwargs.get('int_scale', 1)
    assert out['flow'].shape == (1, 3, 32 * s, 32 * s, 32 * s)
    assert out['inv_flow'].shape == out['flow'].shape
    assert model.int_size == (32 * s,) * 3 and model.integrator is not None
    assert len(out['energies']) == kwargs['time_steps']
    key = 'momentum' if kwargs['param'] == 'shooting' else 'pre_velocities'
    assert key in out
    (out['flow'].abs().mean() + out['energy']).backward()
    assert model.backbone.flow.weight.grad is not None
    assert model((mov, fix)).shape == out['flow'].shape


def test_wrapper_lessnet_and_svf_warning():
    torch.manual_seed(0)
    config = configs_LessNet.get_LessNet_default_config()
    config.img_size = (32, 32, 32)
    model = LDDMMWrapper(LessNet(config, SVF=False), time_steps=2)
    mov, fix = torch.rand(1, 1, 32, 32, 32), torch.rand(1, 1, 32, 32, 32)
    assert model((mov, fix)).shape == (1, 3, 32, 32, 32)
    with pytest.warns(UserWarning):
        LDDMMWrapper(LessNet(config, SVF=True), time_steps=2)


def test_wrapper_custom_adapter_and_tuple_output():
    torch.manual_seed(0)
    vxm = _tiny_vxm()
    vxm.gen_output = True  # forward returns (warped, flow); the last element is the field
    vxm.transformer = SpatialTransformer((32, 32, 32))
    model = LDDMMWrapper(vxm, time_steps=2)
    mov, fix = torch.rand(1, 1, 32, 32, 32), torch.rand(1, 1, 32, 32, 32)
    assert model((mov, fix)).shape == (1, 3, 32, 32, 32)

    calls = []

    def adapter(backbone, m, f):
        calls.append(1)
        return backbone((m, f))[-1]

    model = LDDMMWrapper(vxm, param='tvf', time_steps=3, adapter=adapter)
    model((mov, fix))
    assert len(calls) == 3  # tvf re-applies the backbone at every time step

    with pytest.raises(ValueError):
        LDDMMWrapper(vxm, adapter=lambda b, m, f: torch.zeros(1, 2, 32, 32, 32))((mov, fix))


def test_wrapper_state_dict_roundtrip():
    torch.manual_seed(0)
    model = LDDMMWrapper(_tiny_vxm(), time_steps=2)
    mov, fix = torch.rand(1, 1, 32, 32, 32), torch.rand(1, 1, 32, 32, 32)
    ref = model((mov, fix))
    fresh = LDDMMWrapper(_tiny_vxm(), time_steps=2)
    fresh.load_state_dict(model.state_dict(), strict=False)  # integrators are lazy, buffers non-persistent
    assert torch.allclose(fresh((mov, fix)), ref, atol=1e-6)


# --------------------------------------------------------------------------- #
# Spatially varying kernel (region-specific LDDMM)
# --------------------------------------------------------------------------- #
from MIR.models import SpatiallyVaryingKernel, KernelWeightHead


def _random_weights(kernel, seed=0):
    torch.manual_seed(seed)
    return kernel.weights_from_logits(torch.randn(1, kernel.n_kernels, *SIZE) * 3)


def test_spatially_varying_kernel_is_symmetric_psd_and_reduces_to_gaussian():
    kernel = SpatiallyVaryingKernel([1.0, 2.0, 4.0], weight_smooth_sigma=1.0)
    w = _random_weights(kernel)
    assert torch.allclose(w.sum(dim=1), torch.ones(1, *SIZE), atol=1e-5) and w.min() > 0
    p, q = torch.randn(1, 3, *SIZE), torch.randn(1, 3, *SIZE)
    assert abs(kernel.inner(p, kernel(q, w)) - kernel.inner(q, kernel(p, w))) < 1e-4
    assert kernel.inner(p, kernel(p, w)) > 0
    onehot = torch.zeros_like(w)
    onehot[:, 1] = 1
    assert torch.allclose(kernel(p, onehot), FourierKernel('gaussian', sigma=2.0)(p), atol=1e-5)
    assert kernel.omt(onehot * 0 + torch.tensor([0., 0., 1.]).view(1, 3, 1, 1, 1)) == 0
    assert 0 < kernel.omt(torch.full_like(w, 1 / 3)) <= 1
    assert kernel.effective_sigma(onehot).mean() == 2.0


def test_shoot_with_advected_weights_is_diffeomorphic_and_conserves_energy():
    kernel = SpatiallyVaryingKernel([1.0, 2.0, 4.0], weight_smooth_sigma=1.0)
    w = _random_weights(kernel)
    integ = LDDMMIntegrator(SIZE, 8, kernel)
    m0 = FourierKernel('gaussian', sigma=3.0)(torch.randn(1, 3, *SIZE))
    m0 = m0 / kernel(m0, w).abs().max() * 2.5
    out = integ.shoot(m0, weights=w)
    energies = torch.stack(out['energies'])
    assert ((energies / energies[0]) - 1).abs().max() < 0.15
    u, winv = out['flow'], out['inv_flow']
    st = SpatialTransformer(SIZE)
    residual = (u + st(winv, u)).norm(dim=1)[0, 2:-2, 2:-2, 2:-2]
    assert residual.mean() < 0.1 * u.norm(dim=1).mean()
    assert det3x3(jacobian_fd(u)).min() > 0
    with pytest.raises(ValueError):
        integ.shoot(m0)  # weights are required with a spatially varying kernel


def test_kernel_weight_head_init_favours_smoothest():
    head = KernelWeightHead(4, init='smoothest')
    kernel = SpatiallyVaryingKernel([1, 2, 4, 8])
    w = kernel.weights_from_logits(head(torch.rand(1, 1, *SIZE), torch.rand(1, 1, *SIZE), torch.randn(1, 3, *SIZE)))
    assert w[:, -1].mean() > 0.8


@pytest.mark.parametrize('param', ['shooting', 'tvf'])
def test_wrapper_spatially_varying(param):
    torch.manual_seed(0)
    model = LDDMMWrapper(_tiny_vxm(), param=param, time_steps=3, sigmas=[1, 2, 4], weight_smooth_sigma=0.5)
    mov, fix = torch.rand(1, 1, 32, 32, 32), torch.rand(1, 1, 32, 32, 32)
    out = model((mov, fix), return_dict=True)
    assert out['kernel_weights'].shape == (1, 3, 32, 32, 32)
    assert 0 <= out['omt'] <= 1
    (out['flow'].abs().mean() + out['energy'] + out['omt']).backward()
    assert model.weight_head.net[-1].weight.grad is not None
    assert model.backbone.flow.weight.grad is not None


def test_wrapper_spatially_varying_int_scale_and_custom_head():
    torch.manual_seed(0)
    head = lambda mov, fix, field: torch.zeros(field.shape[0], 2, *field.shape[2:])
    model = LDDMMWrapper(_tiny_vxm(), time_steps=2, sigmas=[1, 3], weight_head=head, int_scale=2)
    out = model((torch.rand(1, 1, 32, 32, 32), torch.rand(1, 1, 32, 32, 32)), return_dict=True)
    assert out['flow'].shape == (1, 3, 64, 64, 64)
    assert out['kernel_weights'].shape == (1, 2, 64, 64, 64)
    assert torch.allclose(out['kernel_weights'].sum(dim=1), torch.ones(1, 64, 64, 64), atol=1e-4)


def test_transmorph_lddmm_spatially_varying():
    torch.manual_seed(0)
    for param in ['shooting', 'tvf']:
        model = TransMorphLDDMM(_tiny_config(), param=param, time_steps=2, sigmas=[1, 2, 4])
        mov, fix = torch.rand(1, 1, 32, 32, 32), torch.rand(1, 1, 32, 32, 32)
        out = model((mov, fix), return_dict=True)
        assert out['kernel_weights'].shape == (1, 3, 32, 32, 32) and 'omt' in out
        (out['flow'].abs().mean() + out['energy'] + out['omt']).backward()
        assert model.wts_head[-1].weight.grad is not None


# --------------------------------------------------------------------------- #
# SPR integration
# --------------------------------------------------------------------------- #
from MIR.models import SPRHead
from MIR.deformation_regularizer import logBeta


def test_weights_from_spr_endpoints_and_interpolation():
    kernel = SpatiallyVaryingKernel([1.0, 2.0, 4.0], weight_smooth_sigma=0.0)
    ones = torch.ones(1, 1, *SIZE)
    w = kernel.weights_from_spr(ones)
    assert torch.allclose(w[:, -1], torch.ones(1, *SIZE), atol=1e-4)      # lam = 1 -> smoothest
    w = kernel.weights_from_spr(ones * 0)
    assert torch.allclose(w[:, 0], torch.ones(1, *SIZE), atol=1e-4)       # lam = 0 -> sharpest
    w = kernel.weights_from_spr(ones * 0.5)
    assert torch.allclose(w[:, 1], torch.ones(1, *SIZE), atol=1e-4)       # halfway -> middle sigma
    w = kernel.weights_from_spr(ones * 0.25)
    assert torch.allclose(w[:, 0], ones[:, 0] * 0.5, atol=1e-4) and torch.allclose(w[:, 1], ones[:, 0] * 0.5, atol=1e-4)
    assert torch.allclose(w.sum(dim=1), torch.ones(1, *SIZE), atol=1e-5)
    two = SpatiallyVaryingKernel([1.0, 4.0], weight_smooth_sigma=0.0)
    w = two.weights_from_spr(ones * 0.3)
    assert torch.allclose(w[:, 0], ones[:, 0] * 0.7, atol=1e-4)           # [1 - lam, lam]


def test_spr_head_range_and_init():
    head = SPRHead()
    lam = head(torch.rand(1, 1, *SIZE), torch.rand(1, 1, *SIZE), torch.randn(1, 3, *SIZE))
    assert lam.shape == (1, 1, *SIZE) and lam.min() > 0 and lam.max() <= 1
    assert abs(lam.mean().item() - torch.sigmoid(torch.tensor(2.0)).item()) < 1e-4


@pytest.mark.parametrize('spr,param', [('kernel', 'shooting'), ('kernel', 'tvf'), ('energy', 'shooting'), ('energy', 'tvf')])
def test_wrapper_spr_with_head(spr, param):
    torch.manual_seed(0)
    kw = dict(sigmas=[1, 2, 4]) if spr == 'kernel' else {}
    model = LDDMMWrapper(_tiny_vxm(), param=param, time_steps=2, spr=spr, **kw)
    mov, fix = torch.rand(1, 1, 32, 32, 32), torch.rand(1, 1, 32, 32, 32)
    out = model((mov, fix), return_dict=True)
    assert out['spr_weight'].shape == (1, 1, 32, 32, 32)
    assert ('kernel_weights' in out) == (spr == 'kernel')
    assert 'omt' not in out  # the Beta prior replaces the OMT penalty
    prior = logBeta()(out['spr_weight'], 1.1)
    (out['flow'].abs().mean() + out['energy'] + prior).backward()
    assert model.spr_head.net[-1].bias.grad is not None and model.spr_head.net[-1].bias.grad.abs().sum() > 0


def test_wrapper_spr_from_backbone():
    torch.manual_seed(0)

    class FakeSPRBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vxm = _tiny_vxm()
            self.lam_bias = torch.nn.Parameter(torch.zeros(1))

        def forward(self, inputs):
            flow = self.vxm(inputs)
            return flow, torch.sigmoid(self.lam_bias + flow[:, :1] * 0)

    model = LDDMMWrapper(FakeSPRBackbone(), time_steps=2, spr='kernel', spr_source='backbone', sigmas=[1, 3])
    mov, fix = torch.rand(1, 1, 32, 32, 32), torch.rand(1, 1, 32, 32, 32)
    out = model((mov, fix), return_dict=True)
    assert model.spr_head is None and out['spr_weight'].shape == (1, 1, 32, 32, 32)
    assert torch.allclose(out['kernel_weights'][:, 1], torch.full((1, 32, 32, 32), 0.5), atol=1e-3)
    (out['energy'] + logBeta()(out['spr_weight'], 1.1)).backward()
    assert model.backbone.lam_bias.grad is not None
    with pytest.raises(ValueError):
        LDDMMWrapper(_tiny_vxm(), spr='kernel', spr_source='backbone', sigmas=[1, 3])((mov, fix))
    with pytest.raises(ValueError):
        LDDMMWrapper(_tiny_vxm(), spr='kernel')  # sigmas required


def test_energy_mode_weights_the_integrand():
    kernel = FourierKernel()
    integ = LDDMMIntegrator(SIZE, 3, kernel)
    m0 = _smooth_momentum(kernel, v0_max=2.0)
    full = integ.shoot(m0)['energy']
    half = integ.shoot(m0, energy_weight=torch.full((1, 1, *SIZE), 0.5))['energy']
    assert torch.allclose(half, 0.5 * full, atol=1e-6)


@pytest.mark.parametrize('spr', ['kernel', 'energy'])
def test_transmorph_lddmm_spr(spr):
    torch.manual_seed(0)
    kw = dict(sigmas=[1, 2, 4]) if spr == 'kernel' else {}
    for param in ['shooting', 'tvf']:
        model = TransMorphLDDMM(_tiny_config(), param=param, time_steps=2, spr=spr, **kw)
        mov, fix = torch.rand(1, 1, 32, 32, 32), torch.rand(1, 1, 32, 32, 32)
        out = model((mov, fix), return_dict=True)
        assert out['spr_weight'].shape == (1, 1, 32, 32, 32) and 'omt' not in out
        (out['flow'].abs().mean() + out['energy'] + logBeta()(out['spr_weight'], 1.1)).backward()
        assert model.sigma_2.bias.grad is not None
