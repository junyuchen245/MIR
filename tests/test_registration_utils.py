import torch

from MIR.models.registration_utils import SpatialTransformer, VecInt


def test_spatial_transformer_identity_2d():
    torch.manual_seed(0)
    src = torch.rand(1, 1, 5, 6)
    flow = torch.zeros(1, 2, 5, 6)
    st = SpatialTransformer((5, 6))
    out = st(src, flow)
    assert torch.allclose(out, src, atol=1e-5)


def test_vecint_zero_steps_identity():
    torch.manual_seed(0)
    vec = torch.randn(1, 2, 5, 6)
    integrator = VecInt((5, 6), nsteps=0)
    out = integrator(vec)
    assert torch.allclose(out, vec, atol=1e-5)
