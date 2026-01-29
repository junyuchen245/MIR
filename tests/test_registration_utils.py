import torch

from MIR.models.registration_utils import SpatialTransformer, VecInt

def test_spatial_transformer_zero_flow_shape_finite_3d():
    torch.manual_seed(0)
    src = torch.ones(1, 1, 24, 24, 24)
    flow = torch.zeros(1, 3, 24, 24, 24)
    st = SpatialTransformer((24, 24, 24))
    out = st(src, flow)
    assert out.shape == src.shape
    assert torch.isfinite(out).all()


def test_vecint_zero_steps_identity():
    torch.manual_seed(0)
    vec = torch.randn(1, 3, 24, 24, 24)
    integrator = VecInt((24, 24, 24), nsteps=0)
    out = integrator(vec)
    assert torch.allclose(out, vec, atol=1e-5)
