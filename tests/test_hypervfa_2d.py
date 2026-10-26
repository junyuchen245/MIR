"""Tests for the 2D code paths used by the browser playground tutorial:
HyperVFA / HyperVFASPR in 2D, the ``nb_hyp_units`` option, ``LocalGrad2d``, and the
ONNX-friendly patch extraction in the VFA decoders."""
import pytest
import torch

from MIR.models import HyperVFA, HyperVFASPR, VFA, VFASPR, SpatialTransformer
import MIR.models.configs_VFA as configs_VFA
from MIR.deformation_regularizer import LocalGrad2d, LocalGrad3d, logBeta


def _cfg(img_size, hyp_units=8):
    c = configs_VFA.get_VFA_default_config()
    c.img_size = img_size
    c.downsamples = 2
    c.start_channels = 4
    c.matching_channels = 4
    c.max_channels = 16
    c.nb_hyp_units = hyp_units
    return c


def test_hypervfaspr_2d_forward_shapes():
    torch.manual_seed(0)
    model = HyperVFASPR(_cfg((32, 32)), device="cpu").eval()
    mov, fix = torch.rand(2, 1, 32, 32), torch.rand(2, 1, 32, 32)
    flow, wts = model((mov, fix), torch.tensor([[0.3, 0.7]]))
    assert flow.shape == (2, 2, 32, 32)
    assert wts.shape == (2, 1, 32, 32)
    assert torch.all(wts > 0) and torch.all(wts <= 1)
    assert torch.isfinite(flow).all()


def test_hypervfa_2d_and_3d_forward():
    torch.manual_seed(0)
    m2 = HyperVFA(_cfg((32, 32)), device="cpu").eval()
    assert m2((torch.rand(1, 1, 32, 32), torch.rand(1, 1, 32, 32)), torch.tensor([[0.5]])).shape == (1, 2, 32, 32)
    m3 = HyperVFA(_cfg((16, 16, 16)), device="cpu").eval()
    assert m3((torch.rand(1, 1, 16, 16, 16), torch.rand(1, 1, 16, 16, 16)), torch.tensor([[0.5]])).shape == (1, 3, 16, 16, 16)


def test_nb_hyp_units_sets_embedding_width():
    narrow = HyperVFASPR(_cfg((32, 32), hyp_units=8), device="cpu")
    wide = HyperVFASPR(_cfg((32, 32), hyp_units=24), device="cpu")
    assert narrow.hyper_model.fcs[-1].out_features == 8
    assert wide.hyper_model.fcs[-1].out_features == 24
    assert sum(p.numel() for p in wide.parameters()) > sum(p.numel() for p in narrow.parameters())
    assert narrow.nb_hyp_units == 8 and wide.nb_hyp_units == 24


def test_hypervfa_default_width_without_config_key():
    c = _cfg((32, 32))
    del c.nb_hyp_units
    model = HyperVFA(c, device="cpu")
    assert model.hyper_model.fcs[-1].out_features == 96


def test_localgrad2d_matches_manual_and_3d_convention():
    torch.manual_seed(0)
    flow = torch.randn(2, 2, 12, 10)
    wts = torch.rand(2, 1, 12, 10)
    reg = LocalGrad2d(penalty="l2")(flow, wts)
    dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]) ** 2
    dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]) ** 2
    manual = (torch.mean(dx * wts[:, :, :, 1:]) + torch.mean(dy * wts[:, :, 1:, :])) / 2.0
    assert torch.allclose(reg, manual)
    # l1 penalty and uniform weights: plain mean absolute finite difference
    ones = torch.ones_like(wts)
    l1 = LocalGrad2d(penalty="l1")(flow, ones)
    manual_l1 = (torch.mean(dx.sqrt()) + torch.mean(dy.sqrt())) / 2.0
    assert torch.allclose(l1, manual_l1)
    assert LocalGrad3d(penalty="l2")(flow.unsqueeze(-1).repeat(1, 1, 1, 1, 2), ones.unsqueeze(-1).repeat(1, 1, 1, 1, 2)).item() > 0


def test_logbeta_prior_pulls_weights_up():
    wts = torch.full((1, 1, 8, 8), 0.5, requires_grad=True)
    loss = logBeta()(wts, torch.tensor(1.2))
    loss.backward()
    assert loss.item() > 0
    assert torch.all(wts.grad < 0)  # decreasing the loss means increasing lambda


@pytest.mark.parametrize("model_cls", [VFA, VFASPR, HyperVFA])
def test_export_friendly_patch_extraction_matches_unfold(model_cls):
    torch.manual_seed(0)
    cfg = _cfg((32, 32))
    model = model_cls(cfg, device="cpu").eval()
    mov, fix = torch.rand(1, 1, 32, 32), torch.rand(1, 1, 32, 32)
    args = ((mov, fix),) if model_cls in (VFA, VFASPR) else ((mov, fix), torch.tensor([[0.5]]))
    with torch.no_grad():
        ref = model(*args)
        model.decoder.export_friendly = True
        alt = model(*args)
    ref = ref[0] if isinstance(ref, tuple) else ref
    alt = alt[0] if isinstance(alt, tuple) else alt
    assert torch.allclose(ref, alt, atol=1e-6)


def test_onnx_export_hypervfaspr_2d(tmp_path):
    onnx = pytest.importorskip("onnx")
    torch.manual_seed(0)
    model = HyperVFASPR(_cfg((32, 32)), device="cpu").eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, mov, fix, h):
            return self.m((mov, fix), h)

    mov, fix, h = torch.rand(1, 1, 32, 32), torch.rand(1, 1, 32, 32), torch.tensor([[0.5, 0.5]])
    path = str(tmp_path / "hypervfaspr_2d.onnx")
    torch.onnx.export(Wrap(model), (mov, fix, h), path, input_names=["moving", "fixed", "hyper"],
                      output_names=["flow", "weights"], opset_version=16, do_constant_folding=True)
    onnx.checker.check_model(path)
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    flow, wts = sess.run(None, {"moving": mov.numpy(), "fixed": fix.numpy(), "hyper": h.numpy()})
    with torch.no_grad():
        ref_flow, ref_wts = model((mov, fix), h)
    assert abs(flow - ref_flow.numpy()).max() < 1e-3
    assert abs(wts - ref_wts.numpy()).max() < 1e-4


def test_spatial_transformer_2d_identity():
    stn = SpatialTransformer((8, 6))
    img = torch.rand(1, 1, 8, 6)
    assert torch.allclose(stn(img, torch.zeros(1, 2, 8, 6)), img, atol=1e-6)
