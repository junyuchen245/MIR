"""Regression tests for bugs fixed during the 2026-10 code-quality pass (see docs/about/changelog.rst)."""

import pytest
import torch

import MIR
import MIR.image_similarity as S
import MIR.models as M
import MIR.models.configs_TransMorph as configs_TransMorph

N = 32


def _vol(seed, c=1, n=N):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, c, n, n, n, generator=g)


def _tm_config():
    cfg = configs_TransMorph.get_3DTransMorph3Lvl_config()
    cfg.img_size = (N, N, N)
    cfg.window_size = (2, 2, 2)
    cfg.out_chan = 3
    return cfg


def test_version_attribute():
    assert isinstance(MIR.__version__, str) and MIR.__version__


@pytest.mark.parametrize("module", ["MIR.models.registration_utils", "MIR.utils.registration_utils"])
def test_spatial_transformer_ignores_stale_grid_in_checkpoint(module):
    """Checkpoints saved before the grid became a non-persistent buffer must still load with strict=True."""
    import importlib

    st_cls = importlib.import_module(module).SpatialTransformer
    st = st_cls((8, 8, 8))
    state = {"grid": torch.zeros(1, 3, 8, 8, 8)}
    st.load_state_dict(state, strict=True)  # must not raise "Unexpected key(s)"
    assert "grid" not in st.state_dict()


def test_transmorph_tvf_loads_legacy_checkpoint_strict():
    model = M.TransMorphTVF(_tm_config(), time_steps=2)
    legacy = dict(model.state_dict())
    legacy["spatial_trans.grid"] = torch.zeros(1, 3, N, N, N)  # what old checkpoints contain
    legacy["spatial_trans_.grid"] = torch.zeros(1, 3, N // 2, N // 2, N // 2)
    model.load_state_dict(legacy, strict=True)


def test_hyper_transmorph_tvf_spr_forward_runs():
    model = M.HyperTransMorphTVFSPR(_tm_config(), time_steps=2).eval()
    with torch.no_grad():
        out = model((_vol(1), _vol(2)), torch.tensor([[0.3]]))
    flow_pos, flow_neg, weight = out
    assert flow_pos.shape == (1, 3, N, N, N)
    assert weight.shape[1] == 1 and float(weight.min()) > 0


@pytest.mark.parametrize(
    "factory",
    [
        S.CorrRatio,
        S.LocalCorrRatio,
        S.MutualInformation,
        S.localMutualInformation,
        S.NCC_gauss,
        S.NCC_vxm,
        S.NCC,
        S.MIND_loss,
    ],
    ids=lambda f: f.__name__,
)
def test_losses_run_on_cpu(factory):
    """These losses used to hard-code CUDA; they must work on CPU tensors."""
    loss = factory()
    with torch.no_grad():
        val = loss(_vol(1), _vol(2))
    assert torch.isfinite(torch.as_tensor(val)).all()


def test_invert_warp_via_velocity_on_cpu():
    g = torch.Generator().manual_seed(0)
    warp = torch.randn(1, 3, 16, 16, 16, generator=g) * 0.5
    inv = M.invert_warp_via_velocity(warp, nb_steps=3, iters=3)
    assert inv.shape == warp.shape and inv.device.type == "cpu"


def test_affine_transformer_on_cpu():
    tf = M.AffineTransformer()
    src = _vol(1, n=16)
    zeros = torch.zeros(1, 3)
    out = tf(src, zeros, torch.ones(1, 3), zeros, torch.zeros(1, 6))
    warped = out[0] if isinstance(out, (tuple, list)) else out
    assert warped.shape == src.shape
    assert torch.allclose(warped, src, atol=1e-5)  # identity parameters leave the image unchanged


# --------------------------------------------------------------------------------------------------------------------
# Second round of fixes (2026-10-27)
# --------------------------------------------------------------------------------------------------------------------
def _smooth_field(seed, n=24, amp=3.0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(1, 3, n, n, n, generator=g)
    for _ in range(4):
        v = torch.nn.functional.avg_pool3d(torch.nn.functional.pad(v, (2,) * 6, mode="replicate"), 5, 1)
    return v / v.abs().max() * amp


def test_fit_warp_to_svf_output_names():
    warp = _smooth_field(0, n=16, amp=1.0)
    kw = dict(nb_steps=4, iters=5, verbose=False)
    torch.manual_seed(0)
    vel = M.fit_warp_to_svf(warp, output_type="velocity", **kw)
    torch.manual_seed(0)
    legacy_vel = M.fit_warp_to_svf(warp, output_type="disp", **kw)  # legacy name, unchanged behaviour
    torch.manual_seed(0)
    disp = M.fit_warp_to_svf(warp, output_type="displacement", **kw)
    torch.manual_seed(0)
    legacy_disp = M.fit_warp_to_svf(warp, output_type="svf", **kw)
    assert torch.equal(vel, legacy_vel) and torch.equal(disp, legacy_disp)
    assert not torch.allclose(vel, disp)
    with pytest.raises(ValueError):
        M.fit_warp_to_svf(warp, objective="huber", **kw)
    with pytest.raises(ValueError):
        M.fit_warp_to_svf(warp, output_type="field", **kw)
    v_fast = M.fit_warp_to_svf_fast(warp, iters=3, refine_iters=1, output_type="velocity")
    d_fast = M.fit_warp_to_svf_fast(warp, iters=3, refine_iters=1, output_type="displacement")
    assert torch.allclose(M.VecInt((16, 16, 16), 5)(v_fast), d_fast, atol=1e-5)


def test_jacobian_determinant_affine_and_batch():
    from MIR.models.registration_utils import jacobian_determinant

    n = 12
    A = torch.tensor([[0.10, 0.02, -0.03], [0.00, -0.05, 0.04], [0.01, 0.03, 0.08]])
    grid = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float32)] * 3, indexing="ij"))
    u = torch.einsum("ij,jdhw->idhw", A, grid)[None]
    det = jacobian_determinant(torch.cat([u, torch.zeros_like(u)]))  # batch of 2: affine field and identity
    assert det.shape == (2, 1, n, n, n)
    assert torch.allclose(det[0], torch.det(torch.eye(3) + A).expand(1, n, n, n), atol=1e-5)
    assert torch.allclose(det[1], torch.ones(1, n, n, n))


def test_fastncc_matches_reference_ncc():
    a, b = _vol(1), _vol(2)
    fast = S.FastNCC(win=9)(a, b)
    ref = S.NCC_vxm(win=[9, 9, 9])(a, b)
    assert torch.allclose(fast, ref, rtol=1e-4)


def test_vfa_return_full_inverse_is_exp_of_negative_velocity():
    import MIR.models.configs_VFA as configs_VFA

    cfg = configs_VFA.get_VFA_default_config()
    cfg.img_size = (N, N, N)
    model = M.VFA(cfg, device="cpu", SVF=True, SVF_steps=4, return_full=True).eval()
    pair = (_vol(1), _vol(2))
    with torch.no_grad():
        _, _, pos, neg = model(pair)
        model.SVF = False  # same weights, raw velocity output
        v = model(pair)
    assert torch.allclose(pos, model.integrate(v), atol=1e-6)
    assert torch.allclose(neg, model.integrate(-v), atol=1e-6)


def test_gradicon_exact_jacobian_on_affine_field():
    import MIR.deformation_regularizer as R

    D, H, W = 20, 24, 28
    A = torch.tensor([[0.05, -0.02, 0.03], [0.01, 0.04, -0.03], [-0.02, 0.02, 0.06]])
    grid = torch.stack(torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in (D, H, W)], indexing="ij"))
    u = torch.einsum("ij,jdhw->idhw", A, grid)[None]
    reg = R.GradICONExact3d((D, H, W))
    pts = torch.tensor([[[10.0, 12.0, 7.0], [20.5, 3.25, 15.0]]])  # (x, y, z) = (axis 2, axis 1, axis 0)
    J = reg._jacobian_samples(u, pts)[0]
    P = torch.tensor([[0, 0, 1.0], [0, 1.0, 0], [1.0, 0, 0]])
    assert torch.allclose(J - torch.eye(3), P @ A @ P.T, atol=1e-3)


def test_mutual_information_respects_minval():
    x = -_vol(3)  # values in [-1, 0]
    mi = S.MutualInformation(minval=-1.0, maxval=1.0)
    clamped_to_zero = S.MutualInformation(minval=0.0, maxval=1.0)
    assert float(-mi(x, x)) > 10 * float(-clamped_to_zero(x, x)) + 1e-3


def test_multivariate_kl_matches_scalar_kl():
    import MIR.deformation_regularizer as R

    g = torch.Generator().manual_seed(0)
    mk = lambda: (torch.randn(1, 3, 4, 4, 4, generator=g), 0.3 * torch.randn(1, 3, 4, 4, 4, generator=g))  # noqa: E731
    P, Q = mk(), mk()
    kl = R.KL_divergence()(P, Q)  # mean over all elements
    mvkl = R.MultiVariateKL_divergence()(P, Q)  # mean over voxels of the sum over the 3 components
    assert torch.allclose(mvkl, 3 * kl, rtol=1e-4)


def test_meanstream_state_stays_non_trainable():
    ms = M.MeanStream(cap=10, in_shape=(4, 4, 4)).train()
    ms(torch.rand(2, 3, 4, 4, 4))
    assert not ms.mean.requires_grad and not ms.count.requires_grad


def test_swin_defaults_construct_and_train_returns_self():
    import MIR.models.Swin_Transformer as swin

    model = swin.SwinTransformer(
        patch_size=4, in_chans=2, embed_dim=24, depths=[1, 1, 1, 1], num_heads=[1, 1, 1, 1], window_size=(2, 2, 2)
    )
    assert model.train() is model
    with torch.no_grad():
        outs = model(torch.rand(1, 2, 64, 64, 64))
    assert outs[-1].shape[1] == 24 * 8


def test_defswin_default_dwin_size_and_transvfa_default_channels():
    import importlib

    import MIR.models.Deformable_Swin_Transformer as dswin

    TV = importlib.import_module("MIR.models.TransVFA")  # the package re-exports a class of the same name

    dswin.DefSwinTransformer(
        img_size=(64, 64, 64), embed_dim=24, depths=[1, 1, 1, 1], num_heads=[1, 1, 1, 1], window_size=(2, 2, 2)
    )
    TV.Feature2VFA(_tm_config())  # default vfa_channels used to raise IndexError


def test_multimorph_fixes():
    import MIR.models.MultiMorph as MM

    x = torch.rand(1, 3, 8, 8, 8)
    assert MM.ResizeTransform(2, 3)(x).shape[2:] == (4, 4, 4)  # downsampling path (factor < 1)
    conv = MM.FastMeanConv3d(2, 4, 3, padding=1).eval()
    batch = torch.rand(2, 3, 2, 8, 8, 8)  # [B, N, C, H, W, D]
    with torch.no_grad():
        joint = conv(batch)
        single = torch.cat([conv(batch[i : i + 1]) for i in range(2)])
    assert torch.allclose(joint, single, atol=1e-6)  # every sample uses its own group summary


def test_vfa_return_orginal_raises_clear_error():
    import MIR.models.configs_VFA as configs_VFA

    cfg = configs_VFA.get_VFA_default_config()
    cfg.img_size = (N, N, N)
    model = M.VFA(cfg, device="cpu", return_orginal=True)
    with pytest.raises(NotImplementedError):
        model((_vol(1), _vol(2)))


def test_ncc_variants_edge_cases():
    a, b = _vol(1), _vol(2)
    loss = S.NCC_mok(win=None)
    assert torch.isfinite(loss(a, b)) and torch.isfinite(loss(a, b))  # second call used to fail
    a1, b1 = a.reshape(1, 1, -1)[..., :256], b.reshape(1, 1, -1)[..., :256]
    assert torch.isfinite(S.NCC_vfa()(a1, b1))  # 1-D input
    with pytest.raises(AssertionError):
        S.NCC_gauss()(a[..., 0], b[..., 0])  # 2-D input: clear error


def test_dice_seg_loss_ignore_index_outside_class_range():
    logits = torch.randn(1, 3, 8, 8, 8)
    target = torch.randint(0, 3, (1, 8, 8, 8))
    target_ignored = target.clone()
    target_ignored[:, :2] = 255
    loss = S.DiceSegLoss(3, ignore_index=255)(logits, target_ignored)
    assert torch.isfinite(loss)


def test_center_crop_pad_uses_constructor_padding_value():
    from MIR.utils import CenterCropPad3D

    out = CenterCropPad3D((6, 6, 6), padding_value=5.0)(torch.zeros(1, 1, 4, 4, 4))
    assert float(out.max()) == 5.0
    out = CenterCropPad3D((6, 6, 6), padding_value=5.0)(torch.zeros(1, 1, 4, 4, 4), 2.0)  # explicit value wins
    assert float(out.max()) == 2.0


def test_deedsbcv_lookup_ignores_current_directory(tmp_path, monkeypatch):
    import MIR.utils.deedsbcv_binary as D

    (tmp_path / "deedsBCV").write_text("#!/bin/sh\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MIR_DEEDSBCV_DIR", raising=False)
    monkeypatch.setattr(D, "get_deedsbcv_dir", lambda *a, **k: tmp_path / "cache")
    assert D.get_deedsbcv_executable("deedsBCV").parent == tmp_path / "cache"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sin_positional_encoding_follows_device():
    import MIR.models.Swin_Transformer as swin

    enc = swin.SinPositionalEncoding3D(24)
    out = enc(torch.rand(1, 24, 4, 4, 4, device="cuda"))
    assert out.is_cuda


def test_random_multicontrast_remap_unbatched_input():
    from MIR import RandomMultiContrastRemap

    torch.manual_seed(0)
    img = torch.rand(1, 16, 16, 16)
    mask = torch.zeros(1, 16, 16, 16, dtype=torch.bool)
    mask[:, 3:13, 3:13, 3:13] = True
    out = RandomMultiContrastRemap()(img, mask)
    out = out[0] if isinstance(out, (tuple, list)) else out
    assert out.shape[-3:] == img.shape[-3:]
