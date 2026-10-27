"""Mixed precision (fp16 / bf16) and multi-process (DDP) behaviour; CPU-only tests run in CI, GPU tests skip."""

import os
import socket
import tempfile

import pytest
import torch
import torch.nn as nn

import MIR.deformation_regularizer as R
import MIR.image_similarity as S
import MIR.models as M
import MIR.models.configs_VoxelMorph as configs_VoxelMorph
from MIR.utils import distributed as D
from MIR.utils import precision as P

N = 16


def _vol(seed, c=1, n=N, b=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, c, n, n, n, generator=g)


def _vxm():
    cfg = configs_VoxelMorph.get_VXM_default_config()
    cfg.img_size = (N, N, N)
    return M.VxmDense(cfg)


# ------------------------------------------------------------------------------------------------------ precision
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "loss_factory", [S.NCC_vxm, S.MutualInformation, S.CorrRatio, S.PCC, S.SSIM3D, R.Grad3d], ids=lambda f: f.__name__
)
def test_losses_compute_in_float32(loss_factory, dtype):
    a, b = _vol(1), _vol(2)
    if loss_factory is R.Grad3d:
        a = a.repeat(1, 3, 1, 1, 1)
        args_lowp, args_ref = (a.to(dtype), None), (a.to(dtype).float(), None)
    else:
        args_lowp, args_ref = (a.to(dtype), b.to(dtype)), (a.to(dtype).float(), b.to(dtype).float())
    loss = loss_factory()
    out = loss(*args_lowp)
    ref = loss(*args_ref)  # the same (quantised) values in float32
    assert out.dtype == torch.float32
    assert torch.equal(out, ref)


def test_losses_ignore_active_autocast():
    a, b = _vol(1), _vol(2)
    ref = S.NCC_vxm()(a, b)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = S.NCC_vxm()(a, b)
    assert out.dtype == torch.float32 and torch.equal(out, ref)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_spatial_transformer_and_vecint_keep_dtype(dtype):
    st, vi = M.SpatialTransformer((N, N, N)), M.VecInt((N, N, N), 4)
    img = _vol(1)
    flow = (_vol(2, c=3) - 0.5) * 2
    warped = st(img.to(dtype), flow.to(dtype))
    disp = vi(flow.to(dtype))
    assert warped.dtype == dtype and disp.dtype == dtype
    assert torch.allclose(warped.float(), st(img.to(dtype).float(), flow.to(dtype).float()), atol=1e-2)
    assert torch.allclose(disp.float(), vi(flow.to(dtype).float()), atol=2e-2)


def test_model_trains_under_cpu_bf16_autocast():
    torch.manual_seed(0)
    model = _vxm()
    mov, fix = _vol(1), _vol(2)
    st = M.SpatialTransformer((N, N, N))
    with P.autocast("bf16", device_type="cpu"):
        flow = model((mov, fix))
        loss = S.NCC_vxm()(fix, st(mov, flow)) + R.Grad3d()(flow, None)
    loss.backward()
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_precision_helpers():
    assert P.precision_dtype("fp16") is torch.float16
    with pytest.raises(ValueError):
        P.precision_dtype("fp8")
    with P.autocast("fp32"):
        pass
    assert not P.grad_scaler("bf16").is_enabled()
    assert not P.grad_scaler("fp32").is_enabled()
    with pytest.raises(ValueError):
        P.autocast("fp16", device_type="cpu")


def test_hypernetwork_accepts_identical_per_sample_hyperparameters():
    from MIR._amp import shared_hyperparameter

    h = torch.tensor([[0.3], [0.3]])
    assert torch.equal(shared_hyperparameter(h), h[:1])
    with pytest.raises(ValueError):
        shared_hyperparameter(torch.tensor([[0.3], [0.4]]))


# ---------------------------------------------------------------------------------------------------- distributed
def test_distributed_helpers_single_process():
    model = _vxm()
    assert D.get_rank() == 0 and D.get_world_size() == 1 and D.is_main_process()
    assert D.wrap_model(model, mode="none") is model
    assert D.wrap_model(model) is model  # "auto" never uses DataParallel for a CPU model
    dp = nn.DataParallel(model)
    assert D.unwrap_model(dp) is model
    sd = D.state_dict_for_saving(dp)
    assert list(sd) == list(model.state_dict())
    prefixed = {"module." + k: v for k, v in model.state_dict().items()}
    model.load_state_dict(D.strip_module_prefix(prefixed), strict=True)
    assert D.all_reduce_mean(1.5) == 1.5 and D.distributed_sampler(list(range(4))) is None


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ddp_worker(rank, world, port, out_dir):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank)
    )
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    model = D.wrap_model(_vxm(), mode="ddp", device="cpu")
    mov, fix = _vol(10 + rank), _vol(20 + rank)  # different data on each rank
    flow = model((mov, fix))
    (flow**2).mean().backward()
    grads = torch.cat([p.grad.flatten() for p in D.unwrap_model(model).parameters() if p.grad is not None])
    ms = M.MeanStream(cap=100, in_shape=(4, 4, 4)).train()
    ms(torch.full((2, 3, 4, 4, 4), float(rank + 1)))  # rank 0 sees 1s, rank 1 sees 2s
    torch.save(
        {
            "grads": grads,
            "keys": list(D.state_dict_for_saving(model)),
            "mean": ms.mean.detach(),
            "count": ms.count.detach(),
            "loss": D.all_reduce_mean(torch.tensor(float(rank))),
        },
        os.path.join(out_dir, f"rank{rank}.pt"),
    )
    D.cleanup()


def test_ddp_two_processes_cpu():
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as d:
        mp.spawn(_ddp_worker, args=(2, _free_port(), d), nprocs=2, join=True)
        r0, r1 = (torch.load(os.path.join(d, f"rank{r}.pt")) for r in (0, 1))
    assert torch.allclose(r0["grads"], r1["grads"])  # gradients are synchronised
    assert r0["keys"] == list(_vxm().state_dict())  # checkpoints carry no "module." prefix
    assert torch.allclose(r0["mean"], r1["mean"]) and torch.allclose(r0["mean"], torch.full_like(r0["mean"], 1.5))
    assert float(r0["count"]) == 4.0  # both ranks counted the global batch
    assert float(r0["loss"]) == 0.5


# ----------------------------------------------------------------------------------------------------------- GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_gpu_autocast_training_step(precision):
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU without bf16")
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = _vxm().to(dev)
    st = M.SpatialTransformer((N, N, N)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = P.grad_scaler(precision)
    mov, fix = _vol(1).to(dev), _vol(2).to(dev)
    with P.autocast(precision):
        flow = model((mov, fix))
        loss = S.NCC_vxm()(fix, st(mov, flow)) + R.Grad3d()(flow, None)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    assert loss.dtype == torch.float32 and torch.isfinite(loss)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_dataparallel_matches_single_gpu():
    model = _vxm().cuda().eval()
    x = (_vol(1, b=2).cuda(), _vol(2, b=2).cuda())
    with torch.no_grad():
        ref = model(x)
        out = D.wrap_model(model, mode="dp")(x)
    assert torch.allclose(ref, out, atol=1e-4)
