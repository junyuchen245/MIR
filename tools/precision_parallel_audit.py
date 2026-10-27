"""Mixed-precision and multi-GPU audit for MIR models, losses and regularizers.

For every model case of ``tools/golden_outputs.py`` (and every loss / regularizer / function case) this runs on
CUDA, with small 3-D inputs:

  fp32        reference forward pass
  amp16       ``torch.autocast("cuda", torch.float16)`` forward + backward
  amp_bf16    ``torch.autocast("cuda", torch.bfloat16)`` forward + backward
  half        pure float16 inference (``model.half()``, float16 inputs)
  dp          ``nn.DataParallel`` over all visible GPUs, batch 2, forward + backward, compared to one GPU

and reports, per case and mode, whether it ran, whether all outputs are finite, and the relative error of the
outputs against fp32. ``--ddp`` runs a 2-process ``DistributedDataParallel`` check (gradient synchronisation and
unused parameters) for every model instead.

Usage::

    python tools/precision_parallel_audit.py [--modes fp32,amp16,amp_bf16,half,dp] [--only NAME] [--json OUT]
    python tools/precision_parallel_audit.py --ddp [--only NAME] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import warnings

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import golden_outputs as G  # noqa: E402


def _floats(obj):
    """All floating-point tensors in a (nested) output."""
    if isinstance(obj, torch.Tensor):
        return [obj] if obj.is_floating_point() else []
    if isinstance(obj, dict):
        return [t for k in sorted(obj, key=str) for t in _floats(obj[k])]
    if isinstance(obj, (list, tuple)):
        return [t for v in obj for t in _floats(v)]
    return []


def _to(x, device=None, dtype=None):
    if isinstance(x, torch.Tensor):
        x = x.to(device) if device is not None else x
        return x.to(dtype) if (dtype is not None and x.is_floating_point()) else x
    if isinstance(x, (list, tuple)):
        return type(x)(_to(v, device, dtype) for v in x)
    if isinstance(x, dict):
        return {k: _to(v, device, dtype) for k, v in x.items()}
    return x


def _rel(a, b):
    a, b = a.detach().double(), b.detach().double().to(a.device)
    if a.shape != b.shape:
        return float("nan")
    return float((a - b).norm() / (b.norm() + 1e-12))


def _scalar(outs):
    ts = [t for t in _floats(outs) if t.requires_grad]
    return sum(t.float().mean() for t in ts) if ts else None


class _Call(nn.Module):
    """Wrap ``call(model)`` (which closes over fixed inputs) as ``forward(*inputs)`` for DataParallel."""

    def __init__(self, model, fn):
        super().__init__()
        self.model, self.fn = model, fn

    def forward(self, *inputs):
        return self.fn(self.model, *inputs)


# model cases with explicit inputs (so they can be cast / batched); (build, inputs, fn(model, *inputs))
def model_cases():
    import MIR.models as M

    cases = {}
    pair = lambda b=1: (G.vol(b=b, s=1), G.vol(b=b, s=2))  # noqa: E731
    raw = G.model_cases()
    hyper_scalar = {"HyperVFA", "HyperVxmDense", "HyperTransMorphTVF", "HyperTransMorphTVFSPR"}
    for name, (build, _call) in raw.items():
        if name in (
            "SpatialTransformer",
            "SpatialTransformer_nearest",
            "VecInt",
            "FourierKernel_helmholtz",
            "FourierKernel_gaussian",
        ):
            continue
        if name.startswith(("SynthesisHead", "EfficientAdvanced", "ConvNeXt")):
            cases[name] = (build, lambda b=1: (G.vol(c=8, b=b),), lambda m, x: m(x))
        elif name == "AdvancedDecoder3D":
            cases[name] = (
                build,
                lambda b=1: (G.vol(c=32, b=b), G.vol(c=64, n=G.N // 2, b=b), G.vol(c=128, n=G.N // 4, b=b)),
                lambda m, a, c, d: m([a, c, d]),
            )
        elif name == "GroupNet3D":
            cases[name] = (build, lambda b=1: (G.vol(b=b).unsqueeze(1).repeat(1, 3, 1, 1, 1, 1),), lambda m, x: m(x))
        elif name == "LessNetUNet":
            cases[name] = (build, pair, lambda m, a, c: m(a, c))
        elif name in hyper_scalar:
            cases[name] = (build, lambda b=1: pair(b) + (torch.full((1, 1), 0.3),), lambda m, a, c, h: m((a, c), h))
        elif name == "HyperVFASPR":
            cases[name] = (build, lambda b=1: pair(b) + (torch.tensor([[0.3, 0.5]]),), lambda m, a, c, h: m((a, c), h))
        elif name.startswith(("TransMorphLDDMM", "LDDMMWrapper")):
            cases[name] = (build, pair, lambda m, a, c: m((a, c), return_dict=True))
        else:
            cases[name] = (build, pair, lambda m, a, c: m((a, c)))
    cases["SpatialTransformer"] = (
        lambda: M.SpatialTransformer((G.N,) * 3),
        lambda b=1: (G.vol(b=b), G.flow().repeat(b, 1, 1, 1, 1)),
        lambda m, x, f: m(x, f),
    )
    cases["VecInt"] = (
        lambda: M.VecInt((G.N,) * 3, 7),
        lambda b=1: (G.flow(scale=1.0).repeat(b, 1, 1, 1, 1),),
        lambda m, f: m(f),
    )
    cases["FourierKernel"] = (
        lambda: M.FourierKernel("helmholtz", alpha=2.0, gamma=1.0),
        lambda b=1: (G.flow().repeat(b, 1, 1, 1, 1),),
        lambda m, f: m(f),
    )
    return cases


def loss_cases():
    """(name, factory, inputs_fn) for every discovered loss / regularizer / metric and the explicit function cases."""
    out = []
    for name, (kind, obj) in G.discover_callables().items():
        if kind != "class":
            continue
        out.append((name, obj))
    return out


DEVICE = "cuda:0"


def run_model(name, build, inputs_fn, fn, mode):
    dev = torch.device(DEVICE)
    G.seed(0)
    model = build()
    G.seed(0)
    if mode == "dp":
        n = torch.cuda.device_count()
        if n < 2:
            return {"status": "skip", "msg": "needs 2 GPUs"}
        ids = [dev.index] + [i for i in range(n) if i != dev.index]
        model = model.to(dev).eval()
        ref_inputs = _to(inputs_fn(2), dev)
        dp = nn.DataParallel(_Call(model, fn), device_ids=ids)
        with torch.no_grad():
            ref = fn(model, *ref_inputs)
            out = dp(*ref_inputs)
        rels = [_rel(a, b) for a, b in zip(_floats(out), _floats(ref))]
        dp.train()
        s = _scalar(dp(*ref_inputs))
        if s is not None:
            s.backward()
        return {
            "status": "ok",
            "finite": all(bool(torch.isfinite(t).all()) for t in _floats(out)),
            "rel": max(rels or [0]),
        }
    model = model.to(dev)
    inputs = _to(inputs_fn(1), dev)
    model.eval()
    with torch.no_grad():
        ref = fn(model, *inputs)
    if mode == "fp32":
        return {"status": "ok", "finite": all(bool(torch.isfinite(t).all()) for t in _floats(ref)), "rel": 0.0}
    if mode in ("amp16", "amp_bf16"):
        model.train()
        dtype = torch.float16 if mode == "amp16" else torch.bfloat16
        with torch.autocast("cuda", dtype=dtype):
            out = fn(model, *inputs)
            s = _scalar(out)
        if s is not None:
            s.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if grads and not all(bool(torch.isfinite(g).all()) for g in grads):
                return {"status": "ok", "finite": False, "rel": float("nan"), "msg": "non-finite gradients"}
        with torch.no_grad():
            model.eval()
            with torch.autocast("cuda", dtype=dtype):
                out = fn(model, *inputs)
    elif mode == "half":
        model = model.half().eval()
        with torch.no_grad():
            out = fn(model, *_to(inputs, dtype=torch.float16))
    rels = [_rel(a.float(), b) for a, b in zip(_floats(out), _floats(ref))]
    return {
        "status": "ok",
        "finite": all(bool(torch.isfinite(t).all()) for t in _floats(out)),
        "rel": max(rels or [0]),
        "out_dtypes": sorted({str(t.dtype) for t in _floats(out)}),
    }


def run_loss(name, cls, mode):
    dev = torch.device(DEVICE)
    pat = G.REF_PATTERNS.get(name) if hasattr(G, "REF_PATTERNS") else None
    patterns = [pat] if pat else list(G.ARG_PATTERNS)
    last = None
    for p in patterns:
        try:
            G.seed(0)
            loss = cls()
            args = _to(G.ARG_PATTERNS[p](), dev)
            ref = loss(*args)
            break
        except Exception as e:  # noqa: BLE001
            last = e
    else:
        return {"status": "skip", "msg": f"no pattern: {type(last).__name__}"}
    if mode == "fp32":
        return {"status": "ok", "finite": all(bool(torch.isfinite(t).all()) for t in _floats(ref)), "rel": 0.0}
    if mode in ("amp16", "amp_bf16"):
        dtype = torch.float16 if mode == "amp16" else torch.bfloat16
        args_g = [
            a.clone().requires_grad_(True) if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in args
        ]
        with torch.autocast("cuda", dtype=dtype):
            out = loss(*args_g)
        s = _scalar(out)
        if s is not None:
            s.backward()
    elif mode == "half":
        out = loss(*_to(args, dtype=torch.float16))
    else:
        return {"status": "skip"}
    rels = [_rel(a.float(), b) for a, b in zip(_floats(out), _floats(ref))]
    return {
        "status": "ok",
        "finite": all(bool(torch.isfinite(t).all()) for t in _floats(out)),
        "rel": max(rels or [0]),
        "out_dtypes": sorted({str(t.dtype) for t in _floats(out)}),
    }


def _guard(f, *a):
    try:
        return f(*a)
    except Exception as e:  # noqa: BLE001
        tb = traceback.extract_tb(e.__traceback__)
        where = next(
            (f"{os.path.basename(fr.filename)}:{fr.lineno}" for fr in reversed(tb) if "/MIR/" in fr.filename), "?"
        )
        return {"status": "error", "msg": f"{type(e).__name__}: {str(e)[:160]}", "where": where}
    finally:
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------------------------- DDP
def _ddp_worker(rank, world, names, port, out_path):
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    warnings.simplefilter("ignore")
    cap = float(os.environ.get("MIR_AUDIT_CAP_GB", "0"))
    if cap:
        total = torch.cuda.get_device_properties(rank).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total), rank)
    results = {}
    cases = model_cases()
    for name in names:
        build, inputs_fn, fn = cases[name]

        def run(build=build, inputs_fn=inputs_fn, fn=fn):
            G.seed(0)
            model = build().cuda(rank).train()
            ddp = nn.parallel.DistributedDataParallel(_Call(model, fn), device_ids=[rank])
            G.seed(rank)
            out = ddp(*_to(inputs_fn(1), torch.device("cuda", rank)))
            s = _scalar(out)
            if s is None:
                return {"status": "ok", "msg": "no differentiable output"}
            s.backward()
            # gradients must be identical on both ranks after the all-reduce
            g = torch.cat(
                [p.grad.flatten() for p in model.parameters() if p.grad is not None]
                or [torch.zeros(1, device=s.device)]
            )
            other = g.clone()
            dist.broadcast(other, src=0)
            unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
            return {
                "status": "ok",
                "grad_sync": bool(torch.allclose(g, other)),
                "unused_params": len(unused),
                "unused_examples": unused[:3],
            }

        try:
            results[name] = run()
        except Exception as e:  # noqa: BLE001
            results[name] = {"status": "error", "msg": f"{type(e).__name__}: {str(e)[:200]}"}
        dist.barrier()
    if rank == 0:
        json.dump(results, open(out_path, "w"), indent=1)
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modes", default="fp32,amp16,amp_bf16,half,dp")
    ap.add_argument("--only", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--ddp", action="store_true")
    ap.add_argument("--device", default="cuda:0", help="device for the single-GPU modes")
    ap.add_argument(
        "--cap-gb",
        type=float,
        default=None,
        help="limit this process to N GB on every GPU (protects other jobs on shared GPUs)",
    )
    a = ap.parse_args()
    global DEVICE
    DEVICE = a.device
    if a.cap_gb and not a.ddp:
        current = torch.cuda.current_device()  # set_per_process_memory_fraction switches the current device (torch 2.0)
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 2**30
            torch.cuda.set_per_process_memory_fraction(min(1.0, a.cap_gb / total), i)
        torch.cuda.set_device(torch.device(DEVICE).index if torch.device(DEVICE).index is not None else current)
    warnings.simplefilter("ignore")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if a.ddp:
        import torch.multiprocessing as mp

        names = [n for n in model_cases() if a.only is None or a.only in n]
        out = a.json or "/tmp/mir_ddp_audit.json"
        if a.cap_gb:
            os.environ["MIR_AUDIT_CAP_GB"] = str(a.cap_gb)
        mp.spawn(_ddp_worker, args=(2, names, 29517, out), nprocs=2, join=True)
        res = json.load(open(out))
        for n, r in res.items():
            print(f"{n:42s} {r}")
        return
    modes = a.modes.split(",")
    results = {}
    for name, (build, inputs_fn, fn) in model_cases().items():
        if a.only and a.only not in name:
            continue
        results[name] = {m: _guard(run_model, name, build, inputs_fn, fn, m) for m in modes}
        print(f"{name:40s} " + "  ".join(f"{m}:{_fmt(results[name][m])}" for m in modes), flush=True)
    for name, cls in loss_cases():
        if a.only and a.only not in name:
            continue
        results[name] = {m: _guard(run_loss, name, cls, m) for m in modes if m != "dp"}
        print(
            f"{name.split('.')[-1]:40s} " + "  ".join(f"{m}:{_fmt(results[name][m])}" for m in results[name]),
            flush=True,
        )
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)


def _fmt(r):
    if r["status"] == "skip":
        return "skip"
    if r["status"] == "error":
        return f"ERR({r.get('where', '?')})"
    if not r.get("finite", True):
        return "NONFINITE"
    return f"{r['rel']:.1e}"


if __name__ == "__main__":
    main()
