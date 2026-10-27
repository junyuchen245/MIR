"""Golden-output regression harness for the MIR package (behaviour-preservation gate for refactoring).

Runs a fixed set of deterministic CPU computations -- model forward passes, losses, regularizers, accuracy measures,
utilities, data augmentation -- with fixed seeds and records a numerical fingerprint of every output, plus every model's
``state_dict`` parameter names and shapes (pretrained checkpoints load by name and shape). Locally available pretrained
checkpoints are loaded with ``strict=True`` and run on a fixed input.

Usage::

    python tools/golden_outputs.py --write golden.json      # record the reference
    python tools/golden_outputs.py --check golden.json      # compare; exit code 1 on any difference

Losses, regularizers and accuracy measures are discovered automatically: every public callable is tried with a fixed
list of argument patterns and the first pattern that runs is recorded (and replayed by ``--check``).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
import traceback
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
N = 32  # spatial size of the synthetic 3-D test volumes
RTOL, ATOL = 1e-6, 1e-7  # CPU runs are expected to be bit-identical; this only absorbs float formatting


def seed(s: int = 0) -> None:
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)


def vol(c=1, n=N, b=1, s=1):
    g = torch.Generator().manual_seed(s)
    x = torch.rand(b, c, n, n, n, generator=g)
    return torch.nn.functional.avg_pool3d(torch.nn.functional.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate"), 3, 1)


def seg(n=N, labels=4, s=3):
    g = torch.Generator().manual_seed(s)
    return torch.randint(0, labels, (1, 1, n, n, n), generator=g).float()


def flow(n=N, scale=2.0, s=5):
    g = torch.Generator().manual_seed(s)
    f = torch.randn(1, 3, n, n, n, generator=g) * scale
    for _ in range(3):
        f = torch.nn.functional.avg_pool3d(torch.nn.functional.pad(f, (1, 1, 1, 1, 1, 1), mode="replicate"), 3, 1)
    return f


# ---------------------------------------------------------------------------------------------------- fingerprints
def _arrays(obj, prefix="out"):
    if isinstance(obj, torch.Tensor):
        yield prefix, obj.detach().double().cpu().numpy()
    elif isinstance(obj, np.ndarray) and obj.dtype != object:
        yield prefix, obj.astype(np.float64)
    elif isinstance(obj, (bool, int, float, np.floating, np.integer)):
        yield prefix, np.asarray(float(obj))
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            yield from _arrays(obj[k], f"{prefix}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _arrays(v, f"{prefix}[{i}]")


def fingerprint(obj) -> dict:
    fp = {}
    for name, a in _arrays(obj):
        flat = a.reshape(-1)
        idx = np.linspace(0, max(flat.size - 1, 0), num=min(8, flat.size)).astype(int) if flat.size else []
        fp[name] = {
            "shape": list(a.shape),
            "sum": float(np.nansum(flat)),
            "abs": float(np.nansum(np.abs(flat))),
            "sq": float(np.nansum(flat**2)),
            "nan": int(np.isnan(flat).sum()),
            "samples": [float(flat[i]) for i in idx],
        }
    return fp


def state_dict_signature(model) -> list:
    return [[k, list(v.shape)] for k, v in model.state_dict().items()]


# ---------------------------------------------------------------------------------------------------------- cases
def _tm_config(name="get_3DTransMorph3Lvl_config"):
    import MIR.models.configs_TransMorph as C

    cfg = getattr(C, name)()
    cfg.img_size = (N, N, N)
    cfg.window_size = (2, 2, 2)
    cfg.out_chan = 3
    return cfg


def _vfa_config():
    import MIR.models.configs_VFA as C

    cfg = C.get_VFA_default_config()
    cfg.img_size = (N, N, N)
    return cfg


def model_cases():
    """name -> builder returning (model, call) where call(model) produces the outputs."""
    import MIR.models as M

    pair = (vol(s=1), vol(s=2))
    cases = {}

    def add(name, build, call):
        cases[name] = (build, call)

    add("TransMorph_SVF", lambda: M.TransMorph(_tm_config()), lambda m: m(pair))
    add("TransMorph_noSVF", lambda: M.TransMorph(_tm_config(), SVF=False), lambda m: m(pair))
    add("TransMorphTVF", lambda: M.TransMorphTVF(_tm_config(), time_steps=3), lambda m: m(pair))
    add("TransMorphTVFSPR", lambda: M.TransMorphTVFSPR(_tm_config(), time_steps=3), lambda m: m(pair))
    add("VFA", lambda: M.VFA(_vfa_config(), device="cpu"), lambda m: m(pair))
    add("VFA_SVF", lambda: M.VFA(_vfa_config(), device="cpu", SVF=True, SVF_steps=3), lambda m: m(pair))
    add("VFASPR", lambda: M.VFASPR(_vfa_config(), device="cpu"), lambda m: m(pair))
    add("HyperVFA", lambda: M.HyperVFA(_vfa_config(), device="cpu"), lambda m: m(pair, torch.tensor([[0.3]])))
    add(
        "HyperVFASPR", lambda: M.HyperVFASPR(_vfa_config(), device="cpu"), lambda m: m(pair, torch.tensor([[0.3, 0.5]]))
    )
    add("TransVFA", lambda: M.TransVFA(_tm_config(), _vfa_config(), device="cpu"), lambda m: m(pair))

    def vxm_cfg():
        import MIR.models.configs_VoxelMorph as C

        c = C.get_VXM_default_config()
        c.img_size = (N, N, N)
        return c

    for cfg_name in ("get_VXM_default_config", "get_VXM_1_config", "get_VXM_BJ_config"):

        def b(cfg_name=cfg_name):
            import MIR.models.configs_VoxelMorph as C

            c = getattr(C, cfg_name)()
            c.img_size = (N, N, N)
            return M.VxmDense(c)

        add(f"VxmDense_{cfg_name}", b, lambda m: m(pair))
    add("HyperVxmDense", lambda: M.HyperVxmDense(vxm_cfg()), lambda m: m(pair, torch.tensor([[0.3]])))
    for cls, cfgf in (
        ("LessNet", "get_LessNet_default_config"),
        ("LessNet", "get_LessNet_SVF_config"),
        ("LessNetSPR", "get_LessNet_SPR_config"),
    ):

        def b(cls=cls, cfgf=cfgf):
            import MIR.models.configs_LessNet as C

            c = getattr(C, cfgf)()
            c.img_size = (N, N, N)
            return getattr(M, cls)(c)

        add(f"{cls}_{cfgf}", b, lambda m: m(pair))
    add("LessNetUNet", lambda: M.LessNetUNet(), lambda m: m(*pair))
    for cls, cfgf in (
        ("FourierNet", "get_FourierNet_default_config"),
        ("FourierNet", "get_FourierNet_SVF_config"),
        ("FourierNetSPR", "get_FourierNet_SPR_config"),
        ("FourierNetPlus", "get_FourierNetPlus_default_config"),
        ("FourierNetPlus", "get_FourierNetPlus_SVF_config"),
        ("FourierNetPlusSPR", "get_FourierNetPlus_SPR_config"),
    ):

        def b(cls=cls, cfgf=cfgf):
            import MIR.models.configs_FourierNet as C

            c = getattr(C, cfgf)()
            c.img_size = (N, N, N)
            return getattr(M, cls)(c)

        add(f"{cls}_{cfgf}", b, lambda m: m(pair))
    add(
        "HyperTransMorphTVF",
        lambda: M.HyperTransMorphTVF(_tm_config(), time_steps=3),
        lambda m: m(pair, torch.tensor([[0.3]])),
    )
    add(
        "HyperTransMorphTVFSPR",
        lambda: M.HyperTransMorphTVFSPR(_tm_config(), time_steps=3),
        lambda m: m(pair, torch.tensor([[0.3]])),
    )
    add(
        "TransMorphLDDMM_shooting",
        lambda: M.TransMorphLDDMM(_tm_config(), time_steps=3),
        lambda m: m(pair, return_dict=True),
    )
    add(
        "TransMorphLDDMM_tvf",
        lambda: M.TransMorphLDDMM(_tm_config(), param="tvf", time_steps=3),
        lambda m: m(pair, return_dict=True),
    )
    add(
        "TransMorphLDDMM_svk",
        lambda: M.TransMorphLDDMM(_tm_config(), time_steps=3, sigmas=[1, 2]),
        lambda m: m(pair, return_dict=True),
    )
    add(
        "TransMorphLDDMM_spr",
        lambda: M.TransMorphLDDMM(_tm_config(), time_steps=3, sigmas=[1, 2], spr="kernel"),
        lambda m: m(pair, return_dict=True),
    )
    add(
        "LDDMMWrapper_vxm",
        lambda: M.LDDMMWrapper(M.VxmDense(vxm_cfg()), time_steps=3),
        lambda m: m(pair, return_dict=True),
    )
    add(
        "GroupNet3D",
        lambda: M.GroupNet3D(features=[8, 8], img_size=[N, N, N], diffeo_steps=3),
        lambda m: m(vol(b=1, c=1).repeat(1, 1, 1, 1, 1).unsqueeze(1).repeat(1, 3, 1, 1, 1, 1)),
    )
    add("SynthesisHead3D", lambda: M.SynthesisHead3D(8), lambda m: m(vol(c=8)))
    add("SynthesisHead3DAdvanced", lambda: M.SynthesisHead3DAdvanced(8, num_res_blocks=2), lambda m: m(vol(c=8)))
    add("EfficientAdvancedSynthHead3D", lambda: M.EfficientAdvancedSynthHead3D(8, num_blocks=2), lambda m: m(vol(c=8)))
    add("ConvNeXtSynthHead3D", lambda: M.ConvNeXtSynthHead3D(8, num_blocks=2), lambda m: m(vol(c=8)))
    add(
        "AdvancedDecoder3D",
        lambda: M.AdvancedDecoder3D([32, 64, 128], aspp_out=32, num_classes=5),
        lambda m: m([vol(c=32, n=N), vol(c=64, n=N // 2), vol(c=128, n=N // 4)]),
    )
    add("SpatialTransformer", lambda: M.SpatialTransformer((N, N, N)), lambda m: m(vol(), flow()))
    add(
        "SpatialTransformer_nearest",
        lambda: M.SpatialTransformer((N, N, N), mode="nearest"),
        lambda m: m(seg(), flow()),
    )
    add("VecInt", lambda: M.VecInt((N, N, N), 7), lambda m: m(flow(scale=1.0)))
    add("FourierKernel_helmholtz", lambda: M.FourierKernel("helmholtz", alpha=2.0, gamma=1.0), lambda m: m(flow()))
    add("FourierKernel_gaussian", lambda: M.FourierKernel("gaussian", sigma=2.0), lambda m: m(flow()))
    return cases


def gpu_model_cases():
    import MIR.models as M

    if not torch.cuda.is_available():
        return {}
    pair = (vol(s=1).cuda(), vol(s=2).cuda())
    return {
        "TransMorph_dwin": (
            lambda: M.TransMorph(_tm_config("get_3DTransMorphDWin3Lvl_config"), swin_type="dswin").cuda(),
            lambda m: m(pair),
        )
    }


def function_cases():
    """Deterministic calls of standalone functions."""
    import MIR.models as M
    import MIR.utils as U

    cases = {
        "jacobian_fd": lambda: M.jacobian_fd(flow()),
        "det3x3": lambda: M.det3x3(M.jacobian_fd(flow())),
        "mk_grid_img": lambda: U.mk_grid_img(8, 1, (N, N, N)),
        "mk_grid_img_dim1": lambda: U.mk_grid_img(8, 1, (N, N, N), dim=1),
        "fit_warp_to_svf": lambda: M.fit_warp_to_svf(flow(scale=1.0), nb_steps=3, iters=5),
        "invert_warp_via_velocity": lambda: M.invert_warp_via_velocity(flow(scale=1.0), nb_steps=3, iters=5),
    }
    import MIR.accuracy_measures as A
    import MIR.deformation_regularizer as R
    import MIR.image_similarity as S

    probs = lambda: torch.softmax(vol(c=4, s=7) * 5, 1)  # noqa: E731
    lab = lambda s=3: seg(s=s)[:, 0].long()  # noqa: E731
    img2d = lambda s: vol(s=s)[..., N // 2]  # noqa: E731
    flow_np = lambda: flow()[0].numpy()  # noqa: E731
    cases.update(
        {
            "DiceLoss": lambda: S.DiceLoss(num_class=4)(probs(), seg().long()),
            "DiceSegLoss": lambda: S.DiceSegLoss(4)(vol(c=4, s=7), lab()),
            "SegmentationLoss": lambda: S.SegmentationLoss(4)(vol(c=4, s=7), lab()),
            "SSIM2D": lambda: S.SSIM2D()(img2d(1), img2d(2)),
            "ssim": lambda: S.ssim(img2d(1), img2d(2)),
            "create_window_3D": lambda: S.create_window_3D(5, 1),
            "DisplacementRegularizer_bending": lambda: R.DisplacementRegularizer("bending")(flow(), None),
            "DisplacementRegularizer_gradient-l2": lambda: R.DisplacementRegularizer("gradient-l2")(flow(), None),
            "GradICON3d": lambda: R.GradICON3d((N, N, N))(flow(s=5), flow(s=6)),
            "GradICONExact3d": lambda: R.GradICONExact3d((N, N, N))(flow(s=5), flow(s=6)),
            "KL_divergence": lambda: R.KL_divergence()(probs(), torch.softmax(vol(c=4, s=8) * 5, 1)),
            "dice_val_all": lambda: A.dice_val_all(seg(s=3).long(), seg(s=4).long(), 4),
            "dice_val_VOI": lambda: A.dice_val_VOI(seg(s=3), seg(s=4), num_clus=4),
            "calc_jac_dets": lambda: A.calc_jac_dets(flow_np()),
            "calc_Jstar_1": lambda: A.calc_Jstar_1(flow_np()),
            "calc_Jstar_2": lambda: A.calc_Jstar_2(flow_np()),
        }
    )
    return cases


ARG_PATTERNS = {
    "img,img": lambda: (vol(s=1), vol(s=2)),
    "seg,seg": lambda: (seg(s=3), seg(s=4)),
    "flow": lambda: (flow(),),
    "flow,img": lambda: (flow(), vol(s=1)),
    "img,img,flow": lambda: (vol(s=1), vol(s=2), flow()),
    "seg,seg,int": lambda: (seg(s=3), seg(s=4), 4),
    "img": lambda: (vol(s=1),),
}


def discover_callables():
    """Public classes/functions in the loss/regularizer/metric modules, instantiated with defaults."""
    import MIR.accuracy_measures as A
    import MIR.deformation_regularizer as R
    import MIR.image_similarity as S

    found = {}
    for mod in (S, R, A):
        for name in sorted(getattr(mod, "__all__", None) or [n for n in dir(mod) if not n.startswith("_")]):
            obj = getattr(mod, name, None)
            if inspect.isclass(obj) and issubclass(obj, torch.nn.Module):
                found[f"{mod.__name__}.{name}"] = ("class", obj)
            elif inspect.isfunction(obj) and obj.__module__.startswith("MIR"):
                found[f"{mod.__name__}.{name}"] = ("function", obj)
    return found


def _to(args, device):
    return tuple(a.to(device) if isinstance(a, torch.Tensor) else a for a in args)


def run_discovered(kind, obj, pattern=None, device=None):
    """Try argument patterns on CPU; fall back to CUDA for callables that hard-code it. Returns (pattern, device, out)."""
    fn = obj
    if kind == "class":
        seed(0)
        fn = obj()
    patterns = [pattern] if pattern else list(ARG_PATTERNS)
    devices = [device] if device else (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    last = None
    for dev in devices:
        for p in patterns:
            seed(0)
            try:
                with torch.no_grad():
                    out = fn(*_to(ARG_PATTERNS[p](), dev))
                return p, dev, out
            except Exception as e:  # noqa: BLE001 -- probing argument patterns
                last = e
    raise RuntimeError(f"no argument pattern worked: {type(last).__name__}: {last}")


def pretrained_cases():
    """Real checkpoints shipped with the tutorials: strict load + forward on a fixed input."""
    import MIR.models as M
    import MIR.models.configs_TransMorph as CT
    import MIR.models.configs_VFA as CV

    tut = os.path.join(REPO, "tutorials")
    cases = {}

    def vfa(path, key):
        def run():
            c = CV.get_VFA_default_config()
            c.img_size = (160, 224, 192)
            m = M.VFA(c, device="cpu")
            m.load_state_dict(torch.load(path, map_location="cpu")[key], strict=True)
            m.eval()
            x = (vol(n=1, s=1), vol(n=1, s=2))
            x = tuple(
                torch.nn.functional.interpolate(t, size=(160, 224, 192), mode="trilinear")
                for t in (torch.rand(1, 1, 20, 28, 24, generator=torch.Generator().manual_seed(s)) for s in (1, 2))
            )
            with torch.no_grad():
                return m, m(x)

        return run

    def tm(path):
        def run():
            c = CT.get_3DTransMorph3Lvl_config()
            c.img_size = (80, 112, 96)
            c.window_size = (2, 3, 3)
            c.out_chan = 3
            m = M.TransMorphTVF(c, time_steps=7)
            m.load_state_dict(torch.load(path, map_location="cpu")["state_dict"], strict=True)
            m.eval()
            x = tuple(
                torch.nn.functional.interpolate(t, size=(80, 112, 96), mode="trilinear")
                for t in (torch.rand(1, 1, 10, 14, 12, generator=torch.Generator().manual_seed(s)) for s in (1, 2))
            )
            with torch.no_grad():
                return m, m(x)

        return run

    for label, path, fn in (
        (
            "VFA-LUMIR24",
            os.path.join(tut, "lumir25_VFA/pretrained_wts/VFA_LUMIR24.pth"),
            lambda p: vfa(p, "model_state_dict"),
        ),
        (
            "VFA-LUMIR25",
            os.path.join(tut, "lumir25_VFA/pretrained_wts/VFA_LUMIR25.pth"),
            lambda p: vfa(p, "state_dict"),
        ),
        (
            "TransMorphTVF-LUMIR24",
            os.path.join(tut, "lumir25_TransMorph/pretrained_wts/TransMorphTVF_LUMIR24.pth.tar"),
            tm,
        ),
    ):
        if os.path.exists(path):
            cases[label] = fn(path)
    return cases


# ------------------------------------------------------------------------------------------------------------ main
def collect(reference=None) -> dict:
    warnings.simplefilter("ignore")
    torch.set_num_threads(8)
    res = {"models": {}, "functions": {}, "discovered": {}, "pretrained": {}}
    for name, (build, call) in model_cases().items():
        try:
            seed(0)
            m = build()
            m.eval()
            seed(0)
            with torch.no_grad():
                out = call(m)
            res["models"][name] = {"state_dict": state_dict_signature(m), "outputs": fingerprint(out)}
        except Exception as e:  # noqa: BLE001
            res["models"][name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    res["gpu_models"] = {}
    for name, (build, call) in gpu_model_cases().items():
        try:
            seed(0)
            m = build()
            m.eval()
            seed(0)
            with torch.no_grad():
                out = call(m)
            res["gpu_models"][name] = {"state_dict": state_dict_signature(m), "outputs": fingerprint(out)}
        except Exception as e:  # noqa: BLE001
            res["gpu_models"][name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    for name, fn in function_cases().items():
        try:
            seed(0)
            res["functions"][name] = {"outputs": fingerprint(fn())}
        except Exception as e:  # noqa: BLE001
            res["functions"][name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    for name, (kind, obj) in discover_callables().items():
        pat = dev = None
        if reference is not None:
            ref = reference.get("discovered", {}).get(name, {})
            pat, dev = ref.get("pattern"), ref.get("device")
            if pat is None:
                continue
        try:
            p, d, out = run_discovered(kind, obj, pat, dev)
            res["discovered"][name] = {"pattern": p, "device": d, "outputs": fingerprint(out)}
        except Exception as e:  # noqa: BLE001
            res["discovered"][name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    for name, run in pretrained_cases().items():
        try:
            seed(0)
            m, out = run()
            res["pretrained"][name] = {"state_dict": state_dict_signature(m), "outputs": fingerprint(out)}
        except Exception as e:  # noqa: BLE001
            res["pretrained"][name] = {"error": f"{type(e).__name__}: {traceback.format_exc()[-400:]}"}
    return res


GPU_RTOL = 1e-4  # CUDA reductions are not bit-reproducible across runs


def _close(a, b, rtol=RTOL):
    return abs(a - b) <= ATOL + rtol * max(abs(a), abs(b))


def compare(old, new) -> list:
    problems = []
    for section in ("models", "gpu_models", "functions", "discovered", "pretrained"):
        for name, o in old.get(section, {}).items():
            rtol = GPU_RTOL if section == "gpu_models" or o.get("device") == "cuda" else RTOL
            n = new.get(section, {}).get(name)
            if n is None:
                if "error" not in o:  # cases that never ran are not replayed
                    problems.append(f"{section}/{name}: missing")
                continue
            if "error" in o:
                if "error" not in n:
                    continue  # a previously failing case now runs: not a regression
                continue
            if "error" in n:
                problems.append(f"{section}/{name}: now fails: {n['error']}")
                continue
            if o.get("state_dict") != n.get("state_dict"):
                problems.append(
                    f"{section}/{name}: state_dict names/shapes changed (pretrained weights would not load)"
                )
            for k, fo in o["outputs"].items():
                fn = n["outputs"].get(k)
                if fn is None:
                    problems.append(f"{section}/{name}: output {k} missing")
                    continue
                if fo["shape"] != fn["shape"] or fo["nan"] != fn["nan"]:
                    problems.append(f"{section}/{name}: output {k} shape/NaN changed {fo['shape']} -> {fn['shape']}")
                    continue
                vals_o = [fo["sum"], fo["abs"], fo["sq"]] + fo["samples"]
                vals_n = [fn["sum"], fn["abs"], fn["sq"]] + fn["samples"]
                if not all(_close(a, b, rtol) for a, b in zip(vals_o, vals_n)):
                    d = max(abs(a - b) for a, b in zip(vals_o, vals_n))
                    problems.append(f"{section}/{name}: output {k} values changed (max abs diff {d:.3g})")
            if set(n["outputs"]) - set(o["outputs"]):
                problems.append(f"{section}/{name}: new outputs {sorted(set(n['outputs']) - set(o['outputs']))}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", metavar="JSON")
    g.add_argument("--check", metavar="JSON")
    a = ap.parse_args()
    if a.write:
        res = collect()
        with open(a.write, "w") as f:
            json.dump(res, f, indent=1)
        for section, entries in res.items():
            ok = sum("error" not in v for v in entries.values())
            print(f"{section:11s}: {ok}/{len(entries)} cases recorded")
            for k, v in entries.items():
                if "error" in v:
                    print(f"   - {k}: {v['error'][:160]}")
        return 0
    with open(a.check) as f:
        old = json.load(f)
    problems = compare(old, collect(reference=old))
    for p in problems:
        print("  !", p)
    total = sum(len(v) for v in old.values())
    print(f"{len(problems)} problem(s) across {total} recorded cases")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
