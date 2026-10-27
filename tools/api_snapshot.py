"""Public-API snapshot for the MIR package: guards against breaking changes during refactoring.

Imports every module under ``MIR`` and records, for each public (non-underscore) attribute, its kind and -- for
functions, classes and their public methods -- the call signature. ``--check`` compares the live package against a
stored snapshot and fails on any removed module, removed name or changed signature. New names are reported but allowed, and so are
removed names that only re-exported a third-party object through an import (``np``, ``nn``, ``Optional``, ...).

Usage::

    python tools/api_snapshot.py --write api_snapshot.json     # record the current API
    python tools/api_snapshot.py --check api_snapshot.json     # compare (exit code 1 on breaking changes)
    python tools/api_snapshot.py --write-test-data             # refresh tests/data/public_api.json
"""

from __future__ import annotations

import argparse
import importlib
import os
import inspect
import json
import pkgutil
import re
import sys
import warnings


_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def _signature(obj) -> str:
    try:
        return _ADDRESS.sub("", str(inspect.signature(obj)))  # default values may print object addresses
    except (TypeError, ValueError):
        return "<no signature>"


def _params(obj) -> list:
    """Parameter names, kinds and defaults -- the parts of a signature callers depend on (annotations excluded)."""
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    return [
        [p.name, p.kind.name, _ADDRESS.sub("", repr(p.default)) if p.default is not p.empty else None]
        for p in sig.parameters.values()
    ]


def _describe(name: str, obj, module_name: str) -> dict:
    if inspect.ismodule(obj):
        return {"kind": "module", "target": obj.__name__}
    if inspect.isclass(obj):
        entry = {
            "kind": "class",
            "qualname": f"{obj.__module__}.{obj.__qualname__}",
            "init": _signature(obj),
            "init_params": _params(obj),
        }
        if obj.__module__.split(".")[0] == "MIR":
            methods = {}
            for m, v in inspect.getmembers(obj):
                if m.startswith("_") and m not in ("__call__",):
                    continue
                if inspect.isfunction(v) or inspect.ismethod(v):
                    methods[m] = _params(v)
                elif isinstance(inspect.getattr_static(obj, m, None), property):
                    methods[m] = "<property>"
            entry["methods"] = methods
        return entry
    if callable(obj):
        return {
            "kind": "function",
            "qualname": f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', name)}",
            "signature": _signature(obj),
            "params": _params(obj),
        }
    simple = (int, float, str, bool, tuple, type(None))
    return {
        "kind": "value",
        "type": type(obj).__name__,
        **({"repr": repr(obj)[:200]} if isinstance(obj, simple) else {}),
    }


def snapshot() -> dict:
    warnings.simplefilter("ignore")
    import MIR

    out = {"modules": {}, "import_errors": {}}
    names = ["MIR"] + [m.name for m in pkgutil.walk_packages(MIR.__path__, "MIR.", onerror=lambda n: None)]
    for mod_name in sorted(set(names)):
        try:
            mod = importlib.import_module(mod_name)
        except BaseException as e:  # noqa: BLE001 -- optional dependencies may be missing; record, do not fail
            out["import_errors"][mod_name] = f"{type(e).__name__}: {str(e)[:200]}"
            continue
        out["modules"][mod_name] = {
            n: _describe(n, getattr(mod, n), mod_name)
            for n in sorted(dir(mod))
            if not n.startswith("_") and hasattr(mod, n)
        }
    return out


def _third_party(entry: dict) -> bool:
    """True for names that merely re-export a non-MIR object (e.g. ``np``, ``nn``, ``Optional``) via an import."""
    if entry["kind"] == "module":
        return not entry["target"].startswith("MIR")
    if entry["kind"] in ("class", "function"):
        return not entry.get("qualname", "MIR").startswith("MIR")
    return entry["kind"] == "value" and entry.get("type") in (
        "_SpecialForm",
        "_GenericAlias",
        "TypeVar",
        "_SpecialGenericAlias",
    )


VENDORED = (
    "MIR.models.SITReg",
    "MIR.models.fireants",
    "MIR.models.convexAdam",
    "MIR.intensity_normalization",
    "MIR.random_image_generation.perlin_numpy",
)
TEST_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "data", "public_api.json")


def _own_methods(cls) -> set:
    """Public methods defined by MIR classes in the MRO (inherited torch.nn.Module methods are not MIR API)."""
    names = set()
    for k in cls.__mro__:
        if k.__module__.split(".")[0] == "MIR":
            names.update(n for n in vars(k) if not n.startswith("_") or n == "__call__")
    return names


def test_data() -> dict:
    """Compact first-party API record for tests/test_public_api.py (kinds and parameters only)."""
    snap = snapshot()
    out = {}
    for mod_name, names in snap["modules"].items():
        if mod_name.startswith(VENDORED):
            continue
        mod = importlib.import_module(mod_name)
        rec = {}
        for n, d in names.items():
            if _third_party(d):
                continue
            e = {"kind": d["kind"]}
            if d["kind"] == "function":
                e["params"] = d["params"]
            elif d["kind"] == "class":
                e["init_params"] = d["init_params"]
                own = _own_methods(getattr(mod, n))
                e["methods"] = {m: p for m, p in d.get("methods", {}).items() if m in own}
            elif d["kind"] == "module":
                e["target"] = d["target"]
            elif "repr" in d:
                e["repr"] = d["repr"]
            rec[n] = e
        out[mod_name] = rec
    return {
        "note": "Public API of first-party MIR modules, checked by tests/test_public_api.py. Regenerate only "
        "after an intended, backwards-compatible addition: python tools/api_snapshot.py --write-test-data",
        "modules": out,
    }


def compare(old: dict, new: dict) -> tuple[list[str], list[str]]:
    breaking, added = [], []
    for mod, names in old["modules"].items():
        if mod not in new["modules"]:
            breaking.append(f"module no longer importable: {mod} ({new['import_errors'].get(mod, 'missing')})")
            continue
        cur = new["modules"][mod]
        for n, d in names.items():
            if n not in cur:
                if _third_party(d):
                    added.append(f"removed third-party import (not MIR API): {mod}.{n}")
                else:
                    breaking.append(f"removed: {mod}.{n}")
                continue
            c = cur[n]
            if d["kind"] != c["kind"]:
                breaking.append(f"kind changed: {mod}.{n}: {d['kind']} -> {c['kind']}")
            elif d["kind"] == "function" and d.get("params") != c.get("params"):
                breaking.append(f"signature changed: {mod}.{n}{d['signature']} -> {c['signature']}")
            elif d["kind"] == "class":
                if d.get("init_params") != c.get("init_params"):
                    breaking.append(f"constructor changed: {mod}.{n}{d['init']} -> {c['init']}")
                for m, s in d.get("methods", {}).items():
                    if m not in c.get("methods", {}):
                        breaking.append(f"method removed: {mod}.{n}.{m}")
                    elif s != c["methods"][m]:
                        breaking.append(f"method signature changed: {mod}.{n}.{m}: {s} -> {c['methods'][m]}")
                for m in set(c.get("methods", {})) - set(d.get("methods", {})):
                    added.append(f"method added: {mod}.{n}.{m}")
            elif d["kind"] == "value" and "repr" in d and d.get("repr") != c.get("repr"):
                breaking.append(f"value changed: {mod}.{n}: {d['repr']} -> {c.get('repr')}")
        for n in set(cur) - set(names):
            added.append(f"added: {mod}.{n}")
    for mod in set(new["modules"]) - set(old["modules"]):
        added.append(f"new module: {mod}")
    return breaking, added


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":  # set/frozenset reprs depend on hash randomization
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", metavar="JSON")
    g.add_argument("--check", metavar="JSON")
    g.add_argument("--write-test-data", action="store_true", help=f"write {os.path.relpath(TEST_DATA)}")
    a = ap.parse_args()
    if a.write_test_data:
        data = test_data()
        with open(TEST_DATA, "w") as f:
            json.dump(data, f, indent=1, sort_keys=True)
        print(f"wrote {TEST_DATA}: {len(data['modules'])} modules")
        return 0
    snap = snapshot()
    if a.write:
        with open(a.write, "w") as f:
            json.dump(snap, f, indent=1, sort_keys=True)
        n = sum(len(v) for v in snap["modules"].values())
        print(
            f"wrote {a.write}: {len(snap['modules'])} modules, {n} public names, {len(snap['import_errors'])} import errors"
        )
        return 0
    with open(a.check) as f:
        old = json.load(f)
    breaking, added = compare(old, snap)
    for line in added:
        print("  +", line)
    for line in breaking:
        print("  !", line)
    print(f"{len(breaking)} breaking change(s), {len(added)} addition(s)")
    return 1 if breaking else 0


if __name__ == "__main__":
    sys.exit(main())
