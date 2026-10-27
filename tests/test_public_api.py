"""Public-API stability: no first-party name may disappear or change its call parameters.

Tutorials, downstream projects and pretrained-weight loading depend on deep import paths (for example
``MIR.models.registration_utils`` or ``MIR.models.VFA.Decoder``), so the whole first-party module tree is treated as
public. The reference is ``tests/data/public_api.json`` (``python tools/api_snapshot.py --write-test-data``).
Type annotations and new names are allowed; removals and parameter changes (names, kinds, defaults) are not.
"""

import importlib
import json
import os
import sys
import warnings

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import api_snapshot  # noqa: E402

with open(os.path.join(HERE, "data", "public_api.json")) as f:
    REFERENCE = json.load(f)["modules"]


@pytest.mark.parametrize("module_name", sorted(REFERENCE))
def test_module_api_unchanged(module_name):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:  # optional third-party dependency missing in this environment
        if e.name and e.name.split(".")[0] != "MIR":
            pytest.skip(f"optional dependency not installed: {e.name}")
        raise
    problems, unavailable = [], []
    for name, ref in REFERENCE[module_name].items():
        if not hasattr(module, name):
            problems.append(f"removed: {name}")
            continue
        if getattr(module, name) is None and ref["kind"] != "value":
            # optional component whose import failed in this environment (e.g. SITReg, fireants): MIR binds the
            # name to None on purpose; the API itself is checked wherever the dependency is installed
            unavailable.append(name)
            continue
        cur = api_snapshot._describe(name, getattr(module, name), module_name)
        if cur["kind"] != ref["kind"]:
            problems.append(f"{name}: kind {ref['kind']} -> {cur['kind']}")
        elif ref["kind"] == "function" and cur.get("params") != ref["params"]:
            problems.append(f"{name}: parameters changed {ref['params']} -> {cur.get('params')}")
        elif ref["kind"] == "class":
            if cur.get("init_params") != ref["init_params"]:
                problems.append(f"{name}: constructor parameters changed")
            for m, params in ref.get("methods", {}).items():
                if m not in cur.get("methods", {}):
                    problems.append(f"{name}.{m}: method removed")
                elif cur["methods"][m] != params:
                    problems.append(f"{name}.{m}: parameters changed {params} -> {cur['methods'][m]}")
        elif ref["kind"] == "module" and cur.get("target") != ref["target"]:
            problems.append(f"{name}: now refers to module {cur.get('target')}")
        elif "repr" in ref and cur.get("repr") != ref["repr"]:
            problems.append(f"{name}: value changed")
    assert not problems, "\n".join(problems)
    if unavailable:
        warnings.warn(f"{module_name}: optional components unavailable here: {unavailable}", stacklevel=1)
