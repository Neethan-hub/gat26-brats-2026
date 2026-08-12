#!/usr/bin/env python3
"""GAT-26 release-environment determinism tests (`python3 tests/test_release_env.py`).

Fail closed if the release container's Python environment is not fully pinned: every line in
configs/release/requirements.lock.txt and every package installed by a `pip install` in the
Dockerfile must carry an exact `==` version. This blocks a silent upgrade/downgrade of torch,
CUDA-related packages, nnU-Net, or the reconstruction dependencies at build time.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REQ = REPO / "configs" / "release" / "requirements.lock.txt"
DOCKERFILE = REPO / "configs" / "release" / "Dockerfile"

FAILS = 0
PIN_RE = re.compile(r"^[A-Za-z0-9_.\-]+==[0-9][^\s]*$")   # name==<version starting with a digit>
# inference-critical packages that MUST appear pinned in the requirements
REQUIRED = {"nnunetv2", "numpy", "scipy", "scikit-image", "nibabel", "SimpleITK",
            "acvl_utils", "dynamic_network_architectures", "batchgenerators", "batchgeneratorsv2",
            "einops"}


def check(name, cond, detail=""):
    global FAILS
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{'' if cond else ' :: ' + str(detail)}")
    if not cond:
        FAILS += 1


def req_lines():
    return [ln.strip() for ln in REQ.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def main():
    print("test_release_env:")
    lines = req_lines()

    # 1. every requirement line is an exact == pin
    unpinned = [ln for ln in lines if not PIN_RE.match(ln)]
    check("all_requirements_exact_pinned", not unpinned, unpinned)

    # 2. no floating operators anywhere in the requirements
    floating = [ln for ln in lines if any(op in ln for op in (">=", "<=", "~=", "!=", ">", "<", "*"))]
    check("no_floating_operators", not floating, floating)

    # 3. the inference-critical packages are present and pinned
    names = {ln.split("==")[0] for ln in lines}
    missing = sorted(REQUIRED - names)
    check("critical_packages_present", not missing, missing)

    # 4. numpy / scipy / scikit-image explicitly pinned (called out by the correction)
    for p in ("numpy", "scipy", "scikit-image"):
        pinned = any(ln.split("==")[0] == p and PIN_RE.match(ln) for ln in lines)
        check(f"{p}_pinned", pinned)

    # 5. Dockerfile: every pip-installed package is pinned; torch is pinned; no bare installs
    df = DOCKERFILE.read_text(encoding="utf-8").replace("\\\n", " ").replace("\\\r\n", " ")
    pip_pkgs = []
    for chunk in df.split("pip install")[1:]:
        chunk = chunk.split("\n")[0].split("&&")[0]      # bound to this install command only
        for tok in chunk.split():
            if tok.startswith("-") or tok.startswith("http") or tok.endswith(".txt") or "/" in tok:
                continue                      # flags, index URL, -r target, paths
            pip_pkgs.append(tok)
    unpinned_df = [t for t in pip_pkgs if "==" not in t]
    check("dockerfile_pip_all_pinned", not unpinned_df, unpinned_df)
    check("torch_pinned_exact", bool(re.search(r"torch==2\.8\.0(\b|\+)", df)))
    check("dockerfile_installs_requirements_lock", "requirements.lock.txt" in df)

    print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
