#!/usr/bin/env python3
"""GAT-26 build-context preflight tests (`python3 tests/test_release_build_preflight.py`).

Verify the deterministic preflight: a complete context passes; missing COPY sources / missing
checkpoints fail; and forbidden material (Synapse config, SSH keys, tokens, validation archives,
raw images, .git) is detected. Synthetic dirs only — no Docker, no build.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import release_build_preflight as P  # noqa: E402

DOCKER = (REPO / "configs" / "release" / "Dockerfile").read_text()
FAILS = 0


def check(name, cond):
    global FAILS
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILS += 1


def make_ctx(td, n_folds=5):
    ctx = Path(td)
    (ctx / "requirements.lock.txt").write_text("nnunetv2==2.8.1\n")
    (ctx / "scripts").mkdir()
    for s in ("release_infer.py", "g4_reconstruct_validate.py", "g3_audit_labeled_archive.py"):
        (ctx / "scripts" / s).write_text("x")
    (ctx / "plans").mkdir()
    (ctx / "plans" / "nnUNetResEncUNetMPlans.json").write_text("{}")
    (ctx / "plans" / "dataset.json").write_text("{}")
    for i in range(n_folds):
        (ctx / "weights" / f"fold_{i}").mkdir(parents=True)
        (ctx / "weights" / f"fold_{i}" / "checkpoint_final.pth").write_text(f"ckpt-{i}")
    return ctx


def main():
    print("test_release_build_preflight:")
    # Dockerfile COPY sources are discovered (non-empty, include the runner + weights + plans)
    srcs = P.dockerfile_copy_sources(DOCKER)
    check("copy_sources_found", len(srcs) >= 4)
    check("copy_sources_include_runner", any("release_infer.py" in s for s in srcs))
    check("copy_sources_include_weights", any(s.rstrip("/") == "weights" for s in srcs))

    with tempfile.TemporaryDirectory() as td:                     # complete final context
        make_ctx(td, 5)
        check("complete_final_context_clean", P.check_context(td, DOCKER, proxy=False) == [])
    with tempfile.TemporaryDirectory() as td:                     # complete proxy context (fold_0)
        make_ctx(td, 1)
        check("complete_proxy_context_clean", P.check_context(td, DOCKER, proxy=True) == [])
    with tempfile.TemporaryDirectory() as td:                     # proxy layout judged in final mode -> fail
        make_ctx(td, 1)
        check("proxy_layout_fails_final_mode", P.check_context(td, DOCKER, proxy=False) != [])
    with tempfile.TemporaryDirectory() as td:                     # missing a COPY source
        make_ctx(td, 5); (Path(td) / "scripts" / "release_infer.py").unlink()
        probs = P.check_context(td, DOCKER, proxy=False)
        check("missing_copy_source_detected", any("release_infer.py" in p for p in probs))
    with tempfile.TemporaryDirectory() as td:                     # missing a checkpoint
        make_ctx(td, 4)
        check("missing_checkpoint_detected", any("fold_4" in p for p in P.check_context(td, DOCKER, proxy=False)))

    # forbidden material anywhere in the context is detected
    for fname, mk in [
        (".synapseConfig", lambda c: (c / ".synapseConfig").write_text("x")),
        ("id_ed25519", lambda c: (c / "id_ed25519").write_text("x")),
        ("secret.pat", lambda c: (c / "secret.pat").write_text("x")),
        (".bash_history", lambda c: (c / ".bash_history").write_text("x")),
        ("validation.zip", lambda c: (c / "validation.zip").write_text("x")),
        ("raw_image.nii.gz", lambda c: (c / "raw_image.nii.gz").write_text("x")),
        (".git", lambda c: (c / ".git").mkdir()),
        ("nnUNet_raw", lambda c: (c / "nnUNet_raw").mkdir()),
    ]:
        with tempfile.TemporaryDirectory() as td:
            ctx = make_ctx(td, 5); mk(ctx)
            probs = P.check_context(td, DOCKER, proxy=False)
            check(f"forbidden_detected::{fname}", len(probs) >= 1)

    print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
