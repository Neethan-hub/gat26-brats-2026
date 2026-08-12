#!/usr/bin/env python3
"""G89 — path-safety and batch-flow tests (`python3 tests/test_g89_batch_and_paths.py`).

Covers behaviour that earlier stages asserted weakly or not at all. Everything here is host-level
and synthetic: no GPU, no checkpoint, no image, no real BraTS data.

Two fixtures that earlier evidence leaned on are explicitly rejected as proof:
  * a **dangling** symlink is not proof of the traversal gate — inside a container its target is
    simply absent, so the entry is not a directory and the gate never runs;
  * a mode-0000 file is not proof of the unreadable gate — the container runs as root, which can
    read it, so that fixture exercises the kernel rather than the runner.

The multi-case failure-atomicity behaviour is measured against the real image during qualification
(a valid case followed by an invalid one); this file pins the pure-logic half of it.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import release_infer as X  # noqa: E402

FAILS = 0


def check(name, cond):
    global FAILS
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILS += 1


def raises(fn, exc=X.ReleaseInputError):
    try:
        fn()
        return False
    except exc:
        return True


def _mods(d: Path, stem: str):
    d.mkdir(parents=True, exist_ok=True)
    for m in ("t1n", "t1c", "t2w", "t2f"):
        (d / f"{stem}-{m}.nii.gz").write_bytes(b"placeholder")


def main():                                                          # noqa: C901
    print("test_g89_batch_and_paths:")

    # 1. a REAL escaping symlink: the target exists and resolves outside the mounted input root
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        inp = root / "input"
        outside = root / "outside" / "SYNTHLAB-30001-000"
        _mods(outside, "SYNTHLAB-30001-000")
        _mods(inp / "SYNTHLAB-30000-000", "SYNTHLAB-30000-000")
        try:
            os.symlink(outside, inp / "SYNTHLAB-30001-000")
            link_made = True
        except OSError:
            link_made = False
        if link_made:
            tgt = (inp / "SYNTHLAB-30001-000")
            check("escaping_symlink_target_really_exists", tgt.is_dir())
            check("escaping_symlink_resolves_outside_root",
                  Path(os.path.realpath(tgt)).is_relative_to(Path(os.path.realpath(root / "outside"))))
            check("real_escaping_symlink_fails_closed",
                  raises(lambda: X.list_case_folders(inp)))
        else:
            print("  ..   symlinks unsupported on this filesystem; gate NOT executed")
            check("real_escaping_symlink_fails_closed", False)

    # 2. a symlink that stays INSIDE the input root is not rejected by the traversal gate
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "input"
        _mods(inp / "SYNTHLAB-30002-000", "SYNTHLAB-30002-000")
        real = inp / "SYNTHLAB-30003-000"
        _mods(real, "SYNTHLAB-30003-000")
        try:
            os.symlink(real, inp / "SYNTHLAB-30004-000")
            names = sorted(p.name for p in X.list_case_folders(inp))
            check("inside_root_symlink_is_accepted", "SYNTHLAB-30004-000" in names)
        except OSError:
            print("  ..   symlinks unsupported; inside-root case NOT executed")

    # 3. a DANGLING symlink is explicitly NOT counted as proof of the traversal gate
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "input"
        _mods(inp / "SYNTHLAB-30005-000", "SYNTHLAB-30005-000")
        try:
            os.symlink(Path(td) / "does-not-exist", inp / "SYNTHLAB-30006-000")
            names = sorted(p.name for p in X.list_case_folders(inp))
            # it is not a directory, so it is skipped -- the traversal check never runs on it
            check("dangling_symlink_is_skipped_not_a_traversal_proof",
                  "SYNTHLAB-30006-000" not in names and names == ["SYNTHLAB-30005-000"])
        except OSError:
            print("  ..   symlinks unsupported; dangling case NOT executed")

    # 4. output-collision detection is unchanged: two folders mapping to one name fail closed
    #    BEFORE any inference, and identical names in different parents are still a collision
    check("duplicate_output_name_fails_closed",
          raises(lambda: X.plan_outputs(["/a/SYNTHLAB-30007-000", "/b/SYNTHLAB-30007-000"])))
    plan = X.plan_outputs(["/in/SYNTHLAB-30008-000", "/in/SYNTHLAB-30008-001"])
    check("distinct_names_do_not_collide", len(plan) == 2)

    # 5. batch planning validates EVERY name before the case loop starts, so an unsafe name in the
    #    batch prevents the run entirely. (Modality validation is per case, inside the loop -- that
    #    is the recorded failure-atomicity limitation and is measured against the real image.)
    check("unsafe_member_blocks_whole_batch",
          raises(lambda: X.plan_outputs(["/in/SYNTHLAB-30009-000", "/in/.hidden"])))
    good = X.plan_outputs(["/in/SYNTHLAB-30010-000", "/in/SYNTHLAB-30011-000"])
    check("valid_batch_plans_all_names", len(good) == 2)

    # 6. the runner exposes no non-finite input contract: this is measured, never gated
    check("no_non_finite_contract_in_runner",
          "isfinite" not in (REPO / "scripts" / "release_infer.py").read_text(encoding="utf-8"))

    # 7. the mode-0000 INPUT fixture is not a runner gate. The runner does check readability with
    #    os.access, but only on CHECKPOINTS inside discover_checkpoints; no input modality file is
    #    permission-checked, so a root-readable mode-0000 input proves nothing about the runner.
    src = (REPO / "scripts" / "release_infer.py").read_text(encoding="utf-8")
    import ast
    tree = ast.parse(src)
    access_scopes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "access"):
                    access_scopes.add(node.name)
    check("readability_check_confined_to_checkpoint_discovery",
          access_scopes == {"discover_checkpoints"})
    check("no_permission_gate_on_input_modalities",
          "os.access" not in ast.get_source_segment(src, next(
              n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "discover_modalities")))

    print(f"test_g89_batch_and_paths: {'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
