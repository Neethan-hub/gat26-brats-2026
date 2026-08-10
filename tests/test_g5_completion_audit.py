#!/usr/bin/env python3
"""GAT-26 completion-audit dispatch regression tests (`python3 tests/test_g5_completion_audit.py`).

These lock the exact defect that made an L run mis-audit against the M architecture: the audit
must derive the plans/tag from the frozen launch manifest, fail closed on absent/unknown/
inconsistent plan or tag, use the SELECTED plan (never a module default) in every verification
path, and emit a tag-aware verdict. Synthetic only — no GPU, no nnU-Net, no real case IDs/hashes.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import g5_completion_audit as A  # noqa: E402
import g5_runner as R            # noqa: E402

FAILS = 0


def check(name, cond):
    global FAILS
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILS += 1


def manifest(tag, plans, of=None):
    of = of or f"/workspace/runs/nnUNet_results/{R.DATASET}/nnUNetTrainer__{plans}__3d_fullres/fold_0"
    return {"tag": tag, "recipe": {"plans": plans}, "output_folder": of}


def main():
    print("test_g5_completion_audit:")

    # 1. M manifest selects the M plan; verdict is M_COMPLETION_PASS on all-ok.
    pM, tM = A.resolve_plans_tag(manifest("M", "nnUNetResEncUNetMPlans"))
    check("M_manifest_selects_M_plan", (pM, tM) == ("nnUNetResEncUNetMPlans", "M"))
    check("M_verdict_pass", A.verdict_for("M", True) == "M_COMPLETION_PASS")
    check("M_verdict_no_go", A.verdict_for("M", False) == "M_COMPLETION_NO_GO")

    # 2. L manifest selects the L plan; verdict is L_COMPLETION_PASS on all-ok.
    pL, tL = A.resolve_plans_tag(manifest("L", "nnUNetResEncUNetLPlans"))
    check("L_manifest_selects_L_plan", (pL, tL) == ("nnUNetResEncUNetLPlans", "L"))
    check("L_verdict_pass", A.verdict_for("L", True) == "L_COMPLETION_PASS")
    check("L_verdict_no_go", A.verdict_for("L", False) == "L_COMPLETION_NO_GO")

    # 3. An L checkpoint can never be silently checked against the M architecture: an L manifest
    #    resolves ONLY to the L plan, and a tag/plan cross-mismatch fails closed.
    def raises(m):
        try:
            A.resolve_plans_tag(m)
            return False
        except A.PlanResolutionError:
            return True
    check("L_tag_with_M_plan_fails_closed", raises(manifest("L", "nnUNetResEncUNetMPlans")))
    check("M_tag_with_L_plan_fails_closed", raises(manifest("M", "nnUNetResEncUNetLPlans")))

    # 4. Unknown / missing / inconsistent plan or tag fails closed (never falls back to M).
    check("unknown_tag_fails_closed", raises(manifest("XL", "nnUNetResEncUNetLPlans")))
    check("unknown_plan_fails_closed", raises(manifest("L", "nnUNetResEncUNetXLPlans")))
    check("absent_plan_fails_closed", raises({"tag": "L", "recipe": {},
                                              "output_folder": "x/y"}))
    check("plan_not_in_output_folder_fails_closed",
          raises({"tag": "L", "recipe": {"plans": "nnUNetResEncUNetLPlans"},
                  "output_folder": "/runs/nnUNetResEncUNetMPlans/fold_0"}))

    # 4b. Backward-compat: an OLDER manifest with a valid plan but NO tag (the real M fold-0 case)
    #     derives the canonical tag from the authoritative plan — it must NOT fail closed.
    old_M = {"recipe": {"plans": "nnUNetResEncUNetMPlans"},
             "output_folder": f"x/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0"}
    check("absent_tag_derives_from_plans", A.resolve_plans_tag(old_M) == ("nnUNetResEncUNetMPlans", "M"))
    old_L = {"recipe": {"plans": "nnUNetResEncUNetLPlans"},
             "output_folder": f"x/nnUNetResEncUNetLPlans/fold_0"}
    check("absent_tag_derives_L_from_plans", A.resolve_plans_tag(old_L) == ("nnUNetResEncUNetLPlans", "L"))

    # 5. Provenance uses the SELECTED plan, not a module default. Stub R.sha to a path-derived
    #    token so the plan-hash check passes ONLY for the plan whose file path is used.
    orig_sha = R.sha
    LC5 = "c" * 40
    try:
        R.sha = lambda p: f"sha::{Path(p).name}"
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["nnUNet_preprocessed"] = tmp
            rundir = Path(tmp) / "rundir"; rundir.mkdir()
            (rundir / "precheck.json").write_text(json.dumps({"source_commit": True}))
            (rundir / "source_version_at_launch.txt").write_text(
                f"launch_commit={LC5}\nlaunch_utc_epoch=1\nmarker_at_launch:\ndeployed_commit={LC5}\n")
            svp = Path(tmp) / "config.json"
            svp.write_text(f"deployed_commit={LC5}\n")
            expect = {
                "commit": LC5, "source_version_path": str(svp),
                "config_path": str(svp), "config_sha256": "sha::config.json",
                "split_sha256": "sha::splits_final.json",
                "plan_sha256": "sha::nnUNetResEncUNetLPlans.json",   # L plan's expected token
                "fingerprint_sha256": "sha::dataset_fingerprint.json",
            }
            rep_L = {"gates": {}}
            okL = A.verify_provenance(rep_L, str(rundir), "of", expect, "nnUNetResEncUNetLPlans")
            rep_M = {"gates": {}}
            okM = A.verify_provenance(rep_M, str(rundir), "of", expect, "nnUNetResEncUNetMPlans")
            check("provenance_passes_with_selected_L_plan", okL is True)
            check("provenance_fails_when_checked_against_M_plan", okM is False)
            check("provenance_plan_hash_uses_selected_plan",
                  rep_L["gates"]["provenance_hashes"]["detail"]["plan_hash"] is True
                  and rep_M["gates"]["provenance_hashes"]["detail"]["plan_hash"] is False)
    finally:
        R.sha = orig_sha
        os.environ.pop("nnUNet_preprocessed", None)

    # 6. No verification path carries a hardcoded R.PLANS default: the plan args are REQUIRED.
    vp = inspect.signature(A.verify_provenance).parameters["plans"]
    sl = inspect.signature(A.strict_checkpoint_load).parameters["plans_id"]
    check("verify_provenance_plan_arg_required", vp.default is inspect.Parameter.empty)
    check("strict_checkpoint_load_plan_arg_required", sl.default is inspect.Parameter.empty)
    check("no_R_PLANS_in_source", "R.PLANS" not in (REPO / "scripts" / "g5_completion_audit.py").read_text())

    # 7. Permitted set is exactly the frozen M/L screen — backward-compatible M identity intact.
    check("permitted_is_exactly_M_L",
          A.PERMITTED_PLANS == {"M": "nnUNetResEncUNetMPlans", "L": "nnUNetResEncUNetLPlans"})

    # 8. STRICT launch-source provenance (exact-field parse; no Boolean-only legacy fallback).
    LC = "a" * 40           # launch commit
    OTHER = "b" * 40        # a later redeployment's commit

    def snaptext(launch=LC, marker_dc=LC, extra_head="", extra_marker=""):
        return (f"{extra_head}launch_commit={launch}\nlaunch_utc_epoch=1\n"
                f"marker_at_launch:\ndeployed_commit={marker_dc}\ndeployed_utc=x\n{extra_marker}")

    def mk(td, snapshot=None, precheck={"source_commit": True}, current=f"deployed_commit={LC}\n",
           expect_commit=LC, snapshot_as_dir=False):
        rd = Path(td)
        marker = rd / "SOURCE_VERSION.txt"; marker.write_text(current)
        p = rd / "source_version_at_launch.txt"
        if snapshot_as_dir:
            p.mkdir()
        elif snapshot is not None:
            p.write_text(snapshot)
        if precheck is not None:
            (rd / "precheck.json").write_text(json.dumps(precheck))
        return A.verify_launch_provenance(str(rd), {"commit": expect_commit, "source_version_path": str(marker)})

    with tempfile.TemporaryDirectory() as td:               # valid, current marker == launch
        ok8, i8 = mk(td, snapshot=snaptext())
        check("strict_valid_passes_no_drift", ok8 and i8["deployment_drift"] is False)
    with tempfile.TemporaryDirectory() as td:               # valid + legitimate later redeploy drift
        ok8, i8 = mk(td, snapshot=snaptext(), current=f"deployed_commit={OTHER}\n")
        check("strict_valid_passes_with_drift_reported", ok8 and i8["deployment_drift"] is True)
    with tempfile.TemporaryDirectory() as td:               # immutable launch mismatch
        ok8, _ = mk(td, snapshot=snaptext(launch=OTHER))
        check("strict_launch_mismatch_fails", ok8 is False)
    with tempfile.TemporaryDirectory() as td:               # NO Boolean-only legacy fallback: precheck true but NO snapshot
        ok8, i8 = mk(td, snapshot=None)
        check("no_boolean_fallback_precheck_only_fails",
              ok8 is False and i8["launch_evidence"] == "absent_launch_snapshot_no_boolean_fallback")
    with tempfile.TemporaryDirectory() as td:               # duplicate launch_commit fields
        ok8, i8 = mk(td, snapshot=snaptext(extra_head=f"launch_commit={LC}\n"))
        check("duplicate_launch_commit_fails", ok8 is False and i8["n_launch_commit_fields"] == 2)
    with tempfile.TemporaryDirectory() as td:               # substring-only (commit only in a comment, no exact field)
        sub = f"# deployed at {LC} historically\nlaunch_utc_epoch=1\nmarker_at_launch:\ndeployed_commit={LC}\n"
        ok8, _ = mk(td, snapshot=sub)
        check("substring_only_launch_commit_fails", ok8 is False)
    with tempfile.TemporaryDirectory() as td:               # marker deployed_commit mismatch
        ok8, _ = mk(td, snapshot=snaptext(marker_dc=OTHER))
        check("marker_deployed_commit_mismatch_fails", ok8 is False)
    with tempfile.TemporaryDirectory() as td:               # missing marker_at_launch section
        ok8, i8 = mk(td, snapshot=f"launch_commit={LC}\nlaunch_utc_epoch=1\n")
        check("missing_marker_section_fails", ok8 is False and i8["launch_evidence"] == "missing_marker_at_launch_section")
    with tempfile.TemporaryDirectory() as td:               # precheck source_commit false
        ok8, _ = mk(td, snapshot=snaptext(), precheck={"source_commit": False})
        check("precheck_false_fails", ok8 is False)
    with tempfile.TemporaryDirectory() as td:               # precheck missing entirely
        ok8, _ = mk(td, snapshot=snaptext(), precheck=None)
        check("precheck_missing_fails", ok8 is False)
    with tempfile.TemporaryDirectory() as td:               # malformed EXPECTED commit
        ok8, i8 = mk(td, snapshot=snaptext(), expect_commit="not-a-commit")
        check("malformed_expected_commit_fails", ok8 is False and i8["launch_evidence"] == "malformed_expected_commit")
    with tempfile.TemporaryDirectory() as td:               # unreadable snapshot (a directory)
        ok8, i8 = mk(td, snapshot_as_dir=True)
        check("unreadable_snapshot_fails", ok8 is False and i8["launch_evidence"] == "unreadable_launch_snapshot")
    with tempfile.TemporaryDirectory() as td:               # drift reported, snapshot never rewritten
        rd = Path(td); (rd / "precheck.json").write_text(json.dumps({"source_commit": True}))
        snap = rd / "source_version_at_launch.txt"; snap.write_text(snaptext())
        marker = rd / "SOURCE_VERSION.txt"; marker.write_text(f"deployed_commit={OTHER}\n")
        before = snap.read_bytes()
        ok8, i8 = A.verify_launch_provenance(str(rd), {"commit": LC, "source_version_path": str(marker)})
        check("launch_record_never_rewritten", snap.read_bytes() == before and ok8 and i8["deployment_drift"] is True)

    print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
