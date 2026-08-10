#!/usr/bin/env python3
"""GAT-26 strict fold-0 completion audit (Part D) — runs in the TRAINING venv, shells out to
the isolated evaluator venv for BraTS-evaluation 0.0.8 GoAT. Does NOT train.

Reuses the preserved checkpoint + predictions of a completed fold-0 run and establishes a
scientifically valid completion result:

  1. training completion (epochs 0..N-1, checkpoint_final loadable, finite loss history,
     provenance == frozen config/plan/split/fingerprint/source-commit, no external weights)
  2. strict checkpoint load (strict=True, explicit _orig_mod handling; never strict=False)
  3. exact fold-0 validation membership equality (C2)
  4. re-run every output validator (flat Task-3 name, {0,1,2,3}, geometry, ET⊆TC⊆WT)
  5. re-run BraTS-evaluation 0.0.8 GoAT via the correct isolated evaluator python (C1 handling)
  6. 271 successful records / 0 errors / set-equal / +inf→373 with full denominators
  7. independent aggregate recompute vs the evaluator summary within 1e-12
  8. loose plumbing-sanity gate (catastrophic-pipeline only, NOT a competitiveness claim)

Sanitized: emits counts / booleans / aggregate metrics only — no real case IDs, no per-case
values, no private hashes, no predictions/checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import g5_runner as R          # sha, output_folder, fold_validation_stems, membership_report
import g5_evaluate as E        # REGIONS, aggregate_components
import g4_reconstruct_validate as RC

REGIONS = E.REGIONS
EXPECT_EPOCHS = R.EXPECT_EPOCHS
EXPECT_VAL = R.EXPECT_VAL_CASES
TOL = 1e-12

# Loose plumbing-sanity thresholds — catastrophic-pipeline detection only.
PLUMB = {"wt_dsc_min": 0.50, "tc_dsc_min": 0.30, "et_dsc_min": 0.20}

# Permitted frozen fold-0 screen architectures: tag -> plans identity. This is the ONLY place a
# plan name is allowed; every verification path takes the manifest-resolved plan, never a module
# default. An absent / unknown / tag-inconsistent plan or tag FAILS CLOSED (never falls back to M).
PERMITTED_PLANS = {"M": "nnUNetResEncUNetMPlans", "L": "nnUNetResEncUNetLPlans"}


class PlanResolutionError(Exception):
    """Raised when the manifest plan/tag is absent, unknown, or mutually inconsistent."""


def resolve_plans_tag(manifest):
    """Derive (plans, tag) from the frozen launch manifest, fail-closed.

    `recipe.plans` is the AUTHORITATIVE architecture identity: it must be present, one of the
    permitted frozen M/L plans, and consistent with the recorded output_folder path. `tag` is the
    canonical label derived from the frozen plans<->tag bijection; if the manifest also carries a
    tag it must be permitted and agree (older M manifests predate the tag field and legitimately
    omit it — the architecture is still fully determined by `plans`, never a module default).

    FAIL CLOSED (raise PlanResolutionError) on: absent/unknown plans, plans not in output_folder,
    or a manifest tag that is unknown or inconsistent with the plan. The audit then emits a
    *_COMPLETION_NO_GO instead of checking a wrong/default architecture.
    """
    plans = (manifest.get("recipe", {}) or {}).get("plans", None)
    of = manifest.get("output_folder", "") or ""
    tag = manifest.get("tag", None)
    if plans not in PERMITTED_PLANS.values():
        raise PlanResolutionError(f"plans not permitted/absent: {plans!r}")
    inv = {v: k for k, v in PERMITTED_PLANS.items()}          # frozen bijection: plan -> tag
    tag_from_plans = inv[plans]
    if tag is not None:
        if tag not in PERMITTED_PLANS:
            raise PlanResolutionError(f"tag not permitted: {tag!r}")
        if tag != tag_from_plans:
            raise PlanResolutionError(f"tag/plans inconsistent: tag={tag!r} plans={plans!r}")
    if plans not in of:
        raise PlanResolutionError("plans not consistent with output_folder path")
    return plans, tag_from_plans


def verdict_for(tag, all_ok):
    """Tag-aware verdict string: {tag}_COMPLETION_{PASS|NO_GO}."""
    return f"{tag}_COMPLETION_{'PASS' if all_ok else 'NO_GO'}"


def fail(report, key, detail=""):
    report["gates"][key] = {"ok": False, "detail": detail}
    return False


def ok(report, key, detail=""):
    report["gates"][key] = {"ok": True, "detail": detail}
    return True


def verify_training_completion(report, of, expect):
    import torch
    ckpt_path = Path(of) / "checkpoint_final.pth"
    if not ckpt_path.exists():
        return fail(report, "checkpoint_final_present")
    ok(report, "checkpoint_final_present")
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    # epochs 0..N-1 complete
    cur = ck.get("current_epoch", None)
    report["training"]["current_epoch"] = cur
    epochs_ok = (cur is not None) and (cur >= EXPECT_EPOCHS - 1)
    ok(report, "epochs_complete", f"current_epoch={cur} expect>={EXPECT_EPOCHS-1}") if epochs_ok \
        else fail(report, "epochs_complete", f"current_epoch={cur}")
    # finite loss history from the checkpoint's own logging
    logging = ck.get("logging", {}) or {}
    tl = logging.get("train_losses", []) or []
    vl = logging.get("val_losses", []) or []
    n_tl = len(tl)
    finite_losses = all(math.isfinite(float(x)) for x in tl) and all(math.isfinite(float(x)) for x in vl)
    report["training"]["train_loss_points"] = n_tl
    (ok if (finite_losses and n_tl >= EXPECT_EPOCHS) else fail)(
        report, "finite_loss_history", f"points={n_tl} finite={finite_losses}")
    # provenance / no external weights
    tname = ck.get("trainer_name", "")
    report["training"]["trainer_name"] = tname
    (ok if tname == R.TRAINER else fail)(report, "trainer_name_matches", tname)
    init_args = ck.get("init_args", {}) or {}
    # no pretrained / external weight path anywhere in init args
    blob = json.dumps(init_args, default=str).lower()
    no_ext = ("pretrained" not in blob) and ("finetune" not in blob) and (".pth" not in blob.replace("checkpoint_final.pth", ""))
    (ok if no_ext else fail)(report, "no_external_weights_in_init")
    return ck


def _wellformed_commit(s):
    return isinstance(s, str) and len(s) == 40 and all(c in "0123456789abcdef" for c in s.lower())


def _exact_field_values(text, key):
    """All values from lines that are EXACTLY `key=value` (leading/trailing whitespace trimmed).
    A commit that only appears as a substring of some other line is NOT matched — this is what
    makes substring-only evidence insufficient."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(key + "="):
            out.append(line[len(key) + 1:].strip())
    return out


def verify_launch_provenance(rundir, expect):
    """Verify the IMMUTABLE per-run launch-source provenance by STRICT exact-field parsing,
    decoupled from the MUTABLE current worker deployment marker.

    STRICT contract (no generic Boolean-only legacy fallback — future G7 runners always write the
    per-run launch snapshot before training, so this is the only accepted path):
      * the expected commit (`expect['commit']`) must be a well-formed 40-hex string;
      * `<rundir>/source_version_at_launch.txt` must exist and be readable, with a `marker_at_launch:`
        section;
      * it must contain EXACTLY ONE `launch_commit=<value>` field whose value equals the expected
        commit (substring-only / duplicate / mismatched / malformed → reject);
      * the launch-time marker snapshot (below `marker_at_launch:`) must contain exactly one
        `deployed_commit=<value>` equal to the expected commit;
      * `<rundir>/precheck.json` must exist and record `source_commit` exactly `true`.
    Any missing / duplicate / malformed / substring-only / mismatched / unreadable evidence → ok
    False. The current deployment marker and the drift boolean are REPORTED ONLY and never gate ok;
    legitimate later redeployment drift is expected and never rewrites the launch record.

    Historical note: the M/L fold-0 runs predate the launch snapshot. Their completion evidence is
    already committed and authoritative; this stricter verifier does NOT retro-pass them (they have
    no snapshot) and does NOT rewrite or invalidate that historical committed evidence.
    """
    lc = expect.get("commit")
    info = {"launch_commit_wellformed": _wellformed_commit(lc)}
    # mutable current deployment marker — reported only, never gates
    try:
        cur = Path(expect.get("source_version_path", "")).read_text()
    except Exception:
        cur = ""
    marker_field = f"deployed_commit={lc}" if _wellformed_commit(lc) else "\x00"
    info["current_marker_matches_launch"] = _wellformed_commit(lc) and (marker_field in cur)
    info["deployment_drift"] = _wellformed_commit(lc) and (marker_field not in cur)

    def done(ev, ok):
        info["launch_evidence"] = ev
        info["ok"] = bool(ok)
        return info["ok"], info

    if not _wellformed_commit(lc):
        return done("malformed_expected_commit", False)
    snap = Path(rundir) / "source_version_at_launch.txt"
    if not snap.exists():
        return done("absent_launch_snapshot_no_boolean_fallback", False)
    try:
        text = snap.read_text()
    except Exception:
        return done("unreadable_launch_snapshot", False)
    if "marker_at_launch:" not in text:
        return done("missing_marker_at_launch_section", False)
    head, marker = text.split("marker_at_launch:", 1)
    launch_vals = _exact_field_values(head, "launch_commit")
    marker_vals = _exact_field_values(marker, "deployed_commit")
    exactly_one_launch = (len(launch_vals) == 1 and launch_vals[0] == lc and _wellformed_commit(launch_vals[0]))
    marker_ok = (len(marker_vals) == 1 and marker_vals[0] == lc)
    pj = Path(rundir) / "precheck.json"
    try:
        precheck_ok = (json.loads(pj.read_text()).get("source_commit") is True)
    except Exception:
        precheck_ok = False
    info["exactly_one_launch_commit_eq_expected"] = exactly_one_launch
    info["marker_deployed_commit_eq_expected"] = marker_ok
    info["precheck_source_commit_true"] = precheck_ok
    info["n_launch_commit_fields"] = len(launch_vals)
    info["n_marker_deployed_commit_fields"] = len(marker_vals)
    return done("per_run_launch_snapshot_strict", exactly_one_launch and marker_ok and precheck_ok)


def verify_provenance(report, rundir, of, expect, plans):
    # Re-verify the immutable data/config/plan hashes + the IMMUTABLE launch-source provenance.
    # `plans` is the frozen plans identity of THIS run (resolved fail-closed from the launch
    # manifest); a module-default constant here would silently mismatch the plan hash for an L run.
    # The launch-source gate checks the per-run launch record, NOT the mutable current marker.
    import os
    pp = Path(os.environ["nnUNet_preprocessed"]) / R.DATASET
    lp_ok, lp = verify_launch_provenance(rundir, expect)
    checks = {
        "launch_source_provenance": lp_ok,
        "public_config_hash": R.sha(expect["config_path"]) == expect["config_sha256"],
        "split_full_hash": R.sha(pp / "splits_final.json") == expect["split_sha256"],
        "plan_hash": R.sha(pp / f"{plans}.json") == expect["plan_sha256"],
        "fingerprint_hash": R.sha(pp / "dataset_fingerprint.json") == expect["fingerprint_sha256"],
    }
    # deployment marker / drift are reported separately — they never gate the verdict
    report["deployment"] = {
        "launch_commit_wellformed": lp["launch_commit_wellformed"],
        "launch_evidence": lp["launch_evidence"],
        "launch_source_provenance_ok": lp["ok"],
        "exactly_one_launch_commit_eq_expected": lp.get("exactly_one_launch_commit_eq_expected"),
        "marker_deployed_commit_eq_expected": lp.get("marker_deployed_commit_eq_expected"),
        "precheck_source_commit_true": lp.get("precheck_source_commit_true"),
        "current_marker_matches_launch": lp["current_marker_matches_launch"],
        "deployment_drift_expected_after_redeploy": lp["deployment_drift"],
    }
    (ok if all(checks.values()) else fail)(
        report, "provenance_hashes", {k: v for k, v in checks.items()})
    return all(checks.values())


def strict_checkpoint_load(report, of, ck, plans_id):
    import torch
    import os
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    pp = Path(os.environ["nnUNet_preprocessed"]) / R.DATASET
    plans = json.loads((pp / f"{plans_id}.json").read_text())
    dataset_json = json.loads((pp / "dataset.json").read_text())
    pm = PlansManager(plans)
    cm = pm.get_configuration(R.CONFIG)
    lm = pm.get_label_manager(dataset_json)
    num_in = len(dataset_json["channel_names"])
    net = get_network_from_plans(
        cm.network_arch_class_name, cm.network_arch_init_kwargs,
        cm.network_arch_init_kwargs_req_import, num_in, lm.num_segmentation_heads,
        allow_init=True, deep_supervision=False)
    sd = ck["network_weights"]
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    try:
        net.load_state_dict(sd, strict=True)      # never strict=False
        ok(report, "strict_checkpoint_load", "strict=True, _orig_mod handled, no missing/unexpected")
        return True
    except Exception as e:
        return fail(report, "strict_checkpoint_load", type(e).__name__)


def verify_membership(report, of, fold):
    import os
    pp = Path(os.environ["nnUNet_preprocessed"]) / R.DATASET
    expected = R.fold_validation_stems(pp / "splits_final.json", fold=fold)
    val_dir = Path(of) / "validation"
    actual = {p.name[:-len(".nii.gz")] for p in val_dir.glob("*.nii.gz")}
    mem = R.membership_report(expected, actual)
    report["membership"] = mem                    # counts/booleans only
    (ok if mem["exact_set_equal"] else fail)(report, "exact_membership_equal", {k: mem[k] for k in
                                             ("expected_count", "actual_count", "missing", "extra")})
    return mem["exact_set_equal"]


def rerun_output_validators(report, of, expect_val):
    import nibabel as nib
    import os
    raw = Path(os.environ["nnUNet_raw"]) / R.DATASET / "labelsTr"
    val_dir = Path(of) / "validation"
    preds = sorted(val_dir.glob("*.nii.gz"))
    bad = 0
    for p in preds:
        cid = p.name[:-len(".nii.gz")]
        ref = nib.load(str(raw / f"{cid}.nii.gz"))
        v = RC.validate_output_file(str(p), ref.affine, ref.shape, ref.header.get_zooms(),
                                    nib.aff2axcodes(ref.affine))
        if not v["ok"]:
            bad += 1
    report["output_validation"] = {"predictions": len(preds), "valid": len(preds) - bad, "invalid": bad}
    (ok if (bad == 0 and len(preds) == expect_val) else fail)(
        report, "output_validators", {"n": len(preds), "invalid": bad, "expected": expect_val})
    return bad == 0 and len(preds) == expect_val


def run_official_evaluator(report, of, eval_python, rundir, expect_val):
    import os
    raw = Path(os.environ["nnUNet_raw"]) / R.DATASET / "labelsTr"
    val_dir = Path(of) / "validation"
    priv = Path(rundir) / "official_eval_private_posthoc.json"
    summ = Path(rundir) / "official_eval_summary_posthoc.json"
    ev = subprocess.run([eval_python, str(REPO / "g5_evaluate.py"),
                         "--preds", str(val_dir), "--gt", str(raw),
                         "--out", str(priv), "--summary-out", str(summ),
                         "--expected-n", str(expect_val)],
                        capture_output=True, text=True)
    (Path(rundir) / "posthoc_eval.stdout.log").write_text(ev.stdout or "")
    (Path(rundir) / "posthoc_eval.stderr.log").write_text(ev.stderr or "")
    if ev.returncode != 0:
        fail(report, "official_evaluator_run", f"rc={ev.returncode}; {(ev.stderr or '')[-200:]}")
        return None, None
    priv_data = json.loads(priv.read_text())
    summ_data = json.loads(summ.read_text())
    n_ok = summ_data.get("n")
    errs = priv_data.get("errors", [])
    good = (n_ok == expect_val) and (len(errs) == 0)
    (ok if good else fail)(report, "official_evaluator_run",
                           {"n": n_ok, "n_errors": len(errs), "expected": expect_val})
    return (priv_data if good else None), (summ_data if good else None)


def independent_recompute(report, priv_data, summ_data, expect_val):
    recs = list(priv_data["per_subject_records"].values())
    comps = E.aggregate_components(recs, n_expected=expect_val)
    recomputed = {k: v[0] for k, v in comps.items()}
    denom = {k: v[1] for k, v in comps.items()}
    primary = summ_data["components_mean"]
    max_diff = 0.0
    for k in recomputed:
        max_diff = max(max_diff, abs(recomputed[k] - primary[k]))
    agree = max_diff <= TOL
    denom_ok = all(d == expect_val for d in denom.values())
    (ok if (agree and denom_ok) else fail)(
        report, "independent_recompute_within_1e-12", {"max_diff": max_diff, "denoms": denom})
    return agree and denom_ok


def check_hd95_handling(report, priv_data, summ_data):
    """Confirm every +inf HD95 was converted (373) and none dropped from any denominator."""
    hist = priv_data.get("status_histogram", {})
    n_penalty = sum(v for k, v in hist.items() if k.startswith("penalty"))
    n_tn = hist.get("empty_both_tn", 0)
    # all hd95 aggregate values must be finite and <= 373; denominators already checked = 271
    recs = list(priv_data["per_subject_records"].values())
    all_finite = all(math.isfinite(rec[f"{r}_hd95"]) and rec[f"{r}_hd95"] <= E.HD95_PENALTY
                     for rec in recs for r in REGIONS)
    report["hd95_handling"] = {"penalty_conversions": n_penalty, "empty_both_tn": n_tn,
                               "status_histogram": hist}
    (ok if all_finite else fail)(report, "hd95_inf_to_373_no_dropped_denominator",
                                 {"penalty_conversions": n_penalty, "all_finite": all_finite})
    return all_finite


def plumbing_sanity(report, summ_data):
    cm = summ_data["components_mean"]
    checks = {
        "wt_dsc>=0.50": cm["wt_dsc"] >= PLUMB["wt_dsc_min"],
        "tc_dsc>=0.30": cm["tc_dsc"] >= PLUMB["tc_dsc_min"],
        "et_dsc>=0.20": cm["et_dsc"] >= PLUMB["et_dsc_min"],
        "not_all_background": summ_data["missed_region_rate"] < 1.0,
        "all_aggregates_finite": all(math.isfinite(v) for v in cm.values())
        and math.isfinite(summ_data["dsc_p05"]) and math.isfinite(summ_data["hd95_p95"]),
    }
    (ok if all(checks.values()) else fail)(report, "plumbing_sanity", checks)
    return all(checks.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--eval-python", required=True)
    ap.add_argument("--report-out", required=True)     # sanitized report (committable)
    args = ap.parse_args()

    rundir = Path(args.rundir)
    manifest = json.loads((rundir / "launch_manifest.json").read_text())
    of = manifest["output_folder"]
    expect = json.loads((rundir / "expect.json").read_text())

    # fold of THIS run (from the frozen manifest), and the fold's expected validation size derived
    # from the frozen split itself (271 for fold 0, 270 for folds 1-4) — never a module default.
    import os as _os
    fold = manifest.get("recipe", {}).get("fold", manifest.get("fold", 0))
    _pp = Path(_os.environ["nnUNet_preprocessed"]) / R.DATASET
    expect_val = len(R.fold_validation_stems(_pp / "splits_final.json", fold=fold))

    # frozen plans identity + tag of THIS run, resolved FAIL-CLOSED (M or L), never a module default
    try:
        plans_id, tag = resolve_plans_tag(manifest)
    except PlanResolutionError as e:
        report = {"audit": "g5_fold0_completion", "trains": False, "plans": None, "tag": None,
                  "gates": {"plans_tag_resolution": {"ok": False, "detail": str(e)}}, "training": {},
                  "verdict": "UNKNOWN_COMPLETION_NO_GO"}
        Path(args.report_out).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"verdict": report["verdict"], "plans": None, "tag": None,
                          "gates": {"plans_tag_resolution": False}}))
        return 7

    report = {"audit": "g5_fold0_completion", "trains": False, "plans": plans_id,
              "tag": tag, "gates": {"plans_tag_resolution": {"ok": True, "detail": f"{tag}:{plans_id}"}},
              "training": {}}

    ck = verify_training_completion(report, of, expect)
    prov_ok = verify_provenance(report, str(rundir), of, expect, plans_id)
    load_ok = strict_checkpoint_load(report, of, ck, plans_id) if isinstance(ck, dict) else False
    report["fold"] = fold
    report["expected_val_cases"] = expect_val
    mem_ok = verify_membership(report, of, fold)
    val_ok = rerun_output_validators(report, of, expect_val)

    priv_data = summ_data = None
    if all(g["ok"] for g in report["gates"].values()):
        priv_data, summ_data = run_official_evaluator(report, of, args.eval_python, str(rundir), expect_val)
        if priv_data is not None:
            check_hd95_handling(report, priv_data, summ_data)
            independent_recompute(report, priv_data, summ_data, expect_val)
            plumbing_sanity(report, summ_data)

    all_ok = all(g["ok"] for g in report["gates"].values()) and (summ_data is not None)
    report["verdict"] = verdict_for(tag, all_ok)
    # A run whose gates all pass but whose current worker marker has drifted from its launch commit
    # (a legitimate later redeployment) is a backward-compatibility re-audit, NOT a fresh completion.
    drift = report.get("deployment", {}).get("deployment_drift_expected_after_redeploy", False)
    report["classification"] = (
        f"{tag}_BACKWARD_COMPATIBILITY_PASS_WITH_EXPECTED_DEPLOYMENT_DRIFT"
        if (all_ok and drift) else report["verdict"])
    if summ_data is not None:
        report["aggregate"] = {                       # sanitized aggregate only
            "n": summ_data["n"],
            "components_mean": summ_data["components_mean"],
            "component_denominator": summ_data["component_denominator"],
            "dsc_p05": summ_data["dsc_p05"], "hd95_p95": summ_data["hd95_p95"],
            "smallest_volume_wt_dsc": summ_data["smallest_volume_wt_dsc"],
            "empty_reference_fp_rate": summ_data["empty_reference_fp_rate"],
            "missed_region_rate": summ_data["missed_region_rate"],
            "status_histogram": summ_data["status_histogram"]}
    Path(args.report_out).write_text(json.dumps(report, indent=2) + "\n")
    # sanitized stdout: verdict/classification + non-private plan/tag identifiers + gate booleans
    print(json.dumps({"verdict": report["verdict"], "classification": report["classification"],
                      "plans": plans_id, "tag": tag,
                      "deployment_drift": report.get("deployment", {}).get(
                          "deployment_drift_expected_after_redeploy", False),
                      "gates": {k: v["ok"] for k, v in report["gates"].items()}}))
    return 0 if all_ok else 7


if __name__ == "__main__":
    sys.exit(main())
