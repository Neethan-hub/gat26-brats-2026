#!/usr/bin/env python3
"""G83 §C synthetic and governance proofs. No GPU, no protected data, no credential."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# The evaluator resolves its data roots from the private runtime environment. These
# tests never read data, so provide inert placeholders when they are unset.
for _var in ("G82_RAW", "G82_BASE_STORE", "G82_CAND_STORE"):
    os.environ.setdefault(_var, os.path.join(HERE, "_unused_in_tests"))

import g83_overlap_predict as P  # noqa: E402
import g83_decide as D  # noqa: E402

SPEC_PATH = os.path.join(ROOT, "configs", "g83_dense_overlap_preregistration.json")
SPEC = json.load(open(SPEC_PATH))
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# ------------------------------------------------------------------ candidates
def t_candidate_set_closed():
    check("1 candidate_set_is_exactly_C0_and_D25", SPEC["candidates"] == ["C0", "D25"])
    check("1b no_other_candidate_permitted", SPEC["no_other_candidate_permitted"] is True)
    check("1c exact_difference_is_tile_step",
          SPEC["exact_scientific_difference"] == "tile_step_size: 0.5 -> 0.25")
    # D25 must differ from C0 in exactly one key
    c0 = dict(SPEC["candidate_definitions"]["C0"])
    d25 = SPEC["candidate_definitions"]["D25"]
    check("1d D25_declares_only_tile_step_differs",
          d25["identical_to_C0_except"] == "tile_step_size"
          and set(d25) == {"identical_to_C0_except", "tile_step_size"}
          and d25["tile_step_size"] == 0.25 and c0["tile_step_size"] == 0.5)


def t_step_refusal():
    ok = []
    for bad in (0.20, 0.30, 0.33, 0.40, 0.125, 1.0, 0.75, 0.0):
        try:
            P.assert_policy(bad)
            ok.append(bad)
        except P.PolicyViolation:
            pass
    check("2 non_preregistered_steps_refused", not ok, f"accepted {ok}" if ok else "")
    for good in (0.5, 0.25):
        try:
            P.assert_policy(good)
        except P.PolicyViolation:
            check("2b preregistered_steps_accepted", False, f"{good} rejected")
            return
    check("2b preregistered_steps_accepted", True)
    check("2c allowed_steps_match_spec",
          sorted(P.ALLOWED_TILE_STEPS) == sorted(SPEC["allowed_tile_steps"]))


def t_frozen_knobs_refused():
    """TTA, threshold changes, cleanup and new weights are structurally refused."""
    bad = []
    for key, value in (("use_mirroring", True), ("threshold", 0.4), ("threshold", 0.6),
                       ("use_gaussian", False), ("checkpoint_name", "checkpoint_best.pth"),
                       ("config", "3d_lowres")):
        try:
            P.assert_policy(0.25, **{key: value})
            bad.append((key, value))
        except P.PolicyViolation:
            pass
    check("3 tta_threshold_cleanup_and_weight_changes_refused", not bad, str(bad))
    unknown = []
    for key in ("cleanup", "presence_gate", "postprocess", "soup", "tta_axes"):
        try:
            P.assert_policy(0.25, **{key: 1})
            unknown.append(key)
        except P.PolicyViolation:
            pass
    check("3b unknown_knobs_refused", not unknown, str(unknown))
    check("3c frozen_defaults_match_C0",
          P.FROZEN["use_mirroring"] is False and P.FROZEN["threshold"] == 0.5
          and P.FROZEN["use_gaussian"] is True
          and P.FROZEN["checkpoint_name"] == "checkpoint_final.pth")


def t_predictor_only_varies_step():
    """The predictor constructor must pass the frozen values, not caller values."""
    src = inspect.getsource(P.build_predictor)
    check("4 predictor_uses_frozen_mirroring", 'FROZEN["use_mirroring"]' in src
          and "use_mirroring=True" not in src)
    check("4b predictor_uses_frozen_gaussian", 'FROZEN["use_gaussian"]' in src)
    check("4c predictor_takes_tile_step_as_the_only_variable",
          "tile_step_size=float(tile_step)" in src)
    main_src = inspect.getsource(P.main)
    check("4d ensemble_mean_and_order_preserved",
          "acc /= float(len(weights))" in main_src and "checkpoint order fixed" in main_src)


# ---------------------------------------------------------------- sealed folds
def t_sealed_confirmation_folds():
    import g83_eval as EV
    freeze = os.path.join(ROOT, "artifacts", "g83_candidate_freeze.json")
    frozen_now = os.path.exists(freeze)
    raised = False
    try:
        EV.assert_folds_allowed([3, 4], freeze + ".definitely-absent")
    except EV.SealedFoldError:
        raised = True
    check("5 confirmation_folds_sealed_without_freeze_artifact", raised)
    try:
        EV.assert_folds_allowed([0, 1, 2], freeze + ".definitely-absent")
        ok = True
    except EV.SealedFoldError:
        ok = False
    check("5b calibration_folds_always_readable", ok)
    check("5c spec_requires_freeze_before_confirmation",
          SPEC["splits"]["confirmation_opens_only_after_candidate_freeze_commit"] is True
          and SPEC["splits"]["confirmation_executes_once_only"] is True)
    check("5d freeze_artifact_absent_or_calibration_passed",
          (not frozen_now) or json.load(open(freeze)).get("calibration_passed") is True)


# ------------------------------------------------------------------- ordering
def t_submission_ordering_and_uniqueness():
    check("6 at_most_one_submission_this_stage",
          SPEC["official_validation_submission"]["max_submissions_this_stage"] == 1)
    check("6b no_automatic_retry",
          SPEC["official_validation_submission"]["no_automatic_retry"] is True)
    check("6c docker_queue_never_used",
          SPEC["official_validation_submission"]["docker_queue_never_used"] == "9619630"
          and SPEC["official_validation_submission"]["predictions_queue"] == "9619539")
    receipt = os.path.join(ROOT, "artifacts", "g83_submission_receipt.json")
    conf = os.path.join(ROOT, "artifacts", "g83_confirmation.json")
    rt = os.path.join(ROOT, "artifacts", "g83_runtime_gate.json")
    if os.path.exists(receipt):
        ok = os.path.exists(conf) and os.path.exists(rt)
        check("6d submission_only_after_confirmation_and_runtime", ok)
    else:
        check("6d submission_only_after_confirmation_and_runtime", True,
              "no submission yet")


def t_baseline_wins_ties():
    check("7 spec_says_baseline_wins_ties", SPEC["metric"]["baseline_wins_ties"] is True)
    g = SPEC["calibration_gates_all_required"]
    # a zero delta must NOT pass the tau=1 gate
    ev = {"t10": {"common_support": {"delta_U": 0.0, "component_deltas":
                                     {f"{r}_{m}": 0.0 for r in ("ET", "TC", "WT")
                                      for m in ("DSC", "NSD")}, "n_subjects": 806},
                  "official": {"delta_U": 0.0, "denominator_changes": {}},
                  "bootstrap": {"prob_positive": 0.5, "ci95": [-0.1, 0.1]},
                  "fold_deltas": {"0": 0.0, "1": 0.0, "2": 0.0},
                  "zero_dsc": {"baseline_region_cases": 33, "candidate_region_cases": 33,
                               "baseline_unique_subjects": 24,
                               "candidate_unique_subjects": 24,
                               "baseline_et": 22, "candidate_et": 22}},
          "t05": {"common_support": {"delta_U": 0.0, "component_deltas":
                                     {f"{r}_{m}": 0.0 for r in ("ET", "TC", "WT")
                                      for m in ("DSC", "NSD")}},
                  "official": {"delta_U": 0.0}},
          "lesion": {"baseline": {"FN": 10, "FP": 10}, "candidate": {"FN": 10, "FP": 10},
                     "fn_increase_fraction": 0.0, "fp_increase_fraction": 0.0},
          "evaluator_errors": 0, "independent_recompute_agrees": True, "n_cases": 811}
    ch = D._checks(ev, g, pooled=False)
    check("7b exact_tie_does_not_advance", not all(ch.values()),
          "failing: " + ",".join(k for k, v in ch.items() if not v))


def t_tau_explicit():
    import g83_eval as EV
    check("8 both_tolerances_preregistered",
          sorted(SPEC["metric"]["nsd_tolerances_explicit"]) == [0.5, 1.0])
    check("8b tolerance_never_implicit",
          SPEC["metric"]["tolerance_never_implicit"] is True)
    check("8c evaluator_evaluates_both_taus", set(EV.TAUS) == {0.5, 1.0})
    src = open(os.path.join(ROOT, "scripts", "g82_eval.py")).read()
    check("8d adapter_called_with_explicit_tau",
          "region_components(seg_ref, seg_c, 0.5)" in src
          and "region_components(seg_ref, seg_c, 1.0)" in src)
    adapter = open("/workspace/brats-2026-gat26/scripts/g79v_tau_nsd_adapter.py").read() \
        if os.path.exists("/workspace/brats-2026-gat26/scripts/g79v_tau_nsd_adapter.py") else ""
    if adapter:
        check("8e adapter_refuses_implicit_tau", "tau must be explicit" in adapter)


# ------------------------------------------------------------ tracked-artifact
def t_no_identifiers_or_percase_in_tracked_artifacts():
    # scanner-patterns-begin
    # Provider resource identifiers are supplied by the private runtime environment
    # and are deliberately NOT written here: this file is part of the public export.
    protected = [p for p in os.environ.get("GAT26_PROTECTED_IDS", "").split(",") if p]
    pats = {
        "case_identifier": r"BraTS-GoAT-\d{5}",
        "runpod_key": r"rpa_[A-Za-z0-9]{20,}",
        "s3_secret": r"rps_[A-Za-z0-9]{20,}",
        "private_path": r"/(?:root|dev/shm)/[A-Za-z0-9_]",
    }
    if protected:
        pats["provider_resource_id"] = "|".join(re.escape(p) for p in protected)
    # scanner-patterns-end
    offenders = []
    scanned = 0
    for sub in ("configs", "scripts", "tests", "artifacts"):
        d = os.path.join(ROOT, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            # G83 owns only its own files; pre-existing stage files are governed by
            # their own committed scans and are out of scope here.
            if not (fn.startswith("g83") or fn.startswith("test_g83")):
                continue
            p = os.path.join(d, fn)
            if not os.path.isfile(p):
                continue
            scanned += 1
            txt = open(p, encoding="utf8", errors="replace").read()
            # this file necessarily contains the patterns themselves; exclude that block
            if "scanner-patterns-begin" in txt:
                head, _, rest = txt.partition("# scanner-patterns-begin")
                _, _, tail = rest.partition("# scanner-patterns-end")
                txt = head + tail
            for name, pat in pats.items():
                m = re.search(pat, txt)
                if m:
                    offenders.append((fn, name))
    check("9 no_identifiers_or_private_paths_in_g83_artifacts", not offenders,
          f"scanned {scanned}" if not offenders else str(offenders))
    check("9b spec_forbids_percase_upload",
          "per-case metrics" in SPEC["runtime_gate_hf_a10g_proxy"]["never_upload_to_hf"])


def t_c0_immutable():
    src_all = ""
    for sub in ("scripts", "tests"):
        d = os.path.join(ROOT, sub)
        for fn in sorted(os.listdir(d)):
            if fn.startswith("g83") and fn.endswith(".py"):
                src_all += open(os.path.join(d, fn)).read()
    # markers are assembled from fragments so this scanner does not match itself,
    # nor trip the equivalent G82 scanner that also reads this directory
    markers = ["build_" + "context/weights", "checkpoint_final.pth\", \"w",
               "shutil." + "rmtree", "os." + "remove("]
    bad = [w for w in markers if w in src_all]
    check("10 g83_code_never_writes_or_deletes_c0", not bad, str(bad))
    check("10b spec_forbids_new_weights",
          any("new weights" in x for x in SPEC["prohibited"])
          and any("training or fine-tuning" in x for x in SPEC["prohibited"]))


def t_preregistration_immutable():
    """The preregistration must be committed and its digest recorded."""
    digest = hashlib.sha256(open(SPEC_PATH, "rb").read()).hexdigest()
    stamp = os.path.join(ROOT, "artifacts", "g83_preregistration_digest.json")
    if os.path.exists(stamp):
        rec = json.load(open(stamp))
        check("11 preregistration_byte_identical_since_commit",
              rec.get("sha256") == digest,
              "digest drifted" if rec.get("sha256") != digest else "")
    else:
        check("11 preregistration_byte_identical_since_commit", True,
              "digest stamp not yet written")
    check("11b no_result_dependent_edits_declared",
          SPEC["no_result_dependent_edits"] is True)


def t_resource_policy():
    rp = SPEC["resource_policy"]
    check("12 evaluator_process_cap_at_proven_value", rp["max_evaluator_processes"] == 48)
    check("12b thread_caps_declared",
          rp["evaluator_thread_caps"] == {"OMP_NUM_THREADS": 1, "MKL_NUM_THREADS": 1,
                                          "OPENBLAS_NUM_THREADS": 1})
    check("12c no_broad_pgrep_in_g83_code",
          all("pgrep -f" not in open(os.path.join(ROOT, "scripts", f)).read()
              for f in os.listdir(os.path.join(ROOT, "scripts")) if f.startswith("g83")))
    import g83_eval as EV
    src = inspect.getsource(EV.main)
    check("12d evaluator_refuses_more_than_48_processes", "48" in src and "refusing" in src)
    check("12e gpu_cap_is_four", rp["max_simultaneous_project_gpus_all_providers"] == 4)
    check("12f cost_cap_is_40", rp["incremental_cost_cap_usd"] == 40)


def main() -> int:
    t_candidate_set_closed()
    t_step_refusal()
    t_frozen_knobs_refused()
    t_predictor_only_varies_step()
    t_sealed_confirmation_folds()
    t_submission_ordering_and_uniqueness()
    t_baseline_wins_ties()
    t_tau_explicit()
    t_no_identifiers_or_percase_in_tracked_artifacts()
    t_c0_immutable()
    t_preregistration_immutable()
    t_resource_policy()

    n = len(RESULTS)
    ok = sum(1 for _, o, _ in RESULTS if o)
    print(f"\n{ok}/{n} checks passed")
    if ok != n:
        print("FAILED:", [r[0] for r in RESULTS if not r[1]])
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
