#!/usr/bin/env python3
"""G84 §C synthetic and governance proofs. No GPU, no protected data, no credential."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

for _v in ("G84_RAW", "G84_CAND_STORE"):
    os.environ.setdefault(_v, os.path.join(HERE, "_unused_in_tests"))

import g84_tta_predict as P  # noqa: E402
import g84_decide as D  # noqa: E402

SPEC_PATH = os.path.join(ROOT, "configs", "g84_release_tta_preregistration.json")
SPEC = json.load(open(SPEC_PATH))
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# ------------------------------------------------------------------ candidates
def t_candidate_set():
    check("1 candidate_set_is_exactly_C0_and_M8", SPEC["candidates"] == ["C0", "M8"])
    check("1b no_other_candidate_permitted", SPEC["no_other_candidate_permitted"] is True)
    check("1c exact_difference_is_use_mirroring",
          SPEC["exact_scientific_difference"] == "use_mirroring: false -> true")
    check("1d code_candidate_set_matches_spec", sorted(P.POLICIES) == sorted(SPEC["candidates"]))
    bad = []
    for name in ("D25", "M4", "TDG", "C0_P", "M8_D25", "soup"):
        try:
            P.assert_policy(name)
            bad.append(name)
        except P.PolicyViolation:
            pass
    check("1e candidate_set_cannot_expand", not bad, str(bad))


def t_m8_differs_only_in_mirroring():
    c0, m8 = SPEC["candidate_definitions"]["C0"], SPEC["candidate_definitions"]["M8"]
    check("2 spec_says_m8_identical_except_mirroring",
          m8["identical_to_C0_except"] == "use_mirroring")
    check("2b m8_tile_step_unchanged",
          m8["tile_step_size"] == c0["tile_step_size"] == 0.5)
    check("2c code_policies_differ_only_in_mirroring",
          P.POLICIES["C0"] == {"use_mirroring": False}
          and P.POLICIES["M8"] == {"use_mirroring": True})
    check("2d frozen_values_match_release_c0",
          P.FROZEN["tile_step_size"] == 0.5 and P.FROZEN["use_gaussian"] is True
          and P.FROZEN["threshold"] == 0.5
          and P.FROZEN["checkpoint_name"] == "checkpoint_final.pth"
          and P.FROZEN["cleanup"] == "none" and P.FROZEN["presence_gate"] == "none")


def t_axes_exact():
    check("3 mirror_axes_are_exactly_012", P.MIRROR_AXES == (0, 1, 2))
    check("3b spec_axes_match_code",
          tuple(SPEC["candidate_definitions"]["M8"]["allowed_mirroring_axes"]) == P.MIRROR_AXES)
    bad = []
    for axes in ((0, 1), (0,), (1, 2), (0, 1, 2, 3), (), (2, 1, 0)):
        try:
            P.assert_axes(axes)
            bad.append(axes)
        except P.PolicyViolation:
            pass
    check("3c wrong_axis_sets_refused", not bad, str(bad))
    try:
        P.assert_axes((0, 1, 2))
        ok = True
    except P.PolicyViolation:
        ok = False
    check("3d exact_axes_accepted", ok)


def t_frozen_knobs():
    bad = []
    for key, value in (("tile_step_size", 0.25), ("tile_step_size", 0.4),
                       ("threshold", 0.4), ("threshold", 0.6), ("use_gaussian", False),
                       ("checkpoint_name", "checkpoint_best.pth"),
                       ("fold_weighting", "weighted"), ("cleanup", "size_filter"),
                       ("presence_gate", "on"), ("reconstruction", "argmax"),
                       ("config", "3d_lowres")):
        try:
            P.assert_policy("M8", **{key: value})
            bad.append((key, value))
        except P.PolicyViolation:
            pass
    check("4 step_threshold_gaussian_checkpoint_foldweight_cleanup_presence_refused",
          not bad, str(bad))
    unknown = []
    for key in ("tta_axes", "soup", "blend", "postprocess", "adaptation"):
        try:
            P.assert_policy("M8", **{key: 1})
            unknown.append(key)
        except P.PolicyViolation:
            pass
    check("4b unknown_knobs_refused", not unknown, str(unknown))
    src = inspect.getsource(P.build_predictor)
    check("4c predictor_uses_frozen_tile_step_and_gaussian",
          "tile_step_size=TILE_STEP" in src and 'FROZEN["use_gaussian"]' in src)
    check("4d predictor_verifies_axes_at_construction",
          "assert_axes(pred.allowed_mirroring_axes)" in src)
    check("4e c0_carries_no_mirroring_axes", "must carry no mirroring axes" in src)
    ens = inspect.getsource(P.ensemble_probabilities)
    check("4f equal_mean_in_original_checkpoint_order",
          "acc / float(len(weights))" in ens and "for k in range(len(weights))" in ens)


# ----------------------------------------------------------------- sealed folds
def t_sealed_folds():
    import g84_eval as EV
    freeze = os.path.join(ROOT, "artifacts", "g84_candidate_freeze.json")
    raised = False
    try:
        EV.assert_folds_allowed([3, 4], freeze + ".absent")
    except EV.SealedFoldError:
        raised = True
    check("5 confirmation_folds_sealed_without_freeze", raised)
    try:
        EV.assert_folds_allowed([0, 1, 2], freeze + ".absent")
        ok = True
    except EV.SealedFoldError:
        ok = False
    check("5b calibration_folds_readable", ok)
    check("5c spec_requires_freeze_first",
          SPEC["splits"]["confirmation_opens_only_after_candidate_freeze_commit"] is True
          and SPEC["splits"]["confirmation_executes_once_only"] is True)
    if os.path.exists(freeze):
        check("5d freeze_artifact_records_calibration_pass",
              json.load(open(freeze)).get("calibration_passed") is True)
    else:
        check("5d freeze_artifact_records_calibration_pass", True, "not yet frozen")


def t_submission_ordering():
    s = SPEC["official_validation_submission"]
    check("6 one_submission_max", s["max_submissions_this_stage"] == 1)
    check("6b no_automatic_retry", s["no_automatic_retry"] is True)
    check("6c queues_redacted_in_public_export",
          s["predictions_queue"] is None and s["docker_queue_never_used"] is None
          and "intentionally removed from the public export"
          in s.get("queue_identifiers_redacted_for_public_export", ""))
    receipt = os.path.join(ROOT, "artifacts", "g84_submission_receipt.json")
    need = [os.path.join(ROOT, "artifacts", f) for f in
            ("g84_calibration_decision.json", "g84_candidate_freeze.json",
             "g84_confirmation_decision.json", "g84_runtime_qualification.json")]
    if os.path.exists(receipt):
        check("6d submission_only_after_all_prior_gates", all(os.path.exists(p) for p in need),
              "missing: " + ",".join(os.path.basename(p) for p in need
                                     if not os.path.exists(p)))
    else:
        check("6d submission_only_after_all_prior_gates", True, "no submission yet")


def t_ties_retain_c0():
    check("7 baseline_wins_ties", SPEC["metric"]["baseline_wins_ties"] is True)
    g = SPEC["calibration_gates_all_required"]
    zeros = {c: 0.0 for c in
             [f"{r}_{m}" for m in ("DSC", "NSD") for r in ("ET", "TC", "WT")]}
    ev = {"t10": {"common_support": {"delta_U": 0.0, "component_deltas": dict(zeros),
                                     "n_subjects": 806},
                  "official": {"delta_U": 0.0, "denominator_changes": {},
                               "baseline_denominators": {}},
                  "bootstrap": {"prob_positive": 0.5, "ci95": [-0.001, 0.001]},
                  "fold_deltas": {"0": 0.0, "1": 0.0, "2": 0.0},
                  "zero_dsc": {"baseline_region_cases": 33, "candidate_region_cases": 33,
                               "baseline_unique_subjects": 24,
                               "candidate_unique_subjects": 24,
                               "baseline_et": 22, "candidate_et": 22}},
          "t05": {"common_support": {"delta_U": 0.0, "component_deltas": dict(zeros)},
                  "official": {"delta_U": 0.0}},
          "lesion": {"baseline": {"FN": 10, "FP": 10}, "candidate": {"FN": 10, "FP": 10},
                     "fn_increase_fraction": 0.0, "fp_increase_fraction": 0.0},
          "evaluator_errors": 0, "independent_recompute_agrees": True,
          "exact_membership": True, "n_cases": 811}
    ch = D.calibration_checks(ev, g)
    check("7b exact_tie_does_not_advance", not all(ch.values()),
          "failing: " + ",".join(sorted(k for k, v in ch.items() if not v)))


# --- r11 authorized exceptions -----------------------------------------------------------------
# These guards exist so a frozen stage's committed files cannot drift after its audit. The r11
# correction pass was explicitly instructed to change three of them, and each change is a
# publication or fail-closed correction, never a scientific one: no recorded result, metric, count,
# interval, threshold, gate outcome, candidate decision or release decision is touched by any of
# them. They are enumerated here so the guard keeps failing closed on anything else.
R11_AUTHORIZED_CHANGES = {
    "tests/test_g83_science.py":
        "r11: check 8e loaded the NSD adapter from a machine-absolute private path behind a "
        "conditional, so on every public export the check left the tally with no failure while the "
        "file still reported all checks passed. The load is now repository-relative and fail-closed.",
    "scripts/g79v_tau_nsd_adapter.py":
        "r11: published, so check 8e can execute in a public tree. The only edit is removal of a "
        "machine-absolute worker path default, which is now resolved from the environment.",
    "scripts/g84_eval.py":
        "r11: the script root defaulted to a machine-absolute path that shadowed the module's own "
        "directory on any other checkout. It is now repository-relative.",
    "tests/test_g84_science.py":
        "r11: this file carries the same authorized-exception list, for the same three changes.",
}


def t_frozen_prior_stage_files():
    """G82 and G83 committed files must be byte-identical to their commits."""
    prior = subprocess.run(
        ["git", "-C", ROOT, "diff", "--name-only",
         "d0fbbb388f45fd45f1ad7236dcbcf20a95daf1e5", "--"],
        capture_output=True, text=True).stdout.split()
    touched = [f for f in prior
               if (f.startswith(("scripts/g82", "scripts/g83", "configs/g82", "configs/g83",
                                 "tests/test_g82", "tests/test_g83"))
                   or f.startswith("artifacts/G82") or f.startswith("artifacts/g82")
                   or f.startswith("artifacts/G83") or f.startswith("artifacts/g83"))]
    unauthorized = [f for f in touched if f not in R11_AUTHORIZED_CHANGES]
    check("8 g82_and_g83_frozen_files_unmodified", not unauthorized, str(unauthorized))
    for f in sorted(set(touched) & set(R11_AUTHORIZED_CHANGES)):
        print(f"  ..   authorized r11 change, still guarded elsewhere: {f}")


def t_c0_immutable():
    src = ""
    for sub in ("scripts", "tests"):
        d = os.path.join(ROOT, sub)
        for fn in sorted(os.listdir(d)):
            if fn.startswith(("g84", "test_g84")) and fn.endswith(".py"):
                src += open(os.path.join(d, fn)).read()
    markers = ["build_" + "context/weights", "shutil." + "rmtree", "os." + "remove(",
               "checkpoint_final.pth\", \"w"]
    check("9 g84_never_writes_or_deletes_c0", not [m for m in markers if m in src])
    check("9b spec_forbids_training_and_checkpoint_change",
          any("training" in x for x in SPEC["prohibited"])
          and any("checkpoint change" in x for x in SPEC["prohibited"])
          and any("checkpoint deletion" in x for x in SPEC["prohibited"]))


def t_no_identifiers():
    # scanner-patterns-begin
    protected = [p for p in os.environ.get("GAT26_PROTECTED_IDS", "").split(",") if p]
    pats = {"case_identifier": r"BraTS-GoAT-\d{5}",
            "runpod_key": r"rpa_[A-Za-z0-9]{20,}",
            "s3_secret": r"rps_[A-Za-z0-9]{20,}",
            "private_path": r"/(?:root|dev/shm)/[A-Za-z0-9_]"}
    if protected:
        pats["provider_resource_id"] = "|".join(re.escape(p) for p in protected)
    # scanner-patterns-end
    offenders, scanned = [], 0
    for sub in ("configs", "scripts", "tests", "artifacts"):
        d = os.path.join(ROOT, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not (fn.startswith("g84") or fn.startswith("test_g84")
                    or fn.startswith("G84")):
                continue
            p = os.path.join(d, fn)
            if not os.path.isfile(p):
                continue
            scanned += 1
            txt = open(p, encoding="utf8", errors="replace").read()
            if "scanner-patterns-begin" in txt:
                h, _, rest = txt.partition("# scanner-patterns-begin")
                _, _, t = rest.partition("# scanner-patterns-end")
                txt = h + t
            for name, pat in pats.items():
                if re.search(pat, txt):
                    offenders.append((fn, name))
    check("10 no_identifiers_or_private_paths_in_g84_files", not offenders,
          f"scanned {scanned}" if not offenders else str(offenders))


def t_disclosure_and_history():
    d = SPEC["candidate_choice_disclosure"]
    check("11 stage_declared_not_blind", d["blind"] is False)
    check("11b historical_evidence_disclosed",
          abs(d["historical_planning_observations"]["tau1_aggregate_uplift_approx"]
              - 0.002509) < 1e-9
          and abs(d["historical_planning_observations"]["tau05_aggregate_uplift_approx"]
                  - 0.004132) < 1e-9
          and d["historical_planning_observations"]["all_six_component_differences_positive"]
          is True)
    check("11c historical_values_not_admissible_as_candidate_evidence",
          "NOT admissible" in d["historical_planning_observations"]["provenance"])
    check("11d baseline_is_release_path_not_tta",
          SPEC["baseline"]["never_use_the_internal_validation_tta_values_as_baseline"] is True)


def t_prereg_immutable_and_resources():
    digest = hashlib.sha256(open(SPEC_PATH, "rb").read()).hexdigest()
    stamp = os.path.join(ROOT, "artifacts", "g84_preregistration_digest.json")
    if os.path.exists(stamp):
        check("12 preregistration_byte_identical",
              json.load(open(stamp)).get("sha256") == digest)
    else:
        check("12 preregistration_byte_identical", True, "digest not yet stamped")
    rp = SPEC["resource_policy"]
    check("12b evaluator_cap_48_and_80_88_prohibited",
          rp["max_evaluator_processes"] == 48
          and rp["prohibited_evaluator_process_counts"] == [80, 88])
    check("12c gpu_and_cost_caps", rp["max_simultaneous_project_gpus"] == 4
          and rp["max_new_workers"] == 3 and rp["incremental_cost_cap_usd"] == 60
          and rp["max_price_per_gpu_hour_usd"] == 2.50)
    check("12d no_broad_pgrep_in_g84_code",
          all("pgrep -f" not in open(os.path.join(ROOT, "scripts", f)).read()
              for f in os.listdir(os.path.join(ROOT, "scripts")) if f.startswith("g84")))
    import g84_eval as EV
    check("12e evaluator_refuses_more_than_48", EV.MAX_PROCS == 48
          and "refusing more than" in inspect.getsource(EV.main))
    check("12f segmentations_stored_directly",
          SPEC["storage"]["mode"] == "final segmentations stored directly")


def main() -> int:
    for fn in (t_candidate_set, t_m8_differs_only_in_mirroring, t_axes_exact,
               t_frozen_knobs, t_sealed_folds, t_submission_ordering, t_ties_retain_c0,
               t_frozen_prior_stage_files, t_c0_immutable, t_no_identifiers,
               t_disclosure_and_history, t_prereg_immutable_and_resources):
        fn()
    n = len(RESULTS)
    ok = sum(1 for _, o, _ in RESULTS if o)
    print(f"\n{ok}/{n} checks passed")
    if ok != n:
        print("FAILED:", [r[0] for r in RESULTS if not r[1]])
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
