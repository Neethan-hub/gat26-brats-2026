#!/usr/bin/env python3
"""GAT-26 selection-policy v2 tests (`python3 tests/test_g45_selection_policy.py`).

Exercises the executable engine end to end: metric direction, fractional ties, valid
INF->373, fail-closed on missing/error/subject-mismatch, bootstrap determinism, a known
L-win and a known tie, each noninferiority boundary, one-margin violation rejecting L, the
fold-0 decision, two-fold directional confirmation, and the simpler-model tie rule.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import g45_selection_policy as SP  # noqa: E402

FAILS = []


def check(name, cond):
    if not cond:
        FAILS.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def rec(subject, dsc, hd95, ok=True):
    """dsc/hd95: dict region->value for the three regions."""
    m = {(r, "DSC"): dsc[r] for r in ("ET", "TC", "WT")}
    m.update({(r, "HD95"): hd95[r] for r in ("ET", "TC", "WT")})
    return {"subject": subject, "evaluated_ok": ok, "metrics": m}


def make_fold(subjects, dsc_M, hd_M, dsc_L, hd_L):
    M = [rec(s, dsc_M, hd_M) for s in subjects]
    L = [rec(s, dsc_L, hd_L) for s in subjects]
    return {"M": M, "L": L}


def main():
    good_dsc = {"ET": 0.85, "TC": 0.90, "WT": 0.92}
    good_hd = {"ET": 3.0, "TC": 2.5, "WT": 2.0}

    # 1. direction: strictly better L wins award rank
    aggM = {(r, "DSC"): 0.8 for r in ("ET", "TC", "WT")}; aggM.update({(r, "HD95"): 5.0 for r in ("ET", "TC", "WT")})
    aggL = {(r, "DSC"): 0.9 for r in ("ET", "TC", "WT")}; aggL.update({(r, "HD95"): 3.0 for r in ("ET", "TC", "WT")})
    ranks = SP.award_ranks({"M": aggM, "L": aggL})
    check("better_L_lower_rank", ranks["L"] < ranks["M"])
    check("exact_tie_equal_ranks", abs(SP.award_ranks({"M": aggM, "L": dict(aggM)})["M"] - 1.5) < 1e-9)

    # 2. INF -> 373 for a valid evaluated HD95; error/missing stay hard failures
    check("valid_inf_hd95_to_373", SP.parse_metric(float("inf"), "HD95", True) == 373.0)
    try:
        SP.parse_metric(float("inf"), "HD95", False); check("error_inf_hardfail", False)
    except SP.HardFailure:
        check("error_inf_hardfail", True)
    try:
        SP.parse_metric(None, "DSC", True); check("missing_dsc_hardfail", False)
    except SP.HardFailure:
        check("missing_dsc_hardfail", True)
    try:
        SP.parse_metric(float("inf"), "DSC", True); check("inf_dsc_hardfail", False)
    except SP.HardFailure:
        check("inf_dsc_hardfail", True)

    # 3. subject mismatch / duplicate hard failures
    subs = [f"S{i:03d}" for i in range(30)]
    fold = make_fold(subs, good_dsc, good_hd, good_dsc, good_hd)
    try:
        SP.validate_subject_records(fold["M"], fold["L"][:-1]); check("subject_mismatch_hardfail", False)
    except SP.HardFailure:
        check("subject_mismatch_hardfail", True)
    dup = fold["M"][:-1] + [fold["M"][0]]
    try:
        SP.validate_subject_records(dup, fold["L"]); check("duplicate_subject_hardfail", False)
    except SP.HardFailure:
        check("duplicate_subject_hardfail", True)
    # evaluator error record -> hard failure (not disguised as penalty)
    err = [dict(r) for r in fold["L"]]; err[0] = {**err[0], "evaluated_ok": False}
    try:
        SP.validate_subject_records(fold["M"], err); check("evaluator_error_hardfail", False)
    except SP.HardFailure:
        check("evaluator_error_hardfail", True)

    # 4. bootstrap determinism (identical input+seed -> identical result)
    s2, mM, mL = SP.validate_subject_records(fold["M"], fold["L"])
    b1 = SP.paired_bootstrap_compare(s2, mM, mL, seed=SP.SEED, n=500)
    b2 = SP.paired_bootstrap_compare(s2, mM, mL, seed=SP.SEED, n=500)
    check("bootstrap_deterministic", json.dumps(b1, sort_keys=True) == json.dumps(b2, sort_keys=True))
    check("tie_bootstrap_point_zero", abs(b1["delta_point"]) < 1e-9)  # identical M/L -> tie

    # 5. known L-win: L strictly better on every subject -> delta_point<0 and CI upper<0
    dsc_L = {"ET": 0.90, "TC": 0.95, "WT": 0.97}; hd_L = {"ET": 1.5, "TC": 1.0, "WT": 0.8}
    fold_Lwin = make_fold(subs, good_dsc, good_hd, dsc_L, hd_L)
    s3, mM3, mL3 = SP.validate_subject_records(fold_Lwin["M"], fold_Lwin["L"])
    bw = SP.paired_bootstrap_compare(s3, mM3, mL3, seed=SP.SEED, n=500)
    check("known_L_win_favors_L", bw["delta_point"] < 0 and bw["ci_excludes_tie_favoring_L"])

    # 6. noninferiority boundaries
    aggM = SP.aggregate_components(s2, mM)
    # exactly at DSC boundary: L = M - margin -> pass (>=)
    aggL_edge = dict(aggM); aggL_edge[("ET", "DSC")] = aggM[("ET", "DSC")] - SP.NONINF_MARGINS["regional_dsc_abs"]
    ni = SP.evaluate_noninferiority(aggM, aggL_edge)
    check("dsc_boundary_inclusive_pass", ni["dsc_ET"] is True)
    # just beyond -> fail
    aggL_bad = dict(aggM); aggL_bad[("ET", "DSC")] = aggM[("ET", "DSC")] - SP.NONINF_MARGINS["regional_dsc_abs"] - 1e-6
    check("dsc_beyond_boundary_fail", SP.evaluate_noninferiority(aggM, aggL_bad)["dsc_ET"] is False)
    # HD95 boundary
    aggL_h = dict(aggM); aggL_h[("WT", "HD95")] = aggM[("WT", "HD95")] + SP.NONINF_MARGINS["regional_hd95_mm"]
    check("hd95_boundary_inclusive_pass", SP.evaluate_noninferiority(aggM, aggL_h)["hd95_WT"] is True)
    aggL_hbad = dict(aggM); aggL_hbad[("WT", "HD95")] = aggM[("WT", "HD95")] + SP.NONINF_MARGINS["regional_hd95_mm"] + 1e-6
    check("hd95_beyond_boundary_fail", SP.evaluate_noninferiority(aggM, aggL_hbad)["hd95_WT"] is False)

    extra_ok = {"smallest_volume_dsc_M": 0.7, "smallest_volume_dsc_L": 0.75,
                "dsc_p05_M": 0.6, "dsc_p05_L": 0.66, "hd95_p95_M": 8.0, "hd95_p95_L": 7.0,
                "empty_ref_fp_M": 0.05, "empty_ref_fp_L": 0.05, "missed_region_M": 0.03, "missed_region_L": 0.03,
                "runtime_M": 100.0, "runtime_L": 118.0, "cost_M": 100.0, "cost_L": 118.0}

    # 7. B1 FAIL-CLOSED: fold-0 with MISSING auxiliary gates can NEVER trigger L (even a clear L-win)
    d0_missing = SP.decide_after_fold0(fold_Lwin)   # no extra provided
    check("fold0_missing_aux_gates_selects_M", d0_missing["decision"] == "select_M")
    check("fold0_missing_aux_gates_flagged", bool(d0_missing.get("fail_closed_missing_gates")))

    # 8. one-margin violation rejects L even with all aux gates supplied
    fold_violate = {**make_fold(subs, good_dsc, good_hd, dsc_L, hd_L), "extra": dict(extra_ok)}
    for r in fold_violate["L"]:
        r["metrics"][("TC", "HD95")] = good_hd["TC"] + 50.0   # violate hd95_TC
    check("one_margin_violation_rejects_L", SP.decide_after_fold0(fold_violate)["decision"] == "select_M")
    # a single failing auxiliary gate (runtime blowout) also rejects L
    bad_runtime = dict(extra_ok); bad_runtime["runtime_L"] = 200.0   # > M*1.25
    check("one_aux_gate_fail_selects_M",
          SP.decide_after_fold0({**fold_Lwin, "extra": bad_runtime})["decision"] == "select_M")

    # 9. all gates supplied AND passing AND genuine L benefit -> fold-1 confirmation (never expand on fold0)
    d0w = SP.decide_after_fold0({**fold_Lwin, "extra": extra_ok})
    check("fold0_all_gates_L_benefit_triggers_fold1", d0w["decision"] == "confirm_L_on_fold1")
    check("fold0_tie_selects_M", SP.decide_after_fold0({**fold, "extra": extra_ok})["decision"] == "select_M")

    # 10. two-fold confirmation requires full gates provided; missing extras -> select_M (fail-closed)
    check("twofold_missing_gates_fail_closed",
          SP.decide_after_two_folds(fold_Lwin, fold_Lwin)["decision"] == "select_M")
    f0 = {**fold_Lwin, "extra": extra_ok}; f1 = {**fold_Lwin, "extra": extra_ok}
    check("twofold_full_gates_L_may_expand",
          SP.decide_after_two_folds(f0, f1)["decision"] == "L_may_expand_subject_to_owner_budget_regate")

    # 10. emit + validate v2 document
    out = REPO / "configs" / "g45_selection_policy.json"
    if out.exists():
        import argparse
        check("emitted_v2_validates", SP.cmd_validate(argparse.Namespace(file=str(out))) == 0)
        p = json.loads(out.read_text())
        check("policy_is_v2", p["policy_id"] == "gat26_g45_selection_policy_v2")
        check("tie_rule_selects_M", p["tie_rule"]["rule"] == "select_M_the_cheaper_simpler_plan")

    print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
