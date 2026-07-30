#!/usr/bin/env python3
"""G84 gate arbitration for calibration (§G) and confirmation (§H).

Every threshold comes from the frozen preregistration. A failed gate is reported as
failed and never reinterpreted, and the baseline wins exact ties.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REGIONS = ("ET", "TC", "WT")
COMPONENTS = [f"{r}_{m}" for m in ("DSC", "NSD") for r in REGIONS]


def _common(ev, tau):
    return ev[tau]["common_support"]


def calibration_checks(ev, g):
    t1, t05 = ev["t10"], ev["t05"]
    c1, c05 = t1["common_support"], t05["common_support"]
    boot, z1, les = t1["bootstrap"], t1["zero_dsc"], ev["lesion"]
    ch = {}
    ch["dU_common_tau1"] = c1["delta_U"] >= g["delta_U_common_tau1_min"]
    ch["dU_common_tau05"] = c05["delta_U"] >= g["delta_U_common_tau05_min"]
    ch["official_deltas_positive"] = (t1["official"]["delta_U"] > 0
                                      and t05["official"]["delta_U"] > 0)
    ch["all_components_nonnegative_tau1"] = all(
        v >= 0 for v in c1["component_deltas"].values())
    ch["all_components_nonnegative_tau05"] = all(
        v >= 0 for v in c05["component_deltas"].values())
    ch["bootstrap_probability"] = boot["prob_positive"] >= g["bootstrap_prob_tau1_positive_min"]
    ch["bootstrap_lower_bound_above_zero"] = boot["ci95"][0] > 0
    ch["enough_folds_nonnegative"] = (
        sum(1 for v in t1["fold_deltas"].values() if v >= 0)
        >= g["min_folds_with_nonnegative_tau1_delta"])
    ch["no_fold_collapses"] = all(v >= g["min_fold_delta"]
                                  for v in t1["fold_deltas"].values())
    ch["et_zero_dsc_not_increased"] = z1["candidate_et"] <= z1["baseline_et"]
    ch["total_zero_dsc_not_increased"] = (z1["candidate_region_cases"]
                                          <= z1["baseline_region_cases"])
    ch["unique_subject_zero_dsc_not_increased"] = (z1["candidate_unique_subjects"]
                                                   <= z1["baseline_unique_subjects"])
    ch["lesion_fn_not_increased"] = les["candidate"]["FN"] <= les["baseline"]["FN"]
    ch["lesion_fp_within_limit"] = (les["fp_increase_fraction"]
                                    <= g["max_lesion_fp_increase_fraction"])
    dchg = t1["official"]["denominator_changes"]
    ch["no_denominator_only_gain"] = all(v == 0 for v in dchg.values()) or c1["delta_U"] > 0
    ch["exact_membership"] = ev.get("exact_membership", False)
    ch["zero_evaluator_errors"] = ev.get("evaluator_errors", 0) == 0
    ch["independent_recompute_agrees"] = ev.get("independent_recompute_agrees", False)
    return ch


def confirmation_checks(ev, g):
    t1, t05 = ev["t10"], ev["t05"]
    c1, c05 = t1["common_support"], t05["common_support"]
    boot, z1, les = t1["bootstrap"], t1["zero_dsc"], ev["lesion"]
    ch = {}
    ch["dU_common_tau1"] = c1["delta_U"] >= g["delta_U_common_tau1_min"]
    ch["dU_common_tau05"] = c05["delta_U"] >= g["delta_U_common_tau05_min"]
    ch["official_deltas_positive"] = (t1["official"]["delta_U"] > 0
                                      and t05["official"]["delta_U"] > 0)
    ch["bootstrap_probability"] = boot["prob_positive"] >= g["bootstrap_prob_positive_min"]
    ch["bootstrap_lower_bound_above_zero"] = boot["ci95"][0] > 0
    ch["five_of_six_nonnegative_tau1"] = (
        sum(1 for v in c1["component_deltas"].values() if v >= 0)
        >= g["min_nonnegative_components_each_tolerance"])
    ch["five_of_six_nonnegative_tau05"] = (
        sum(1 for v in c05["component_deltas"].values() if v >= 0)
        >= g["min_nonnegative_components_each_tolerance"])
    ch["dsc_within_limit"] = all(c1["component_deltas"][f"{r}_DSC"] >= g["min_dsc_component_delta"]
                                 for r in REGIONS)
    ch["nsd_within_limit"] = all(c1["component_deltas"][f"{r}_NSD"] >= g["min_nsd_component_delta"]
                                 for r in REGIONS)
    fold_vals = list(t1["fold_deltas"].values())
    ch["no_fold_collapses"] = all(v >= g["min_fold_delta"] for v in fold_vals)
    ch["at_least_one_fold_positive"] = any(v > 0 for v in fold_vals)
    ch["et_zero_dsc_not_increased"] = z1["candidate_et"] <= z1["baseline_et"]
    ch["total_zero_dsc_not_increased"] = (z1["candidate_region_cases"]
                                          <= z1["baseline_region_cases"])
    ch["unique_subject_zero_dsc_not_increased"] = (z1["candidate_unique_subjects"]
                                                   <= z1["baseline_unique_subjects"])
    ch["lesion_fn_not_increased"] = les["candidate"]["FN"] <= les["baseline"]["FN"]
    dchg = t1["official"]["denominator_changes"]
    ch["no_denominator_only_gain"] = all(v == 0 for v in dchg.values()) or c1["delta_U"] > 0
    ch["exact_membership"] = ev.get("exact_membership", False)
    ch["zero_evaluator_errors"] = ev.get("evaluator_errors", 0) == 0
    return ch


def pooled_checks(ev, g):
    t1, t05 = ev["t10"], ev["t05"]
    c1, c05 = t1["common_support"], t05["common_support"]
    ch = {}
    ch["pooled_dU_common_tau1"] = c1["delta_U"] >= g["delta_U_common_tau1_min"]
    ch["pooled_dU_common_tau05"] = c05["delta_U"] >= g["delta_U_common_tau05_min"]
    ch["pooled_bootstrap_probability"] = (t1["bootstrap"]["prob_positive"]
                                          >= g["bootstrap_prob_positive_min"])
    ch["pooled_five_of_six_tau1"] = (
        sum(1 for v in c1["component_deltas"].values() if v >= 0)
        >= g["min_nonnegative_components_each_tolerance"])
    ch["pooled_five_of_six_tau05"] = (
        sum(1 for v in c05["component_deltas"].values() if v >= 0)
        >= g["min_nonnegative_components_each_tolerance"])
    z1, les = t1["zero_dsc"], ev["lesion"]
    ch["pooled_zero_dsc_not_increased"] = (
        z1["candidate_region_cases"] <= z1["baseline_region_cases"]
        and z1["candidate_et"] <= z1["baseline_et"]
        and z1["candidate_unique_subjects"] <= z1["baseline_unique_subjects"])
    ch["pooled_lesion_fn_not_increased"] = les["candidate"]["FN"] <= les["baseline"]["FN"]
    return ch


def evidence(ev):
    t1, t05 = ev["t10"], ev["t05"]
    return {"n_cases": ev["n_cases"], "n_expected": ev.get("n_expected"),
            "exact_membership": ev.get("exact_membership"),
            "n_common_tau1": t1["common_support"]["n_subjects"],
            "dU_common_tau1": t1["common_support"]["delta_U"],
            "dU_common_tau05": t05["common_support"]["delta_U"],
            "dU_official_tau1": t1["official"]["delta_U"],
            "dU_official_tau05": t05["official"]["delta_U"],
            "component_deltas_tau1": t1["common_support"]["component_deltas"],
            "component_deltas_tau05": t05["common_support"]["component_deltas"],
            "baseline_means_tau1": t1["common_support"]["baseline_means"],
            "candidate_means_tau1": t1["common_support"]["candidate_means"],
            "baseline_means_tau05": t05["common_support"]["baseline_means"],
            "candidate_means_tau05": t05["common_support"]["candidate_means"],
            "denominators_tau1": t1["official"]["baseline_denominators"],
            "denominator_changes_tau1": t1["official"]["denominator_changes"],
            "nan_pattern_changes_tau1": t1.get("nan_pattern_changes"),
            "fold_deltas_tau1": t1["fold_deltas"], "bootstrap": t1["bootstrap"],
            "zero_dsc": t1["zero_dsc"], "lesion": ev["lesion"],
            "independent_recompute_max_abs_difference":
                ev.get("independent_recompute_max_abs_difference")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--phase", choices=("calibration", "confirmation"), required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--pooled-eval", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec = json.load(open(a.spec))
    ev = json.load(open(a.eval))
    if a.phase == "calibration":
        g = spec["calibration_gates_all_required"]
        ch = calibration_checks(ev, g)
        fail_status = g["on_failure"]
        pass_status = "CALIBRATION_PASSED"
    else:
        g = spec["confirmation_gates_all_required"]
        ch = confirmation_checks(ev, g)
        fail_status = g["on_failure"]
        pass_status = "CONFIRMATION_PASSED"

    report = {"schema": "gat26.g84.decision.v1", "phase": a.phase,
              "checks": ch, "evidence": evidence(ev)}

    if a.phase == "confirmation" and a.pooled_eval and os.path.exists(a.pooled_eval):
        pooled = json.load(open(a.pooled_eval))
        pch = pooled_checks(pooled, g["pooled_all_five_folds"])
        report["pooled_all_five_folds"] = {"checks": pch, "evidence": evidence(pooled)}
        ch = {**ch, **pch}
        report["checks"] = ch

    report["passes"] = all(ch.values())
    report["failed_gates"] = sorted(k for k, v in ch.items() if not v)
    report["terminal_status"] = pass_status if report["passes"] else fail_status

    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, a.out)

    e = report["evidence"]
    print(f"{a.phase}: dU1={e['dU_common_tau1']:+.6f} dU05={e['dU_common_tau05']:+.6f} "
          f"P={e['bootstrap']['prob_positive']:.4f} "
          f"CI=[{e['bootstrap']['ci95'][0]:+.6f},{e['bootstrap']['ci95'][1]:+.6f}]")
    if report["failed_gates"]:
        print("FAILED GATES:", report["failed_gates"])
    print("STATUS:", report["terminal_status"])
    return 0 if report["passes"] else 3


if __name__ == "__main__":
    sys.exit(main())
