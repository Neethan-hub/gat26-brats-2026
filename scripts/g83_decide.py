#!/usr/bin/env python3
"""G83 gate arbitration for calibration (§F) and confirmation (§G).

Every threshold is read from the frozen preregistration; nothing is derived from
the observed numbers, and a failed gate is reported as failed rather than
reinterpreted. The baseline wins all ties.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REGIONS = ("ET", "TC", "WT")


def _checks(ev: dict, g: dict, pooled: bool) -> dict:
    t1, t05 = ev["t10"], ev["t05"]
    c1, c05 = t1["common_support"], t05["common_support"]
    boot = t1["bootstrap"]
    z1 = t1["zero_dsc"]
    les = ev["lesion"]
    cd1, cd05 = c1["component_deltas"], c05["component_deltas"]
    pre = "pooled_" if pooled else ""

    ch = {}
    ch["delta_U_common_tau1"] = c1["delta_U"] >= g[f"{pre}delta_U_common_tau1_min"]
    if pooled:
        ch["delta_U_common_tau05_positive"] = c05["delta_U"] > 0
    else:
        ch["delta_U_common_tau05"] = c05["delta_U"] >= g["delta_U_common_tau05_min"]
    ch["official_deltas_positive_both_tolerances"] = (
        t1["official"]["delta_U"] > 0 and t05["official"]["delta_U"] > 0)
    ch["bootstrap_probability"] = (
        boot["prob_positive"] >= g[("bootstrap_prob_positive_min" if pooled
                                    else "bootstrap_prob_delta_U_common_tau1_positive_min")])
    if not pooled:
        ch["bootstrap_lower_bound_above_zero"] = boot["ci95"][0] > 0
        ch["nonnegative_components_tau05"] = (
            sum(1 for v in cd05.values() if v >= 0)
            >= g["min_nonnegative_common_support_components_tau05"])
        ch["enough_folds_nonnegative"] = (
            sum(1 for v in t1["fold_deltas"].values() if v >= 0)
            >= g["min_folds_with_nonnegative_tau1_delta"])
        ch["no_fold_collapses"] = all(v >= g["min_fold_delta"]
                                      for v in t1["fold_deltas"].values())
    else:
        ch["no_individual_fold_collapses"] = all(
            v >= g["min_individual_fold_delta"] for v in t1["fold_deltas"].values())
    ch["nonnegative_components_tau1"] = (
        sum(1 for v in cd1.values() if v >= 0)
        >= g[("min_nonnegative_pooled_components_tau1" if pooled
              else "min_nonnegative_common_support_components_tau1")])
    ch["dsc_regression_within_limit"] = all(
        cd1[f"{r}_DSC"] >= -g["max_dsc_component_regression"] for r in REGIONS)
    ch["nsd_regression_within_limit"] = all(
        cd1[f"{r}_NSD"] >= -g["max_nsd_component_regression"] for r in REGIONS)
    ch["et_zero_dsc_not_increased"] = z1["candidate_et"] <= z1["baseline_et"]
    ch["total_zero_dsc_not_increased"] = (
        z1["candidate_region_cases"] <= z1["baseline_region_cases"])
    ch["unique_subject_zero_dsc_not_increased"] = (
        z1["candidate_unique_subjects"] <= z1["baseline_unique_subjects"])
    ch["lesion_fn_not_increased"] = les["candidate"]["FN"] <= les["baseline"]["FN"]
    ch["lesion_fp_within_limit"] = (
        les["fp_increase_fraction"] <= g["max_lesion_fp_increase_fraction"])
    dchg = t1["official"]["denominator_changes"]
    ch["no_denominator_only_gain"] = (
        all(v == 0 for v in dchg.values()) or c1["delta_U"] > 0)
    ch["zero_evaluator_errors"] = ev.get("evaluator_errors", 0) == 0
    ch["independent_recompute_agrees"] = ev.get("independent_recompute_agrees", False)
    return ch


def evidence(ev: dict) -> dict:
    t1, t05 = ev["t10"], ev["t05"]
    return {
        "n_cases": ev["n_cases"],
        "n_common_support_tau1": t1["common_support"]["n_subjects"],
        "delta_U_common_tau1": t1["common_support"]["delta_U"],
        "delta_U_common_tau05": t05["common_support"]["delta_U"],
        "delta_U_official_tau1": t1["official"]["delta_U"],
        "delta_U_official_tau05": t05["official"]["delta_U"],
        "component_deltas_tau1": t1["common_support"]["component_deltas"],
        "component_deltas_tau05": t05["common_support"]["component_deltas"],
        "denominators_tau1": t1["official"]["baseline_denominators"],
        "denominator_changes_tau1": t1["official"]["denominator_changes"],
        "fold_deltas_tau1": t1["fold_deltas"],
        "bootstrap": t1["bootstrap"],
        "zero_dsc": t1["zero_dsc"],
        "lesion": ev["lesion"],
        "independent_recompute_max_abs_difference":
            ev.get("independent_recompute_max_abs_difference"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--phase", choices=("calibration", "confirmation"), required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--all-folds-eval", default="",
                    help="confirmation only: the pooled five-fold evaluation")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec = json.load(open(a.spec))
    ev = json.load(open(a.eval))
    pooled = a.phase == "confirmation"
    g = spec["confirmation_gates_all_required" if pooled
             else "calibration_gates_all_required"]

    ch = _checks(ev, g, pooled)
    report = {"schema": "gat26.g83.decision.v1", "phase": a.phase,
              "checks": ch, "evidence": evidence(ev)}

    if pooled and a.all_folds_eval and os.path.exists(a.all_folds_eval):
        allev = json.load(open(a.all_folds_eval))
        ag = g["all_five_folds"]
        a1, a05 = allev["t10"], allev["t05"]
        report["all_five_folds"] = {
            "checks": {
                "delta_U_common_tau1": (a1["common_support"]["delta_U"]
                                        >= ag["delta_U_common_tau1_min"]),
                "delta_U_common_tau05_positive": a05["common_support"]["delta_U"] > 0,
                "bootstrap_probability": (a1["bootstrap"]["prob_positive"]
                                          >= ag["bootstrap_prob_positive_min"]),
                "effect_direction_agrees": (
                    (a1["common_support"]["delta_U"] > 0)
                    == (ev["t10"]["common_support"]["delta_U"] > 0)),
            },
            "evidence": evidence(allev)}
        ch = {**ch, **{f"all_folds_{k}": v
                       for k, v in report["all_five_folds"]["checks"].items()}}
        report["checks"] = ch

    report["passes"] = all(ch.values())
    report["failed_gates"] = sorted(k for k, v in ch.items() if not v)
    if report["passes"]:
        report["terminal_status"] = ("CONFIRMATION_PASSED" if pooled
                                     else "CALIBRATION_PASSED")
    else:
        report["terminal_status"] = ("G83_RETAIN_C0_D25_CONFIRMATION_FAILURE" if pooled
                                     else "G83_RETAIN_C0_D25_CALIBRATION_FAILURE")

    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, a.out)

    e = report["evidence"]
    print(f"{a.phase}: dU1={e['delta_U_common_tau1']:+.6f} "
          f"dU05={e['delta_U_common_tau05']:+.6f} "
          f"P={e['bootstrap']['prob_positive']:.4f} "
          f"CI=[{e['bootstrap']['ci95'][0]:+.6f},{e['bootstrap']['ci95'][1]:+.6f}]")
    if report["failed_gates"]:
        print("FAILED GATES:", report["failed_gates"])
    print("STATUS:", report["terminal_status"])
    return 0 if report["passes"] else 3


if __name__ == "__main__":
    sys.exit(main())
