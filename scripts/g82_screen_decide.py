#!/usr/bin/env python3
"""G82 §G screen decision: apply the preregistered eligibility gates and pick at
most one recipe. Purely mechanical — every threshold comes from the frozen
preregistration file, nothing is computed from the observed numbers.

Selection: among eligible recipes maximise min(dU_common_tau1, dU_common_tau05).
The baseline wins all ties; differences under 1e-4 count as ties; among tied
non-baseline recipes prefer T, then DG, then TDG.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

TIE = 1e-4
PREFERENCE = ["T", "DG", "TDG"]
REGIONS = ("ET", "TC", "WT")


def eligibility(ev: dict, spec: dict) -> dict:
    g = spec["screen"]["eligibility_all_required"]
    t1, t05 = ev["t10"], ev["t05"]
    c1, c05 = t1["common_support"], t05["common_support"]
    checks = {}

    checks["dU_common_tau1_nonneg"] = c1["delta_U"] >= g["delta_U_common_tau1_min"]
    checks["dU_common_tau05_nonneg"] = c05["delta_U"] >= g["delta_U_common_tau05_min"]
    checks["dU_official_tau1_nonneg"] = (
        t1["official"]["delta_U"] >= g["delta_U_official_tau1_min"])

    worst_dsc = min(c1["component_deltas"][f"{r}_DSC"] for r in REGIONS)
    worst_nsd = min(c1["component_deltas"][f"{r}_NSD"] for r in REGIONS)
    checks["no_dsc_component_regression_beyond_limit"] = (
        worst_dsc >= -g["max_common_support_dsc_component_regression"])
    checks["no_nsd_component_regression_beyond_limit"] = (
        worst_nsd >= -g["max_common_support_nsd_component_regression"])

    folds_ok = sum(1 for v in t1["fold_deltas"].values() if v >= 0)
    checks["enough_folds_nonnegative"] = (
        folds_ok >= g["min_folds_with_nonnegative_delta_U_common_tau1"])

    z = t1["zero_dsc"]
    checks["total_zero_dsc_not_increased"] = (
        z["candidate_region_cases"] <= z["baseline_region_cases"])

    les = ev["lesion"]
    checks["lesion_fn_within_limit"] = (
        les["fn_increase_fraction"] <= g["max_lesion_fn_increase_fraction"])
    checks["lesion_fp_within_limit"] = (
        les["fp_increase_fraction"] <= g["max_lesion_fp_increase_fraction"])
    checks["no_evaluator_errors"] = ev.get("evaluator_errors", 0) == 0

    return {
        "checks": checks,
        "eligible": all(checks.values()),
        "score": min(c1["delta_U"], c05["delta_U"]),
        "evidence": {
            "dU_common_tau1": c1["delta_U"],
            "dU_common_tau05": c05["delta_U"],
            "dU_official_tau1": t1["official"]["delta_U"],
            "dU_official_tau05": t05["official"]["delta_U"],
            "worst_dsc_component_delta": worst_dsc,
            "worst_nsd_component_delta": worst_nsd,
            "folds_nonnegative": folds_ok,
            "n_common_subjects": c1["n_subjects"],
            "zero_dsc_region_cases": [z["baseline_region_cases"], z["candidate_region_cases"]],
            "zero_dsc_unique_subjects": [z["baseline_unique_subjects"],
                                         z["candidate_unique_subjects"]],
            "zero_dsc_et": [z["baseline_et"], z["candidate_et"]],
            "lesion": les,
            "bootstrap_prob_positive_tau1": t1["bootstrap"]["prob_positive"],
            "denominator_changes_tau1": t1["official"]["denominator_changes"],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--eval", nargs="+", required=True,
                    help="recipe=path/to/eval.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec = json.load(open(a.spec))
    report = {"schema": "gat26.g82.screen_decision.v1",
              "tie_threshold": TIE, "preference_order": PREFERENCE, "recipes": {}}

    for item in a.eval:
        recipe, path = item.split("=", 1)
        assert recipe in spec["candidates"], f"{recipe} is not a preregistered candidate"
        if not os.path.exists(path):
            report["recipes"][recipe] = {"eligible": False,
                                         "checks": {"result_present": False},
                                         "score": float("-inf"),
                                         "evidence": {"note": "no evaluation produced"}}
            continue
        report["recipes"][recipe] = eligibility(json.load(open(path)), spec)

    eligible = {r: v for r, v in report["recipes"].items() if v["eligible"]}
    if not eligible:
        report["selected"] = None
        report["terminal_status"] = "G82_RETAIN_C0_NO_SCREEN_ADVANCEMENT"
    else:
        best = max(v["score"] for v in eligible.values())
        # the baseline wins ties: a non-positive best score is not an advancement
        tied = [r for r, v in eligible.items() if best - v["score"] <= TIE]
        if best <= TIE:
            report["selected"] = None
            report["terminal_status"] = "G82_RETAIN_C0_NO_SCREEN_ADVANCEMENT"
            report["note"] = ("the best eligible gain is within the tie threshold of "
                              "zero, and the baseline wins all ties")
        else:
            tied.sort(key=lambda r: PREFERENCE.index(r))
            report["selected"] = tied[0]
            report["tied_with"] = tied
            report["terminal_status"] = "SCREEN_ADVANCED"

    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, a.out)

    for r, v in sorted(report["recipes"].items()):
        e = v["evidence"]
        print(f"{r:4s} eligible={str(v['eligible']):5s} "
              f"score={v['score']:+.6f} "
              f"dU1={e.get('dU_common_tau1', float('nan')):+.6f} "
              f"dU05={e.get('dU_common_tau05', float('nan')):+.6f}")
        if not v["eligible"]:
            print("      failed:", [k for k, ok in v["checks"].items() if not ok])
    print("SELECTED:", report["selected"], "|", report["terminal_status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
