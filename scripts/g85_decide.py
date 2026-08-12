#!/usr/bin/env python3
"""G85 gate arbitration: independent confirmation (§E) and the supportive pooled
all-five-fold analysis.

Every threshold comes from the frozen preregistration. A failed gate is reported as
failed, never reinterpreted. The corrected lesion analysis contributes only
noninferiority safety checks; its diagnostic precision/recall/F1 can never override
an official or safety gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REGIONS = ("ET", "TC", "WT")


def _checks(ev: dict, g: dict, phase: str) -> dict:
    t1, t05 = ev["t10"], ev["t05"]
    c1, c05 = t1["common_support"], t05["common_support"]
    boot, z1 = t1["bootstrap"], t1["zero_dsc"]
    cd1, cd05 = c1["component_deltas"], c05["component_deltas"]
    ni = ev["lesion_noninferiority"]
    ch = {}

    ch["dU_common_tau1"] = c1["delta_U"] >= g["delta_U_common_tau1_min"]
    ch["dU_common_tau05"] = c05["delta_U"] >= g["delta_U_common_tau05_min"]
    ch["bootstrap_probability"] = (
        boot["prob_positive"] >= g[("bootstrap_prob_tau1_positive_min" if phase == "confirmation"
                                    else "bootstrap_prob_positive_min")])
    ch["bootstrap_lower_bound_above_zero"] = boot["ci95"][0] > 0
    need = g["min_nonnegative_components_each_tolerance"]
    ch["components_nonnegative_tau1"] = sum(1 for v in cd1.values() if v >= 0) >= need
    ch["components_nonnegative_tau05"] = sum(1 for v in cd05.values() if v >= 0) >= need

    if phase == "confirmation":
        ch["official_deltas_positive"] = (t1["official"]["delta_U"] > 0
                                          and t05["official"]["delta_U"] > 0)
        ch["dsc_within_limit"] = all(cd1[f"{r}_DSC"] >= g["min_dsc_component_delta"]
                                     for r in REGIONS)
        ch["nsd_within_limit"] = all(cd1[f"{r}_NSD"] >= g["min_nsd_component_delta"]
                                     for r in REGIONS)
        fd = t1["fold_deltas"]
        ch["fold3_nonnegative"] = fd.get("3", -1.0) >= 0
        ch["fold4_nonnegative"] = fd.get("4", -1.0) >= 0

    ch["et_zero_dsc_not_increased"] = z1["candidate_et"] <= z1["baseline_et"]
    ch["total_zero_dsc_not_increased"] = (z1["candidate_region_cases"]
                                         <= z1["baseline_region_cases"])
    ch["unique_subject_zero_dsc_not_increased"] = (z1["candidate_unique_subjects"]
                                                   <= z1["baseline_unique_subjects"])

    # corrected reference-component noninferiority (§D). n_ref invariance is a
    # correctness assertion, not a scientific gate, but a violation must still stop us.
    ch["n_ref_total_invariant"] = ni["checks"]["n_ref_total_invariant"]
    ch["miss_rate_point_within_margin"] = (
        ni["bootstrap"]["point_delta"] <= g["point_miss_rate_increase_max"]
        if "point_miss_rate_increase_max" in g
        else ni["checks"]["point_miss_rate_within_margin"])
    ch["miss_rate_upper_bound_within_margin"] = (
        ni["bootstrap"]["upper_95_one_sided"] <= g["one_sided_miss_rate_upper_bound_max"]
        if "one_sided_miss_rate_upper_bound_max" in g
        else ni["checks"]["upper_bound_within_margin"])
    ch["no_region_miss_rate_regression"] = ni["checks"]["no_region_miss_rate_regression"]
    ch["fp_pred_within_limit"] = ni["checks"]["fp_within_limit"]

    dchg = t1["official"]["denominator_changes"]
    ch["no_denominator_only_gain"] = all(v == 0 for v in dchg.values()) or c1["delta_U"] > 0
    ch["exact_membership"] = ev.get("exact_membership", False)
    ch["zero_evaluator_errors"] = ev.get("evaluator_errors", 0) == 0
    ch["independent_recompute_agrees"] = ev.get("independent_recompute_agrees", False)
    return ch


def evidence(ev: dict) -> dict:
    t1, t05 = ev["t10"], ev["t05"]
    ni = ev["lesion_noninferiority"]
    return {
        "n_cases": ev["n_cases"], "n_expected": ev.get("n_expected"),
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
        "zero_dsc": t1["zero_dsc"],
        "lesion_noninferiority": {
            "n_ref_total": ni["baseline"]["n_ref_total"],
            "miss_rate_baseline": ni["baseline"]["miss_rate"],
            "miss_rate_candidate": ni["candidate"]["miss_rate"],
            "point_delta": ni["bootstrap"]["point_delta"],
            "upper_95_one_sided": ni["bootstrap"]["upper_95_one_sided"],
            "ci95_two_sided": ni["bootstrap"]["ci95_two_sided"],
            "fn_ref_total": [ni["baseline"]["fn_ref_total"], ni["candidate"]["fn_ref_total"]],
            "fp_pred_total": [ni["baseline"]["fp_pred_total"], ni["candidate"]["fp_pred_total"]],
            "region_deltas": ni["region_deltas"], "margins": ni["margins"],
            "checks": ni["checks"]},
        "lesion_diagnostic_only": ni["diagnostic"],
        "independent_recompute_max_abs_difference":
            ev.get("independent_recompute_max_abs_difference"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--phase", choices=("confirmation", "pooled"), required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec = json.load(open(a.spec, encoding="utf-8"))
    ev = json.load(open(a.eval, encoding="utf-8"))
    if a.phase == "confirmation":
        g = dict(spec["confirmation"]["primary_gates_all_required"])
        g.update(spec["corrected_lesion_analysis"]["margins_frozen_before_confirmation"])
        g["point_miss_rate_increase_max"] = g["point_miss_rate_increase_max"]
        g["one_sided_miss_rate_upper_bound_max"] = g["one_sided_95_upper_bound_max"]
        fail = spec["confirmation"]["on_failure"]
    else:
        g = dict(spec["pooled_all_five_folds"]["gates_all_required"])
        fail = spec["confirmation"]["on_failure"]

    ch = _checks(ev, g, a.phase)
    report = {"schema": "gat26.g85.decision.v1", "phase": a.phase,
              "checks": ch, "evidence": evidence(ev),
              "passes": all(ch.values()),
              "failed_gates": sorted(k for k, v in ch.items() if not v)}
    report["terminal_status"] = (("CONFIRMATION_PASSED" if a.phase == "confirmation"
                                 else "POOLED_PASSED") if report["passes"] else fail)

    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, a.out)

    e = report["evidence"]
    ni = e["lesion_noninferiority"]
    print(f"{a.phase}: dU1={e['dU_common_tau1']:+.6f} dU05={e['dU_common_tau05']:+.6f} "
          f"P={e['bootstrap']['prob_positive']:.4f} "
          f"CI=[{e['bootstrap']['ci95'][0]:+.6f},{e['bootstrap']['ci95'][1]:+.6f}]")
    print(f"  miss-rate {ni['miss_rate_baseline']:.6f} -> {ni['miss_rate_candidate']:.6f} "
          f"delta={ni['point_delta']:+.6f} upper95={ni['upper_95_one_sided']:+.6f} "
          f"(margins {ni['margins']['point_max']}/{ni['margins']['upper_bound_max']})")
    if report["failed_gates"]:
        print("FAILED GATES:", report["failed_gates"])
    print("STATUS:", report["terminal_status"])
    return 0 if report["passes"] else 3


if __name__ == "__main__":
    sys.exit(main())
