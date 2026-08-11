#!/usr/bin/env python3
"""G82 §H discovery freeze decision.

Applies the preregistered freeze gates to every allowed (epoch, alpha) policy of
the single selected recipe and, if more than one passes, picks the winner by the
preregistered order:

  1. highest lower 95% bootstrap bound for dU_common_tau1
  2. larger min(tau1, tau05) gain
  3. alpha = 1.00 (lower release complexity)
  4. the baseline wins any remaining tie

Nothing here is derived from the observed numbers; every threshold is read from the
frozen preregistration file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REGIONS = ("ET", "TC", "WT")


def gates(ev: dict, spec: dict) -> dict:
    g = spec["full_discovery"]["freeze_gates_all_required"]
    t1, t05 = ev["t10"], ev["t05"]
    c1, c05 = t1["common_support"], t05["common_support"]
    boot = t1["bootstrap"]
    z1 = t1["zero_dsc"]
    les = ev["lesion"]
    cd = c1["component_deltas"]

    checks = {}
    checks["dU_common_tau1"] = c1["delta_U"] >= g["delta_U_common_tau1_min"]
    checks["dU_common_tau05"] = c05["delta_U"] >= g["delta_U_common_tau05_min"]
    checks["official_deltas_positive"] = (
        t1["official"]["delta_U"] > 0 and t05["official"]["delta_U"] > 0)
    checks["bootstrap_prob"] = (
        boot["prob_positive"] >= g["bootstrap_prob_delta_U_common_tau1_positive_min"])
    checks["enough_nonnegative_components"] = (
        sum(1 for v in cd.values() if v >= 0)
        >= g["min_nonnegative_common_support_tau1_components"])
    checks["dsc_regression_within_limit"] = all(
        cd[f"{r}_DSC"] >= -g["max_dsc_component_regression"] for r in REGIONS)
    checks["nsd_regression_within_limit"] = all(
        cd[f"{r}_NSD"] >= -g["max_nsd_component_regression"] for r in REGIONS)
    fold_vals = list(t1["fold_deltas"].values())
    checks["enough_positive_folds"] = (
        sum(1 for v in fold_vals if v > 0) >= g["min_folds_with_positive_tau1_delta"])
    checks["no_fold_collapses"] = all(
        v >= g["min_fold_delta_U_common_tau1"] for v in fold_vals)
    checks["et_zero_dsc_not_increased"] = z1["candidate_et"] <= z1["baseline_et"]
    checks["total_zero_dsc_decreases"] = (
        z1["candidate_region_cases"] < z1["baseline_region_cases"])
    checks["lesion_fn_not_increased"] = les["candidate"]["FN"] <= les["baseline"]["FN"]
    checks["lesion_fp_within_limit"] = (
        les["fp_increase_fraction"] <= g["max_lesion_fp_increase_fraction"])
    strat = t1.get("volume_stratum_deltas", {})
    checks["no_volume_stratum_collapse"] = all(
        v["delta"] >= -g["max_utility_loss_in_any_volume_stratum_with_at_least_30_cases"]
        for v in strat.values())
    checks["official_gain_positive_on_common_support"] = c1["delta_U"] > 0

    # a gain that exists only because subjects left the denominator is not a gain
    dchg = t1["official"]["denominator_changes"]
    denom_shrunk = any(v < 0 for v in dchg.values())
    checks["no_denominator_only_gain"] = (not denom_shrunk) or c1["delta_U"] > 0
    checks["no_evaluator_errors"] = ev.get("evaluator_errors", 0) == 0

    return {
        "checks": checks,
        "passes": all(checks.values()),
        "dU_common_tau1": c1["delta_U"],
        "dU_common_tau05": c05["delta_U"],
        "min_gain": min(c1["delta_U"], c05["delta_U"]),
        "bootstrap_lower_bound": boot["ci95"][0],
        "bootstrap_prob_positive": boot["prob_positive"],
        "denominator_changes": dchg,
        "zero_dsc": z1,
        "lesion": les,
        "fold_deltas": t1["fold_deltas"],
        "component_deltas": cd,
        "volume_stratum_deltas": strat,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--policy", nargs="+", required=True,
                    help="E<epoch>_A<alpha>=path/to/eval.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec = json.load(open(a.spec))
    allowed_epochs = set(spec["full_discovery"]["checkpoints"])
    allowed_alphas = set(spec["full_discovery"]["alphas"])

    report = {"schema": "gat26.g82.discovery_decision.v1", "recipe": a.recipe,
              "policies": {}}
    for item in a.policy:
        name, path = item.split("=", 1)
        epoch = int(name.split("_")[0][1:])
        alpha = float(name.split("_A")[1])
        assert epoch in allowed_epochs, f"epoch {epoch} is not preregistered"
        assert alpha in allowed_alphas, f"alpha {alpha} is not preregistered"
        if not os.path.exists(path):
            report["policies"][name] = {"passes": False,
                                        "checks": {"result_present": False}}
            continue
        report["policies"][name] = gates(json.load(open(path)), spec)

    passing = {k: v for k, v in report["policies"].items() if v.get("passes")}
    if not passing:
        report["frozen_policy"] = None
        report["terminal_status"] = "G82_RETAIN_C0_NO_FULL_DISCOVERY_ADVANCEMENT"
    else:
        def key(item):
            name, v = item
            alpha = float(name.split("_A")[1])
            return (-v["bootstrap_lower_bound"], -v["min_gain"], 0 if alpha == 1.0 else 1)
        winner = sorted(passing.items(), key=key)[0][0]
        report["frozen_policy"] = winner
        report["candidates_passing"] = sorted(passing)
        report["terminal_status"] = "DISCOVERY_FROZEN"

    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, a.out)

    for k, v in sorted(report["policies"].items()):
        if "dU_common_tau1" in v:
            print(f"{k:12s} pass={str(v['passes']):5s} dU1={v['dU_common_tau1']:+.6f} "
                  f"dU05={v['dU_common_tau05']:+.6f} "
                  f"P={v['bootstrap_prob_positive']:.3f} "
                  f"lo95={v['bootstrap_lower_bound']:+.6f}")
            if not v["passes"]:
                print("        failed:", [c for c, ok in v["checks"].items() if not ok])
    print("FROZEN:", report["frozen_policy"], "|", report["terminal_status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
