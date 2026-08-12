#!/usr/bin/env python3
"""GAT-26 selection policy v2 (Stage 4A-R / G4.5-R) — executable, testable engine.

Supersedes v1 (which only declared aggregate award ranks). v2 adds the executable,
pure functions the report depends on: subject-record validation (fail-closed),
component aggregation, paired subject-level bootstrap, noninferiority evaluation, and
the fold-0 / two-fold M-vs-L decision logic.

Frozen (pre-result, unbiased) contract — unchanged from v1 where noted:
  * evaluator = BraTS-evaluation 0.0.8, GoAT config (ET=[3], TC=[1,3], WT=[1,2,3]).
  * six equally weighted components ET/TC/WT x DSC/HD95; DSC higher-is-better,
    HD95 lower-is-better; NSD and connected-component metrics OUTSIDE the award utility.
  * tie -> select M; G5 inference threshold 0.5, no TTA / post-processing / presence gate.

Fail-closed input contract (Part B.4/B.5):
  * paired M/L records for exactly the same held-out subjects; all six components present.
  * duplicate / missing / extra subjects -> HardFailure.
  * evaluator error records, missing predictions/outputs, malformed metrics -> HardFailure.
  * a missing output is NEVER disguised as an HD95 penalty.
  * only a VALID evaluated region under the official zero-TP infinite-HD95 condition is
    parsed to 373; errors/missing/malformed stay hard failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SEED = 21072026
BOOTSTRAP_RESAMPLES = 10000
HD95_PENALTY = 373.0
COMPONENTS = [("ET", "DSC"), ("TC", "DSC"), ("WT", "DSC"),
              ("ET", "HD95"), ("TC", "HD95"), ("WT", "HD95")]
DIRECTION = {"DSC": "higher_is_better", "HD95": "lower_is_better"}
MEANINGFUL_RANK_GAIN = 1.0 / 6.0  # L must beat M mean award rank by >= 1/6 (one component of six)

# Noninferiority margins (frozen now, pre-result). Each: max tolerated degradation of L vs M.
NONINF_MARGINS = {
    "regional_dsc_abs": 0.005,          # L DSC >= M DSC - 0.005 per region
    "regional_hd95_mm": 0.75,           # L HD95 <= M HD95 + 0.75 mm per region
    "smallest_volume_stratum_dsc_abs": 0.01,
    "dsc_5th_percentile_abs": 0.01,
    "hd95_95th_percentile_mm": 1.5,
    "empty_reference_false_positive_rate_abs": 0.02,
    "missed_region_rate_abs": 0.02,
    "runtime_regression_frac": 0.25,
    "cost_regression_frac": 0.25,
}
NONINF_RATIONALE = {
    "regional_dsc_abs": "0.005 Dice ~ sub-noise floor at 1,351 cases; larger L drops are real regressions. Inequality: L_DSC >= M_DSC - 0.005.",
    "regional_hd95_mm": "0.75 mm < one isotropic voxel (1 mm); tighter would flag resampling jitter. Inequality: L_HD95 <= M_HD95 + 0.75.",
    "smallest_volume_stratum_dsc_abs": "0.01 on the smallest-lesion stratum where Dice is noisiest. Inequality: L >= M - 0.01.",
    "dsc_5th_percentile_abs": "0.01 guards the hard-case tail (5th pct). Inequality: L_p05 >= M_p05 - 0.01.",
    "hd95_95th_percentile_mm": "1.5 mm guards the worst-boundary tail (95th pct). Inequality: L_p95 <= M_p95 + 1.5.",
    "empty_reference_false_positive_rate_abs": "0.02 caps extra false positives on empty-reference regions. Inequality: L_fp <= M_fp + 0.02.",
    "missed_region_rate_abs": "0.02 caps extra fully-missed present regions. Inequality: L_miss <= M_miss + 0.02.",
    "runtime_regression_frac": "L may not cost >25% more wall time than M without benefit. Inequality: L_rt <= M_rt * 1.25.",
    "cost_regression_frac": "L may not cost >25% more $ than M without benefit. Inequality: L_cost <= M_cost * 1.25.",
}


class HardFailure(Exception):
    pass


# ----------------------------- fail-closed parsing -----------------------------
def parse_metric(value, metric, evaluated_ok):
    """Return a finite float, or raise HardFailure. Only a valid evaluated HD95 with the
    official zero-TP infinite condition becomes HD95_PENALTY. Errors/missing never do."""
    if not evaluated_ok:
        raise HardFailure(f"evaluator error / missing output for {metric} (not a 373 penalty)")
    if value is None:
        raise HardFailure(f"missing {metric} value")
    if isinstance(value, str):
        if metric == "HD95" and value.strip().lower() in ("inf", "infinity", "+inf"):
            return HD95_PENALTY
        raise HardFailure(f"malformed {metric} value: {value!r}")
    v = float(value)
    if math.isnan(v):
        raise HardFailure(f"NaN {metric}")
    if math.isinf(v):
        if metric == "HD95" and v > 0:
            return HD95_PENALTY               # valid zero-TP infinite HD95
        raise HardFailure(f"non-finite {metric}")
    return v


def validate_subject_records(recs_M, recs_L):
    """recs_*: list of {subject, evaluated_ok, metrics:{(region,metric):value}}.
    Returns (subjects, mM, mL) with parsed finite metrics, or raises HardFailure."""
    def index(recs, tag):
        subs = [r["subject"] for r in recs]
        if len(subs) != len(set(subs)):
            raise HardFailure(f"duplicate subject in {tag}")
        out = {}
        for r in recs:
            m = {}
            for comp in COMPONENTS:
                if comp not in r.get("metrics", {}):
                    raise HardFailure(f"missing component {comp} for {tag} subject")
                m[comp] = parse_metric(r["metrics"][comp], comp[1], r.get("evaluated_ok", False))
            out[r["subject"]] = m
        return out
    mM = index(recs_M, "M"); mL = index(recs_L, "L")
    if set(mM) != set(mL):
        raise HardFailure("M and L held-out subject sets differ (missing/extra)")
    subjects = sorted(mM)
    return subjects, mM, mL


# ----------------------------- aggregation & ranks -----------------------------
def aggregate_components(subjects, metrics):
    return {c: sum(metrics[s][c] for s in subjects) / len(subjects) for c in COMPONENTS}


def award_ranks(agg_by_model):
    """agg_by_model: {model: {component: value}} -> {model: mean fractional rank} (lower better)."""
    models = list(agg_by_model)
    ranks = {m: [] for m in models}
    for comp in COMPONENTS:
        direction = DIRECTION[comp[1]]
        key = (lambda m: agg_by_model[m][comp]) if direction == "lower_is_better" \
            else (lambda m: -agg_by_model[m][comp])
        order = sorted(models, key=key)
        vals = [key(m) for m in order]
        i, pos = 0, 1
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[j + 1] == vals[i]:
                j += 1
            avg = sum(range(pos, pos + (j - i + 1))) / (j - i + 1)
            for k in range(i, j + 1):
                ranks[order[k]].append(avg)
            pos += (j - i + 1); i = j + 1
    return {m: sum(r) / len(r) for m, r in ranks.items()}


# ----------------------------- paired bootstrap -----------------------------
def paired_bootstrap_compare(subjects, mM, mL, seed=SEED, n=BOOTSTRAP_RESAMPLES):
    """Resample paired subjects with replacement; per resample aggregate the six
    components and recompute M/L fractional ranks; delta = rank_L - rank_M (negative
    favors L). Deterministic for fixed input+seed. Returns point estimate + 95% CI."""
    import numpy as np
    idx = np.arange(len(subjects))
    # point estimate on the full sample
    aggM = aggregate_components(subjects, mM); aggL = aggregate_components(subjects, mL)
    r = award_ranks({"M": aggM, "L": aggL})
    point = r["L"] - r["M"]
    rng = np.random.default_rng(seed)
    deltas = np.empty(n, dtype=np.float64)
    S = len(subjects)
    for b in range(n):
        take = rng.integers(0, S, size=S)
        subs = [subjects[i] for i in take]
        aM = aggregate_components(subs, mM); aL = aggregate_components(subs, mL)
        rr = award_ranks({"M": aM, "L": aL})
        deltas[b] = rr["L"] - rr["M"]
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_point": float(point), "delta_ci_low": float(lo), "delta_ci_high": float(hi),
            "n_resamples": n, "seed": seed, "favors_L_point": point < 0,
            "ci_excludes_tie_favoring_L": float(hi) < 0.0}


# ----------------------------- noninferiority -----------------------------
def evaluate_noninferiority(aggM, aggL, extra=None, margins=None):
    """Return {gate: pass, ...} and all_pass. `extra` optionally provides the stratum/
    percentile/rate/runtime inputs; gates without inputs are marked 'not_provided' and do
    NOT count as passing (fail-closed for expansion decisions)."""
    m = margins or NONINF_MARGINS
    extra = extra or {}
    res = {}
    # regional DSC / HD95 from aggregates
    for region in ("ET", "TC", "WT"):
        res[f"dsc_{region}"] = aggL[(region, "DSC")] >= aggM[(region, "DSC")] - m["regional_dsc_abs"]
        res[f"hd95_{region}"] = aggL[(region, "HD95")] <= aggM[(region, "HD95")] + m["regional_hd95_mm"]

    def opt_ge(key, mk):  # L >= M - margin (higher better)
        if key + "_M" in extra and key + "_L" in extra:
            return extra[key + "_L"] >= extra[key + "_M"] - m[mk]
        return "not_provided"

    def opt_le(key, mk):  # L <= M + margin (lower better)
        if key + "_M" in extra and key + "_L" in extra:
            return extra[key + "_L"] <= extra[key + "_M"] + m[mk]
        return "not_provided"

    def opt_frac(key, mk):  # L <= M*(1+frac)
        if key + "_M" in extra and key + "_L" in extra:
            return extra[key + "_L"] <= extra[key + "_M"] * (1 + m[mk])
        return "not_provided"

    res["smallest_volume_stratum_dsc"] = opt_ge("smallest_volume_dsc", "smallest_volume_stratum_dsc_abs")
    res["dsc_5th_percentile"] = opt_ge("dsc_p05", "dsc_5th_percentile_abs")
    res["hd95_95th_percentile"] = opt_le("hd95_p95", "hd95_95th_percentile_mm")
    res["empty_reference_fp_rate"] = opt_le("empty_ref_fp", "empty_reference_false_positive_rate_abs")
    res["missed_region_rate"] = opt_le("missed_region", "missed_region_rate_abs")
    res["runtime"] = opt_frac("runtime", "runtime_regression_frac")
    res["cost"] = opt_frac("cost", "cost_regression_frac")

    provided = {k: v for k, v in res.items() if v is not True and v is not False}
    all_provided_pass = all(v is True for v in res.values() if isinstance(v, bool))
    # expansion requires every gate PROVIDED and passing
    all_gates_pass = all(v is True for v in res.values())
    res["_all_provided_pass"] = all_provided_pass
    res["_all_gates_provided_and_pass"] = all_gates_pass
    res["_gates_not_provided"] = [k for k, v in res.items() if v == "not_provided"]
    return res


# ----------------------------- decisions -----------------------------
def _fold_benefit(fold, require_all_gates):
    subjects, mM, mL = validate_subject_records(fold["M"], fold["L"])
    aggM = aggregate_components(subjects, mM); aggL = aggregate_components(subjects, mL)
    ranks = award_ranks({"M": aggM, "L": aggL})
    rank_gain = ranks["M"] - ranks["L"]                      # positive => L better
    boot = paired_bootstrap_compare(subjects, mM, mL,
                                    seed=fold.get("seed", SEED),
                                    n=fold.get("n", BOOTSTRAP_RESAMPLES))
    noninf = evaluate_noninferiority(aggM, aggL, extra=fold.get("extra"))
    gates_ok = noninf["_all_gates_provided_and_pass"] if require_all_gates else noninf["_all_provided_pass"]
    meaningful = (rank_gain >= MEANINGFUL_RANK_GAIN
                  and boot["ci_excludes_tie_favoring_L"]
                  and gates_ok)
    return {"rank_M": ranks["M"], "rank_L": ranks["L"], "rank_gain_L_over_M": rank_gain,
            "bootstrap": boot, "noninferiority": noninf, "meaningful_L_benefit": bool(meaningful)}


def decide_after_fold0(fold0):
    """Fold-0 alone can only trigger fold-1 confirmation; it can never expand L on its own.
    FAIL-CLOSED: every frozen noninferiority gate (regional DSC/HD95 AND the auxiliary
    smallest-volume/tail/false-positive/missed-region/runtime/cost gates) must be SUPPLIED
    and passing. A missing auxiliary input can never trigger L confirmation -> select M."""
    b = _fold_benefit(fold0, require_all_gates=True)
    missing = b["noninferiority"].get("_gates_not_provided", [])
    if missing:
        return {"decision": "select_M", "fold0": b, "fail_closed_missing_gates": missing}
    if b["meaningful_L_benefit"]:
        decision = "confirm_L_on_fold1"        # spend on L fold-1 only
    else:
        decision = "select_M"                  # tie / no benefit / noninferiority fail -> M, do not pay for L fold1
    return {"decision": decision, "fold0": b}


def decide_after_two_folds(fold0, fold1):
    b0 = _fold_benefit(fold0, require_all_gates=True)
    b1 = _fold_benefit(fold1, require_all_gates=True)
    same_direction = b0["meaningful_L_benefit"] and b1["meaningful_L_benefit"]
    if same_direction:
        decision = "L_may_expand_subject_to_owner_budget_regate"
    else:
        decision = "select_M"
    return {"decision": decision, "fold0": b0, "fold1": b1, "confirmed_same_direction": same_direction}


# ----------------------------- policy document -----------------------------
def build_policy():
    return {
        "policy_id": "gat26_g45_selection_policy_v2",
        "supersedes": "gat26_g45_selection_policy_v1",
        "frozen": True, "frozen_before_any_fold_result": True,
        "result_dependent_editing_forbidden": True, "change_requires_new_owner_gate": True,
        "evaluator": {"package": "BraTS-evaluation", "version": "0.0.8", "config": "GoAT",
                      "regions": {"ET": [3], "TC": [1, 3], "WT": [1, 2, 3]},
                      "hd95_zero_tp_penalty_mm": HD95_PENALTY,
                      "fail_closed": "evaluator errors / missing outputs / malformed metrics are HARD failures, never a 373 penalty"},
        "primary_award_utility": {
            "components": [{"region": r, "metric": mt, "direction": DIRECTION[mt]} for r, mt in COMPONENTS],
            "weighting": "equal", "aggregation": "mean_fractional_award_rank_over_six_components",
            "excluded_from_award": ["NSD", "connected_component_metrics", "sensitivity", "specificity", "precision"]},
        "bootstrap": {"method": "paired_subject_level_within_fold_with_replacement",
                      "seed": SEED, "resamples": BOOTSTRAP_RESAMPLES,
                      "delta_definition": "delta = rank_L - rank_M (negative favors L)",
                      "report": ["delta_point", "percentile_95_ci"], "deterministic": True},
        "meaningful_benefit_rule": {
            "min_mean_award_rank_gain_L_over_M": MEANINGFUL_RANK_GAIN,
            "bootstrap_ci_upper_bound_delta_lt_0": True,
            "must_pass_every_frozen_noninferiority_gate": True,
            "fold0_alone_only_triggers_fold1_confirmation": True,
            "L_expands_only_if_fold1_confirms_same_direction": True},
        "m_decision_rule": {"select_M_if": ["L fails meaningful benefit on fold 0",
                                            "L violates any noninferiority gate", "tie"],
                            "M_is_baseline_control_and_tie_winner": True,
                            "no_result_dependent_policy_change": True},
        "noninferiority_margins": NONINF_MARGINS, "noninferiority_rationale": NONINF_RATIONALE,
        "tie_rule": {"rule": "select_M_the_cheaper_simpler_plan"},
        "frozen_g5_inference": {"threshold": 0.5, "reconstruction": "hierarchy_safe_projection_WT_TC_ET",
                                "tta": False, "connected_component_filtering": False, "presence_gate": False,
                                "learned_threshold": False, "learned_ensemble_weight": False,
                                "primary_checkpoint": "checkpoint_final"},
        "public_validation_use": {"purpose": "integration_confirmation_only",
                                  "never_used_for": ["architecture_choice", "threshold_choice",
                                                     "post_processing_choice", "fold_selection", "checkpoint_choice"]},
    }


def cmd_emit(args):
    Path(args.out).write_text(json.dumps(build_policy(), indent=2) + "\n", encoding="utf-8")
    h = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
    print(json.dumps({"written": args.out, "policy_id": "gat26_g45_selection_policy_v2", "sha256": h}))
    return 0


def cmd_validate(args):
    p = json.loads(Path(args.file).read_text(encoding="utf-8"))
    checks = {
        "v2": p["policy_id"] == "gat26_g45_selection_policy_v2",
        "six_components": len(p["primary_award_utility"]["components"]) == 6,
        "dsc_up_hd95_down": all((c["direction"] == "higher_is_better") == (c["metric"] == "DSC")
                                for c in p["primary_award_utility"]["components"]),
        "nsd_cc_excluded": {"NSD", "connected_component_metrics"} <= set(p["primary_award_utility"]["excluded_from_award"]),
        "bootstrap_seed": p["bootstrap"]["seed"] == SEED and p["bootstrap"]["resamples"] == BOOTSTRAP_RESAMPLES,
        "tie_to_M": p["tie_rule"]["rule"] == "select_M_the_cheaper_simpler_plan",
        "g5_no_tta_no_cc": p["frozen_g5_inference"]["tta"] is False and p["frozen_g5_inference"]["connected_component_filtering"] is False,
        "fail_closed_declared": "HARD" in p["evaluator"]["fail_closed"],
        "margins_have_rationale": set(p["noninferiority_margins"]) == set(p["noninferiority_rationale"]),
    }
    out = {"valid": all(checks.values()), "checks": checks}
    print(json.dumps(out))
    return 0 if out["valid"] else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("emit"); p.add_argument("--out", required=True); p.set_defaults(func=cmd_emit)
    p = sub.add_parser("validate"); p.add_argument("--file", required=True); p.set_defaults(func=cmd_validate)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    import sys
    sys.exit(main())
