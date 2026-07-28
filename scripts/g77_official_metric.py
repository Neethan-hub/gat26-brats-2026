#!/usr/bin/env python3
"""GAT-26 G7.7 — official BraTS-2026 Task-3 metric utility (DSC + NSD), executable and testable.

The official Task-3 leaderboard ranks on SIX components, all higher-is-better:

    global_dsc_{et,tc,wt}   and   global_nsd_{et,tc,wt}

(source: BraTS-evaluation 0.0.8 @ 88e3e39c `config_GoAT.yaml` `global_metrics: [DSC, NSD, HD95]`
and `metrics_parser.py`, whose emitted field names are exactly the Task-3 leaderboard columns).

HD95 is deliberately **not** part of this utility. It is retained elsewhere as a secondary
diagnostic only and must never select the release policy.

Aggregation reproduces the official parser exactly: an arithmetic mean over per-subject values with
**skipna semantics** (`pandas.DataFrame.mean`), so NaN entries are dropped from both numerator and
denominator and per-component denominators may legitimately differ.

This module is pure and importable so the decision logic is unit-testable without an evaluator, a
GPU, or protected data. It deliberately does NOT modify, import from, or reinterpret the historical
DSC+HD95 policy in `scripts/g45_selection_policy.py`.
"""
from __future__ import annotations

SEED = 21072026
BOOTSTRAP_RESAMPLES = 10000

REGIONS = ("ET", "TC", "WT")
# The six official ranking components. Order is fixed; all are higher-is-better.
OFFICIAL_COMPONENTS = [(r, m) for m in ("DSC", "NSD") for r in REGIONS]
OFFICIAL_FIELDS = {(r, m): f"global_{m.lower()}_{r.lower()}" for (r, m) in OFFICIAL_COMPONENTS}
DIRECTION = {"DSC": "higher_is_better", "NSD": "higher_is_better"}

# A candidate must beat the baseline by at least one full component of six.
MEANINGFUL_RANK_GAIN = 1.0 / 6.0


class HardFailure(Exception):
    """Fail-closed: any malformed record, subject-set mismatch, or illegal value."""


def _finite(value):
    """Return a float, None for a legitimately absent (NaN) component, else raise."""
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise HardFailure(f"non-numeric metric value {value!r}") from exc
    if x != x:                      # NaN -> legitimately absent, skipna drops it
        return None
    if x in (float("inf"), float("-inf")):
        raise HardFailure("infinite value is not legal for DSC/NSD")
    if not (0.0 <= x <= 1.0):
        raise HardFailure(f"DSC/NSD out of range: {x}")
    return x


def validate_subject_records(recs_base, recs_cand):
    """recs_*: [{subject, metrics:{(region,metric): value}}]. Returns (subjects, base, cand).

    Fail-closed on: subject-set mismatch, duplicates, missing components, illegal values.
    """
    def index(recs, tag):
        out = {}
        for r in recs:
            sid = r["subject"]
            if sid in out:
                raise HardFailure(f"duplicate subject in {tag}")
            m = {}
            for comp in OFFICIAL_COMPONENTS:
                if comp not in r["metrics"]:
                    raise HardFailure(f"missing component {comp} for a subject in {tag}")
                m[comp] = _finite(r["metrics"][comp])
            out[sid] = m
        return out

    b = index(recs_base, "baseline")
    c = index(recs_cand, "candidate")
    if set(b) != set(c):
        raise HardFailure(
            f"subject sets differ: baseline={len(b)} candidate={len(c)} "
            f"missing={len(set(b) - set(c))} extra={len(set(c) - set(b))}")
    return sorted(b), b, c


def aggregate_components(subjects, metrics):
    """Official aggregation: arithmetic mean per component with skipna (NaN dropped)."""
    agg = {}
    for comp in OFFICIAL_COMPONENTS:
        vals = [metrics[s][comp] for s in subjects if metrics[s][comp] is not None]
        agg[comp] = (sum(vals) / len(vals)) if vals else None
    return agg


def component_denominators(subjects, metrics):
    """Per-component count of non-NaN subjects actually entering the mean."""
    return {comp: sum(1 for s in subjects if metrics[s][comp] is not None)
            for comp in OFFICIAL_COMPONENTS}


def award_ranks(agg_by_model):
    """{model: {component: value}} -> {model: mean fractional rank} (LOWER rank is better).

    All six official components are higher-is-better, so the best value gets rank 0.
    """
    models = list(agg_by_model)
    totals = {m: 0.0 for m in models}
    for comp in OFFICIAL_COMPONENTS:
        vals = {m: agg_by_model[m][comp] for m in models}
        if any(v is None for v in vals.values()):
            raise HardFailure(f"cannot rank component {comp}: an aggregate is undefined")
        # higher is better -> descending; ties share the average rank
        order = sorted(models, key=lambda m: -vals[m])
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            shared = sum(range(i, j + 1)) / (j - i + 1)
            for k in range(i, j + 1):
                totals[order[k]] += shared
            i = j + 1
    return {m: totals[m] / len(OFFICIAL_COMPONENTS) for m in models}


def paired_bootstrap_compare(subjects, base, cand, seed=SEED, n=BOOTSTRAP_RESAMPLES):
    """Paired subject-level bootstrap on the six official components.

    delta = rank_candidate - rank_baseline, so POSITIVE favours the baseline and a candidate wins
    only when the whole interval is below zero. Deterministic for a fixed input and seed.
    """
    import numpy as np
    aggB = aggregate_components(subjects, base)
    aggC = aggregate_components(subjects, cand)
    r = award_ranks({"base": aggB, "cand": aggC})
    point = r["cand"] - r["base"]
    rng = np.random.default_rng(seed)
    S = len(subjects)
    deltas = np.empty(n, dtype=np.float64)
    for i in range(n):
        take = rng.integers(0, S, size=S)
        subs = [subjects[t] for t in take]
        rr = award_ranks({"base": aggregate_components(subs, base),
                          "cand": aggregate_components(subs, cand)})
        deltas[i] = rr["cand"] - rr["base"]
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_point": float(point), "delta_ci_low": float(lo), "delta_ci_high": float(hi),
            "n_resamples": n, "seed": seed,
            "favors_candidate_point": point < 0,
            "ci_excludes_tie_favoring_candidate": float(hi) < 0.0}


def evaluate_candidate(recs_base, recs_cand, expected_n=None, evaluator_errors=0,
                       output_contract_ok=True):
    """Full fail-closed official-metric comparison of one candidate against the baseline."""
    if evaluator_errors:
        raise HardFailure(f"{evaluator_errors} evaluator error(s): fail closed, never a penalty")
    subjects, base, cand = validate_subject_records(recs_base, recs_cand)
    if expected_n is not None and len(subjects) != expected_n:
        raise HardFailure(f"subject count {len(subjects)} != expected {expected_n}")
    aggB = aggregate_components(subjects, base)
    aggC = aggregate_components(subjects, cand)
    ranks = award_ranks({"base": aggB, "cand": aggC})
    rank_gain = ranks["base"] - ranks["cand"]          # positive => candidate better
    boot = paired_bootstrap_compare(subjects, base, cand)
    advances = bool(rank_gain >= MEANINGFUL_RANK_GAIN
                    and boot["ci_excludes_tie_favoring_candidate"]
                    and output_contract_ok)
    return {
        "n_subjects": len(subjects),
        "baseline_aggregate": {f"{r}_{m}": aggB[(r, m)] for (r, m) in OFFICIAL_COMPONENTS},
        "candidate_aggregate": {f"{r}_{m}": aggC[(r, m)] for (r, m) in OFFICIAL_COMPONENTS},
        "baseline_denominators": {f"{r}_{m}": v for (r, m), v in
                                  component_denominators(subjects, base).items()},
        "candidate_denominators": {f"{r}_{m}": v for (r, m), v in
                                   component_denominators(subjects, cand).items()},
        "rank_baseline": ranks["base"], "rank_candidate": ranks["cand"],
        "rank_gain_over_baseline": rank_gain,
        "bootstrap": boot,
        "output_contract_ok": bool(output_contract_ok),
        "advances": advances,
    }


# The frozen candidate list. Anything outside this set is an unauthorised expansion and is
# rejected fail-closed, so a stray key can never win selection.
FROZEN_CANDIDATES = ("C1", "C2", "C3", "C0_et10", "C0_et25", "C0_et50", "S1", "S2")


def choose_strongest(results):
    """Preregistered tie-break among advancing candidates: rank gain, then bootstrap bound,
    then proximity to the frozen baseline (later in CANDIDATE_PROXIMITY = closer).

    Fail-closed: any candidate name outside FROZEN_CANDIDATES raises, so results cannot be
    widened after the fact by introducing a new key.
    """
    CANDIDATE_PROXIMITY = ["C3", "C1", "C2", "S2", "S1", "C0_et50", "C0_et25", "C0_et10"]
    unknown = sorted(set(results) - set(FROZEN_CANDIDATES))
    if unknown:
        raise HardFailure(f"unauthorised candidate(s) not in the frozen list: {unknown}")
    adv = [k for k, v in results.items() if v.get("advances")]
    if not adv:
        return None

    def key(name):
        r = results[name]
        prox = CANDIDATE_PROXIMITY.index(name)
        return (r["rank_gain_over_baseline"], -r["bootstrap"]["delta_ci_high"], prox)

    return sorted(adv, key=key, reverse=True)[0]
