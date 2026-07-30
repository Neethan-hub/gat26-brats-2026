#!/usr/bin/env python3
"""G85 reference-component ("lesion") safety audit, corrected counters.

WHY this module exists: the earlier lesion counters mixed two different spaces.
They counted TP/FP in *prediction* space and FN in *reference* space, then
reported precision/recall/F1 as if TP+FN were a fixed property of the case. It
is not: merging two predicted components into one, or bridging two reference
components with a single predicted blob, changes TP without changing the
reference at all. Any gate built on that quantity can be gamed by a policy that
simply predicts larger, more connected masks.

The corrected quantity anchored here is the reference-side miss rate:

    miss_rate = (reference components with zero predicted overlap) / (reference components)

Its denominator, ``n_ref``, is a function of the reference segmentation alone and
is therefore invariant across policies compared on the same cases. That makes the
policy comparison a paired comparison of numerators over a shared, fixed
denominator, which is what a safety statement about missed lesions requires.

The margins in ``MARGINS_FROZEN`` are frozen before confirmation and are
noninferiority *safety* gates: they neither reward an increase in missed
reference components nor erase any prior stage's failure, and passing them is
never evidence of superiority. Precision/recall/F1 derived from the old counters
are retained under ``diagnostic`` only and must never override an official
metric or a safety gate.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REGIONS = ("ET", "TC", "WT")
GT_LABELS = {"ET": (3,), "TC": (1, 3), "WT": (1, 2, 3)}

# 26-connectivity: a full 3x3x3 neighbourhood, matching the connected-component
# convention used by the official BraTS lesion-wise tooling.
STRUCTURE = np.ones((3, 3, 3), bool)

MARGINS_FROZEN = {
    "point_max": 0.0025,
    "upper_bound_max": 0.0050,
    "region_max": 0.0050,
    "fp_max_increase_fraction": 0.05,
}

DIAGNOSTIC_NOTE = ("naive precision/recall/F1 from the legacy prediction-space counters; "
                   "diagnostic only, must never override an official metric or a safety gate")

DEFAULT_SEED = 20260730
DEFAULT_RESAMPLES = 10000


def component_stats(seg_ref: np.ndarray, seg_pred: np.ndarray, region: str) -> dict:
    """Reference- and prediction-side component counts for one region of one case.

    ``n_ref`` and ``ref_component_voxels`` are computed from ``seg_ref`` only, so
    they cannot move when the prediction changes.
    """
    from scipy import ndimage

    labels = GT_LABELS[region]
    ref = np.isin(seg_ref, labels)
    pred = np.isin(seg_pred, labels)
    lab_ref, n_ref = ndimage.label(ref, structure=STRUCTURE)
    lab_pred, n_pred = ndimage.label(pred, structure=STRUCTURE)

    # bincount over the label image gives every component's voxel count in one
    # pass; bincount restricted to the other mask's voxels gives, per component,
    # how many of its voxels the other mask covers. Zero means no overlap.
    ref_sizes = np.bincount(lab_ref.ravel(), minlength=n_ref + 1)
    ref_overlap = np.bincount(lab_ref[pred].ravel(), minlength=n_ref + 1)
    pred_overlap = np.bincount(lab_pred[ref].ravel(), minlength=n_pred + 1)

    ref_component_voxels = [int(ref_sizes[i]) for i in range(1, n_ref + 1)]
    missed_component_voxels = [int(ref_sizes[i]) for i in range(1, n_ref + 1)
                               if ref_overlap[i] == 0]
    tp_pred = int(sum(1 for i in range(1, n_pred + 1) if pred_overlap[i] > 0))

    return {
        "n_ref": int(n_ref),
        "fn_ref": len(missed_component_voxels),
        "fp_pred": int(n_pred) - tp_pred,
        "tp_pred_diagnostic": tp_pred,
        "ref_component_voxels": sorted(ref_component_voxels),
        "missed_component_voxels": sorted(missed_component_voxels),
    }


def per_case_stats(seg_ref: np.ndarray, seg_pred: np.ndarray) -> dict:
    """All three regions of one case, keyed by region name."""
    return {region: component_stats(seg_ref, seg_pred, region) for region in REGIONS}


def _selected(records: Sequence[Mapping[str, Any]],
              subjects: Iterable[str] | None) -> list[Mapping[str, Any]]:
    if subjects is None:
        return list(records)
    keep = set(subjects)
    return [rec for rec in records if rec["case"] in keep]


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def aggregate_miss_rate(records: Sequence[Mapping[str, Any]], policy_key: str,
                        subjects: Iterable[str] | None = None) -> dict:
    """Pool component counts over subjects and regions for one policy.

    Pooling counts and dividing once (a ratio estimator) is deliberate: a
    mean of per-case miss rates would weight a one-component case the same as a
    twenty-component case, which is not the quantity a safety gate needs.
    """
    selected = _selected(records, subjects)
    per_region = {region: {"n_ref": 0, "fn_ref": 0, "fp_pred": 0,
                           "tp_pred_diagnostic": 0} for region in REGIONS}
    for rec in selected:
        policy = rec[policy_key]
        for region in REGIONS:
            stats = policy.get(region)
            if stats is None:
                continue
            acc = per_region[region]
            for key in ("n_ref", "fn_ref", "fp_pred", "tp_pred_diagnostic"):
                acc[key] += int(stats.get(key, 0))

    n_ref_total = sum(acc["n_ref"] for acc in per_region.values())
    fn_ref_total = sum(acc["fn_ref"] for acc in per_region.values())
    fp_pred_total = sum(acc["fp_pred"] for acc in per_region.values())
    tp_total = sum(acc["tp_pred_diagnostic"] for acc in per_region.values())

    return {
        "n_ref_total": n_ref_total,
        "fn_ref_total": fn_ref_total,
        "fp_pred_total": fp_pred_total,
        "tp_pred_diagnostic_total": tp_total,
        "miss_rate": _ratio(fn_ref_total, n_ref_total),
        "n_subjects": len(selected),
        "per_region": {region: {"n_ref": acc["n_ref"], "fn_ref": acc["fn_ref"],
                                "fp_pred": acc["fp_pred"],
                                "miss_rate": _ratio(acc["fn_ref"], acc["n_ref"])}
                       for region, acc in per_region.items()},
    }


def _subject_totals(records: Sequence[Mapping[str, Any]], policy_key: str) -> tuple:
    """Per-subject (fn_ref, n_ref) sums; the unit of resampling is one subject."""
    fn = np.zeros(len(records), dtype=np.float64)
    n_ref = np.zeros(len(records), dtype=np.float64)
    for i, rec in enumerate(records):
        policy = rec[policy_key]
        for region in REGIONS:
            stats = policy.get(region)
            if stats is None:
                continue
            fn[i] += int(stats.get("fn_ref", 0))
            n_ref[i] += int(stats.get("n_ref", 0))
    return fn, n_ref


def paired_miss_rate_bootstrap(records: Sequence[Mapping[str, Any]], base_key: str,
                               cand_key: str, subjects: Iterable[str] | None = None,
                               seed: int = DEFAULT_SEED,
                               n: int = DEFAULT_RESAMPLES) -> dict:
    """Paired subject-level bootstrap of the miss-rate delta (candidate - baseline).

    Subjects, never individual components, are resampled: components inside one
    subject are correlated, so resampling them would understate the variance.
    Both numerator and denominator are re-summed inside every resample and the
    two miss rates are recomputed from scratch before the delta is taken, which
    keeps the ratio estimator's denominator variation inside the interval.
    """
    selected = _selected(records, subjects)
    base = aggregate_miss_rate(selected, base_key)
    cand = aggregate_miss_rate(selected, cand_key)
    point_delta = cand["miss_rate"] - base["miss_rate"]
    k = len(selected)
    if k == 0 or n <= 0:
        return {"point_delta": point_delta, "mean_delta": 0.0, "std_delta": 0.0,
                "upper_95_one_sided": point_delta, "ci95_two_sided": [point_delta,
                                                                      point_delta],
                "prob_increase": 0.0, "n_subjects": k, "seed": seed, "resamples": 0}

    b_fn, b_n = _subject_totals(selected, base_key)
    c_fn, c_n = _subject_totals(selected, cand_key)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n, k))
    b_rate = _resampled_rate(b_fn, b_n, idx)
    c_rate = _resampled_rate(c_fn, c_n, idx)
    deltas = c_rate - b_rate

    return {
        "point_delta": point_delta,
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std()),
        "upper_95_one_sided": float(np.quantile(deltas, 0.95)),
        "ci95_two_sided": [float(np.quantile(deltas, 0.025)),
                           float(np.quantile(deltas, 0.975))],
        "prob_increase": float((deltas > 0).mean()),
        "n_subjects": k,
        "seed": seed,
        "resamples": int(n),
    }


def _resampled_rate(fn: np.ndarray, n_ref: np.ndarray, idx: np.ndarray) -> np.ndarray:
    num = fn[idx].sum(axis=1)
    den = n_ref[idx].sum(axis=1)
    return np.where(den > 0, num / np.maximum(den, 1.0), 0.0)


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2.0 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp_pred_diagnostic": int(tp), "fp_pred": int(fp), "fn_ref": int(fn)}


def noninferiority(records: Sequence[Mapping[str, Any]], base_key: str, cand_key: str,
                   margins: Mapping[str, float] | None = None,
                   subjects: Iterable[str] | None = None, *,
                   seed: int = DEFAULT_SEED, resamples: int = DEFAULT_RESAMPLES) -> dict:
    """Frozen-margin noninferiority decision on the reference-side miss rate.

    Every check must hold. ``n_ref_total_invariant`` is a correctness assertion
    on the audit itself, not a scientific gate: if the two policies disagree on
    the number of reference components then the counters were computed against
    different references or different case sets, and no comparison is valid.
    """
    used = dict(MARGINS_FROZEN)
    if margins:
        used.update({k: float(v) for k, v in margins.items()})

    base = aggregate_miss_rate(records, base_key, subjects)
    cand = aggregate_miss_rate(records, cand_key, subjects)
    boot = paired_miss_rate_bootstrap(records, base_key, cand_key, subjects,
                                      seed=seed, n=resamples)

    region_deltas = {region: (cand["per_region"][region]["miss_rate"]
                              - base["per_region"][region]["miss_rate"])
                     for region in REGIONS}
    base_fp = base["fp_pred_total"]
    fp_increase_fraction = _ratio(cand["fp_pred_total"] - base_fp, base_fp)

    checks = {
        "n_ref_total_invariant": base["n_ref_total"] == cand["n_ref_total"],
        "point_miss_rate_within_margin": boot["point_delta"] <= used["point_max"],
        "upper_bound_within_margin": (boot["upper_95_one_sided"]
                                      <= used["upper_bound_max"]),
        "no_region_miss_rate_regression": all(d <= used["region_max"]
                                              for d in region_deltas.values()),
        "fp_within_limit": fp_increase_fraction <= used["fp_max_increase_fraction"],
    }

    return {
        "passes": all(checks.values()),
        "checks": checks,
        "margins": used,
        "baseline": base,
        "candidate": cand,
        "bootstrap": boot,
        "region_deltas": region_deltas,
        "diagnostic": {
            "note": DIAGNOSTIC_NOTE,
            "baseline": _prf(base["tp_pred_diagnostic_total"], base["fp_pred_total"],
                             base["fn_ref_total"]),
            "candidate": _prf(cand["tp_pred_diagnostic_total"], cand["fp_pred_total"],
                              cand["fn_ref_total"]),
            "fp_increase_fraction": fp_increase_fraction,
        },
    }
