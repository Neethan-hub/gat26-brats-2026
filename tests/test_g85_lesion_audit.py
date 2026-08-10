#!/usr/bin/env python3
"""G85 lesion-audit proofs on synthetic volumes. No protected data, no GPU.

WHY these particular tests: the corrected audit only earns trust if the failure
modes of the counters it replaces are demonstrated, not asserted. Each test below
either pins an invariance the old counters lacked (the reference-component
denominator), exhibits a prediction-space quantity that moves while the reference
stands still, or shows that a frozen safety margin actually stops progression.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import g85_lesion_audit as A  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []
SHAPE = (24, 24, 24)


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# ------------------------------------------------------------------- synthetic volumes
def _empty() -> np.ndarray:
    return np.zeros(SHAPE, dtype=np.int16)


def _box(vol: np.ndarray, origin, size, label: int = 3) -> np.ndarray:
    """Fill a half-open box. Boxes must leave a two-voxel gap to stay 26-separated."""
    z, y, x = origin
    dz, dy, dx = size if isinstance(size, tuple) else (size, size, size)
    vol[z:z + dz, y:y + dy, x:x + dx] = label
    return vol


def _ref_three_cubes() -> np.ndarray:
    ref = _empty()
    _box(ref, (2, 2, 2), 3)
    _box(ref, (2, 2, 10), 3)
    _box(ref, (2, 2, 18), 3)
    return ref


def _rec(case: str, base: dict, cand: dict) -> dict:
    return {"case": case, "base": base, "cand": cand}


def _stats(n_ref: int, fn_ref: int, fp_pred: int, tp: int | None = None) -> dict:
    """Fabricated per-region counters, used where exact numbers must be controlled."""
    return {"n_ref": n_ref, "fn_ref": fn_ref, "fp_pred": fp_pred,
            "tp_pred_diagnostic": (n_ref - fn_ref) if tp is None else tp,
            "ref_component_voxels": [], "missed_component_voxels": []}


# -------------------------------------------------------- 1 denominator is policy-free
def t_n_ref_invariant() -> None:
    ref = _ref_three_cubes()
    pred_none = _empty()
    pred_all = np.full(SHAPE, 3, dtype=np.int16)
    a = A.per_case_stats(ref, pred_none)
    b = A.per_case_stats(ref, pred_all)
    check("1 component_stats_keys_exact",
          set(a["ET"]) == {"n_ref", "fn_ref", "fp_pred", "tp_pred_diagnostic",
                           "ref_component_voxels", "missed_component_voxels"},
          str(sorted(a["ET"])))
    check("1b n_ref_three_reference_components_all_regions",
          all(a[r]["n_ref"] == 3 for r in A.REGIONS),
          str({r: a[r]["n_ref"] for r in A.REGIONS}))
    check("1c n_ref_identical_under_opposite_predictions",
          all(a[r]["n_ref"] == b[r]["n_ref"] for r in A.REGIONS),
          str({r: (a[r]["n_ref"], b[r]["n_ref"]) for r in A.REGIONS}))
    # The saturated prediction is a single component covering all three reference
    # components, which is exactly why a prediction-space TP count cannot stand in
    # for the reference-component denominator.
    check("1d predictions_really_were_different",
          a["ET"]["fn_ref"] == 3 and b["ET"]["fn_ref"] == 0
          and a["ET"]["fp_pred"] == 0 and b["ET"]["tp_pred_diagnostic"] == 1,
          f"fn {a['ET']['fn_ref']}/{b['ET']['fn_ref']} "
          f"tp {a['ET']['tp_pred_diagnostic']}/{b['ET']['tp_pred_diagnostic']}")
    recs = [_rec("c1", a, b)]
    agg_a = A.aggregate_miss_rate(recs, "base")
    agg_b = A.aggregate_miss_rate(recs, "cand")
    check("1e aggregate_n_ref_total_policy_invariant",
          agg_a["n_ref_total"] == agg_b["n_ref_total"] == 9,
          f"{agg_a['n_ref_total']} vs {agg_b['n_ref_total']}")
    check("1f miss_rate_uses_reference_denominator",
          agg_a["miss_rate"] == 1.0 and agg_b["miss_rate"] == 0.0,
          f"{agg_a['miss_rate']} {agg_b['miss_rate']}")
    check("1g reference_voxel_sizes_policy_invariant",
          a["ET"]["ref_component_voxels"] == b["ET"]["ref_component_voxels"] == [27, 27, 27],
          str(a["ET"]["ref_component_voxels"]))
    check("1h missed_voxels_listed_only_for_missed_components",
          a["ET"]["missed_component_voxels"] == [27, 27, 27]
          and b["ET"]["missed_component_voxels"] == [])


# ---------------------------------------------------- 2 TP+FN is not a case invariant
def t_tp_plus_fn_not_invariant() -> None:
    ref = _empty()
    _box(ref, (2, 2, 2), 3)
    _box(ref, (2, 2, 12), 3)
    pred_one = _box(_empty(), (2, 2, 2), 3)
    # A single predicted blob that bridges both reference components: one TP covers
    # two reference lesions, so TP+FN drops while the reference is unchanged.
    pred_bridge = _box(_empty(), (2, 2, 2), (3, 3, 13))
    a = A.per_case_stats(ref, pred_one)["ET"]
    b = A.per_case_stats(ref, pred_bridge)["ET"]
    sum_a = a["tp_pred_diagnostic"] + a["fn_ref"]
    sum_b = b["tp_pred_diagnostic"] + b["fn_ref"]
    check("2 n_ref_equal_across_the_two_predictions", a["n_ref"] == b["n_ref"] == 2,
          f"{a['n_ref']} vs {b['n_ref']}")
    check("2b tp_plus_fn_differs_so_invariance_must_not_be_assumed", sum_a != sum_b,
          f"{sum_a} vs {sum_b}")
    check("2c bridging_hides_a_miss_in_prediction_space",
          a["fn_ref"] == 1 and b["fn_ref"] == 0
          and a["tp_pred_diagnostic"] == 1 and b["tp_pred_diagnostic"] == 1)
    recs = [_rec("c1", A.per_case_stats(ref, pred_one),
                 A.per_case_stats(ref, pred_bridge))]
    agg_a = A.aggregate_miss_rate(recs, "base")
    agg_b = A.aggregate_miss_rate(recs, "cand")
    check("2d audit_denominator_survives_the_bridge",
          agg_a["n_ref_total"] == agg_b["n_ref_total"],
          f"{agg_a['n_ref_total']} vs {agg_b['n_ref_total']}")
    check("2e audit_reports_the_miss_that_tp_plus_fn_hid",
          agg_a["fn_ref_total"] > agg_b["fn_ref_total"])


# ------------------------------------- 3 splitting a predicted component is inert
def t_split_prediction_cannot_move_denominator() -> None:
    ref = _empty()
    _box(ref, (2, 2, 2), 3)
    _box(ref, (14, 14, 14), 3)
    whole = _box(_empty(), (2, 2, 2), (3, 3, 18))
    split = whole.copy()
    split[:, :, 10:12] = 0  # two-voxel cut keeps the pieces 26-disconnected
    a = A.per_case_stats(ref, whole)
    b = A.per_case_stats(ref, split)
    recs = [_rec("c1", a, b)]
    agg_a = A.aggregate_miss_rate(recs, "base")
    agg_b = A.aggregate_miss_rate(recs, "cand")
    check("3 split_leaves_n_ref_total_untouched",
          agg_a["n_ref_total"] == agg_b["n_ref_total"] == 6,
          f"{agg_a['n_ref_total']} vs {agg_b['n_ref_total']}")
    check("3b split_leaves_fn_ref_untouched",
          agg_a["fn_ref_total"] == agg_b["fn_ref_total"],
          f"{agg_a['fn_ref_total']} vs {agg_b['fn_ref_total']}")
    check("3c split_did_change_prediction_space_counters",
          b["ET"]["fp_pred"] > a["ET"]["fp_pred"],
          f"fp {a['ET']['fp_pred']} -> {b['ET']['fp_pred']}")
    check("3d split_touched_no_new_reference_component",
          a["ET"]["missed_component_voxels"] == b["ET"]["missed_component_voxels"] == [27])


# -------------------------------------------------- 4 the bootstrap resamples subjects
class _RecordingRNG:
    """Wraps a real Generator to capture the resampling index range."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[tuple] = []

    def integers(self, low, high=None, size=None, **kw):
        self.calls.append((low, high, size))
        return self.inner.integers(low, high, size=size, **kw)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def t_bootstrap_resamples_subjects() -> None:
    single = [_rec("s0", {"ET": _stats(4, 0, 0)}, {"ET": _stats(4, 1, 0)})]
    boot1 = A.paired_miss_rate_bootstrap(single, "base", "cand", n=500)
    check("4 single_subject_gives_zero_variance", boot1["std_delta"] == 0.0,
          f"std={boot1['std_delta']}")
    check("4b single_subject_interval_is_degenerate",
          boot1["ci95_two_sided"][0] == boot1["ci95_two_sided"][1] == boot1["point_delta"]
          == boot1["mean_delta"] == boot1["upper_95_one_sided"] == 0.25,
          str(boot1["ci95_two_sided"]))

    many = [_rec(f"s{i}", {"ET": _stats(4, 0, 0)},
                 {"ET": _stats(4, 1 if i < 3 else 0, 0)}) for i in range(11)]
    captured: list[_RecordingRNG] = []
    real = np.random.default_rng

    def fake(seed=None):
        rng = _RecordingRNG(real(seed))
        captured.append(rng)
        return rng

    np.random.default_rng = fake
    try:
        boot2 = A.paired_miss_rate_bootstrap(many, "base", "cand", n=300)
    finally:
        np.random.default_rng = real
    calls = captured[0].calls if captured else []
    check("4c resample_index_range_is_the_subject_count",
          bool(calls) and calls[0][0] == 0 and calls[0][1] == 11,
          str(calls))
    check("4d resample_draws_one_index_per_subject_per_replicate",
          bool(calls) and tuple(calls[0][2]) == (300, 11), str(calls))
    check("4e n_subjects_equals_record_count", boot2["n_subjects"] == len(many) == 11,
          f"{boot2['n_subjects']} vs {len(many)}")
    check("4f multi_subject_bootstrap_has_spread", boot2["std_delta"] > 0.0,
          f"std={boot2['std_delta']}")
    check("4g subject_resampling_never_touches_component_counts",
          len(calls) == 1, f"{len(calls)} integer draws")


# ------------------------------------------------------------- 5 exact equality passes
def t_exact_equality_passes() -> None:
    recs = []
    for i in range(4):
        ref = _empty()
        _box(ref, (2, 2, 2 + i), 3)
        _box(ref, (12, 12, 12), 3)
        pred = _box(_empty(), (2, 2, 2 + i), 3)
        stats = A.per_case_stats(ref, pred)
        recs.append(_rec(f"c{i}", stats, A.per_case_stats(ref, pred)))
    boot = A.paired_miss_rate_bootstrap(recs, "base", "cand", n=1000)
    check("5 identical_predictions_give_zero_point_delta", boot["point_delta"] == 0.0,
          str(boot["point_delta"]))
    check("5b identical_predictions_give_non_positive_upper_bound",
          boot["upper_95_one_sided"] <= 0.0, str(boot["upper_95_one_sided"]))
    res = A.noninferiority(recs, "base", "cand", None, resamples=1000)
    check("5c exact_equality_passes_all_checks", res["passes"] is True,
          "failing: " + ",".join(sorted(k for k, v in res["checks"].items() if not v)))
    check("5d equality_case_has_real_misses_to_detect",
          res["baseline"]["fn_ref_total"] > 0,
          f"fn={res['baseline']['fn_ref_total']}")


# ------------------------------------------------------------ 6 point failure blocks
def t_point_failure_blocks() -> None:
    recs = [_rec(f"s{i}", {"ET": _stats(10, 0, 2)},
                 {"ET": _stats(10, 1 if i == 0 else 0, 2)}) for i in range(10)]
    res = A.noninferiority(recs, "base", "cand", None, resamples=2000)
    check("6 point_margin_exceeded_fails",
          res["checks"]["point_miss_rate_within_margin"] is False,
          f"delta={res['bootstrap']['point_delta']:.6f}")
    check("6b point_failure_blocks_progression", res["passes"] is False)
    check("6c denominator_still_invariant_under_failure",
          res["checks"]["n_ref_total_invariant"] is True)


# ------------------------------------------------------- 7 upper-bound failure blocks
def t_upper_bound_failure_blocks() -> None:
    # 200 subjects, five reference components each. One subject loses two components,
    # so the pooled point estimate (0.002) sits inside the 0.0025 margin, but a
    # resample that draws that subject several times pushes the 95th percentile out.
    recs = [_rec(f"s{i}", {"ET": _stats(5, 0, 1)},
                 {"ET": _stats(5, 2 if i == 0 else 0, 1)}) for i in range(200)]
    res = A.noninferiority(recs, "base", "cand", None)
    boot = res["bootstrap"]
    check("7 point_estimate_inside_margin",
          res["checks"]["point_miss_rate_within_margin"] is True,
          f"point={boot['point_delta']:.6f}")
    check("7b upper_bound_outside_margin",
          res["checks"]["upper_bound_within_margin"] is False,
          f"upper={boot['upper_95_one_sided']:.6f} > {res['margins']['upper_bound_max']}")
    check("7c upper_bound_failure_blocks_progression", res["passes"] is False)
    check("7d one_sided_bound_is_above_the_point_estimate",
          boot["upper_95_one_sided"] > boot["point_delta"] > 0.0)


# ------------------------------------------------------- 8 per-region regression blocks
def t_region_regression_blocks() -> None:
    # ET carries 100 reference components, WT carries 100000, so a single extra ET
    # miss is invisible in the pooled rate and must be caught per region.
    recs = [_rec(f"s{i}", {"ET": _stats(2, 0, 1), "WT": _stats(2000, 0, 1)},
                 {"ET": _stats(2, 1 if i == 0 else 0, 1), "WT": _stats(2000, 0, 1)})
            for i in range(50)]
    res = A.noninferiority(recs, "base", "cand", None, resamples=2000)
    check("8 pooled_point_passes",
          res["checks"]["point_miss_rate_within_margin"] is True,
          f"point={res['bootstrap']['point_delta']:.8f}")
    check("8b pooled_upper_bound_passes",
          res["checks"]["upper_bound_within_margin"] is True,
          f"upper={res['bootstrap']['upper_95_one_sided']:.8f}")
    check("8c region_regression_detected",
          res["checks"]["no_region_miss_rate_regression"] is False,
          f"ET delta={res['region_deltas']['ET']:.6f}")
    check("8d region_regression_blocks_progression", res["passes"] is False)


# ------------------------------------------------------------------ 9 FP limit blocks
def t_fp_increase_blocks() -> None:
    recs = [_rec(f"s{i}", {"ET": _stats(10, 0, 5)}, {"ET": _stats(10, 0, 6)})
            for i in range(20)]
    res = A.noninferiority(recs, "base", "cand", None, resamples=2000)
    check("9 fp_increase_beyond_five_percent_fails",
          res["checks"]["fp_within_limit"] is False,
          f"fraction={res['diagnostic']['fp_increase_fraction']:.4f}")
    check("9b miss_rate_checks_still_pass",
          res["checks"]["point_miss_rate_within_margin"] is True
          and res["checks"]["upper_bound_within_margin"] is True)
    check("9c fp_failure_blocks_progression", res["passes"] is False)
    ok = A.noninferiority([_rec("s0", {"ET": _stats(10, 0, 100)},
                                {"ET": _stats(10, 0, 105)})],
                          "base", "cand", None, resamples=200)
    check("9d exactly_five_percent_is_allowed", ok["checks"]["fp_within_limit"] is True,
          f"fraction={ok['diagnostic']['fp_increase_fraction']:.4f}")


# --------------------------------------------------- 10 diagnostic F1 cannot override
def t_diagnostic_f1_cannot_override() -> None:
    # Candidate trades misses for precision: fewer false positives, more missed
    # reference components. Legacy F1 improves; the safety gate must still refuse.
    recs = [_rec(f"s{i}", {"ET": _stats(11, 1, 5)}, {"ET": _stats(11, 3, 0)})
            for i in range(10)]
    res = A.noninferiority(recs, "base", "cand", None, resamples=2000)
    d = res["diagnostic"]
    check("10 candidate_diagnostic_f1_is_higher",
          d["candidate"]["f1"] > d["baseline"]["f1"],
          f"{d['baseline']['f1']:.4f} -> {d['candidate']['f1']:.4f}")
    check("10b candidate_diagnostic_precision_is_higher",
          d["candidate"]["precision"] > d["baseline"]["precision"])
    check("10c safety_gate_still_fails", res["passes"] is False,
          "failing: " + ",".join(sorted(k for k, v in res["checks"].items() if not v)))
    check("10d failure_is_the_miss_rate_not_the_f1",
          res["checks"]["point_miss_rate_within_margin"] is False
          and res["checks"]["fp_within_limit"] is True)
    check("10e diagnostic_is_labelled_as_non_overriding",
          "diagnostic only" in d["note"] and "never override" in d["note"], d["note"])


# ---------------------------------------------------------------- 11 frozen margins
def t_margins_frozen() -> None:
    check("11 margins_frozen_exact_values",
          A.MARGINS_FROZEN == {"point_max": 0.0025, "upper_bound_max": 0.0050,
                               "region_max": 0.0050, "fp_max_increase_fraction": 0.05},
          str(A.MARGINS_FROZEN))
    check("11b default_margins_are_the_frozen_margins",
          A.noninferiority([_rec("s0", {"ET": _stats(4, 0, 0)}, {"ET": _stats(4, 0, 0)})],
                           "base", "cand", None, resamples=100)["margins"]
          == A.MARGINS_FROZEN)
    check("11c connectivity_is_26_connected",
          A.STRUCTURE.shape == (3, 3, 3) and bool(A.STRUCTURE.all())
          and A.STRUCTURE.dtype == np.bool_)
    check("11d regions_match_the_label_convention",
          A.GT_LABELS == {"ET": (3,), "TC": (1, 3), "WT": (1, 2, 3)})


def main() -> int:
    for fn in (t_n_ref_invariant, t_tp_plus_fn_not_invariant,
               t_split_prediction_cannot_move_denominator, t_bootstrap_resamples_subjects,
               t_exact_equality_passes, t_point_failure_blocks,
               t_upper_bound_failure_blocks, t_region_regression_blocks,
               t_fp_increase_blocks, t_diagnostic_f1_cannot_override, t_margins_frozen):
        fn()
    n = len(RESULTS)
    ok = sum(1 for _, o, _ in RESULTS if o)
    print(f"\n{ok}/{n} checks passed")
    if ok != n:
        print("FAILED:", [r[0] for r in RESULTS if not r[1]])
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
