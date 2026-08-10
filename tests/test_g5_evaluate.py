#!/usr/bin/env python3
"""GAT-26 G5 evaluator unit tests (`python3 tests/test_g5_evaluate.py`).

Pure, synthetic checks of the fail-closed HD95 / metric normalization (C1) -- no
brats_evaluation, panoptica, nibabel, GPU, or real data. All stems/values are synthetic;
no real case IDs or private hashes appear here.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import g5_evaluate as E  # noqa: E402

FAILS = []


def check(name, cond):
    if not cond:
        FAILS.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def hardfails(d, h, ge, pe):
    try:
        E.classify_region(d, h, ge, pe)
        return False
    except E.HardFail:
        return True


def main():
    INF = math.inf
    NAN = math.nan

    # ---- legitimate cases ----
    check("empty_both_tn -> (1.0, 0.0)",
          E.classify_region(NAN, NAN, 1, 1) == (1.0, 0.0, "empty_both_tn"))
    check("empty_both_tn even if evaluator gave None",
          E.classify_region(None, None, 1, 1) == (1.0, 0.0, "empty_both_tn"))
    check("gt_empty xor pred_empty -> penalty (0.0, 373.0)",
          E.classify_region(0.0, INF, 1, 0) == (0.0, 373.0, "penalty_zero_tp"))
    check("miss (gt present, pred empty) -> penalty (0.0, 373.0)",
          E.classify_region(0.0, INF, 0, 1) == (0.0, 373.0, "penalty_zero_tp"))

    # ---- both non-empty: finite HD95 unchanged ----
    d0, h0, s0 = E.classify_region(0.83, 4.5, 0, 0)
    check("finite hd95 unchanged", d0 == 0.83 and h0 == 4.5 and s0 == "ok")
    # ---- both non-empty but disjoint (+inf HD95) -> 373, DSC kept ----
    check("both-nonempty +inf hd95 -> 373 (disjoint), dsc kept",
          E.classify_region(0.0, INF, 0, 0) == (0.0, 373.0, "penalty_disjoint"))

    # ---- hard failures (both non-empty) ----
    check("NaN dsc fails", hardfails(NAN, 3.0, 0, 0))
    check("NaN hd95 fails", hardfails(0.7, NAN, 0, 0))
    check("-inf hd95 fails", hardfails(0.7, -INF, 0, 0))
    check("missing dsc fails", hardfails(None, 3.0, 0, 0))
    check("missing hd95 fails", hardfails(0.7, None, 0, 0))
    check("dsc>1 out of range fails", hardfails(1.5, 3.0, 0, 0))
    check("dsc<0 out of range fails", hardfails(-0.1, 3.0, 0, 0))
    check("finite negative hd95 fails", hardfails(0.7, -2.0, 0, 0))

    # ---- aggregate: +inf->373 retained in mean AND denominator ----
    # three synthetic subjects; one region disjoint (hd95 +inf -> 373), all others finite.
    recs = []
    for i, hd in enumerate([10.0, 10.0, INF]):
        d, h, _ = E.classify_region(0.5, hd, 0, 0)          # +inf -> 373 via classify
        recs.append({"et_dsc": 0.5, "et_hd95": h,
                     "tc_dsc": 0.5, "tc_hd95": 5.0,
                     "wt_dsc": 0.5, "wt_hd95": 5.0})
    comps = E.aggregate_components(recs, n_expected=3)
    et_mean, et_denom = comps["et_hd95"]
    check("et_hd95 denominator retained = 3 (inf not dropped)", et_denom == 3)
    check("et_hd95 mean includes 373 penalty",
          abs(et_mean - (10.0 + 10.0 + 373.0) / 3.0) < 1e-12)
    check("dsc denominator retained = 3", comps["et_dsc"][1] == 3)

    # ---- aggregate: denominator-mismatch fails closed ----
    try:
        E.aggregate_components(recs, n_expected=271)
        check("aggregate denom mismatch fails", False)
    except E.HardFail:
        check("aggregate denom mismatch fails", True)

    # ---- aggregate: a nonfinite that slipped through fails closed ----
    bad = [dict(recs[0]), dict(recs[1]), dict(recs[2])]
    bad[0]["wt_hd95"] = INF
    try:
        E.aggregate_components(bad, n_expected=3)
        check("aggregate nonfinite fails", False)
    except E.HardFail:
        check("aggregate nonfinite fails", True)

    print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
