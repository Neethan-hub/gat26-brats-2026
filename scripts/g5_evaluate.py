#!/usr/bin/env python3
"""GAT-26 G5 official evaluation — BraTS-evaluation 0.0.8 GoAT (runs in the eval venv).

Evaluates frozen fold-0 validation predictions against ground truth, producing PRIVATE
per-subject records plus the sanitized aggregate/tail metrics selection-policy v2 needs.

Fail-closed HD95 / metric handling (C1):
  * A successful evaluator record may return legitimate positive infinity HD95 for a zero
    true-positive region. Every legitimate +inf HD95 is converted to 373.0 BEFORE any
    component mean / percentile / tail / selection-policy input, and is NEVER discarded
    from the mean or the denominator: each ET/TC/WT DSC and HD95 component keeps exactly
    the full evaluated-subject count.
  * A region that is empty in BOTH ground truth and prediction is a legitimate true-negative
    (0/0 is undefined -> panoptica emits NaN). Its emptiness is derived INDEPENDENTLY from
    the label arrays, so it is scored by the standard BraTS closure convention DSC=1.0,
    HD95=0.0 -- grounded in the data, not in masking a NaN.
  * Everything else that is nonfinite/absent/out-of-range is a HARD failure (exit != 0):
    NaN or None DSC in a region that is not empty-both, DSC outside [0,1], negative-infinity
    HD95, finite HD95 < 0, missing metric, malformed evaluator record, evaluator exception,
    missing prediction, or missing ground truth. A missing prediction / evaluator error is
    NEVER converted into 373.

Pure helpers (classify_region, aggregate_components) import only stdlib math so unit tests
run without brats_evaluation/panoptica/nibabel/torch. Sanitized stdout: counts + a
normalization-status histogram only -- no real case IDs, no per-case values.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REGIONS = ["et", "tc", "wt"]
HD95_PENALTY = 373.0            # BraTS zero-TP HD95 penalty
DSC_LO, DSC_HI = 0.0, 1.0


class HardFail(Exception):
    """A genuine, non-legitimate metric malformation -> the whole evaluation must fail."""


# ------------------------------ pure, testable core ------------------------------
def classify_region(d, h, gt_empty, pred_empty):
    """Normalize one region's (dsc, hd95) into aggregate-ready finite values.

    d, h are the raw evaluator DSC / HD95 (may be None / NaN / +-inf / finite).
    gt_empty, pred_empty are booleans derived INDEPENDENTLY from the integer label arrays.

    Returns (dsc_agg, hd95_agg, status). Raises HardFail on a genuine malformation.
    Legitimacy is decided by data-derived emptiness, never by the possibly-NaN metric.
    """
    ge, pe = bool(gt_empty), bool(pred_empty)

    # 1) true-negative region: both empty -> perfect agreement on absence (BraTS closure).
    if ge and pe:
        return 1.0, 0.0, "empty_both_tn"

    # 2) exactly one side empty -> overlap impossible -> DSC=0, HD95 penalty (grounded).
    if ge != pe:
        return 0.0, HD95_PENALTY, "penalty_zero_tp"

    # 3) both regions non-empty -> the evaluator must give a well-formed record.
    if d is None:
        raise HardFail("dsc_missing")
    d = float(d)
    if math.isnan(d):
        raise HardFail("dsc_nan")
    if math.isinf(d):
        raise HardFail("dsc_inf")
    if not (DSC_LO <= d <= DSC_HI):
        raise HardFail("dsc_out_of_range")
    if h is None:
        raise HardFail("hd95_missing")
    h = float(h)
    if math.isnan(h):
        raise HardFail("hd95_nan")
    if math.isinf(h):
        if h < 0:
            raise HardFail("hd95_neg_inf")
        # both non-empty but disjoint (zero TP) -> legitimate +inf -> penalty, DSC kept.
        return d, HD95_PENALTY, "penalty_disjoint"
    if h < 0:
        raise HardFail("hd95_negative_finite")
    return d, h, "ok"


def aggregate_components(records, regions=REGIONS, n_expected=None):
    """records: list of per-subject dicts with normalized f'{r}_dsc'/f'{r}_hd95' (finite).

    Returns {f'{r}_dsc': (mean, denom), f'{r}_hd95': (mean, denom)}. Each denom must equal
    the number of records (and n_expected if given). Raises HardFail on any nonfinite value
    or denominator mismatch.
    """
    n = len(records)
    if n_expected is not None and n != n_expected:
        raise HardFail(f"denominator_mismatch:{n}!={n_expected}")
    comps = {}
    for r in regions:
        for m in ("dsc", "hd95"):
            vals = [rec[f"{r}_{m}"] for rec in records]
            if len(vals) != n:
                raise HardFail("component_length_mismatch")
            for v in vals:
                if not math.isfinite(v):
                    raise HardFail(f"nonfinite_after_norm:{r}_{m}")
            comps[f"{r}_{m}"] = (sum(vals) / len(vals), len(vals))
    return comps


def region_masks(seg):
    seg = np.rint(np.asarray(seg)).astype(np.int16)
    return {"et": (seg == 3), "tc": (seg == 1) | (seg == 3),
            "wt": (seg == 1) | (seg == 2) | (seg == 3)}


# ------------------------------ evaluator glue (heavy deps, lazy) ------------------------------
def load_evaluator():
    import brats_evaluation as be
    from panoptica import Panoptica_Evaluator
    return Panoptica_Evaluator.load_from_config(str(be.config_path("GoAT")))


def region_metric(res, region, metric):
    key = {"dsc": "global_bin_dsc", "hd95": "global_bin_hd95"}[metric]
    reg = res.get(region)
    if isinstance(reg, dict) and key in reg:
        try:
            return float(reg[key])
        except (TypeError, ValueError):
            return None
    return None


def _raw(x):
    """JSON-safe raw value: None stays None, finite -> float, +-inf/NaN -> its string tag."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else str(x)


def evaluate_all(preds, gt, evaluate_single, ev):
    """Run the evaluator over every prediction; normalize each region fail-closed.

    Returns (records, errors, per_status, vols, fp_empty, missed). `records` is a dict keyed
    by real case id (PRIVATE). Each value holds raw + normalized region metrics. A subject's
    accumulators are committed atomically only after all three regions normalize -- a
    mid-subject HardFail contributes nothing to any component.
    """
    import nibabel as nib
    records, errors = {}, []
    per_status = {}
    vols = {r: [] for r in REGIONS}
    fp_empty = {r: [] for r in REGIONS}
    missed = {r: [] for r in REGIONS}
    for p in preds:
        cid = p.name[:-len(".nii.gz")]
        ref = gt / f"{cid}.nii.gz"
        if not ref.exists():
            errors.append({"subject": cid, "reason": "missing_ground_truth"})
            continue
        try:
            res = evaluate_single(str(p), str(ref), cid, ev)
        except Exception as e:                       # evaluator raised -> HARD, never penalty
            errors.append({"subject": cid, "reason": f"evaluator_exception:{type(e).__name__}"})
            continue
        if not isinstance(res, dict):
            errors.append({"subject": cid, "reason": "malformed_result"})
            continue
        gseg = np.asanyarray(nib.load(str(ref)).dataobj)
        pseg = np.asanyarray(nib.load(str(p)).dataobj)
        gm, pm = region_masks(gseg), region_masks(pseg)
        rec, s_status, s_vol, s_fp, s_missed = {}, {}, {}, {}, {}
        try:
            for r in REGIONS:
                d = region_metric(res, r, "dsc")
                h = region_metric(res, r, "hd95")
                ge = int(gm[r].sum() == 0)
                pe = int(pm[r].sum() == 0)
                dsc_agg, hd95_agg, status = classify_region(d, h, ge, pe)
                rec[f"{r}_dsc"] = dsc_agg
                rec[f"{r}_hd95"] = hd95_agg
                rec[f"{r}_dsc_raw"] = _raw(d)
                rec[f"{r}_hd95_raw"] = _raw(h)
                rec[f"{r}_status"] = status
                s_status[r] = status
                s_vol[r] = int(gm[r].sum())
                s_fp[r] = 1 if (gm[r].sum() == 0 and pm[r].sum() > 0) else 0
                s_missed[r] = 1 if (gm[r].sum() > 0 and pm[r].sum() == 0) else 0
        except HardFail as hf:
            errors.append({"subject": cid, "reason": f"malformed_metrics:{hf}"})
            continue
        records[cid] = rec                            # commit this subject atomically
        for r in REGIONS:
            per_status[s_status[r]] = per_status.get(s_status[r], 0) + 1
            vols[r].append(s_vol[r])
            fp_empty[r].append(s_fp[r])
            missed[r].append(s_missed[r])
    return records, errors, per_status, vols, fp_empty, missed


def build_aggregate(records, vols, fp_empty, missed, n_expected):
    recs = list(records.values())
    comps = aggregate_components(recs, n_expected=n_expected)
    components_mean = {k: v[0] for k, v in comps.items()}
    denominators = {k: v[1] for k, v in comps.items()}
    all_dsc = np.array([rec[f"{r}_dsc"] for rec in recs for r in REGIONS], dtype=float)
    all_hd95 = np.array([rec[f"{r}_hd95"] for rec in recs for r in REGIONS], dtype=float)
    wt_vol = np.array(vols["wt"], dtype=float)
    wt_dsc = np.array([rec["wt_dsc"] for rec in recs], dtype=float)
    order = np.argsort(wt_vol)
    k = max(1, len(order) // 5)
    agg = {
        "n": len(recs),
        "components_mean": components_mean,
        "component_denominator": denominators,
        "dsc_p05": float(np.percentile(all_dsc, 5)),
        "hd95_p95": float(np.percentile(all_hd95, 95)),
        "smallest_volume_wt_dsc": float(np.mean(wt_dsc[order[:k]])),
        "empty_reference_fp_rate": float(np.mean([np.mean(fp_empty[r]) for r in REGIONS])),
        "missed_region_rate": float(np.mean([np.mean(missed[r]) for r in REGIONS])),
    }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)                 # PRIVATE per-subject (real ids) -- ignored
    ap.add_argument("--summary-out", default=None)          # SANITIZED aggregate -- committable
    # REQUIRED, no default: the frozen per-fold validation denominator (271 for fold 0, 270 for
    # folds 1-4). Omitting it must fail immediately rather than silently assuming a fold-0 count.
    ap.add_argument("--expected-n", type=int, required=True)
    args = ap.parse_args()

    preds = sorted(Path(args.preds).glob("*.nii.gz"))
    gt = Path(args.gt)
    if not preds:
        print(json.dumps({"error": "no_predictions"}))
        return 2

    import brats_evaluation as be
    ev = load_evaluator()
    records, errors, per_status, vols, fp_empty, missed = evaluate_all(
        preds, gt, be.evaluate_single_exam, ev)

    if errors:
        Path(args.out).write_text(json.dumps(
            {"errors": errors, "n_ok": len(records), "status_histogram": per_status},
            indent=2) + "\n")
        print(json.dumps({"error": "hard_failures", "n_errors": len(errors),
                          "n_ok": len(records)}))
        return 3

    agg = build_aggregate(records, vols, fp_empty, missed, args.expected_n)

    Path(args.out).write_text(json.dumps({
        "evaluator": "BraTS-evaluation==0.0.8", "config": "GoAT",
        "per_subject_records": records,           # PRIVATE (real ids) -- never committed/printed
        "errors": [], "status_histogram": per_status, "aggregate": agg}, indent=2) + "\n")

    if args.summary_out:                          # SANITIZED: aggregate + status counts only
        Path(args.summary_out).write_text(json.dumps({
            "evaluator": "BraTS-evaluation==0.0.8", "config": "GoAT",
            "n": agg["n"], "status_histogram": per_status,
            "components_mean": agg["components_mean"],
            "component_denominator": agg["component_denominator"],
            "dsc_p05": agg["dsc_p05"], "hd95_p95": agg["hd95_p95"],
            "smallest_volume_wt_dsc": agg["smallest_volume_wt_dsc"],
            "empty_reference_fp_rate": agg["empty_reference_fp_rate"],
            "missed_region_rate": agg["missed_region_rate"]}, indent=2) + "\n")

    print(json.dumps({"evaluated": len(records), "errors": 0, "aggregate_n": agg["n"],
                      "status_histogram": per_status}))     # sanitized counts only
    return 0


if __name__ == "__main__":
    sys.exit(main())
