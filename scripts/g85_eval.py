#!/usr/bin/env python3
"""G85 confirmation evaluation: frozen C0 versus frozen M8 on a given fold set.

Unlike G84's evaluator, which read its baseline from a preserved calibration cache,
this module scores BOTH policies from segmentations generated fresh in this stage
through the exact raw-NIfTI release path. That is required because no C0 baseline
cache exists for the confirmation folds.

Nothing about either policy is decided here; both are frozen by
artifacts/g85_candidate_freeze.json. This module only measures.

Reported per tolerance (explicitly 0.5 and 1.0, never implicit):
official skip-NaN aggregates with realized denominators, common-support aggregates,
per-fold deltas, a subject-level paired bootstrap, zero-DSC tails, and the corrected
reference-component miss-rate noninferiority audit from g85_lesion_audit.
An independent Kahan-summed recomputation must agree to 1e-12.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REGIONS = ("ET", "TC", "WT")
METRICS = ("DSC", "NSD")
COMPONENTS = [(r, m) for m in METRICS for r in REGIONS]
LOCKED_FOLDS = (3, 4)
RECOMPUTE_TOL = 1e-12
MAX_PROCS = 48

RAW = os.environ["G85_RAW"]
STORE_C0 = os.environ["G85_STORE_C0"]
STORE_M8 = os.environ["G85_STORE_M8"]
# Repository-relative by construction (see g84_eval.py). G85_SCRIPTS remains an explicit override.
# Inserted at the END so it can never shadow this module's own directory, which line 29 put first.
SCRIPTS = os.environ.get("G85_SCRIPTS") or os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPTS)


class SealedFoldError(RuntimeError):
    pass


def assert_folds_allowed(folds, freeze_artifact: str) -> None:
    """Confirmation folds stay unreadable until the committed freeze artifact exists."""
    locked = sorted(set(int(f) for f in folds) & set(LOCKED_FOLDS))
    if locked and not os.path.exists(freeze_artifact):
        raise SealedFoldError(
            f"folds {locked} are sealed until the candidate-freeze artifact exists")


def _eval_mod():
    import g79v_tau_nsd_adapter as A
    sp = A.eval_site_packages()
    if sp and sp not in sys.path:
        sys.path.insert(0, sp)
    return A


def read_gt(cid):
    import SimpleITK as sitk
    p = os.path.join(RAW, cid, f"{cid}-seg.nii.gz")
    return np.rint(sitk.GetArrayFromImage(sitk.ReadImage(p))).astype(np.int16)


def read_seg(store, cid, fold):
    return np.asarray(np.load(os.path.join(store, f"f{fold}", "seg", f"{cid}.npz"))["seg"]
                      ).astype(np.int16)


def _one(args):
    cid, fold = args
    A = _eval_mod()
    import g85_lesion_audit as LA
    ref = read_gt(cid)
    out = {"case": cid, "fold": fold}
    for key, store in (("C0", STORE_C0), ("M8", STORE_M8)):
        pred = read_seg(store, cid, fold)
        if ref.shape != pred.shape:
            raise RuntimeError("geometry mismatch between prediction and reference")
        c05 = A.region_components(ref, pred, 0.5)
        c10 = A.region_components(ref, pred, 1.0)
        out[f"{key}_t05"] = {f"{r}_{m}": c05[(r, m)] for r, m in COMPONENTS}
        out[f"{key}_t10"] = {f"{r}_{m}": c10[(r, m)] for r, m in COMPONENTS}
        out[key] = LA.per_case_stats(ref, pred)
    return out


def _finite(x):
    return x is not None and isinstance(x, float) and math.isfinite(x)


def aggregate(records, prefix, tau_key, subjects=None):
    means, denom = {}, {}
    for r, m in COMPONENTS:
        vals = [rec[f"{prefix}_{tau_key}"][f"{r}_{m}"] for rec in records
                if subjects is None or rec["case"] in subjects]
        good = [v for v in vals if _finite(v)]
        means[f"{r}_{m}"] = float(np.mean(good)) if good else float("nan")
        denom[f"{r}_{m}"] = len(good)
    return means, denom, float(np.mean([means[f"{r}_{m}"] for r, m in COMPONENTS]))


def independent_recompute(records, prefix, tau_key, subjects=None):
    means = {}
    for r, m in COMPONENTS:
        key = f"{r}_{m}"
        total = comp = 0.0
        n = 0
        for rec in records:
            if subjects is not None and rec["case"] not in subjects:
                continue
            v = rec[f"{prefix}_{tau_key}"][key]
            if not _finite(v):
                continue
            y = v - comp
            t = total + y
            comp = (t - total) - y
            total = t
            n += 1
        means[key] = (total / n) if n else float("nan")
    u = sum(means[f"{r}_{m}"] for r, m in COMPONENTS) / len(COMPONENTS)
    return means, u


def common_support(records, tau_key):
    keep = set()
    for rec in records:
        b, c = rec[f"C0_{tau_key}"], rec[f"M8_{tau_key}"]
        if all(_finite(b[f"{r}_{m}"]) and _finite(c[f"{r}_{m}"]) for r, m in COMPONENTS):
            keep.add(rec["case"])
    return keep


def per_case_utility(records, prefix, tau_key, subjects):
    return {rec["case"]: float(np.mean([rec[f"{prefix}_{tau_key}"][f"{r}_{m}"]
                                        for r, m in COMPONENTS]))
            for rec in records if rec["case"] in subjects}


def zero_dsc(records, prefix, tau_key, subjects):
    region_cases, subs, et = 0, set(), 0
    for rec in records:
        if rec["case"] not in subjects:
            continue
        d = rec[f"{prefix}_{tau_key}"]
        hit = False
        for r in REGIONS:
            v = d[f"{r}_DSC"]
            if _finite(v) and v == 0.0:
                region_cases += 1
                hit = True
                if r == "ET":
                    et += 1
        if hit:
            subs.add(rec["case"])
    return region_cases, len(subs), et


def paired_bootstrap(subjects, ub, uc, seed=20260730, n=10000):
    subs = sorted(subjects)
    d = np.array([uc[s] - ub[s] for s in subs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(subs), size=(n, len(subs)))
    means = d[idx].mean(axis=1)
    return {"point": float(d.mean()), "prob_positive": float((means > 0).mean()),
            "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]}


def build_result(recs, folds, tag):
    import g85_lesion_audit as LA
    result = {"schema": "gat26.g85.eval.v1", "tag": tag, "folds": folds,
              "n_cases": len(recs),
              "policies": {"baseline": "C0 use_mirroring=False",
                           "candidate": "M8 use_mirroring=True axes (0,1,2)"}}
    worst_rc = 0.0
    for tau_key, tau in (("t10", 1.0), ("t05", 0.5)):
        allsub = {r["case"] for r in recs}
        bm, bd, bu = aggregate(recs, "C0", tau_key)
        cm_, cd, cu = aggregate(recs, "M8", tau_key)
        common = common_support(recs, tau_key)
        bmc, bdc, buc = aggregate(recs, "C0", tau_key, common)
        cmc, cdc, cuc = aggregate(recs, "M8", tau_key, common)
        for prefix, ref_m, ref_u, subj in (("C0", bm, bu, None), ("M8", cm_, cu, None),
                                          ("C0", bmc, buc, common), ("M8", cmc, cuc, common)):
            im, iu = independent_recompute(recs, prefix, tau_key, subj)
            for k in ref_m:
                if not (math.isnan(ref_m[k]) and math.isnan(im[k])):
                    worst_rc = max(worst_rc, abs(ref_m[k] - im[k]))
            worst_rc = max(worst_rc, abs(ref_u - iu))
        ub = per_case_utility(recs, "C0", tau_key, common)
        uc = per_case_utility(recs, "M8", tau_key, common)
        fold_deltas = {}
        for f in folds:
            sub = {r["case"] for r in recs if r["fold"] == f} & common
            if sub:
                fold_deltas[str(f)] = float(np.mean([uc[s] for s in sub])
                                            - np.mean([ub[s] for s in sub]))
        zb = zero_dsc(recs, "C0", tau_key, allsub)
        zc = zero_dsc(recs, "M8", tau_key, allsub)
        result[tau_key] = {
            "tau": tau,
            "official": {"baseline_means": bm, "candidate_means": cm_,
                         "baseline_denominators": bd, "candidate_denominators": cd,
                         "U_baseline": bu, "U_candidate": cu, "delta_U": cu - bu,
                         "denominator_changes": {k: cd[k] - bd[k] for k in bd}},
            "common_support": {"n_subjects": len(common), "baseline_means": bmc,
                               "candidate_means": cmc, "denominators": bdc,
                               "U_baseline": buc, "U_candidate": cuc,
                               "delta_U": cuc - buc,
                               "component_deltas": {f"{r}_{m}": cmc[f"{r}_{m}"] - bmc[f"{r}_{m}"]
                                                    for r, m in COMPONENTS}},
            "bootstrap": paired_bootstrap(common, ub, uc),
            "fold_deltas": fold_deltas,
            "zero_dsc": {"baseline_region_cases": zb[0], "candidate_region_cases": zc[0],
                         "baseline_unique_subjects": zb[1],
                         "candidate_unique_subjects": zc[1],
                         "baseline_et": zb[2], "candidate_et": zc[2]},
            "nan_pattern_changes": {k: cd[k] - bd[k] for k in bd if cd[k] != bd[k]} or "none",
        }
    result["lesion_noninferiority"] = LA.noninferiority(recs, "C0", "M8", LA.MARGINS_FROZEN)
    result["evaluator_errors"] = 0
    result["independent_recompute_max_abs_difference"] = worst_rc
    result["independent_recompute_agrees"] = worst_rc <= RECOMPUTE_TOL
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--records-out", default="", help="private per-case records (never committed)")
    ap.add_argument("--nproc", type=int, default=44)
    ap.add_argument("--freeze-artifact", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts", "g85_candidate_freeze.json"))
    a = ap.parse_args()
    if a.nproc > MAX_PROCS:
        raise SystemExit(f"refusing more than {MAX_PROCS} evaluator processes")

    folds = [int(x) for x in a.folds.split(",")]
    assert_folds_allowed(folds, a.freeze_artifact)

    splits = json.load(open(a.splits, encoding="utf-8"))
    jobs, missing = [], []
    for f in folds:
        for cid in splits[f]["val"]:
            if all(os.path.exists(os.path.join(s, f"f{f}", "seg", f"{cid}.npz"))
                   for s in (STORE_C0, STORE_M8)):
                jobs.append((cid, f))
            else:
                missing.append(cid)
    expected = sum(len(splits[f]["val"]) for f in folds)
    print(f"[{a.tag}] evaluating {len(jobs)} of {expected} expected cases, folds {folds}"
          + (f" | MISSING {len(missing)}" if missing else ""), flush=True)

    with ProcessPoolExecutor(a.nproc) as ex:
        recs = list(ex.map(_one, jobs, chunksize=1))

    result = build_result(recs, folds, a.tag)
    result["n_expected"] = expected
    result["exact_membership"] = len(recs) == expected

    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, a.out)
    if a.records_out:
        with open(a.records_out, "w", encoding="utf-8") as f:
            json.dump(recs, f)

    t1, t05 = result["t10"], result["t05"]
    ni = result["lesion_noninferiority"]
    print(f"[{a.tag}] dU_common tau1={t1['common_support']['delta_U']:+.6f} "
          f"tau0.5={t05['common_support']['delta_U']:+.6f} | official "
          f"tau1={t1['official']['delta_U']:+.6f} tau0.5={t05['official']['delta_U']:+.6f} | "
          f"P={t1['bootstrap']['prob_positive']:.4f} "
          f"CI=[{t1['bootstrap']['ci95'][0]:+.6f},{t1['bootstrap']['ci95'][1]:+.6f}]", flush=True)
    print(f"[{a.tag}] miss-rate delta={ni['bootstrap']['point_delta']:+.6f} "
          f"upper95={ni['bootstrap']['upper_95_one_sided']:+.6f} "
          f"noninferior={ni['passes']} | recompute={result['independent_recompute_agrees']}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
