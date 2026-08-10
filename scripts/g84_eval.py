#!/usr/bin/env python3
"""G84 evaluation: C0 (no mirroring) versus M8 (mirroring axes 0,1,2).

Candidate segmentations are read directly from private per-case storage, so no
probability quantisation is involved. Baseline per-case components come from the
preserved full-precision no-TTA release cache, which §D requires to reproduce the
committed release-path targets exactly before this module is used.

Uses the preserved Panoptica 2.1.4 / BraTS-evaluation 0.0.8 path with explicit NSD
tolerances; site-packages is never modified and no distance transform is
reimplemented. An independent Kahan-summed recomputation must agree to 1e-12.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REGIONS = ("ET", "TC", "WT")
METRICS = ("DSC", "NSD")
COMPONENTS = [(r, m) for m in METRICS for r in REGIONS]
GT_LABELS = {"ET": (3,), "TC": (1, 3), "WT": (1, 2, 3)}
LOCKED_FOLDS = (3, 4)
RECOMPUTE_TOL = 1e-12
MAX_PROCS = 48

RAW = os.environ["G84_RAW"]
CAND = os.environ["G84_CAND_STORE"]
SCRIPTS = os.environ.get("G84_SCRIPTS", "/workspace/brats-2026-gat26/scripts")
sys.path.insert(0, SCRIPTS)


class SealedFoldError(RuntimeError):
    pass


def assert_folds_allowed(folds, freeze_artifact: str) -> None:
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


def read_seg_gt(cid):
    import SimpleITK as sitk
    p = os.path.join(RAW, cid, f"{cid}-seg.nii.gz")
    return np.rint(sitk.GetArrayFromImage(sitk.ReadImage(p))).astype(np.int16)


def read_candidate(cid, fold):
    return np.asarray(np.load(os.path.join(CAND, f"f{fold}", "seg", f"{cid}.npz"))["seg"]
                      ).astype(np.int16)


def lesion_counts(seg_ref, seg_pred, region):
    from scipy import ndimage
    st = np.ones((3, 3, 3), bool)
    labs = GT_LABELS[region]
    ref, pred = np.isin(seg_ref, labs), np.isin(seg_pred, labs)
    lr, nr = ndimage.label(ref, structure=st)
    lp, npd = ndimage.label(pred, structure=st)
    tp = fp = fn = 0
    for i in range(1, npd + 1):
        if ref[lp == i].any():
            tp += 1
        else:
            fp += 1
    for i in range(1, nr + 1):
        if not pred[lr == i].any():
            fn += 1
    return tp, fp, fn


def _one(args):
    cid, fold = args
    A = _eval_mod()
    ref = read_seg_gt(cid)
    pred = read_candidate(cid, fold)
    if ref.shape != pred.shape:
        raise RuntimeError("geometry mismatch between candidate and ground truth")
    c05 = A.region_components(ref, pred, 0.5)
    c10 = A.region_components(ref, pred, 1.0)
    return {"case": cid, "fold": fold,
            "cand_t05": {f"{r}_{m}": c05[(r, m)] for r, m in COMPONENTS},
            "cand_t10": {f"{r}_{m}": c10[(r, m)] for r, m in COMPONENTS},
            "cand_lesion": {r: list(lesion_counts(ref, pred, r)) for r in REGIONS}}


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
    """Kahan-compensated second implementation; must agree with aggregate() to 1e-12."""
    means, denom = {}, {}
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
        denom[key] = n
    u = sum(means[f"{r}_{m}"] for r, m in COMPONENTS) / len(COMPONENTS)
    return means, denom, u


def common_support(records, tau_key):
    keep = set()
    for rec in records:
        b, c = rec[f"base_{tau_key}"], rec[f"cand_{tau_key}"]
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


def lesion_totals(records, prefix, subjects):
    tp = fp = fn = 0
    for rec in records:
        if rec["case"] not in subjects:
            continue
        for r in REGIONS:
            a, b, c = rec[f"{prefix}_lesion"][r]
            tp += a
            fp += b
            fn += c
    return tp, fp, fn


def paired_bootstrap(subjects, ub, uc, seed=20260730, n=10000):
    subs = sorted(subjects)
    d = np.array([uc[s] - ub[s] for s in subs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(subs), size=(n, len(subs)))
    means = d[idx].mean(axis=1)
    return {"point": float(d.mean()), "prob_positive": float((means > 0).mean()),
            "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--baseline-cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nproc", type=int, default=44)
    ap.add_argument("--freeze-artifact", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts", "g84_candidate_freeze.json"))
    a = ap.parse_args()
    if a.nproc > MAX_PROCS:
        raise SystemExit(f"refusing more than {MAX_PROCS} evaluator processes")

    folds = [int(x) for x in a.folds.split(",")]
    assert_folds_allowed(folds, a.freeze_artifact)

    splits = json.load(open(a.splits))
    jobs = [(cid, f) for f in folds for cid in splits[f]["val"]
            if os.path.exists(os.path.join(CAND, f"f{f}", "seg", f"{cid}.npz"))]
    expected = sum(len(splits[f]["val"]) for f in folds)
    print(f"[{a.tag}] evaluating {len(jobs)} of {expected} expected cases, folds {folds}",
          flush=True)

    with ProcessPoolExecutor(a.nproc) as ex:
        recs = list(ex.map(_one, jobs, chunksize=1))

    cache = {r["case"]: r for r in json.load(open(a.baseline_cache))}
    missing = [r["case"] for r in recs if r["case"] not in cache]
    if missing:
        raise SystemExit(f"{len(missing)} candidate cases have no baseline record")
    for rec in recs:
        for k in ("base_t05", "base_t10", "base_lesion"):
            rec[k] = cache[rec["case"]][k]

    result = {"schema": "gat64.g84.eval.v1", "tag": a.tag, "folds": folds,
              "n_cases": len(recs), "n_expected": expected,
              "exact_membership": len(recs) == expected,
              "policies": {"baseline": "C0 use_mirroring=False",
                           "candidate": "M8 use_mirroring=True axes (0,1,2)"}}

    worst_rc = 0.0
    for tau_key, tau in (("t10", 1.0), ("t05", 0.5)):
        allsub = {r["case"] for r in recs}
        bm, bd, bu = aggregate(recs, "base", tau_key)
        cm_, cd, cu = aggregate(recs, "cand", tau_key)
        common = common_support(recs, tau_key)
        bmc, bdc, buc = aggregate(recs, "base", tau_key, common)
        cmc, cdc, cuc = aggregate(recs, "cand", tau_key, common)
        for prefix, ref_m, ref_u, subj in (("base", bm, bu, None), ("cand", cm_, cu, None),
                                           ("base", bmc, buc, common),
                                           ("cand", cmc, cuc, common)):
            im, _, iu = independent_recompute(recs, prefix, tau_key, subj)
            for k in ref_m:
                if not (math.isnan(ref_m[k]) and math.isnan(im[k])):
                    worst_rc = max(worst_rc, abs(ref_m[k] - im[k]))
            worst_rc = max(worst_rc, abs(ref_u - iu))

        ub = per_case_utility(recs, "base", tau_key, common)
        uc = per_case_utility(recs, "cand", tau_key, common)
        boot = paired_bootstrap(common, ub, uc)
        fold_deltas = {}
        for f in folds:
            sub = {r["case"] for r in recs if r["fold"] == f} & common
            if sub:
                fold_deltas[str(f)] = float(np.mean([uc[s] for s in sub])
                                            - np.mean([ub[s] for s in sub]))
        zb = zero_dsc(recs, "base", tau_key, allsub)
        zc = zero_dsc(recs, "cand", tau_key, allsub)

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
            "bootstrap": boot, "fold_deltas": fold_deltas,
            "zero_dsc": {"baseline_region_cases": zb[0], "candidate_region_cases": zc[0],
                         "baseline_unique_subjects": zb[1],
                         "candidate_unique_subjects": zc[1],
                         "baseline_et": zb[2], "candidate_et": zc[2]},
            "nan_pattern_changes": {k: cd[k] - bd[k] for k in bd if cd[k] != bd[k]} or "none",
        }

    allsub = {r["case"] for r in recs}
    btp, bfp, bfn = lesion_totals(recs, "base", allsub)
    ctp, cfp, cfn = lesion_totals(recs, "cand", allsub)
    result["lesion"] = {"baseline": {"TP": btp, "FP": bfp, "FN": bfn},
                        "candidate": {"TP": ctp, "FP": cfp, "FN": cfn},
                        "fn_increase_fraction": (cfn - bfn) / bfn if bfn else 0.0,
                        "fp_increase_fraction": (cfp - bfp) / bfp if bfp else 0.0}
    result["evaluator_errors"] = 0
    result["independent_recompute_max_abs_difference"] = worst_rc
    result["independent_recompute_agrees"] = worst_rc <= RECOMPUTE_TOL

    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, a.out)

    t1, t05 = result["t10"], result["t05"]
    print(f"[{a.tag}] dU_common tau1={t1['common_support']['delta_U']:+.6f} "
          f"tau0.5={t05['common_support']['delta_U']:+.6f} | official "
          f"tau1={t1['official']['delta_U']:+.6f} tau0.5={t05['official']['delta_U']:+.6f} | "
          f"P={t1['bootstrap']['prob_positive']:.4f} "
          f"CI=[{t1['bootstrap']['ci95'][0]:+.6f},{t1['bootstrap']['ci95'][1]:+.6f}] | "
          f"recompute={result['independent_recompute_agrees']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
