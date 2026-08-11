#!/usr/bin/env python3
"""G82 evaluation: official and common-support utilities for one candidate policy.

Baseline and candidate probabilities live in the same uint8/bbox store, so the
only difference between the two policies is the network weights (and, for a
blend, the mixing weight alpha). Reconstruction is the unchanged C0 rule:
threshold 0.5 with ET subset TC subset WT enforced. No threshold search, no
component filtering, no presence gate.

Reports, per §G:
  U_official  organizer-compatible NaN skipping with the realized denominators
  U_common    every component recomputed on the per-case finite intersection
              shared by C0 and the candidate
plus denominators, NaN-pattern changes, region-wise and deduplicated
unique-subject zero-DSC counts, lesion TP/FP/FN, positive-volume quartiles,
bottom-decile utility, fold-wise deltas and a subject-level paired bootstrap.
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
QMAX = 255.0
TAU_RECON = 0.5

DATA = os.environ["G82_RAW"]
BASE_STORE = os.environ["G82_BASE_STORE"]      # G81 C0 out-of-fold store root
CAND_STORE = os.environ["G82_CAND_STORE"]      # candidate store root
SCRIPTS = os.environ.get("G82_SCRIPTS", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)


def _eval_mod():
    import g79v_tau_nsd_adapter as A
    sp = A.eval_site_packages()
    if sp and sp not in sys.path:
        sys.path.insert(0, sp)
    return A


def read_seg(cid):
    import SimpleITK as sitk
    p = os.path.join(DATA, cid, f"{cid}-seg.nii.gz")
    return np.rint(sitk.GetArrayFromImage(sitk.ReadImage(p))).astype(np.int16)


def dequant_full(path):
    """(3, Z, Y, X) float32 probabilities in channel order WT, TC, ET.

    Voxels outside the stored bounding box had every projected region probability
    below 0.05 and are restored as zero, which cannot change a 0.5 threshold.
    """
    z = np.load(path)
    q = z["q"]
    b = [int(x) for x in z["bbox"]]
    shape = tuple(int(x) for x in z["shape"])
    out = np.zeros((3,) + shape, dtype=np.float32)
    out[:, b[0]:b[1], b[2]:b[3], b[4]:b[5]] = q.astype(np.float32) / QMAX
    return out


def reconstruct(p):
    """Unchanged C0 hierarchy-safe reconstruction at threshold 0.5."""
    wt, tc, et = p[0] >= TAU_RECON, p[1] >= TAU_RECON, p[2] >= TAU_RECON
    tc = tc | et
    wt = wt | tc
    seg = np.zeros(wt.shape, dtype=np.uint8)
    seg[wt] = 2
    seg[tc] = 1
    seg[et] = 3
    return seg


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
    cid, fold, alpha, want_baseline = args
    A = _eval_mod()
    seg_ref = read_seg(cid)
    base_npz = os.path.join(BASE_STORE, f"f{fold}", "probs", f"{cid}.npz")
    cand_npz = os.path.join(CAND_STORE, f"f{fold}", "probs", f"{cid}.npz")

    out = {"case": cid, "fold": fold,
           "gt_sizes": {r: int(np.isin(seg_ref, GT_LABELS[r]).sum()) for r in REGIONS}}

    pc = dequant_full(cand_npz)
    if alpha < 1.0:
        pb = dequant_full(base_npz)
        pc = (1.0 - alpha) * pb + alpha * pc
        del pb
    seg_c = reconstruct(pc)
    del pc
    c05, c10 = A.region_components(seg_ref, seg_c, 0.5), A.region_components(seg_ref, seg_c, 1.0)
    out["cand_t05"] = {f"{r}_{m}": c05[(r, m)] for r, m in COMPONENTS}
    out["cand_t10"] = {f"{r}_{m}": c10[(r, m)] for r, m in COMPONENTS}
    out["cand_lesion"] = {r: list(lesion_counts(seg_ref, seg_c, r)) for r in REGIONS}

    if want_baseline:
        seg_b = reconstruct(dequant_full(base_npz))
        b05 = A.region_components(seg_ref, seg_b, 0.5)
        b10 = A.region_components(seg_ref, seg_b, 1.0)
        out["base_t05"] = {f"{r}_{m}": b05[(r, m)] for r, m in COMPONENTS}
        out["base_t10"] = {f"{r}_{m}": b10[(r, m)] for r, m in COMPONENTS}
        out["base_lesion"] = {r: list(lesion_counts(seg_ref, seg_b, r)) for r in REGIONS}
    return out


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #

def _finite(x):
    return x is not None and isinstance(x, float) and math.isfinite(x)


def aggregate(records, prefix, tau_key, subjects=None):
    """Organizer-compatible: NaN components are skipped and leave the denominator."""
    means, denom = {}, {}
    for r, m in COMPONENTS:
        vals = [rec[f"{prefix}_{tau_key}"][f"{r}_{m}"] for rec in records
                if subjects is None or rec["case"] in subjects]
        good = [v for v in vals if _finite(v)]
        means[f"{r}_{m}"] = float(np.mean(good)) if good else float("nan")
        denom[f"{r}_{m}"] = len(good)
    u = float(np.mean([means[f"{r}_{m}"] for r, m in COMPONENTS]))
    return means, denom, u


def common_support(records, tau_key):
    """Cases where every component is finite under BOTH policies."""
    keep = set()
    for rec in records:
        b, c = rec[f"base_{tau_key}"], rec[f"cand_{tau_key}"]
        if all(_finite(b[f"{r}_{m}"]) and _finite(c[f"{r}_{m}"]) for r, m in COMPONENTS):
            keep.add(rec["case"])
    return keep


def per_case_utility(records, prefix, tau_key, subjects):
    out = {}
    for rec in records:
        if rec["case"] not in subjects:
            continue
        d = rec[f"{prefix}_{tau_key}"]
        out[rec["case"]] = float(np.mean([d[f"{r}_{m}"] for r, m in COMPONENTS]))
    return out


def zero_dsc(records, prefix, tau_key, subjects):
    """Region-case count and the deduplicated unique-subject count."""
    region_cases, subs = 0, set()
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
        if hit:
            subs.add(rec["case"])
    return region_cases, len(subs)


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
    """Subject-level paired bootstrap on the per-case utility difference."""
    subs = sorted(subjects)
    d = np.array([uc[s] - ub[s] for s in subs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(subs), size=(n, len(subs)))
    means = d[idx].mean(axis=1)
    return {
        "point": float(d.mean()),
        "prob_positive": float((means > 0).mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def bottom_decile(u, subjects):
    v = np.array([u[s] for s in sorted(subjects)])
    return float(v[v <= np.quantile(v, 0.10)].mean())


def volume_strata(records, subjects):
    """Positive-WT-volume quartile membership, for the stratum-regression gate."""
    vols = {rec["case"]: rec["gt_sizes"]["WT"] for rec in records if rec["case"] in subjects}
    pos = np.array([v for v in vols.values() if v > 0], dtype=np.float64)
    if pos.size < 4:
        return {}
    q = np.quantile(pos, [0.25, 0.5, 0.75])
    out = {f"Q{i}": set() for i in range(1, 5)}
    for cid, v in vols.items():
        if v <= 0:
            continue
        k = 1 + int(v > q[0]) + int(v > q[1]) + int(v > q[2])
        out[f"Q{k}"].add(cid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nproc", type=int, default=40)
    ap.add_argument("--baseline-cache", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    folds = [int(x) for x in a.folds.split(",")]
    splits = json.load(open(a.splits))
    jobs = []
    for f in folds:
        for cid in splits[f]["val"]:
            if os.path.exists(os.path.join(CAND_STORE, f"f{f}", "probs", f"{cid}.npz")):
                jobs.append((cid, f))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"[{a.tag}] evaluating {len(jobs)} cases over folds {folds}, alpha={a.alpha}",
          flush=True)

    cache = {}
    if a.baseline_cache and os.path.exists(a.baseline_cache):
        cache = {r["case"]: r for r in json.load(open(a.baseline_cache))}
        print(f"  reusing cached baseline evaluation for {len(cache)} cases", flush=True)

    args = [(cid, f, a.alpha, cid not in cache) for cid, f in jobs]
    with ProcessPoolExecutor(a.nproc) as ex:
        recs = list(ex.map(_one, args, chunksize=1))
    for rec in recs:
        if rec["case"] in cache:
            for k in ("base_t05", "base_t10", "base_lesion"):
                rec[k] = cache[rec["case"]][k]

    if a.baseline_cache and not os.path.exists(a.baseline_cache):
        tmp = a.baseline_cache + ".tmp"
        with open(tmp, "w") as f:
            json.dump([{ "case": r["case"], "base_t05": r["base_t05"],
                         "base_t10": r["base_t10"], "base_lesion": r["base_lesion"]}
                       for r in recs], f)
        os.replace(tmp, a.baseline_cache)

    result = {"schema": "gat26.g82.eval.v1", "tag": a.tag, "alpha": a.alpha,
              "folds": folds, "n_cases": len(recs)}

    for tau_key, tau in (("t10", 1.0), ("t05", 0.5)):
        allsub = {r["case"] for r in recs}
        bm, bd, bu = aggregate(recs, "base", tau_key)
        cm, cd, cu = aggregate(recs, "cand", tau_key)
        common = common_support(recs, tau_key)
        bmc, bdc, buc = aggregate(recs, "base", tau_key, common)
        cmc, cdc, cuc = aggregate(recs, "cand", tau_key, common)

        ub = per_case_utility(recs, "base", tau_key, common)
        uc = per_case_utility(recs, "cand", tau_key, common)
        boot = paired_bootstrap(common, ub, uc)

        fold_deltas = {}
        for f in folds:
            sub = {r["case"] for r in recs if r["fold"] == f} & common
            if sub:
                fold_deltas[str(f)] = float(
                    np.mean([uc[s] for s in sub]) - np.mean([ub[s] for s in sub]))

        strata = volume_strata(recs, common)
        strat_delta = {}
        for name, members in strata.items():
            if len(members) >= 30:
                strat_delta[name] = {
                    "n": len(members),
                    "delta": float(np.mean([uc[s] for s in members])
                                   - np.mean([ub[s] for s in members]))}

        zb_rc, zb_su = zero_dsc(recs, "base", tau_key, allsub)
        zc_rc, zc_su = zero_dsc(recs, "cand", tau_key, allsub)
        et_zb = sum(1 for r in recs if _finite(r[f"base_{tau_key}"]["ET_DSC"])
                    and r[f"base_{tau_key}"]["ET_DSC"] == 0.0)
        et_zc = sum(1 for r in recs if _finite(r[f"cand_{tau_key}"]["ET_DSC"])
                    and r[f"cand_{tau_key}"]["ET_DSC"] == 0.0)

        result[tau_key] = {
            "tau": tau,
            "official": {"baseline_means": bm, "candidate_means": cm,
                         "baseline_denominators": bd, "candidate_denominators": cd,
                         "U_baseline": bu, "U_candidate": cu, "delta_U": cu - bu,
                         "denominator_changes": {k: cd[k] - bd[k] for k in bd}},
            "common_support": {
                "n_subjects": len(common),
                "baseline_means": bmc, "candidate_means": cmc,
                "denominators": bdc,
                "U_baseline": buc, "U_candidate": cuc, "delta_U": cuc - buc,
                "component_deltas": {f"{r}_{m}": cmc[f"{r}_{m}"] - bmc[f"{r}_{m}"]
                                     for r, m in COMPONENTS},
            },
            "bootstrap": boot,
            "fold_deltas": fold_deltas,
            "volume_stratum_deltas": strat_delta,
            "bottom_decile": {"baseline": bottom_decile(ub, common),
                              "candidate": bottom_decile(uc, common)},
            "zero_dsc": {"baseline_region_cases": zb_rc, "candidate_region_cases": zc_rc,
                         "baseline_unique_subjects": zb_su,
                         "candidate_unique_subjects": zc_su,
                         "baseline_et": et_zb, "candidate_et": et_zc},
        }

    allsub = {r["case"] for r in recs}
    btp, bfp, bfn = lesion_totals(recs, "base", allsub)
    ctp, cfp, cfn = lesion_totals(recs, "cand", allsub)
    result["lesion"] = {"baseline": {"TP": btp, "FP": bfp, "FN": bfn},
                        "candidate": {"TP": ctp, "FP": cfp, "FN": cfn},
                        "fn_increase_fraction": (cfn - bfn) / bfn if bfn else 0.0,
                        "fp_increase_fraction": (cfp - bfp) / bfp if bfp else 0.0}
    result["evaluator_errors"] = 0

    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, a.out)

    t1, t05 = result["t10"], result["t05"]
    print(f"[{a.tag}] dU_common tau1={t1['common_support']['delta_U']:+.6f} "
          f"tau0.5={t05['common_support']['delta_U']:+.6f} | "
          f"dU_official tau1={t1['official']['delta_U']:+.6f} | "
          f"P(dU>0)={t1['bootstrap']['prob_positive']:.4f} | "
          f"zeroDSC {t1['zero_dsc']['baseline_region_cases']}->"
          f"{t1['zero_dsc']['candidate_region_cases']} | "
          f"lesion FN {bfn}->{cfn} FP {bfp}->{cfp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
