#!/usr/bin/env python3
"""G83 evaluation: C0 (tile step 0.5) versus D25 (tile step 0.25).

Reuses the G79-VR-verified metric path and the G82 common-support evaluator
verbatim -- Panoptica is never replaced, site-packages is never modified, and no
distance transform is reimplemented. This module adds only what G83 needs:

  * an explicit, never-implicit NSD tolerance on every call;
  * an independent second aggregate recomputation that must agree to 1e-12;
  * a structural refusal to read confirmation folds before the freeze commit.
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

import g82_eval as E  # noqa: E402  (proven G82 evaluator: reconstruction + components)

REGIONS = E.REGIONS
COMPONENTS = E.COMPONENTS
TAUS = (1.0, 0.5)
LOCKED_FOLDS = (3, 4)
RECOMPUTE_TOL = 1e-12


class SealedFoldError(RuntimeError):
    pass


def assert_folds_allowed(folds, freeze_artifact: str) -> None:
    """Confirmation folds are unreadable until the candidate-freeze commit exists."""
    locked = sorted(set(int(f) for f in folds) & set(LOCKED_FOLDS))
    if locked and not os.path.exists(freeze_artifact):
        raise SealedFoldError(
            f"folds {locked} are sealed until the candidate-freeze artifact exists")


def independent_recompute(records, prefix, tau_key, subjects=None):
    """A deliberately different implementation of the same aggregate.

    Uses running Kahan-compensated sums rather than numpy means, so an error in
    either path shows up as a disagreement rather than cancelling out.
    """
    means, denom = {}, {}
    for r, m in COMPONENTS:
        key = f"{r}_{m}"
        total = 0.0
        comp = 0.0
        n = 0
        for rec in records:
            if subjects is not None and rec["case"] not in subjects:
                continue
            v = rec[f"{prefix}_{tau_key}"][key]
            if v is None or (isinstance(v, float) and math.isnan(v)):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nproc", type=int, default=40)
    ap.add_argument("--baseline-cache", default="")
    ap.add_argument("--freeze-artifact",
                    default=os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), "artifacts",
                        "g83_candidate_freeze.json"))
    a = ap.parse_args()
    if a.nproc > 48:
        raise SystemExit("refusing more than the 48 evaluator processes proven stable in G82")

    folds = [int(x) for x in a.folds.split(",")]
    assert_folds_allowed(folds, a.freeze_artifact)

    splits = json.load(open(a.splits))
    jobs = []
    for f in folds:
        for cid in splits[f]["val"]:
            if os.path.exists(os.path.join(E.CAND_STORE, f"f{f}", "probs", f"{cid}.npz")):
                jobs.append((cid, f))
    print(f"[{a.tag}] evaluating {len(jobs)} cases over folds {folds}", flush=True)

    cache = {}
    if a.baseline_cache and os.path.exists(a.baseline_cache):
        cache = {r["case"]: r for r in json.load(open(a.baseline_cache))}
        print(f"  reusing verified baseline records for {len(cache)} cases", flush=True)

    args = [(cid, f, 1.0, cid not in cache) for cid, f in jobs]
    with ProcessPoolExecutor(a.nproc) as ex:
        recs = list(ex.map(E._one, args, chunksize=1))
    for rec in recs:
        if rec["case"] in cache:
            for k in ("base_t05", "base_t10", "base_lesion"):
                rec[k] = cache[rec["case"]][k]

    result = {"schema": "gat26.g83.eval.v1", "tag": a.tag, "folds": folds,
              "n_cases": len(recs), "policies": {"baseline": "C0 tile_step 0.5",
                                                 "candidate": "D25 tile_step 0.25"},
              "recompute_tolerance": RECOMPUTE_TOL}

    worst_recompute = 0.0
    for tau_key, tau in (("t10", 1.0), ("t05", 0.5)):
        allsub = {r["case"] for r in recs}
        bm, bd, bu = E.aggregate(recs, "base", tau_key)
        cm_, cd, cu = E.aggregate(recs, "cand", tau_key)
        common = E.common_support(recs, tau_key)
        bmc, bdc, buc = E.aggregate(recs, "base", tau_key, common)
        cmc, cdc, cuc = E.aggregate(recs, "cand", tau_key, common)

        # independent second implementation must agree to 1e-12
        for prefix, ref_means, ref_u, subj in (("base", bm, bu, None), ("cand", cm_, cu, None),
                                               ("base", bmc, buc, common), ("cand", cmc, cuc, common)):
            im, idn, iu = independent_recompute(recs, prefix, tau_key, subj)
            for k in ref_means:
                if not (math.isnan(ref_means[k]) and math.isnan(im[k])):
                    worst_recompute = max(worst_recompute, abs(ref_means[k] - im[k]))
            worst_recompute = max(worst_recompute, abs(ref_u - iu))

        ub = E.per_case_utility(recs, "base", tau_key, common)
        uc = E.per_case_utility(recs, "cand", tau_key, common)
        boot = E.paired_bootstrap(common, ub, uc)

        fold_deltas = {}
        for f in folds:
            sub = {r["case"] for r in recs if r["fold"] == f} & common
            if sub:
                fold_deltas[str(f)] = float(np.mean([uc[s] for s in sub])
                                            - np.mean([ub[s] for s in sub]))

        zb_rc, zb_su = E.zero_dsc(recs, "base", tau_key, allsub)
        zc_rc, zc_su = E.zero_dsc(recs, "cand", tau_key, allsub)
        et_zb = sum(1 for r in recs if E._finite(r[f"base_{tau_key}"]["ET_DSC"])
                    and r[f"base_{tau_key}"]["ET_DSC"] == 0.0)
        et_zc = sum(1 for r in recs if E._finite(r[f"cand_{tau_key}"]["ET_DSC"])
                    and r[f"cand_{tau_key}"]["ET_DSC"] == 0.0)

        result[tau_key] = {
            "tau": tau,
            "official": {"baseline_means": bm, "candidate_means": cm_,
                         "baseline_denominators": bd, "candidate_denominators": cd,
                         "U_baseline": bu, "U_candidate": cu, "delta_U": cu - bu,
                         "denominator_changes": {k: cd[k] - bd[k] for k in bd}},
            "common_support": {
                "n_subjects": len(common), "baseline_means": bmc, "candidate_means": cmc,
                "denominators": bdc, "U_baseline": buc, "U_candidate": cuc,
                "delta_U": cuc - buc,
                "component_deltas": {f"{r}_{m}": cmc[f"{r}_{m}"] - bmc[f"{r}_{m}"]
                                     for r, m in COMPONENTS}},
            "bootstrap": boot,
            "fold_deltas": fold_deltas,
            "zero_dsc": {"baseline_region_cases": zb_rc, "candidate_region_cases": zc_rc,
                         "baseline_unique_subjects": zb_su,
                         "candidate_unique_subjects": zc_su,
                         "baseline_et": et_zb, "candidate_et": et_zc},
            "nan_pattern_changes": {
                k: cd[k] - bd[k] for k in bd if cd[k] != bd[k]} or "none",
        }

    allsub = {r["case"] for r in recs}
    btp, bfp, bfn = E.lesion_totals(recs, "base", allsub)
    ctp, cfp, cfn = E.lesion_totals(recs, "cand", allsub)
    result["lesion"] = {"baseline": {"TP": btp, "FP": bfp, "FN": bfn},
                        "candidate": {"TP": ctp, "FP": cfp, "FN": cfn},
                        "fn_increase_fraction": (cfn - bfn) / bfn if bfn else 0.0,
                        "fp_increase_fraction": (cfp - bfp) / bfp if bfp else 0.0}
    result["evaluator_errors"] = 0
    result["independent_recompute_max_abs_difference"] = worst_recompute
    result["independent_recompute_agrees"] = worst_recompute <= RECOMPUTE_TOL

    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, a.out)

    t1, t05 = result["t10"], result["t05"]
    print(f"[{a.tag}] dU_common tau1={t1['common_support']['delta_U']:+.6f} "
          f"tau0.5={t05['common_support']['delta_U']:+.6f} | "
          f"dU_official tau1={t1['official']['delta_U']:+.6f} "
          f"tau0.5={t05['official']['delta_U']:+.6f} | "
          f"P={t1['bootstrap']['prob_positive']:.4f} "
          f"CI=[{t1['bootstrap']['ci95'][0]:+.6f},{t1['bootstrap']['ci95'][1]:+.6f}] | "
          f"recompute_agrees={result['independent_recompute_agrees']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
