#!/usr/bin/env python3
"""Seed the C0 baseline evaluation cache from the preserved out-of-fold records.

The baseline's six official components at tau=0.5 and tau=1.0 were already
computed, case by case, with the same preserved official evaluator and the same
frozen C0 inference path. Recomputing them would cost ~42 s of panoptica time per
case for no new information, so they are reused directly. Only the baseline lesion
TP/FP/FN counts are computed here, because those were not stored.

Before reusing anything, a random sample is recomputed from scratch and required to
agree exactly; otherwise the seeding is refused.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import g82_eval as E  # noqa: E402

REGIONS = E.REGIONS
COMPONENTS = E.COMPONENTS


def record_path(fold, cid):
    return os.path.join(E.BASE_STORE, f"f{fold}", "records", f"{cid}.json")


def _lesion_only(args):
    cid, fold = args
    seg_ref = E.read_seg(cid)
    seg_b = E.reconstruct(E.dequant_full(
        os.path.join(E.BASE_STORE, f"f{fold}", "probs", f"{cid}.npz")))
    return cid, {r: list(E.lesion_counts(seg_ref, seg_b, r)) for r in REGIONS}


def _fresh_components(args):
    cid, fold = args
    A = E._eval_mod()
    seg_ref = E.read_seg(cid)
    seg_b = E.reconstruct(E.dequant_full(
        os.path.join(E.BASE_STORE, f"f{fold}", "probs", f"{cid}.npz")))
    c05 = A.region_components(seg_ref, seg_b, 0.5)
    c10 = A.region_components(seg_ref, seg_b, 1.0)
    return cid, ({f"{r}_{m}": c05[(r, m)] for r, m in COMPONENTS},
                 {f"{r}_{m}": c10[(r, m)] for r, m in COMPONENTS})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nproc", type=int, default=48)
    ap.add_argument("--validate", type=int, default=16,
                    help="how many cases to recompute from scratch and compare")
    a = ap.parse_args()

    folds = [int(x) for x in a.folds.split(",")]
    splits = json.load(open(a.splits))
    pairs = []
    for f in folds:
        for cid in splits[f]["val"]:
            if os.path.exists(record_path(f, cid)):
                pairs.append((cid, f))
    print(f"seeding baseline for {len(pairs)} cases over folds {folds}", flush=True)

    # 1. validate the stored components against a fresh recomputation
    rnd = random.Random(20260730)
    sample = rnd.sample(pairs, min(a.validate, len(pairs)))
    with ProcessPoolExecutor(min(a.nproc, len(sample))) as ex:
        fresh = dict((cid, v) for cid, v in ex.map(_fresh_components, sample))
    worst = 0.0
    mismatched_nan = 0
    for cid, fold in sample:
        rec = json.load(open(record_path(fold, cid)))
        stored = (rec["baseline"], rec["baseline_tau1"])
        for s, f_ in zip(stored, fresh[cid]):
            for k in s:
                sv, fv = s[k], f_[k]
                if sv is None or fv is None or (isinstance(sv, float) and sv != sv) \
                        or (isinstance(fv, float) and fv != fv):
                    if bool(sv is None or (isinstance(sv, float) and sv != sv)) != \
                       bool(fv is None or (isinstance(fv, float) and fv != fv)):
                        mismatched_nan += 1
                    continue
                worst = max(worst, abs(float(sv) - float(fv)))
    print(f"VALIDATION on {len(sample)} cases: max abs component difference {worst:.3e}, "
          f"NaN-pattern mismatches {mismatched_nan}", flush=True)
    if worst > 1e-12 or mismatched_nan:
        print("REFUSING to seed: stored baseline components do not reproduce exactly")
        return 3

    # 2. lesion counts (not stored anywhere) for every case
    with ProcessPoolExecutor(a.nproc) as ex:
        lesions = dict(ex.map(_lesion_only, pairs, chunksize=2))

    out = []
    for cid, fold in pairs:
        rec = json.load(open(record_path(fold, cid)))
        out.append({"case": cid,
                    "base_t05": rec["baseline"],
                    "base_t10": rec["baseline_tau1"],
                    "base_lesion": lesions[cid]})
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, a.out)
    tp = sum(sum(v[r][0] for r in REGIONS) for v in lesions.values())
    fp = sum(sum(v[r][1] for r in REGIONS) for v in lesions.values())
    fn = sum(sum(v[r][2] for r in REGIONS) for v in lesions.values())
    print(f"SEEDED {len(out)} baseline records | lesion TP={tp} FP={fp} FN={fn}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
