#!/usr/bin/env python3
"""Derive T's subject-level strata from training-partition labels only.

Strata (all defined *inside* each fold's own training partition, never using a
held-out case and never using any validation or test data):

  small_et    ET volume in the lowest positive-volume quartile
  small_tc    TC volume in the lowest positive-volume quartile
  multifocal  WT component count >= the partition's 75th percentile and >= 2
  et_absent   exactly zero ET voxels
  wt_extreme  WT volume in the bottom or top decile

Emits one JSON per fold mapping identifier -> list of stratum names. No case
identifier is ever committed; the output lives outside the repository.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys

import numpy as np

RAW = os.environ["G82_RAW"]                     # raw label tree (private, never committed)
SPLITS = os.environ["G82_SPLITS"]               # nnU-Net splits_final.json
OUT = os.environ["G82_STRATA_OUT"]              # private output directory

# label convention {0,1,2,3}; ET=[3], TC=[1,3], WT=[1,2,3]
ET, TC, WT = (3,), (1, 3), (1, 2, 3)


def measure(ident: str):
    import nibabel as nib
    from scipy import ndimage
    seg = np.asarray(nib.load(f"{RAW}/{ident}/{ident}-seg.nii.gz").dataobj).astype(np.uint8)
    et = int(np.isin(seg, ET).sum())
    tc = int(np.isin(seg, TC).sum())
    wt_mask = np.isin(seg, WT)
    wt = int(wt_mask.sum())
    ncomp = 0
    if wt:
        structure = np.ones((3, 3, 3), dtype=np.uint8)          # 26-connectivity
        _, ncomp = ndimage.label(wt_mask, structure=structure)
    return ident, et, tc, wt, int(ncomp)


def strata_for_partition(rows: dict) -> dict:
    """rows: identifier -> (et, tc, wt, ncomp). Returns identifier -> [strata]."""
    idents = sorted(rows)
    et = np.array([rows[i][0] for i in idents], dtype=np.float64)
    tc = np.array([rows[i][1] for i in idents], dtype=np.float64)
    wt = np.array([rows[i][2] for i in idents], dtype=np.float64)
    nc = np.array([rows[i][3] for i in idents], dtype=np.float64)

    out = {i: [] for i in idents}

    pos_et = et[et > 0]
    if pos_et.size:
        q1 = np.quantile(pos_et, 0.25)
        for k, i in enumerate(idents):
            if 0 < et[k] <= q1:
                out[i].append("small_et")
    pos_tc = tc[tc > 0]
    if pos_tc.size:
        q1 = np.quantile(pos_tc, 0.25)
        for k, i in enumerate(idents):
            if 0 < tc[k] <= q1:
                out[i].append("small_tc")

    p75 = np.quantile(nc, 0.75)
    for k, i in enumerate(idents):
        if nc[k] >= max(p75, 2.0) and nc[k] >= 2:
            out[i].append("multifocal")

    for k, i in enumerate(idents):
        if et[k] == 0:
            out[i].append("et_absent")

    pos_wt = wt[wt > 0]
    if pos_wt.size:
        lo, hi = np.quantile(pos_wt, 0.10), np.quantile(pos_wt, 0.90)
        for k, i in enumerate(idents):
            if wt[k] > 0 and (wt[k] <= lo or wt[k] >= hi):
                out[i].append("wt_extreme")
    return out


def main(folds) -> int:
    splits = json.load(open(SPLITS))
    need = sorted({i for f in folds for i in splits[f]["train"]})
    print(f"measuring {len(need)} training labels for folds {folds}", flush=True)
    with mp.Pool(min(32, os.cpu_count() or 8)) as pool:
        res = pool.map(measure, need, chunksize=8)
    rows = {r[0]: (r[1], r[2], r[3], r[4]) for r in res}

    os.makedirs(OUT, exist_ok=True)
    for f in folds:
        part = {i: rows[i] for i in splits[f]["train"]}
        st = strata_for_partition(part)
        path = f"{OUT}/fold{f}.json"
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(st, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        counts = {}
        for v in st.values():
            for s in v:
                counts[s] = counts.get(s, 0) + 1
        print(f"fold {f}: n={len(st)} strata sizes {dict(sorted(counts.items()))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main([int(x) for x in (sys.argv[1:] or ["0", "1", "2"])]))
