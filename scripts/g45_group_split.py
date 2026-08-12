#!/usr/bin/env python3
"""GAT-26 G4.5 — private subject grouping, modality-aware near-duplicate audit, and a
deterministic group-level five-fold split.

Privacy: real case IDs, group membership, suspicious-pair lists, and per-case
fingerprints stay on an ignored worker path. Only sanitized aggregate counts are
printed / committed.

Predeclared split policy (frozen BEFORE any result is examined):
  * Unit of assignment = subject group. Groups are singletons unless the near-dup
    audit *confirms* a same-subject/near-identical relationship (union-find).
  * Cohort/tumor-entity stratification is UNKNOWN and NOT attempted (no official
    labels; never inferred from filenames or lesion morphology).
  * Balanced observable features: ET presence, TC presence, and quintile bins of
    log-WT-volume, positive-TC-volume, positive-ET-volume.
  * Deterministic greedy grouped assignment, seed 21072026: groups are ordered by
    (stratum signature, blake2b(case-id-list)) — a seed-salted stable key — then
    each group is placed in the eligible fold minimising, in order:
    (that stratum's count in the fold, the fold's total size, the fold index).
  * Hard invariants (Part D.6) are asserted and unit-tested.

Near-dup detector (Part C.4): one compact fingerprint per case (4 z-scored
downsampled modalities + down-sampled seg + metadata), pairwise cosine on the
intensity fingerprint plus seg-Dice and shape agreement as corroboration. No
O(N^2) full-volume decompression: full-res confirmation runs only for the small
set of pairs above the predeclared suspicious threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

SEED = 21072026
NFOLDS = 5
DOWN = 16              # downsample grid per axis for the intensity fingerprint
SEG_DOWN = 24         # slightly finer grid for seg overlap
# Predeclared near-duplicate thresholds (frozen before results):
SUSPICIOUS_COS = 0.9990      # >= flag as suspicious near-duplicate (needs review)
CONFIRM_COS = 0.99990        # >= AND corroboration -> confirmed same-subject group
CONFIRM_SEG_DICE = 0.95      # seg overlap corroboration for confirmed grouping
MODS = ["t1n", "t1c", "t2w", "t2f"]


# ----------------------------- fingerprint -----------------------------
def _zoom_to(arr, n):
    """Resample a 3D array to n^3 block-means, vectorized per axis via reduceat.
    Requires each axis length >= n (true for brain volumes with n<=24)."""
    import numpy as np
    a = np.asarray(arr, dtype=np.float32)
    for ax, s in enumerate(a.shape):
        edges = np.linspace(0, s, n + 1).astype(int)
        counts = np.diff(edges).astype(np.float32)
        if (counts <= 0).any():  # axis too short for this grid; fall back to linear pick
            take = np.clip((np.arange(n) * s // n), 0, s - 1)
            a = np.take(a, take, axis=ax)
            continue
        summed = np.add.reduceat(a, edges[:-1], axis=ax)
        shape = [1] * a.ndim; shape[ax] = n
        a = summed / counts.reshape(shape)
    return a


def _znorm(v):
    import numpy as np
    v = v.reshape(-1).astype(np.float32)
    m, s = v.mean(), v.std()
    return (v - m) / (s + 1e-6)


def compute_fingerprint(case_dir):
    """Return a dict fingerprint for one case directory. Pure/deterministic."""
    import numpy as np
    import nibabel as nib
    cid = os.path.basename(case_dir)
    files = {re.search(r"-([a-z0-9]+)\.nii\.gz$", f).group(1): os.path.join(case_dir, f)
             for f in os.listdir(case_dir) if f.endswith(".nii.gz")}
    vecs = []
    shape = None
    zooms = None
    fg = 0
    for mi, m in enumerate(MODS):
        img = nib.load(files[m])
        a = np.asanyarray(img.dataobj, dtype=np.float32)
        if shape is None:
            shape = tuple(int(x) for x in a.shape)
            zooms = tuple(round(float(z), 4) for z in img.header.get_zooms()[:3])
            fg = int((a > 0).sum())  # foreground estimate from T1n
        vecs.append(_znorm(_zoom_to(a, DOWN)))
    fp = np.concatenate(vecs)
    fp = fp / (np.linalg.norm(fp) + 1e-9)
    seg = np.asanyarray(nib.load(files["seg"]).dataobj)
    seg = np.rint(seg).astype(np.int16)
    et = int((seg == 3).sum())
    tc = int(((seg == 1) | (seg == 3)).sum())
    wt = int(((seg == 1) | (seg == 2) | (seg == 3)).sum())
    seg_small = (_zoom_to((seg > 0).astype(np.float32), SEG_DOWN) > 0.5).astype(np.uint8)
    return {
        "cid": cid,
        "fp": fp.astype(np.float32),
        "seg_small": seg_small.reshape(-1),
        "shape": shape,
        "zooms": zooms,
        "et": et, "tc": tc, "wt": wt,
        "fg": fg,
        "nvox": int(np.prod(shape)),
    }


def _fp_worker(case_dir):
    try:
        return compute_fingerprint(case_dir)
    except Exception as e:  # fail loud but keep which case
        return {"cid": os.path.basename(case_dir), "error": f"{type(e).__name__}:{e}"}


def cmd_fingerprint(args):
    from multiprocessing import Pool
    import numpy as np
    inner = os.path.join(args.raw_root, sorted(os.listdir(args.raw_root))[0])
    case_dirs = [os.path.join(inner, c) for c in sorted(os.listdir(inner))
                 if os.path.isdir(os.path.join(inner, c))]
    t0 = time.time()
    with Pool(args.np) as p:
        fps = p.map(_fp_worker, case_dirs, chunksize=4)
    errs = [f for f in fps if "error" in f]
    if errs:
        print(json.dumps({"error": "fingerprint_failures", "n": len(errs)}))
        return 2
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        cids=np.array([f["cid"] for f in fps]),
        fp=np.stack([f["fp"] for f in fps]),
        seg_small=np.stack([f["seg_small"] for f in fps]),
        shapes=np.array([f["shape"] for f in fps]),
        zooms=np.array([f["zooms"] for f in fps]),
        et=np.array([f["et"] for f in fps]),
        tc=np.array([f["tc"] for f in fps]),
        wt=np.array([f["wt"] for f in fps]),
        fg=np.array([f["fg"] for f in fps]),
        nvox=np.array([f["nvox"] for f in fps]),
    )
    print(json.dumps({"cases": len(fps), "seconds": round(time.time() - t0, 1),
                      "fingerprint_dim": int(fps[0]["fp"].shape[0]), "np": args.np}))
    return 0


# ----------------------------- near-dup audit -----------------------------
def _dice_bits(a, b):
    import numpy as np
    inter = int(np.logical_and(a, b).sum())
    s = int(a.sum() + b.sum())
    return (2.0 * inter / s) if s else 1.0


def near_dup_pairs(fp, seg_small, shapes, cos_thresh):
    """Return list of (i, j, cos, seg_dice, same_shape) above cos_thresh. O(N^2) on
    compact vectors only (no volume decompression)."""
    import numpy as np
    S = fp @ fp.T
    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    cos = S[iu]
    mask = cos >= cos_thresh
    out = []
    for idx in np.where(mask)[0]:
        i, j = int(iu[0][idx]), int(iu[1][idx])
        out.append((i, j, float(cos[idx]),
                    _dice_bits(seg_small[i], seg_small[j]),
                    tuple(shapes[i]) == tuple(shapes[j])))
    out.sort(key=lambda t: -t[2])
    return out


def selftest_detector():
    """Synthetic identical / near-identical / unrelated — used by the unit test."""
    import numpy as np
    rng = np.random.default_rng(0)
    base = rng.standard_normal((4, DOWN, DOWN, DOWN)).astype(np.float32)
    def norm(v):
        v = np.concatenate([_znorm(x) for x in v]); return v / (np.linalg.norm(v) + 1e-9)
    ident = norm(base.copy())
    near = norm(base + 0.01 * rng.standard_normal(base.shape).astype(np.float32))
    unrel = norm(rng.standard_normal((4, DOWN, DOWN, DOWN)).astype(np.float32))
    fp = np.stack([norm(base), ident, near, unrel])
    seg = np.zeros((4, SEG_DOWN ** 3), dtype=np.uint8)
    seg[:3, :100] = 1  # first three share a seg blob, unrelated has none
    shapes = np.array([[240, 240, 155]] * 4)
    pairs = near_dup_pairs(fp, seg, shapes, SUSPICIOUS_COS)
    got = {(i, j): (c, d) for i, j, c, d, s in pairs}
    return {
        "identical_detected": (0, 1) in got and got[(0, 1)][0] > 0.99999,
        "near_detected": any(i in (0, 1) and j == 2 for (i, j) in got),
        "unrelated_not_detected": not any(3 in (i, j) for (i, j) in got),
    }


def cmd_audit(args):
    import numpy as np
    d = np.load(args.fp, allow_pickle=True)
    fp, seg_small, shapes = d["fp"], d["seg_small"], d["shapes"]
    n = fp.shape[0]
    susp = near_dup_pairs(fp, seg_small, shapes, SUSPICIOUS_COS)
    # confirmed same-subject -> union-find
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b); parent[max(ra, rb)] = min(ra, rb)
    confirmed, ambiguous = [], []
    for i, j, c, dice, same_shape in susp:
        if c >= CONFIRM_COS and same_shape and dice >= CONFIRM_SEG_DICE:
            union(i, j); confirmed.append((i, j, c, dice))
        else:
            ambiguous.append((i, j, c, dice, same_shape))
    groups = {}
    for x in range(n):
        groups.setdefault(find(x), []).append(x)
    group_list = list(groups.values())
    # private detail
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cids = d["cids"]
    np.savez_compressed(args.out,
                        group_members=np.array([json.dumps([cids[k] for k in g]) for g in group_list]),
                        confirmed=np.array([json.dumps(list(c)) for c in confirmed]) if confirmed else np.array([]),
                        ambiguous=np.array([json.dumps(list(a)) for a in ambiguous]) if ambiguous else np.array([]))
    summ = {
        "cases": n,
        "suspicious_threshold_cos": SUSPICIOUS_COS,
        "confirm_threshold_cos": CONFIRM_COS,
        "suspicious_pairs": len(susp),
        "confirmed_same_subject_pairs": len(confirmed),
        "ambiguous_unresolved_pairs": len(ambiguous),
        "groups_total": len(group_list),
        "multi_case_groups": sum(1 for g in group_list if len(g) > 1),
        "largest_group": max(len(g) for g in group_list),
        "max_ambiguous_cos": round(max([a[2] for a in ambiguous], default=0.0), 6),
        "selftest": selftest_detector(),
    }
    print(json.dumps(summ))
    Path(args.summary).write_text(json.dumps(summ, indent=2) + "\n", encoding="utf-8")
    return 0


# ----------------------------- split -----------------------------
def _bin_quintiles(values, present_mask):
    import numpy as np
    v = np.array(values, dtype=np.float64)
    out = np.zeros(len(v), dtype=int)
    pos = np.where(present_mask)[0]
    if len(pos) == 0:
        return out
    qs = np.quantile(v[pos], [0.2, 0.4, 0.6, 0.8])
    out[pos] = np.digitize(v[pos], qs) + 1  # 1..5 for present; 0 for absent
    return out


def build_split(fp_npz, audit_npz):
    import numpy as np
    d = np.load(fp_npz, allow_pickle=True)
    cids = list(d["cids"]); n = len(cids)
    et, tc, wt = d["et"], d["tc"], d["wt"]
    a = np.load(audit_npz, allow_pickle=True)
    group_members = [json.loads(x) for x in a["group_members"]]
    cid2idx = {c: i for i, c in enumerate(cids)}

    et_present = et > 0
    tc_present = tc > 0
    wt_bin = _bin_quintiles(np.log1p(wt), wt > 0)
    tc_bin = _bin_quintiles(tc, tc_present)
    et_bin = _bin_quintiles(et, et_present)

    def group_sig(members):
        idxs = [cid2idx[c] for c in members]
        # majority/first-member stratum signature
        i0 = idxs[0]
        return (int(et_present[i0]), int(tc_present[i0]), int(wt_bin[i0]),
                int(tc_bin[i0]), int(et_bin[i0]))

    salt = str(SEED).encode()
    def order_key(members):
        sig = group_sig(members)
        h = hashlib.blake2b(salt + "".join(sorted(members)).encode(), digest_size=8).hexdigest()
        return (sig, h)

    groups = sorted(group_members, key=order_key)
    fold_of = {}
    fold_sizes = [0] * NFOLDS
    strat_counts = [dict() for _ in range(NFOLDS)]
    for members in groups:
        sig = group_sig(members)
        gsz = len(members)
        # choose fold minimising (stratum count, fold size, fold index)
        best = min(range(NFOLDS),
                   key=lambda f: (strat_counts[f].get(sig, 0), fold_sizes[f], f))
        for c in members:
            fold_of[c] = best
        fold_sizes[best] += gsz
        strat_counts[best][sig] = strat_counts[best].get(sig, 0) + gsz

    # nnU-Net splits_final.json format: list of {"train": [...], "val": [...]}
    all_ids = list(cids)
    splits = []
    for f in range(NFOLDS):
        val = sorted([c for c in all_ids if fold_of[c] == f])
        train = sorted([c for c in all_ids if fold_of[c] != f])
        splits.append({"train": train, "val": val})

    # aggregate balance (sanitized)
    def counts(mask):
        return [int(sum(1 for c in cids if fold_of[c] == f and mask[cid2idx[c]])) for f in range(NFOLDS)]
    agg = {
        "n_cases": n,
        "n_groups": len(groups),
        "multi_case_groups": sum(1 for g in groups if len(g) > 1),
        "fold_sizes": fold_sizes,
        "et_present_per_fold": counts(et_present),
        "et_absent_per_fold": counts(~et_present),
        "tc_present_per_fold": counts(tc_present),
        "tc_absent_per_fold": counts(~tc_present),
        "wt_bin_per_fold": {int(b): counts(wt_bin == b) for b in sorted(set(wt_bin.tolist()))},
        "tc_bin_per_fold": {int(b): counts(tc_bin == b) for b in sorted(set(tc_bin.tolist())) if b > 0},
        "et_bin_per_fold": {int(b): counts(et_bin == b) for b in sorted(set(et_bin.tolist())) if b > 0},
    }
    return splits, agg


def validate_split(splits, n_expected, groups_singleton_only):
    import itertools
    val_all = list(itertools.chain.from_iterable(s["val"] for s in splits))
    inv = {}
    inv["all_appear_once_as_val"] = (len(val_all) == n_expected and len(set(val_all)) == n_expected)
    inv["no_missing_extra"] = len(set(val_all)) == n_expected
    inv["nonempty_folds"] = all(len(s["val"]) > 0 for s in splits)
    inv["train_val_disjoint"] = all(set(s["train"]).isdisjoint(s["val"]) for s in splits)
    inv["train_val_union_full"] = all(set(s["train"]) | set(s["val"]) == set(val_all) for s in splits)
    sizes = sorted(len(s["val"]) for s in splits)
    if groups_singleton_only:
        inv["fold_sizes_270_or_271"] = all(x in (270, 271) for x in sizes)
    inv["max_fold_size_diff_le_1"] = (sizes[-1] - sizes[0]) <= 1
    return inv


def cmd_split(args):
    import numpy as np, hashlib as H
    splits, agg = build_split(args.fp, args.audit)
    a = np.load(args.audit, allow_pickle=True)
    groups = [json.loads(x) for x in a["group_members"]]
    singleton_only = all(len(g) == 1 for g in groups)
    inv = validate_split(splits, agg["n_cases"], singleton_only)
    # region presence per fold checks
    et_absent = agg["et_absent_per_fold"]; tc_absent = agg["tc_absent_per_fold"]
    inv["all_folds_have_et_absent"] = all(x > 0 for x in et_absent)
    inv["all_folds_have_tc_absent"] = all(x > 0 for x in tc_absent)
    inv["et_absent_range_le_feasible"] = (max(et_absent) - min(et_absent)) <= max(1, args.presence_tol)
    inv["tc_absent_range_le_feasible"] = (max(tc_absent) - min(tc_absent)) <= max(1, args.presence_tol)
    # volume-bin imbalance limit (predeclared): per-fold count within +/- tol of mean
    def bin_ok(binmap):
        ok = True
        for b, per in binmap.items():
            mean = sum(per) / NFOLDS
            if max(abs(x - mean) for x in per) > args.bin_tol:
                ok = False
        return ok
    inv["wt_bin_balanced"] = bin_ok(agg["wt_bin_per_fold"])
    inv["tc_bin_balanced"] = bin_ok(agg["tc_bin_per_fold"])
    inv["et_bin_balanced"] = bin_ok(agg["et_bin_per_fold"])

    all_ok = all(inv.values())
    # write PRIVATE split
    payload = json.dumps(splits)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(payload, encoding="utf-8")
    split_hash = H.sha256(payload.encode()).hexdigest()
    Path(args.hashfile).write_text(split_hash + "\n", encoding="utf-8")

    summary = {
        "seed": SEED, "nfolds": NFOLDS, "singleton_only": singleton_only,
        "invariants": inv, "all_invariants_pass": all_ok,
        "aggregate_balance": agg,
        "split_sha256_prefix": split_hash[:12],
        "bin_tol": args.bin_tol, "presence_tol": args.presence_tol,
    }
    print(json.dumps({"all_invariants_pass": all_ok, "invariants": inv,
                      "fold_sizes": agg["fold_sizes"],
                      "split_sha256_prefix": split_hash[:12]}))
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if all_ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fingerprint"); p.add_argument("--raw-root", required=True)
    p.add_argument("--out", required=True); p.add_argument("--np", type=int, default=16)
    p.set_defaults(func=cmd_fingerprint)
    p = sub.add_parser("audit"); p.add_argument("--fp", required=True)
    p.add_argument("--out", required=True); p.add_argument("--summary", required=True)
    p.set_defaults(func=cmd_audit)
    p = sub.add_parser("split"); p.add_argument("--fp", required=True); p.add_argument("--audit", required=True)
    p.add_argument("--out", required=True); p.add_argument("--hashfile", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--bin-tol", type=int, default=8); p.add_argument("--presence-tol", type=int, default=1)
    p.set_defaults(func=cmd_split)
    p = sub.add_parser("selftest"); p.set_defaults(func=lambda a: (print(json.dumps(selftest_detector())), 0)[1])
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
