#!/usr/bin/env python3
"""GAT-26 G4.5 grouping/split tests (`python3 tests/test_g45_group_split.py`).

Covers the split invariants (duplicate/omitted IDs, train/val overlap, group leakage,
rare-region imbalance), determinism (byte-identical double run), near-dup detector, and
identifier non-leakage into sanitized summaries. Uses a small synthetic fingerprint set.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import g45_group_split as GS  # noqa: E402

FAILS = []


def check(name, cond):
    if not cond:
        FAILS.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _synth(tmp, n=60, planted_dup=True):
    rng = np.random.default_rng(7)
    cids = np.array([f"CASE-{i:05d}" for i in range(n)])
    fp = rng.standard_normal((n, GS.DOWN ** 3 * 4)).astype(np.float32)
    fp /= np.linalg.norm(fp, axis=1, keepdims=True) + 1e-9
    seg_small = (rng.random((n, GS.SEG_DOWN ** 3)) > 0.7).astype(np.uint8)
    shapes = np.array([[240, 240, 155]] * n)
    et = (rng.random(n) > 0.15).astype(int) * rng.integers(50, 5000, n)   # ~15% ET-absent
    tc = np.maximum(et, (rng.random(n) > 0.08).astype(int) * rng.integers(80, 9000, n))
    wt = tc + rng.integers(1000, 30000, n)
    if planted_dup:
        fp[1] = fp[0]; seg_small[1] = seg_small[0]; shapes[1] = shapes[0]  # exact near-dup 0==1
    fpz = os.path.join(tmp, "fp.npz")
    np.savez_compressed(fpz, cids=cids, fp=fp, seg_small=seg_small, shapes=shapes,
                        et=et, tc=tc, wt=wt, fg=np.full(n, 1000), nvox=np.full(n, 8928000))
    return fpz, cids


def main():
    check("detector_selftest", all(GS.selftest_detector().values()))
    with tempfile.TemporaryDirectory() as tmp:
        fpz, cids = _synth(tmp)
        import argparse
        az = os.path.join(tmp, "groups.npz"); asum = os.path.join(tmp, "audit.json")
        GS.cmd_audit(argparse.Namespace(fp=fpz, out=az, summary=asum))
        a = json.load(open(asum, encoding="utf-8"))
        check("planted_dup_detected", a["confirmed_same_subject_pairs"] >= 1)
        check("no_unresolved_ambiguous", a["ambiguous_unresolved_pairs"] == 0)

        splits, agg = GS.build_split(fpz, az)
        n = len(cids)
        inv = GS.validate_split(splits, n, groups_singleton_only=False)
        check("all_once_as_val", inv["all_appear_once_as_val"])
        check("no_missing_extra", inv["no_missing_extra"])
        check("train_val_disjoint", inv["train_val_disjoint"])
        check("nonempty_folds", inv["nonempty_folds"])
        check("max_fold_diff_le_1_or_group", inv["max_fold_size_diff_le_1"])

        # group leakage: planted dup pair (CASE-00000, CASE-00001) must share a fold
        dup = {"CASE-00000", "CASE-00001"}
        val_fold = {c: f for f, s in enumerate(splits) for c in s["val"]}
        check("planted_group_no_fold_crossing", val_fold["CASE-00000"] == val_fold["CASE-00001"])

        # determinism: byte-identical second build
        s1 = json.dumps(GS.build_split(fpz, az)[0])
        s2 = json.dumps(GS.build_split(fpz, az)[0])
        check("deterministic_double_run", s1 == s2)

        # rare-region: every fold has >=1 ET-absent and >=1 TC-absent where feasible
        check("all_folds_et_absent", all(x > 0 for x in agg["et_absent_per_fold"]))

        # identifier non-leakage: sanitized aggregate must not contain raw case ids
        blob = json.dumps(agg)
        check("no_case_ids_in_aggregate", not any(str(c) in blob for c in cids))

    # regression: committed governance/artifact text must not overclaim "distinct subjects"
    import re
    overclaim = re.compile(r"distinct\s+(singleton\s+)?subject|confirmed\s+distinct", re.I)
    offenders = []
    for rel in ("RUN_STATE.json", "DECISIONS.md", "RELEASE_CHECKLIST.md", "DATA_PROVENANCE.md",
                "artifacts/G45_PRETRAINING_AUDIT.md"):
        p = REPO / rel
        if p.exists() and overclaim.search(p.read_text(encoding="utf-8")):
            offenders.append(rel)
    check("no_distinct_subject_overclaim_in_committed_text", not offenders)

    print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
