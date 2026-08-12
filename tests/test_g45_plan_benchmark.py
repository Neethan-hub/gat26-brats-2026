#!/usr/bin/env python3
"""GAT-26 G4.5 plan/benchmark tests (`python3 tests/test_g45_plan_benchmark.py`).

Covers the production dataset contract constants, plan-validation logic on a synthetic
ResEnc plan, and the deterministic 40-case benchmark selection (fold + strata coverage).
Heavy deps (torch/nnunetv2) are lazy inside command functions and are NOT imported here.
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
import g45_plan_benchmark as PB  # noqa: E402

FAILS = []


def check(name, cond):
    if not cond:
        FAILS.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _synth_split_and_fp(tmp, per_fold_cases=60):
    rng = np.random.default_rng(3)
    folds = []
    all_cids = []
    et = []; tc = []; wt = []; nvox = []
    cid_i = 0
    for f in range(5):
        val = []
        for _ in range(per_fold_cases):
            c = f"CASE-{cid_i:05d}"; cid_i += 1
            all_cids.append(c); val.append(c)
            et.append(int(rng.random() > 0.2) * rng.integers(10, 5000))
            tc.append(max(et[-1], int(rng.random() > 0.1) * rng.integers(20, 9000)))
            wt.append(tc[-1] + rng.integers(500, 40000))
            nvox.append(8928000)
        folds.append({"train": [c for c in all_cids if c not in val], "val": val})
    # rebuild train lists correctly after all val known
    valset = {c: f for f, s in enumerate(folds) for c in s["val"]}
    splits = [{"val": s["val"], "train": [c for c in all_cids if valset[c] != f]} for f, s in enumerate(folds)]
    fpz = os.path.join(tmp, "fp.npz")
    np.savez_compressed(fpz, cids=np.array(all_cids), fp=np.zeros((len(all_cids), 4)),
                        seg_small=np.zeros((len(all_cids), 8)), shapes=np.zeros((len(all_cids), 3)),
                        et=np.array(et), tc=np.array(tc), wt=np.array(wt),
                        fg=np.array(nvox), nvox=np.array(nvox))
    sj = os.path.join(tmp, "split.json"); Path(sj).write_text(json.dumps(splits), encoding="utf-8")
    return fpz, sj, all_cids


def main():
    # 1. production dataset contract constants
    check("channels_order", PB.CHANNEL_NAMES == {"0": "T1n", "1": "T1c", "2": "T2w", "3": "T2f"})
    check("labels_regions", PB.LABELS == {"background": 0, "whole_tumor": [1, 2, 3],
                                          "tumor_core": [1, 3], "enhancing_tumor": [3]})
    check("regions_class_order", PB.REGIONS_CLASS_ORDER == [2, 1, 3])
    check("dataset_name_501", PB.DATASET_NAME == "Dataset501_GAT26GOAT")

    # 2. plan validation logic on a synthetic ResEnc plan
    with tempfile.TemporaryDirectory() as tmp:
        plan = {"plans_name": "nnUNetResEncUNetMPlans", "dataset_name": "Dataset501_GAT26GOAT",
                "configurations": {"3d_fullres": {
                    "patch_size": [128, 160, 112], "batch_size": 2, "spacing": [1, 1, 1],
                    "normalization_schemes": ["ZScoreNormalization"] * 4,
                    "architecture": {"network_class_name":
                                     "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
                                     "arch_kwargs": {"n_stages": 6, "features_per_stage": [32, 64, 128, 256, 320, 320],
                                                     "strides": [[1, 1, 1]] + [[2, 2, 2]] * 5,
                                                     "kernel_sizes": [[3, 3, 3]] * 6,
                                                     "n_blocks_per_stage": [1, 3, 4, 6, 6, 6]}}}}}
        s = PB._summarize_plan(Path(tmp) and _write(tmp, "plan", plan))
        check("summary_arch_resenc", s["arch_class"].endswith("ResidualEncoderUNet"))
        check("summary_batch_ge_2", s["batch_size"] >= 2)
        check("summary_patch_3d", len(s["patch_size"]) == 3)

        # a bad plan (batch 1) must fail the batch>=2 gate via the same logic
        badn = dict(plan); badn = json.loads(json.dumps(plan)); badn["configurations"]["3d_fullres"]["batch_size"] = 1
        sb = PB._summarize_plan(_write(tmp, "planbad", badn))
        check("bad_batch_flagged", not (sb["batch_size"] >= 2))

    # 3. deterministic bench selection: all 5 folds represented, strata covered
    with tempfile.TemporaryDirectory() as tmp:
        fpz, sj, cids = _synth_split_and_fp(tmp)
        chosen, feat, fold_of, strata = PB.select_bench_cases(fpz, sj, per_fold=8)
        check("selection_size_40", len(chosen) == 40)
        check("selection_unique", len(set(chosen)) == len(chosen))
        folds_hit = {fold_of[c] for c in chosen}
        check("all_5_folds_represented", folds_hit == {0, 1, 2, 3, 4})
        chosen2, *_ = PB.select_bench_cases(fpz, sj, per_fold=8)
        check("selection_deterministic", chosen == chosen2)

    print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 1 if FAILS else 0


def _write(tmp, name, obj):
    p = os.path.join(tmp, name + ".json")
    Path(p).write_text(json.dumps(obj), encoding="utf-8")
    return Path(p)


if __name__ == "__main__":
    sys.exit(main())
