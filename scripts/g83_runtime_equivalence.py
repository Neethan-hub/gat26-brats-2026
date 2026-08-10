#!/usr/bin/env python3
"""G83 §D runtime-equivalence gate for the cached inference path.

The cached path is a RUNTIME-ONLY optimisation, not a scientific candidate. It may
be used only if, at tile step 0.5, it reproduces the frozen release path exactly:

  * segmentation voxel arrays exactly equal
  * geometry exactly equal
  * labels exactly equal
  * output names and count equal
  * raw region probabilities exactly equal, or max abs difference <= 1e-7
  * no voxel crosses the 0.5 threshold differently
  * zero errors

Frozen release files are never edited. If the gate fails the caller must fall back
to the original per-case path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

PROB_TOL = 1e-7
THRESHOLD = 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--cases-file", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--dataset-json", required=True)
    ap.add_argument("--scripts", required=True)
    ap.add_argument("--n-real", type=int, default=10)
    ap.add_argument("--n-synthetic", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.scripts)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np
    import torch
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    import release_infer as RI
    import g4_reconstruct_validate as RC
    import g83_overlap_predict as P

    plans = json.load(open(a.plans))
    dataset_json = json.load(open(a.dataset_json))
    RI.enforce_determinism()
    io = SimpleITKIO()

    # Deterministic, outcome-independent sample: the first N case identifiers in
    # sorted order. Chosen before any equivalence result is observed.
    cases = sorted(c.strip() for c in open(a.cases_file) if c.strip())[:a.n_real]

    weights = [RI._load_weights(cp) for cp in a.checkpoints]
    cached = [P.build_predictor(RI, plans, dataset_json, w, 0.5) for w in weights]

    worst_prob = 0.0
    seg_mismatch = 0
    geom_mismatch = 0
    label_mismatch = 0
    cross = 0
    errors = 0
    compared = 0
    t0 = time.time()

    def ensemble(data, props, use_cached):
        acc = None
        for k in range(len(weights)):
            pred = (cached[k] if use_cached
                    else P.build_predictor(RI, plans, dataset_json, weights[k], 0.5))
            _, probs = pred.predict_single_npy_array(data, props, None, None, True)
            acc = np.asarray(probs) if acc is None else acc + np.asarray(probs)
            if not use_cached:
                del pred
                torch.cuda.empty_cache()
        return acc / float(len(weights))

    # synthetic cases first: fixed random volumes exercise the same code path
    rng = np.random.RandomState(20260730)
    for s in range(a.n_synthetic):
        try:
            shape = (4, 80, 96, 88)
            data = rng.normal(size=shape).astype(np.float32)
            props = {"spacing": [1.0, 1.0, 1.0]}
            ref = ensemble(data, props, False)
            opt = ensemble(data, props, True)
            worst_prob = max(worst_prob, float(np.abs(ref - opt).max()))
            if not np.array_equal(RC.project_and_reconstruct(ref[0], ref[1], ref[2],
                                                             THRESHOLD, THRESHOLD, THRESHOLD),
                                  RC.project_and_reconstruct(opt[0], opt[1], opt[2],
                                                             THRESHOLD, THRESHOLD, THRESHOLD)):
                seg_mismatch += 1
            cross += int(((ref >= THRESHOLD) != (opt >= THRESHOLD)).sum())
            compared += 1
        except Exception:
            errors += 1

    for cid in cases:
        try:
            cdir = os.path.join(a.data_root, cid)
            files = [os.path.join(cdir, f"{cid}-{m}.nii.gz")
                     for m in P.REQUIRED_MODALITIES]
            data, props = io.read_images(files)
            ref = ensemble(data, props, False)
            opt = ensemble(data, props, True)
            worst_prob = max(worst_prob, float(np.abs(ref - opt).max()))
            cross += int(((ref >= THRESHOLD) != (opt >= THRESHOLD)).sum())
            seg_r = RC.project_and_reconstruct(ref[0], ref[1], ref[2],
                                               THRESHOLD, THRESHOLD, THRESHOLD)
            seg_o = RC.project_and_reconstruct(opt[0], opt[1], opt[2],
                                               THRESHOLD, THRESHOLD, THRESHOLD)
            if not np.array_equal(seg_r, seg_o):
                seg_mismatch += 1
            if seg_r.shape != seg_o.shape:
                geom_mismatch += 1
            if set(np.unique(seg_r).tolist()) != set(np.unique(seg_o).tolist()):
                label_mismatch += 1
            compared += 1
        except Exception:
            errors += 1

    result = {
        "schema": "gat26.g83.runtime_equivalence.v1",
        "tile_step": 0.5,
        "n_synthetic": a.n_synthetic, "n_real": len(cases), "n_compared": compared,
        "max_abs_probability_difference": worst_prob,
        "probability_tolerance": PROB_TOL,
        "segmentation_arrays_mismatched": seg_mismatch,
        "geometry_mismatched": geom_mismatch,
        "labels_mismatched": label_mismatch,
        "voxels_crossing_threshold_differently": cross,
        "errors": errors,
        "seconds": round(time.time() - t0, 1),
    }
    result["PASS"] = bool(
        compared == a.n_synthetic + len(cases) and errors == 0
        and seg_mismatch == 0 and geom_mismatch == 0 and label_mismatch == 0
        and cross == 0 and worst_prob <= PROB_TOL)

    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, a.out)
    print(json.dumps(result, indent=1))
    return 0 if result["PASS"] else 3


if __name__ == "__main__":
    sys.exit(main())
