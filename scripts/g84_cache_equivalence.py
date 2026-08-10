#!/usr/bin/env python3
"""G84 §F predictor-cache equivalence gate, run separately for C0 and M8.

The cache is a runtime-only optimisation and never a scientific candidate. For the
policy under test it must reproduce the original per-case rebuild/reload path
exactly on real calibration cases with complete real image properties:

  * identical checkpoint and accumulation order
  * max raw probability difference <= 1e-7
  * no voxel crossing threshold 0.5 differently
  * exact segmentation arrays, geometry, labels, names and counts
  * zero errors

Synthetic cases may supplement but never replace the real-case gate. Tolerances are
fixed here and are never relaxed after a result is observed.
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
    ap.add_argument("--policy", required=True, choices=("C0", "M8"))
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--cases-file", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--dataset-json", required=True)
    ap.add_argument("--scripts", required=True)
    ap.add_argument("--n-real", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.scripts)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    import release_infer as RI
    import g4_reconstruct_validate as RC
    import g84_tta_predict as P

    plans = json.load(open(a.plans))
    dataset_json = json.load(open(a.dataset_json))
    RI.enforce_determinism()
    io = SimpleITKIO()

    # Deterministic, outcome-independent selection made before any result is read:
    # sort the candidate identifiers, then take an evenly spaced stride so the sample
    # spans the geometry range rather than clustering at one end.
    pool = sorted(c.strip() for c in open(a.cases_file) if c.strip())
    n = min(a.n_real, len(pool))
    stride = max(1, len(pool) // n)
    cases = [pool[i * stride] for i in range(n)]

    weights = [RI._load_weights(cp) for cp in a.checkpoints]
    cached = [P.build_predictor(plans, dataset_json, w, a.policy) for w in weights]

    axes_ok = True
    for pr in cached:
        if a.policy == "M8" and tuple(pr.allowed_mirroring_axes or ()) != P.MIRROR_AXES:
            axes_ok = False
        if a.policy == "C0" and pr.allowed_mirroring_axes is not None:
            axes_ok = False

    worst_prob = 0.0
    seg_mismatch = geom_mismatch = label_mismatch = cross = errors = compared = 0
    shapes = []
    fg_extents = []
    t0 = time.time()

    for cid in cases:
        try:
            cdir = os.path.join(a.data_root, cid)
            files = [os.path.join(cdir, f"{cid}-{m}.nii.gz") for m in P.REQUIRED_MODALITIES]
            data, props = io.read_images(files)
            arr = np.asarray(data)
            shapes.append(list(arr.shape[1:]))
            # BraTS raw volumes share one voxel grid, so grid shape alone cannot show
            # geometry spread. Record the nonzero-foreground extent, which does vary.
            nz = np.nonzero(arr[0] != 0)
            fg_extents.append([int(x.max() - x.min() + 1) if len(x) else 0 for x in nz])
            ref = P.ensemble_probabilities(None, weights, plans, dataset_json,
                                           a.policy, data, props)
            opt = P.ensemble_probabilities(cached, weights, plans, dataset_json,
                                           a.policy, data, props)
            worst_prob = max(worst_prob, float(np.abs(ref - opt).max()))
            cross += int(((ref >= THRESHOLD) != (opt >= THRESHOLD)).sum())
            sr = RC.project_and_reconstruct(ref[0], ref[1], ref[2],
                                            THRESHOLD, THRESHOLD, THRESHOLD)
            so = RC.project_and_reconstruct(opt[0], opt[1], opt[2],
                                            THRESHOLD, THRESHOLD, THRESHOLD)
            if not np.array_equal(sr, so):
                seg_mismatch += 1
            if sr.shape != so.shape:
                geom_mismatch += 1
            if set(np.unique(sr).tolist()) != set(np.unique(so).tolist()):
                label_mismatch += 1
            compared += 1
        except Exception:
            errors += 1

    result = {"schema": "gat26.g84.cache_equivalence.v1", "policy": a.policy,
              "use_mirroring": P.POLICIES[a.policy]["use_mirroring"],
              "mirror_axes_verified": axes_ok,
              "n_real_cases": len(cases), "n_compared": compared,
              "distinct_input_grid_shapes": len({tuple(s) for s in shapes}),
              "input_grid_shape_note": "BraTS raw volumes share one voxel grid by construction",
              "distinct_foreground_extents": len({tuple(e) for e in fg_extents}),
              "foreground_extent_voxel_span": ([min(e[i] for e in fg_extents) for i in range(3)],
                                               [max(e[i] for e in fg_extents) for i in range(3)]) if fg_extents else None,
              "max_abs_probability_difference": worst_prob,
              "probability_tolerance": PROB_TOL,
              "segmentation_arrays_mismatched": seg_mismatch,
              "geometry_mismatched": geom_mismatch,
              "labels_mismatched": label_mismatch,
              "voxels_crossing_threshold_differently": cross,
              "errors": errors, "seconds": round(time.time() - t0, 1)}
    result["PASS"] = bool(compared == len(cases) and errors == 0 and axes_ok
                          and seg_mismatch == 0 and geom_mismatch == 0
                          and label_mismatch == 0 and cross == 0
                          and worst_prob <= PROB_TOL)

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
