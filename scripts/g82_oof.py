#!/usr/bin/env python3
"""G82 out-of-fold probability generation for one candidate checkpoint.

Uses the *frozen C0 inference path* unchanged (same predictor construction, tile
step 0.5, Gaussian weighting, no TTA, no mirroring) so the only difference
between baseline and candidate is the network weights.

Writes projected hierarchy-consistent probabilities quantised to uint8 and
cropped to the foreground bounding box, in exactly the storage format the
baseline store already uses, so blends and evaluation are directly comparable.

Ground truth is never read here; evaluation happens on the controller with the
preserved official evaluator.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REQUIRED_MODALITIES = ("t1n", "t1c", "t2w", "t2f")
QMAX = 255
CROP_AT = 0.05


def atomic_write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def project(probs):
    """nnU-Net region channel order is (WT, TC, ET); project to nested probabilities."""
    import numpy as np
    q_wt, q_tc, q_et = probs[0], probs[1], probs[2]
    p_et = q_et
    p_tc = np.maximum(q_tc, p_et)
    p_wt = np.maximum(q_wt, p_tc)
    return p_wt, p_tc, p_et


def quantise_and_crop(p_wt, p_tc, p_et):
    import numpy as np
    m = (p_wt >= CROP_AT) | (p_tc >= CROP_AT) | (p_et >= CROP_AT)
    if not m.any():
        bbox = (0, 1, 0, 1, 0, 1)
    else:
        idx = np.where(m)
        bbox = (int(idx[0].min()), int(idx[0].max()) + 1,
                int(idx[1].min()), int(idx[1].max()) + 1,
                int(idx[2].min()), int(idx[2].max()) + 1)
    sl = (slice(bbox[0], bbox[1]), slice(bbox[2], bbox[3]), slice(bbox[4], bbox[5]))
    stack = np.stack([p_wt[sl], p_tc[sl], p_et[sl]])
    q = np.rint(np.clip(stack, 0.0, 1.0) * QMAX).astype(np.uint8)
    return q, bbox


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--dataset-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scripts", required=True, help="directory holding release_infer.py")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, a.scripts)
    import numpy as np
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    import release_infer as RI

    plans = json.load(open(a.plans))
    dataset_json = json.load(open(a.dataset_json))
    cases = list(json.load(open(a.splits))[a.fold]["val"])
    if a.limit:
        cases = cases[:a.limit]

    prob_dir = os.path.join(a.out_dir, "probs")
    os.makedirs(prob_dir, exist_ok=True)

    RI.enforce_determinism()
    pred = RI._build_predictor(plans, dataset_json, RI._load_weights(a.checkpoint))
    io = SimpleITKIO()

    t0 = time.time()
    done = skipped = failed = 0
    for i, cid in enumerate(cases):
        prob_path = os.path.join(prob_dir, f"{cid}.npz")
        if os.path.exists(prob_path):
            skipped += 1
            continue
        try:
            cdir = os.path.join(a.data_root, cid)
            files = [os.path.join(cdir, f"{cid}-{m}.nii.gz") for m in REQUIRED_MODALITIES]
            data, props = io.read_images(files)
            _, probs = pred.predict_single_npy_array(data, props, None, None, True)
            p_wt, p_tc, p_et = project(np.asarray(probs))
            q, bbox = quantise_and_crop(p_wt, p_tc, p_et)
            tmp = prob_path + ".tmp.npz"
            np.savez_compressed(tmp, q=q, bbox=np.array(bbox, dtype=np.int32),
                                shape=np.array(p_wt.shape, dtype=np.int32))
            os.replace(tmp, prob_path)
            done += 1
        except Exception as e:
            failed += 1
            atomic_write_json(os.path.join(prob_dir, f"{cid}.FAILED.json"),
                              {"case": cid, "error": f"{type(e).__name__}: {e}"})
        if (i + 1) % 25 == 0 or i + 1 == len(cases):
            el = time.time() - t0
            print(json.dumps({"i": i + 1, "of": len(cases), "done": done,
                              "skipped": skipped, "failed": failed,
                              "s_per_case": round(el / max(done, 1), 2)}), flush=True)

    print(json.dumps({"phase": "DONE", "fold": a.fold, "done": done, "skipped": skipped,
                      "failed": failed, "minutes": round((time.time() - t0) / 60, 1)}),
          flush=True)
    return 0 if failed == 0 else 7


if __name__ == "__main__":
    sys.exit(main())
