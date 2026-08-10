#!/usr/bin/env python3
"""G83 dense-overlap inference: C0 (tile step 0.5) and D25 (tile step 0.25).

The ONLY permitted difference between the two policies is ``tile_step_size``.
Every other knob is frozen and structurally refused if a caller tries to change
it: no mirroring/TTA, no threshold search, no cleanup, no presence gate, no new
weights, no checkpoint_best, no soup.

Two execution modes, both producing identical output:

  ``per_case``  the original frozen release path -- rebuild and reload all five
                predictors for every case.
  ``cached``    a RUNTIME-ONLY optimisation that builds the five predictors once
                and reuses them. It is not a scientific candidate; it may be used
                only after g83_runtime_equivalence.py proves it produces
                bit-identical output at step 0.5.

Checkpoint order and probability accumulation order are preserved exactly in both
modes, because the ensemble mean is order-sensitive in floating point.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ALLOWED_TILE_STEPS = (0.5, 0.25)
FROZEN = {
    "use_gaussian": True,
    "use_mirroring": False,
    "threshold": 0.5,
    "config": "3d_fullres",
    "checkpoint_name": "checkpoint_final.pth",
}
REQUIRED_MODALITIES = ("t1n", "t1c", "t2w", "t2f")
QMAX = 255
CROP_AT = 0.05


class PolicyViolation(RuntimeError):
    """Raised when a caller tries to change anything except the tile step."""


def assert_policy(tile_step: float, **overrides) -> None:
    """Fail closed on any deviation from the frozen policy."""
    if float(tile_step) not in ALLOWED_TILE_STEPS:
        raise PolicyViolation(
            f"tile_step_size {tile_step} is not preregistered; only "
            f"{ALLOWED_TILE_STEPS} are permitted")
    for key, value in overrides.items():
        if key not in FROZEN:
            raise PolicyViolation(f"{key} is not a tunable parameter in G83")
        if value != FROZEN[key]:
            raise PolicyViolation(
                f"{key}={value!r} deviates from the frozen C0 value {FROZEN[key]!r}")


def policy_name(tile_step: float) -> str:
    return "C0" if float(tile_step) == 0.5 else "D25"


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
    return np.rint(np.clip(stack, 0.0, 1.0) * QMAX).astype(np.uint8), bbox


def build_predictor(RI, plans, dataset_json, weights, tile_step):
    """Frozen release predictor with the tile step as the only variable."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    pm = PlansManager(plans)
    cm = pm.get_configuration(FROZEN["config"])
    lm = pm.get_label_manager(dataset_json)
    net = get_network_from_plans(
        cm.network_arch_class_name, cm.network_arch_init_kwargs,
        cm.network_arch_init_kwargs_req_import, len(dataset_json["channel_names"]),
        lm.num_segmentation_heads, allow_init=True, deep_supervision=False)
    pred = nnUNetPredictor(tile_step_size=float(tile_step),
                           use_gaussian=FROZEN["use_gaussian"],
                           use_mirroring=FROZEN["use_mirroring"],
                           device=torch.device("cuda"), verbose=False,
                           verbose_preprocessing=False, allow_tqdm=False)
    pred.manual_initialization(net, pm, cm, [weights], dataset_json, "nnUNetTrainer", None)
    return pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile-step", type=float, required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="one checkpoint for out-of-fold work, five for the ensemble")
    ap.add_argument("--cases-file", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--dataset-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scripts", required=True)
    ap.add_argument("--mode", choices=("per_case", "cached"), default="cached")
    ap.add_argument("--emit", choices=("probs", "nifti"), default="probs")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    assert_policy(a.tile_step)
    sys.path.insert(0, a.scripts)
    import numpy as np
    import torch
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    import release_infer as RI
    import g4_reconstruct_validate as RC

    for cp in a.checkpoints:
        if os.path.basename(cp) != FROZEN["checkpoint_name"] and "c0_fold" not in os.path.basename(cp):
            raise PolicyViolation(f"refusing non-final checkpoint {os.path.basename(cp)}")

    plans = json.load(open(a.plans))
    dataset_json = json.load(open(a.dataset_json))
    cases = [c.strip() for c in open(a.cases_file) if c.strip()]
    if a.limit:
        cases = cases[:a.limit]

    out_probs = os.path.join(a.out_dir, "probs")
    out_nii = os.path.join(a.out_dir, "pred")
    os.makedirs(out_probs if a.emit == "probs" else out_nii, exist_ok=True)

    RI.enforce_determinism()
    io = SimpleITKIO()
    weights = [RI._load_weights(cp) for cp in a.checkpoints]     # order preserved

    predictors = None
    if a.mode == "cached":
        predictors = [build_predictor(RI, plans, dataset_json, w, a.tile_step)
                      for w in weights]

    t0 = time.time()
    done = skipped = failed = 0
    per_case_seconds = []
    for i, cid in enumerate(cases):
        dst = (os.path.join(out_probs, f"{cid}.npz") if a.emit == "probs"
               else os.path.join(out_nii, f"{cid}.nii.gz"))
        if os.path.exists(dst):
            skipped += 1
            continue
        c0 = time.time()
        try:
            cdir = os.path.join(a.data_root, cid)
            files = [os.path.join(cdir, f"{cid}-{m}.nii.gz") for m in REQUIRED_MODALITIES]
            data, props = io.read_images(files)
            acc = None
            for k in range(len(weights)):                        # checkpoint order fixed
                pred = (predictors[k] if predictors is not None
                        else build_predictor(RI, plans, dataset_json, weights[k], a.tile_step))
                _, probs = pred.predict_single_npy_array(data, props, None, None, True)
                acc = np.asarray(probs) if acc is None else acc + np.asarray(probs)
                if predictors is None:
                    del pred
                    torch.cuda.empty_cache()
            acc /= float(len(weights))                           # equal mean, order fixed

            if a.emit == "probs":
                p_wt, p_tc, p_et = project(acc)
                q, bbox = quantise_and_crop(p_wt, p_tc, p_et)
                tmp = dst + ".tmp.npz"
                np.savez_compressed(tmp, q=q, bbox=np.array(bbox, dtype=np.int32),
                                    shape=np.array(p_wt.shape, dtype=np.int32))
                os.replace(tmp, dst)
            else:
                seg = RC.project_and_reconstruct(acc[0], acc[1], acc[2],
                                                 FROZEN["threshold"], FROZEN["threshold"],
                                                 FROZEN["threshold"])
                tmp = dst + ".tmp.nii.gz"
                io.write_seg(seg.astype(np.uint8), tmp, props)
                os.replace(tmp, dst)
            done += 1
            per_case_seconds.append(time.time() - c0)
        except Exception as e:
            failed += 1
            atomic_write_json(os.path.join(a.out_dir, f"{cid}.FAILED.json"),
                              {"case": cid, "error": f"{type(e).__name__}: {e}"})
        if (i + 1) % 25 == 0 or i + 1 == len(cases):
            el = time.time() - t0
            print(json.dumps({"i": i + 1, "of": len(cases), "done": done,
                              "skipped": skipped, "failed": failed,
                              "s_per_case": round(el / max(done, 1), 2)}), flush=True)

    peak = 0.0
    try:
        import torch as _t
        peak = _t.cuda.max_memory_reserved() / 2 ** 30
    except Exception:
        pass
    summary = {"phase": "DONE", "policy": policy_name(a.tile_step),
               "tile_step": a.tile_step, "mode": a.mode, "emit": a.emit,
               "done": done, "skipped": skipped, "failed": failed,
               "n_checkpoints": len(weights),
               "minutes": round((time.time() - t0) / 60, 2),
               "peak_reserved_vram_gib": round(peak, 3)}
    if per_case_seconds:
        s = sorted(per_case_seconds)
        summary["case_seconds"] = {
            "n": len(s), "mean": round(sum(s) / len(s), 3),
            "p95": round(s[min(len(s) - 1, int(0.95 * len(s)))], 3),
            "max": round(s[-1], 3),
            "stdev": round((sum((x - sum(s) / len(s)) ** 2 for x in s) / len(s)) ** 0.5, 3)}
    atomic_write_json(os.path.join(a.out_dir, "summary.json"), summary)
    print(json.dumps(summary), flush=True)
    return 0 if failed == 0 else 7


if __name__ == "__main__":
    sys.exit(main())
