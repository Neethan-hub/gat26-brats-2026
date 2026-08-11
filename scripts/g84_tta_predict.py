#!/usr/bin/env python3
"""G84 exact release-path inference for C0 and M8.

The ONLY permitted scientific difference between the two policies is

    use_mirroring: False (C0)  ->  True (M8)

with mirroring axes required to be exactly (0, 1, 2), i.e. the original prediction
plus all seven nonempty flip combinations. Everything else is frozen and refused
structurally: tile step stays 0.5, Gaussian weighting stays on, threshold stays
0.5, only checkpoint_final is loaded, fold probabilities are combined by an equal
arithmetic mean in the original checkpoint order, and reconstruction is the
unchanged hierarchy-safe rule with no cleanup and no presence gate.

Final segmentations are stored directly, so no probability quantisation is
involved and no quantisation-induced threshold difference is possible.

Two execution modes produce identical output: ``per_case`` rebuilds and reloads
every predictor for every case (the original release path); ``cached`` builds them
once. ``cached`` is a runtime-only optimisation and may be used only after
g84_cache_equivalence.py proves it bit-identical for the policy in question.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

TILE_STEP = 0.5
MIRROR_AXES = (0, 1, 2)
THRESHOLD = 0.5
POLICIES = {"C0": {"use_mirroring": False}, "M8": {"use_mirroring": True}}
FROZEN = {
    "tile_step_size": 0.5,
    "use_gaussian": True,
    "threshold": 0.5,
    "config": "3d_fullres",
    "checkpoint_name": "checkpoint_final.pth",
    "fold_weighting": "equal arithmetic mean in original checkpoint order",
    "reconstruction": "hierarchy-safe ET subset TC subset WT",
    "cleanup": "none",
    "presence_gate": "none",
}
REQUIRED_MODALITIES = ("t1n", "t1c", "t2w", "t2f")


class PolicyViolation(RuntimeError):
    """Raised when a caller tries to change anything except use_mirroring."""


def assert_policy(policy: str, **overrides) -> None:
    if policy not in POLICIES:
        raise PolicyViolation(
            f"policy {policy!r} is not preregistered; only {sorted(POLICIES)} exist")
    for key, value in overrides.items():
        if key not in FROZEN:
            raise PolicyViolation(f"{key} is not a tunable parameter in G84")
        if value != FROZEN[key]:
            raise PolicyViolation(
                f"{key}={value!r} deviates from the frozen release value {FROZEN[key]!r}")


def assert_axes(axes) -> None:
    if tuple(axes) != MIRROR_AXES:
        raise PolicyViolation(
            f"allowed_mirroring_axes must be exactly {MIRROR_AXES}, got {tuple(axes)!r}")


def atomic_write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def build_predictor(plans, dataset_json, weights, policy: str):
    """Frozen release predictor; use_mirroring is the only variable."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    assert_policy(policy)
    mirror = POLICIES[policy]["use_mirroring"]
    pm = PlansManager(plans)
    cm = pm.get_configuration(FROZEN["config"])
    lm = pm.get_label_manager(dataset_json)
    net = get_network_from_plans(
        cm.network_arch_class_name, cm.network_arch_init_kwargs,
        cm.network_arch_init_kwargs_req_import, len(dataset_json["channel_names"]),
        lm.num_segmentation_heads, allow_init=True, deep_supervision=False)
    pred = nnUNetPredictor(tile_step_size=TILE_STEP, use_gaussian=FROZEN["use_gaussian"],
                           use_mirroring=mirror, device=torch.device("cuda"),
                           verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    pred.manual_initialization(net, pm, cm, [weights], dataset_json, "nnUNetTrainer",
                               MIRROR_AXES if mirror else None)
    if mirror:
        assert_axes(pred.allowed_mirroring_axes)
    else:
        if pred.allowed_mirroring_axes is not None:
            raise PolicyViolation("C0 must carry no mirroring axes")
    return pred


def ensemble_probabilities(predictors_or_none, weights, plans, dataset_json, policy,
                           data, props):
    """Equal arithmetic mean over checkpoints, in the original order."""
    import numpy as np
    import torch
    acc = None
    for k in range(len(weights)):
        pred = (predictors_or_none[k] if predictors_or_none is not None
                else build_predictor(plans, dataset_json, weights[k], policy))
        _, probs = pred.predict_single_npy_array(data, props, None, None, True)
        acc = np.asarray(probs) if acc is None else acc + np.asarray(probs)
        if predictors_or_none is None:
            del pred
            torch.cuda.empty_cache()
    return acc / float(len(weights))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=sorted(POLICIES), required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="one checkpoint for out-of-fold work, five for the ensemble")
    ap.add_argument("--cases-file", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--dataset-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scripts", required=True)
    ap.add_argument("--mode", choices=("per_case", "cached"), default="cached")
    ap.add_argument("--emit", choices=("seg_npz", "nifti"), default="seg_npz")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    assert_policy(a.policy)
    sys.path.insert(0, a.scripts)
    import numpy as np
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    import release_infer as RI
    import g4_reconstruct_validate as RC

    for cp in a.checkpoints:
        base = os.path.basename(cp)
        if FROZEN["checkpoint_name"] not in base and "c0_fold" not in base:
            raise PolicyViolation(f"refusing non-final checkpoint {base}")

    plans = json.load(open(a.plans))
    dataset_json = json.load(open(a.dataset_json))
    cases = [c.strip() for c in open(a.cases_file) if c.strip()]
    if a.limit:
        cases = cases[:a.limit]

    out_dir = os.path.join(a.out_dir, "seg")
    os.makedirs(out_dir, exist_ok=True)

    RI.enforce_determinism()
    io = SimpleITKIO()
    weights = [RI._load_weights(cp) for cp in a.checkpoints]     # order preserved
    predictors = ([build_predictor(plans, dataset_json, w, a.policy) for w in weights]
                  if a.mode == "cached" else None)

    t0 = time.time()
    done = skipped = failed = 0
    per_case = []
    for i, cid in enumerate(cases):
        dst = os.path.join(out_dir, f"{cid}.npz" if a.emit == "seg_npz"
                           else f"{cid}.nii.gz")
        if os.path.exists(dst):
            skipped += 1
            continue
        c0 = time.time()
        try:
            cdir = os.path.join(a.data_root, cid)
            files = [os.path.join(cdir, f"{cid}-{m}.nii.gz") for m in REQUIRED_MODALITIES]
            data, props = io.read_images(files)
            acc = ensemble_probabilities(predictors, weights, plans, dataset_json,
                                         a.policy, data, props)
            seg = RC.project_and_reconstruct(acc[0], acc[1], acc[2],
                                             THRESHOLD, THRESHOLD, THRESHOLD)
            tmp = dst + ".tmp"
            if a.emit == "seg_npz":
                np.savez_compressed(tmp + ".npz", seg=seg.astype(np.uint8))
                os.replace(tmp + ".npz", dst)
            else:
                io.write_seg(seg.astype(np.uint8), tmp + ".nii.gz", props)
                os.replace(tmp + ".nii.gz", dst)
            done += 1
            per_case.append(time.time() - c0)
        except Exception as e:
            failed += 1
            atomic_write_json(os.path.join(a.out_dir, f"{cid}.FAILED.json"),
                              {"case": cid, "error": f"{type(e).__name__}: {e}"})
        if (i + 1) % 20 == 0 or i + 1 == len(cases):
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
    summary = {"phase": "DONE", "policy": a.policy, "mode": a.mode,
               "use_mirroring": POLICIES[a.policy]["use_mirroring"],
               "mirror_axes": list(MIRROR_AXES) if POLICIES[a.policy]["use_mirroring"] else None,
               "tile_step": TILE_STEP, "n_checkpoints": len(weights),
               "done": done, "skipped": skipped, "failed": failed,
               "minutes": round((time.time() - t0) / 60, 2),
               "peak_reserved_vram_gib": round(peak, 3)}
    if per_case:
        s = sorted(per_case)
        mean = sum(s) / len(s)
        summary["case_seconds"] = {
            "n": len(s), "mean": round(mean, 3),
            "p95": round(s[min(len(s) - 1, int(0.95 * len(s)))], 3),
            "max": round(s[-1], 3),
            "stdev": round((sum((x - mean) ** 2 for x in s) / len(s)) ** 0.5, 3)}
    atomic_write_json(os.path.join(a.out_dir, "summary.json"), summary)
    print(json.dumps(summary), flush=True)
    return 0 if failed == 0 else 7


if __name__ == "__main__":
    sys.exit(main())
