#!/usr/bin/env python3
"""GAT-26 G4.5-R two-epoch pipeline canary (NOT G5).

Runs the EXACT production recipe (official nnUNetTrainer / data loader / augmentation /
loss / AMP / scaler, random init, frozen fold 0, candidate plan) with the ONLY intentional
divergence being num_epochs=2. Emits sanitized timing/memory/finiteness evidence. It does
NOT run held-out inference, compute or reveal DSC/HD95, choose a model, or keep the
checkpoint as a candidate. Outputs live under an isolated canary-only private directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Dataset501_GAT26GOAT")
    ap.add_argument("--plans", required=True)
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out-dir", required=True)           # isolated canary dir
    ap.add_argument("--result", required=True)
    ap.add_argument("--expect-split-sha256", required=True)
    ap.add_argument("--expect-plan-sha256", required=True)
    ap.add_argument("--test-resume", action="store_true")
    args = ap.parse_args()

    import os
    import numpy as np
    import torch
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    pp = Path(os.environ["nnUNet_preprocessed"]) / args.dataset
    # (1) verify frozen split + plan hashes BEFORE start (fail closed)
    split_sha = _sha(pp / "splits_final.json")
    plan_sha = _sha(pp / f"{args.plans}.json")
    if split_sha != args.expect_split_sha256:
        print(json.dumps({"error": "split_hash_mismatch"})); return 3
    if plan_sha != args.expect_plan_sha256:
        print(json.dumps({"error": "plan_hash_mismatch"})); return 3

    plans = json.loads((pp / f"{args.plans}.json").read_text())
    dataset_json = json.loads((pp / "dataset.json").read_text())
    plans_t = dict(plans); plans_t["continue_training"] = False
    trainer = nnUNetTrainer(plans=plans_t, configuration=args.config, fold=args.fold,
                            dataset_json=dataset_json, device=torch.device("cuda"))
    # isolate outputs into a canary-only directory (set BEFORE initialize so logging lands here)
    outdir = Path(args.out_dir)
    fold_dir = outdir / f"fold_{args.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    trainer.output_folder_base = str(outdir)
    trainer.output_folder = str(fold_dir)
    trainer.log_file = str(fold_dir / "canary_training.log")
    # ONLY intentional divergence from G5:
    trainer.num_epochs = 2
    trainer.initialize()
    net_class = trainer.network.__class__.__name__

    # timing wrappers (do not change recipe, only measure)
    tstats = {"train_s": 0.0, "val_s": 0.0, "train_steps": 0, "val_steps": 0}
    finite = {"loss": True, "params": True}
    epoch_walls = []
    _epoch_mark = {"t": None}
    _train_step = trainer.train_step
    _val_step = trainer.validation_step
    _on_epoch_start = trainer.on_epoch_start
    _on_epoch_end = trainer.on_epoch_end

    def on_epoch_start():
        _epoch_mark["t"] = time.time(); return _on_epoch_start()

    def on_epoch_end():
        r = _on_epoch_end()
        if _epoch_mark["t"] is not None:
            epoch_walls.append(round(time.time() - _epoch_mark["t"], 2))
        return r

    trainer.on_epoch_start = on_epoch_start
    trainer.on_epoch_end = on_epoch_end

    def train_step(b):
        t = time.time(); out = _train_step(b); torch.cuda.synchronize()
        tstats["train_s"] += time.time() - t; tstats["train_steps"] += 1
        lv = float(out["loss"]) if isinstance(out, dict) and "loss" in out else float("nan")
        if not np.isfinite(lv):
            finite["loss"] = False
        return out

    def val_step(b):
        t = time.time(); out = _val_step(b); torch.cuda.synchronize()
        tstats["val_s"] += time.time() - t; tstats["val_steps"] += 1
        return out

    trainer.train_step = train_step
    trainer.validation_step = val_step

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.run_training()          # 2 epochs: train + online val + final checkpoint
    total_s = time.time() - t0

    # per-epoch wall from our on_epoch_start/on_epoch_end wrappers (train+val+overhead)
    compile_overhead = round(epoch_walls[0] - epoch_walls[1], 2) if len(epoch_walls) >= 2 else None

    # finite params
    finite["params"] = all(bool(torch.isfinite(p).all()) for p in trainer.network.parameters())
    gfin = all((p.grad is None) or bool(torch.isfinite(p.grad).all()) for p in trainer.network.parameters())

    # checkpoint strict reload (explicit torch.compile _orig_mod. handling; never strict=False)
    ckpt = Path(trainer.output_folder) / "checkpoint_final.pth"
    reload_ok = False
    if ckpt.exists():
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["network_weights"]
        sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
        raw = getattr(trainer.network, "_orig_mod", trainer.network)
        raw.load_state_dict(sd, strict=True)
        reload_ok = True

    # optional resume test: a fresh trainer that continues from the canary checkpoint
    resume_ok = None
    if args.test_resume and ckpt.exists():
        try:
            plans_r = dict(plans); plans_r["continue_training"] = False
            tr2 = nnUNetTrainer(plans=plans_r, configuration=args.config, fold=args.fold,
                                dataset_json=dataset_json, device=torch.device("cuda"))
            tr2.output_folder_base = str(outdir); tr2.output_folder = str(outdir / f"fold_{args.fold}")
            tr2.log_file = None; tr2.num_epochs = 2
            tr2.initialize()
            tr2.load_checkpoint(str(ckpt))
            resume_ok = (tr2.current_epoch == 2)
            try: tr2.on_train_end()
            except Exception: pass
        except Exception as e:
            resume_ok = f"ERR:{type(e).__name__}"

    out = {
        "label": "g45r_canary_only", "no_accuracy_claim": True, "not_G5": True,
        "not_fold_training": True, "num_epochs": 2, "only_divergence": "num_epochs=2",
        "plans": args.plans, "network_class": net_class, "fold": args.fold,
        "split_hash_verified": True, "plan_hash_verified": True,
        "total_wall_s": round(total_s, 1),
        "epoch_walls_s": epoch_walls,
        "compile_overhead_s": compile_overhead,
        "steady_epoch_wall_s": epoch_walls[1] if len(epoch_walls) >= 2 else None,
        "train_step_mean_s": round(tstats["train_s"] / max(tstats["train_steps"], 1), 3),
        "val_step_mean_s": round(tstats["val_s"] / max(tstats["val_steps"], 1), 3),
        "train_steps": tstats["train_steps"], "val_steps": tstats["val_steps"],
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "loss_finite": finite["loss"], "grad_finite": bool(gfin), "params_finite": finite["params"],
        "checkpoint_saved": ckpt.exists(), "strict_reload_ok": reload_ok, "resume_ok": resume_ok,
        "batch_size": plans["configurations"][args.config]["batch_size"],
        "patch_size": plans["configurations"][args.config]["patch_size"],
    }
    try: trainer.on_train_end()
    except Exception: pass
    print(json.dumps(out))
    Path(args.result).write_text(json.dumps(out, indent=2) + "\n")
    ok = (finite["loss"] and finite["params"] and gfin and out["checkpoint_saved"] and reload_ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
