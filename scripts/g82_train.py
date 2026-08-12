#!/usr/bin/env python3
"""G82 training entrypoint. Runs one candidate/fold fine-tuning job.

Everything that varies is read from the JSON pointed at by ``G82_CONFIG`` so no
private path or credential is ever visible in a process listing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _train_losses(logger) -> list:
    """nnU-Net 2.8.1 wraps the epoch logger in a MetaLogger; reach the real one."""
    for obj in [logger] + list(getattr(logger, "loggers", []) or []) + \
               list(vars(logger).values()):
        d = getattr(obj, "my_fantastic_logging", None)
        if isinstance(d, dict) and "train_losses" in d:
            return d["train_losses"]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True)
    ap.add_argument("--dataset-json", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--g82-config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.environ["G82_CONFIG"] = a.g82_config
    from g82_trainer import nnUNetTrainerG82

    plans = json.load(open(a.plans))
    dataset_json = json.load(open(a.dataset_json))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr = nnUNetTrainerG82(plans, a.config, a.fold, dataset_json, device=device)
    t0 = time.time()
    tr.run_training()
    wall = time.time() - t0

    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    losses = [float(x) for x in _train_losses(tr.logger)]
    rec = {
        "schema": "gat26.g82.trainjob.v1",
        "recipe": tr.g82_recipe,
        "fold": a.fold,
        "epochs": tr.num_epochs,
        "initial_lr": tr.initial_lr,
        "seed": tr.g82_seed,
        "wall_seconds": wall,
        "seconds_per_epoch": wall / max(tr.num_epochs, 1),
        "peak_vram_gib": peak,
        "train_losses": losses,
        "all_losses_finite": all(x == x and abs(x) != float("inf") for x in losses),
        "output_folder": tr.output_folder,
        "init_report": tr.g82_init_report,
    }
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, a.out)
    print(json.dumps({k: v for k, v in rec.items() if k != "train_losses"}, indent=1))
    return 0 if rec["all_losses_finite"] else 4


if __name__ == "__main__":
    sys.exit(main())
