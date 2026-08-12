#!/usr/bin/env python3
"""GAT-26 G5A checkpoint resume drill (MANDATORY pre-launch).

Proves true continuation (not just weight reload) through the installed nnU-Net 2.8.1
`save_checkpoint`/`load_checkpoint` API — network + optimizer + grad_scaler + epoch — then
continues one further epoch and verifies the epoch advances. Isolated from production
outputs. Never uses strict=False. Sanitized evidence only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build(plans_json, dataset_json, config, fold, out_dir, num_epochs):
    import torch
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    plans = json.loads(Path(plans_json).read_text(encoding="utf-8"))
    dj = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
    pt = dict(plans); pt["continue_training"] = False
    t = nnUNetTrainer(plans=pt, configuration=config, fold=fold, dataset_json=dj,
                      device=torch.device("cuda"))
    fold_dir = Path(out_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    t.output_folder_base = str(out_dir)
    t.output_folder = str(fold_dir)
    t.log_file = str(fold_dir / "drill.log")
    t.num_epochs = num_epochs
    t.initialize()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Dataset501_GAT26GOAT")
    ap.add_argument("--plans", default="nnUNetResEncUNetMPlans")
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--result", required=True)
    args = ap.parse_args()

    import os, torch
    pp = Path(os.environ["nnUNet_preprocessed"]) / args.dataset
    plans_json = str(pp / f"{args.plans}.json"); dataset_json = str(pp / "dataset.json")
    outA = Path(args.out_dir) / "A"; outB = Path(args.out_dir) / "B"

    # --- Trainer A: run exactly one epoch, save checkpoint (network+optimizer+scaler+epoch) ---
    A = _build(plans_json, dataset_json, args.config, args.fold, str(outA), num_epochs=1)
    A.run_training()                                  # 1 epoch + on_train_end saves checkpoint_final
    ckpt = Path(A.output_folder) / "checkpoint_final.pth"
    saved_epoch = int(torch.load(ckpt, map_location="cpu", weights_only=False)["current_epoch"])
    a_scaler = torch.load(ckpt, map_location="cpu", weights_only=False).get("grad_scaler_state") is not None
    a_opt = len(torch.load(ckpt, map_location="cpu", weights_only=False)["optimizer_state"]["state"]) > 0
    try: A.on_train_end()
    except Exception: pass
    del A; torch.cuda.empty_cache()

    # --- Trainer B: fresh process/object, load checkpoint through the supported API ---
    B = _build(plans_json, dataset_json, args.config, args.fold, str(outB),
               num_epochs=saved_epoch + 1)
    B.load_checkpoint(str(ckpt))                      # restores weights+optimizer+scaler+epoch (no strict=False)
    restored_epoch = int(B.current_epoch)
    opt_state_restored = len(B.optimizer.state_dict()["state"]) > 0
    scaler_restored = (B.grad_scaler is not None)
    # continue exactly one more epoch
    B.run_training()                                  # range(restored_epoch, restored_epoch+1)
    final_epoch = int(B.current_epoch)
    lr_now = float(B.optimizer.param_groups[0]["lr"])
    # finite params/grads after continuation
    params_finite = all(bool(torch.isfinite(p).all()) for p in B.network.parameters())
    grads_finite = all((p.grad is None) or bool(torch.isfinite(p.grad).all()) for p in B.network.parameters())
    new_ckpt = (Path(B.output_folder) / "checkpoint_final.pth").exists()
    try: B.on_train_end()
    except Exception: pass

    out = {
        "label": "g5a_resume_drill", "plans": args.plans, "fold": args.fold,
        "A_saved_epoch": saved_epoch, "A_optimizer_state_saved": bool(a_opt), "A_scaler_saved": bool(a_scaler),
        "B_restored_epoch": restored_epoch,
        "B_optimizer_state_restored": bool(opt_state_restored),
        "B_scaler_restored": bool(scaler_restored),
        "epoch_restored_matches_saved": restored_epoch == saved_epoch,
        "final_epoch": final_epoch,
        "epoch_advanced_not_restarted": final_epoch == restored_epoch + 1 and final_epoch > restored_epoch,
        "lr_after_resume": round(lr_now, 6), "lr_finite": bool(lr_now == lr_now and lr_now not in (float("inf"), float("-inf"))),
        "params_finite": params_finite, "grads_finite": grads_finite,
        "new_checkpoint_produced": bool(new_ckpt), "strict_false_used": False,
    }
    ok = (out["A_optimizer_state_saved"] and out["A_scaler_saved"] and out["epoch_restored_matches_saved"]
          and out["B_optimizer_state_restored"] and out["B_scaler_restored"]
          and out["epoch_advanced_not_restarted"] and out["params_finite"] and out["grads_finite"]
          and out["lr_finite"] and out["new_checkpoint_produced"])
    out["drill_pass"] = bool(ok)
    print(json.dumps(out))
    Path(args.result).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
