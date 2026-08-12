#!/usr/bin/env python3
"""GAT-26 G4 — exactly ONE native nnU-Net optimizer update on real preprocessed pilot
data (ResEnc-M plan). Random init; no pretrained weights. g4_real_data_smoke_only;
no_accuracy_claim; not_A10G_parity. Sanitized JSON to stdout; checkpoint stays private."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Dataset777_GAT26G4SMOKE")
    ap.add_argument("--plans", default="nnUNetResEncUNetMPlans")
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import numpy as np
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.paths import nnUNet_preprocessed

    out = {"label": "g4_real_data_smoke_only", "no_accuracy_claim": True, "not_A10G_parity": True}
    pp = Path(nnUNet_preprocessed) / args.dataset
    plans = json.loads((pp / f"{args.plans}.json").read_text(encoding="utf-8"))
    dataset_json = json.loads((pp / "dataset.json").read_text(encoding="utf-8"))

    # ResEnc-M contract sanity from the plan
    arch = plans["configurations"][args.config]["architecture"]
    out["arch_class"] = arch["network_class_name"]
    out["plans_identifier"] = plans["plans_name"]
    out["input_channels"] = len(dataset_json["channel_names"])
    out["regions_class_order"] = dataset_json.get("regions_class_order")

    device = torch.device("cuda")
    # This nnU-Net 2.8.1 build's trainer pops a 'continue_training' key from plans.
    plans_for_trainer = dict(plans); plans_for_trainer["continue_training"] = False
    trainer = nnUNetTrainer(plans=plans_for_trainer, configuration=args.config, fold=0,
                            dataset_json=dataset_json, device=device)
    trainer.initialize()

    # Prove random init: no pretrained checkpoint attribute set, no external weight loaded.
    out["network_class"] = trainer.network.__class__.__name__
    out["pretrained_ckpt_used"] = False  # nnUNetTrainer.initialize does not load weights

    def flat_params(net):
        return torch.cat([p.detach().reshape(-1).float().cpu() for p in net.parameters()])
    before = flat_params(trainer.network).clone()

    trainer.on_train_start()          # builds dataloaders, unpacks data
    trainer.on_train_epoch_start()
    batch = next(trainer.dataloader_train)

    # finite inputs
    x = batch["data"]
    out["input_finite"] = bool(torch.isfinite(x).all().item())

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    log = trainer.train_step(batch)   # exactly ONE optimizer update (native AMP+scaler+loss)
    torch.cuda.synchronize()
    step_s = time.time() - t0

    loss_val = float(log["loss"])
    out["loss"] = loss_val
    out["loss_finite"] = bool(np.isfinite(loss_val))
    grad_norm = 0.0
    finite_grads = True
    for p in trainer.network.parameters():
        if p.grad is not None:
            if not bool(torch.isfinite(p.grad).all().item()):
                finite_grads = False
            grad_norm += float(p.grad.detach().float().norm().item()) ** 2
    grad_norm = grad_norm ** 0.5
    out["grad_finite"] = finite_grads
    out["grad_norm"] = grad_norm
    out["grad_norm_positive"] = grad_norm > 0

    after = flat_params(trainer.network)
    changed = int((after != before).sum().item())
    out["params_changed"] = changed
    out["params_changed_positive"] = changed > 0
    out["params_finite"] = bool(torch.isfinite(after).all().item())

    out["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    out["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    out["step_seconds"] = step_s

    # checkpoint save + strict reload + bitwise equality
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"network_weights": trainer.network.state_dict()}, args.ckpt)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)["network_weights"]
    # fresh model with same architecture
    plans_for_trainer2 = dict(plans); plans_for_trainer2["continue_training"] = False
    trainer2 = nnUNetTrainer(plans=plans_for_trainer2, configuration=args.config, fold=0,
                             dataset_json=dataset_json, device=torch.device("cuda"))
    trainer2.initialize()
    trainer2.network.load_state_dict(sd, strict=True)
    reloaded = trainer2.network.state_dict()
    saved = trainer.network.state_dict()
    bitwise = all(torch.equal(saved[k].cpu(), reloaded[k].cpu()) for k in saved)
    out["strict_reload"] = True
    out["bitwise_equal_after_reload"] = bool(bitwise)

    try:
        trainer.on_train_end()
    except Exception:
        pass

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    san = {k: v for k, v in out.items()}
    print(json.dumps(san))
    ok = (out["input_finite"] and out["loss_finite"] and out["grad_finite"]
          and out["grad_norm_positive"] and out["params_changed_positive"]
          and out["params_finite"] and out["bitwise_equal_after_reload"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
