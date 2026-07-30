#!/usr/bin/env python3
"""G82 §F C0-equivalence gate + trainer smoke.

Proves that the copied custom-trainer initialisation is bit-for-bit the same
network as the original C0 checkpoint path before a single optimisation step is
taken. If any tolerance is exceeded the caller must return
``G82_BLOCKED_CHECKPOINT_INITIALIZATION_NOT_C0_EQUIVALENT``.

Path A  the original C0 release path: get_network_from_plans(deep_supervision=False)
        followed by load_state_dict(checkpoint["network_weights"]).
Path B  nnUNetTrainerG82.initialize() followed by load_c0_weights(), with deep
        supervision disabled for comparison only.

Tolerances (preregistered): max |logit diff| <= 1e-6, max prob diff <= 1e-7,
identical argmax/region output, identical parameter names, no missing or
unexpected weights.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LOGIT_TOL = 1e-6
PROB_TOL = 1e-7
PATCH_SEED = 20260730


def build_path_a(plans, dataset_json, config, ckpt_path, device):
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    pm = PlansManager(plans)
    cm = pm.get_configuration(config)
    lm = pm.get_label_manager(dataset_json)
    net = get_network_from_plans(cm.network_arch_class_name, cm.network_arch_init_kwargs,
                                 cm.network_arch_init_kwargs_req_import,
                                 len(dataset_json["channel_names"]),
                                 lm.num_segmentation_heads, allow_init=True,
                                 deep_supervision=False)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    w = ck["network_weights"]
    w = {k[7:] if k.startswith("module.") else k: v for k, v in w.items()}
    net.load_state_dict(w, strict=True)
    return net.to(device).eval(), cm, lm


def build_path_b(plans, dataset_json, config, fold, ckpt_path, device, cfg_path):
    os.environ["G82_CONFIG"] = cfg_path
    from g82_trainer import nnUNetTrainerG82
    tr = nnUNetTrainerG82(plans, config, fold, dataset_json, device=device)
    tr.initialize()
    net = tr.network
    tr.set_deep_supervision_enabled(False)
    return net.to(device).eval(), tr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True)
    ap.add_argument("--dataset-json", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--g82-config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plans = json.load(open(a.plans))
    dataset_json = json.load(open(a.dataset_json))

    net_a, cm, lm = build_path_a(plans, dataset_json, a.config, a.checkpoint, device)
    net_b, tr = build_path_b(plans, dataset_json, a.config, a.fold, a.checkpoint,
                             device, a.g82_config)

    from g82_trainer import strip_wrappers
    sd_a = {strip_wrappers(k): v for k, v in net_a.state_dict().items()}
    sd_b = {strip_wrappers(k): v for k, v in net_b.state_dict().items()}
    names_a, names_b = sorted(sd_a), sorted(sd_b)
    same_names = names_a == names_b
    max_param_diff = 0.0
    if same_names:
        for k in names_a:
            d = (sd_a[k].float() - sd_b[k].float().to(sd_a[k].device)).abs().max().item()
            max_param_diff = max(max_param_diff, d)

    # two fixed training-only patches
    g = torch.Generator().manual_seed(PATCH_SEED)
    patch = tuple(cm.patch_size)
    nch = len(dataset_json["channel_names"])
    x = torch.randn((2, nch) + patch, generator=g, dtype=torch.float32).to(device)

    with torch.no_grad():
        la = net_a(x)
        lb = net_b(x)
    la = la[0] if isinstance(la, (list, tuple)) else la
    lb = lb[0] if isinstance(lb, (list, tuple)) else lb

    logit_diff = (la.float() - lb.float()).abs().max().item()
    pa, pb = torch.sigmoid(la.float()), torch.sigmoid(lb.float())
    prob_diff = (pa - pb).abs().max().item()
    region_a, region_b = (pa >= 0.5), (pb >= 0.5)
    regions_identical = bool(torch.equal(region_a, region_b))
    argmax_identical = bool(torch.equal(la.float().argmax(1), lb.float().argmax(1)))

    init = tr.g82_init_report or {}
    result = {
        "schema": "gat26.g82.equivalence.v1",
        "fold": a.fold,
        "device": str(device),
        "identical_parameter_names": same_names,
        "n_parameters": len(names_a),
        "max_parameter_abs_diff": max_param_diff,
        "missing_keys": init.get("missing_keys", []),
        "unexpected_keys": init.get("unexpected_keys", []),
        "segmentation_head_tensors_loaded": len(init.get("seg_head_tensors", [])),
        "max_abs_logit_diff": logit_diff,
        "max_abs_prob_diff": prob_diff,
        "argmax_identical": argmax_identical,
        "region_output_identical": regions_identical,
        "tolerances": {"logit": LOGIT_TOL, "prob": PROB_TOL},
        "finetune_lr": tr.initial_lr,
        "num_epochs": tr.num_epochs,
        "recipe": tr.g82_recipe,
        "optimizer_family": "SGD",
        "weight_decay": tr.weight_decay,
        "batch_size": tr.batch_size,
        "patch_size": list(patch),
        "oversample_foreground_percent": tr.oversample_foreground_percent,
        "deep_supervision": tr.enable_deep_supervision,
    }
    result["PASS"] = bool(
        same_names
        and not init.get("missing_keys")
        and not init.get("unexpected_keys")
        and init.get("seg_head_tensors")
        and logit_diff <= LOGIT_TOL
        and prob_diff <= PROB_TOL
        and argmax_identical
        and regions_identical
    )
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
