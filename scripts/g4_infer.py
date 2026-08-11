#!/usr/bin/env python3
"""GAT-26 G4 — full-case sliding-window inference on the two pilots using the one-step
smoke checkpoint. Region probabilities -> GAT-26 hierarchy reconstruction -> validated
flat Task-3 output. g4_real_data_smoke_only / no_accuracy_claim / not_A10G_parity.
Predictions and geometry stay private; only sanitized booleans/timings are returned.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g4_reconstruct_validate as RC  # noqa: E402


def build_predictor(plans, dataset_json, config, ckpt_weights):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    pm = PlansManager(plans)
    cm = pm.get_configuration(config)
    lm = pm.get_label_manager(dataset_json)
    num_in = len(dataset_json["channel_names"])
    net = get_network_from_plans(
        cm.network_arch_class_name, cm.network_arch_init_kwargs,
        cm.network_arch_init_kwargs_req_import, num_in, lm.num_segmentation_heads,
        allow_init=True, deep_supervision=False)
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                           device=torch.device("cuda"), verbose=False, verbose_preprocessing=False,
                           allow_tqdm=False)
    pred.manual_initialization(net, pm, cm, [ckpt_weights], dataset_json,
                               "nnUNetTrainer", None)
    return pred, lm


def infer_case(pred, files_in_modality_order):
    """Return (region_probs [3,X,Y,Z] in source geometry, properties)."""
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    data, props = SimpleITKIO().read_images(files_in_modality_order)
    seg, probs = pred.predict_single_npy_array(data, props, None, None, True)
    return probs, props


def reconstruct_from_probs(probs):
    # nnU-Net region channels are ordered as the labels dict regions: WT, TC, ET.
    q_wt, q_tc, q_et = probs[0], probs[1], probs[2]
    return RC.project_and_reconstruct(q_wt, q_tc, q_et)


def save_output(seg_uint8, props, out_dir, five_digit):
    """Write via nnU-Net's own SimpleITKIO so geometry (spacing/origin/direction) is
    restored exactly as nnU-Net read it — no RAS/LPS mismatch."""
    import numpy as np
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    name = RC.task3_output_name(five_digit)
    path = Path(out_dir) / name
    SimpleITKIO().write_seg(np.ascontiguousarray(seg_uint8.astype("uint8")), str(path), props)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--dataset", default="Dataset777_GAT26G4SMOKE")
    ap.add_argument("--plans", default="nnUNetResEncUNetMPlans")
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--result-json", required=True)
    args = ap.parse_args()

    import torch
    import numpy as np  # noqa
    import nibabel as nib  # noqa
    from nnunetv2.paths import nnUNet_preprocessed

    pp = Path(nnUNet_preprocessed) / args.dataset
    plans = json.loads((pp / f"{args.plans}.json").read_text())
    dataset_json = json.loads((pp / "dataset.json").read_text())
    weights = torch.load(args.ckpt, map_location="cpu", weights_only=True)["network_weights"]
    # torch.compile wraps the module -> strip the _orig_mod. prefix for a plain network.
    weights = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
               for k, v in weights.items()}
    mapping = json.loads(Path(args.mapping).read_text())

    pred, lm = build_predictor(plans, dataset_json, args.config, weights)
    out = {"label": "g4_real_data_smoke_only", "no_accuracy_claim": True,
           "not_A10G_parity": True, "cases": {}, "controls": {}}

    aliases = {"pilot_case_A": "00001", "pilot_case_B": "00002"}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    per_case_hash = {}
    ref_paths = {}
    for pseudo, mods in mapping.items():
        order = [mods["t1n"], mods["t1c"], mods["t2w"], mods["t2f"]]
        ref_paths[pseudo] = mods["t1n"]
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        probs, props = infer_case(pred, order)
        seg = reconstruct_from_probs(probs)
        infer_s = time.time() - t0
        path = save_output(seg, props, args.out_dir, aliases[pseudo])
        # validate saved file: compare output vs raw T1n, both read by nibabel
        ref = nib.load(str(mods["t1n"]))
        v = RC.validate_output_file(path, ref.affine, ref.shape,
                                    ref.header.get_zooms(), nib.aff2axcodes(ref.affine))
        per_case_hash[pseudo] = RC.mask_hash(seg)
        out["cases"][aliases[pseudo]] = {
            "validator_ok": v["ok"], "name_ok": v["name_ok"], "geometry_ok": v["geometry_ok"],
            "mask_ok": v["mask_ok"], "value_set": v["mask_detail"]["value_set"],
            "et_subset_tc": v["mask_detail"]["et_subset_tc"],
            "tc_subset_wt": v["mask_detail"]["tc_subset_wt"],
            "infer_seconds": round(infer_s, 2),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}

    # flat-output check: only 2 .nii.gz, nothing else
    files = sorted(Path(args.out_dir).iterdir())
    out["flat_output_ok"] = (len(files) == 2 and all(f.name.endswith(".nii.gz") for f in files))
    out["output_count_eq_case_count"] = len(files) == len(mapping)

    # determinism: rerun pilot_case_A once, require identical hash
    a = "pilot_case_A"
    probs2, _ = infer_case(pred, [mapping[a]["t1n"], mapping[a]["t1c"], mapping[a]["t2w"], mapping[a]["t2f"]])
    h2 = RC.mask_hash(reconstruct_from_probs(probs2))
    out["controls"]["deterministic_repeat_ok"] = (h2 == per_case_hash[a])

    # shuffled discovery: derive channel order by modality suffix from a shuffled list
    shuffled = [mapping[a]["t2f"], mapping[a]["seg"], mapping[a]["t1c"], mapping[a]["t1n"], mapping[a]["t2w"]]
    def by_mod(paths):
        import g3_audit_labeled_archive as AUD
        d = {AUD.classify_modality(Path(p).name): p for p in paths}
        return [d["t1n"], d["t1c"], d["t2w"], d["t2f"]]
    probs3, _ = infer_case(pred, by_mod(shuffled))
    h3 = RC.mask_hash(reconstruct_from_probs(probs3))
    out["controls"]["shuffled_order_ok"] = (h3 == per_case_hash[a])

    # missing-modality negative: only 3 modalities -> must fail, no output written
    n_before = len(list(Path(args.out_dir).iterdir()))
    try:
        infer_case(pred, [mapping[a]["t1n"], mapping[a]["t1c"], mapping[a]["t2w"]])
        out["controls"]["missing_modality_failed"] = False
    except Exception:
        out["controls"]["missing_modality_failed"] = True
    out["controls"]["missing_modality_no_extra_output"] = (
        len(list(Path(args.out_dir).iterdir())) == n_before)

    Path(args.result_json).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))
    ok = (all(c["validator_ok"] for c in out["cases"].values())
          and out["flat_output_ok"] and out["output_count_eq_case_count"]
          and out["controls"]["deterministic_repeat_ok"]
          and out["controls"]["shuffled_order_ok"]
          and out["controls"]["missing_modality_failed"]
          and out["controls"]["missing_modality_no_extra_output"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
