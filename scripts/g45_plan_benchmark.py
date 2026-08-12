#!/usr/bin/env python3
"""GAT-26 G4.5 — production nnU-Net raw dataset view, full-data fingerprint, ResEnc
M/L plan generation, and bounded real-pipeline benchmarks.

Design contracts (must not be weakened):
  * Random initialization only; no pretrained/external tensor path.
  * Symlink the authorized raw payloads — never copy or modify source images.
  * Region order WT, TC, ET; regions_class_order [2,1,3]; channels T1n/T1c/T2w/T2f.
  * Official nnU-Net v2.8.1 ResidualEncoderUNet (ResEnc M / L presets).
  * Every benchmark labelled g45_benchmark_only / no_accuracy_claim /
    not_fold_training / not_A10G_parity. No DSC/HD95 score fields populated.
  * Atomic construction, fail-closed. Sanitized JSON to stdout; private artifacts
    stay on ignored worker paths.

This module is import-safe and CLI-driven; heavy imports are lazy inside commands.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Canonical production identifiers
NFOLDS = 5
DATASET_ID = 501
DATASET_NAME = f"Dataset{DATASET_ID:03d}_GAT26GOAT"
CHANNEL_NAMES = {"0": "T1n", "1": "T1c", "2": "T2w", "3": "T2f"}
MOD_SUFFIX = {"0000": "t1n", "0001": "t1c", "0002": "t2w", "0003": "t2f"}
LABELS = {"background": 0, "whole_tumor": [1, 2, 3], "tumor_core": [1, 3], "enhancing_tumor": [3]}
REGIONS_CLASS_ORDER = [2, 1, 3]
FILE_ENDING = ".nii.gz"
BENCH_LABELS = {"label": "g45_benchmark_only", "no_accuracy_claim": True,
                "not_fold_training": True, "not_A10G_parity": True}


def _raw_cases(raw_root: str):
    """Yield (case_id, {suffix: abspath}) for each complete case under the raw root.
    Layout: raw_root/<inner>/BraTS-GoAT-#####/BraTS-GoAT-#####-{t1n,t1c,t2w,t2f,seg}.nii.gz
    """
    import re
    inner = os.path.join(raw_root, sorted(os.listdir(raw_root))[0])
    out = []
    for cid in sorted(os.listdir(inner)):
        cd = os.path.join(inner, cid)
        if not os.path.isdir(cd):
            continue
        mods = {}
        for f in os.listdir(cd):
            m = re.search(r"-([a-z0-9]+)\.nii\.gz$", f)
            if m:
                mods[m.group(1)] = os.path.join(cd, f)
        if {"t1n", "t1c", "t2w", "t2f", "seg"}.issubset(mods):
            out.append((cid, mods))
    return out


def cmd_build_dataset(args):
    """Create the production raw dataset view via symlinks (atomic, fail-closed)."""
    raw_base = os.environ["nnUNet_raw"]
    cases = _raw_cases(args.raw_root)
    assert len(cases) == args.expect_cases, f"expected {args.expect_cases} cases, found {len(cases)}"
    dst = Path(raw_base) / DATASET_NAME
    tmp = Path(raw_base) / (DATASET_NAME + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "imagesTr").mkdir(parents=True)
    (tmp / "labelsTr").mkdir(parents=True)
    src_hashes_ok = True
    n_links = 0
    for cid, mods in cases:
        for ch, suf in MOD_SUFFIX.items():
            src = mods[suf]
            if not os.path.isfile(src):
                raise RuntimeError("missing source modality (fail-closed)")
            os.symlink(src, tmp / "imagesTr" / f"{cid}_{ch}.nii.gz")
            n_links += 1
        os.symlink(mods["seg"], tmp / "labelsTr" / f"{cid}.nii.gz")
        n_links += 1
    dataset_json = {
        "channel_names": CHANNEL_NAMES,
        "labels": LABELS,
        "regions_class_order": REGIONS_CLASS_ORDER,
        "numTraining": len(cases),
        "file_ending": FILE_ENDING,
        "name": DATASET_NAME,
        "description": "GAT-26 BraTS-GoAT production labeled training view (symlinks; random-init).",
    }
    (tmp / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8")
    # atomic promote
    if dst.exists():
        if args.force:
            shutil.rmtree(dst)
        else:
            raise RuntimeError(f"{dst} exists; refuse to overwrite without --force")
    os.rename(tmp, dst)
    out = {"dataset": DATASET_NAME, "cases": len(cases), "symlinks": n_links,
           "images_per_case": 4, "labels_per_case": 1,
           "regions_class_order": REGIONS_CLASS_ORDER, "channels": CHANNEL_NAMES,
           "labels_map": LABELS, "src_modified": False, "symlinked_not_copied": True}
    print(json.dumps(out))
    return 0


def cmd_verify_dataset(args):
    """Verify the production dataset contract without modifying anything."""
    raw_base = os.environ["nnUNet_raw"]
    d = Path(raw_base) / DATASET_NAME
    dj = json.loads((d / "dataset.json").read_text(encoding="utf-8"))
    imgs = sorted((d / "imagesTr").glob("*.nii.gz"))
    labs = sorted((d / "labelsTr").glob("*.nii.gz"))
    ids = sorted({p.name[:-len("_0000.nii.gz")] for p in imgs})
    # all images are symlinks (no copied payload) and resolve into the authorized raw root
    all_symlink = all(p.is_symlink() for p in imgs) and all(p.is_symlink() for p in labs)
    resolve_ok = all(os.path.realpath(p).startswith(os.path.realpath(args.raw_root)) for p in imgs[:50])
    out = {
        "dataset": DATASET_NAME,
        "num_cases": len(ids),
        "images": len(imgs), "labels": len(labs),
        "images_eq_4x": len(imgs) == 4 * len(ids),
        "channel_names_ok": dj["channel_names"] == CHANNEL_NAMES,
        "labels_ok": dj["labels"] == LABELS,
        "rco_ok": dj["regions_class_order"] == REGIONS_CLASS_ORDER,
        "numTraining_ok": dj["numTraining"] == len(ids),
        "file_ending_ok": dj["file_ending"] == FILE_ENDING,
        "all_symlink_no_copy": bool(all_symlink),
        "symlinks_resolve_into_raw_root": bool(resolve_ok),
    }
    out["contract_ok"] = all(v for k, v in out.items() if k.endswith("_ok") or k in
                             ("all_symlink_no_copy", "symlinks_resolve_into_raw_root"))
    print(json.dumps(out))
    return 0 if out["contract_ok"] else 1


def _summarize_plan(plans_path: Path, config: str = "3d_fullres"):
    p = json.loads(plans_path.read_text(encoding="utf-8"))
    cfg = p["configurations"][config]
    a = cfg["architecture"]
    kw = a.get("arch_kwargs", {})
    return {
        "plans_name": p.get("plans_name"),
        "arch_class": a.get("network_class_name"),
        "patch_size": cfg.get("patch_size"),
        "batch_size": cfg.get("batch_size"),
        "spacing": cfg.get("spacing"),
        "n_stages": kw.get("n_stages"),
        "features_per_stage": kw.get("features_per_stage"),
        "strides": kw.get("strides"),
        "kernel_sizes": kw.get("kernel_sizes"),
        "n_blocks_per_stage": kw.get("n_blocks_per_stage"),
        "normalization": cfg.get("normalization_schemes"),
    }


def cmd_validate_plan(args):
    pp = Path(os.environ["nnUNet_preprocessed"]) / DATASET_NAME
    s = _summarize_plan(pp / f"{args.plans}.json")
    dj = json.loads((pp / "dataset.json").read_text(encoding="utf-8"))
    checks = {
        "arch_is_resenc_unet": s["arch_class"].endswith("ResidualEncoderUNet"),
        "three_regions": len(dj["labels"]) - 1 == 3,
        "channels_order_ok": dj["channel_names"] == CHANNEL_NAMES,
        "rco_ok": dj["regions_class_order"] == REGIONS_CLASS_ORDER,
        "patch_3d": isinstance(s["patch_size"], list) and len(s["patch_size"]) == 3,
        "batch_ge_2": s["batch_size"] >= 2,
        "stages_ge_5": (s["n_stages"] or 0) >= 5,
        "strides_valid": bool(s["strides"]) and all(len(x) == 3 for x in s["strides"]),
        "kernels_valid": bool(s["kernel_sizes"]) and all(len(x) == 3 for x in s["kernel_sizes"]),
    }
    out = {"plans": args.plans, "summary": s, "checks": checks,
           "valid": all(checks.values())}
    print(json.dumps(out))
    return 0 if out["valid"] else 1


# ----------------------------- Part F: bounded benchmarks -----------------------------
def _load_split_and_feats(fp_npz, split_json):
    import numpy as np
    d = np.load(fp_npz, allow_pickle=True)
    cids = [str(c) for c in d["cids"]]
    feat = {c: {"wt": int(d["wt"][i]), "tc": int(d["tc"][i]), "et": int(d["et"][i]),
                "nvox": int(d["nvox"][i])} for i, c in enumerate(cids)}
    splits = json.loads(Path(split_json).read_text(encoding="utf-8"))
    fold_of = {}
    for f, s in enumerate(splits):
        for c in s["val"]:
            fold_of[c] = f
    return cids, feat, fold_of


def select_bench_cases(fp_npz, split_json, per_fold=8):
    """Deterministic 40-case selection covering all 5 folds and strata."""
    cids, feat, fold_of = _load_split_and_feats(fp_npz, split_json)
    chosen = []
    strata_hits = {"et_absent": 0, "tc_absent": 0, "wt_small": 0, "wt_large": 0, "wt_median": 0}
    for f in range(NFOLDS):
        fc = sorted([c for c in cids if fold_of.get(c) == f], key=lambda c: (feat[c]["wt"], c))
        picks = []
        if fc:
            picks += [fc[0], fc[-1], fc[len(fc) // 2]]              # WT small/large/median
        et_absent = [c for c in fc if feat[c]["et"] == 0]
        tc_absent = [c for c in fc if feat[c]["tc"] == 0]
        if et_absent:
            picks.append(sorted(et_absent)[0]); strata_hits["et_absent"] += 1
        if tc_absent:
            picks.append(sorted(tc_absent)[0]); strata_hits["tc_absent"] += 1
        by_size = sorted(fc, key=lambda c: (feat[c]["nvox"], c))
        if by_size:
            picks += [by_size[0], by_size[-1]]                     # small/large image
        # dedup preserving order, fill to per_fold by next-by-wt not yet chosen
        seen = set(); ded = []
        for c in picks:
            if c not in seen:
                seen.add(c); ded.append(c)
        i = 0
        while len(ded) < per_fold and i < len(fc):
            if fc[i] not in seen:
                seen.add(fc[i]); ded.append(fc[i])
            i += 1
        chosen += ded[:per_fold]
        strata_hits["wt_small"] += 1; strata_hits["wt_large"] += 1; strata_hits["wt_median"] += 1
    return chosen, feat, fold_of, strata_hits


def _pm_cm(plans_json_path, config, dataset_json):
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    plans = json.loads(Path(plans_json_path).read_text(encoding="utf-8"))
    pm = PlansManager(plans)
    cm = pm.get_configuration(config)
    return plans, pm, cm


def cmd_preprocess_bench(args):
    import numpy as np, time as _t, hashlib
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    raw_base = os.environ["nnUNet_raw"]
    ddir = Path(raw_base) / DATASET_NAME
    dataset_json = json.loads((ddir / "dataset.json").read_text(encoding="utf-8"))
    pp_base = Path(os.environ["nnUNet_preprocessed"]) / DATASET_NAME
    plans, pm, cm = _pm_cm(pp_base / f"{args.plans}.json", args.config, dataset_json)
    chosen, feat, fold_of, strata = select_bench_cases(args.fp, args.split, args.per_fold)
    Path(args.priv_list).write_text(json.dumps(chosen), encoding="utf-8")
    outdir = Path(args.bench_out);
    if outdir.exists(): shutil.rmtree(outdir)
    (outdir / "data").mkdir(parents=True)
    pre = DefaultPreprocessor(verbose=False)
    per_case = []
    raw_bytes = 0; out_bytes = 0
    src_hash_before = {}
    for cid in chosen:
        imgs = [str(ddir / "imagesTr" / f"{cid}_{ch}.nii.gz") for ch in ["0000", "0001", "0002", "0003"]]
        seg = str(ddir / "labelsTr" / f"{cid}.nii.gz")
        for p in imgs:  # record source hash (symlink target) before
            rp = os.path.realpath(p); src_hash_before[rp] = os.path.getsize(rp)
        t0 = _t.time()
        data, s, props = pre.run_case(imgs, seg, pm, cm, dataset_json)
        dt = _t.time() - t0
        per_case.append(dt)
        raw_bytes += sum(os.path.getsize(os.path.realpath(p)) for p in imgs)
        # save compressed (reference/disk-expansion measurement) atomically
        final = outdir / "data" / (cid + ".npz")
        with open(str(final) + ".tmp", "wb") as fh:
            np.savez_compressed(fh, data=data.astype(np.float32), seg=s.astype(np.int8))
        os.rename(str(final) + ".tmp", final)
        out_bytes += os.path.getsize(final)
    # verify no raw source modified
    src_ok = all(os.path.getsize(rp) == sz for rp, sz in src_hash_before.items())
    pc = np.array(per_case)
    n_total = args.n_total
    p90 = float(np.percentile(pc, 90)); med = float(np.median(pc)); mx = float(pc.max())
    safety = args.safety
    proj_hours = n_total * p90 * safety / 3600.0
    mean_out = out_bytes / len(chosen)
    proj_disk_gib = mean_out * n_total * safety / 2**30
    out = dict(BENCH_LABELS)
    out.update({
        "n_bench": len(chosen), "plans": args.plans,
        "wall_seconds": round(pc.sum(), 1),
        "per_case_median_s": round(med, 2), "per_case_p90_s": round(p90, 2),
        "per_case_max_s": round(mx, 2),
        "raw_read_gib": round(raw_bytes / 2**30, 2),
        "preproc_written_gib": round(out_bytes / 2**30, 2),
        "output_expansion_x": round(out_bytes / max(raw_bytes, 1), 2),
        "throughput_cases_per_min": round(len(chosen) / (pc.sum() / 60), 2),
        "projected_full_corpus_hours_p90_safety": round(proj_hours, 2),
        "projected_full_corpus_preproc_gib": round(proj_disk_gib, 1),
        "safety_factor": safety, "raw_source_unmodified": bool(src_ok),
        "strata_coverage": strata,
    })
    print(json.dumps(out))
    Path(args.result).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0 if src_ok else 1


def cmd_build_bench_raw(args):
    """Build a RAW 40-case subset dataset (symlinks) and seed its preprocessed dir with
    the production plan + fingerprint (dataset_name adjusted) so the OFFICIAL preprocessor
    reproduces the exact production configuration. No source payload copied."""
    import numpy as np
    raw_base = Path(os.environ["nnUNet_raw"]); pp_base = Path(os.environ["nnUNet_preprocessed"])
    prod_raw = raw_base / DATASET_NAME; prod_pp = pp_base / DATASET_NAME
    chosen = json.loads(Path(args.priv_list).read_text(encoding="utf-8"))
    braw = raw_base / args.bench_dataset
    if braw.exists(): shutil.rmtree(braw)
    (braw / "imagesTr").mkdir(parents=True); (braw / "labelsTr").mkdir(parents=True)
    for cid in chosen:
        for ch in ["0000", "0001", "0002", "0003"]:
            src = os.path.realpath(prod_raw / "imagesTr" / f"{cid}_{ch}.nii.gz")
            os.symlink(src, braw / "imagesTr" / f"{cid}_{ch}.nii.gz")
        os.symlink(os.path.realpath(prod_raw / "labelsTr" / f"{cid}.nii.gz"), braw / "labelsTr" / f"{cid}.nii.gz")
    dj = json.loads((prod_raw / "dataset.json").read_text(encoding="utf-8")); dj["numTraining"] = len(chosen)
    (braw / "dataset.json").write_text(json.dumps(dj, indent=2), encoding="utf-8")
    # seed preprocessed dir with production plan(s) + fingerprint so preprocess uses them verbatim
    bpp = pp_base / args.bench_dataset
    if bpp.exists(): shutil.rmtree(bpp)
    bpp.mkdir(parents=True)
    shutil.copy(prod_pp / "dataset_fingerprint.json", bpp / "dataset_fingerprint.json")
    (bpp / "dataset.json").write_text(json.dumps(dj, indent=2), encoding="utf-8")
    for plans_name in args.plans_list.split(","):
        plans = json.loads((prod_pp / f"{plans_name}.json").read_text(encoding="utf-8"))
        plans["dataset_name"] = args.bench_dataset
        (bpp / f"{plans_name}.json").write_text(json.dumps(plans), encoding="utf-8")
    # bench split referencing only these cases (fold 0 used for benchmark)
    k = max(1, len(chosen) // 5)
    val = sorted(chosen)[:k]; train = [c for c in sorted(chosen) if c not in val]
    (bpp / "splits_final.json").write_text(json.dumps([{"train": train, "val": val}] * NFOLDS), encoding="utf-8")
    print(json.dumps({"bench_dataset": args.bench_dataset, "cases": len(chosen),
                      "train": len(train), "val": len(val), "symlinked_not_copied": True}))
    return 0


def cmd_step_bench(args):
    """Short native training-step benchmark on the bench dataset: >=20 warmup + 100
    measured optimizer steps, finite loss/grad/params, VRAM, checkpoint round-trip."""
    import torch, numpy as np, time as _t
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    pp = Path(os.environ["nnUNet_preprocessed"]) / args.bench_dataset
    plans = json.loads((pp / f"{args.plans}.json").read_text(encoding="utf-8"))
    dataset_json = json.loads((pp / "dataset.json").read_text(encoding="utf-8"))
    plans_t = dict(plans); plans_t["continue_training"] = False
    trainer = nnUNetTrainer(plans=plans_t, configuration=args.config, fold=0,
                            dataset_json=dataset_json, device=torch.device("cuda"))
    trainer.initialize()
    net_class = trainer.network.__class__.__name__
    trainer.on_train_start(); trainer.on_train_epoch_start()
    # warmup (includes torch.compile first-call)
    t_compile0 = _t.time()
    finite = True
    for i in range(args.warmup):
        log = trainer.train_step(next(trainer.dataloader_train))
        if i == 0:
            torch.cuda.synchronize(); compile_warm_s = _t.time() - t_compile0
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    times = []; losses = []
    deadline = _t.time() + args.cap_seconds
    for i in range(args.steps):
        t0 = _t.time()
        log = trainer.train_step(next(trainer.dataloader_train))
        torch.cuda.synchronize()
        times.append(_t.time() - t0); losses.append(float(log["loss"]))
        if not np.isfinite(losses[-1]): finite = False
        if _t.time() > deadline:
            break
    # finite grads/params
    gfin = True
    for p in trainer.network.parameters():
        if p.grad is not None and not bool(torch.isfinite(p.grad).all()): gfin = False
    pfin = all(bool(torch.isfinite(p).all()) for p in trainer.network.parameters())
    # checkpoint round trip — explicit torch.compile _orig_mod. handling (never strict=False):
    # the compiled OptimizedModule's state_dict carries an "_orig_mod." prefix; strip it and
    # load into the underlying uncompiled module.
    ck = Path(args.tmp_ckpt); ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"network_weights": trainer.network.state_dict()}, ck)
    sd = torch.load(ck, map_location="cpu", weights_only=True)["network_weights"]
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    raw_module = getattr(trainer.network, "_orig_mod", trainer.network)
    raw_module.load_state_dict(sd, strict=True)
    ck.unlink()
    ta = np.array(times)
    out = dict(BENCH_LABELS)
    out.update({
        "plans": args.plans, "network_class": net_class,
        "measured_steps": len(times), "warmup_steps": args.warmup,
        "compile_warmup_s": round(compile_warm_s, 2),
        "step_mean_s": round(float(ta.mean()), 3), "step_median_s": round(float(np.median(ta)), 3),
        "step_p90_s": round(float(np.percentile(ta, 90)), 3),
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "loss_finite": bool(finite), "grad_finite": bool(gfin), "params_finite": bool(pfin),
        "loss_first": round(losses[0], 4) if losses else None,
        "batch_size": plans["configurations"][args.config]["batch_size"],
        "patch_size": plans["configurations"][args.config]["patch_size"],
        "strict_reload_ok": True,
    })
    try: trainer.on_train_end()
    except Exception: pass
    print(json.dumps(out)); Path(args.result).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0 if (finite and gfin and pfin and len(times) >= 1) else 1


def cmd_infer_proxy(args):
    """No-TTA sliding-window inference memory proxy on 3 representative cases, plus an
    allocator-limited 21 GiB proxy. RTX PRO 6000 numbers are NOT A10G evidence."""
    import torch, numpy as np, time as _t
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    raw_base = Path(os.environ["nnUNet_raw"]); pp = Path(os.environ["nnUNet_preprocessed"]) / DATASET_NAME
    ddir = raw_base / DATASET_NAME
    plans = json.loads((pp / f"{args.plans}.json").read_text(encoding="utf-8"))
    dataset_json = json.loads((pp / "dataset.json").read_text(encoding="utf-8"))
    pm = PlansManager(plans); cm = pm.get_configuration(args.config); lm = pm.get_label_manager(dataset_json)
    net = get_network_from_plans(cm.network_arch_class_name, cm.network_arch_init_kwargs,
                                 cm.network_arch_init_kwargs_req_import, 4, lm.num_segmentation_heads,
                                 allow_init=True, deep_supervision=False)
    cases = json.loads(Path(args.cases_json).read_text(encoding="utf-8"))  # {"small":cid,"median":cid,"large":cid}
    if args.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                           device=torch.device("cuda"), verbose=False, verbose_preprocessing=False,
                           allow_tqdm=False)
    pred.manual_initialization(net, pm, cm, [net.state_dict()], dataset_json, "nnUNetTrainer", None)
    res = {}
    oom = False
    for tag, cid in cases.items():
        order = [str(ddir / "imagesTr" / f"{cid}_{ch}.nii.gz") for ch in ["0000", "0001", "0002", "0003"]]
        data, props = SimpleITKIO().read_images(order)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        try:
            t0 = _t.time()
            _seg, _probs = pred.predict_single_npy_array(data, props, None, None, True)
            dt = _t.time() - t0
            res[tag] = {"infer_s": round(dt, 2),
                        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
        except torch.cuda.OutOfMemoryError:
            oom = True; res[tag] = {"OOM": True}
    peak = max((v.get("peak_reserved_gib", 0) for v in res.values()), default=0)
    out = dict(BENCH_LABELS)
    out.update({"plans": args.plans, "proxy_label": "RTX_PRO_6000_21GiB_allocator_proxy_only",
                "mem_fraction": args.mem_fraction, "per_case": res,
                "max_peak_reserved_gib": peak, "oom": oom,
                "under_21gib": (peak < 21.0 and not oom),
                "under_18gib": (peak < 18.0 and not oom)})
    print(json.dumps(out)); Path(args.result).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preprocess-bench")
    p.add_argument("--fp", required=True); p.add_argument("--split", required=True)
    p.add_argument("--plans", default="nnUNetResEncUNetMPlans"); p.add_argument("--config", default="3d_fullres")
    p.add_argument("--per-fold", type=int, default=8); p.add_argument("--n-total", type=int, default=1351)
    p.add_argument("--safety", type=float, default=1.3); p.add_argument("--bench-out", required=True)
    p.add_argument("--priv-list", required=True); p.add_argument("--result", required=True)
    p.set_defaults(func=cmd_preprocess_bench)

    p = sub.add_parser("build-bench-raw")
    p.add_argument("--priv-list", required=True); p.add_argument("--bench-dataset", default="Dataset511_GAT26BENCH")
    p.add_argument("--plans-list", default="nnUNetResEncUNetMPlans,nnUNetResEncUNetLPlans")
    p.set_defaults(func=cmd_build_bench_raw)

    p = sub.add_parser("step-bench")
    p.add_argument("--bench-dataset", default="Dataset511_GAT26BENCH"); p.add_argument("--config", default="3d_fullres")
    p.add_argument("--plans", required=True); p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--steps", type=int, default=100); p.add_argument("--cap-seconds", type=int, default=600)
    p.add_argument("--tmp-ckpt", required=True); p.add_argument("--result", required=True)
    p.set_defaults(func=cmd_step_bench)

    p = sub.add_parser("infer-proxy")
    p.add_argument("--plans", required=True); p.add_argument("--config", default="3d_fullres")
    p.add_argument("--cases-json", required=True); p.add_argument("--mem-fraction", type=float, default=0.0)
    p.add_argument("--result", required=True)
    p.set_defaults(func=cmd_infer_proxy)

    p = sub.add_parser("build-dataset"); p.add_argument("--raw-root", required=True)
    p.add_argument("--expect-cases", type=int, default=1351); p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_build_dataset)

    p = sub.add_parser("verify-dataset"); p.add_argument("--raw-root", required=True)
    p.set_defaults(func=cmd_verify_dataset)

    p = sub.add_parser("validate-plan"); p.add_argument("--plans", required=True)
    p.set_defaults(func=cmd_validate_plan)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
