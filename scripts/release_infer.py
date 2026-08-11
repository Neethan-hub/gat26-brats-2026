#!/usr/bin/env python3
"""GAT-26 clean-room release inference runner (Task 3 / BraTS-GoAT).

Container entrypoint contract (frozen G5 selection = ResEnc-M):
  * iterate every case folder under a read-only /input; write exactly one flat .nii.gz per case
    to /output (no sub-folders); the output name is derived dynamically from the COMPLETE input
    case-folder basename, which is treated as an opaque identifier — no case ID is parsed out of
    it, no trailing-digit count is required and no folder prefix is assumed.
  * ResEnc-M plan only; random-init-trained checkpoints only; NO external weights.
  * SEQUENTIAL five-checkpoint ensemble (mean of region probabilities) — one checkpoint resident at
    a time, freed before the next, so peak VRAM is a single model, not five.
  * frozen inference rules ONLY: threshold 0.5, hierarchy-safe WT/TC/ET reconstruction, NO TTA
    (mirroring off), NO connected-component filtering, NO presence gate, NO learned threshold or
    learned ensemble weight, primary checkpoint_final.
  * strict modality discovery by suffix (independent of directory order); missing / duplicate /
    unknown modality FAILS (nonzero) BEFORE any output is written.
  * exact source geometry restored via nnU-Net's own SimpleITKIO writer.
  * deterministic; bounded temp with cleanup; explicit nonzero exit on any invalid input or
    incomplete output. No network / download / cache at runtime; weights baked into the image.

The pure helpers (modality discovery, naming, ensemble accumulation) are import-safe for unit
tests — heavy deps (torch / nnU-Net / SimpleITK) are imported only inside the GPU path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g4_reconstruct_validate as RC          # noqa: E402
import g3_audit_labeled_archive as AUD        # noqa: E402

REQUIRED_MODALITIES = ("t1n", "t1c", "t2w", "t2f")   # nnU-Net channel order for this dataset
PLANS = "nnUNetResEncUNetMPlans"                      # frozen selected plan (ResEnc-M) ONLY
CONFIG = "3d_fullres"
N_CHECKPOINTS = 5                                     # final design: five distinct fold checkpoints
TAU = 0.5                                             # frozen threshold; no learned threshold
DEFAULT_WEIGHTS_DIR = os.environ.get("GAT26_WEIGHTS_DIR", "/opt/gat26/weights")  # baked, not a worker path


class ReleaseInputError(Exception):
    """Fail-closed input error — the run must exit nonzero WITHOUT writing output for this case."""


# ------------------------------ pure, testable helpers ------------------------------
def discover_modalities(filenames):
    """Map input files to the required modalities by suffix, independent of directory order.
    FAIL CLOSED on missing, duplicate, or unknown modality. Returns paths in REQUIRED_MODALITIES
    order. `filenames` is an iterable of paths/str (only .nii.gz image files)."""
    imgs = [f for f in filenames if str(f).endswith((".nii.gz", ".nii"))]
    by_mod = {}
    for f in imgs:
        m = AUD.classify_modality(Path(f).name)
        if m == "unknown":
            raise ReleaseInputError(f"unknown modality file: {Path(f).name}")
        if m in by_mod:
            raise ReleaseInputError(f"duplicate modality {m}: {Path(by_mod[m]).name} and {Path(f).name}")
        by_mod[m] = f
    missing = [m for m in REQUIRED_MODALITIES if m not in by_mod]
    if missing:
        raise ReleaseInputError(f"missing modalities: {missing}")
    extra = [m for m in by_mod if m not in REQUIRED_MODALITIES]
    if extra:
        raise ReleaseInputError(f"unexpected modalities: {extra}")
    return [by_mod[m] for m in REQUIRED_MODALITIES]


import re as _re
# Characters that can never appear in a safe case-folder basename: path separators, NUL and any
# other control character. Everything else is opaque — no case ID is parsed from the basename.
_UNSAFE_BASENAME_RE = _re.compile(r"[\x00-\x1f\x7f/\\]")


def validate_case_folder_basename(case_folder_basename: str) -> str:
    """Accept an organizer case-folder basename as an OPAQUE identifier; return it UNCHANGED.

    The official Task-3 container contract derives each output name from the complete input-folder
    basename. It imposes no trailing-digit count and no folder prefix, so nothing is extracted,
    truncated, normalised or reconstructed here and the basename is preserved byte-for-byte.

    FAIL CLOSED only on names that are genuinely unsafe or structurally invalid: empty or
    whitespace-only, '.', '..' or any other name beginning with a dot (a hidden entry, which would
    also produce a hidden output), an embedded path separator, and NUL or any other control
    character."""
    name = str(case_folder_basename)
    if (not name.strip() or name.startswith(".")
            or _UNSAFE_BASENAME_RE.search(name) is not None):
        raise ReleaseInputError(f"invalid case folder basename: {case_folder_basename!r}")
    return name


def release_output_name(case_folder_basename: str) -> str:
    """Derive the flat output name from the COMPLETE input case-folder basename:
    `<complete basename>` -> `<complete basename>.nii.gz`, prefix-agnostic and digit-agnostic."""
    return f"{validate_case_folder_basename(case_folder_basename)}.nii.gz"


def plan_outputs(case_folders):
    """Validate every input case folder and map folder -> flat output name, FAIL CLOSED (before any
    inference/writing) on an invalid name or a duplicate/ambiguous output-name collision."""
    plan = {}
    seen = {}
    for folder in case_folders:
        name = release_output_name(Path(folder).name)
        if name in seen:
            raise ReleaseInputError(
                f"duplicate/ambiguous output name {name!r} from {Path(folder).name!r} and {seen[name]!r}")
        seen[name] = Path(folder).name
        plan[str(folder)] = name
    return plan


def list_case_folders(input_dir):
    """Every immediate sub-folder of /input is one case (sorted, deterministic).

    Structural safety: the declared input root is resolved once and every candidate directory must
    resolve to a path inside it, so a symlink that escapes /input FAILS CLOSED rather than being
    followed. Non-directory entries are not cases and are skipped. Hidden (leading-dot) entries are
    never organizer case folders; they are skipped with a notice rather than silently dropped."""
    root = Path(os.path.realpath(str(input_dir)))
    cases = []
    for p in sorted(Path(input_dir).iterdir(), key=lambda q: q.name):
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            print(f"NOTICE: skipping hidden entry under input root: {p.name!r}", file=sys.stderr)
            continue
        target = Path(os.path.realpath(str(p)))
        if target != root and root not in target.parents:
            raise ReleaseInputError(
                f"case folder resolves outside the input root: {p.name!r}")
        cases.append(p)
    return cases


def accumulate_mean(acc, prob):
    """Streaming ensemble accumulator: acc += prob (returns prob on first call). Divide by count
    afterwards. Keeps only ONE model's probs plus the running sum in memory."""
    if acc is None:
        return prob.copy()
    acc += prob
    return acc


# ------------------------------ GPU inference path ------------------------------
def _sha256_file(path):
    """SHA-256 of a file, streamed. Used ONLY to compare checkpoint content locally — the digest is
    never printed, returned, logged, or committed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_checkpoints(weights_dir, allow_duplicate_proxy=False):
    """Resolve exactly N_CHECKPOINTS checkpoint paths, FAIL CLOSED.

    A10G-2 FINAL mode (allow_duplicate_proxy=False): require exactly fold_0..fold_4/checkpoint_final.pth,
    each a REGULAR READABLE file, and require all five to have DISTINCT content (SHA-256 computed
    privately; five copied/identical files fail closed). Returns "five_distinct_fold_checkpoints".

    A10G-1 PROXY mode (allow_duplicate_proxy=True): permit ONLY the existing M fold-0 checkpoint
    replicated into five runtime slots. Returns "runtime_only_duplicate_weight_proxy"
    (no_accuracy_claim / not_final_ensemble). The digest is never emitted."""
    wd = Path(weights_dir)
    if allow_duplicate_proxy:
        m0 = wd / "fold_0" / "checkpoint_final.pth"
        if not (m0.is_file() and os.access(m0, os.R_OK)):
            raise ReleaseInputError("A10G-1 proxy mode requires a readable fold_0/checkpoint_final.pth")
        return [m0] * N_CHECKPOINTS, "runtime_only_duplicate_weight_proxy"

    slots = [wd / f"fold_{i}" / "checkpoint_final.pth" for i in range(N_CHECKPOINTS)]
    for p in slots:
        if not p.is_file():
            raise ReleaseInputError(
                f"final acceptance requires fold_0..fold_{N_CHECKPOINTS-1}/checkpoint_final.pth; "
                f"missing or not a regular file: {p.parent.name}/{p.name}")
        if not os.access(p, os.R_OK):
            raise ReleaseInputError(f"checkpoint not readable: {p.parent.name}/{p.name}")
    digests = [_sha256_file(p) for p in slots]                 # private; never emitted
    if len(set(digests)) != N_CHECKPOINTS:
        raise ReleaseInputError(
            "final acceptance requires five DISTINCT checkpoints — identical/duplicated content "
            "detected; pass --allow-duplicate-proxy for the A10G-1 smoke ONLY (never for acceptance)")
    return slots, "five_distinct_fold_checkpoints"


def _load_weights(ckpt_path):
    import torch
    # nnU-Net checkpoints embed a numpy scalar global in their logging block, so weights_only=True
    # cannot deserialize the whole file until it is allowlisted. Allowlist the numpy scalar and load
    # weights-only. The checkpoint is our OWN random-init-trained file, baked into the image at build
    # (zero runtime network), so a fallback full load is safe if the allowlist path still trips.
    safe = []
    for modname in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            safe.append(__import__(modname, fromlist=["scalar"]).scalar)
        except Exception:
            pass
    if safe:
        try:
            torch.serialization.add_safe_globals(safe)
        except Exception:
            pass
    try:
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except Exception:
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)  # trusted baked file
    w = ck["network_weights"]
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in w.items()}


def _build_predictor(plans, dataset_json, ckpt_weights):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    pm = PlansManager(plans)
    cm = pm.get_configuration(CONFIG)
    lm = pm.get_label_manager(dataset_json)
    num_in = len(dataset_json["channel_names"])
    net = get_network_from_plans(cm.network_arch_class_name, cm.network_arch_init_kwargs,
                                 cm.network_arch_init_kwargs_req_import, num_in,
                                 lm.num_segmentation_heads, allow_init=True, deep_supervision=False)
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,  # NO TTA
                           device=torch.device("cuda"), verbose=False,
                           verbose_preprocessing=False, allow_tqdm=False)
    pred.manual_initialization(net, pm, cm, [ckpt_weights], dataset_json, "nnUNetTrainer", None)
    return pred


def infer_case_ensemble(ckpt_paths, plans, dataset_json, ordered_files):
    """Sequential mean-probability ensemble over ckpt_paths. Returns (seg_uint8, props). One model
    resident at a time; freed before the next (memory-safe)."""
    import torch
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    data, props = SimpleITKIO().read_images(ordered_files)
    acc = None
    for cp in ckpt_paths:
        pred = _build_predictor(plans, dataset_json, _load_weights(cp))
        _, probs = pred.predict_single_npy_array(data, props, None, None, True)
        acc = accumulate_mean(acc, probs)
        del pred
        torch.cuda.empty_cache()
    acc /= float(len(ckpt_paths))
    seg = RC.project_and_reconstruct(acc[0], acc[1], acc[2], TAU, TAU, TAU)  # WT,TC,ET channel order
    return seg, props


def _check_gpu(count, name, require_name):
    """Pure GPU-gate logic: require exactly ONE CUDA GPU; if a name is required, it must match
    exactly. FAIL CLOSED otherwise (abort before inference)."""
    if count != 1:
        raise ReleaseInputError(f"expected exactly one CUDA GPU (need --gpus all), saw {count}")
    if require_name and name != require_name:
        raise ReleaseInputError(f"GPU name {name!r} != required {require_name!r} — abort before inference")
    return name


def verify_gpu(require_name=None):
    """Verify from inside the container that torch sees exactly one CUDA GPU (and, for a genuine-A10G
    run, that its name is exactly the required name). Returns the observed GPU name."""
    import torch
    if not torch.cuda.is_available():
        raise ReleaseInputError("no CUDA GPU visible to torch (did you pass --gpus all?)")
    return _check_gpu(torch.cuda.device_count(), torch.cuda.get_device_name(0), require_name)


def enforce_determinism():
    """Bit-exact, reproducible inference: fixed seeds + deterministic cuDNN/cuBLAS. Without this,
    GPU convolutions use nondeterministic reduction orders and a few borderline voxels can flip
    across the 0.5 threshold between runs. warn_only=True keeps ops that lack a deterministic kernel
    from crashing while using deterministic implementations everywhere they exist."""
    import os as _os
    import random
    import numpy as np
    import torch
    _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required for deterministic cuBLAS
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/input")
    ap.add_argument("--output", default="/output")
    ap.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR)
    ap.add_argument("--dataset-json", required=True)   # baked dataset.json (channel order / labels)
    ap.add_argument("--plans-json", required=True)     # baked M plans
    ap.add_argument("--allow-duplicate-proxy", action="store_true")
    ap.add_argument("--require-gpu-name", default=os.environ.get("GAT26_REQUIRE_GPU_NAME"),
                    help="abort unless torch's single GPU has exactly this name (e.g. 'NVIDIA A10G')")
    ap.add_argument("--result-json", default=None)     # optional sanitized run summary
    args = ap.parse_args()

    import numpy as np
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    enforce_determinism()                                  # bit-exact, reproducible inference
    plans = json.loads(Path(args.plans_json).read_text())
    dataset_json = json.loads(Path(args.dataset_json).read_text())
    if plans.get("plans_name", PLANS) not in (PLANS,) and PLANS not in json.dumps(plans):
        print(f"FATAL: plans is not the selected {PLANS}", file=sys.stderr); return 3

    # GPU gate: exactly one CUDA GPU (+ exact name for a genuine-A10G run) BEFORE any inference.
    try:
        gpu_name = verify_gpu(args.require_gpu_name)
    except ReleaseInputError as e:
        print(f"FATAL gpu gate: {e}", file=sys.stderr); return 8

    ckpts, ckpt_mode = discover_checkpoints(args.weights_dir, args.allow_duplicate_proxy)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    try:
        cases = list_case_folders(args.input)
    except ReleaseInputError as e:
        print(f"FATAL invalid case-folder naming: {e}", file=sys.stderr); return 5
    if not cases:
        print("FATAL: no case folders under /input", file=sys.stderr); return 4

    # Validate EVERY case-folder name and detect duplicate/ambiguous output collisions BEFORE any
    # inference or writing (fail closed on the whole batch, not mid-way).
    try:
        out_names = plan_outputs(cases)
    except ReleaseInputError as e:
        print(f"FATAL invalid case-folder naming: {e}", file=sys.stderr); return 5

    summary = {"plan": PLANS, "checkpoint_mode": ckpt_mode, "n_checkpoints": len(ckpts),
               "gpu_name": gpu_name, "no_tta": True, "cc_filtering": False, "threshold": TAU,
               "n_cases": len(cases), "written": 0, "peak_reserved_gib": 0.0, "case_seconds": []}
    import torch
    for folder in cases:
        try:
            files = [p for p in folder.iterdir() if p.is_file()]
            ordered = discover_modalities(files)                       # fail-closed BEFORE inference
            name = out_names[str(folder)]
        except ReleaseInputError as e:
            print(f"FATAL invalid input for case {folder.name}: {e}", file=sys.stderr); return 5
        t0 = time.time(); torch.cuda.reset_peak_memory_stats()
        seg, props = infer_case_ensemble(ckpts, plans, dataset_json, ordered)
        out_path = out_dir / name
        SimpleITKIO().write_seg(np.ascontiguousarray(seg.astype("uint8")), str(out_path), props)
        # validate the written file against the source geometry; fail closed on any defect
        import nibabel as nib
        ref = nib.load(str(ordered[0]))
        v = RC.validate_output_file(out_path, ref.affine, ref.shape,
                                    ref.header.get_zooms(), nib.aff2axcodes(ref.affine),
                                    expected_name=name)
        if not v["ok"]:
            print(f"FATAL output validation failed for {folder.name}: {v}", file=sys.stderr); return 6
        summary["written"] += 1
        summary["case_seconds"].append(round(time.time() - t0, 2))
        summary["peak_reserved_gib"] = max(summary["peak_reserved_gib"],
                                           round(torch.cuda.max_memory_reserved() / 2**30, 2))

    # flat-output + completeness: exactly one .nii.gz per case, nothing else
    produced = sorted(out_dir.iterdir())
    flat_ok = all(f.is_file() and f.name.endswith(".nii.gz") for f in produced)
    if not (flat_ok and len(produced) == len(cases) and summary["written"] == len(cases)):
        print(f"FATAL incomplete/invalid output: produced={len(produced)} cases={len(cases)}",
              file=sys.stderr); return 7
    summary["flat_output_ok"] = True
    if args.result_json:
        Path(args.result_json).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("plan", "checkpoint_mode", "n_checkpoints", "n_cases",
                                              "written", "peak_reserved_gib", "flat_output_ok")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
