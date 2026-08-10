#!/usr/bin/env python3
"""GAT-26 G4 — build a private two-case SMOKE-ONLY nnU-Net raw dataset (hardened).

Exactly pilot_case_A + pilot_case_B. Synthetic 5-digit aliases (evaluator-compatible).
Atomic promote; fails closed leaving no partial dataset; source files never modified.
Never prints/commits real case IDs, filenames, or hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

CHANNELS = OrderedDict([("0000", "t1n"), ("0001", "t1c"), ("0002", "t2w"), ("0003", "t2f")])
REQUIRED = ("t1n", "t1c", "t2w", "t2f", "seg")
# Synthetic 5-digit aliases (end in 5 digits so the official evaluator does not skip them).
PILOT_ALIASES = OrderedDict([("pilot_case_A", "GAT26SMOKE_80000"),
                             ("pilot_case_B", "GAT26SMOKE_80001")])


def build_dataset_json(num_training: int) -> "OrderedDict":
    d = OrderedDict()
    d["channel_names"] = OrderedDict([("0", "T1n"), ("1", "T1c"), ("2", "T2w"), ("3", "T2f")])
    labels = OrderedDict()
    labels["background"] = 0
    labels["whole_tumor"] = [1, 2, 3]
    labels["tumor_core"] = [1, 3]
    labels["enhancing_tumor"] = [3]
    d["labels"] = labels
    d["regions_class_order"] = [2, 1, 3]
    d["numTraining"] = num_training
    d["file_ending"] = ".nii.gz"
    d["name"] = "Dataset777_GAT26G4SMOKE"
    d["smoke_only"] = True
    return d


def verify_dataset_json(d: dict) -> list[str]:
    errs = []
    if list(d["labels"].keys()) != ["background", "whole_tumor", "tumor_core", "enhancing_tumor"]:
        errs.append("label keys reordered")
    if d["labels"]["whole_tumor"] != [1, 2, 3]:
        errs.append("whole_tumor != [1,2,3]")
    if d["labels"]["tumor_core"] != [1, 3]:
        errs.append("tumor_core != [1,3]")
    if d["labels"]["enhancing_tumor"] != [3]:
        errs.append("enhancing_tumor != [3]")
    if d["regions_class_order"] != [2, 1, 3]:
        errs.append("regions_class_order != [2,1,3]")
    if list(d["channel_names"].keys()) != ["0", "1", "2", "3"]:
        errs.append("channel order wrong")
    return errs


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def validate_mapping(mapping: dict, raw_root: Path) -> None:
    """Fail-closed checks on the private pilot mapping before any filesystem change."""
    if list(mapping.keys()) != ["pilot_case_A", "pilot_case_B"]:
        raise ValueError("mapping keys must be exactly pilot_case_A, pilot_case_B")
    real_ids = set()
    seen_paths = set()
    raw_root = raw_root.resolve()
    for pseudonym, mods in mapping.items():
        if set(mods.keys()) != set(REQUIRED):
            raise ValueError(f"{pseudonym}: modalities must be exactly {REQUIRED}")
        # derive a coarse real-id token to check alias non-collision + distinctness
        real_ids.add(Path(mods["seg"]).name)
        for m in REQUIRED:
            p = Path(mods[m]).resolve()
            if not p.exists():
                raise ValueError(f"{pseudonym}/{m}: source path does not exist")
            if raw_root not in p.parents and p != raw_root:
                raise ValueError(f"{pseudonym}/{m}: source escapes authorized raw root")
            if str(p) in seen_paths:
                raise ValueError("two pilots share a source file")
            seen_paths.add(str(p))
    if len(real_ids) != 2:
        raise ValueError("the two pilots must be distinct source cases")
    # aliases must be synthetic and not collide with any real filename token
    for alias in PILOT_ALIASES.values():
        for rid in real_ids:
            if alias in rid:
                raise ValueError("synthetic alias collides with a real case identifier")


def build(mapping: dict, raw_root: Path, dataset_root: Path, dataset_name: str,
          link: bool = True) -> dict:
    """Build into a temp dir; validate; atomically promote. No partial on failure."""
    validate_mapping(mapping, raw_root)
    final = dataset_root / dataset_name
    if final.exists() and any(final.iterdir()):
        raise ValueError("target dataset already exists and is non-empty; refusing to overwrite")
    tmp = Path(tempfile.mkdtemp(prefix="g4ds_", dir=str(dataset_root)))
    try:
        images = tmp / "imagesTr"; labels = tmp / "labelsTr"
        images.mkdir(parents=True); labels.mkdir(parents=True)
        provenance = {}
        for pseudonym, mods in mapping.items():
            alias = PILOT_ALIASES[pseudonym]
            prov = provenance.setdefault(pseudonym, {"alias": alias, "in": {}})
            for ch, m in CHANNELS.items():
                src = Path(mods[m]); prov["in"][m] = _sha256(src)
                dst = images / f"{alias}_{ch}.nii.gz"
                (dst.symlink_to(src.resolve()) if link else shutil.copy2(src, dst))
            src = Path(mods["seg"]); prov["in"]["seg"] = _sha256(src)
            dst = labels / f"{alias}.nii.gz"
            (dst.symlink_to(src.resolve()) if link else shutil.copy2(src, dst))
        dj = build_dataset_json(len(mapping))
        errs = verify_dataset_json(dj)
        if errs:
            raise ValueError(f"dataset.json contract failed: {errs}")
        (tmp / "dataset.json").write_text(json.dumps(dj, indent=2) + "\n", encoding="utf-8")
        # re-verify source hashes unchanged after linking/copy
        for pseudonym, mods in mapping.items():
            for m in REQUIRED:
                if _sha256(Path(mods[m])) != provenance[pseudonym]["in"][m]:
                    raise ValueError("source hash changed during build")
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(final)  # atomic same-filesystem promote
        return {"dataset_dir": str(final), "provenance": provenance, "dataset_json_ok": True}
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--raw-root", required=True, help="authorized protected raw root")
    parser.add_argument("--dataset-root", required=True, help="nnUNet_raw root (private/ignored)")
    parser.add_argument("--dataset-name", default="Dataset777_GAT26G4SMOKE")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    mapping = json.loads(Path(args.mapping).read_text())
    res = build(mapping, Path(args.raw_root), Path(args.dataset_root), args.dataset_name,
                link=not args.copy)
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({"dataset_json_ok": res["dataset_json_ok"], "num_cases": len(mapping),
                      "smoke_only": True}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
