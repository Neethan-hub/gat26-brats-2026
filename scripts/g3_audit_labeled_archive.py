#!/usr/bin/env python3
"""GAT-26 G3 — ZIP safety, path-safe extraction, and protected-data audit of the
labeled `train_with_gt` archive. Never prints member/case filenames or per-case
records. Pure logic (safety checks, modality mapping, region math, pilot selection)
is separated from I/O so it is unit-testable with synthetic inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
import zipfile
from pathlib import Path

# ---- ZIP safety limits ----
MAX_MEMBERS = 50_000
MAX_MEMBER_UNCOMPRESSED = 20 * 2**30           # 20 GiB
EXPANSION_RATIO_LIMIT = 1000                     # for members >= 100 MiB
EXPANSION_MIN_SIZE = 100 * 2**20                 # 100 MiB
MIN_FREE_GIB_AFTER = 50
MAX_TOTAL_LOGICAL_GIB = 250

# ---- modality convention (canonical BraTS-GoAT; unambiguous aliases only) ----
MOD_ALIASES = {
    "t1n": "t1n", "t1": "t1n", "t1native": "t1n",
    "t1c": "t1c", "t1ce": "t1c", "t1gd": "t1c", "t1contrast": "t1c",
    "t2w": "t2w", "t2": "t2w",
    "t2f": "t2f", "flair": "t2f", "t2flair": "t2f",
    "seg": "seg", "segmentation": "seg",
}
REQUIRED_MODS = ("t1n", "t1c", "t2w", "t2f")
LEGAL_LABELS = frozenset({0, 1, 2, 3})


def _norm(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def is_unsafe_member(zinfo: zipfile.ZipInfo) -> str | None:
    """Return a reason string if the member is unsafe, else None."""
    name = zinfo.filename
    if not name or name.endswith("/"):
        return None  # directory entry
    if name.startswith("/") or name.startswith("\\"):
        return "absolute path"
    if ".." in Path(name.replace("\\", "/")).parts:
        return "parent traversal"
    if "\\" in name or (len(name) > 1 and name[1] == ":"):
        return "windows/backslash path"
    if any(ord(c) < 32 for c in name):
        return "control/NUL char"
    mode = (zinfo.external_attr >> 16) & 0o170000
    if mode == 0o120000:
        return "symlink"
    if mode in (0o140000, 0o060000, 0o020000, 0o010000):
        return "socket/device/fifo"
    if zinfo.flag_bits & 0x1:
        return "encrypted entry"
    # An exec bit on a recognized data/text file is a benign archive artifact (BraTS
    # archives store .nii.gz as mode 0o700). Flag executables only for non-data
    # extensions, which is where a runnable payload would hide. We never execute any
    # extracted file regardless.
    exec_mode = (zinfo.external_attr >> 16) & 0o111
    lname = name.lower()
    is_data = lname.endswith((".nii.gz", ".nii", ".json", ".csv", ".txt", ".md"))
    if exec_mode and not is_data:
        return "executable entry (non-data)"
    if zinfo.file_size >= MAX_MEMBER_UNCOMPRESSED:
        return "member exceeds 20 GiB"
    if zinfo.file_size >= EXPANSION_MIN_SIZE and zinfo.compress_size > 0:
        if zinfo.file_size / zinfo.compress_size > EXPANSION_RATIO_LIMIT:
            return "suspicious expansion ratio"
    return None


def inspect_zip(zip_path: Path) -> dict:
    """Central-directory safety scan. Raises ValueError on any unsafe condition."""
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
    files = [z for z in infos if not z.filename.endswith("/")]
    if len(files) > MAX_MEMBERS:
        raise ValueError(f"too many members: {len(files)} > {MAX_MEMBERS}")
    seen: dict[str, int] = {}
    total_uncompressed = 0
    for z in files:
        reason = is_unsafe_member(z)
        if reason:
            raise ValueError(f"unsafe member ({reason})")  # no filename printed
        key = _norm(z.filename)
        seen[key] = seen.get(key, 0) + 1
        total_uncompressed += z.file_size
    dups = {k: v for k, v in seen.items() if v > 1}
    if dups:
        raise ValueError(f"{len(dups)} duplicate/case-fold-colliding normalized paths")
    total_gib = total_uncompressed / 2**30
    if total_gib > MAX_TOTAL_LOGICAL_GIB:
        raise ValueError(f"projected logical use {total_gib:.1f} GiB > {MAX_TOTAL_LOGICAL_GIB}")
    return {"member_count": len(files), "total_uncompressed_bytes": total_uncompressed}


def _safe_join(dest: Path, name: str) -> Path:
    target = (dest / name).resolve()
    if not str(target).startswith(str(dest.resolve()) + "/") and target != dest.resolve():
        raise ValueError("path escapes destination")
    return target


def safe_extract(zip_path: Path, staging_dir: Path, free_gib_now: float) -> dict:
    """Path-safe streaming extraction with CRC validation + per-member sha256."""
    info = inspect_zip(zip_path)
    projected_gib = info["total_uncompressed_bytes"] / 2**30
    if free_gib_now - projected_gib < MIN_FREE_GIB_AFTER:
        raise ValueError(f"extraction would leave < {MIN_FREE_GIB_AFTER} GiB free")
    staging_dir.mkdir(parents=True, exist_ok=True)
    member_hashes: dict[str, str] = {}
    extracted = 0
    total_bytes = 0
    with zipfile.ZipFile(zip_path) as zf:
        for z in zf.infolist():
            if z.filename.endswith("/"):
                _safe_join(staging_dir, z.filename).mkdir(parents=True, exist_ok=True)
                continue
            target = _safe_join(staging_dir, z.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with zf.open(z, "r") as src, open(target, "wb") as out:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    h.update(chunk)
                    out.write(chunk)
                    total_bytes += len(chunk)
            member_hashes[z.filename] = h.hexdigest()
            extracted += 1
    if extracted != info["member_count"]:
        raise ValueError(f"extracted {extracted} != expected {info['member_count']}")
    return {"member_count": extracted, "total_bytes": total_bytes,
            "member_hashes": member_hashes}


def classify_modality(filename: str) -> str:
    stem = filename.lower()
    for suf in (".nii.gz", ".nii"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    token = stem.split("-")[-1].split("_")[-1]
    return MOD_ALIASES.get(token, "unknown")


def parse_case_id(filename: str) -> str:
    base = filename.split("/")[-1]
    for suf in (".nii.gz", ".nii"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    parts = base.replace("_", "-").split("-")
    return "-".join(parts[:-1]) if len(parts) > 1 else base


def label_set_status(valset) -> str:
    """'label4' if legacy 4 present; 'illegal' if any non-{0,1,2,3}; else 'ok'."""
    vs = set(int(v) for v in valset)
    if 4 in vs:
        return "label4"
    if not vs.issubset(LEGAL_LABELS):
        return "illegal"
    return "ok"


def region_voxels(seg_flat) -> dict:
    """ET=3; TC={1,3}; WT={1,2,3}. Input: iterable of int label counts by value."""
    counts = seg_flat
    et = counts.get(3, 0)
    tc = counts.get(1, 0) + counts.get(3, 0)
    wt = counts.get(1, 0) + counts.get(2, 0) + counts.get(3, 0)
    return {"ET": et, "TC": tc, "WT": wt}


def select_pilots(cases: list[dict]) -> dict:
    """A: median WT among complete valid cases. B: smallest positive ET (fallback TC)."""
    valid = [c for c in cases if c.get("complete") and c.get("valid")]
    if not valid:
        return {"pilot_case_A": None, "pilot_case_B": None, "note": "no complete valid cases"}
    def tiebreak(c):
        return hashlib.sha256(c["case_id"].encode()).hexdigest()
    by_wt = sorted(valid, key=lambda c: (c["wt"], tiebreak(c)))
    a = by_wt[len(by_wt) // 2]  # median WT
    rest = [c for c in valid if c["case_id"] != a["case_id"]]
    et_pos = sorted([c for c in rest if c["et"] > 0], key=lambda c: (c["et"], tiebreak(c)))
    fallback = False
    if et_pos:
        b = et_pos[0]
    else:
        tc_pos = sorted([c for c in rest if c["tc"] > 0], key=lambda c: (c["tc"], tiebreak(c)))
        b = tc_pos[0] if tc_pos else None
        fallback = True
    return {
        "pilot_case_A": {"pseudonym": "pilot_case_A", "criteria": "median WT burden among complete valid labeled cases",
                         "_private_case_id": a["case_id"]},
        "pilot_case_B": ({"pseudonym": "pilot_case_B",
                          "criteria": ("smallest positive TC burden (ET fallback)" if fallback
                                       else "smallest positive ET burden among remaining complete valid cases"),
                          "_private_case_id": b["case_id"]} if b else None),
        "fallback_used": fallback,
    }


# ---- data audit (nibabel; runs on worker) ----
# Exhaustive checks. Tolerances (explicitly recorded): affine atol=1e-3,
# spacing atol=1e-3. Orientation compared via nibabel axis codes.
AFFINE_ATOL = 1e-3
SPACING_ATOL = 1e-3


def group_case_files(files):
    """Group by case with duplicate-modality/seg detection BEFORE insertion."""
    by_case: dict = {}
    dup_modality = 0
    dup_seg = 0
    unknown = 0
    for p in files:
        cid = parse_case_id(p.name)
        mod = classify_modality(p.name)
        slot = by_case.setdefault(cid, {})
        if mod == "unknown":
            unknown += 1
            slot.setdefault("_unknown", []).append(p)
            continue
        if mod in slot:  # never silently overwrite
            if mod == "seg":
                dup_seg += 1
            else:
                dup_modality += 1
            slot.setdefault("_dups", []).append((mod, p))
            continue
        slot[mod] = p
    return by_case, {"dup_modality": dup_modality, "dup_seg": dup_seg, "unknown": unknown}


def finite_full_array(arr) -> bool:
    """Exhaustive NaN/Inf check over EVERY voxel of an already-loaded array."""
    import numpy as np
    return bool(np.isfinite(arr).all())


def _array_content_hash(arr) -> str:
    """Deterministic canonical voxel-content hash from an in-memory array."""
    import numpy as np
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode()); h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def geometry_report(ref, img) -> dict:
    """Full geometry comparison of img vs ref (shape/affine/spacing/orientation)."""
    import numpy as np
    import nibabel as nib
    out = {"shape": img.shape == ref.shape,
           "dims3": len(img.shape) == 3 and all(s > 0 for s in img.shape),
           "affine_finite": bool(np.all(np.isfinite(img.affine))),
           "affine": bool(np.allclose(img.affine, ref.affine, atol=AFFINE_ATOL)),
           "spacing": bool(np.allclose(np.asarray(img.header.get_zooms()[:3]),
                                       np.asarray(ref.header.get_zooms()[:3]), atol=SPACING_ATOL)),
           "orientation": nib.aff2axcodes(img.affine) == nib.aff2axcodes(ref.affine)}
    return out


def validate_case(mods: dict) -> dict:
    """Exhaustive per-case validation. Each volume is loaded ONCE (memory-bounded per
    volume, not the whole dataset) and reused for finiteness + content hash + geometry."""
    import numpy as np
    import nibabel as nib
    rec = {"complete": False, "valid": False, "et": 0, "tc": 0, "wt": 0,
           "flags": {}, "content_hashes": {}, "qform_sform_discrepancy": False}
    have = [m for m in REQUIRED_MODS if m in mods]
    if len(have) != 4 or "seg" not in mods:
        rec["flags"]["missing_modality"] = True
        return rec
    rec["complete"] = True
    ref = nib.load(str(mods["t1n"]))
    geo_bad = 0; modality_nonfinite = 0
    for m in REQUIRED_MODS:
        img = nib.load(str(mods[m]))
        arr = np.asarray(img.dataobj)          # single decompression per volume
        g = geometry_report(ref, img)
        if not all(g.values()):
            geo_bad += 1
        if not finite_full_array(arr):
            modality_nonfinite += 1            # counts ONLY modality failures
        rec["content_hashes"][m] = _array_content_hash(arr)
        try:
            if ref.header.get("qform_code") != img.header.get("qform_code") or \
               ref.header.get("sform_code") != img.header.get("sform_code"):
                rec["qform_sform_discrepancy"] = True
        except Exception:
            pass
    seg = nib.load(str(mods["seg"]))
    seg_geo = geometry_report(ref, seg)
    if not all(seg_geo.values()):
        geo_bad += 1
    sdata = np.asarray(seg.dataobj)            # single load, reused below
    seg_nonfinite = not np.isfinite(sdata).all()   # counts ONLY segmentation failures
    rec["content_hashes"]["seg"] = _array_content_hash(sdata)
    sdata = sdata.astype("float64")
    label_bad = "ok"
    if not seg_nonfinite:
        if not np.allclose(np.round(sdata), sdata):
            label_bad = "fractional"
        else:
            vals = set(int(v) for v in np.unique(sdata.astype("int64")))
            label_bad = label_set_status(vals)
    rec["flags"] = {"geometry_bad": geo_bad, "modality_nonfinite": modality_nonfinite,
                    "seg_nonfinite": seg_nonfinite, "label_status": label_bad}
    if geo_bad == 0 and modality_nonfinite == 0 and not seg_nonfinite and label_bad == "ok":
        vals, cnts = np.unique(sdata.astype("int64"), return_counts=True)
        vc = {int(v): int(c) for v, c in zip(vals, cnts)}
        r = region_voxels(vc)
        rec.update(et=r["ET"], tc=r["TC"], wt=r["WT"], valid=True, label_counts=vc)
    return rec


def audit_extracted(root: Path) -> dict:
    files = [p for p in root.rglob("*") if p.is_file()
             and (p.name.endswith(".nii.gz") or p.name.endswith(".nii"))]
    by_case, group_anom = group_case_files(files)
    convention = "brats-goat-t1n/t1c/t2w/t2f/seg"
    anomalies = {"unknown_modality": group_anom["unknown"],
                 "duplicate_modality": group_anom["dup_modality"],
                 "duplicate_seg": group_anom["dup_seg"],
                 "missing_modality": 0, "geometry_mismatch": 0,
                 "modality_nonfinite": 0, "seg_nonfinite": 0,
                 "fractional_label": 0, "illegal_label": 0, "label4": 0,
                 "qform_sform_discrepancy": 0}
    label_totals: dict = {}
    region_case_counts = {"ET": 0, "TC": 0, "WT": 0}
    empty_counts = {"empty_mask": 0, "missing_ET": 0, "missing_TC": 0, "missing_WT": 0}
    cases = []
    all_hashes = {}
    canonical_file_count = 0
    for cid, mods in by_case.items():
        clean_mods = {k: v for k, v in mods.items() if not k.startswith("_")}
        canonical_file_count += len([k for k in clean_mods if k in REQUIRED_MODS or k == "seg"])
        rec = {"case_id": cid, "complete": False, "valid": False, "et": 0, "tc": 0, "wt": 0}
        vr = validate_case(clean_mods)
        rec["complete"] = vr["complete"]
        if not vr["complete"]:
            anomalies["missing_modality"] += 1
            cases.append(rec); continue
        f = vr["flags"]
        if f.get("geometry_bad"): anomalies["geometry_mismatch"] += 1
        if f.get("modality_nonfinite"):
            anomalies["modality_nonfinite"] += 1
        if f.get("seg_nonfinite"): anomalies["seg_nonfinite"] += 1
        if f.get("label_status") == "fractional": anomalies["fractional_label"] += 1
        elif f.get("label_status") == "label4": anomalies["label4"] += 1
        elif f.get("label_status") == "illegal": anomalies["illegal_label"] += 1
        if vr.get("qform_sform_discrepancy"): anomalies["qform_sform_discrepancy"] += 1
        for m, h in vr["content_hashes"].items():
            all_hashes.setdefault(h, 0)
            all_hashes[h] += 1
        if vr["valid"]:
            for k, v in vr.get("label_counts", {}).items():
                label_totals[k] = label_totals.get(k, 0) + v
            rec.update(et=vr["et"], tc=vr["tc"], wt=vr["wt"], valid=True)
            for reg in ("ET", "TC", "WT"):
                if rec[reg.lower()] > 0: region_case_counts[reg] += 1
                else: empty_counts[f"missing_{reg}"] += 1
            if rec["wt"] == 0: empty_counts["empty_mask"] += 1
        cases.append(rec)
    content_dupes = sum(c - 1 for c in all_hashes.values() if c > 1)
    pilots = select_pilots(cases)
    complete = sum(1 for c in cases if c["complete"])
    valid = sum(1 for c in cases if c["valid"])
    return {
        "convention": convention, "case_count": len(cases),
        "canonical_file_count": canonical_file_count,
        "complete_cases": complete, "incomplete_cases": len(cases) - complete,
        "valid_cases": valid, "label_set": sorted(label_totals.keys()),
        "label_voxel_totals": label_totals, "region_case_counts": region_case_counts,
        "empty_region_counts": empty_counts, "anomalies": anomalies,
        "voxel_content_duplicate_count": content_dupes,
        "tolerances": {"affine_atol": AFFINE_ATOL, "spacing_atol": SPACING_ATOL,
                       "finiteness": "exhaustive per-voxel (one full-array load per volume)"},
        "cohort": "UNKNOWN (not unambiguously derivable from filenames)",
        "near_duplicate_note": "Near-duplicate / subject-family analysis deferred to G4.5.",
        "pilots": pilots, "_private_cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("inspect", "extract", "audit"), required=True)
    parser.add_argument("--zip")
    parser.add_argument("--staging")
    parser.add_argument("--root")
    parser.add_argument("--free-gib", type=float, default=250.0)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.mode == "inspect":
        print(json.dumps(inspect_zip(Path(args.zip))))
    elif args.mode == "extract":
        res = safe_extract(Path(args.zip), Path(args.staging), args.free_gib)
        out = {"member_count": res["member_count"], "total_bytes": res["total_bytes"]}
        if args.out:
            Path(args.out).write_text(json.dumps(res) + "\n")
        print(json.dumps(out))
    elif args.mode == "audit":
        res = audit_extracted(Path(args.root))
        if args.out:
            Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
        san = {k: v for k, v in res.items() if not k.startswith("_")}
        san["pilots"] = {"pilot_case_A": res["pilots"].get("pilot_case_A", {}).get("criteria") if res["pilots"].get("pilot_case_A") else None,
                         "pilot_case_B": res["pilots"].get("pilot_case_B", {}).get("criteria") if res["pilots"].get("pilot_case_B") else None,
                         "fallback_used": res["pilots"].get("fallback_used")}
        print(json.dumps(san))
    return 0


if __name__ == "__main__":
    sys.exit(main())
