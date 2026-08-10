#!/usr/bin/env python3
"""Unit tests for G4 smoke dataset contract + fail-closed reconstruction/validator."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


DS = _load("g4_smoke_dataset")
RC = _load("g4_reconstruct_validate")


def test_dataset_json_contract():
    d = DS.build_dataset_json(2)
    errs = DS.verify_dataset_json(d)
    if list(d["labels"].keys()) != ["background", "whole_tumor", "tumor_core", "enhancing_tumor"]:
        errs.append("label insertion order not preserved")
    if d["regions_class_order"] != [2, 1, 3]:
        errs.append("regions_class_order wrong")
    if d["channel_names"] != {"0": "T1n", "1": "T1c", "2": "T2w", "3": "T2f"}:
        errs.append("channel mapping wrong")
    return errs


def test_channel_order_and_aliases():
    errs = []
    if list(DS.CHANNELS.items()) != [("0000", "t1n"), ("0001", "t1c"), ("0002", "t2w"), ("0003", "t2f")]:
        errs.append("channel mapping constant wrong")
    # B3: aliases must end in exactly five digits (evaluator-compatible)
    import re
    for a in DS.PILOT_ALIASES.values():
        if not re.search(r"\d{5}$", a):
            errs.append(f"alias {a} does not end in 5 digits")
    return errs


def test_region_round_trip():
    import numpy as np
    lab = np.array([0, 1, 2, 3], dtype="uint8").reshape(1, 1, 4)
    q_et = (lab == 3).astype("float32")
    q_tc = ((lab == 1) | (lab == 3)).astype("float32")
    q_wt = ((lab == 1) | (lab == 2) | (lab == 3)).astype("float32")
    rec = RC.project_and_reconstruct(q_wt, q_tc, q_et)
    return [] if np.array_equal(rec.reshape(-1), np.array([0, 1, 2, 3])) else ["round-trip failed"]


def test_nesting_unequal_thresholds():
    import numpy as np
    seg = RC.project_and_reconstruct(np.array([0.1]).reshape(1, 1, 1),
                                     np.array([0.2]).reshape(1, 1, 1),
                                     np.array([0.9]).reshape(1, 1, 1))
    v = RC.validate_mask(seg)
    return [] if (v["et_subset_tc"] and v["tc_subset_wt"]) else ["nesting not enforced"]


def test_validate_mask_failclosed():
    """B1: fractional/NaN/Inf/label4/negative rejected; float mask with integral values rejected."""
    import numpy as np
    errs = []
    good = np.zeros((4, 4, 4), dtype="uint8"); good[0, 0, 0] = 2; good[1, 1, 1] = 1; good[2, 2, 2] = 3
    if not RC.validate_mask(good)["ok"]:
        errs.append("valid uint8 mask rejected")
    good16 = good.astype("int16")
    if not RC.validate_mask(good16)["ok"]:
        errs.append("valid int16 mask rejected")
    # float32 mask with only 0/1/2/3 must be rejected as a stored output dtype
    fmask = good.astype("float32")
    if RC.validate_mask(fmask, require_integer_dtype=True)["ok"]:
        errs.append("float32 integral mask wrongly accepted as stored output")
    # fractional 1.5
    frac = good.astype("float32"); frac[0, 0, 0] = 1.5
    if RC.validate_mask(frac, require_integer_dtype=False)["ok"]:
        errs.append("fractional 1.5 accepted")
    # NaN / Inf
    nan = good.astype("float32"); nan[0, 0, 0] = np.nan
    if RC.validate_mask(nan, require_integer_dtype=False)["ok"]:
        errs.append("NaN accepted")
    inf = good.astype("float32"); inf[0, 0, 0] = np.inf
    if RC.validate_mask(inf, require_integer_dtype=False)["ok"]:
        errs.append("Inf accepted")
    # label 4 / negative
    l4 = good.copy(); l4[0, 0, 0] = 4
    if RC.validate_mask(l4)["ok"]:
        errs.append("label 4 accepted")
    neg = good.astype("int16"); neg[0, 0, 0] = -1
    if RC.validate_mask(neg)["ok"]:
        errs.append("negative label accepted")
    return errs


def test_task3_naming_strict():
    """B2: strict 5-digit naming; reject timepoint/4/6-digit/nested/alt-extension."""
    errs = []
    if RC.task3_output_name("01234") != "01234.nii.gz":
        errs.append("valid 5-digit failed")
    for bad in ("42", "123456", "abcde", "01234/x"):
        try:
            RC.task3_output_name(bad); errs.append(f"task3_output_name accepted bad {bad!r}")
        except ValueError:
            pass
    # basename validator
    if RC.validate_task3_basename("00042.nii.gz") != "00042":
        errs.append("valid basename rejected")
    for bad in ("00042-000.nii.gz", "0042.nii.gz", "000042.nii.gz",
                "sub/00042.nii.gz", "00042.nii", "00042.txt"):
        try:
            RC.validate_task3_basename(bad); errs.append(f"validate_task3_basename accepted {bad!r}")
        except ValueError:
            pass
    return errs


def test_builder_hardening():
    """B4: build fails closed on bad mapping; leaves no partial dataset."""
    import numpy as np
    import nibabel as nib
    errs = []
    tmp = Path(tempfile.mkdtemp())
    raw = tmp / "raw"; raw.mkdir()
    # make two synthetic cases with 5 modalities each
    mapping = {}
    for pn, cid in (("pilot_case_A", "AA"), ("pilot_case_B", "BB")):
        d = raw / cid; d.mkdir()
        m = {}
        for mm in ("t1n", "t1c", "t2w", "t2f"):
            p = d / f"{cid}-{mm}.nii.gz"; nib.save(nib.Nifti1Image(np.ones((6, 6, 6), "float32"), np.eye(4)), str(p)); m[mm] = str(p)
        p = d / f"{cid}-seg.nii.gz"; nib.save(nib.Nifti1Image(np.zeros((6, 6, 6), "int16"), np.eye(4)), str(p)); m["seg"] = str(p)
        mapping[pn] = m
    dsroot = tmp / "nnraw"; dsroot.mkdir()
    res = DS.build(mapping, raw, dsroot, "Dataset777_GAT26G4SMOKE", link=True)
    if not res["dataset_json_ok"]:
        errs.append("valid build failed")
    # imagesTr has 8 channel files, labelsTr 2
    ds = Path(res["dataset_dir"])
    if len(list((ds / "imagesTr").glob("*_0000.nii.gz"))) != 2:
        errs.append("channel 0000 count wrong")
    # bad mapping: wrong keys
    try:
        DS.build({"x": mapping["pilot_case_A"]}, raw, dsroot, "DatasetBad", link=True)
        errs.append("wrong keys accepted")
    except ValueError:
        pass
    if (dsroot / "DatasetBad").exists():
        errs.append("partial dataset left after failure")
    # missing modality
    bad = {k: dict(v) for k, v in mapping.items()}
    del bad["pilot_case_A"]["t1c"]
    try:
        DS.build(bad, raw, dsroot, "DatasetBad2", link=True)
        errs.append("missing modality accepted")
    except ValueError:
        pass
    return errs


def main() -> int:
    tests = [
        ("dataset.json contract", test_dataset_json_contract),
        ("channel order + 5-digit aliases", test_channel_order_and_aliases),
        ("region round-trip {0,1,2,3}", test_region_round_trip),
        ("nesting unequal thresholds", test_nesting_unequal_thresholds),
        ("B1 fail-closed mask validation", test_validate_mask_failclosed),
        ("B2 strict Task3 naming", test_task3_naming_strict),
        ("B4 builder hardening", test_builder_hardening),
    ]
    all_errs = []
    for name, fn in tests:
        errs = fn()
        print(f"[{'PASS' if not errs else 'FAIL'}] {name}")
        for e in errs:
            print(f"    - {e}")
        all_errs.extend(errs)
    print(f"\nG4 TESTS: {'PASS' if not all_errs else 'FAIL'} ({len(all_errs)} issue(s))")
    return 0 if not all_errs else 1


if __name__ == "__main__":
    sys.exit(main())
