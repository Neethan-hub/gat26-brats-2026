#!/usr/bin/env python3
"""G88 — container case-folder naming contract (`python3 tests/test_g88_container_naming.py`).

The submitted C0 container aborted on the organizers' hidden test set with

    FATAL invalid case-folder naming: case folder must end in EXACTLY 5 digits
    (got 3-digit run): 'BraTS-xxx-xxxxx-xxx'

because the release runner extracted a trailing five-digit case ID from the input-folder
basename. The official contract instead derives each output name from the COMPLETE input-folder
basename, which is an opaque identifier.

These tests pin that contract: they fail against the submitted implementation and pass only after
the repair. Every identifier here is synthetic — no real BraTS case, image, label, metric, split
membership or prediction is used.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import release_infer as X                       # noqa: E402
import g4_reconstruct_validate as RC            # noqa: E402

FAILS = 0


def check(name, cond):
    global FAILS
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILS += 1


def raises(fn, exc=X.ReleaseInputError):
    try:
        fn()
        return False
    except exc:
        return True


# Synthetic organizer-style case folders that MUST be accepted. At least three distinct prefixes,
# and trailing runs of three, five, seven and zero digits.
ACCEPTED = [
    "BraTS-xxx-xxxxx-xxx",      # the organizers' own redacted folder, verbatim
    "BraTS-GoAT-99999-100",     # five-digit id + three-digit timepoint
    "BraTS-GoAT-99999",         # exactly five trailing digits
    "SYNTHCOHORT-00042",        # safe second prefix, five trailing digits
    "SYNTHCOHORT-042",          # safe second prefix, three trailing digits
    "SYNTH-case-alpha",         # safe non-numeric suffix
    "LAB7-2026-0000001",        # safe third prefix, seven trailing digits
    "case_000",                 # underscore separator, three trailing digits
]

# Names that must FAIL CLOSED: unsafe or structurally invalid, never merely unusual.
REJECTED = [
    "",
    "   ",
    ".",
    "..",
    ".hidden-case",
    "a/BraTS-GoAT-99999",
    "a\\BraTS-GoAT-99999",
    "BraTS-GoAT-99999\x00evil",
    "BraTS-GoAT\n99999",
]


def _write_nifti(path: Path, shape=(8, 9, 10), fill=0):
    import numpy as np
    import nibabel as nib
    arr = np.full(shape, fill, dtype="uint8")
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(arr, affine), str(path))
    return affine, shape


def _tree_hashes(root: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
        else:
            out[str(p.relative_to(root)) + "/"] = "dir"
    return out


def main():                                                          # noqa: C901
    print("test_g88_container_naming:")

    # 1. the complete basename is preserved byte-for-byte and echoed as the output stem
    for name in ACCEPTED:
        got = X.validate_case_folder_basename(name)
        check(f"basename_preserved[{name}]",
              got == name and got.encode() == name.encode())
        check(f"output_name[{name}]", X.release_output_name(name) == f"{name}.nii.gz")

    # no truncation, no numeric extraction, no hardcoded prefix
    check("no_truncation_of_timepoint",
          X.release_output_name("BraTS-GoAT-99999-100") == "BraTS-GoAT-99999-100.nii.gz")
    check("three_digit_run_accepted",
          X.release_output_name("SYNTHCOHORT-042") == "SYNTHCOHORT-042.nii.gz")
    check("seven_digit_run_accepted",
          X.release_output_name("LAB7-2026-0000001") == "LAB7-2026-0000001.nii.gz")
    check("nonnumeric_suffix_accepted",
          X.release_output_name("SYNTH-case-alpha") == "SYNTH-case-alpha.nii.gz")
    check("prefix_agnostic",
          X.release_output_name("ZZZ-unfamiliar-prefix-7") == "ZZZ-unfamiliar-prefix-7.nii.gz")

    # 2. unsafe / structurally invalid names still fail closed
    for bad in REJECTED:
        check(f"rejected[{bad!r}]", raises(lambda b=bad: X.release_output_name(b)))

    # 3. planning over one input root with many cases and three distinct prefixes
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "input"
        for name in ACCEPTED:
            (inp / name).mkdir(parents=True)
        before = _tree_hashes(inp)
        cases = X.list_case_folders(inp)
        check("all_cases_enumerated", len(cases) == len(ACCEPTED))
        plan = X.plan_outputs(cases)
        check("one_output_per_case", len(plan) == len(ACCEPTED))
        check("plan_names_are_full_basenames",
              sorted(plan.values()) == sorted(f"{n}.nii.gz" for n in ACCEPTED))
        check("plan_names_are_flat",
              all("/" not in v and "\\" not in v for v in plan.values()))
        prefixes = {v.split("-")[0] for v in plan.values()}
        check("at_least_three_distinct_prefixes", len(prefixes) >= 3)
        check("input_not_mutated_by_planning", _tree_hashes(inp) == before)

        # duplicate/ambiguous output names still fail closed
        check("duplicate_output_name_fails",
              raises(lambda: X.plan_outputs(["/a/BraTS-GoAT-99999-100",
                                             "/b/BraTS-GoAT-99999-100"])))

    # 4. enumeration safety: hidden entries skipped, files skipped, escaping symlink fails closed
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "input"
        outside = Path(td) / "outside"
        (inp / "BraTS-GoAT-99999-100").mkdir(parents=True)
        (inp / ".hidden-dir").mkdir()
        (inp / "README.txt").write_text("not a case\n", encoding="utf-8")
        outside.mkdir()
        names = [p.name for p in X.list_case_folders(inp)]
        check("hidden_entry_skipped", ".hidden-dir" not in names)
        check("regular_file_skipped", "README.txt" not in names)
        check("real_case_enumerated", names == ["BraTS-GoAT-99999-100"])
        try:
            os.symlink(outside, inp / "ESCAPE-000")
            check("symlink_escaping_input_root_fails",
                  raises(lambda: X.list_case_folders(inp)))
        except OSError:
            check("symlink_escaping_input_root_fails", True)   # symlinks unsupported here

    # 5. the SECOND enforcement point: the written-output validator must accept the planned name
    with tempfile.TemporaryDirectory() as td:
        import nibabel as nib
        import numpy as np
        d = Path(td)
        stem = "BraTS-xxx-xxxxx-xxx"
        out = d / f"{stem}.nii.gz"
        affine, shape = _write_nifti(out)
        zooms = nib.load(str(out)).header.get_zooms()
        axcodes = nib.aff2axcodes(affine)
        v = RC.validate_output_file(out, affine, shape, zooms, axcodes, expected_name=out.name)
        check("output_validator_accepts_full_basename_name", v["ok"] and v["name_ok"])
        # a produced name that is not the planned name must fail closed
        v2 = RC.validate_output_file(out, affine, shape, zooms, axcodes,
                                     expected_name="SOMETHING-ELSE.nii.gz")
        check("output_validator_rejects_unplanned_name", not v2["ok"] and not v2["name_ok"])
        check("container_basename_helper_rejects_nested",
              raises(lambda: RC.validate_container_output_basename("a/b.nii.gz", "a/b.nii.gz"),
                     ValueError))
        check("container_basename_helper_rejects_wrong_extension",
              raises(lambda: RC.validate_container_output_basename("x.nii", "x.nii"), ValueError))
        # the legacy validation-upload contract is untouched when expected_name is omitted
        check("legacy_five_digit_contract_intact",
              RC.validate_task3_basename("00042.nii.gz") == "00042"
              and raises(lambda: RC.validate_task3_basename("BraTS-GoAT-99999-100.nii.gz"),
                         ValueError))
        _ = np  # silence unused-import linters

    # 6. source guard — the production runner must not reintroduce a fixed-digit naming rule
    src = (REPO / "scripts" / "release_infer.py").read_text(encoding="utf-8")
    forbidden = {
        "exactly_five_digits_phrase": "EXACTLY 5 digits" in src or "exactly five digits" in src.lower(),
        "fixed_trailing_digit_length_check": ("len(run) != 5" in src or "len(run) == 5" in src
                                              or r"\d{5}" in src or "{5}$" in src),
        "output_name_from_numeric_suffix_only": "task3_output_name(" in src,
        # a hardcoded prefix would have to appear as a string literal the code compares or builds
        # with; the module title naming the sub-challenge is prose, not naming logic
        "hardcoded_goat_prefix": '"BraTS-GoAT' in src or "'BraTS-GoAT" in src,
    }
    for k, hit in forbidden.items():
        check(f"guard_no_{k}", not hit)

    # 7. the repair is a real regression fix: the frozen submitted source rejects what we accept
    frozen = subprocess.run(
        ["git", "-C", str(REPO), "show", "6ed2f9f:scripts/release_infer.py"],
        capture_output=True, text=True)
    if frozen.returncode == 0:
        fs = frozen.stdout
        check("frozen_source_had_the_five_digit_rule", "EXACTLY 5 digits" in fs)
        with tempfile.TemporaryDirectory() as td:
            import importlib.util
            src_dir = Path(td)
            (src_dir / "release_infer.py").write_text(fs, encoding="utf-8")
            for rel in ("g4_reconstruct_validate.py", "g3_audit_labeled_archive.py"):
                (src_dir / rel).write_text(subprocess.run(
                    ["git", "-C", str(REPO), "show", f"6ed2f9f:scripts/{rel}"],
                    capture_output=True, text=True, check=True).stdout, encoding="utf-8")
            sys.path.insert(0, str(src_dir))
            spec = importlib.util.spec_from_file_location("g88_frozen",
                                                          src_dir / "release_infer.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.path.remove(str(src_dir))
            rejected_by_frozen = 0
            for name in ACCEPTED:
                try:
                    mod.release_output_name(name)
                except mod.ReleaseInputError:
                    rejected_by_frozen += 1
            check("frozen_source_rejects_organizer_shapes", rejected_by_frozen >= 5)
    else:
        check("frozen_source_had_the_five_digit_rule", True)       # shallow clone: skip
        check("frozen_source_rejects_organizer_shapes", True)

    print(f"test_g88_container_naming: {'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
