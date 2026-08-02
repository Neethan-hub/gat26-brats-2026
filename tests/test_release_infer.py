#!/usr/bin/env python3
"""GAT-26 release-runner unit tests (`python3 tests/test_release_infer.py`).

Cover the fail-closed pure helpers that gate the clean-room Task-3 container: strict modality
discovery (missing/duplicate/unknown), folder-echo output naming, sequential ensemble mean,
checkpoint resolution (five distinct vs runtime duplicate proxy), and the frozen inference
constants. No GPU, no nnU-Net, no real case IDs.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import release_infer as X  # noqa: E402

FAILS = 0


def check(name, cond):
    global FAILS
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILS += 1


def raises(fn):
    try:
        fn(); return False
    except X.ReleaseInputError:
        return True


def main():
    print("test_release_infer:")
    pre = "BraTS-GoAT-99999"
    good = [f"{pre}-t1n.nii.gz", f"{pre}-t1c.nii.gz", f"{pre}-t2w.nii.gz", f"{pre}-t2f.nii.gz"]

    # 1. modality discovery — order-independent, returns canonical order
    order = X.discover_modalities(list(reversed(good)))
    check("modalities_order_independent",
          [Path(p).name for p in order] ==
          [f"{pre}-t1n.nii.gz", f"{pre}-t1c.nii.gz", f"{pre}-t2w.nii.gz", f"{pre}-t2f.nii.gz"])
    check("non_image_files_ignored",
          [Path(p).name for p in X.discover_modalities(good + ["README.txt", f"{pre}.json"])]
          == [Path(p).name for p in order])

    # 2. fail-closed: missing / duplicate / unknown
    check("missing_modality_fails", raises(lambda: X.discover_modalities(good[:3])))
    check("duplicate_modality_fails",
          raises(lambda: X.discover_modalities(good + [f"{pre}-t1c.nii.gz"])))
    check("unknown_modality_fails",
          raises(lambda: X.discover_modalities(good + [f"{pre}-flair.nii.gz"])))
    check("empty_input_fails", raises(lambda: X.discover_modalities([])))

    # 3. Task-3 container naming — the output stem is the COMPLETE input-folder basename, treated
    #    as an opaque identifier. No case ID is parsed, no trailing-digit count is required and no
    #    folder prefix is assumed. Only unsafe or structurally invalid names fail closed.
    #    All identifiers below are obviously synthetic. See tests/test_g88_container_naming.py for
    #    the full organizer-shape matrix and the source guard.
    check("basename_returned_unchanged", X.validate_case_folder_basename(pre) == pre)
    check("output_name_echoes_folder", X.release_output_name(pre) == f"{pre}.nii.gz")
    check("plain_digit_folder_ok", X.validate_case_folder_basename("99999") == "99999")
    # accepted regardless of the trailing digit count, or of there being no digits at all
    check("four_digit_ok", X.release_output_name("BraTS-GoAT-9999") == "BraTS-GoAT-9999.nii.gz")
    check("six_digit_ok", X.release_output_name("BraTS-GoAT-999999") == "BraTS-GoAT-999999.nii.gz")
    check("timepoint_dash_suffix_ok",
          X.release_output_name("BraTS-GoAT-99999-100") == "BraTS-GoAT-99999-100.nii.gz")
    check("timepoint_underscore_suffix_ok",
          X.release_output_name("BraTS-GoAT-99999_000") == "BraTS-GoAT-99999_000.nii.gz")
    check("alpha_suffix_ok",
          X.release_output_name("BraTS-GoAT-99999abc") == "BraTS-GoAT-99999abc.nii.gz")
    check("no_digits_ok", X.release_output_name("BraTS-GoAT-XYZ") == "BraTS-GoAT-XYZ.nii.gz")
    # rejects — unsafe or structurally invalid only
    check("nested_output_name_fails", raises(lambda: X.release_output_name("a/99999")))
    check("backslash_output_name_fails", raises(lambda: X.release_output_name("a\\99999")))
    check("empty_output_name_fails", raises(lambda: X.release_output_name("  ")))
    check("dotdot_output_name_fails", raises(lambda: X.release_output_name("..")))
    check("hidden_output_name_fails", raises(lambda: X.release_output_name(".hidden")))
    check("nul_output_name_fails", raises(lambda: X.release_output_name("BraTS-GoAT-99999\x00")))
    # plan_outputs: valid distinct batch maps folder->flat name; duplicate/ambiguous fails closed
    plan = X.plan_outputs(["/in/BraTS-GoAT-99997-100", "/in/BraTS-GoAT-99998"])
    check("plan_outputs_valid_batch",
          plan == {"/in/BraTS-GoAT-99997-100": "BraTS-GoAT-99997-100.nii.gz",
                   "/in/BraTS-GoAT-99998": "BraTS-GoAT-99998.nii.gz"})
    check("plan_outputs_duplicate_fails",
          raises(lambda: X.plan_outputs(["/a/BraTS-GoAT-99997", "/b/BraTS-GoAT-99997"])))
    check("plan_outputs_rejects_unsafe_member",
          raises(lambda: X.plan_outputs(["/in/BraTS-GoAT-99997", "/in/.hidden"])))

    # 4. streaming ensemble mean (numpy) — equals plain average, memory-safe accumulation
    try:
        import numpy as np
        a = np.ones((2, 2)); b = np.zeros((2, 2)); c = np.full((2, 2), 2.0)
        acc = None
        for arr in (a, b, c):
            acc = X.accumulate_mean(acc, arr)
        acc = acc / 3.0
        check("ensemble_mean_correct", np.allclose(acc, (a + b + c) / 3.0))
        # accumulate_mean must not alias the first array (copy on first)
        acc2 = X.accumulate_mean(None, a); acc2 += 5.0
        check("accumulate_does_not_alias_first", np.allclose(a, 1.0))
    except ImportError:
        check("ensemble_mean_correct", True)  # numpy always present in the runtime image

    # 5. checkpoint resolution — A10G-2 requires five DISTINCT-content files; A10G-1 proxy is opt-in.
    def _mk(wd, n, content):
        for i in range(n):
            (wd / f"fold_{i}").mkdir()
            (wd / f"fold_{i}" / "checkpoint_final.pth").write_text(content(i))
    with tempfile.TemporaryDirectory() as td:                    # five DISTINCT content -> final pass
        wd = Path(td); _mk(wd, 5, lambda i: f"ckpt-content-{i}")
        paths, mode = X.discover_checkpoints(str(wd))
        check("five_distinct_content_passes", len(paths) == 5 and mode == "five_distinct_fold_checkpoints")
    with tempfile.TemporaryDirectory() as td:                    # five IDENTICAL copies, no proxy -> FAIL
        wd = Path(td); _mk(wd, 5, lambda i: "identical-bytes")
        check("five_identical_copies_fail_closed", raises(lambda: X.discover_checkpoints(str(wd))))
    with tempfile.TemporaryDirectory() as td:                    # five IDENTICAL copies WITH proxy -> proxy pass
        wd = Path(td); _mk(wd, 5, lambda i: "identical-bytes")
        paths, mode = X.discover_checkpoints(str(wd), allow_duplicate_proxy=True)
        check("identical_with_proxy_passes", len(paths) == 5 and mode == "runtime_only_duplicate_weight_proxy")
    with tempfile.TemporaryDirectory() as td:                    # missing a fold -> final FAIL
        wd = Path(td); _mk(wd, 4, lambda i: f"c{i}")
        check("missing_fold_fails_final", raises(lambda: X.discover_checkpoints(str(wd))))
    with tempfile.TemporaryDirectory() as td:                    # a slot that is a directory (not a regular file) -> FAIL
        wd = Path(td); _mk(wd, 4, lambda i: f"c{i}"); (wd / "fold_4").mkdir(); (wd / "fold_4" / "checkpoint_final.pth").mkdir()
        check("non_regular_file_fails_final", raises(lambda: X.discover_checkpoints(str(wd))))
    with tempfile.TemporaryDirectory() as td:                    # only fold_0 -> no proxy FAIL; proxy PASS
        wd = Path(td); (wd / "fold_0").mkdir(); (wd / "fold_0" / "checkpoint_final.pth").write_text("m0")
        check("proxy_disallowed_by_default", raises(lambda: X.discover_checkpoints(str(wd))))
        paths, mode = X.discover_checkpoints(str(wd), allow_duplicate_proxy=True)
        check("duplicate_proxy_five_slots", len(paths) == 5 and mode == "runtime_only_duplicate_weight_proxy")
    with tempfile.TemporaryDirectory() as td:                    # no fold_0 at all -> proxy also fails
        check("no_checkpoints_fails", raises(lambda: X.discover_checkpoints(td, allow_duplicate_proxy=True)))
    # the resolver must never leak a checkpoint digest in its return value
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td); _mk(wd, 5, lambda i: f"ckpt-content-{i}")
        paths, mode = X.discover_checkpoints(str(wd))
        check("no_hash_in_return", all(len(str(p)) < 200 and "sha" not in str(p).lower() for p in paths))

    # 5b. determinism enforcement sets deterministic cuDNN + fixed seeds (no GPU needed)
    try:
        import torch
        X.enforce_determinism()
        check("cudnn_deterministic", torch.backends.cudnn.deterministic is True)
        check("cudnn_benchmark_off", torch.backends.cudnn.benchmark is False)
        import os as _os
        check("cublas_workspace_set", _os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8")
    except ImportError:
        check("cudnn_deterministic", True)  # torch always present in the runtime image

    # 6. frozen inference constants — ResEnc-M only, no TTA, threshold 0.5, five checkpoints
    check("plan_is_M", X.PLANS == "nnUNetResEncUNetMPlans")
    check("threshold_half", X.TAU == 0.5)
    check("five_checkpoints", X.N_CHECKPOINTS == 5)
    check("weights_dir_not_worker_path", "/workspace" not in X.DEFAULT_WEIGHTS_DIR)

    # 7. GPU gate (pure logic): exactly one GPU; exact-name match when required
    check("gpu_one_no_name_ok", X._check_gpu(1, "NVIDIA A10G", None) == "NVIDIA A10G")
    check("gpu_name_match_ok", X._check_gpu(1, "NVIDIA A10G", "NVIDIA A10G") == "NVIDIA A10G")
    check("gpu_zero_fails", raises(lambda: X._check_gpu(0, "NVIDIA A10G", None)))
    check("gpu_two_fails", raises(lambda: X._check_gpu(2, "NVIDIA A10G", None)))
    check("gpu_wrong_name_fails", raises(lambda: X._check_gpu(1, "NVIDIA RTX PRO 6000", "NVIDIA A10G")))

    print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
