#!/usr/bin/env python3
"""GAT-26 G5A production-runner unit tests (`python3 tests/test_g5_runner.py`).

Synthetic paths/status only — no GPU, no nnU-Net. Covers the authoritative output-path order,
watchdog thresholds, epoch-log parsing, atomic state writes, and the precheck hash/absence
logic via a synthetic expectation bundle.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import g5_runner as R  # noqa: E402

FAILS = []


def check(name, cond):
    if not cond:
        FAILS.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def main():
    # 1. authoritative output-path order = trainer__plans__config
    os.environ["nnUNet_results"] = "/tmp/nn_results"
    of = R.output_folder()
    check("output_path_trainer_plans_config",
          of.endswith("Dataset501_GAT26GOAT/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0"))

    # 1b. plans parameterization: distinct L output path, M default unchanged
    check("output_path_L_plans",
          R.output_folder("nnUNetResEncUNetLPlans").endswith(
              "Dataset501_GAT26GOAT/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_0"))
    check("output_path_default_is_M", R.output_folder() == R.output_folder("nnUNetResEncUNetMPlans"))
    # parameterized watchdog: a longer L ceiling does not trip at the M ceiling
    check("wd_L_ceiling_not_tripped_at_M",
          R.watchdog_verdict(R.HARD_CEILING_S, 0, True, 0, 999, hard_ceiling_s=137160) == "OK")
    check("wd_L_disk_floor_150",
          R.watchdog_verdict(0, 0, True, 0, 149, disk_floor_gib=150) == "STOPPED_HARD_CEILING")

    # 2. watchdog thresholds
    check("wd_ceiling", R.watchdog_verdict(R.HARD_CEILING_S, 0, True, 0, 999) == "STOPPED_HARD_CEILING")
    check("wd_disk_floor", R.watchdog_verdict(0, 0, True, 0, R.DISK_FLOOR_GIB - 1) == "STOPPED_HARD_CEILING")
    check("wd_no_first_epoch", R.watchdog_verdict(0, 0, False, R.FIRST_EPOCH_S, 999).startswith("FAILED_TRAINING_no_first_epoch"))
    check("wd_no_progress", R.watchdog_verdict(0, R.NO_PROGRESS_S, True, 0, 999).startswith("FAILED_TRAINING_no_progress"))
    check("wd_ok", R.watchdog_verdict(10, 10, True, 10, 999) == "OK")
    check("wd_first_epoch_grace", R.watchdog_verdict(0, 0, False, R.FIRST_EPOCH_S - 1, 999) == "OK")

    # 3. epoch-log parsing
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "training_log_2026_1_1_00_00_00.txt").write_text(
            "2026: Epoch 0\nloss\n2026: Epoch 1\n2026: Epoch 7\n")
        ep, mt = R.parse_latest_epoch(tmp)
        check("parse_latest_epoch", ep == 7 and mt > 0)
        ep2, _ = R.parse_latest_epoch(tempfile.mkdtemp())
        check("parse_no_logs", ep2 == -1)

    # 4. atomic state write
    with tempfile.TemporaryDirectory() as tmp:
        R.set_state(tmp, "TRAINING")
        check("atomic_state", (Path(tmp) / "state.txt").read_text().strip() == "TRAINING")
        check("state_sequence", R.STATES == ["PRECHECK", "TRAINING", "VALIDATING",
                                             "OUTPUT_VALIDATION", "OFFICIAL_EVALUATION", "PASS"])

    # 5. precheck hash/absence logic (synthetic; skip GPU/job checks which need nvidia-smi)
    with tempfile.TemporaryDirectory() as tmp:
        pp = Path(tmp) / "pp" / R.DATASET; pp.mkdir(parents=True)
        for fn in ("splits_final.json", f"{R.PLANS}.json", "dataset_fingerprint.json"):
            (pp / fn).write_text(fn)
        def sh(name): return hashlib.sha256(name.encode()).hexdigest()
        cfg = Path(tmp) / "cfg.json"; cfg.write_text("CFG")
        sv = Path(tmp) / "sv.txt"; sv.write_text("deployed_commit=ABC123\n")
        os.environ["nnUNet_preprocessed"] = str(Path(tmp) / "pp")
        os.environ["nnUNet_results"] = str(Path(tmp) / "res")
        expect = {"commit": "ABC123", "source_version_path": str(sv),
                  "config_path": str(cfg), "config_sha256": hashlib.sha256(b"CFG").hexdigest(),
                  "split_sha256": sh("splits_final.json"), "plan_sha256": sh(f"{R.PLANS}.json"),
                  "fingerprint_sha256": sh("dataset_fingerprint.json")}
        # exercise only the deterministic file-hash checks (not nvidia-smi)
        checks = {}
        checks["source_commit"] = expect["commit"] in Path(expect["source_version_path"]).read_text()
        checks["public_config_hash"] = R.sha(expect["config_path"]) == expect["config_sha256"]
        checks["split_full_hash"] = R.sha(pp / "splits_final.json") == expect["split_sha256"]
        checks["plan_hash"] = R.sha(pp / f"{R.PLANS}.json") == expect["plan_sha256"]
        checks["fingerprint_hash"] = R.sha(pp / "dataset_fingerprint.json") == expect["fingerprint_sha256"]
        checks["output_dir_absent"] = not os.path.isdir(R.output_folder())
        check("precheck_hashes_all_true", all(checks.values()))
        # tamper -> fail
        expect_bad = dict(expect, config_sha256="deadbeef")
        check("precheck_detects_tampered_config", R.sha(expect_bad["config_path"]) != expect_bad["config_sha256"])

    # 6. exact fold-membership equality (C2) -- synthetic stems, no real ids
    exp = {"S0001", "S0002", "S0003"}
    m_ok = R.membership_report(exp, {"S0001", "S0002", "S0003"})
    check("membership_exact_set_equal", m_ok["exact_set_equal"] and m_ok["missing"] == 0 and m_ok["extra"] == 0)
    m_swap = R.membership_report(exp, {"S0001", "S0002", "S0009"})   # equal count, 1 missing + 1 extra
    check("membership_equal_count_missing_plus_extra_fails",
          (not m_swap["exact_set_equal"]) and m_swap["expected_count"] == m_swap["actual_count"]
          and m_swap["missing"] == 1 and m_swap["extra"] == 1)
    m_miss = R.membership_report(exp, {"S0001", "S0002"})            # missing only
    check("membership_missing_only_fails",
          (not m_miss["exact_set_equal"]) and m_miss["missing"] == 1 and m_miss["extra"] == 0)
    m_extra = R.membership_report(exp, {"S0001", "S0002", "S0003", "S0004"})  # extra only
    check("membership_extra_only_fails",
          (not m_extra["exact_set_equal"]) and m_extra["missing"] == 0 and m_extra["extra"] == 1)

    # 7. fold_validation_stems parses the frozen split format (synthetic)
    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "splits_final.json"
        sp.write_text(json.dumps([
            {"train": ["S0003"], "val": ["S0001", "S0002"]},
            {"train": ["S0001"], "val": ["S0003", "S0004"]},
        ]))
        check("fold_validation_stems_fold0", R.fold_validation_stems(sp, 0) == {"S0001", "S0002"})
        check("fold_validation_stems_fold1", R.fold_validation_stems(sp, 1) == {"S0003", "S0004"})

    print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
