#!/usr/bin/env python3
"""GAT-26 G7 sequential fold supervisor — ResEnc-M folds 1..4, one at a time.

Durable, worker-side, independent of any interactive operator session: run fold 1 (train -> validate ->
official eval via g5_runner), then run the strict completion audit; launch the NEXT fold ONLY if the
current fold reaches an audited PASS. Stop the entire chain on any failure, anomaly, ceiling, or
ambiguous state. Atomic chain state + heartbeat survive SSH/browser disconnection.

Never retrains fold 0 (g5_runner fails closed unless --fold in {1,2,3,4}); each fold uses isolated
output/state/prediction/private-evidence paths; no cross-fold checkpoint reuse.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
FOLDS = (1, 2, 3, 4)


def atomic_write(path, text):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def set_chain_state(g7dir, state):
    atomic_write(Path(g7dir) / "g7_state.txt", state + "\n")


def chain_heartbeat(g7dir, current_fold, chain_state, start_ts, stop_by_epoch, folds_passed,
                    fold_status=None):
    now = time.time()
    atomic_write(Path(g7dir) / "g7_status.json", json.dumps({
        "chain_state": chain_state, "current_fold": current_fold,
        "folds_passed": folds_passed, "elapsed_s": round(now - start_ts, 1),
        "seconds_to_stop_by": round(stop_by_epoch - now, 1),
        "heartbeat_utc_epoch": round(now, 0),
        "fold_status": fold_status,        # sanitized copy of the active fold's status.json
    }, indent=2) + "\n")


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def audit_is_pass(rep, fold):
    """Pure gate: the NEXT fold launches ONLY when this fold's completion audit is an unambiguous
    audited PASS — report present, verdict M_COMPLETION_PASS, this exact fold, and every gate ok."""
    if not rep or rep.get("verdict") != "M_COMPLETION_PASS" or rep.get("fold") != fold:
        return False
    gates = rep.get("gates", {})
    return bool(gates) and all(g.get("ok") for g in gates.values())


def fold_output_folder(fold):
    base = os.environ["nnUNet_results"]
    return Path(base) / "Dataset501_GAT26GOAT" / \
        "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres" / f"fold_{fold}"


def run_fold(g7dir, fold, eval_python, per_fold_hours, disk_floor, expect_src,
             start_ts, stop_by_epoch, folds_passed):
    """Train+validate+eval (g5_runner) then strict audit for one fold. Returns (ok, terminal_reason)."""
    fold_rundir = Path(g7dir) / f"fold_{fold}"
    fold_rundir.mkdir(parents=True, exist_ok=True)
    # fold isolation: this fold's results dir must not already hold a checkpoint (no collision)
    of = fold_output_folder(fold)
    if (of / "checkpoint_final.pth").exists():
        return False, f"FAILED_FOLD{fold}_output_collision"
    shutil.copyfile(expect_src, fold_rundir / "expect.json")   # audit reads rundir/expect.json

    set_chain_state(g7dir, f"FOLD{fold}_TRAINING")
    cmd = [os.environ.get("GAT26_TRAIN_PY", str(REPO.parent / ".venv" / "bin" / "python")),
           str(REPO / "g5_runner.py"),
           "--rundir", str(fold_rundir), "--expect", str(expect_src),
           "--eval-python", eval_python, "--plans", "nnUNetResEncUNetMPlans",
           "--fold", str(fold), "--tag", "M",
           "--hard-ceiling-hours", str(per_fold_hours), "--disk-floor-gib", str(disk_floor),
           "--poll", "30"]
    log = open(fold_rundir / "runner.stdout.log", "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    # poll the runner; refresh chain heartbeat; enforce the OVERALL stop-by ceiling
    while proc.poll() is None:
        now = time.time()
        chain_heartbeat(g7dir, fold, f"FOLD{fold}_TRAINING", start_ts, stop_by_epoch, folds_passed,
                        fold_status=read_json(fold_rundir / "status.json"))
        if now >= stop_by_epoch:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
            return False, f"STOPPED_OVERALL_STOP_BY_FOLD{fold}"
        time.sleep(60)
    fold_state = (fold_rundir / "state.txt").read_text(encoding="utf-8").strip() if (fold_rundir / "state.txt").exists() else "UNKNOWN"
    if fold_state != "PASS":
        return False, f"FAILED_FOLD{fold}_{fold_state}"

    # ---- strict completion audit (only proceed on audited PASS) ----
    set_chain_state(g7dir, f"FOLD{fold}_AUDIT")
    chain_heartbeat(g7dir, fold, f"FOLD{fold}_AUDIT", start_ts, stop_by_epoch, folds_passed)
    gates = fold_rundir / "completion_gates.json"
    aud = subprocess.run([os.environ.get("GAT26_TRAIN_PY", str(REPO.parent / ".venv" / "bin" / "python")),
                          str(REPO / "g5_completion_audit.py"),
                          "--rundir", str(fold_rundir), "--eval-python", eval_python,
                          "--report-out", str(gates)],
                         capture_output=True, text=True)
    (fold_rundir / "audit.stdout.log").write_text(aud.stdout or "", encoding="utf-8")
    (fold_rundir / "audit.stderr.log").write_text(aud.stderr or "", encoding="utf-8")
    rep = read_json(gates)
    if aud.returncode != 0 or not audit_is_pass(rep, fold):
        return False, f"FAILED_FOLD{fold}_AUDIT_{(rep or {}).get('verdict')}"
    return True, f"FOLD{fold}_PASS"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g7dir", required=True)
    ap.add_argument("--expect", required=True)          # shared G7 expect.json (commit-bound)
    ap.add_argument("--eval-python", required=True)
    ap.add_argument("--stop-by-utc-epoch", type=int, required=True)
    ap.add_argument("--overall-ceiling-hours", type=float, default=72.0)
    ap.add_argument("--per-fold-hours", type=float, default=15 + 42 / 60)  # 15 h 42 m = 15.7 h
    ap.add_argument("--disk-floor-gib", type=int, default=120)
    args = ap.parse_args()

    g7dir = Path(args.g7dir); g7dir.mkdir(parents=True, exist_ok=True)
    start_ts = time.time()
    stop_by = min(float(args.stop_by_utc_epoch), start_ts + args.overall_ceiling_hours * 3600)
    set_chain_state(g7dir, "CHAIN_START")
    folds_passed = []
    for fold in FOLDS:
        if time.time() >= stop_by:
            set_chain_state(g7dir, f"STOPPED_OVERALL_CEILING_before_fold{fold}")
            chain_heartbeat(g7dir, fold, "STOPPED_OVERALL_CEILING", start_ts, stop_by, folds_passed)
            return 4
        ok, reason = run_fold(g7dir, fold, args.eval_python, args.per_fold_hours,
                              args.disk_floor_gib, args.expect, start_ts, stop_by, folds_passed)
        if not ok:
            set_chain_state(g7dir, reason)
            chain_heartbeat(g7dir, fold, reason, start_ts, stop_by, folds_passed)
            print(json.dumps({"chain_state": reason, "folds_passed": folds_passed}))
            return 3
        folds_passed.append(fold)
        set_chain_state(g7dir, reason)   # FOLD{fold}_PASS
        chain_heartbeat(g7dir, fold, reason, start_ts, stop_by, folds_passed)

    set_chain_state(g7dir, "ALL_FOLDS_PASS")
    chain_heartbeat(g7dir, None, "ALL_FOLDS_PASS", start_ts, stop_by, folds_passed)
    print(json.dumps({"chain_state": "ALL_FOLDS_PASS", "folds_passed": folds_passed}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
