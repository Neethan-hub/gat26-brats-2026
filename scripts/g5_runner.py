#!/usr/bin/env python3
"""GAT-26 production runner — ResEnc-M single fold (Dataset501_GAT26GOAT).

Orchestrates the frozen recipe end to end with atomic private state/heartbeat and a hard
watchdog: PRECHECK -> TRAINING -> VALIDATING -> OUTPUT_VALIDATION -> OFFICIAL_EVALUATION ->
PASS (failures: FAILED_<STAGE> / STOPPED_HARD_CEILING). Uses the installed official nnU-Net
2.8.1 CLI. Never commits protected predictions/checkpoints/hashes.

The `--fold` argument is REQUIRED and fails closed unless it is one of {1,2,3,4}: fold 0
(M_COMPLETION_PASS) must never be retrained, and each fold trains from its own recorded random
initialization with isolated output/state/prediction paths (no cross-fold checkpoint reuse). The
expected validation-set size is derived from that fold's frozen split (271 for fold 0, 270 for
folds 1-4), so exact-membership equality is enforced per fold.

Sanitized: no real case IDs are printed. Pure helpers (precheck checks, validators, watchdog
thresholds) are import-safe for unit tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# ---- frozen recipe / limits ----
DATASET = "Dataset501_GAT26GOAT"
TRAINER = "nnUNetTrainer"
PLANS = "nnUNetResEncUNetMPlans"
CONFIG = "3d_fullres"
FOLD = 0
EXPECT_EPOCHS = 1000
EXPECT_VAL_CASES = 271
HARD_CEILING_S = int(15 * 3600 + 42 * 60)        # 15 h 42 m
NO_PROGRESS_S = 20 * 60                            # 20 min
FIRST_EPOCH_S = 20 * 60                            # first epoch within 20 min of init
DISK_FLOOR_GIB = 180
QUOTA_TOTAL_BYTES = 300 * 10**9
STATES = ["PRECHECK", "TRAINING", "VALIDATING", "OUTPUT_VALIDATION", "OFFICIAL_EVALUATION", "PASS"]


def atomic_write(path, text):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def set_state(rundir, state):
    atomic_write(Path(rundir) / "state.txt", state + "\n")


def project_free_gib(mount=None):
    """Disk usage of the project data mount.

    The mount is resolved from GAT26_DATA_MOUNT, falling back to the repository root, so the
    function carries no machine-absolute default.
    """
    mount = mount or os.environ.get("GAT26_DATA_MOUNT") or str(Path(__file__).resolve().parent.parent)
    used = int(subprocess.run(["du", "-sb", mount], capture_output=True, text=True).stdout.split()[0])
    return (QUOTA_TOTAL_BYTES - used) / 2**30


def heartbeat(rundir, stage, epoch, launch_ts, last_progress_ts, gpu_mem_gib, pid, eta_utc):
    now = time.time()
    atomic_write(Path(rundir) / "status.json", json.dumps({
        "stage": stage, "current_epoch": epoch,
        "elapsed_s": round(now - launch_ts, 1),
        "since_last_progress_s": round(now - last_progress_ts, 1),
        "gpu_mem_reserved_gib": gpu_mem_gib, "disk_free_gib": round(project_free_gib(), 1),
        "pid": pid, "eta_utc": eta_utc, "heartbeat_utc_epoch": round(now, 0),
    }, indent=2) + "\n")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def output_folder(plans=PLANS, fold=FOLD):
    base = os.environ["nnUNet_results"]
    return os.path.join(base, DATASET, f"{TRAINER}__{plans}__{CONFIG}", f"fold_{fold}")


# ------------------------------ exact fold-membership equality (C2) ------------------------------
def fold_validation_stems(splits_final_path, fold=FOLD):
    """Return the exact set of fold-`fold` validation case stems from the frozen split."""
    data = json.loads(Path(splits_final_path).read_text(encoding="utf-8"))
    return set(data[fold]["val"])


def membership_report(expected_stems, actual_stems):
    """Set-equality of expected vs actual prediction stems. Counts/booleans only -- the caller
    must never print or commit the membership itself. A one-missing + one-extra replacement
    fails (exact_set_equal False) even though the counts stay equal."""
    exp, act = set(expected_stems), set(actual_stems)
    missing, extra = exp - act, act - exp
    return {"expected_count": len(exp), "actual_count": len(act),
            "missing": len(missing), "extra": len(extra),
            "exact_set_equal": (len(missing) == 0 and len(extra) == 0 and len(exp) == len(act))}


# ------------------------------ precheck ------------------------------
def precheck(expect, plans=PLANS, disk_floor_gib=DISK_FLOOR_GIB, fold=FOLD):
    """expect: dict of expected hashes/commit. Returns (ok, checks)."""
    pp = Path(os.environ["nnUNet_preprocessed"]) / DATASET
    checks = {}
    checks["source_commit"] = expect["commit"] in Path(expect["source_version_path"]).read_text(encoding="utf-8")
    checks["public_config_hash"] = sha(expect["config_path"]) == expect["config_sha256"]
    checks["split_full_hash"] = sha(pp / "splits_final.json") == expect["split_sha256"]
    checks["plan_hash"] = sha(pp / f"{plans}.json") == expect["plan_sha256"]
    checks["fingerprint_hash"] = sha(pp / "dataset_fingerprint.json") == expect["fingerprint_sha256"]
    checks["split_installed"] = (pp / "splits_final.json").exists()
    of = output_folder(plans, fold)
    checks["output_dir_absent_or_empty"] = (not os.path.isdir(of)) or (not os.listdir(of))
    checks["no_checkpoint_present"] = not os.path.exists(os.path.join(of, "checkpoint_final.pth"))
    # exactly one GPU
    gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines()
    checks["one_gpu"] = len(gpus) == 1
    procs = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.strip()
    checks["no_gpu_compute"] = procs == ""
    # no competing training/eval job
    ps = subprocess.run(["bash", "-c", "ps -eo cmd --no-headers"], capture_output=True, text=True).stdout
    checks["no_competing_job"] = sum(1 for l in ps.splitlines()
                                     if any(k in l for k in ("nnUNetv2_train", "nnUNetv2_predict"))) == 0
    checks["disk_floor"] = project_free_gib() >= disk_floor_gib
    # no unexpected pretrained-weight argument is structurally impossible here (we build the argv)
    checks["random_init_no_pretrained"] = True
    return all(checks.values()), checks


# ------------------------------ prediction validators ------------------------------
def validate_prediction(path, ref_affine, ref_shape, ref_zooms, ref_axcodes):
    """Reuse the proven G4 validator: flat name, integer {0,1,2,3}, exact geometry, finite,
    ET subset TC subset WT."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import g4_reconstruct_validate as RC
    return RC.validate_output_file(path, ref_affine, ref_shape, ref_zooms, ref_axcodes)


# ------------------------------ watchdog ------------------------------
def watchdog_verdict(elapsed_s, since_progress_s, first_epoch_seen, since_init_s, disk_free_gib,
                     hard_ceiling_s=HARD_CEILING_S, disk_floor_gib=DISK_FLOOR_GIB):
    if elapsed_s >= hard_ceiling_s:
        return "STOPPED_HARD_CEILING"
    if disk_free_gib < disk_floor_gib:
        return "STOPPED_HARD_CEILING"
    if not first_epoch_seen and since_init_s >= FIRST_EPOCH_S:
        return "FAILED_TRAINING_no_first_epoch"
    if first_epoch_seen and since_progress_s >= NO_PROGRESS_S:
        return "FAILED_TRAINING_no_progress"
    return "OK"


def parse_latest_epoch(log_dir):
    """Return (latest_epoch, last_mtime) from nnU-Net training logs."""
    logs = sorted(Path(log_dir).glob("training_log_*.txt"))
    if not logs:
        return -1, 0.0
    latest = logs[-1]
    epoch = -1
    for line in latest.read_text(errors="ignore", encoding="utf-8").splitlines():
        if "Epoch " in line:
            try:
                epoch = max(epoch, int(line.split("Epoch ")[1].split()[0]))
            except (ValueError, IndexError):
                pass
    return epoch, latest.stat().st_mtime


def _gpu_mem_gib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip().splitlines()
        return round(int(out[0]) / 1024, 2)
    except Exception:
        return None


def run(args):
    fold = args.fold
    rundir = Path(args.rundir); rundir.mkdir(parents=True, exist_ok=True)
    expect = json.loads(Path(args.expect).read_text(encoding="utf-8"))
    plans = args.plans
    hard_ceiling_s = int(round(args.hard_ceiling_hours * 3600))
    disk_floor = args.disk_floor_gib
    launch_ts = time.time()
    hard_stop = launch_ts + hard_ceiling_s
    atomic_write(rundir / "launch_manifest.json", json.dumps({
        "launch_utc_epoch": round(launch_ts), "hard_stop_utc_epoch": round(hard_stop),
        "tag": args.tag, "fold": fold, "hard_ceiling_s": hard_ceiling_s, "disk_floor_gib": disk_floor,
        "recipe": {"dataset": DATASET, "trainer": TRAINER, "plans": plans, "config": CONFIG,
                   "fold": fold, "epochs": EXPECT_EPOCHS, "random_init": True},
        "output_folder": output_folder(plans, fold)}, indent=2) + "\n")

    # Immutable per-run launch-source record: snapshot the source-version marker + launch commit
    # binding BEFORE training. The current worker marker is mutable (advances on later redeploys);
    # this frozen record is what the completion audit verifies launch provenance against.
    try:
        marker_text = Path(expect["source_version_path"]).read_text(encoding="utf-8")
    except Exception:
        marker_text = ""
    atomic_write(rundir / "source_version_at_launch.txt",
                 f"launch_commit={expect.get('commit', '')}\n"
                 f"launch_utc_epoch={round(launch_ts)}\n"
                 f"marker_at_launch:\n{marker_text}")

    set_state(rundir, "PRECHECK")
    ok, checks = precheck(expect, plans=plans, disk_floor_gib=disk_floor, fold=fold)
    atomic_write(rundir / "precheck.json", json.dumps(checks, indent=2) + "\n")
    if not ok:
        set_state(rundir, "FAILED_PRECHECK")
        print(json.dumps({"state": "FAILED_PRECHECK", "checks": checks}))
        return 2

    # ---- TRAINING ----
    set_state(rundir, "TRAINING")
    of = output_folder(plans, fold)
    train_cmd = ["nnUNetv2_train", "501", CONFIG, str(fold), "-tr", TRAINER, "-p", plans]
    tlog = open(rundir / "train.stdout.log", "w", encoding="utf-8")
    proc = subprocess.Popen(train_cmd, stdout=tlog, stderr=subprocess.STDOUT,
                            start_new_session=True)  # own process group
    atomic_write(rundir / "train.pid", str(proc.pid) + "\n")
    init_ts = time.time(); last_progress = time.time(); last_epoch = -1; first_epoch_seen = False
    fail_state = None
    while True:
        rc = proc.poll()
        epoch, log_mtime = parse_latest_epoch(of)
        if epoch > last_epoch:
            last_epoch = epoch; last_progress = time.time()
            if epoch >= 0:
                first_epoch_seen = True
        elapsed = time.time() - launch_ts
        eta = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(
            launch_ts + (elapsed / max(last_epoch + 1, 1)) * EXPECT_EPOCHS)) if last_epoch >= 0 else "n/a"
        heartbeat(rundir, "TRAINING", last_epoch, launch_ts, last_progress, _gpu_mem_gib(), proc.pid, eta)
        # nonfinite / OOM / worker death detection
        tail = Path(rundir / "train.stdout.log").read_text(errors="ignore", encoding="utf-8")[-4000:]
        if any(k in tail for k in ("out of memory", "CUDA out of memory")):
            fail_state = "FAILED_TRAINING_OOM"; break
        if "no longer alive" in tail:
            fail_state = "FAILED_TRAINING_worker_death"; break
        if "nan" in tail.lower() and "loss" in tail.lower() and "nan," in tail.lower():
            fail_state = "FAILED_TRAINING_nonfinite"; break
        verdict = watchdog_verdict(elapsed, time.time() - last_progress, first_epoch_seen,
                                   time.time() - init_ts, project_free_gib(),
                                   hard_ceiling_s=hard_ceiling_s, disk_floor_gib=disk_floor)
        if verdict != "OK":
            fail_state = verdict; break
        if rc is not None:
            break
        time.sleep(args.poll)
    if fail_state:
        try: os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception: pass
        set_state(rundir, fail_state)
        print(json.dumps({"state": fail_state, "last_epoch": last_epoch}))
        return 3
    if proc.returncode != 0 or not os.path.exists(os.path.join(of, "checkpoint_final.pth")):
        set_state(rundir, "FAILED_TRAINING")
        print(json.dumps({"state": "FAILED_TRAINING", "rc": proc.returncode}))
        return 3

    # ---- VALIDATING (frozen fold-0 validation with checkpoint_final via official --val) ----
    set_state(rundir, "VALIDATING")
    heartbeat(rundir, "VALIDATING", last_epoch, launch_ts, time.time(), _gpu_mem_gib(), os.getpid(), "n/a")
    val_cmd = ["nnUNetv2_train", "501", CONFIG, str(fold), "-tr", TRAINER, "-p", plans, "--val"]
    vlog = open(rundir / "val.stdout.log", "w", encoding="utf-8")
    vrc = subprocess.run(val_cmd, stdout=vlog, stderr=subprocess.STDOUT).returncode
    val_dir = os.path.join(of, "validation")
    preds = sorted(p for p in Path(val_dir).glob("*.nii.gz")) if os.path.isdir(val_dir) else []
    if vrc != 0 or len(preds) == 0:
        set_state(rundir, "FAILED_VALIDATING")
        print(json.dumps({"state": "FAILED_VALIDATING", "rc": vrc, "preds": len(preds)}))
        return 4

    # ---- OUTPUT_VALIDATION ----
    set_state(rundir, "OUTPUT_VALIDATION")
    import nibabel as nib
    raw = Path(os.environ["nnUNet_raw"]) / DATASET / "labelsTr"
    # exact fold-membership equality (C2): predicted stems must equal the frozen fold-0 val set
    pp = Path(os.environ["nnUNet_preprocessed"]) / DATASET
    expected_stems = fold_validation_stems(pp / "splits_final.json", fold=fold)
    expected_n = len(expected_stems)                       # 271 for fold 0, 270 for folds 1-4
    actual_stems = {p.name[:-len(".nii.gz")] for p in preds}
    mem = membership_report(expected_stems, actual_stems)
    bad = 0
    for p in preds:
        cid = p.name[:-len(".nii.gz")]
        ref = nib.load(str(raw / f"{cid}.nii.gz"))
        v = validate_prediction(str(p), ref.affine, ref.shape, ref.header.get_zooms(),
                                nib.aff2axcodes(ref.affine))
        if not v["ok"]:
            bad += 1
    ok_count = len(preds) - bad
    atomic_write(rundir / "output_validation.json", json.dumps({
        "predictions": len(preds), "valid": ok_count, "invalid": bad,
        "expected_val_cases": expected_n, "membership": mem}, indent=2) + "\n")
    if bad != 0 or len(preds) != expected_n or not mem["exact_set_equal"]:
        set_state(rundir, "FAILED_OUTPUT_VALIDATION")
        print(json.dumps({"state": "FAILED_OUTPUT_VALIDATION", "valid": ok_count,
                          "n": len(preds), "membership_ok": mem["exact_set_equal"]}))
        return 5

    # ---- OFFICIAL_EVALUATION (BraTS-evaluation 0.0.8 GoAT, in the eval venv) ----
    set_state(rundir, "OFFICIAL_EVALUATION")
    heartbeat(rundir, "OFFICIAL_EVALUATION", last_epoch, launch_ts, time.time(), _gpu_mem_gib(), os.getpid(), "n/a")
    eval_out = rundir / "official_eval_private.json"
    ev = subprocess.run([args.eval_python, str(Path(__file__).resolve().parent / "g5_evaluate.py"),
                         "--preds", val_dir, "--gt", str(raw),
                         "--out", str(eval_out),
                         "--summary-out", str(rundir / "official_eval_summary.json"),
                         "--expected-n", str(expected_n)],   # this fold's frozen denominator (270 / 271)
                        capture_output=True, text=True)
    # C3: persist private evaluator stdout/stderr + exact exit status (git-ignored worker path)
    atomic_write(rundir / "eval.stdout.log", ev.stdout or "")
    atomic_write(rundir / "eval.stderr.log", ev.stderr or "")
    atomic_write(rundir / "eval.exit", str(ev.returncode) + "\n")
    # C3: detect an evaluator-returned error dict, not only a raised exception / nonzero exit
    eval_error_dict = True
    try:
        j = json.loads(eval_out.read_text(encoding="utf-8"))
        eval_error_dict = bool(j.get("errors"))          # non-empty errors list => failure
    except Exception:
        eval_error_dict = True                            # unreadable => fail closed
    if ev.returncode != 0 or eval_error_dict:
        set_state(rundir, "FAILED_OFFICIAL_EVALUATION")
        print(json.dumps({"state": "FAILED_OFFICIAL_EVALUATION", "rc": ev.returncode,
                          "eval_error_dict": eval_error_dict, "err": (ev.stderr or "")[-300:]}))
        return 6

    set_state(rundir, "PASS")
    heartbeat(rundir, "PASS", last_epoch, launch_ts, time.time(), _gpu_mem_gib(), os.getpid(), "done")
    print(json.dumps({"state": "PASS", "epochs": last_epoch + 1, "val_predictions": len(preds)}))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--expect", required=True)        # JSON with expected hashes/commit/paths
    ap.add_argument("--eval-python", required=True)    # eval-venv python
    ap.add_argument("--poll", type=int, default=30)
    # arch/limit parameters (defaults reproduce the ResEnc-M recipe exactly)
    ap.add_argument("--plans", default=PLANS)                 # e.g. nnUNetResEncUNetLPlans for L
    ap.add_argument("--fold", type=int, required=True)        # G7 production folds ONLY
    ap.add_argument("--hard-ceiling-hours", type=float, default=HARD_CEILING_S / 3600.0)
    ap.add_argument("--disk-floor-gib", type=int, default=DISK_FLOOR_GIB)
    ap.add_argument("--tag", default="M")
    args = ap.parse_args()
    # FAIL CLOSED: this runner trains ONLY production folds 1-4. Fold 0 (M_COMPLETION_PASS) must
    # never be retrained, and no fold outside {1,2,3,4} is a valid production target.
    if args.fold not in (1, 2, 3, 4):
        print(json.dumps({"error": f"fold must be one of {{1,2,3,4}} (got {args.fold}); "
                          "fold 0 must not be retrained"}), file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
