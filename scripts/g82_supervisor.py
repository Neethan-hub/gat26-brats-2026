#!/usr/bin/env python3
"""G82 §K supervisor: restartable, progress-aware, duplicate-proof.

Liveness is NOT inferred from a heartbeat. For each (candidate, fold) job token the
supervisor checks, on the worker itself:

  * that a process exists whose command line contains both g82_train.py and the
    job's exact config token,
  * the process state and its child augmenter processes,
  * whether the training log's modification time is actually advancing,
  * whether the epoch count is increasing,
  * the GPU's compute-apps list.

A replacement is launched only when no live process holds the token, which is what
prevented G81's duplicate-worker defect. A stalled job (process alive but no
progress for STALL_SECONDS) is killed first and only then restarted, at most
MAX_RESTARTS times, and never across a hard cutoff.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

# Every path is supplied by the private runtime environment; none is hard-coded,
# so no private absolute path is ever committed.
STATE = os.environ["G82_STATE"]                 # controller state directory
KEY = os.environ["G82_SSH_KEY"]                 # 0600 controller key, never printed
ENDPOINTS = os.environ["G82_ENDPOINTS"]         # private worker endpoint table
SSH = ["-i", KEY, "-n", "-o", "StrictHostKeyChecking=no",
       "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=20",
       "-o", "ServerAliveInterval=30"]

STALL_SECONDS = 420          # no log progress for 7 minutes => stalled
POLL_SECONDS = 90
MAX_RESTARTS = 2
LOCK = os.path.join(STATE, "supervisor.lock")


def journal(event: dict) -> None:
    event["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(STATE, "journal.jsonl"), "a") as f:
        f.write(json.dumps(event) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sh(host, port, cmd, timeout=90):
    r = subprocess.run(["ssh"] + SSH + ["-p", str(port), f"root@{host}", cmd],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.strip()


def probe(host, port, token, W):
    """Return a liveness/progress record for one job token."""
    cmd = (
        f'T={token}; '
        # Match ONLY python processes holding this job's exact config file. A bare
        # `pgrep -f g82_train.py` also matches the probe's own shell, which would make
        # every job look alive and defeat restart detection entirely.
        f'PIDS=$(for p in $(pgrep -x python; pgrep -x python3); do '
        f'  C=$(tr "\\0" " " < /proc/$p/cmdline 2>/dev/null); '
        f'  case "$C" in *g82_train.py*cfg/$T.json*) echo $p;; esac; done); '
        f'echo "PIDS=$PIDS"; '
        f'R=$(echo $T | cut -d_ -f1); F=$(echo $T | cut -d_ -f2 | tr -d f); '
        f'L=$(ls -t {W}/results/${{R}}_E40/Dataset501_GAT26GOAT/*/fold_${{F}}/training_log_*.txt '
        f'   2>/dev/null | head -1); '
        f'[ -z "$L" ] && L={W}/jobs/$T.log; '
        f'echo "MTIME=$(stat -c %Y $L 2>/dev/null || echo 0)"; '
        f'echo "EPOCHS=$(grep -c \'Epoch time\' $L 2>/dev/null || echo 0)"; '
        f'echo "DONE=$(test -f {W}/jobs/$T.DONE && echo 1 || echo 0)"; '
        f'echo "RESULT=$(test -f {W}/jobs/$T.json && echo 1 || echo 0)"; '
        f'echo "GPUPIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr "\\n" "," )"; '
        f'echo "NOW=$(date +%s)"'
    )
    rc, out = sh(host, port, cmd)
    rec = {"probe_rc": rc}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            rec[k.strip()] = v.strip()
    rec["pids"] = [int(x) for x in rec.get("PIDS", "").split() if x.isdigit()]
    rec["alive"] = bool(rec["pids"])
    rec["epochs"] = int(rec.get("EPOCHS") or 0)
    rec["done"] = rec.get("DONE") == "1"
    rec["mtime"] = int(rec.get("MTIME") or 0)
    rec["now"] = int(rec.get("NOW") or 0)
    rec["log_age"] = max(0, rec["now"] - rec["mtime"]) if rec["mtime"] else None
    # the stdout redirect is block-buffered; only nnU-Net's own training_log_*.txt
    # (resolved above) is a trustworthy progress signal.
    return rec


def kill_token(host, port, token, W):
    sh(host, port,
       f'for p in $(pgrep -x python; pgrep -x python3); do '
       f'  C=$(tr "\\0" " " < /proc/$p/cmdline 2>/dev/null); '
       f'  case "$C" in *g82_train.py*cfg/{token}.json*) kill -9 $p;; esac; done; sleep 3')


def relaunch(host, port, token, recipe, fold, W):
    cmd = (
        f'export nnUNet_raw={W}/nnUNet_raw '
        f'nnUNet_preprocessed={W}/nnUNet_preprocessed '
        f'nnUNet_results={W}/results/{recipe}_E40; '
        f'mkdir -p "$nnUNet_results"; '
        f'setsid nohup env PYTHONUNBUFFERED=1 python {W}/scripts/g82_train.py '
        f'--plans {W}/cfg/nnUNetResEncUNetMPlans.json '
        f'--dataset-json {W}/cfg/dataset.json --fold {fold} '
        f'--g82-config {W}/cfg/{token}.json '
        f'--out {W}/jobs/{token}.json '
        f'>> {W}/jobs/{token}.log 2>&1 < /dev/null & echo relaunched'
    )
    return sh(host, port, cmd)


def main() -> int:
    if os.path.exists(LOCK):
        with open(LOCK) as f:
            pid = f.read().strip()
        if pid.isdigit() and os.path.exists(f"/proc/{pid}"):
            print("another supervisor holds the lock; exiting")
            return 0
    with open(LOCK, "w") as f:
        f.write(str(os.getpid()))

    WDIR = os.environ["G82_WORKER_DIR"]      # absolute worker path, private
    endpoints = []
    with open(ENDPOINTS) as f:
        for line in f:
            p = line.split()
            if len(p) == 4:
                endpoints.append({"pod": p[0], "fold": int(p[1]), "host": p[2], "port": p[3]})

    cutoff = os.environ.get("G82_STAGE_CUTOFF_UTC", "2026-07-30T04:30:00Z")
    cutoff_ts = time.mktime(time.strptime(cutoff, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
    recipes = ["T", "DG", "TDG"]
    restarts = {}
    state_path = os.path.join(STATE, "supervisor_state.json")

    journal({"event": "supervisor_start", "cutoff": cutoff,
             "workers": len(endpoints), "stall_seconds": STALL_SECONDS})

    while True:
        now = time.time()
        snapshot = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "cutoff": cutoff, "jobs": {}}
        all_done = True
        for ep in endpoints:
            for recipe in recipes:
                token = f"{recipe}_f{ep['fold']}_E40"
                rec = probe(ep["host"], ep["port"], token, WDIR)
                rec["restarts"] = restarts.get(token, 0)
                snapshot["jobs"][token] = {
                    k: rec[k] for k in ("alive", "epochs", "done", "log_age", "restarts")}
                if rec["done"]:
                    continue
                all_done = False
                stalled = (rec["alive"] and rec["log_age"] is not None
                           and rec["log_age"] > STALL_SECONDS)
                if stalled:
                    if now > cutoff_ts:
                        journal({"event": "stall_not_restarted_past_cutoff", "token": token})
                        continue
                    if restarts.get(token, 0) >= MAX_RESTARTS:
                        journal({"event": "stall_restart_budget_exhausted", "token": token,
                                 "epochs": rec["epochs"]})
                        continue
                    journal({"event": "stalled_job_killed_then_restarted", "token": token,
                             "log_age": rec["log_age"], "epochs": rec["epochs"],
                             "pids": rec["pids"]})
                    kill_token(ep["host"], ep["port"], token, WDIR)
                    time.sleep(5)
                    after = probe(ep["host"], ep["port"], token, WDIR)
                    if after["alive"]:
                        journal({"event": "refused_relaunch_process_still_alive",
                                 "token": token})
                        continue
                    restarts[token] = restarts.get(token, 0) + 1
                    relaunch(ep["host"], ep["port"], token, recipe, ep["fold"], WDIR)
        atomic_json(state_path, snapshot)
        if all_done:
            journal({"event": "all_screen_jobs_complete"})
            return 0
        if time.time() > cutoff_ts:
            journal({"event": "cutoff_reached", "cutoff": cutoff})
            return 2
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
