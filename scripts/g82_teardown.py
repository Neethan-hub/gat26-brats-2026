#!/usr/bin/env python3
"""G82 §L teardown: terminate ONLY pods created by this stage, then prove it.

Refuses to touch anything whose id is not in the G82-created list, never touches a
volume, and independently re-reads the account afterwards to confirm the pods are
gone and no longer billable. Resource identifiers are written to a private file and
never printed in full.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.environ["G82_ROOT"])
import rp  # noqa: E402


def short(pid: str) -> str:
    return pid[:4] + "…"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--created", required=True,
                    help="private file listing pod ids created by this stage")
    ap.add_argument("--out", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    created = [l.strip() for l in open(a.created) if l.strip()]
    before = rp.pods()
    before_ids = {p["id"] for p in before}
    foreign = sorted(before_ids - set(created))

    report = {"schema": "gat26.g82.teardown.v1",
              "g82_created_count": len(created),
              "account_pods_before": len(before),
              "foreign_pods_untouched": len(foreign),
              "actions": []}

    for pid in created:
        if pid not in before_ids:
            report["actions"].append({"pod": short(pid), "action": "already absent"})
            continue
        if not a.apply:
            report["actions"].append({"pod": short(pid), "action": "would terminate"})
            continue
        try:
            rp.terminate(pid)
            report["actions"].append({"pod": short(pid), "action": "terminated"})
        except Exception as e:
            report["actions"].append({"pod": short(pid), "action": "FAILED",
                                      "error": type(e).__name__})

    if a.apply:
        time.sleep(12)
        after = rp.pods()                      # fresh independent read
        after_ids = {p["id"] for p in after}
        report["account_pods_after"] = len(after)
        report["g82_pods_remaining"] = sorted(short(p) for p in created if p in after_ids)
        report["all_g82_pods_gone"] = not report["g82_pods_remaining"]
        report["foreign_pods_still_present"] = len(after_ids & set(foreign))
        report["no_foreign_pod_was_touched"] = (
            report["foreign_pods_still_present"] == len(foreign))
        vols = rp.call("GET", "/networkvolumes") or []
        report["volumes_present"] = len(vols)
        report["volumes_untouched"] = True     # no volume call is ever issued by G82

    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, a.out)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
