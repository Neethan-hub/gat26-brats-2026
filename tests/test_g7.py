#!/usr/bin/env python3
"""GAT-26 G7 fold-isolation / routing / supervisor tests (`python3 tests/test_g7.py`).

Covers: fold isolation & routing, no output collision, no fold-0 retraining (fail-closed), no
cross-fold checkpoint loading, stop-on-failure, launch-next-only-after-audited-PASS, and exact
per-fold membership. Synthetic only — no GPU, no nnU-Net, no real case IDs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import g5_runner as R          # noqa: E402
import g7_supervisor as S      # noqa: E402

FAILS = 0
RUNNER = str(REPO / "scripts" / "g5_runner.py")


def check(name, cond):
    global FAILS
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILS += 1


def main():
    print("test_g7:")

    # 1. no fold-0 retraining / fail-closed fold routing (subprocess the runner CLI)
    def fold_exit(fold):
        r = subprocess.run([sys.executable, RUNNER, "--rundir", "/tmp/nope", "--expect", "/tmp/nope.json",
                            "--eval-python", "/bin/false", "--fold", str(fold)],
                           capture_output=True, text=True)
        return r.returncode
    check("fold0_fails_closed", fold_exit(0) == 2)          # fold 0 must never be retrained
    check("fold5_fails_closed", fold_exit(5) == 2)
    check("foldneg_fails_closed", fold_exit(-1) == 2)
    # a valid fold passes the gate (then fails later on the missing expect.json -> NOT exit 2)
    check("fold1_passes_gate", fold_exit(1) != 2)
    check("fold4_passes_gate", fold_exit(4) != 2)

    # 2. exact fold routing — output_folder is fold-specific and distinct per fold
    os.environ["nnUNet_results"] = "/RESULTS"
    outs = {f: R.output_folder("nnUNetResEncUNetMPlans", f) for f in (0, 1, 2, 3, 4)}
    check("output_folder_routes_by_fold", all(outs[f].endswith(f"fold_{f}") for f in outs))
    check("output_folders_distinct", len(set(outs.values())) == 5)
    sup = {f: str(S.fold_output_folder(f)) for f in (1, 2, 3, 4)}
    check("supervisor_fold_paths_distinct", len(set(sup.values())) == 4
          and all(f"fold_{f}" == Path(sup[f]).name for f in sup))

    # 3. no cross-fold checkpoint loading — the train command is the frozen 8-token argv with the
    #    fold, and carries NO pretrained/checkpoint/continue flag (structural guard on the source).
    src = (REPO / "scripts" / "g5_runner.py").read_text()
    check("train_cmd_is_frozen",
          'train_cmd = ["nnUNetv2_train", "501", CONFIG, str(fold), "-tr", TRAINER, "-p", plans]' in src)
    check("no_pretrained_or_continue_flag",
          "-pretrained_weights" not in src and '"--c"' not in src and "'--c'" not in src
          and "checkpoint_" not in src.split("def run")[1].split("train_cmd")[0][-200:])

    # 4. exact per-fold membership — each fold's val set is correct, disjoint from its train, and
    #    the five val sets partition the case universe exactly once.
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "splits_final.json"
        # synthetic 5-fold split over 10 synthetic ids, 2 val per fold
        ids = [f"S{i:04d}" for i in range(10)]
        splits = [{"val": ids[2 * k:2 * k + 2],
                   "train": [x for x in ids if x not in ids[2 * k:2 * k + 2]]} for k in range(5)]
        sp.write_text(json.dumps(splits))
        vals = {f: R.fold_validation_stems(sp, fold=f) for f in range(5)}
        check("membership_fold_specific", vals[1] == {"S0002", "S0003"} and vals[3] == {"S0006", "S0007"})
        union = set().union(*vals.values())
        check("membership_partitions_once",
              len(union) == 10 and sum(len(v) for v in vals.values()) == 10)
        check("membership_val_train_disjoint",
              all(vals[f].isdisjoint(set(splits[f]["train"])) for f in range(5)))
    # membership_report catches a same-count missing+extra swap
    swap = R.membership_report({"a", "b"}, {"a", "c"})
    check("membership_swap_fails", not swap["exact_set_equal"] and swap["missing"] == 1 and swap["extra"] == 1)

    # 5. launch-next-only-after-audited-PASS — the pure audit gate
    good = {"verdict": "M_COMPLETION_PASS", "fold": 2,
            "gates": {"a": {"ok": True}, "b": {"ok": True}}}
    check("audit_pass_accepts_clean", S.audit_is_pass(good, 2) is True)
    check("audit_pass_rejects_wrong_fold", S.audit_is_pass(good, 3) is False)
    check("audit_pass_rejects_no_go",
          S.audit_is_pass({**good, "verdict": "M_COMPLETION_NO_GO"}, 2) is False)
    check("audit_pass_rejects_failed_gate",
          S.audit_is_pass({**good, "gates": {"a": {"ok": True}, "b": {"ok": False}}}, 2) is False)
    check("audit_pass_rejects_missing_report", S.audit_is_pass(None, 2) is False)
    check("audit_pass_rejects_empty_gates",
          S.audit_is_pass({"verdict": "M_COMPLETION_PASS", "fold": 2, "gates": {}}, 2) is False)

    # 6. stop-on-failure — the supervisor iterates FOLDS 1..4 and returns nonzero on a failed fold
    check("supervisor_folds_are_1234", S.FOLDS == (1, 2, 3, 4))
    check("supervisor_stops_on_fail_source",
          "if not ok:" in (REPO / "scripts" / "g7_supervisor.py").read_text()
          and "return 3" in (REPO / "scripts" / "g7_supervisor.py").read_text())

    # 7. evaluator-denominator fail-closed recovery (G7 fold-1 posthoc fix)
    src_r = (REPO / "scripts" / "g5_runner.py").read_text()
    src_e = (REPO / "scripts" / "g5_evaluate.py").read_text()
    src_a = (REPO / "scripts" / "g5_completion_audit.py").read_text()
    # (a) the runner passes a DYNAMICALLY DERIVED denominator, not a constant
    check("runner_passes_expected_n", '"--expected-n", str(expected_n)' in src_r)
    check("runner_expected_n_from_membership",
          "expected_n = len(expected_stems)" in src_r
          and "fold_validation_stems(pp / \"splits_final.json\", fold=fold)" in src_r)
    # (b) the audit passes the fold-derived denominator (never a hardcoded 271)
    check("audit_passes_expect_val", '"--expected-n", str(expect_val)' in src_a)
    # (c) the evaluator REQUIRES --expected-n (no default of 271)
    check("evaluator_expected_n_required", 'add_argument("--expected-n", type=int, required=True)' in src_e)
    check("evaluator_no_default_271", "default=271" not in src_e)
    # (d) omitting --expected-n fails closed (argparse exit 2)
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "g5_evaluate.py"),
                        "--preds", "/tmp/none", "--gt", "/tmp/none", "--out", "/tmp/none.json"],
                       capture_output=True, text=True)
    check("evaluator_omission_fails_closed", r.returncode == 2 and "--expected-n" in r.stderr)
    # (e) an incorrect denominator is a hard failure (aggregate_components), pure
    import g5_evaluate as E  # noqa: E402
    def bad_denom():
        recs = [{f"{reg}_dsc": 1.0, f"{reg}_hd95": 0.0, f"{reg}_status": "ok"} for reg in ("et",)]
        return E.aggregate_components(recs, n_expected=len(recs) + 1)
    try:
        bad_denom(); denom_raises = False
    except Exception:
        denom_raises = True
    check("incorrect_denominator_hard_fails", denom_raises)
    # (f) folds 1-4 derive 270 / fold 0 derives 271 from a fold-size-mimicking synthetic split
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "splits_final.json"
        # fold 0 val = 3 (like 271); folds 1-4 val = 2 (like 270)
        allids = [f"S{i:04d}" for i in range(11)]
        sizes = [3, 2, 2, 2, 2]
        splits, start = [], 0
        for sz in sizes:
            val = allids[start:start + sz]; start += sz
            splits.append({"val": val, "train": [x for x in allids if x not in val]})
        sp.write_text(json.dumps(splits))
        n = {f: len(R.fold_validation_stems(sp, fold=f)) for f in range(5)}
        check("fold0_denominator_larger", n[0] == 3 and all(n[f] == 2 for f in (1, 2, 3, 4)))

    print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
