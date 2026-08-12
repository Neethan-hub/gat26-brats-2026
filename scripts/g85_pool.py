#!/usr/bin/env python3
"""G85 supportive all-five-fold analysis.

Combines the immutable G84 fold-0-2 calibration records with the new G85 fold-3-4
confirmation records. This is supportive evidence only and never substitutes for the
independent confirmation decision.

Fold 0-2 provenance, stated precisely because it mixes two sources:

  * M8 comes from the calibration segmentations preserved by G84 and is re-measured
    here with the same evaluator and the same corrected lesion code as folds 3-4.
  * C0 comes from the preserved full-precision no-TTA release cache, whose per-case
    components and lesion counters were reproduced exactly by G84's Part D gate. Its
    stored counters were independently verified to agree with the corrected counters
    (tp/fp/fn map onto tp_pred_diagnostic/fp_pred/fn_ref with zero disagreement), and
    ``n_ref`` is recomputed from the reference alone, which is policy-invariant.

No C0 segmentation is regenerated for folds 0-2, so no calibration rerun occurs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import g85_eval as EV  # noqa: E402
import g85_lesion_audit as LA  # noqa: E402

CAL_M8_STORE = os.environ["G85_CAL_M8_STORE"]
CAL_C0_CACHE = os.environ["G85_CAL_C0_CACHE"]
COMPONENTS = EV.COMPONENTS
REGIONS = LA.REGIONS


def _cal_one(args):
    """One calibration case: M8 measured from its preserved segmentation, C0 from cache."""
    cid, fold, cached = args
    A = EV._eval_mod()
    ref = EV.read_gt(cid)
    pred = np.asarray(np.load(os.path.join(CAL_M8_STORE, f"f{fold}", "seg", f"{cid}.npz"))["seg"]
                      ).astype(np.int16)
    c05 = A.region_components(ref, pred, 0.5)
    c10 = A.region_components(ref, pred, 1.0)
    rec = {"case": cid, "fold": fold,
           "M8_t05": {f"{r}_{m}": c05[(r, m)] for r, m in COMPONENTS},
           "M8_t10": {f"{r}_{m}": c10[(r, m)] for r, m in COMPONENTS},
           "M8": LA.per_case_stats(ref, pred),
           "C0_t05": cached["base_t05"], "C0_t10": cached["base_t10"]}
    # n_ref depends only on the reference, so C0 shares M8's reference counts exactly.
    rec["C0"] = {}
    for r in REGIONS:
        tp, fp, fn = cached["base_lesion"][r]
        rec["C0"][r] = {"n_ref": rec["M8"][r]["n_ref"], "fn_ref": int(fn),
                        "fp_pred": int(fp), "tp_pred_diagnostic": int(tp),
                        "ref_component_voxels": rec["M8"][r]["ref_component_voxels"],
                        "missed_component_voxels": []}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--confirmation-records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nproc", type=int, default=44)
    a = ap.parse_args()
    if a.nproc > EV.MAX_PROCS:
        raise SystemExit(f"refusing more than {EV.MAX_PROCS} evaluator processes")

    splits = json.load(open(a.splits, encoding="utf-8"))
    cache = {r["case"]: r for r in json.load(open(CAL_C0_CACHE, encoding="utf-8"))}
    jobs = []
    for f in (0, 1, 2):
        for cid in splits[f]["val"]:
            if (cid in cache
                    and os.path.exists(os.path.join(CAL_M8_STORE, f"f{f}", "seg", f"{cid}.npz"))):
                jobs.append((cid, f, cache[cid]))
    print(f"rebuilding {len(jobs)} calibration records (folds 0-2)", flush=True)
    with ProcessPoolExecutor(a.nproc) as ex:
        cal = list(ex.map(_cal_one, jobs, chunksize=1))

    conf = json.load(open(a.confirmation_records, encoding="utf-8"))
    recs = cal + conf
    folds = [0, 1, 2, 3, 4]
    expected = sum(len(splits[f]["val"]) for f in folds)
    print(f"pooled records: {len(recs)} of {expected} expected", flush=True)

    result = EV.build_result(recs, folds, "pooled_all_five_folds")
    result["n_expected"] = expected
    result["exact_membership"] = len(recs) == expected
    result["provenance"] = {
        "folds_0_2_M8": "preserved G84 calibration segmentations, re-measured here",
        "folds_0_2_C0": "preserved full-precision no-TTA release cache (no regeneration)",
        "folds_3_4": "generated fresh in G85 through the exact raw release path",
        "supportive_only": True}

    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, a.out)

    t1, t05 = result["t10"], result["t05"]
    ni = result["lesion_noninferiority"]
    print(f"[pooled] dU_common tau1={t1['common_support']['delta_U']:+.6f} "
          f"tau0.5={t05['common_support']['delta_U']:+.6f} | "
          f"P={t1['bootstrap']['prob_positive']:.4f} "
          f"CI=[{t1['bootstrap']['ci95'][0]:+.6f},{t1['bootstrap']['ci95'][1]:+.6f}]", flush=True)
    print(f"[pooled] miss-rate delta={ni['bootstrap']['point_delta']:+.6f} "
          f"upper95={ni['bootstrap']['upper_95_one_sided']:+.6f} "
          f"noninferior={ni['passes']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
