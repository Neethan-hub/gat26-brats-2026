#!/usr/bin/env python3
"""GAT-26 G4 — official BraTS-evaluation 0.0.8 GoAT controls (run in the eval venv).

Semantic assertions on produced JSON/CSV (never trust CLI exit status). Actual
two-case smoke scores are labeled g4_real_data_smoke_only / no_accuracy_claim and
kept private. Sanitized PASS/FAIL + aggregate facts to stdout."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import brats_evaluation as be
from panoptica import Panoptica_Evaluator


def _write(arr, path, spacing=(1.0, 1.0, 1.0)):
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr.astype("uint8")))
    img.SetSpacing(spacing)
    sitk.WriteImage(img, str(path))


def _positive_ref(shape=(32, 32, 32)):
    a = np.zeros(shape, dtype="uint8")
    a[4:12, 4:12, 4:12] = 1          # NCR -> TC, WT
    a[6:9, 6:9, 6:9] = 3             # ET  -> ET, TC, WT
    a[16:22, 16:22, 16:22] = 2       # ED  -> WT
    return a


def load_evaluator():
    return Panoptica_Evaluator.load_from_config(str(be.config_path("GoAT")))


def _regions(result: dict):
    """Return the per-region DSC/HD95 from an evaluate_single_exam result dict."""
    return result


def verify_config() -> dict:
    txt = Path(be.config_path("GoAT")).read_text(encoding="utf-8")
    return {
        "et_[3]": "et:" in txt and "[3]" in txt,
        "tc_[1,3]": "[1, 3]" in txt,
        "wt_[1,2,3]": "[1, 2, 3]" in txt,
        "global_DSC_NSD_HD95": all(m in txt for m in ("DSC", "NSD", "HD95")),
        "hd95_zeroTP_INF": "HD95" in txt and "INF" in txt,
    }


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def parser_json(records):
    """Wrap panoptica result dicts in the parser's expected {metrics, missings} shape."""
    return {"metrics": [_jsonable(r) for r in records], "missings": [], "errors": []}


def run_eval(pred, ref, subject, ev, tmp):
    return be.evaluate_single_exam(str(pred), str(ref), subject, ev)


def region_metric(res, region, metric):
    """Extract the official GLOBAL region metric (global_bin_dsc/hd95/nsd)."""
    key = {"dsc": "global_bin_dsc", "hd95": "global_bin_hd95", "nsd": "global_bin_nsd"}[metric.lower()]
    reg = res.get(region)
    if isinstance(reg, dict) and key in reg:
        return float(reg[key])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actual-pred-dir", required=True)
    ap.add_argument("--actual-ref-json", required=True, help="private json: {alias: {pred, ref}}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {"config": verify_config(), "controls": {}, "actual": {}}
    ev = load_evaluator()
    tmp = tempfile.mkdtemp()

    ref = _positive_ref()
    # 1. perfect prediction
    rp = Path(tmp) / "perf_ref.nii.gz"; pp = Path(tmp) / "perf_pred.nii.gz"
    _write(ref, rp); _write(ref.copy(), pp)
    res = run_eval(pp, rp, "GAT26SMOKE_90000", ev, tmp)
    dsc_et = region_metric(res, "et", "dsc"); hd_et = region_metric(res, "et", "hd95")
    out["controls"]["perfect"] = {
        "et_dsc": dsc_et, "et_hd95": hd_et,
        "ok": (dsc_et is not None and abs(float(dsc_et) - 1.0) < 1e-3
               and hd_et is not None and abs(float(hd_et)) < 1e-3)}

    # 2. all-zero prediction on positive reference
    zp = Path(tmp) / "zero_pred.nii.gz"; _write(np.zeros_like(ref), zp)
    zj = Path(tmp) / "zero.json"
    res0 = be.evaluate_single_exam(str(zp), str(rp), "GAT26SMOKE_90001", ev)
    zj.write_text(json.dumps(parser_json([res0])), encoding="utf-8")
    zcsv = Path(tmp) / "zero.csv"
    be.parse_seg_results(str(zj), str(zcsv))
    csvtext = zcsv.read_text(encoding="utf-8")
    dsc0 = region_metric(res0, "et", "dsc"); hd0 = region_metric(res0, "et", "hd95")
    out["controls"]["all_zero"] = {
        "et_dsc": dsc0, "et_hd95_raw": (None if hd0 is None else (float(hd0) if np.isfinite(float(hd0)) else "INF")),
        "parser_maps_373": "373" in csvtext,
        "ok": (dsc0 is not None and float(dsc0) == 0.0 and "373" in csvtext)}

    # The evaluator RECORDS errors (returns an "error" dict) rather than raising, so the
    # GAT-26 wrapper inspects the result semantically and treats it as a hard failure.
    def is_hard_failure(pred_path, subject):
        try:
            r = be.evaluate_single_exam(str(pred_path), str(rp), subject, ev)
        except Exception:
            return True                     # raising is also a hard failure
        return isinstance(r, dict) and ("error" in r)

    # 3. missing / wrong-name control: nonexistent prediction path
    out["controls"]["missing_name"] = {}
    hf = is_hard_failure(Path(tmp) / "does_not_exist.nii.gz", "GAT26SMOKE_90002")
    out["controls"]["missing_name"] = {"wrapper_hard_fail": hf, "ok": hf}

    # 4. geometry mismatch: different shape
    gp = Path(tmp) / "geo_pred.nii.gz"; _write(np.zeros((16, 16, 16), "uint8"), gp)
    hf2 = is_hard_failure(gp, "GAT26SMOKE_90003")
    out["controls"]["geometry_mismatch"] = {"wrapper_hard_fail": hf2, "ok": hf2}

    # 5. actual one-step outputs (private scores)
    refmap = json.loads(Path(args.actual_ref_json).read_text(encoding="utf-8"))
    actual_json = Path(tmp) / "actual.json"; records = {}
    n_err = 0
    for alias, pr in refmap.items():
        try:
            r = be.evaluate_single_exam(str(pr["pred"]), str(pr["ref"]), alias, ev)
            records[alias] = r
        except Exception:
            n_err += 1
    actual_json.write_text(json.dumps(parser_json(list(records.values()))), encoding="utf-8")
    actual_csv = Path(tmp) / "actual.csv"
    be.parse_seg_results(str(actual_json), str(actual_csv))
    ctext = actual_csv.read_text(encoding="utf-8").lower()
    six = all(f"global_{m}_{reg}" in ctext
              for reg in ("et", "tc", "wt") for m in ("dsc", "hd95"))
    out["actual"] = {
        "label": "g4_real_data_smoke_only", "no_accuracy_claim": True,
        "n_metric_records": len(records), "n_errors": n_err, "n_missings": 0,
        "six_global_columns_present": six,
        "ok": (len(records) == len(refmap) and n_err == 0 and six)}

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    # sanitized (no numeric actual scores)
    san = {"config": out["config"],
           "controls": {k: {kk: vv for kk, vv in v.items() if kk in ("ok", "parser_maps_373", "wrapper_hard_fail")}
                        for k, v in out["controls"].items()},
           "actual": {k: v for k, v in out["actual"].items() if k != "scores"}}
    print(json.dumps(san))
    allok = (all(v["ok"] for v in out["controls"].values())
             and out["actual"]["ok"] and all(out["config"].values()))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
