#!/usr/bin/env python3
"""GAT-26 G4 — hierarchy-safe reconstruction + fail-closed output validation."""
from __future__ import annotations

import hashlib
import re

LEGAL = frozenset({0, 1, 2, 3})
# Strict Task 3 basename: <prefix>NNNNN.nii.gz where NNNNN is exactly 5 digits, no
# timepoint. The 5-digit block must be the final token before .nii.gz and must not be
# preceded or followed by another digit.
_TASK3_RE = re.compile(r"^(?:[^/\\]*?[^0-9])?(?<!\d)(\d{5})\.nii\.gz$")


def project_and_reconstruct(q_wt, q_tc, q_et, tau_wt=0.5, tau_tc=0.5, tau_et=0.5):
    """p_ET=q_ET; p_TC=max(q_TC,p_ET); p_WT=max(q_WT,p_TC); explicit nested masks."""
    import numpy as np
    q_wt = np.asarray(q_wt); q_tc = np.asarray(q_tc); q_et = np.asarray(q_et)
    p_et = q_et
    p_tc = np.maximum(q_tc, p_et)
    p_wt = np.maximum(q_wt, p_tc)
    m_wt = p_wt >= tau_wt
    m_tc = (p_tc >= tau_tc) & m_wt
    m_et = (p_et >= tau_et) & m_tc
    seg = np.zeros(q_wt.shape, dtype="uint8")
    seg[m_wt] = 2
    seg[m_tc] = 1
    seg[m_et] = 3
    return seg


def validate_mask(seg, require_integer_dtype=True) -> dict:
    """Fail-closed validation. NO int()/round() before checks — fractional/NaN/Inf/4/
    negative are rejected. `require_integer_dtype` enforces a stored-output integer dtype."""
    import numpy as np
    seg = np.asarray(seg)
    res = {"ok": False, "dims3": seg.ndim == 3, "numeric": np.issubdtype(seg.dtype, np.number),
           "integer_dtype": bool(np.issubdtype(seg.dtype, np.integer)),
           "all_finite": None, "integer_legal": False, "value_set": None,
           "et_subset_tc": False, "tc_subset_wt": False}
    if not res["numeric"] or not res["dims3"]:
        return res
    # finiteness on the raw values (float masks may carry NaN/Inf)
    finite = bool(np.isfinite(seg).all()) if np.issubdtype(seg.dtype, np.floating) else True
    res["all_finite"] = finite
    if not finite:
        return res
    if require_integer_dtype and not res["integer_dtype"]:
        return res  # a float mask (even if values are integral) is rejected as stored output
    uniq = np.unique(seg)
    # reject any non-integral value WITHOUT casting first
    if np.issubdtype(seg.dtype, np.floating) and not np.array_equal(uniq, np.floor(uniq)):
        return res
    vals = set(int(v) for v in uniq)  # safe now: all integral and finite
    res["value_set"] = sorted(vals)
    res["integer_legal"] = vals.issubset(LEGAL)
    if not res["integer_legal"]:
        return res
    et = seg == 3
    tc = (seg == 1) | (seg == 3)
    wt = (seg == 1) | (seg == 2) | (seg == 3)
    res["et_subset_tc"] = bool(np.all(tc[et]))
    res["tc_subset_wt"] = bool(np.all(wt[tc]))
    res["ok"] = res["integer_legal"] and res["et_subset_tc"] and res["tc_subset_wt"]
    return res


def mask_hash(seg) -> str:
    import numpy as np
    seg = np.ascontiguousarray(np.asarray(seg).astype("uint8"))
    h = hashlib.sha256(); h.update(str(seg.shape).encode()); h.update(seg.tobytes())
    return h.hexdigest()


def task3_output_name(case_five_digit: str) -> str:
    """Strict: input is a bare 5-digit case ID; output ends in exactly that ID + .nii.gz."""
    s = str(case_five_digit)
    if "/" in s or "\\" in s:
        raise ValueError("case id must not be a path")
    if not re.fullmatch(r"\d{5}", s):
        raise ValueError(f"case id must be exactly 5 digits, got {s!r}")
    return f"{s}.nii.gz"


def validate_task3_basename(name: str) -> str:
    """Validate a produced output basename; return the 5-digit id or raise."""
    if "/" in name or "\\" in name:
        raise ValueError("output must be a flat basename, not a nested path")
    m = _TASK3_RE.match(name)
    if not m:
        raise ValueError("name must end in exactly one 5-digit case id + .nii.gz (no timepoint)")
    # reject 4/6-digit blocks: ensure the digit run immediately before .nii.gz is exactly 5
    tail = name[:-len(".nii.gz")]
    run = re.search(r"(\d+)$", tail).group(1)
    if len(run) != 5:
        raise ValueError(f"trailing digit run must be exactly 5, got {len(run)}")
    return m.group(1)


def validate_output_file(path, ref_affine, ref_shape, ref_zooms, ref_axcodes,
                         atol=1e-4) -> dict:
    """Validate a saved .nii.gz on reload against the source reference geometry."""
    import numpy as np
    import nibabel as nib
    res = {"ok": False, "readable": False, "name_ok": False, "geometry_ok": False,
           "mask_ok": False}
    try:
        validate_task3_basename(str(path).split("/")[-1]); res["name_ok"] = True
    except ValueError:
        return res
    try:
        img = nib.load(str(path)); arr = np.asarray(img.dataobj)
        res["readable"] = True
    except Exception:
        return res
    mv = validate_mask(arr, require_integer_dtype=True)
    res["mask_ok"] = mv["ok"]
    res["mask_detail"] = mv
    geo = (tuple(img.shape) == tuple(ref_shape)
           and bool(np.all(np.isfinite(img.affine)))
           and bool(np.allclose(img.affine, ref_affine, rtol=0.0, atol=atol))
           and bool(np.allclose(np.asarray(img.header.get_zooms()[:3]),
                                np.asarray(ref_zooms[:3]), rtol=0.0, atol=atol))
           and nib.aff2axcodes(img.affine) == tuple(ref_axcodes))
    res["geometry_ok"] = geo
    res["ok"] = res["name_ok"] and res["readable"] and res["mask_ok"] and res["geometry_ok"]
    return res


def geometry_matches(ref_affine, ref_shape, ref_zooms, out_affine, out_shape, out_zooms,
                     atol=1e-4) -> bool:
    import numpy as np
    return (tuple(out_shape) == tuple(ref_shape)
            and bool(np.allclose(out_affine, ref_affine, rtol=0.0, atol=atol))
            and bool(np.allclose(np.asarray(out_zooms[:3]), np.asarray(ref_zooms[:3]),
                                 rtol=0.0, atol=atol)))
