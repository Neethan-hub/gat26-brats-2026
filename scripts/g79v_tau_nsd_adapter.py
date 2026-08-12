#!/usr/bin/env python3
"""GAT-26 Stage G79-V — explicit-tolerance NSD adapter over the PRESERVED Panoptica build.

WHY THIS EXISTS
---------------
The organizers confirmed (Synapse thread 14186, reply 36573, 2026-07-28) that the final Task-3
ranking uses global DSC + NSD with ``tau=1``. In the preserved evaluator stack
(``brats_evaluation`` 0.0.8 + ``panoptica`` 2.1.4) the *global* NSD tolerance is unreachable from any
supported configuration surface: ``Panoptica_Evaluator.__init__`` has no such parameter, the YAML
lists ``!Metric NSD`` as a bare entry, and ``PanopticaResult._calc_global_bin_metric`` calls the
metric with neither ``threshold`` nor ``voxelspacing`` — so it always falls back to ``0.5``.

This adapter therefore calls Panoptica's OWN installed metric callables directly with an EXPLICIT
threshold, reproducing the global-binary code path exactly. It does NOT:

  * patch or edit site-packages;
  * monkey-patch the evaluator;
  * install, upgrade, or replace any package;
  * reimplement NSD, DSC, or any distance transform;
  * use a different Panoptica version;
  * alter prediction or reference files.

FAITHFULNESS IS PROVEN, NOT ASSERTED
------------------------------------
Running this adapter at ``tau=0.5`` must reproduce the committed G7.7 aggregates for all nine
policies. That equivalence gate is the evidence that the binarization and edge-case handling match
the official evaluator. If it fails, no ``tau=1`` result may be computed.

THIS IS NOT EXACT FINAL-SCORER REPRODUCTION. Several official scorer details remain unconfirmed
(exact Panoptica version, tau units, and how tau is injected). Results are a *sensitivity audit*.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Region label groups, exactly as in the official brats_evaluation config_GoAT.yaml.
REGIONS: dict[str, tuple[int, ...]] = {"ET": (3,), "TC": (1, 3), "WT": (1, 2, 3)}
METRICS = ("DSC", "NSD")

# Calibration-only. Confirmation folds 3-4 are LOCKED for this stage and this module refuses them.
ALLOWED_FOLDS = (0, 1, 2)
LOCKED_FOLDS = (3, 4)

# r11: no machine-absolute default. The restored evaluator stack lives wherever the operator put
# it and is located only through GAT26_EVAL_SITE_PACKAGES. Absent that variable the adapter
# reports "not available" and callers fail closed rather than scoring against a wrong build.
DEFAULT_EVAL_SITE_PACKAGES = os.environ.get("GAT26_EVAL_SITE_PACKAGES", "")


class AdapterError(RuntimeError):
    """Fail-closed: any unsupported threshold path, locked-fold access, or malformed input."""


def eval_site_packages() -> str | None:
    """Locate the RESTORED evaluator site-packages (never installed, only restored read-only)."""
    p = os.environ.get("GAT26_EVAL_SITE_PACKAGES", DEFAULT_EVAL_SITE_PACKAGES)
    return p if p and Path(p, "panoptica").is_dir() else None


def _load_panoptica():
    """Import the preserved Panoptica callables, failing closed if the threshold is unsupported."""
    sp = eval_site_packages()
    if sp is None:
        raise AdapterError(
            "preserved Panoptica build not found; set GAT26_EVAL_SITE_PACKAGES to the restored "
            "evaluator site-packages. This adapter never installs or upgrades packages.")
    if sp not in sys.path:
        sys.path.insert(0, sp)
    import inspect

    from panoptica.metrics.dice import _compute_dice_coefficient as dsc
    from panoptica.metrics.normalized_surface_dice import (
        _compute_normalized_surface_dice as nsd,
    )

    params = inspect.signature(nsd).parameters
    if "threshold" not in params:
        raise AdapterError(
            "the installed Panoptica NSD callable exposes no explicit 'threshold' parameter; "
            "an exact explicit-tolerance evaluation is impossible without patching, which is "
            "forbidden. FAIL CLOSED.")
    return dsc, nsd


def panoptica_versions() -> dict:
    sp = eval_site_packages()
    if sp is None:
        return {}
    if sp not in sys.path:
        sys.path.insert(0, sp)
    import importlib.metadata as md
    out = {}
    for pkg in ("panoptica", "brats_evaluation", "numpy", "SimpleITK"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:  # noqa: BLE001
            out[pkg] = None
    return out


def assert_fold_allowed(fold: int) -> None:
    """Confirmation folds are locked for this stage. Refuse them structurally, not by convention."""
    f = int(fold)
    if f in LOCKED_FOLDS:
        raise AdapterError(
            f"fold {f} is a LOCKED confirmation fold for Stage G79-V; reading it is forbidden")
    if f not in ALLOWED_FOLDS:
        raise AdapterError(f"fold {f} is not a calibration fold {ALLOWED_FOLDS}")


def region_components(reference_arr, prediction_arr, tau: float) -> dict:
    """Six official components for ONE subject at an EXPLICIT NSD tolerance.

    Replicates Panoptica's global-binary path:
      * binarize the group labels (`!= 0` after label-group extraction is `isin(arr, labels)`);
      * if there is no overlap, apply the official ``config_GoAT.yaml`` zero-TP edge cases —
        both empty -> NaN (``no_instances_result``), otherwise 0.0
        (``empty_prediction_result`` / ``empty_reference_result`` / ``normal``, all ZERO for
        DSC and NSD);
      * otherwise call Panoptica's own DSC and NSD, the latter with the explicit threshold.

    Returns {(region, metric): float | None}; ``None`` marks a legitimately absent (NaN) component,
    which the official aggregation skips.
    """
    import numpy as np

    if tau is None:
        raise AdapterError("tau must be explicit; implicit fallback is exactly the defect audited")
    tau = float(tau)
    if not (tau > 0.0):
        raise AdapterError(f"tau must be positive, got {tau}")

    dsc_fn, nsd_fn = _load_panoptica()
    ref = np.asarray(reference_arr)
    pred = np.asarray(prediction_arr)
    if ref.shape != pred.shape:
        raise AdapterError(f"shape mismatch: reference {ref.shape} vs prediction {pred.shape}")

    out: dict = {}
    for region, labels in REGIONS.items():
        ref_b = np.isin(ref, labels)
        pred_b = np.isin(pred, labels)
        ref_empty = not ref_b.any()
        pred_empty = not pred_b.any()
        overlap = bool((ref_b & pred_b).any())

        if not overlap:
            # Official zero-TP edge cases for DSC and NSD (config_GoAT.yaml).
            val = None if (ref_empty and pred_empty) else 0.0
            out[(region, "DSC")] = val
            out[(region, "NSD")] = val
            continue

        r8 = ref_b.astype(np.uint8)
        p8 = pred_b.astype(np.uint8)
        out[(region, "DSC")] = float(dsc_fn(reference=r8, prediction=p8))
        out[(region, "NSD")] = float(nsd_fn(reference=r8, prediction=p8, threshold=tau))
    return out


def nsd_at(reference_arr, prediction_arr, tau: float) -> float:
    """Direct explicit-threshold NSD on two binary arrays (used by the synthetic tests)."""
    import numpy as np
    _, nsd_fn = _load_panoptica()
    return float(nsd_fn(reference=np.asarray(reference_arr).astype("uint8"),
                        prediction=np.asarray(prediction_arr).astype("uint8"),
                        threshold=float(tau)))


if __name__ == "__main__":
    v = panoptica_versions()
    print("preserved evaluator:", v or "NOT FOUND")
    print("regions:", REGIONS)
    print("allowed folds:", ALLOWED_FOLDS, "| locked folds:", LOCKED_FOLDS)
