#!/usr/bin/env python3
"""CPU-only executable contracts for the GAT-26 BraTS-GoAT system.

These tests deliberately avoid the protected challenge data and PyTorch. They
exercise label semantics, hierarchy reconstruction, numerical edge cases,
filename rules, topology arithmetic, and the separation between award-ranking
and diagnostic metrics. They are preflight tests, not evidence of accuracy.

HISTORICAL SNAPSHOT -- not a current release validator.
This script is the pre-training preflight package (version 1.2, 21 July 2026). It is retained
unmodified so that its historical 24/24 result stays reproducible; it reproduces a historical
decision rather than reasserting a current rule. It is superseded for every current claim.

Current truth, as of the r11 camera-ready:
  * Official ranking uses DSC and NSD at tau=1; HD95 is diagnostic only and is not ranked.
  * Output naming preserves the complete opaque input case-folder basename. There is no
    five-digit rule and no assumed cohort prefix.
  * Any [160,160,128] patch size here is synthetic / reference-only; the final trained plan
    is [128,160,112].
  * The corrected container image was submitted, but no organizer execution log, no
    hidden-test result, no rank and no A10G measurement exist for that corrected image.
  * The camera-ready paper and the current top-level README.md are authoritative wherever
    they differ from this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

import numpy as np


CHANNELS = (32, 64, 128, 256, 320, 320)
OFFICIAL_ENCODER_BLOCKS = (1, 3, 4, 6, 6, 6)
OFFICIAL_DECODER_CONVS = (1, 1, 1, 1, 1)
ISOTROPIC_STRIDES = (
    (1, 1, 1),
    (2, 2, 2),
    (2, 2, 2),
    (2, 2, 2),
    (2, 2, 2),
    (2, 2, 2),
)
REGIONS_CLASS_ORDER = (2, 1, 3)
AWARD_METRICS = ("DSC", "HD95")
DIAGNOSTIC_METRICS = ("NSD", "SENSITIVITY", "SPECIFICITY", "PRECISION")
GOAT_OUTPUT_RE = re.compile(r"(?:^|[^0-9])(?P<case_id>[0-9]{5})[.]nii[.]gz$")


def _validate_numeric(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be numeric")
    if np.issubdtype(array.dtype, np.complexfloating) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite real values")
    return array


def labels_to_regions(labels: np.ndarray) -> np.ndarray:
    array = _validate_numeric("labels", labels)
    if np.any(array != np.rint(array)):
        raise ValueError("labels must be integers")
    legal = np.isin(array, (0, 1, 2, 3))
    if not bool(np.all(legal)):
        bad = np.unique(array[~legal]).tolist()
        raise ValueError(f"illegal GoAT labels: {bad}")
    wt = np.isin(array, (1, 2, 3))
    tc = np.isin(array, (1, 3))
    et = array == 3
    return np.stack((wt, tc, et), axis=0)


def regions_to_labels(regions: np.ndarray) -> np.ndarray:
    array = _validate_numeric("regions", regions)
    if array.ndim < 2 or array.shape[0] != 3:
        raise ValueError("regions must have first dimension WT, TC, ET")
    if not bool(np.all(np.isin(array, (0, 1, False, True)))):
        raise ValueError("regions must be binary")
    wt, tc_raw, et_raw = array.astype(bool)
    tc = tc_raw & wt
    et = et_raw & tc
    out = np.zeros(wt.shape, dtype=np.uint8)
    out[wt] = 2
    out[tc] = 1
    out[et] = 3
    return out


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    array = _validate_numeric("probabilities", probabilities).astype(np.float64)
    if array.ndim < 2 or array.shape[0] != 3:
        raise ValueError("probabilities must have first dimension WT, TC, ET")
    if np.any((array < 0) | (array > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    return array


def project_probabilities(probabilities: np.ndarray) -> np.ndarray:
    q = _validate_probabilities(probabilities)
    p_et = q[2]
    p_tc = np.maximum(q[1], p_et)
    p_wt = np.maximum(q[0], p_tc)
    return np.stack((p_wt, p_tc, p_et), axis=0)


def threshold_nested(probabilities: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    p = _validate_probabilities(probabilities)
    t = _validate_numeric("thresholds", np.asarray(thresholds)).astype(np.float64)
    if t.shape != (3,) or np.any((t < 0) | (t > 1)):
        raise ValueError("thresholds must be three finite values in [0, 1]")
    wt = p[0] >= t[0]
    tc = (p[1] >= t[1]) & wt
    et = (p[2] >= t[2]) & tc
    return np.stack((wt, tc, et), axis=0)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    z = _validate_numeric("logits", logits).astype(np.float64)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def surrogate_dice_bce_loss(
    logits: np.ndarray, targets: np.ndarray, smooth: float = 1e-5
) -> float:
    """Numerical surrogate used only for edge-case testing.

    This is intentionally not described as nnU-Net's native PyTorch loss. The
    GPU harness imports the actual nnU-Net loss implementation.
    """

    z = _validate_numeric("logits", logits).astype(np.float64)
    y = _validate_numeric("targets", targets).astype(np.float64)
    if z.shape != y.shape or z.ndim < 2:
        raise ValueError("logits and targets must have equal non-scalar shapes")
    if np.any((y < 0) | (y > 1)):
        raise ValueError("targets must be in [0, 1]")
    bce = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    p = sigmoid(z)
    axes = tuple(range(1, p.ndim))
    intersection = np.sum(p * y, axis=axes)
    denominator = np.sum(p, axis=axes) + np.sum(y, axis=axes)
    dice = (2 * intersection + smooth) / (denominator + smooth)
    return float(np.mean(bce) + np.mean(1.0 - dice))


def hierarchy_penalty(probabilities: np.ndarray) -> float:
    q = _validate_probabilities(probabilities)
    return float(
        np.mean(np.maximum(q[1] - q[0], 0) + np.maximum(q[2] - q[1], 0))
    )


def stage_shapes(
    patch: Sequence[int], strides: Iterable[Sequence[int]]
) -> list[tuple[int, int, int]]:
    current = tuple(int(v) for v in patch)
    if len(current) != 3 or any(v <= 0 for v in current):
        raise ValueError("patch must have three positive dimensions")
    shapes: list[tuple[int, int, int]] = []
    for stride_values in strides:
        stride = tuple(int(v) for v in stride_values)
        if len(stride) != 3 or any(v <= 0 for v in stride):
            raise ValueError("each stride must have three positive dimensions")
        if any(size % step for size, step in zip(current, stride)):
            raise ValueError(f"shape {current} is not divisible by stride {stride}")
        current = tuple(size // step for size, step in zip(current, stride))
        shapes.append(current)
    return shapes


def parse_goat_output_name(filename: str) -> str:
    name = PurePosixPath(filename).name
    if name != filename or not GOAT_OUTPUT_RE.search(name):
        raise ValueError("Task 3 output must be flat and end with 5-digit case ID + .nii.gz")
    match = GOAT_OUTPUT_RE.search(name)
    assert match is not None
    return match.group("case_id")


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, object]] = []

    def check(self, name: str, condition: bool, detail: object) -> None:
        passed = bool(condition)
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            raise AssertionError(f"{name}: {detail}")


def _raises_value_error(callable_) -> bool:
    try:
        callable_()
    except ValueError:
        return True
    return False


def run() -> dict[str, object]:
    audit = Audit()
    rng = np.random.default_rng(21072026)

    legal = np.array([0, 1, 2, 3], dtype=np.uint8)
    audit.check(
        "exhaustive legal label-region round trip",
        np.array_equal(legal, regions_to_labels(labels_to_regions(legal))),
        legal.tolist(),
    )

    random_round_trips = True
    for _ in range(200):
        labels = rng.integers(0, 4, size=(7, 6, 5), dtype=np.uint8)
        random_round_trips &= np.array_equal(
            labels, regions_to_labels(labels_to_regions(labels))
        )
    audit.check("random legal round trips", random_round_trips, "200 arrays")

    invalid_labels = [
        np.array([0, 4]),
        np.array([0, -1]),
        np.array([0.0, 1.5]),
        np.array([0.0, np.nan]),
    ]
    audit.check(
        "illegal/noninteger/nonfinite labels rejected",
        all(_raises_value_error(lambda x=x: labels_to_regions(x)) for x in invalid_labels),
        "-1, 4, 1.5, NaN",
    )

    audit.check(
        "malformed region tensors rejected",
        _raises_value_error(lambda: regions_to_labels(np.zeros((2, 2, 2))))
        and _raises_value_error(lambda: regions_to_labels(np.full((3, 2, 2), 0.5))),
        "wrong channel count and nonbinary values",
    )

    adversarial_regions = np.zeros((3, 2, 2), dtype=np.uint8)
    adversarial_regions[1, 0, 0] = 1
    adversarial_regions[2, 1, 1] = 1
    repaired_labels = regions_to_labels(adversarial_regions)
    audit.check(
        "reconstruction cannot emit child region outside parent",
        np.count_nonzero(repaired_labels) == 0,
        repaired_labels.tolist(),
    )

    invalid_probabilities = [
        np.zeros((2, 2, 2)),
        np.full((3, 2, 2), -0.01),
        np.full((3, 2, 2), 1.01),
        np.full((3, 2, 2), np.nan),
    ]
    audit.check(
        "invalid probabilities rejected",
        all(
            _raises_value_error(lambda x=x: project_probabilities(x))
            for x in invalid_probabilities
        ),
        "wrong channels, out-of-range, and NaN",
    )

    audit.check(
        "invalid thresholds rejected",
        _raises_value_error(lambda: threshold_nested(np.zeros((3, 1)), (0.5, 0.5)))
        and _raises_value_error(
            lambda: threshold_nested(np.zeros((3, 1)), (0.5, 1.1, 0.5))
        )
        and _raises_value_error(
            lambda: threshold_nested(np.zeros((3, 1)), (0.5, np.nan, 0.5))
        ),
        "wrong count, out-of-range, and NaN",
    )

    hand = np.array([[0.2], [0.8], [0.6]])
    expected_projection = np.array([[0.8], [0.8], [0.6]])
    audit.check(
        "cumulative-max projection exact",
        np.array_equal(project_probabilities(hand), expected_projection),
        project_probabilities(hand).tolist(),
    )

    projected = project_probabilities(rng.random((3, 13, 11, 7)))
    audit.check(
        "projection hierarchy",
        np.all(projected[0] >= projected[1])
        and np.all(projected[1] >= projected[2]),
        "WT >= TC >= ET",
    )
    audit.check(
        "projection idempotence",
        np.array_equal(project_probabilities(projected), projected),
        "P(P(q)) == P(q)",
    )

    counter_p = np.array([[[0.55]], [[0.54]], [[0.10]]])
    counter_t = (0.60, 0.50, 0.50)
    raw = counter_p[:, 0, 0] >= np.asarray(counter_t)
    audit.check(
        "unequal-threshold counterexample reproduced",
        bool(raw[1] and not raw[0]),
        raw.tolist(),
    )
    fixed = threshold_nested(counter_p, counter_t)
    audit.check(
        "explicit intersection repairs counterexample",
        not bool(fixed[1, 0, 0]),
        fixed[:, 0, 0].tolist(),
    )

    nested_property = True
    monotone_et_property = True
    for _ in range(1000):
        q = project_probabilities(rng.random((3, 9, 8, 7)))
        thresholds = rng.uniform(0.05, 0.95, size=3)
        masks = threshold_nested(q, thresholds)
        nested_property &= bool(
            np.all(~masks[2] | masks[1]) and np.all(~masks[1] | masks[0])
        )
        raised = thresholds.copy()
        raised[2] = min(1.0, raised[2] + 0.05)
        masks_raised = threshold_nested(q, raised)
        monotone_et_property &= bool(np.all(~masks_raised[2] | masks[2]))
    audit.check("randomized nested-mask property", nested_property, "1000 trials")
    audit.check(
        "raising ET threshold cannot add ET voxels",
        monotone_et_property,
        "1000 trials",
    )

    zero = np.zeros((3, 4, 4, 4), dtype=np.float64)
    one = np.ones_like(zero)
    extreme = np.full_like(zero, 1000.0)
    losses = {
        "empty/extreme-positive": surrogate_dice_bce_loss(extreme, zero),
        "full/extreme-negative": surrogate_dice_bce_loss(-extreme, one),
        "empty/neutral": surrogate_dice_bce_loss(zero, zero),
    }
    audit.check(
        "surrogate loss finite at numerical edge cases",
        all(math.isfinite(v) for v in losses.values()),
        losses,
    )
    correct_loss = surrogate_dice_bce_loss(np.where(one > 0, 8.0, -8.0), one)
    inverse_loss = surrogate_dice_bce_loss(np.where(one > 0, -8.0, 8.0), one)
    audit.check(
        "surrogate loss direction sanity",
        correct_loss < inverse_loss,
        {"correct": correct_loss, "inverse": inverse_loss},
    )

    nested_q = np.array([0.9, 0.7, 0.4])[:, None]
    violating_q = np.array([0.2, 0.8, 0.9])[:, None]
    penalties = {
        "nested": hierarchy_penalty(nested_q),
        "violating": hierarchy_penalty(violating_q),
    }
    audit.check(
        "hierarchy penalty direction",
        penalties["nested"] == 0 and penalties["violating"] > 0,
        penalties,
    )

    shapes = stage_shapes((160, 160, 128), ISOTROPIC_STRIDES)
    expected_shapes = [
        (160, 160, 128),
        (80, 80, 64),
        (40, 40, 32),
        (20, 20, 16),
        (10, 10, 8),
        (5, 5, 4),
    ]
    audit.check("reference patch stage arithmetic", shapes == expected_shapes, shapes)
    audit.check(
        "invalid patch/stride rejected",
        _raises_value_error(
            lambda: stage_shapes((159, 160, 128), ISOTROPIC_STRIDES)
        )
        and _raises_value_error(lambda: stage_shapes((160, 160), ISOTROPIC_STRIDES))
        and _raises_value_error(lambda: stage_shapes((160, 160, 128), ((0, 2, 2),))),
        "nondivisible, wrong-rank, and zero stride",
    )

    feature_elements = sum(
        int(np.prod(shape)) * channels for shape, channels in zip(shapes, CHANNELS)
    )
    audit.check(
        "exact encoder feature lower bound",
        feature_elements == 139_552_000,
        feature_elements,
    )

    contract = {
        "channels": CHANNELS,
        "encoder_blocks": OFFICIAL_ENCODER_BLOCKS,
        "decoder_convs": OFFICIAL_DECODER_CONVS,
        "conv_bias": True,
        "normalization": "InstanceNorm3d(affine=True,eps=1e-5)",
    }
    contract_json = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract_sha256 = hashlib.sha256(contract_json.encode("utf-8")).hexdigest()
    audit.check(
        "pinned ResEnc contract sentinel",
        contract_sha256 == "24e8e6b1f1c0ff32f1d81ccc3c79db26b2cc3fde57be48b19804efdc5346ddfd",
        contract_sha256,
    )

    audit.check(
        "GoAT reconstruction class order",
        REGIONS_CLASS_ORDER == (2, 1, 3),
        REGIONS_CLASS_ORDER,
    )
    audit.check(
        "award and diagnostic metrics are not conflated",
        AWARD_METRICS == ("DSC", "HD95")
        and "NSD" not in AWARD_METRICS
        and "NSD" in DIAGNOSTIC_METRICS,
        {"award": AWARD_METRICS, "diagnostic": DIAGNOSTIC_METRICS},
    )

    valid_names = (
        "BraTS-GoAT-02610.nii.gz",
        "02610.nii.gz",
        "prediction_02610.nii.gz",
    )
    invalid_names = (
        "BraTS-GoAT-02610-000.nii.gz",
        "0261.nii.gz",
        "026100.nii.gz",
        "nested/02610.nii.gz",
        "02610.nii",
    )
    audit.check(
        "Task 3 output-name and flat-directory contract",
        all(parse_goat_output_name(name) == "02610" for name in valid_names)
        and all(
            _raises_value_error(lambda name=name: parse_goat_output_name(name))
            for name in invalid_names
        ),
        {"valid": valid_names, "invalid": invalid_names},
    )

    passed = sum(int(item["passed"]) for item in audit.checks)
    return {
        "suite": "GAT-26 static architecture and release contracts",
        "suite_version": "1.2",
        "status": "PASS" if passed == len(audit.checks) else "FAIL",
        "passed": passed,
        "total": len(audit.checks),
        "checks": audit.checks,
        "official_award_metrics": list(AWARD_METRICS),
        "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
        "resenc_contract_sha256": contract_sha256,
        "reference_feature_elements_encoder_only": feature_elements,
        "reference_feature_memory_mib_encoder_only": {
            "fp16": feature_elements * 2 / 2**20,
            "fp32": feature_elements * 4 / 2**20,
        },
        "limitations": [
            "This suite does not execute PyTorch convolutions or nnU-Net.",
            "The NumPy loss is a numerical surrogate, not nnU-Net's native loss.",
            "It does not measure full-case sliding-window A10G memory or runtime.",
            "It cannot estimate GoAT accuracy without protected data and training.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("static_results.json"))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: result[key] for key in ("suite", "status", "passed", "total")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
