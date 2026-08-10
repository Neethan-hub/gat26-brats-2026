#!/usr/bin/env python3
"""Emit the sanitized aggregate evidence published with the camera-ready paper.

Reads the committed private audit artifacts and writes only aggregate, non-identifying
quantities: fold-level and subset-level means, deltas, bootstrap summaries, lesion counts
and the complete decision-check matrices. It never emits per-case values, case
identifiers, split membership, file paths, hashes, or any submission or cloud identifier.

Usage:  python3 scripts/g91_public_evidence.py <private_repo_root> <public_repo_root>
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

COMPONENTS = ["ET_DSC", "TC_DSC", "WT_DSC", "ET_NSD", "TC_NSD", "WT_NSD"]

# Keys that must never reach the public export, checked against every emitted string.
FORBIDDEN_SUBSTRINGS = (
    "/workspace", "/root", "runpod", "synapse.org", "docker.io", "ghcr.io",
    "vericerno", "sha256:",
)
# Patterns for identifiers that must not be published. These are deliberately written as shape
# patterns rather than literals: spelling the actual submission ids out here would publish, in this
# very file, the values the check exists to keep out of the export.
FORBIDDEN_PATTERNS = (
    r"\b\d{7}\b",                              # challenge submission ids
    r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{12,}\b",    # image digests / manifest hashes
)


def _load(p: Path) -> dict:
    with p.open() as fh:
        return json.load(fh)


def _subset_rows(label: str, block: dict, tau_keys: tuple[str, str],
                 n_common: int | None = None) -> list[dict]:
    """Flatten one evaluation subset into per-component rows at both tolerances.

    `n_common` supplies the common-support size for subsets that record it outside the
    per-tolerance block (the development subset stores it once at the top level).
    """
    rows: list[dict] = []
    for tau_key, tau in zip(tau_keys, ("1.0", "0.5")):
        tb = block.get(tau_key)
        if not tb:
            continue
        base = tb.get("baseline_means", {})
        cand = tb.get("candidate_means", {})
        delta = tb.get("component_deltas", {})
        for comp in COMPONENTS:
            if comp not in base:
                continue
            rows.append({
                "subset": label,
                "nsd_tolerance": tau,
                "component": comp,
                "baseline_C0": f"{base[comp]:.9f}",
                "candidate_M8": f"{cand.get(comp, float('nan')):.9f}",
                "delta": f"{delta.get(comp, float('nan')):+.9f}",
                "n_common_support": tb.get("n_common") or n_common or "",
            })
    return rows


def build(priv: Path) -> dict[str, object]:
    g84 = _load(priv / "artifacts" / "g84_result.json")
    g85 = _load(priv / "artifacts" / "g85_result.json")
    g80 = _load(priv / "artifacts" / "g80_official_validation_result.json")

    subsets = [
        ("development_folds_0_2", g84["calibration"], ("tau1", "tau05"), g84["common_support"]),
        ("policy_selection_holdout_folds_3_4", g85["confirmation"], ("t10", "t05"), None),
        ("pooled_all_folds", g85["pooled_all_five_folds"], ("t10", "t05"), None),
    ]

    components: list[dict] = []
    summary: dict[str, object] = {}
    for label, block, taus, n_common in subsets:
        components.extend(_subset_rows(label, block, taus, n_common))
        entry: dict[str, object] = {}
        for tau_key, tau in zip(taus, ("tau_1.0", "tau_0.5")):
            tb = block.get(tau_key)
            if not tb:
                continue
            bs = tb.get("bootstrap", {})
            entry[tau] = {
                "delta_U_common_support": tb.get("delta_U_common"),
                "n_common_support": tb.get("n_common") or n_common,
                "bootstrap_positive_fraction": bs.get("prob_positive"),
                "bootstrap_ci95_percentile": bs.get("ci95"),
                "per_fold_delta_U": tb.get("fold_deltas"),
            }
        summary[label] = entry

    checks = {
        "development_folds_0_2": {
            "n_total": g84["n_gates_total"],
            "n_passed": g84["n_gates_passed"],
            "failed": g84["gates"]["failed"],
            "checks": g84["gates"]["checks"],
        },
        "policy_selection_holdout_folds_3_4": {
            "n_total": g85["confirmation_gates"]["n_total"],
            "n_passed": g85["confirmation_gates"]["n_passed"],
            "failed": g85["confirmation_gates"]["failed"],
            "checks": g85["confirmation_gates"]["checks"],
        },
        "pooled_all_folds_supportive_only": {
            "n_total": g85["pooled_gates"]["n_total"],
            "n_passed": g85["pooled_gates"]["n_passed"],
            "failed": g85["pooled_gates"]["failed"],
            "checks": g85["pooled_gates"]["checks"],
        },
    }

    ln = g85["confirmation"]["lesion_noninferiority"]
    lesion = {
        "note": (
            "Reference-lesion miss-rate noninferiority on the policy-selection holdout. "
            "Margins are operational, chosen to be strict relative to observed between-fold "
            "variation; they are not externally validated clinical thresholds."
        ),
        "n_reference_components_baseline_candidate": ln["n_ref_total"],
        "missed_reference_components_baseline_candidate": ln["fn_ref_total"],
        "predicted_false_positive_components_baseline_candidate": ln["fp_pred_total"],
        "miss_rate_baseline_candidate": ln["miss_rate"],
        "miss_rate_point_delta": ln["bootstrap"]["point_delta"],
        "miss_rate_one_sided_95_upper": ln["bootstrap"]["upper_95_one_sided"],
        "per_region_miss_rate_delta": ln["region_deltas"],
        "margins": ln["margins"],
        "checks": ln["checks"],
        "passes": ln["passes"],
    }

    official = {
        "note": (
            "Organizer-reported official validation values for the frozen released ensemble, "
            "from a single submission, exactly as returned. DSC and NSD are the ranked metrics; "
            "HD95 is diagnostic only. The organizers did not expose the NSD surface tolerance, "
            "so none is labelled. No rank was exposed and none is inferred. 451 predictions were "
            "submitted; the participant-visible per-case file contained 450 scored rows, a "
            "discrepancy whose cause was not disclosed and which is attributed to neither party."
        ),
        "n_predictions_submitted": 451,
        "n_scored_rows_in_participant_visible_file": 450,
        "finite_per_case_denominator": {"ET": 440, "TC": 449, "WT": 450},
        "DSC": {
            "ET": 0.772185975381129,
            "TC": 0.823529052186837,
            "WT": 0.879421230596971,
        },
        "NSD": {
            "ET": 0.539788495869978,
            "TC": 0.499919028770286,
            "WT": 0.483422275392118,
        },
        "HD95_mm_diagnostic_only": {"ET": 44.82, "TC": 23.64, "WT": 17.56},
    }
    # Cross-check the hard-coded official values against the committed artifact.
    for metric in ("DSC", "NSD"):
        for region, value in official[metric].items():
            key = f"global_{metric.lower()}_{region.lower()}"
            recorded = _find_scalar(g80, key)
            if recorded is not None and abs(recorded - value) > 1e-12:
                raise SystemExit(f"official value mismatch for {key}: {recorded} != {value}")

    return {
        "components": components,
        "summary": summary,
        "checks": checks,
        "lesion": lesion,
        "official": official,
    }


def _find_scalar(obj, key):
    """Depth-first search for the first scalar stored under `key`."""
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], (int, float)):
            return obj[key]
        for value in obj.values():
            found = _find_scalar(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_scalar(value, key)
            if found is not None:
                return found
    return None


def _assert_clean(text: str, where: str) -> None:
    lowered = text.lower()
    for bad in FORBIDDEN_SUBSTRINGS:
        if bad.lower() in lowered:
            raise SystemExit(f"restricted string {bad!r} would be published in {where}")
    for pattern in FORBIDDEN_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            raise SystemExit(
                f"restricted identifier shape {pattern!r} matched in {where} "
                f"at offset {match.start()}"
            )


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    priv, pub = Path(sys.argv[1]), Path(sys.argv[2])
    built = build(priv)

    out = pub / "evidence"
    out.mkdir(parents=True, exist_ok=True)

    comp_path = out / "policy_audit_components.csv"
    with comp_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(built["components"][0]))
        writer.writeheader()
        writer.writerows(built["components"])

    payloads = {
        "policy_audit_summary.json": {
            "note": (
                "Aggregate policy-audit evidence for the mirroring candidate M8 against the "
                "released baseline C0. Development = folds 0-2; the policy-selection holdout = "
                "folds 3-4. Comparisons are on common support: a region-case contributes only "
                "if both policies yield a finite value. Uncertainty is a paired subject-level "
                "bootstrap, seed 20260730, 10,000 resamples, percentile intervals."
            ),
            "utility": (
                "U(P) is the unweighted mean of six fractional ranks over ET/TC/WT x DSC/NSD. "
                "HD95 never enters U. Ties resolve to the baseline."
            ),
            "subsets": built["summary"],
        },
        "policy_audit_checks.json": {
            "note": (
                "Complete decision-check matrices. The development subset carried 18 frozen "
                "checks and the policy-selection holdout 23. A candidate whose expected "
                "evaluation is missing, errored or membership-mismatched is ineligible: the "
                "audit fails closed rather than comparing a surviving subset."
            ),
            "matrices": built["checks"],
        },
        "lesion_noninferiority.json": built["lesion"],
        "official_validation_scores.json": built["official"],
    }
    for name, payload in payloads.items():
        text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
        _assert_clean(text, name)
        (out / name).write_text(text)

    _assert_clean(comp_path.read_text(), comp_path.name)

    print(f"wrote {len(payloads) + 1} evidence files to {out}")
    print(f"  component rows: {len(built['components'])}")
    for key, matrix in built["checks"].items():
        print(f"  {key}: {matrix['n_passed']}/{matrix['n_total']} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
