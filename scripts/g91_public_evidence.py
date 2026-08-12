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
# The leading (?<![0-9.]) on each pattern excludes NUMERIC context. Both rules are identifier-shape
# heuristics, and a JSON float is not an identifier: the mantissa of 8.881416674766385e-05 contains
# the 16-character run "881416674766385e", which the digest rule would otherwise flag, and a value
# like 0.1234567 would trip the submission-id rule. An actual digest or id in this output is always
# preceded by a quote, colon, slash or space, never by a digit or a decimal point.
FORBIDDEN_PATTERNS = (
    r"(?<![0-9.])\b\d{7}\b",                            # challenge submission ids
    r"(?<![0-9.])\b(?=[0-9a-f]*[a-f])[0-9a-f]{12,}\b",   # image digests / manifest hashes
)


def _load(p: Path) -> dict:
    with p.open(encoding="utf-8") as fh:
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


# --- r11: sanitized aggregate inputs for the supplement generator ------------------------------
# scripts/g92_build_supplement.py reads artifacts/g84_result.json and artifacts/g85_result.json, and
# artifacts/ is never exported. Without a public input the documented regeneration command cannot run
# outside the private tree. This projection publishes EXACTLY the aggregate fields the generator
# reads -- component means and deltas, common-support sizes, bootstrap summaries, per-fold deltas,
# gate matrices and lesion COUNTS -- at full float precision, so a public run reproduces the
# committed supplement byte-for-byte. It publishes no per-case value, no case identifier, no fold
# membership, no prediction and no path. The whitelist below is explicit: a key that is not named
# here is not published.

_TAU_FIELDS = ("delta_U_common", "component_deltas", "baseline_means", "candidate_means",
               "n_common", "bootstrap", "fold_deltas")


def _tau_block(block: dict | None) -> dict | None:
    """Project one tolerance block down to the fields the supplement generator reads."""
    if not block:
        return None
    out = {k: block[k] for k in _TAU_FIELDS if k in block}
    bs = out.get("bootstrap")
    if isinstance(bs, dict):
        out["bootstrap"] = {k: bs[k] for k in ("prob_positive", "ci95") if k in bs}
    return out


AB_POLICIES = ("C0", "C1", "C2", "C3", "C0_et10", "C0_et25", "C0_et50", "S1", "S2")
AB_DECISION_METRICS = ("et_dsc", "et_hd95", "tc_dsc", "tc_hd95", "wt_dsc", "wt_hd95")
SCREEN_ARMS = ("T", "DG", "TDG")


def _audit_ab(g75: dict, g76: dict, g77: dict, g79v: dict) -> dict:
    """Per-component evidence for every executed Audit A / Audit B candidate.

    Two distinct measurement families are published side by side and must not be merged:

      * the numbers the audits DECIDED on -- ET/TC/WT DSC and HD95 from the G7.5 / G7.6 records,
        aggregated over all n=811 development subjects;
      * the official-metric re-scoring -- ET/TC/WT DSC and NSD at tau=1 and tau=0.5 from the later
        G7.7 / G79-V records, computed post hoc from the same preserved predictions under the
        evaluator's skipna rule, so each policy carries its OWN denominators.

    The two families use different aggregation rules and therefore disagree in the third decimal;
    they are never combined into one column.
    """
    dec = {}
    for name in AB_POLICIES:
        if name in ("S1", "S2"):
            src = g76["candidates"][name]
        elif name == "C0":
            src = g75["calibration_aggregates"]["C0"]
        else:
            src = g75["calibration_aggregates"][name]
        dec[name] = {k: src[k] for k in AB_DECISION_METRICS if k in src}

    # SIGN CONVENTION -- the two orientations in the frozen record, kept apart on purpose.
    #
    #   rank_gain_over_C0 = rank(C0) - rank(candidate)      positive favours the CANDIDATE
    #   bootstrap_ci      = rank(candidate) - rank(C0)      the OPPOSITE orientation
    #
    # The frozen artifact stores the point estimate in the first orientation and the bootstrap
    # interval in the second. Displaying them side by side without saying so reads as though a
    # negative point estimate came with a positive interval. We therefore derive, and publish under
    # its own name, the interval in the SAME orientation as the point estimate:
    #
    #   rank_gain_ci = [-hi, -lo]   from   bootstrap_ci = [lo, hi]
    #
    # Negating an interval reverses its endpoints, which is why hi and lo swap. The raw interval is
    # retained too, under a name that states its orientation, so nothing is hidden and no frozen
    # value is edited. This is a presentation transform only: no experiment is recomputed and no
    # advancement outcome changes.
    stats = {}
    for name in AB_POLICIES:
        st = g79v["tau10_statistics"].get(name)
        if st is None:
            continue
        lo, hi = st["bootstrap_ci"]
        stats[name] = {
            "rank_gain_over_C0": st["rank_gain_over_C0"],
            "rank_gain_ci": [-hi, -lo],
            "bootstrap_delta_ci_candidate_minus_C0": [lo, hi],
            "advances": st["advances"],
        }

    return {
        "note": (
            "Audit A (G7.5 inference-policy) and Audit B (G7.6 weight-averaging) candidates. "
            "DECISION metrics are DSC and HD95 over all 811 development subjects, the pair the "
            "audits were actually ranked on. OFFICIAL metrics are DSC and NSD re-scored post hoc "
            "from the same preserved predictions; each policy carries its own skipna denominators, "
            "so these are NOT on the paired common support used for Audit C and are not "
            "line-comparable with it. All values come from the end-of-training validation lineage, "
            "which applies eightfold mirroring, not from the release path."
        ),
        "subset": {
            "role": "development subset, folds 0-2",
            "n_subjects": g75["design"]["calibration_n"],
            "folds": g75["design"]["calibration_folds"],
            "evaluator": g75["design"]["evaluator"],
            "confirmation_folds_opened": bool(g75["design"].get("confirmation_run")),
        },
        "definitions": dict(g75["candidates"], **g76["design"]["candidate_definitions"]),
        "decision_metrics_dsc_hd95": dec,
        "official_metrics": {
            "tau_1.0": {n: {"aggregates": g79v["tau10"][n]["aggregates"],
                            "denominators": g79v["tau10"][n]["denominators"]}
                        for n in AB_POLICIES},
            "tau_0.5": {n: {"aggregates": g79v["tau05"][n]["aggregates"],
                            "denominators": g79v["tau05"][n]["denominators"]}
                        for n in AB_POLICIES},
        },
        "statistics_tau_1.0": stats,
        "statistics_sign_convention": {
            "rank_gain_over_C0": "rank(C0) - rank(candidate); POSITIVE favours the candidate",
            "rank_gain_ci": ("the paired subject-level percentile interval for rank_gain_over_C0, "
                             "in the SAME orientation as the point estimate. Derived as [-hi, -lo] "
                             "from the frozen bootstrap interval; negating an interval reverses its "
                             "endpoints."),
            "bootstrap_delta_ci_candidate_minus_C0": ("the frozen interval exactly as stored, in "
                                                      "the OPPOSITE orientation: rank(candidate) - "
                                                      "rank(C0). Retained unmodified for audit."),
            "note": ("A presentation transform only. No frozen artifact was edited, no experiment "
                     "recomputed, and no advancement outcome, threshold, candidate decision or "
                     "release decision changed: every candidate still fails to advance."),
        },
        "outcomes": {
            "g75": g75["decision"], "g76": g76["decision"],
            "g77": g77.get("outcome"), "g79v": g79v["outcome"],
            "advancing_candidates": g79v["advancing_candidates"],
            "release_policy_changed": g79v["release_policy_changed"],
        },
    }


def _screen_g82(g82: dict, prereg: dict) -> dict:
    """The frozen 40-epoch T / DG / TDG fine-tuning screen.

    Only DELTAS versus C0 were recorded for the candidate arms; absolute per-component candidate
    means were never computed, and HD95 was excluded by the preregistration. Both facts are stated
    rather than worked around.
    """
    arms = {}
    for arm in SCREEN_ARMS:
        r = g82["recipes"][arm]
        d = g82["screen_decision"]["recipes"][arm]
        arms[arm] = {
            "definition": prereg["candidate_definitions"][arm],
            "component_deltas_tau_1.0": r["tau1"]["component_deltas"],
            "component_deltas_tau_0.5": r["tau05"]["component_deltas"],
            "delta_U_common_tau_1.0": r["tau1"]["delta_U_common"],
            "delta_U_common_tau_0.5": r["tau05"]["delta_U_common"],
            "delta_U_official_tau_1.0": r["tau1"]["delta_U_official"],
            "delta_U_official_tau_0.5": r["tau05"]["delta_U_official"],
            "bootstrap_tau_1.0": r["tau1"]["bootstrap"],
            "fold_deltas_tau_1.0": r["tau1"]["fold_deltas"],
            "zero_dsc_tau_1.0": r["tau1"]["zero_dsc"],
            "lesion": r["lesion"],
            "gates": d["checks"],
            "eligible": d["eligible"],
            "score": d["score"],
        }
    return {
        "note": (
            "Frozen 40-epoch fine-tuning screen. Only deltas versus the C0 baseline were recorded "
            "for the candidate arms: absolute per-component candidate means do not exist in the "
            "frozen record and are not reconstructed here. HD95 was excluded by preregistration "
            "and no HD95 value exists for any arm. Bootstrap, per-fold deltas and tail statistics "
            "exist at tau=1 only; the tau=0.5 block carries deltas alone."
        ),
        "design": {
            "epochs": prereg["screen"]["epochs"],
            "finetune_lr": prereg["initialization"]["finetune_lr"],
            "seed": prereg["screen"]["seed"],
            "folds": g82["discovery_folds"],
            "n_common_support": g82["n_common_support"],
            "selection_rule": prereg["screen"]["selection"],
            "confirmation_folds_opened": g82["confirmation_folds_opened"],
        },
        "baseline_U_common": {"tau_1.0": g82["recipes"]["T"]["tau1"]["U_common_baseline"],
                              "tau_0.5": g82["recipes"]["T"]["tau05"]["U_common_baseline"]},
        "arms": arms,
        "terminal_status": g82["terminal_status"],
        "convergence_caveat": (
            "The screen tested exactly one point in the fine-tuning design space -- 40 epochs at "
            "5\\% of the original learning rate, from a converged model, on three folds. A bounded "
            "run of that size cannot speak to convergence, and it does not establish that "
            "fine-tuning cannot help: a longer schedule, a higher learning rate or a different "
            "intervention was never tested, because the preregistration made the 40-epoch screen "
            "the gate for spending more compute and the screen did not pass."
        ),
    }


def _d25(g83: dict) -> dict:
    """D25: preregistered, never executed, zero predictions."""
    return {
        "executed": False,
        "predictions_generated": g83["d25_predictions_generated"],
        "failed_gate": g83["failed_gate"],
        "root_cause": g83["root_cause"],
        "conflict": g83["conflict"],
        "tile_multiplier_analysis": g83["tile_multiplier_analysis"],
        "note": (
            "D25 differed from C0 in exactly one parameter, the sliding-window tile step "
            "($0.5 \\rightarrow 0.25$). It was never run. The prerequisite baseline-reproduction gate failed "
            "because the twelve required baseline values are reproducible only by the eightfold "
            "mirroring lineage, while the specification defines C0 with mirroring disabled and "
            "prohibits test-time augmentation, so the targets cannot be reproduced by any run that "
            "complies with the specification. No D25 prediction, DSC, NSD, runtime or utility "
            "value exists. This is a specification/lineage mismatch, not evidence that D25 failed; "
            "D25 remains scientifically unresolved. An analytic tile-count study was nevertheless "
            "computed, without generating any prediction."
        ),
    }


def _provenance(ab: dict, screen: dict) -> dict:
    """Machine-readable map from every rendered supplement value to its frozen source field.

    Each entry names the private record and the exact key path the value was read from, and the
    public projection field that carries it. It exists so a reviewer can trace any number in the
    supplement back to an immutable artifact without access to the private tree.
    """
    def rows(policies, source, keypath, fields, public):
        return [{"policies": list(policies), "source_record": source,
                 "source_key_path": keypath, "fields": list(fields),
                 "public_projection_field": public} for _ in (0,)]

    return {
        "note": (
            "Provenance for every value rendered in the supplement's Audit A/B, bounded-screen and "
            "D25 material. 'source_record' is the frozen private artifact; 'source_key_path' is the "
            "exact path within it; 'public_projection_field' is where the same value appears in "
            "evidence/supplement_inputs.json, which is what a public regeneration reads. Every "
            "value is read directly from its source record EXCEPT two categories, both of which "
            "are explicitly marked derived: true in the entries below. (1) The candidate-minus-C0 "
            "component deltas, which are differences of two published means. (2) The displayed "
            "rank-gain intervals, derived as rank_gain_ci = [-raw_high, -raw_low] from the frozen "
            "opposite-orientation bootstrap interval, so that the interval carries the same "
            "orientation as the point estimate it accompanies. Nothing else is computed here: no "
            "experiment is re-run and no frozen record is edited."
        ),
        "entries": [
            {"what": "Audit A/B official-metric component means, tau=1",
             "policies": list(AB_POLICIES),
             "source_record": "artifacts/g79v_tau1_sensitivity_results.json",
             "source_key_path": "tau10.<policy>.aggregates.{DSC_ET,DSC_TC,DSC_WT,NSD_ET,NSD_TC,NSD_WT}",
             "public_projection_field": "audit_ab.official_metrics.tau_1.0.<policy>.aggregates",
             "derived": False},
            {"what": "Audit A/B official-metric component means, tau=0.5",
             "policies": list(AB_POLICIES),
             "source_record": "artifacts/g79v_tau1_sensitivity_results.json",
             "source_key_path": "tau05.<policy>.aggregates.{six components}",
             "public_projection_field": "audit_ab.official_metrics.tau_0.5.<policy>.aggregates",
             "derived": False},
            {"what": "Audit A/B candidate-minus-C0 component deltas, both tolerances",
             "policies": [p for p in AB_POLICIES if p != "C0"],
             "source_record": "artifacts/g79v_tau1_sensitivity_results.json",
             "source_key_path": "tau{05,10}.<policy>.aggregates MINUS tau{05,10}.C0.aggregates",
             "public_projection_field": "audit_ab.official_metrics.<tau>.<policy>.aggregates",
             "derived": True,
             "derivation": "candidate mean minus baseline C0 mean, component-wise"},
            {"what": "Audit A/B per-policy denominators",
             "policies": list(AB_POLICIES),
             "source_record": "artifacts/g79v_tau1_sensitivity_results.json",
             "source_key_path": "tau10.<policy>.denominators.{DSC_ET,DSC_TC,DSC_WT}",
             "public_projection_field": "audit_ab.official_metrics.tau_1.0.<policy>.denominators",
             "derived": False},
            {"what": "Audit A decision metrics (DSC, HD95)",
             "policies": ["C0", "C1", "C2", "C3", "C0_et10", "C0_et25", "C0_et50"],
             "source_record": "artifacts/g75_inference_policy_decision.json",
             "source_key_path": "calibration_aggregates.<policy>.{et,tc,wt}_{dsc,hd95}",
             "public_projection_field": "audit_ab.decision_metrics_dsc_hd95.<policy>",
             "derived": False},
            {"what": "Audit B decision metrics (DSC, HD95) for the two soups",
             "policies": ["S1", "S2"],
             "source_record": "artifacts/g76_checkpoint_soup_decision.json",
             "source_key_path": "candidates.<policy>.{et,tc,wt}_{dsc,hd95}",
             "public_projection_field": "audit_ab.decision_metrics_dsc_hd95.<policy>",
             "derived": False},
            {"what": ("Rank gain point estimate, the frozen opposite-orientation bootstrap "
                      "interval, and advancement status, at tau=1"),
             "policies": [p for p in AB_POLICIES if p != "C0"],
             "source_record": "artifacts/g79v_tau1_sensitivity_results.json",
             "source_key_path": "tau10_statistics.<policy>.{rank_gain_over_C0,bootstrap_ci,advances}",
             "public_projection_field": ("audit_ab.statistics_tau_1.0.<policy>."
                                         "{rank_gain_over_C0,"
                                         "bootstrap_delta_ci_candidate_minus_C0,advances}"),
             "orientation": {
                 "rank_gain_over_C0": "R(C0) - R(candidate); positive favours the candidate",
                 "bootstrap_delta_ci_candidate_minus_C0":
                     "R(candidate) - R(C0); the OPPOSITE orientation, exactly as frozen",
             },
             "derived": False},
            {"what": ("The interval DISPLAYED in supplement Table S18, converted to the point "
                      "estimate's orientation"),
             "policies": [p for p in AB_POLICIES if p != "C0"],
             "source_record": "artifacts/g79v_tau1_sensitivity_results.json",
             "source_key_path": "tau10_statistics.<policy>.bootstrap_ci",
             "public_projection_field": "audit_ab.statistics_tau_1.0.<policy>.rank_gain_ci",
             "derived": True,
             "derivation": "rank_gain_ci = [-raw_high, -raw_low]",
             "derivation_note": ("The frozen interval is stored as R(candidate) - R(C0), the "
                                 "opposite orientation to the point estimate. Negating it puts it "
                                 "in the same orientation, and negating an interval reverses its "
                                 "endpoints, which is why high and low swap. No experiment is "
                                 "recomputed and the frozen record is not edited; the unconverted "
                                 "interval remains published beside it.")},
            {"what": "Subset role, n, folds and evaluator",
             "policies": list(AB_POLICIES),
             "source_record": "artifacts/g75_inference_policy_decision.json",
             "source_key_path": "design.{calibration_n,calibration_folds,evaluator,confirmation_run}",
             "public_projection_field": "audit_ab.subset",
             "derived": False},
            {"what": "Audit outcomes",
             "policies": list(AB_POLICIES),
             "source_record": ("artifacts/g75_inference_policy_decision.json, "
                               "artifacts/g76_checkpoint_soup_decision.json, "
                               "artifacts/g77_official_metric_decision.json, "
                               "artifacts/g79v_tau1_sensitivity_results.json"),
             "source_key_path": "decision | decision | outcome | outcome",
             "public_projection_field": "audit_ab.outcomes",
             "derived": False},
            {"what": "Bounded 40-epoch screen component deltas, both tolerances",
             "policies": list(SCREEN_ARMS),
             "source_record": "artifacts/g82_result.json",
             "source_key_path": "recipes.<arm>.{tau1,tau05}.component_deltas.{six components}",
             "public_projection_field": "screen_g82.arms.<arm>.component_deltas_<tau>",
             "derived": False},
            {"what": "Bounded screen aggregate utility change and gate tallies",
             "policies": list(SCREEN_ARMS),
             "source_record": "artifacts/g82_result.json",
             "source_key_path": ("recipes.<arm>.{tau1,tau05}.delta_U_common; "
                                 "screen_decision.recipes.<arm>.checks"),
             "public_projection_field": ("screen_g82.arms.<arm>.delta_U_common_<tau>; "
                                         "screen_g82.arms.<arm>.gates"),
             "derived": False},
            {"what": "Bounded screen bootstrap point estimates and positive fractions (tau=1 only)",
             "policies": list(SCREEN_ARMS),
             "source_record": "artifacts/g82_result.json",
             "source_key_path": "recipes.<arm>.tau1.bootstrap.{point,prob_positive,ci95}",
             "public_projection_field": "screen_g82.arms.<arm>.bootstrap_tau_1.0",
             "derived": False},
            {"what": "Bounded screen design, terminal status and convergence caveat",
             "policies": list(SCREEN_ARMS),
             "source_record": ("artifacts/g82_result.json, configs/g82_preregistration.json, "
                               "artifacts/G82_GENERALIZATION_FINETUNE.md"),
             "source_key_path": ("terminal_status; screen.{epochs,seed,selection}; "
                                 "initialization.finetune_lr; section 10 limitation"),
             "public_projection_field": "screen_g82.{design,terminal_status,convergence_caveat}",
             "derived": False},
            {"what": "D25 non-execution, cause and analytic tile-count study",
             "policies": ["D25"],
             "source_record": "artifacts/g83_result.json",
             "source_key_path": ("d25_predictions_generated, failed_gate, root_cause, conflict, "
                                 "tile_multiplier_analysis"),
             "public_projection_field": "d25",
             "derived": False},
        ],
        "values_that_do_not_exist": [
            {"what": "Absolute per-component means for the T, DG and TDG arms",
             "why": ("artifacts/g82_result.json records component_deltas against C0 only; candidate "
                     "absolutes were never computed. They are reported as deltas and nothing is "
                     "reconstructed.")},
            {"what": "Any HD95 value for T, DG or TDG",
             "why": ("configs/g82_preregistration.json metric.official_ranking_uses excludes HD95; "
                     "no HD95 was computed for any screen arm.")},
            {"what": "Bootstrap, per-fold, zero-DSC and tail statistics at tau=0.5 for the screen",
             "why": "recipes.<arm>.tau05 carries deltas only; those statistics exist at tau=1 only."},
            {"what": "Any D25 measurement",
             "why": ("d25_predictions_generated = 0. D25 was never executed, so no DSC, NSD, "
                     "runtime or utility value exists for it.")},
            {"what": "Folds 3-4 values for any Audit A/B candidate or screen arm",
             "why": ("the policy-selection holdout was never opened for these audits "
                     "(confirmation_run false; confirmation_folds_opened false).")},
            {"what": "Common-support (paired) figures for Audit A/B candidates",
             "why": ("the frozen records aggregate under the evaluator's per-policy skipna rule; "
                     "no paired common-support aggregation was computed for these candidates and "
                     "none is invented here.")},
        ],
    }


def _supplement_inputs(g84: dict, g85: dict, extra: dict) -> dict:
    """Whitelisted aggregate projection of the frozen private audit records."""
    conf = g85["confirmation"]
    lesion_ni = conf["lesion_noninferiority"]
    diag = conf["lesion_diagnostic_only"]
    return {
        "note": (
            "Aggregate inputs for scripts/g92_build_supplement.py, projected from the frozen "
            "private audit records artifacts/g84_result.json and artifacts/g85_result.json. Only "
            "the fields the generator reads are published, at full float precision, so that a "
            "public regeneration reproduces paper/supplement.tex byte-for-byte. No per-case value, "
            "case identifier, fold membership, prediction or path is included. Lesion figures are "
            "component COUNTS over the whole subset, not per-case data."
        ),
        "g84": {
            "common_support": g84["common_support"],
            "n_gates_total": g84["n_gates_total"],
            "n_gates_passed": g84["n_gates_passed"],
            "gates": {"checks": g84["gates"]["checks"]},
            "calibration": {
                "tau1": _tau_block(g84["calibration"].get("tau1")),
                "tau05": _tau_block(g84["calibration"].get("tau05")),
            },
            "lesion": {
                "baseline": {k: g84["lesion"]["baseline"][k] for k in ("TP", "FP", "FN")},
                "candidate": {k: g84["lesion"]["candidate"][k] for k in ("TP", "FP", "FN")},
            },
        },
        "g85": {
            "confirmation": {
                "t10": _tau_block(conf.get("t10")),
                "t05": _tau_block(conf.get("t05")),
                "lesion_noninferiority": {
                    "n_ref_total": lesion_ni["n_ref_total"],
                    "miss_rate": lesion_ni["miss_rate"],
                    "region_deltas": lesion_ni["region_deltas"],
                    "margins": lesion_ni["margins"],
                    "bootstrap": {k: lesion_ni["bootstrap"][k] for k in
                                  ("point_delta", "upper_95_one_sided", "n_subjects",
                                   "seed", "resamples")},
                },
                "lesion_diagnostic_only": {
                    side: {k: diag[side][k] for k in
                           ("tp_pred_diagnostic", "fp_pred", "fn_ref")}
                    for side in ("baseline", "candidate")
                },
            },
            "pooled_all_five_folds": {
                "t10": _tau_block(g85["pooled_all_five_folds"].get("t10")),
                "t05": _tau_block(g85["pooled_all_five_folds"].get("t05")),
            },
            "confirmation_gates": {k: g85["confirmation_gates"][k]
                                   for k in ("checks", "n_passed", "n_total")},
            "pooled_gates": {k: g85["pooled_gates"][k] for k in ("checks", "n_passed", "n_total")},
        },
        "audit_ab": extra["audit_ab"],
        "screen_g82": extra["screen_g82"],
        "d25": extra["d25"],
        "provenance": extra["provenance"],
    }


def build(priv: Path) -> dict[str, object]:
    g84 = _load(priv / "artifacts" / "g84_result.json")
    g85 = _load(priv / "artifacts" / "g85_result.json")
    g80 = _load(priv / "artifacts" / "g80_official_validation_result.json")
    # r11: reviewer-requested per-component evidence for the earlier audits and the bounded screen.
    g75 = _load(priv / "artifacts" / "g75_inference_policy_decision.json")
    g76 = _load(priv / "artifacts" / "g76_checkpoint_soup_decision.json")
    g77 = _load(priv / "artifacts" / "g77_official_metric_decision.json")
    g79v = _load(priv / "artifacts" / "g79v_tau1_sensitivity_results.json")
    g82 = _load(priv / "artifacts" / "g82_result.json")
    g82p = _load(priv / "configs" / "g82_preregistration.json")
    g83 = _load(priv / "artifacts" / "g83_result.json")
    extra = {
        "audit_ab": _audit_ab(g75, g76, g77, g79v),
        "screen_g82": _screen_g82(g82, g82p),
        "d25": _d25(g83),
    }
    extra["provenance"] = _provenance(extra["audit_ab"], extra["screen_g82"])

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
            reported = tb.get("delta_U_common")
            deltas = tb.get("component_deltas", {})
            recomputed = sum(deltas[c] for c in COMPONENTS) / 6
            if abs(recomputed - reported) > 1e-12:
                raise SystemExit(
                    f"utility arithmetic mismatch in {label}/{tau_key}: reported "
                    f"{reported!r} but the mean of the six component deltas is {recomputed!r}"
                )
            entry[tau] = {
                "role": ("official-ranking-aligned" if tau == "tau_1.0"
                         else "sensitivity analysis (Panoptica default tolerance)"),
                "delta_U_common_support": reported,
                "delta_U_recomputed_as_mean_of_component_deltas": recomputed,
                "n_common_support": tb.get("n_common") or n_common,
                # Absent where the committed record holds no bootstrap for this tolerance.
                # Never inferred or back-filled: null means "not computed", not "zero".
                "bootstrap_positive_fraction": bs.get("prob_positive"),
                "bootstrap_ci95_percentile": bs.get("ci95"),
                "bootstrap_available": bool(bs),
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
            "HD95 is diagnostic only. The organizers confirmed that the FINAL RANKING uses NSD "
            "at tau=1, but did not disclose which tolerance generated these particular "
            "participant-visible validation values, so no tolerance is attached to them here. "
            "No rank was exposed and none is inferred. 451 predictions were submitted; the "
            "participant-visible per-case file contained 450 scored rows, a discrepancy whose "
            "cause was not disclosed and which is attributed to neither party."
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
        "supplement_inputs": _supplement_inputs(g84, g85, extra),
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
    with comp_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(built["components"][0]))
        writer.writeheader()
        writer.writerows(built["components"])

    payloads = {
        "policy_audit_summary.json": {
            "note": (
                "Aggregate policy-audit evidence for the mirroring candidate M8 against the "
                "released baseline C0. Development = folds 0-2; the policy-selection holdout = "
                "folds 3-4. Comparisons are on common subject support: a subject contributes "
                "only if all six required components are finite under both policies. "
                "Uncertainty is a paired subject-level bootstrap, seed 20260730, 10,000 "
                "resamples, percentile intervals. Where a bootstrap is absent from the "
                "committed record it is reported as null and is never inferred."
            ),
            "utility": (
                "U_tau(P; S_tau) is the unweighted arithmetic mean of the six RAW component "
                "means -- ET/TC/WT x DSC/NSD(tau) -- over the common subject set S_tau. "
                "delta U_tau = U_tau(candidate) - U_tau(baseline), which is identically the "
                "arithmetic mean of the six component deltas. It is NOT a rank statistic: no "
                "ranking, fractional ranking or tie-breaking enters it. HD95 never enters "
                "U_tau. The separate fold-0 architecture screen uses a distinct fractional-rank "
                "statistic R; R and U_tau are different objects and are never combined."
            ),
            "tolerances": (
                "The organizers confirmed that the final challenge ranking uses DSC and NSD "
                "with NSD at tau=1, and that HD95 is diagnostic only. tau=1 is therefore the "
                "official-ranking-aligned analysis. tau=0.5 is Panoptica's default tolerance "
                "and is reported as a prespecified sensitivity analysis, not as an equally "
                "official quantity."
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
        "supplement_inputs.json": built["supplement_inputs"],
    }
    for name, payload in payloads.items():
        text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
        _assert_clean(text, name)
        (out / name).write_text(text, encoding="utf-8")

    _assert_clean(comp_path.read_text(encoding="utf-8"), comp_path.name)

    print(f"wrote {len(payloads) + 1} evidence files to {out}")
    print(f"  component rows: {len(built['components'])}")
    for key, matrix in built["checks"].items():
        print(f"  {key}: {matrix['n_passed']}/{matrix['n_total']} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
