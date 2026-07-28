#!/usr/bin/env python3
"""GAT-26 G7.7 official-metric alignment regression tests
(`python3 tests/test_g77_official_metric.py`).

Locks the preregistered protocol against the failure modes the stage prompt names:
  * candidate-list expansion,
  * fold leakage between calibration and confirmation,
  * premature confirmation access,
  * metric substitution (HD95 must never re-enter the selection utility),
  * post-result alteration of the frozen configuration,
plus the official-metric definition itself (six higher-is-better DSC/NSD components, skipna
aggregation, fail-closed validation, and the baseline-wins-ties rule).

Pure stdlib + numpy; no evaluator, GPU, or protected data required.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import g77_official_metric as G  # noqa: E402

CFG_PATH = REPO / "configs" / "g77_official_metric_alignment.json"
FAILS = 0

# --- private governance state ----------------------------------------------------------------
# RUN_STATE.json is the live internal operational record (resource identifiers, cost accounting,
# blockers, persistence audit). It is deliberately NOT redistributed in the public source export.
PRIVATE_GOVERNANCE = ("RUN_STATE.json",)
SKIP_MESSAGE = "SKIP_PUBLIC_EXPORT_PRIVATE_GOVERNANCE"


def public_export_mode(required_private) -> bool:
    """True ONLY inside a validated sanitized public export. Fail-closed by construction.

    Every condition must hold:
      * a well-formed root ``EXPORT_MANIFEST.json`` exists and declares the Apache-2.0 sanitized
        export;
      * that manifest actually describes THIS tree — it lists this test file and ``LICENSE``;
      * every ``required_private`` path is genuinely absent.

    Deleting files inside the development repository can NEVER activate this: the private
    repository does not contain, and never commits, ``EXPORT_MANIFEST.json`` — the manifest is
    produced only by ``scripts/make_code_export.py`` into an export directory. So missing
    governance stays a hard failure everywhere except a real export.
    """
    try:
        man = json.loads((REPO / "EXPORT_MANIFEST.json").read_text())
    except (OSError, ValueError):
        return False
    if man.get("declared_license") != "Apache-2.0":
        return False
    listed = {e.get("path") for e in man.get("files", []) if isinstance(e, dict)}
    if not listed or "LICENSE" not in listed:
        return False
    if Path(__file__).resolve().relative_to(REPO).as_posix() not in listed:
        return False
    return all(not (REPO / p).exists() for p in required_private)


def check(name, cond):
    global FAILS
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILS += 1


def rec(sid, dsc, nsd):
    """One subject with the six official components (dsc/nsd are 3-tuples ET,TC,WT)."""
    m = {}
    for i, r in enumerate(G.REGIONS):
        m[(r, "DSC")] = dsc[i]
        m[(r, "NSD")] = nsd[i]
    return {"subject": sid, "metrics": m}


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except G.HardFailure:
        return True
    except Exception:
        return False
    return False


def main():
    print("test_g77_official_metric:")
    cfg = json.loads(CFG_PATH.read_text())

    # ---- 1. official metric definition ------------------------------------------------
    check("six_official_components", len(G.OFFICIAL_COMPONENTS) == 6)
    check("components_are_dsc_and_nsd_only",
          {m for _, m in G.OFFICIAL_COMPONENTS} == {"DSC", "NSD"})
    check("regions_et_tc_wt", {r for r, _ in G.OFFICIAL_COMPONENTS} == {"ET", "TC", "WT"})
    check("all_higher_is_better",
          all(v == "higher_is_better" for v in G.DIRECTION.values()))
    check("field_names_match_leaderboard_columns",
          set(G.OFFICIAL_FIELDS.values()) ==
          {"global_dsc_et", "global_dsc_tc", "global_dsc_wt",
           "global_nsd_et", "global_nsd_tc", "global_nsd_wt"})

    # ---- 2. METRIC SUBSTITUTION: HD95 must never enter the selection utility -----------
    src = (REPO / "scripts" / "g77_official_metric.py").read_text()
    check("hd95_not_a_component", not any(m == "HD95" for _, m in G.OFFICIAL_COMPONENTS))
    check("hd95_absent_from_direction_table", "HD95" not in G.DIRECTION)
    check("no_hd95_field_in_official_fields",
          not any("hd95" in f for f in G.OFFICIAL_FIELDS.values()))
    check("module_declares_hd95_excluded", "not** part of this utility" in src)
    check("config_marks_hd95_diagnostic_only",
          "SECONDARY_DIAGNOSTIC_ONLY" in cfg["official_metric_definition"]["hd95_role"])
    check("infinite_value_rejected", raises(G._finite, float("inf")))
    check("out_of_range_rejected", raises(G._finite, 1.5))
    check("negative_rejected", raises(G._finite, -0.1))

    # ---- 3. aggregation reproduces the official parser (skipna) ------------------------
    subs = ["a", "b", "c"]
    metrics = {"a": {c: 1.0 for c in G.OFFICIAL_COMPONENTS},
               "b": {c: 0.0 for c in G.OFFICIAL_COMPONENTS},
               "c": {c: None for c in G.OFFICIAL_COMPONENTS}}      # NaN row
    agg = G.aggregate_components(subs, metrics)
    check("nan_skipped_from_numerator_and_denominator",
          all(abs(agg[c] - 0.5) < 1e-12 for c in G.OFFICIAL_COMPONENTS))
    den = G.component_denominators(subs, metrics)
    check("denominator_excludes_nan", all(den[c] == 2 for c in G.OFFICIAL_COMPONENTS))
    check("config_documents_skipna",
          "skipna=True" in cfg["official_metric_definition"]["subject_aggregation"])
    check("config_documents_error_skip",
          "SKIPPED" in cfg["official_metric_definition"]["failed_case_handling"])
    check("config_documents_missing_penalty",
          "0 for non-HD95" in cfg["official_metric_definition"]["missing_case_handling"])

    # ---- 4. ranking + baseline-wins-ties ----------------------------------------------
    base = [rec("s1", (0.9, 0.9, 0.9), (0.5, 0.5, 0.5)),
            rec("s2", (0.8, 0.8, 0.8), (0.4, 0.4, 0.4))]
    same = [rec("s1", (0.9, 0.9, 0.9), (0.5, 0.5, 0.5)),
            rec("s2", (0.8, 0.8, 0.8), (0.4, 0.4, 0.4))]
    tie = G.evaluate_candidate(base, same, expected_n=2)
    check("identical_candidate_gains_nothing", abs(tie["rank_gain_over_baseline"]) < 1e-12)
    check("identical_candidate_does_not_advance", tie["advances"] is False)
    check("tie_ci_does_not_exclude_tie",
          tie["bootstrap"]["ci_excludes_tie_favoring_candidate"] is False)

    worse = [rec("s1", (0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
             rec("s2", (0.4, 0.4, 0.4), (0.1, 0.1, 0.1))]
    w = G.evaluate_candidate(base, worse, expected_n=2)
    check("worse_candidate_rank_gain_negative", w["rank_gain_over_baseline"] < 0)
    check("worse_candidate_does_not_advance", w["advances"] is False)

    better = [rec("s1", (0.99, 0.99, 0.99), (0.9, 0.9, 0.9)),
              rec("s2", (0.98, 0.98, 0.98), (0.8, 0.8, 0.8))]
    bt = G.evaluate_candidate(base, better, expected_n=2)
    check("better_candidate_rank_gain_positive", bt["rank_gain_over_baseline"] > 0)
    check("better_candidate_can_advance", bt["advances"] is True)
    check("delta_sign_convention_positive_favours_baseline",
          w["bootstrap"]["delta_point"] > 0 and bt["bootstrap"]["delta_point"] < 0)

    # ---- 5. fail-closed validation ----------------------------------------------------
    check("subject_set_mismatch_fails",
          raises(G.evaluate_candidate, base, [rec("s1", (0.9,) * 3, (0.5,) * 3)]))
    check("expected_n_mismatch_fails",
          raises(G.evaluate_candidate, base, same, 3))
    check("evaluator_errors_fail_closed",
          raises(G.evaluate_candidate, base, same, 2, 1))
    check("duplicate_subject_fails",
          raises(G.validate_subject_records, base + [rec("s1", (0.9,) * 3, (0.5,) * 3)], base + [rec("s1", (0.9,) * 3, (0.5,) * 3)]))
    dropped = [{"subject": "s1", "metrics": {("ET", "DSC"): 0.9}},
               {"subject": "s2", "metrics": {("ET", "DSC"): 0.8}}]
    check("missing_component_fails", raises(G.validate_subject_records, base, dropped))
    check("output_contract_failure_blocks_advance",
          G.evaluate_candidate(base, better, 2, 0, False)["advances"] is False)

    # ---- 6. CANDIDATE EXPANSION is forbidden ------------------------------------------
    allowed = cfg["candidates"]["list"]
    check("candidate_list_is_exactly_the_eight_preexisting",
          allowed == ["C1", "C2", "C3", "C0_et10", "C0_et25", "C0_et50", "S1", "S2"])
    check("additions_forbidden_flag", cfg["candidates"]["additions_forbidden"] is True)
    check("no_new_thresholds_flag", cfg["candidates"]["no_new_thresholds_or_variants"] is True)
    check("baseline_not_in_candidate_list", "C0" not in allowed)
    check("frozen_candidate_tuple_matches_config", list(G.FROZEN_CANDIDATES) == allowed)
    check("choose_strongest_REJECTS_unauthorised_candidate",
          raises(G.choose_strongest, {"NOT_AUTHORISED": {"advances": True,
                                                         "rank_gain_over_baseline": 9.0,
                                                         "bootstrap": {"delta_ci_high": -9.0}}}))
    check("choose_strongest_rejects_mixed_authorised_and_unauthorised",
          raises(G.choose_strongest, {"C1": {"advances": False},
                                      "C9_INVENTED": {"advances": True,
                                                      "rank_gain_over_baseline": 9.0,
                                                      "bootstrap": {"delta_ci_high": -9.0}}}))
    check("choose_strongest_picks_higher_rank_gain",
          G.choose_strongest({
              "C1": {"advances": True, "rank_gain_over_baseline": 0.5,
                     "bootstrap": {"delta_ci_high": -0.1}},
              "C2": {"advances": True, "rank_gain_over_baseline": 0.2,
                     "bootstrap": {"delta_ci_high": -0.9}}}) == "C1")
    check("choose_strongest_returns_none_when_no_advance",
          G.choose_strongest({"C1": {"advances": False}}) is None)

    # ---- 7. FOLD LEAKAGE / premature confirmation -------------------------------------
    d = cfg["design"]
    check("calibration_folds_0_1_2", d["calibration_folds"] == [0, 1, 2])
    check("confirmation_folds_3_4", d["confirmation_folds"] == [3, 4])
    check("folds_disjoint",
          not set(d["calibration_folds"]) & set(d["confirmation_folds"]))
    check("calibration_n_811", d["calibration_n"] == 811)
    check("confirmation_n_540", d["confirmation_n"] == 540)
    check("calibration_plus_confirmation_is_full_cv",
          d["calibration_n"] + d["confirmation_n"] == 1351)
    check("confirmation_locked_flag",
          d["confirmation_locked_until_a_candidate_passes_calibration"] is True)
    check("confirmation_at_most_once", d["confirmation_may_run_at_most_once"] is True)
    check("baseline_wins_ties", d["baseline_wins_every_tie"] is True)
    check("bootstrap_seed_frozen", d["bootstrap"]["seed"] == 21072026)
    check("bootstrap_resamples_frozen", d["bootstrap"]["resamples"] == 10000)
    check("module_seed_matches_config", G.SEED == d["bootstrap"]["seed"])
    check("module_resamples_match_config",
          G.BOOTSTRAP_RESAMPLES == d["bootstrap"]["resamples"])
    check("module_threshold_matches_config",
          abs(G.MEANINGFUL_RANK_GAIN - d["meaningful_benefit_min_rank_gain"]) < 1e-12)

    # ---- 8. POST-RESULT ALTERATION guards ---------------------------------------------
    check("config_frozen_flag", cfg["frozen"] is True)
    check("config_frozen_before_results",
          cfg["frozen_before_any_candidate_official_metric_result_was_computed_or_read"] is True)
    check("result_dependent_editing_forbidden",
          cfg["result_dependent_editing_forbidden"] is True)
    check("change_requires_new_owner_gate", cfg["change_requires_new_owner_gate"] is True)
    check("history_not_rewritten_declared", "does_not_rewrite" in cfg)
    check("supersedes_key_is_scoped_to_release_selection_only",
          "supersedes_for_current_release_selection_only" in cfg)
    check("historical_evidence_explicitly_not_rewritten",
          "remain valid as recorded" in cfg["does_not_rewrite"]
          and "ONLY for current official-metric release selection" in cfg["does_not_rewrite"])
    check("reason_is_external_correction",
          cfg["reason"]["trigger"] == "EXTERNAL_OFFICIAL_METRIC_CORRECTION")
    check("candidate_source_restricted_to_preexisting",
          "already existed" in cfg["reason"]["candidate_source_restriction"])

    # ---- 9. evaluator identity + outcome vocabulary ------------------------------------
    o = cfg["official_metric_definition"]
    check("evaluator_version_0_0_8", o["evaluator_version"] == "0.0.8")
    check("evaluator_commit_pinned",
          o["evaluator_commit"] == "88e3e39cd5c4137b0831345c78d16bd393624c3a")
    check("installed_files_verified_identical",
          "configs/config_GoAT.yaml" in o["installed_files_byte_identical_to_official_repo_at_that_commit"])
    check("nsd_tolerance_uncertainty_recorded",
          o["nsd_tolerance"]["set_in_official_config"] is False
          and "residual_uncertainty" in o["nsd_tolerance"])
    check("valid_outcomes_enumerated",
          set(cfg["valid_outcomes"]) == {
              "G77_RETAIN_C0_OFFICIAL_METRIC_ALIGNED",
              "G77_SELECT_<EXISTING_CANDIDATE>_OFFICIAL_METRIC_ALIGNED",
              "G77_ARCHITECTURE_REVIEW_REQUIRED",
              "G77_BLOCKED_OFFICIAL_EVALUATOR_MISMATCH"})
    check("architecture_diagnostic_fold0_only",
          cfg["architecture_diagnostic"]["scope"].startswith("fold 0"))
    check("training_L_forbidden",
          any("training ResEnc-L" in f for f in cfg["architecture_diagnostic"]["forbidden"]))
    check("no_hidden_test_or_validation_claim",
          "hidden-test performance" in cfg["claims_forbidden"]
          and "official validation performance or leaderboard rank" in cfg["claims_forbidden"])

    # ---- 10. historical DSC+HD95 policy untouched --------------------------------------
    hist = json.loads((REPO / "configs" / "g45_selection_policy.json").read_text())
    check("historical_policy_still_v2", hist["policy_id"] == "gat26_g45_selection_policy_v2")
    check("historical_policy_still_frozen", hist["frozen"] is True)
    check("historical_policy_still_dsc_hd95",
          {c["metric"] for c in hist["primary_award_utility"]["components"]} == {"DSC", "HD95"})
    # the module may *mention* the historical policy in prose, but must never import it
    check("g77_does_not_import_historical_policy",
          re.search(r"^\s*(import|from)\s+g45_selection_policy", src, re.M) is None)

    # ---- Sections 11-12 read the private governance record ---------------------------------
    # Everything above this point — the scientific and configuration tests, candidate
    # restrictions, metric-definition checks, fold-isolation checks, bootstrap checks and
    # historical-policy checks — has already run and runs in every context.
    # Sections 11 and 12 assert on RUN_STATE.json, which is internal operational state and is not
    # redistributed publicly. In the development repository it is MANDATORY: a missing file is a
    # hard failure. Only inside a validated sanitized public export are these sections skipped,
    # and no substitute value is invented for them.
    if public_export_mode(PRIVATE_GOVERNANCE):
        print(f"  {SKIP_MESSAGE} — current-state consistency and persistence-audit checks require "
              f"the private governance record, which is not redistributed. All scientific, "
              f"configuration, candidate-restriction, metric-definition, fold-isolation, bootstrap "
              f"and historical-policy checks still ran.")
        print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
        return 1 if FAILS else 0

    # ---- 11. CURRENT-STATE CONSISTENCY: G7.7 complete XOR metric decision pending ----------
    state = json.loads((REPO / "RUN_STATE.json").read_text())
    g77 = state.get("g77_official_metric_alignment", {})
    if g77.get("status") == "COMPLETE":
        phase = state.get("phase", "").lower()
        gate = state.get("gate", {})
        blockers = " ".join(state.get("blockers", []))
        low_bl = blockers.lower()
        omd = state.get("official_metric_discrepancy", {})

        check("phase_declares_g77_complete",
              "g7.7 complete" in phase or "stage g7.7 complete" in phase)
        check("phase_states_g77_outcome",
              "g77_retain_c0_official_metric_aligned" in phase)
        check("phase_declares_g7_g75_g76_complete",
              "g7 complete" in phase and "g7.5 complete" in phase and "g7.6 complete" in phase)
        check("phase_declares_policy_final_frozen",
              "final frozen" in phase or "final frozen model" in phase)
        check("phase_declares_no_further_search",
              "no additional accuracy-policy search is authorized" in phase)
        check("phase_declares_unranked",
              "unranked_no_official_validation_score" in phase)
        check("phase_declares_a10g2_next_gate",
              "a10g-2" in phase and ("next release-compute gate" in phase or "unrun" in phase))
        check("gate_last_completed_is_g77",
              "G7.7" in gate.get("last_completed", ""))
        check("gate_last_completed_not_stale_g76",
              not gate.get("last_completed", "").startswith("G7.6"))
        check("gate_current_consistent_with_post_g77",
              "G7.7" in gate.get("current", "") or "Post-G7.7" in gate.get("current", ""))
        check("g77_decision_recorded",
              g77.get("decision") == "G77_RETAIN_C0_OFFICIAL_METRIC_ALIGNED")

        # the decisive guard: no blocker may still say the official-metric OWNER DECISION is pending
        check("no_blocker_claims_metric_owner_decision_pending",
              not re.search(r"owner decision required:?\s*official", low_bl))
        check("no_blocker_says_metric_recorded_not_acted_on",
              "recorded, not acted on" not in low_bl)
        check("metric_severity_no_longer_decision_critical",
              omd.get("severity", "") != "DECISION_CRITICAL_OWNER_INPUT_REQUIRED")
        check("metric_severity_marks_resolution",
              "RESOLVED_FOR_RELEASE_SELECTION_BY_G7.7" in omd.get("severity", ""))
        # historical wording must be LABELLED historical, not silently rewritten
        check("historical_recorded_only_text_preserved",
              omd.get("action_taken", "").startswith("RECORDED ONLY"))
        check("historical_recorded_only_text_labelled_historical",
              "HISTORICAL record" in omd.get("action_taken_historical_note", ""))
        check("forum_question_status_is_draft_not_posted",
              omd.get("forum_question_status") == "DRAFT_NOT_POSTED_OWNER_ACTION")
        # the residual must still be tracked, just accurately
        check("residual_nsd_clarification_still_tracked",
              "nsd tolerance" in low_bl and "pending" in low_bl)

    # ---- 12. persistence audit: no resource action, divergence recorded --------------------
    pa = state.get("g77_persistence_audit", {})
    if pa:
        check("persistence_outcome_recorded",
              pa.get("outcome") == "G77_PERSISTENCE_CONFIRMED_EXTERNAL_BACKUP_PENDING_NO_RESOURCE_ACTION")
        check("persistence_no_resource_action", pa.get("resource_actions_taken", "").startswith("NONE"))
        check("persistence_zero_critical_ephemeral",
              pa.get("critical_not_persistent_findings") == 0
              and pa.get("unknown_requires_owner_decision_findings") == 0)
        check("persistence_all_critical_on_volume",
              pa.get("critical_categories_checked") == pa.get("critical_categories_on_persistent_volume"))
        check("persistence_backup_not_executed",
              "NOT_EXECUTED" in pa.get("external_backup_status", ""))
        check("persistence_termination_not_safe_yet",
              pa.get("termination_safety", "").startswith("NOT SAFE"))
        # The invariant guarded here is "no execution may ever run on stale worker source".
        # It has two legitimate states, and the guard must stay strict in both:
        #   (a) a worker exists and is deliberately diverged -> expected is True and the
        #       record demands a redeploy before the next execution;
        #   (b) the worker has been terminated (G77B) -> `expected` is correctly False
        #       because no worker filesystem exists, and the record must instead demand a
        #       replacement Pod that deploys the latest tracked commit before any execution.
        # Asserting `expected is True` unconditionally would force a false statement once
        # the worker is gone, so it is scoped to case (a).
        wdd = state.get("worker_deployment_divergence", {})
        worker_terminated = bool(state.get("budget", {}).get("worker_terminated_utc"))
        if worker_terminated:
            check("worker_divergence_closed_by_termination", wdd.get("expected") is False)
            check("worker_divergence_requires_replacement_deploy",
                  "deploy" in wdd.get("required_next", "").lower()
                  and "before any execution" in wdd.get("required_next", "").lower())
        else:
            check("worker_divergence_recorded", wdd.get("expected") is True)
            check("worker_divergence_requires_redeploy_next",
                  "redeploy" in wdd.get("required_next", "").lower())

    print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
