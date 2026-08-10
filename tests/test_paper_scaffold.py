#!/usr/bin/env python3
"""GAT-26 paper-scaffold static validation (`python3 tests/test_paper_scaffold.py`).

No LaTeX toolchain required. Verifies the LNCS structure, that every populated numeric claim matches
committed evidence, that unknown values are conspicuous fail-closed placeholders (not fabricated), and
that NO protected material (case IDs, private hashes, private paths, credentials) leaks into paper/.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAPER = REPO / "paper"
MAIN = (PAPER / "main.tex").read_text()
BIB = (PAPER / "references.bib").read_text()
FAILS = 0

# --- private aggregate evidence -------------------------------------------------------------
# These fold-0 summaries back every populated numeric claim in the paper. They are internal
# evidence records and are deliberately NOT redistributed in the public source export.
PRIVATE_EVIDENCE = (
    "artifacts/g5_m_fold0_official_eval_summary.json",
    "artifacts/g5_l_fold0_official_eval_summary.json",
    "artifacts/g5_fold0_selection_decision.json",
)
SKIP_MESSAGE = "SKIP_PUBLIC_EXPORT_PRIVATE_EVIDENCE"


def public_export_mode(required_private) -> bool:
    """True ONLY inside a validated sanitized public export. Fail-closed by construction.

    Every condition must hold:
      * a well-formed root ``EXPORT_MANIFEST.json`` exists and declares the Apache-2.0 sanitized
        export;
      * that manifest actually describes THIS tree — it lists this test file and ``LICENSE``;
      * every ``required_private`` path is genuinely absent.

    Deleting files inside the development repository can NEVER activate this: the private
    repository does not contain, and never commits, ``EXPORT_MANIFEST.json`` — the manifest is
    produced only by ``scripts/make_code_export.py`` into an export directory. So missing evidence
    stays a hard failure everywhere except a real export.
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


def r3(x):
    return f"{round(x, 3):.3f}"


def r2(x):
    return f"{round(x, 2):.2f}"


def main():
    print("test_paper_scaffold:")

    # 1. LNCS structure
    check("documentclass_llncs", re.search(r"\\documentclass(\[[^\]]*\])?\{llncs\}", MAIN) is not None)
    check("has_document_env", "\\begin{document}" in MAIN and "\\end{document}" in MAIN)
    check("has_abstract", "\\begin{abstract}" in MAIN and "\\end{abstract}" in MAIN)
    check("has_keywords", "\\keywords" in MAIN)
    for sec in ("Introduction", "Results"):
        check(f"section_{sec.lower()}", f"\\section{{{sec}}}" in MAIN)
    check("has_discussion_or_limitations", "Limitations" in MAIN or "\\section{Discussion}" in MAIN)
    check("has_conclusion", "\\section{Conclusion}" in MAIN)
    check("has_bibliography", "\\bibliography{references}" in MAIN and "splncs04" in MAIN)
    check("braces_balanced", MAIN.count("{") == MAIN.count("}"))
    check("no_bare_todo", "TODO" not in MAIN and "XXX" not in MAIN.replace("XXXX", ""))

    # 2. placeholders are conspicuous macros (fail-closed), and unresolved facts use them
    # G87: every placeholder was resolved and the now-unused macro definitions were removed, so
    # requiring the definitions would be requiring dead code. What must hold is that no placeholder
    # is USED anywhere in the manuscript.
    # Stage G86 resolved the last four placeholders, so requiring them to persist would now
    # assert a falsehood -- the same replacement pattern already applied to code availability (c),
    # anonymity (d) and compilation (f) below. Each is replaced by a guard on its NEW true state:
    #   * official validation score -- measured and reported, with no rank claimed (see (e));
    #   * final image tag and digest -- the exact submitted image is recorded WITH the submission,
    #     deliberately not printed in the manuscript, and the paper must say so;
    #   * genuine-A10G peak reserved VRAM -- still NOT measured; the paper must say it holds no
    #     measurement on the organizers' target GPU rather than quietly dropping the caveat;
    #   * hidden-test results -- still do not exist; the paper must report and estimate none.
    # The macros themselves stay defined (asserted above), so any future unresolved fact still has
    # a fail-closed way to be marked.
    check("no_placeholder_macro_used_in_body",
          re.search(r"\\(pending|ownerinput)\{", re.sub(r"\\newcommand\{\\(pending|ownerinput)\}"
                                                       r"\[1\]\{[^\n]*", "", MAIN)) is None)

    # 2b. Author identity, title and acknowledgement resolved by owner decision (Stage G79-S).
    # These are now REQUIRED to be concrete, and their placeholders must be gone.
    # G87 retitled the paper: the previous title implied that five-fold ensembling was the
    # contribution. The current title names what the work actually is.
    check("title_is_approved_final",
          "\\title{GAT-26: Release-Path Auditing and Confirmation-Gated Inference Selection for "
          "Cross-Tumor\nBrain Tumor Segmentation}" in MAIN)
    forbid_title = "Five-Fold Residual Encoder Ensembling" in MAIN
    check("no_superseded_title", not forbid_title)
    check("titlerunning_present", "\\titlerunning{" in MAIN and "\\ownerinput" not in
          MAIN.split("\\titlerunning{")[1].split("}")[0])
    # G92 added an LNCS corresponding-author footnote, so the byline is no longer a bare
    # \author{Nathan Chen}. Accept the marker but keep pinning a single named author.
    check("sole_author_named",
          re.search(r"\\author\{Nathan Chen(\\thanks\{[^}]*\})?\}", MAIN) is not None)
    check("authorrunning_named", "\\authorrunning{N. Chen}" in MAIN)
    check("affiliation_exact",
          "Kang Chiao International School, Xiugang Campus, New Taipei City, Taiwan" in MAIN)
    check("correspondence_email_exact", "\\email{naifenchen52@gmail.com}" in MAIN)
    check("acknowledgement_exact",
          "The author thanks Professor Pin-Yuan Chen for his clinical guidance, methodological "
          "feedback,\nreview of the manuscript, and mentorship." in MAIN)
    # Pin-Yuan Chen is acknowledged ONLY -- never an author, never an affiliation. These inspect
    # TYPESET content: LaTeX comments are stripped first, because a `%` comment explaining the
    # distinction is not itself a byline. (`forbid` is defined further down, so use `check`.)
    typeset = re.sub(r"(?<!\\)%.*", "", MAIN)
    author_block = typeset.split("\\maketitle")[0]
    check("acknowledged_person_not_an_author", "Pin-Yuan" not in author_block)
    check("no_author_placeholder_left",
          re.search(r"\\ownerinput\{(final title|author|affiliation|acknowledgement)", MAIN, re.I)
          is None)
    # Nothing may be invented around the identity: no ORCID, degree, department, funder or grant.
    # Scoped to the typeset author block -- that is where such a claim would actually be made.
    check("no_invented_credentials",
          re.search(r"\\orcid|\bORCID\b|\bPh\.?\s?D\b|\bM\.?\s?D\.\b|\bgrant\b|\bfunded by\b|"
                    r"\bDepartment of\b", author_block, re.I) is None)
    # Exactly one correspondence address, and it is the approved permanent one. Any other address
    # in the author block -- notably the temporary school address -- is a hard failure.
    emails = set(re.findall(r"[\w.+-]+@[\w.-]+\.\w+", author_block))
    check("correspondence_email_is_sole_and_approved", emails == {"naifenchen52@gmail.com"})

    # 3. every populated numeric claim matches committed evidence (rounded as printed).
    #    These comparisons READ the private fold-0 aggregate summaries, which are internal evidence
    #    and are not redistributed publicly. In the development repository they are MANDATORY: a
    #    missing file is a hard failure. Only inside a validated sanitized public export are they
    #    skipped, and nothing is hardcoded, approximated or fabricated in their place.
    if public_export_mode(PRIVATE_EVIDENCE):
        print(f"  {SKIP_MESSAGE} — numeric-evidence comparisons require the private fold-0 "
              f"aggregate summaries, which are not redistributed. All structure, bibliography, "
              f"placeholder, sanitization, publication-state, A10G-state, page-count and "
              f"unsupported-claim guards still ran.")
    else:
        m = json.loads((REPO / "artifacts" / "g5_m_fold0_official_eval_summary.json").read_text())
        l = json.loads((REPO / "artifacts" / "g5_l_fold0_official_eval_summary.json").read_text())
        mc, lc = m["components_mean"], l["components_mean"]
        # G87 compressed the fold-0 screen to the six DSC values; the fold-0 HD95 values and the
        # DSC/HD95 tail percentiles were cut from the manuscript. Values the paper still states must
        # match the evidence exactly; values it no longer states must be genuinely ABSENT rather
        # than replaced by a different number, which is what the second loop enforces.
        # G92 moved the fold-0 architecture screen values from the main paper to the supplement,
        # which ships in the same camera-ready package. A stated value must still match the
        # evidence exactly; it may now live in either document.
        supp_path = PAPER / "supplement.tex"
        SUPP = supp_path.read_text() if supp_path.is_file() else ""
        for label, val, fmt in [
            ("M_et_dsc", mc["et_dsc"], r3), ("M_tc_dsc", mc["tc_dsc"], r3), ("M_wt_dsc", mc["wt_dsc"], r3),
            ("L_et_dsc", lc["et_dsc"], r3), ("L_tc_dsc", lc["tc_dsc"], r3), ("L_wt_dsc", lc["wt_dsc"], r3),
        ]:
            check(f"numeric_matches_evidence::{label}", fmt(val) in MAIN or fmt(val) in SUPP)
        # the fold-0 screen must not report an HD95 or tail-percentile number at all now
        cut = {"M_et_hd95": r2(mc["et_hd95"]), "M_tc_hd95": r2(mc["tc_hd95"]),
               "M_wt_hd95": r2(mc["wt_hd95"]), "L_et_hd95": r2(lc["et_hd95"]),
               "L_tc_hd95": r2(lc["tc_hd95"]), "L_wt_hd95": r2(lc["wt_hd95"]),
               "M_dsc_p05": r3(m["dsc_p05"]), "L_dsc_p05": r3(l["dsc_p05"]),
               "M_hd95_p95": r2(m["hd95_p95"]), "L_hd95_p95": r2(l["hd95_p95"])}
        screen = MAIN.split("On the frozen fold-0 validation set")[-1].split("\n\n")[0]
        check("cut_fold0_values_not_misreported",
              not [k for k, v in cut.items() if v in screen])
        check("val_count_271_matches", str(m["n"]) == "271" and "271" in MAIN)
        sel = json.loads((REPO / "artifacts" / "g5_fold0_selection_decision.json").read_text())
        check("selection_select_M",
              sel["decision"] == "select_M"
              and ("selects ResEnc-M" in MAIN or "selected ResEnc-M" in MAIN))
        n_res = sel["bootstrap_resamples"]
        res_forms = (str(n_res), f"{n_res:,}".replace(",", "{,}"), f"{n_res:,}")  # 10000 / 10{,}000 / 10,000
        check("bootstrap_seed_in_paper", str(sel["seed"]) in MAIN and any(f in MAIN for f in res_forms))

    # 4. required BibTeX entries present (organizer-mandated + method), keyed, no empty keys
    keys = re.findall(r"@\w+\{([^,]+),", BIB)
    check("bib_has_keys", len(keys) >= 10 and all(k.strip() for k in keys))
    for want in ("baid2021brats", "menze2015brats", "isensee2021nnunet"):
        check(f"bib_has::{want}", want in keys)
    check("bib_arxiv_2107", "2107.02314" in BIB)   # flagship id verbatim from the rules page

    # 5. NO protected material anywhere under paper/
    # Stage G86 compiled the manuscript in place, so paper/ now also holds build artifacts (the
    # PDF, .aux, .bbl, .log). Those are byte streams, not UTF-8, and reading them as text raised
    # UnicodeDecodeError and aborted this whole scan. Decode every file leniently instead: the
    # scan must cover the compiled output too, since that is what actually gets submitted.
    # Compressed PDF streams are of course not searchable this way -- the manuscript sources
    # below are the authoritative text, and the PDF is additionally audited by extraction.
    def _text(p: Path) -> str:
        return p.read_bytes().decode("utf-8", errors="replace")

    files = [p for p in PAPER.rglob("*") if p.is_file()]
    joined = "\n".join(_text(p) for p in files)
    # A compiled binary's byte soup can coincidentally contain 64 hex characters, which would be
    # a false hash "leak". The hash guard therefore reads the text sources, where a real private
    # digest would actually have to be written.
    # The OpenReview worksheet legitimately records the compiled PDF's own SHA-256 -- the owner has
    # to verify it against the uploaded file -- and it is excluded from the public export, so it is
    # not part of the manuscript-source hash guard.
    sources = "\n".join(_text(p) for p in files
                        if p.suffix.lower() not in (".pdf", ".aux", ".bbl", ".blg", ".log",
                                                    ".out", ".synctex", ".gz")
                        and p.name != "OPENREVIEW_SUBMISSION_VALUES.md")
    check("no_real_case_ids", re.search(r"BraTS-(GLI|MEN|MET|PED|SSA)-\d{5}", joined) is None)
    check("no_64hex_hash", re.search(r"\b[a-f0-9]{64}\b", sources) is None)
    check("no_private_paths", "/workspace/data" not in joined and "/dev/shm" not in joined
          and "/workspace/runs" not in joined)
    check("no_credentials", not re.search(r"(BEGIN [A-Z ]*PRIVATE KEY|synapseConfig|JUPYTER_PASSWORD|ssh-rsa AAAA)", joined))

    # 6. ACCURACY GUARDS — fail if any premature/unsupported claim reappears (G8P-R).
    low = joined.lower()
    # LaTeX hard-wraps prose, so a phrase may span a newline: match on whitespace-normalized text.
    flat = re.sub(r"\s+", " ", low)

    def forbid(name, present):
        check(name, not present)

    # (a) cohort labels/sizes must NOT be claimed known
    forbid("no_cohort_recorded_privately",
           "recorded privately" in low or "per-cohort labeled-subset sizes are recorded" in low)
    check("cohort_accuracy_stated",
          "reliably derivable" in flat and "did not guess" in flat
          and ("cohort-stratified analysis is unavailable" in flat
               or "cohort-stratified analysis lies outside this work" in flat))

    # (b) no completed container / genuine-A10G validation claimed prematurely
    forbid("no_completed_container_claim",
           "packaged as a zero-network" in low
           or "the submission is a linux/amd64, zero-network clean-room container" in low)
    # The image must still be declared NOT built / NOT accepted (A10G-2 is the gate).
    # Stage G86 built, qualified, pushed and submitted the image, so "not yet built" would now be
    # false. The replacement guards keep the two things that must not drift: the manuscript points
    # to the submission record for the exact image identity instead of printing a digest it cannot
    # keep in sync, and it still refuses to claim measurement on the organizers' target GPU.
    # G91: the camera-ready removed the image-identity sentence entirely rather than deferring it.
    # Saying nothing is strictly safer than saying "recorded elsewhere", so accept either -- but
    # never accept a paper that actually names an image, tag or registry path.
    # The alternative branch is scoped to the manuscript BODY (comments stripped): the process
    # docs under paper/ legitimately discuss image identity as a checklist item, and a LaTeX
    # comment is not a published claim.
    main_body = re.sub(r"(?m)^\s*%.*$", "", MAIN).lower()
    check("container_identity_deferred_to_submission_record",
          "recorded with the challenge submission" in flat
          or not re.search(r"image (digest|identity|tag)|registry|docker\.io|ghcr", main_body))
    # the manuscript must never print the image tag or digest
    check("no_image_digest_in_manuscript",
          re.search(r"sha256|gat26-c0:|docker\.synapse", MAIN, re.I) is None)
    forbid("no_stale_container_unbuilt_claim", "has not yet been built" in flat)
    forbid("no_a10g_validation_passed_claim",
           bool(re.search(r"a10g[- ]?(1|2)?\s*(validation|test)\s*(passed|complete|succeeded|done)", low)))
    # G87-R: an A10G measurement now EXISTS, so requiring the "no measurement" caveat would assert a
    # falsehood. The guard flips to what could actually be overstated: any A10G statement must be
    # backed by committed evidence (checked below in g87r_a10g_claim_backed_by_evidence) and must
    # carry its limitations -- synthetic inputs, not organizer execution, not full runtime parity.
    a10g_in_paper = "nvidia a10g" in flat
    # G91: the historical A10G exercise applies to a SUPERSEDED pre-correction image. The corrected
    # image that was finally submitted has no A10G measurement at all, and the paper must say so.
    check("a10g_claim_carries_its_limits",
          (not a10g_in_paper) or
          ("synthetic" in flat
           and "superseded" in flat
           and "does not qualify the corrected image" in flat
           and "no organizer execution log" in flat))

    # (c) code availability. Stage G79-P actually published the sanitized export and verified it by
    # unauthenticated clone, so the pre-G79-P guards ("currently private" + an \ownerinput
    # placeholder) would now assert a falsehood. They are replaced by guards on the NEW true state:
    # the paper must cite the exact verified public URL, and must still not overclaim around it.
    check("code_availability_public_url",
          "https://github.com/Neethan-hub/gat26-brats-2026" in MAIN)
    check("code_availability_states_apache",
          re.search(r"apache license\s*\n?\s*2\.0", low) is not None)
    check("code_availability_no_data_claim",
          re.search(r"no images, labels,( model)? checkpoints,? or predictions", flat) is not None)
    forbid("no_code_availability_placeholder_left",
           re.search(r"\\ownerinput\{(public source-code url|code-availability)", MAIN, re.I) is not None)
    forbid("no_stale_private_repo_claim", "currently private" in flat)

    # (d) anonymity: RESOLVED in Stage G79-S. The organizers confirmed the review is SINGLE-BLIND,
    # so author identity must NOT be anonymized and naming the author is correct. The earlier guards
    # required this to remain an open owner-verification blocker and would now assert a falsehood.
    # They are replaced by guards that the resolution is recorded WITH its basis -- a non-anonymous
    # manuscript must never rest on "no rule was found".
    check("anonymity_resolved_single_blind", "single-blind" in low)
    check("anonymity_resolution_cites_basis",
          "13911" in low or "discussion thread" in low)
    forbid("no_anonymity_absence_of_rule_justification",
           "no anonymity requirement found" in low or "absence of a rule is not confirmation" in low)

    # (e) pending final evidence must remain PENDING, not presented as completed
    # Post-G7/G7.5/G7.6: the five-fold CV and the trained folds are COMPLETE and must be reported
    # as such; only genuinely unmeasured quantities may remain \pending.
    check("five_fold_cv_reported_not_pending",
          "\\pending{five-fold cv results}" not in MAIN.lower()
          and "1{,}351 cases, each exactly once" in MAIN)
    check("folds_1_4_not_claimed_pending", "\\pending{m folds 1--4}" not in MAIN.lower())
    check("all_five_folds_complete_stated",
          "all five folds (0--4) are complete" in flat or "all five are complete" in flat)
    check("oof_evaluator_zero_errors_stated", "$n=1{,}351$, zero errors" in MAIN or "zero errors" in flat)
    # The released ensemble still cannot be scored on training data; only the wording moved.
    check("ensemble_unmeasurable_stated",
          "cannot be scored without optimistic bias" in flat or "unmeasurable" in flat
          or ("is not the released ensemble's score" in flat
              and "trained four of the five" in flat))
    # Stage G86: an official validation score now EXISTS and is reported. The guard therefore
    # flips from "no submission exists" to "a submission exists and confers no rank", which is the
    # claim that could actually be overstated. Hidden-test results still do not exist.
    check("official_validation_reported_without_rank",
          "carries no rank" in flat or "no official rank" in flat)
    check("hidden_test_still_absent",
          ("no hidden-test measurement exists" in flat
           or "does not exist at the time of writing" in flat)
          and ("we report none and estimate none" in flat
               or "we report and estimate none" in flat))
    forbid("no_hidden_test_number_claimed",
           bool(re.search(r"hidden[- ]test[^.\n]{0,60}(dice|dsc|nsd|score) of\s*\d", low)))
    check("official_metrics_dsc_nsd_stated",
          "dice similarity coefficient" in flat and "normalized surface distance" in flat)
    forbid("no_final_results_faked",
           bool(re.search(r"five-fold cross-validated (dsc|results) (are|is|of)\s*\d", low)))

    # (f) submission blockers documented.
    # Stage G78 actually compiled the paper against the official Springer LNCS class, so the
    # pre-G78 guards ("compilation has not been run" / "no compilation passed claim") would now
    # assert a falsehood. They are replaced by guards on the NEW true state: the compile result and
    # the verified page count must be recorded, and compiling must NOT be mistaken for being
    # submission-ready while owner placeholders remain open.
    ledger = (PAPER / "CITATION_LEDGER.md").read_text().lower()
    checklist = (PAPER / "SUBMISSION_CHECKLIST.md").read_text().lower()
    check("bibtex_hard_blocker", "hard blocker" in ledger and "hard blocker" in checklist)
    check("compile_result_recorded",
          "llncs" in low and "llncs" in checklist
          and re.search(r"excluding references", low) is not None)
    # The count is re-measured whenever the manuscript changes; G78 recorded 9, the G86 final
    # build measures 10. Assert a recorded count that is actually inside the required range,
    # rather than pinning the number of one historical build.
    m_pages = re.findall(r"\b(\d{1,2}) pages? excluding references", low)
    check("page_count_recorded_in_range",
          bool(m_pages) and all(8 <= int(v) <= 10 for v in m_pages) and "8–10" in low)
    check("compile_not_equated_with_submission_ready",
          "not the same as being submission-ready" in low)
    # A clean compile must never be reported as clearing the owner-input placeholders.
    forbid("no_placeholders_resolved_claim",
           bool(re.search(r"(all|every) (owner|placeholder)[^.\n]*(resolved|filled|complete)", low)))

    # ------------------------------------------------------------------ G87 regression guards
    # Each of these encodes a factual correction made in Stage G87. They fail if the manuscript
    # ever drifts back to a claim the committed evidence does not support.

    # (1) The M8 audit is a SINGLE-MODEL out-of-fold comparison; the deployed submission ensembles
    #     all five checkpoints. Conflating them was the highest-severity defect in the G86 draft.
    check("g87_m8_scope_stated",
          "excluded-fold checkpoint" in flat and "all five" in flat
          and "does not directly measure" in flat)
    forbid("g87_no_m8_equals_release_ensemble_claim",
           bool(re.search(r"identical to the (final |released |deployed )?(five-model |five-fold )?"
                          r"ensemble except", low))
           or "the exact released policy with only one changed switch" in low
           or bool(re.search(r"mirroring[^.\n]{0,80}decisively closed", low)))

    # (2) Folds 3--4 are a same-corpus policy-selection holdout, not an independent cohort.
    check("g87_holdout_terminology",
          "policy-selection holdout" in flat and "not an external independent cohort" in flat)
    forbid("g87_no_independence_overclaim",
           "independent confirmation" in flat or "untouched data" in flat
           or "strictly held out throughout development" in flat
           or bool(re.search(r"opened exactly once", low)))

    # (3) The out-of-fold estimate is descriptive and post-selection, never "unbiased".
    check("g87_oof_described_as_post_selection",
          "descriptive" in flat and "post-selection" in flat)
    # Flag only AFFIRMATIVE unbiasedness claims: "not an unbiased estimate" and "rather than
    # unbiased" are the corrected wording and must not trip this guard.
    affirmative_unbiased = [
        m for m in re.finditer(r"\bunbiased\b", low)
        if not re.search(r"(not|never|rather than|neither)\s+(an?\s+)?$", low[max(0, m.start() - 24):m.start()])
    ]
    forbid("g87_no_unbiased_oof_claim", bool(affirmative_unbiased))

    # (4) The G85 protocol was confirmation-frozen but calibration-informed, never result-blind.
    # G91: "calibration" as a subset name is retired; the admission itself is unchanged.
    check("g87_protocol_honesty",
          "not globally result-blind" in flat
          and ("calibration-informed" in flat
               or "designed after seeing the development outcome" in flat))

    # (5) The submitted runner does NOT fail closed on an output collision (G86 measured
    #     exit_code 0, failed_closed false), so no rejection may be claimed anywhere in paper/.
    forbid("g87_no_output_collision_rejection_claim",
           "output-name collision" in flat or "output collisions" in flat
           or bool(re.search(r"reject[^.\n]{0,40}collision", low)))
    check("g87_fresh_output_contract_stated",
          "fresh writable" in flat or "fresh output director" in flat)

    # (6) The organizers DID confirm tau=1 for the final ranking, so "the tolerance was never
    #     exposed" is false and must not reappear. What stays true is narrower: the tolerance behind
    #     the returned participant-visible validation values was not disclosed, so none is attached
    #     to those numbers.
    forbid("g87_no_tolerance_never_exposed_claim",
           "never exposed" in flat or "did not expose the surface tolerance" in flat)
    check("g87_official_nsd_tolerance_not_claimed",
          "did not disclose which tolerance" in flat or "which tolerance produced" in flat)
    forbid("g87_no_tolerance_label_on_official_nsd",
           bool(re.search(r"nsd \(ranked\)[^\n]{0,40}tau", low))
           or bool(re.search(r"official[^.\n]{0,60}nsd[^.\n]{0,30}at (the )?(surface )?tolerance", low)))

    # (7) The bootstrap summary is a resample fraction, not a posterior probability.
    check("g87_bootstrap_fraction_wording",
          "fraction of" in flat or "share of" in flat)
    forbid("g87_no_posterior_probability_wording",
           "paired-bootstrap probability" in flat or "bootstrap probability" in flat
           or "p(\\delta u>0)" in low.replace(" ", ""))

    # (8) The split is duplicate-audited, not proven leakage-free.
    forbid("g87_no_leakage_safe_claim", "leakage-safe" in flat)

    # (9) No adversarial or defensive framing.
    forbid("g87_no_adversarial_framing",
           "a team stopping at calibration" in low or "what a team stopping" in low)

    # ------------------------------------------------------------ G87-R regression guards
    # Stage G87-R removed five claims the committed evidence does not support. These guards are
    # written to separate a PROHIBITED AFFIRMATIVE CLAIM from an accurate negation or limitation:
    # each looks for the affirmative pattern and then checks that it is not inside a negation.
    def affirms(pattern, negators=(r"not\b", r"never\b", r"no\b", r"rather than", r"neither",
                                  r"does not", r"did not", r"cannot", r"stopped at")):
        """True only where `pattern` matches WITHOUT a nearby negator in front of it."""
        out = []
        for m in re.finditer(pattern, flat):
            before = flat[max(0, m.start() - 70):m.start()]
            if not any(re.search(n + r"[^.]{0,60}$", before) for n in negators):
                out.append(m.group(0))
        return out

    # (1) Not every candidate had to survive a holdout; audits A and B stopped at calibration.
    # G91: audits now "stop on the development subset" rather than "at calibration".
    check("g87r_staged_evaluation_stated",
          ("stopped at calibration" in flat or "stopped on the development subset" in flat)
          and ("evaluation is" in flat and "staged" in flat))
    forbid("g87r_no_universal_holdout_claim",
           bool(affirms(r"(each|every) (audit|candidate)[^.]{0,80}(holdout|calibration-then-holdout)"))
           or bool(affirms(r"a candidate is adopted only if[^.]{0,60}holdout"))
           or bool(affirms(r"(each|every) tunable[^.]{0,80}holdout")))

    # (2) Folds 3--4 are a same-corpus policy-selection holdout.
    check("g87r_same_corpus_holdout_stated", "same-corpus policy-selection holdout" in flat)
    forbid("g87r_no_external_cohort_claim",
           bool(affirms(r"holdout[^.]{0,40}(is|was) (an )?(external|independent)")))

    # (3) The reference-lesion opportunity count is POLICY-INVARIANT.
    check("g87r_opportunity_count_invariant_stated",
          "policy-invariant" in flat and "reference lesion components" in flat)
    forbid("g87r_no_opportunity_count_moves_claim",
           "opportunity count can move with the policy" in flat
           or bool(affirms(r"(opportunity|reference)[^.]{0,60}count[^.]{0,40}(move|change|vary)"
                           r"[^.]{0,40}(with the )?policy")))

    # (4) G85 did not retroactively repair or annul G84.
    check("g87r_earlier_audit_stands",
          "correctly applied its own frozen rule" in flat or "stands as recorded" in flat)
    forbid("g87r_no_retroactive_repair_claim",
           bool(affirms(r"(repair|annul|erase|invalidat\w+|fix\w*)[^.]{0,40}(earlier|previous|g84)"))
           or "retroactively repair" in flat and "does not" not in flat)

    # (5) M8 did not measure mirroring on the deployed five-model ensemble.
    forbid("g87r_no_ensemble_measurement_claim",
           bool(affirms(r"(measures?|measured|establishes)[^.]{0,60}mirroring[^.]{0,60}"
                        r"(five-(model|checkpoint)|deployed) ensemble")))

    # (6) The Docker receipt is not evidence of organizer execution.
    forbid("g87r_no_execution_from_receipt_claim",
           bool(affirms(r"(receipt|submission)[^.]{0,60}(proves|confirms|demonstrates)"
                        r"[^.]{0,40}(execution|the organizers ran|was run by the organizers)")))

    # (7) No A10G success may be claimed without committed direct evidence for it.
    # The A10G aggregate lives in artifacts/, which the sanitized public export deliberately does
    # not carry (it records the submitted image's manifest digest). Inside a VALIDATED export the
    # file is legitimately absent -- the same fail-closed pattern used for the fold-0 evidence
    # above -- so the check is satisfied there and stays a hard requirement everywhere else.
    a10g_evidence = (REPO / "artifacts" / "g87r_a10g_qualification.json")
    a10g_passed = False
    if a10g_evidence.exists():
        try:
            a10g_passed = bool(json.loads(a10g_evidence.read_text()).get("overall_pass"))
        except ValueError:
            a10g_passed = False
    elif public_export_mode(("artifacts/g87r_a10g_qualification.json",)):
        a10g_passed = True   # validated sanitized export: private evidence is intentionally absent
    claims_a10g = bool(affirms(r"(on|using) (one |a single )?nvidia a10g")) and \
        not ("we hold no measurement of the container on an nvidia a10g" in flat)
    check("g87r_a10g_claim_backed_by_evidence", (not claims_a10g) or a10g_passed)

    # (8) Publication integrity. The owner removed the voluntary AI-tooling sentence from the
    # Disclosure of Interests; the binding requirement is unchanged and still enforced -- no software
    # system may appear as an author or contributor anywhere in the front matter or credits.
    credits_block = MAIN.split("\\begin{credits}")[-1] if "\\begin{credits}" in MAIN else ""
    software_names = r"(claude|chatgpt|gpt-\d|copilot|codex|gemini|llm|language model|ai assistant)"
    check("g87r_no_software_listed_as_author",
          re.search(software_names, author_block, re.I) is None
          and re.search(r"\\author\{[^}]*" + software_names, MAIN, re.I) is None
          and re.search(software_names + r"[^.\n]{0,40}(is|as)( an?)? (author|contributor)",
                        credits_block, re.I) is None)
    check("g87r_competing_interests_present",
          "declares no competing interests" in flat)

    print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
