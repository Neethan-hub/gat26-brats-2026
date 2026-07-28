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
    check("ownerinput_macro_defined", "\\newcommand{\\ownerinput}" in MAIN)
    check("pending_macro_defined", "\\newcommand{\\pending}" in MAIN)
    # Placeholders that must STILL exist (genuinely unmeasured / not yet owner-supplied).
    # "final title" left this list in Stage G79-S when the owner approved the final title; the
    # remaining entries are all quantities that do not exist yet and must never be invented.
    for tok in ("genuine-A10G peak reserved VRAM", "final image tag and digest",
                "hidden-test results", "official validation score"):
        check(f"placeholder_present::{tok[:22]}", tok in MAIN)

    # 2b. Author identity, title and acknowledgement resolved by owner decision (Stage G79-S).
    # These are now REQUIRED to be concrete, and their placeholders must be gone.
    check("title_is_approved_final",
          "\\title{GAT-26: Five-Fold Residual Encoder Ensembling for Generalizable Brain Tumor "
          "Segmentation}" in MAIN)
    check("titlerunning_present", "\\titlerunning{" in MAIN and "\\ownerinput" not in
          MAIN.split("\\titlerunning{")[1].split("}")[0])
    check("sole_author_named", "\\author{Nathan Chen}" in MAIN)
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
        for label, val, fmt in [
            ("M_et_dsc", mc["et_dsc"], r3), ("M_tc_dsc", mc["tc_dsc"], r3), ("M_wt_dsc", mc["wt_dsc"], r3),
            ("M_et_hd95", mc["et_hd95"], r2), ("M_tc_hd95", mc["tc_hd95"], r2), ("M_wt_hd95", mc["wt_hd95"], r2),
            ("L_et_dsc", lc["et_dsc"], r3), ("L_tc_dsc", lc["tc_dsc"], r3), ("L_wt_dsc", lc["wt_dsc"], r3),
            ("L_et_hd95", lc["et_hd95"], r2), ("L_tc_hd95", lc["tc_hd95"], r2), ("L_wt_hd95", lc["wt_hd95"], r2),
            ("M_dsc_p05", m["dsc_p05"], r3), ("L_dsc_p05", l["dsc_p05"], r3),
            ("M_hd95_p95", m["hd95_p95"], r2), ("L_hd95_p95", l["hd95_p95"], r2),
        ]:
            check(f"numeric_matches_evidence::{label}", fmt(val) in MAIN)
        check("val_count_271_matches", str(m["n"]) == "271" and "271" in MAIN)
        sel = json.loads((REPO / "artifacts" / "g5_fold0_selection_decision.json").read_text())
        check("selection_select_M", sel["decision"] == "select_M" and "selects ResEnc-M" in MAIN)
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
    joined = "\n".join(p.read_text() for p in PAPER.rglob("*") if p.is_file())
    check("no_real_case_ids", re.search(r"BraTS-(GLI|MEN|MET|PED|SSA)-\d{5}", joined) is None)
    check("no_64hex_hash", re.search(r"\b[a-f0-9]{64}\b", joined) is None)
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
          and "cohort-stratified analysis is unavailable" in flat)

    # (b) no completed container / genuine-A10G validation claimed prematurely
    forbid("no_completed_container_claim",
           "packaged as a zero-network" in low
           or "the submission is a linux/amd64, zero-network clean-room container" in low)
    # The image must still be declared NOT built / NOT accepted (A10G-2 is the gate).
    check("container_pending_stated",
          ("has not yet been built" in flat)
          or ("final container image" in flat and "not yet available" in flat)
          or ("\\pending{final image tag and digest}" in MAIN))
    forbid("no_a10g_validation_passed_claim",
           bool(re.search(r"a10g[- ]?(1|2)?\s*(validation|test)\s*(passed|complete|succeeded|done)", low)))

    # (c) code availability. Stage G79-P actually published the sanitized export and verified it by
    # unauthenticated clone, so the pre-G79-P guards ("currently private" + an \ownerinput
    # placeholder) would now assert a falsehood. They are replaced by guards on the NEW true state:
    # the paper must cite the exact verified public URL, and must still not overclaim around it.
    check("code_availability_public_url",
          "https://github.com/Neethan-hub/gat26-brats-2026" in MAIN)
    check("code_availability_states_apache",
          re.search(r"apache license\s*\n?\s*2\.0", low) is not None)
    check("code_availability_no_data_claim",
          "no images, labels, model checkpoints, or predictions" in flat)
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
    check("all_five_folds_complete_stated", "all five folds (0--4) are complete" in flat)
    check("oof_evaluator_zero_errors_stated", "$n=1{,}351$, zero errors" in MAIN or "zero errors" in flat)
    check("ensemble_unmeasurable_stated",
          "cannot be scored without optimistic bias" in flat or "unmeasurable" in flat)
    for item in ("final image tag and digest", "hidden-test results, camera-ready"):
        check(f"still_pending::{item[:20]}", f"\\pending{{{item}}}" in MAIN)
    check("still_pending::genuine_a10g",
          "\\pending{genuine-a10g peak" in MAIN.lower())
    check("no_official_validation_rank_claimed",
          "no official validation submission and therefore no official rank" in flat)
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
    check("page_count_recorded_in_range",
          re.search(r"\b9 pages\b.*excluding references|excluding references.*\b9 pages\b", low)
          is not None and "8–10" in low)
    check("compile_not_equated_with_submission_ready",
          "not the same as being submission-ready" in low)
    # A clean compile must never be reported as clearing the owner-input placeholders.
    forbid("no_placeholders_resolved_claim",
           bool(re.search(r"(all|every) (owner|placeholder)[^.\n]*(resolved|filled|complete)", low)))

    print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
