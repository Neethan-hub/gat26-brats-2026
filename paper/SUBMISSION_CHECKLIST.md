# GAT-26 — BraTS 2026 Task 3 owner submission checklist

Live-verified against the official sources on **2026-07-23T07:23:22Z** and **re-verified
2026-07-25T12:52–12:58Z** via the authenticated Synapse API (Synapse `wiki/639582` Submission
Instructions, `wiki/639585` Challenge Rules, `wiki/639587` Timeline, `wiki/639592` Task-3 Evaluation
[mod **2026-07-24T17:54:36Z**], evaluation queues `9619539`/`9619630`, SubmissionView `syn74508245`).
**Final submission deadline: 30 Jul 2026, 23:59 UTC** (all queues, incl. validation leaderboard,
close). Re-verify before acting.

Every box is **owner-only and interactive**. Claude does not submit, upload, or provision anything.

## A0. Verified queue facts (2026-07-25)

| Fact | Value |
|---|---|
| Task-3 **validation predictions** queue | evaluation `9619539` — round 2026-06-01 → **2026-07-30T23:59:00Z**, limit **DAILY max 2** |
| Task-3 **final Docker** queue | evaluation `9619630` — round 2026-07-03 → **2026-07-30T23:59:00Z**, limit **TOTAL max 1** |
| Queues are independent | A validation-prediction submission does **not** consume the one-shot Docker quota |
| Validation format | one `.zip`/`.tar[.gz]` of `.nii.gz` files; filename must end in the **5-digit case ID**; Task-3 exception: **no** 3-digit timepoint; geometry must match the source exactly or the submission is invalidated |
| **Official Task-3 award metrics** | **DSC + NSD** (`wiki/639592`, mod 2026-07-24). **HD95 is not used and appears nowhere in the challenge wiki.** See `docs/RULE_SNAPSHOT.md` §2026-07-25 |
| Ranking method | summed ranks across metric averages, then 500,000-permutation pairwise significance tiers (DELPHI) |
| Missing-metric handling | organizers do **not** substitute the worst value for a failed case |
| Observed submission risk | of 676 Task-3 validation submissions, **44 INVALID + 11 NOT_SCORED** (~8%) — format errors are common |
| Container verification | organizers do **not** verify containers run until after queues close; a failing container is **disqualified without evaluation** |
| Track rule | a team may submit to **one** track only (EARLY vs standard); GAT-26 is on the standard track |

## A. Eligibility & team
- [ ] Challenge registration complete and team formed on Synapse (project `syn74274097`); note the
      exact **Synapse team name** — it links the short paper to the Docker submission.
- [ ] Confirm eligibility (GoAT-only / random-init compliance already met by the method).

## B. Short paper (MANDATORY — no paper ⇒ Docker not evaluated/ranked)
- [ ] 8–10 pages excluding references, **Springer LNCS format** (`llncs.cls` from the official
      template; not committed here for copyright reasons).
- [ ] Required sections present: Abstract (no citations), keywords, Introduction, Methods, Results,
      Discussion, Acknowledgements (optional), References (must include the Challenge-Rules citations).
- [ ] Fill every `\ownerinput{...}` / `\pending{...}` placeholder — see list in §F.
- [ ] Add the acknowledgement sentence: "Data used in this publication were obtained as part of the
      Challenge project through Synapse ID (syn74274097)."
- [ ] Include the mandated flagship + challenge-specific citations (see `CITATION_LEDGER.md`).
- [x] **HARD BLOCKER (CLEARED, G78 2026-07-28) — complete every BibTeX entry**: every title and
      leading author list re-verified against the **primary source** (arXiv landing pages / publisher
      DOIs). The organizational-author placeholders (`{BraTS Challenge Organizers}`) are gone. Large
      consortium lists are truncated with `and others` → "et al.", a deliberate verified truncation.
- [x] **HARD BLOCKER (CLEARED, G78) — compile with the official LNCS class.** Compiled against the
      official Springer `llncs.cls` **v2.26** + `splncs04.bst` in an isolated git-ignored `/tmp` TeX
      environment (nothing installed globally). **0 undefined citations/references; 12/12 bibliography
      entries typeset.**
- [x] **HARD BLOCKER (CLEARED, G78) — rendered visual review.** All 10 typeset pages rendered and
      inspected: no clipping, no broken references, no malformed tables, no missing citations, no
      visibly unfinished prose. 2 minor overfull boxes remain (7.7 pt in body; 13.1 pt produced by the
      long `\ownerinput{acknowledgements}` placeholder, which resolves when the owner supplies text).
- [x] **HARD BLOCKER (CLEARED, G78) — rendered length verified 8–10 pages excluding references:**
      **9 pages excluding references** (10 pages total; References heading begins on page 9).
- [x] **HARD BLOCKER CLEARED (G79-P, 2026-07-28) — public source-code URL (GitHub).** Timeline wiki
      639587, verbatim: *"Short paper must report a) source code (GitHub link), b) method
      description, and c) results on training and validation data."* The sanitized export is now
      **published under Apache-2.0** at <https://github.com/Neethan-hub/gat26-brats-2026> (PUBLIC,
      one root commit, no inherited private history, verified by unauthenticated clone), and
      `main.tex` cites it — verified rendered and hyperlinked in the compiled PDF. The **private
      development repository remains PRIVATE**.
- [ ] Submit the paper via **OpenReview** (`https://openreview.net/group?id=MICCAI.org/2026/Challenge`)
      and, where prompted (BrainLes CMT), provide the exact Synapse team name. **Confirm** the paper is
      submitted and the team name is correct — organizers only run Docker submissions linked to a paper.
- [ ] **BLOCKER — verify the OpenReview venue anonymity setting** (single/double-blind vs.
      non-anonymous) directly in the OpenReview venue before preparing the author block. No anonymity
      rule was found on the wiki, but **absence of a rule is not confirmation**; do not assume
      non-anonymous.

## C. Every author (exact, per author)
- [ ] Full legal name
- [ ] Email (Synapse-affiliated where required)
- [ ] Affiliation(s)
- [ ] OpenReview profile (existing/created; name matches)

## D. Docker / final submission
- [ ] Final image built and **passing A10G-2 acceptance** (five genuinely distinct ResEnc-M fold_0–4
      checkpoints; genuine NVIDIA A10G; VRAM <21 GiB; runtime <9 h projection; all output assertions).
      See `configs/release/AWS_A10G_RUNBOOK.md`. **A10G-1 smoke does not substitute for A10G-2.**
- [ ] Image pushed to the required registry (`docker.synapse.org/PROJECT_ID/IMAGE_NAME:TAG`).
- [ ] Record the **final image digest** (sha256) and tag.
- [ ] Submit the Docker image in the **Task 3 (Generalizability)** Submission section on Synapse.
- [ ] **Confirm any validation/leaderboard/smoke submission does NOT consume the final ranking
      attempt** (check the Task-3 submission-system limits before submitting the final image; the
      official pages do not state a numeric attempt count — verify in the submission UI). Note the
      Early-vs-Final track rule: submitting during the Early deadline forfeits the Final deadline.

## E. Copyright form
- [ ] Organizers provide the copyright form **at camera-ready**; sign and return it then.

## F. Placeholders that must be resolved (from `main.tex`)
- [ ] Final title
- [ ] Final author list, affiliations, emails, OpenReview profiles
- [ ] Acknowledgements
- [x] Public code repository URL (for the paper's reproducibility statement) — **filled (G79-P)**:
      <https://github.com/Neethan-hub/gat26-brats-2026>
- [ ] M folds 1–4 (produced by G7) — `[PENDING_G7/A10G]`
- [ ] Five-fold cross-validated results — `[PENDING_G7/A10G]`
- [ ] Final ensemble results — `[PENDING_G7/A10G]`
- [ ] Genuine-A10G measurements (A10G-1 and A10G-2: peak VRAM, seconds/case) — `[PENDING_G7/A10G]`
- [ ] Final container image digest/tag — `[PENDING_G7/A10G]`
- [ ] Hidden-test results (camera-ready)
- [ ] Team name, OpenReview submission ID, Synapse submission ID

## G. Final gate
- [ ] `FINAL_AUTHORIZATION.md` (per `RELEASE_CHECKLIST.md`) completed with the exact commit, image
      digest/tag, config/checkpoint/data/split provenance, team name, OpenReview ID, and Synapse
      submission ID, plus every PASS gate — before the human authorizes each exact push/submit.
