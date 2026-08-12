# Camera-ready revision record

The BraTS-GoAT 2026 submission was conditionally accepted subject to a revision addressing all
reviewer comments. This records what changed in the camera-ready version and, where a request was
not adopted, why. It is deliberately concise; the paper itself carries the science.

## Substantive changes

**A research question, and conclusions proportional to the evidence.** The paper now opens with one
explicit question — whether a commit-frozen, multi-gate policy audit can prevent adoption of an
inference modification whose aggregate development-subset benefit does not transfer to a same-corpus
policy-selection holdout — answered as a bounded case study on one system, one corpus and one split.
Three transferable conclusions are stated, and the paper says plainly that it does **not** show the
procedure is preferable to nested validation.

**Terminology.** "Calibration" as the name of a data subset is gone throughout the published text;
those folds are the *development subset*. No probability calibration was ever assessed and the paper
no longer implies otherwise. "Pre-registered" is replaced by "commit-frozen before the specified
comparison" — there was no external time-stamped preregistration, and the paper no longer suggests
one. Historical filenames retain their original names.

**Both NSD tolerances, correctly ranked in authority.** The selective "about one quarter retained"
headline is withdrawn, and both tolerances are reported. But they are not equally official: the
organizers confirmed that the final ranking uses DSC and NSD with NSD at **τ=1**, and excludes HD95.
τ=1 is therefore the official-ranking-aligned analysis and τ=0.5 is a prespecified sensitivity
analysis, retained because the reviewers asked for both. The earlier claim that the final-ranking
tolerance "was never exposed" was false and is withdrawn. What remains true is narrower: the
organizers did not disclose which tolerance produced the returned participant-visible validation
scores, so none is attached to those particular numbers. Retention ratios (28.7 % at τ=1, 59.3 % at
τ=0.5) are descriptive; neither is an effect size.

**An unsupported bootstrap claim is removed.** The committed record contains no development-subset
bootstrap at τ=0.5. The earlier table reported a positive fraction of 1.000 there; that entry is now
"n/a", the interval is unavailable, and "every bootstrap resample was positive" is scoped explicitly
to τ=1. No value was computed after the fact to fill the gap.

**The decision procedure is now fully specified, and its central equation is corrected.** An earlier
camera-ready draft defined the Audit C utility as a mean of *fractional ranks*. That was wrong and is
withdrawn: it is arithmetically incompatible with the reported values, since the candidate improved
all six components in every analysis and a rank utility would therefore give exactly +1.000
everywhere, against recorded deltas of order 1e-3. The committed evidence shows the utility is the
unweighted arithmetic mean of the six **raw** component means on common subject support, so
delta-U is identically the mean of the six component deltas. The fold-0 architecture screen keeps its
own, genuinely rank-based statistic *R*; *R* and *U_tau* are different objects and are never
combined. The equation, equal weighting, aggregation level, common-support rule, empty-region and
non-finite handling, the subject-level bootstrap unit, seed, 10,000 resamples, percentile interval
and baseline-wins-ties rule are all stated, and a regression test recomputes every published delta-U
from its six component deltas. A separate paragraph explains the fail-closed rule: a candidate whose expected evaluation is
missing, errored or membership-mismatched is ineligible, and common support never conceals an
incomplete candidate execution.

**Margins are labelled operational.** The four frozen noninferiority margins are given with their
rationale and identified explicitly as operational, not externally validated clinical thresholds.

**The architecture–metric mismatch is reported honestly, and it is not favourable.** Architecture
selection used DSC/HD95 while the challenge ranks DSC/NSD. We searched the committed record and
rescored the same fold-0 predictions under DSC/NSD at τ=0.5. **The point ordering reverses**:
ResEnc-L is better on four of six official components. The reversal is not robust — the interval
includes zero, so no rank advantage is established — and no architecture review was triggered. The
diagnostic was never repeated at τ=1. The paper therefore no longer claims the conclusion is
unchanged under the ranking metric, and records the mismatch as an open limitation.

**Per-component numbers are published.** The individual ET/TC/WT DSC and NSD means and deltas live in
the **supplement**, one table per subset and tolerance, each with its own bootstrap summary and
per-fold values printed beside it. The complete 18- and 23-check decision matrices, the lesion-safety
detail and the machine-readable aggregates are in [`evidence/`](evidence/).

**Precision.** Machine-precision values are gone from the paper; performance is rounded to four
decimals. The exact official scores are preserved in machine-readable form in
[`evidence/official_validation_scores.json`](evidence/official_validation_scores.json).

**The official-validation gap is no longer attributed.** The paper reports the gap as a measured
fact, identifies the one component it can (the inference-path difference, ≈0.002 DSC), and then
explicitly declines to attribute the remainder — including to cross-tumor transfer, which the
earlier text asserted. There are no cohort labels, no subgroup results and no controlled comparison,
so domain shift, case composition, ET heterogeneity, adaptive reuse and scorer differences cannot be
separated.

**Single model versus deployed ensemble.** The paper states that the mirroring audit, being a
single-checkpoint out-of-fold comparison, does not determine how mirroring would affect the deployed
five-checkpoint ensemble — which could mitigate, preserve or amplify the lesion-miss behaviour.

**Compute.** Only measured runtimes and verified limits are reported. The paper makes no feasibility
claim for full-ensemble mirroring in either direction.

**Container and release contract.** Low-level container mechanics are reduced in the paper and the
operational detail lives here. The output-name contract is corrected: each output file name is the
**complete validated basename** of its input case folder, with no truncation to a fixed-width
identifier. The A10G narrative is corrected — see below.

**Failure analysis.** A qualitative account of *what* failed is given from aggregate evidence only:
lesion false-positive and false-negative movement, zero-DSC region counts, fold-sign heterogeneity
and miss-rate behaviour. No patient-level example, case identifier or per-case metric is published.

**Limitations.** A candid list now covers the same-corpus holdout, adaptive reuse and multiplicity,
the absent external cohort, the architecture–metric mismatch, single-model versus deployed ensemble,
the one absent row in the participant-visible per-case file, the lack of reliable cohort labels, the
absence of hidden-test evidence, the absence of any A10G qualification of the corrected image, and
the absence of any demonstration that the procedure beats nested validation.

**Disclosures.** The paper adds a data-use and ethics statement scoped strictly to what challenge
governance supports (no invented IRB approval, exemption or consent) and a competing-interest
statement.

## Docker and A10G — corrected narrative

A first release container was submitted and **failed in the organizers' execution**: its runner
required the input case-folder basename to end in exactly five digits, and the hidden folders end in
a three-digit run. A corrected image was rebuilt from the same five checkpoints and the same frozen
inference policy — no scientific change, only the naming repair — and was submitted.

The queue recorded the corrected image as received. **Receipt is not execution.** There is no
evidence of successful organizer execution, no hidden-test performance and no rank, and none is
claimed. The historical A10G exercise applies to the **superseded pre-correction image only** and is
retained as [`A10G_QUALIFICATION_SUPERSEDED_IMAGE.md`](A10G_QUALIFICATION_SUPERSEDED_IMAGE.md); its
synthetic fixtures used fixed-width folder names and never exercised the condition that caused the
failure. The corrected image has never been measured on an A10G.

## Requests not adopted, and why

- **New augmentation, domain-adaptation or DA5 training.** Declined. It would change the studied
  system, falls outside a camera-ready revision, and is not needed to correct the scientific account.
  The already-committed bounded 40-epoch augmentation screen is reported accurately, including that
  being bounded at 40 epochs it cannot establish convergence behaviour.
- **A new five-model TTA experiment.** Declined. Any new real-data inference would create
  post-review evidence and could change the submitted system. The ensemble question is instead
  reported as an unresolved limitation, without speculating that ensembling would repair lesion
  misses.
- **Attributing the official-validation gap to tumor type or ET heterogeneity.** Declined. No cohort
  labels and no controlled evidence exist.
- **Patient-level qualitative examples.** Declined. Failure is characterized in aggregate instead.
- **Framing the procedure as preferable to standard or nested validation.** Declined. It is
  presented as a transparent case study, with nested or external validation stated as preferable
  where feasible.
- **Reviewer phrasing not supported by evidence** — "flawless code", "independent confirmation",
  "pre-registered", "~0.002 DSC drop across all regions" as a universal claim, and "seven unused
  hours easily allow TTA" — is not adopted.


## G93 — final consistency corrections

The workflow figure is now the **general** decision procedure, not a picture of what happened: a
candidate failing the development subset is rejected, one that passes reaches the policy-selection
holdout, and there it is either **adopted** or rejected. Both outcomes feed a neutral *selected
release policy* node, and only that node reaches official validation. The figure no longer implies
that the baseline is always retained; the caption states separately that in this study all three
audits ended in the reject branch.

The architecture rank statistic is defined completely in the main paper: `R` is the mean fractional
award rank over the six screened components, lower is better, exact component-level ties take the
average of the tied positions, and the reported difference is ΔR = R(L) − R(M) with a paired
subject-level bootstrap (seed 21072026, 10,000 resamples, 95 % percentile interval). The earlier text
labelled a *difference* as `R` and paired it with the interval of the opposite-signed quantity. This
correction reached the **main paper only**; the supplement still carried the old wording, which G94
below repairs. G93 described the fix as consistent throughout, and that description was wrong.

"Six ranked components" is replaced by "six DSC/NSD components" wherever the audit utility is meant,
so only the genuinely rank-based architecture statistic is described with the word *rank*. Every
"interval did not exclude a tie" becomes "the interval included zero". The bootstrap fraction is
described as the *fraction of paired bootstrap resamples with a positive difference*, never as a
probability; where legacy machine field names such as `bootstrap_probability` are reproduced verbatim
so tables can be matched to the published JSON, they are labelled as identifiers rather than claims.

The inventory table no longer says that everything listed was "tried": it distinguishes executed
candidates from proposals screened out before execution.

Defensive framing is gone from the official-validation discussion, which now states the measurement
and the limits of attribution plainly. The obsolete initial-submission checklist has been removed
from the public source, and the paper build documentation no longer claims cross-version byte
identity or sole Springer copyright over `splncs04.bst`, which is LPPL-licensed upstream work by
Patrick W. Daly adapted by Maurizio Patrignani.

## G94 — statistical-consistency and packaging correction

**The architecture sign error is now fixed in the supplement, at its generating source.** The
supplement is produced by a script from the committed artifacts; the script — not just its output —
was corrected, and the committed `supplement.tex` is exactly what a fresh run of that script emits.
The supplement previously read "the rank gain is R = +0.333 with a 95 % interval [−1.000, +0.667]",
which named a *difference* `R` and then attached to it the interval of the opposite-signed quantity.
It now reads ΔR = R(L) − R(M) = −0.333, 95 % percentile interval [−1.000, +0.667], which includes
zero. The generating implementation is authoritative for this: it computes the bootstrap difference
as `rank_L − rank_M`, so negative values favour ResEnc-L. The committed record also carries the
opposite-signed companion field `rank_gain_L_over_M = +0.333`; the two describe the same result and
only one convention is now used in the published text.

**Two different tie conventions are separated.** The supplement said ties were "resolved to the
smaller model", which conflated two unrelated rules. *Within* `R`, an exact component-level metric
tie assigns both models the average of the tied positions, so a tied component contributes equally to
each and can favour neither. *Separately*, the frozen selection rule resolves a tied or unmet
advancement criterion in favour of ResEnc-M, the cheaper baseline plan. The first is rank
arithmetic; the second is a decision rule. Both are stated, and they are no longer merged into one
sentence.

**Lesion counters are labelled by the space they are counted in.** The lesion table previously
called a prediction-space counter "true positives", although the committed record labels it
diagnostic only. No value changed. The rows now read: predicted components overlapping a reference
component (a diagnostic counter), predicted components with no reference overlap, reference
components with no predicted overlap, and the reference-component total. The supplement states that
prediction-space overlaps and reference-space misses are not complementary under component matching,
so their sum need not equal the reference-component total — and in these data it does not: on the
holdout the two counters sum to 5,424 (baseline) and 5,404 (candidate) against 5,445 reference
components. The safety gate is FN_ref / N_ref, the missed fraction of reference components, not the
legacy recall derived from the prediction-space counter.

**Build claims now match the logs.** The G93 build documentation claimed zero underfull boxes; the
supplement in fact produced six underfull `\hbox` warnings in the experiment-inventory table, and the
G93 report additionally misdescribed them as vboxes. They were removed by reflowing that table —
ragged-right column text and a wider text column — with no change to margins, font sizes, page size
or vertical spacing, and no negative spacing. The supplement's remaining hyperref PDF-string warning
came from the line break in its title, not from mathematics in a section heading; it is resolved with
`\texorpdfstring`, leaving the printed title unchanged. The current logs show zero overfull boxes,
zero underfull boxes and zero hyperref warnings for both documents.

**PDF active-content claims are stated precisely.** Earlier reports said the PDFs contain "no
actions". That is not true of any hyperref output and was never true here. Both PDFs contain ordinary
internal GoTo link actions and external URI link actions. They contain no JavaScript, no Launch,
SubmitForm, Named or Rendition actions, no embedded files and no attachments. Structural validity
under `qpdf --check` is reported separately from that inventory: passing the structural check says
nothing about which actions a file contains.

**Code URL.** The paper now cites the immutable tag `brats-goat-2026-camera-ready-21-r4`. Tags r1, r2
and r3 are unchanged and remain resolvable.

## G95 — completing the architecture-selection rule

**The advancement rule is now stated in full, and it is the rule the code implements.** The earlier
text said only that ResEnc-L "is preferred if the interval excludes zero and every declared
noninferiority gate passes". That was incomplete in four ways, all of which mattered. ResEnc-L could
advance from fold 0 to fold-1 confirmation only if the executable rank-gain threshold on
R(M) − R(L) was met — nominally 1/6, one component of six, stored as the constant
`MEANINGFUL_RANK_GAIN` in the committed policy — the paired-bootstrap 95 % percentile interval for
ΔR = R(L) − R(M) lay entirely below zero, and every frozen auxiliary gate was supplied and passed, a
gate with no input supplied counting as a failure rather than a pass.
Expansion required the same rule to hold on fold 1; otherwise ResEnc-M was retained.

Two points the earlier wording obscured. The rank gain is written R(M) − R(L) and the reported
bootstrap difference is ΔR = R(L) − R(M), so the same result carries opposite signs in the two
quantities. And passing fold 0 never selected or "preferred" ResEnc-L: fold 0 could only trigger
fold-1 confirmation, and nothing but a confirming fold 1 could expand it.

The new regression tests check the published prose against the executable policy rather than against
a fixed sentence: the threshold string is derived from `MEANINGFUL_RANK_GAIN` at test time, and the
decision behaviour is exercised directly — a fold-0 pass returns only `confirm_L_on_fold1`, two
confirming folds are required before expansion, and removing any one auxiliary gate input returns
`select_M`.

**Supplementary Table S11.** Its two numeric column headers were set on one line and read as running
together. They are now compact stacked headers, Baseline over C0 and Candidate over M8, with a wider
gap between the two numeric columns. No value, caption or statement changed. The fix was made in the
generator, and the generated file remains byte-for-byte what a fresh run of that generator emits.

**Float placement.** The supplement now sets standard LaTeX float-placement parameters
(`topnumber`, `totalnumber`, `topfraction`, `bottomfraction`, `textfraction`, `floatpagefraction`).
These govern only where a float may be placed. No margin, font size, page size or spacing changed,
and no negative spacing was introduced. The effect is that each table sits on or beside the page
that discusses it instead of accumulating into half-empty float pages, and the build reports zero
overfull and zero underfull boxes.

**Licence-form signing guidance.** Earlier instructions asserted that Springer does not accept
digital or electronic signatures. The supplied organizer material does not say that: it requires the
corresponding author to sign the last page and is silent on modality. The instructions now state the
requirement accurately and recommend wet ink as the conservative path unless the organizers confirm
that an electronic signature is acceptable. The claim that non-Word converters necessarily
repaginate the form is likewise withdrawn; pagination can be converter-dependent, and confirming the
six-page layout in Word remains the owner's check.

**Publication snapshot.** The public source is additionally published as
`brats-goat-2026-camera-ready-21-r5`, a parentless snapshot commit whose tree is byte-identical to
public `main`. The paper cites r5. Tags r1 through r4 are unchanged and remain resolvable; the
commit metadata recorded in them is historical and has not been rewritten.

## G96 — public-export sanitization and the executable threshold

**Submission-infrastructure identifiers are removed from the public export.** The frozen
preregistration configurations record two organizer evaluation-queue identifiers. They are
submission-infrastructure values: no scientific parameter, gate, threshold, result or chronology
depends on them, and they do not belong in a public source release. In the exported copies of
`configs/g82_preregistration.json`, `g83_dense_overlap_preregistration.json`,
`g84_release_tta_preregistration.json` and `g85_confirmation_preregistration.json` those fields are
`null`, and each redacted object carries a note saying the removal was intentional, that the exact
frozen private originals are preserved unchanged, and that no scientific content was altered.

The redaction is performed by the exporter, not by editing the sources: the private configurations
remain byte-for-byte frozen. The three stage tests that assert those fields are themselves frozen
prior-stage artifacts, so they are likewise not edited in place; the exporter substitutes their
queue assertion for one that verifies the redaction actually happened, and it fails closed if the
substitution does not match exactly once. `EXPORT_MANIFEST.json` lists every redacted file.

This applies to the current tree from this revision onward. Earlier published tags are immutable and
have not been rewritten; no claim is made that they are free of these identifiers.

**The rank-gain threshold is described as what the code executes.** The previous text stated the
exact real-number condition R(M) − R(L) ≥ 1/6. The frozen policy actually evaluates the binary64
comparison `rank_gain >= MEANINGFUL_RANK_GAIN`, and the two are not equivalent at the boundary: the
stored threshold is the double nearest 1/6, 0.16666666666666666, while a rank configuration that is
exactly 1/6 in rational arithmetic computes to about 0.16666666666666652 and is rejected. The
manuscript now says the executable threshold was met, nominally 1/6; the supplement states the
comparison and both values in full.

This is a disclosure, not a correction of any result. The recorded architecture decision was nowhere
near the boundary — the fold-0 rank gain was −0.333 against a required +1/6, and no other declared
condition was met either. `scripts/g45_selection_policy.py` was not modified, and no selection
outcome, evidence file or recorded value changed.

**Publication snapshot.** The paper cites `brats-goat-2026-camera-ready-21-r7`, again a parentless
snapshot whose tree is byte-identical to public `main`. Tags r1 through r6 are unchanged and remain
resolvable. `r6` was tagged before a final supplement typesetting fix and is therefore superseded
rather than cited; it was left in place rather than moved, because published tags in this repository
are never rewritten.

## Post-G96 — credits scope and a new publication snapshot

**The credits section was narrowed.** It now carries exactly two statements: the data-use and
ethics statement, under the heading "Data use and ethics", and the competing-interest statement.
That is the whole of the credits. No scientific content changed: no result, number, figure, table,
method statement, limitation or reference was touched, and the manuscript's claims and evidence are
exactly as they were.

**Pagination.** The main paper is still 12 physical pages and the supplement is unchanged at 11.
Manuscript content still ends on physical page 10; the reference list now begins lower on page 10
rather than at the top of page 11, and still runs to page 12. No margin, font size, page size or
spacing was altered, and no negative spacing was introduced.

**Publication snapshot.** The paper cites `brats-goat-2026-camera-ready-21-r8`, again a parentless
snapshot whose tree is byte-identical to public `main`. Tags r1 through r7 are unchanged and remain
resolvable. `r7` is superseded rather than cited; as with `r6`, it was left in place rather than
moved, because published tags in this repository are never rewritten.

## Post-r8 — vendor-neutral operational documentation

**Historical operational documentation was rewritten into vendor-neutral wording.** Several
passages in the release runbook, the pre-training audit memorandum and one supervisor script header
described the operator tooling used to drive the work by product name. Those passages now describe
the same facts without naming any product. Every hardware, cost, safety, scheduling and execution
claim they carry is unchanged, and the sanitized export now fails closed if a named software-system
reference appears anywhere outside two adjudicated occurrences: the title of a cited third-party
paper, and the detector pattern that enforces that no software system is credited as an author.

No scientific content changed. No result, number, figure, table, method statement, limitation or
reference was touched, and the manuscript's claims and evidence are exactly as they were.

**Publication snapshot.** This revision was prepared as the `brats-goat-2026-camera-ready-21-r9`
candidate. That candidate was never tagged or published; see the r10 entry below. Tags r1 through
r8 are unchanged and remain resolvable, and none was moved, because published tags in this
repository are never rewritten.

## r10 — protocol-provenance, scope and editorial corrections

The r9 candidate was withdrawn before publication after an independent local audit of the
camera-ready artifacts. It was never tagged and never pushed; nothing was published under it. This
entry is appended, and no earlier entry has been rewritten.

**Protocol provenance of the mirroring work is now stated correctly.** Audit C improved all six
components on the development subset but failed its frozen lesion false-negative count veto and
stopped there; the baseline was retained. Its candidate-side phases were conditional on advancement
and were never reached, which is an unreached branch of a frozen protocol rather than an omitted
step, and is unrelated to the separately documented official-validation submission of the retained
baseline. The folds-3--4 holdout analysis was a separately commit-frozen follow-up of the same frozen
candidate, designed after the development outcome was seen and frozen before those folds were opened.
It is development-informed rather than globally result-blind, and it is not advancement under Audit
C's failed gate. All of the holdout evidence -- both tolerances, the bootstrap and fold-heterogeneity
findings, the lesion-miss results and the retention of the baseline -- is retained unchanged; only
the provenance narrative changed, and comparisons between the two are labelled descriptive
cross-analyses.

**The headline scope is narrower.** The abstract no longer says the shipped inference policy is the
object under test. It states that candidate inference modifications were executed end-to-end through
the release machinery on excluded-fold predictions. The conclusion no longer says the five-fold
ensemble was evaluated under commit-frozen rules; it says its release policy was selected and
released following them. The single-excluded-fold-checkpoint limitation is retained.

**Corrections of record.** Supplementary Table S12 now bolds the better WT HD95 value, 3.89, rather
than 4.11; lower HD95 is better and neither value nor the four-of-six conclusion changed. Statements
about hidden-test evidence are scoped to what is defensible: no hidden-test performance result or
rank for the corrected image, and no organizer execution log for it. Three frozen C0 details are
added: sliding-window tile step 0.5, Gaussian importance weighting, and SGD weight decay
3e-5. The memory sentence is corrected: only one fold model is resident at a time, so simultaneous
model residency does not scale with the five folds, but process peak is not that of a bare
single-model run because the running accumulator and current probabilities are also held.

**Editorial.** Source comments no longer point at files absent from the archive, an internal stage
label was removed from the bibliography comments, and headings were normalised.

No scientific content changed: no result, number, interval, denominator, threshold, gate outcome,
candidate decision or release decision was touched.

**Publication snapshot.** The paper cites `brats-goat-2026-camera-ready-21-r10`. That tag does not
yet exist; creating it is a separately authorized publication step.

## r11 — pagination, restored qualifications and current-versus-historical documentation

r10 was **never tagged, never pushed and never submitted**. It failed a *local* validation of its
own camera-ready artifacts: the strict page-limit criterion and the licence-form page count. It was
**not** rejected by the venue or by OpenReview, and the paper's conditional acceptance is unaffected.
r11 corrects what that local validation found.

**Erratum against the r10 entry above.** That entry closes with "No scientific content changed."
That wording is too broad and is withdrawn: r10 *did* correct scientific qualifications — the S12
bolding, the hidden-test scoping and the memory sentence. The accurate statement, for r10 and for
r11, is the narrower one used below. Everything else in the r10 entry stands, and no earlier entry
has been altered.

**The page-limit criterion is now met strictly, and was not before.** The requirement is that all
non-reference manuscript content end on physical page 10 and that `References` be the first
substantive text on page 11. Earlier revisions were recorded as satisfying a 10-content-page limit
on the looser reading that `References` merely *appears* on page 11; measured strictly, r10 still
carried 183 body tokens onto page 11. The r11 manuscript carries **zero**. The space was recovered
by condensing genuine repetition — introductory background, workflow narration duplicated with
Fig. 1, out-of-fold procedural enumeration, low-level container and runtime prose, the opening of
the Discussion, the Conclusion, the bounded-inventory paragraph, the architecture-rescoring tail,
the advancement mechanics and the Decision Procedure's connective prose. **No** template, class,
page-size, margin, font, spacing, float, bibliography-format or figure-scale change was made, and no
required content was moved to the supplement.

**Restored and corrected wording.** The abstract now says the development-subset benefit "does not
transfer *consistently*" to a same-corpus policy-selection holdout, and that "the holdout bootstrap
*intervals* spanned zero *across tolerances*"; the Audit C subsection title matches. Ten prose
qualifications that an earlier, rejected r11 patch had weakened are verified present **inside their
required sections**, together with a further twenty-six section-scoped safeguards: the Conclusion
carries "same-corpus policy-selection holdout", "official-ranking-aligned τ=1" and "both intervals
spanned zero"; the Audit C subsection carries the unreached conditional phases, the separate
development-informed G85 provenance and the "*possible* precision–sensitivity trade-off … remains
only a hypothesis" hedge; Official Validation carries "Per-case cohort labels were not derivable",
"we attribute the discrepancy to neither party" and "One component of the difference"; Runtime
Evidence carries the repaired synthetic-fixture antecedent; and the Fig. 1 caption carries all four
provenance qualifiers. The qualitative failure analysis, the acknowledgment, the organizer-mandated
Synapse sentence, the data-use and ethics statement and the disclosure of interests are unchanged.

**Supplement.** Section headings are now consistently title-cased in both the generator and the
generated file, and the generator still reproduces `paper/supplement.tex` exactly. The supplement
remains 11 physical pages; every table, value, caption, qualification and inventory outcome is
unchanged, and Table S12 still bolds the better WT HD95 value, 3.89.

**Current versus historical documentation.** The public tree now says plainly which records are
current contracts and which are dated evidence. Conspicuous historical-snapshot banners were added
to `preflight/README.md`, `preflight/run_static_preflight.py`,
`docs/BraTS_2026_GoAT_Model_Architecture.md`, `docs/BraTS_2026_GAT26_Preflight_Audit.md` and
`configs/release/AWS_A10G_RUNBOOK.md`; a new `configs/README.md` gives the stage and freeze
chronology, records that `g77_official_metric_alignment.json` supersedes the earlier metric
understanding used by the architecture screen, and states the current position — C0, patch size
`[128,160,112]`, opaque basename, DSC and NSD at τ=1 with HD95 diagnostic, and no corrected-image
A10G, execution-log, hidden-test or rank evidence. `configs/release/README.md` now separates the
failed pre-correction image from the corrected submitted image, records that organizer execution
status for the corrected image is unknown, and marks the retained A10G-1/A10G-2 wording as
historical design that did not qualify the corrected image. The top-level README's repository map no
longer presents archived design and audit artifacts as current contracts. **No frozen historical
constant, JSON record or historical result was rewritten** — only its context is now labelled.

**Memory wording.** The overbroad claims in the top-level README and in `scripts/release_infer.py`
("peak memory is that of a single model", "peak VRAM is a single model, not five", "memory-safe")
are replaced by the accurate statement: one fold model is resident at a time, so simultaneous model
residency does not scale with five folds; the running probability accumulator and the current
per-region probabilities are also resident, so total process peak memory is not that of a bare
single-model run.

**Regression tests.** A new fail-closed public-documentation suite pins the memory wording, the
presence of every required historical banner, the current official-metric and basename statements,
the scoping of historical strings to labelled archival records, exclusion of the bridge and any
nested worktree, and the absence of any harness-directory member from the export. It does not ban
legitimate historical DSC/HD95, five-digit, provider or A10G strings inside clearly labelled
archival records.

No numerical result, interval, threshold, gate outcome, candidate decision, model artifact,
prediction or release decision changed.

**Reviewer-requested per-component evidence is now actually present.** The main paper said the
Audit A/B per-component values were in the supplement; they were not. The supplement now carries
them for every executed candidate --- the best-validation checkpoint, recorded-axis mirroring, the
two combined, enhancing-tumour cleanup at all three declared thresholds, and both weight soups ---
as ET/TC/WT DSC and NSD means and candidate-minus-C0 deltas at the official-ranking `τ=1` and at the
`τ=0.5` sensitivity tolerance, alongside the DSC/HD95 pair the audits actually decided on, the
per-policy denominators, the rank gain with its bootstrap interval, and the outcome. The bounded
40-epoch T/DG/TDG screen is reported as the deltas the frozen record actually holds, and the
supplement states plainly that absolute per-component candidate means and HD95 were never computed
for those arms rather than reconstructing them. `evidence/supplement_inputs.json` carries a
`provenance` block mapping every rendered value to its source record and key path, together with an
explicit list of quantities that do not exist. The bounded inventory is now complete, with a status
column separating executed candidates from proposals screened out before execution, and D25 is
recorded unambiguously as preregistered but never executed, with zero predictions.

**A correction the new evidence forced.** Publishing the soup components exposed a false statement
in the r10 text, which said the six components moved the wrong way "monotonically in the amount
mixed in". They did not: on all six components at both tolerances, and on all three HD95
diagnostics, the ordering is C0 > S2 > S1, so the soup mixing **more** of the best-validation
checkpoint is uniformly **better** than the one mixing less. The source audit record's own
parenthetical already said so. The main paper and the supplement now state that the degradation is
consistent in direction but not monotone. No number, interval, threshold, gate outcome, candidate
decision or release decision changed; the main PDF was rebuilt for this wording alone.

**The documented supplement regeneration now works in this repository.** `scripts/g92_build_supplement.py`
read two records under `artifacts/`, which is never exported, so the documented command failed in
any public tree and the committed `supplement.tex` shipped unreproducible. The public-evidence
generator now also emits `evidence/supplement_inputs.json`, a whitelisted aggregate projection at
full float precision, and the generator reads it when the private records are absent, failing closed
with an explicit diagnostic when neither is available. From a tree containing only `evidence/` and
`scripts/` the documented command reproduces the committed file byte-for-byte, and
`tests/test_r11_supplement_regeneration.py` enforces exactly that.

**A test that silently stopped testing.** `tests/test_g83_science.py` read its NSD adapter from a
machine-absolute private path behind a conditional, so on every public export one check left the
tally without any failure and the file still reported all checks passed. The adapter now ships,
sanitized, as `scripts/g79v_tau_nsd_adapter.py`; the test loads it by a repository-relative path and
fails closed if it is missing. Operational machine-absolute paths were removed from
`scripts/g84_eval.py`, `scripts/g85_eval.py`, `scripts/g5_runner.py`,
`scripts/g3_acquire_labeled_archive.py` and two tests; every remaining occurrence in a file that
ships is either inside a banner-marked historical record or an allowlisted rule literal, and a new
fail-closed check pins that.

**Encoding, and what was deliberately left alone.** Text reads and writes now name UTF-8
explicitly: 236 of the 319 call sites were converted. The remaining 83 sit in the 23 files that the
G82/G83/G84 and G79-V immutability guards hold byte-frozen, and they were reverted rather than
touched --- a frozen stage record is evidence of what was executed, and a cosmetic edit would
destroy the proof it exists to provide. Three frozen files were changed anyway, because the r11
correction required it: `tests/test_g83_science.py`, `scripts/g79v_tau_nsd_adapter.py` and
`scripts/g84_eval.py`. Each change is a publication or fail-closed correction, none touches a
recorded result, and each is now enumerated as an authorized exception inside the guard that covers
it, so the guard still fails closed on anything else. Both suites give identical results under a C
locale and a UTF-8 locale.

**Publication snapshot.** `brats-goat-2026-camera-ready-21-r11` is the intended immutable release
name for this revision, and it is the tag the paper cites. Creating it is a separately authorized
publication step. Publication is complete only when the anonymous, unauthenticated URL for that tag
resolves; until it does, the citation must not be recorded as a passing link check. Tags r1 through
r8 are unchanged and remain resolvable. r9 and r10 were never published.
