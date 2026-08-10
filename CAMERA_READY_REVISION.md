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
governance supports (no invented IRB approval, exemption or consent), a competing-interest
statement, and a truthful generative-AI disclosure.

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
