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

**Both NSD tolerances, equally.** The selective "about one quarter retained" headline is withdrawn.
The paper reports ΔU at τ=1 and τ=0.5 side by side for both subsets, and labels the retention ratios
(28.7 % at τ=1, 59.3 % at τ=0.5) as descriptive, noting neither is the effect size. The official
tolerance was never exposed to participants.

**The decision procedure is now fully specified.** The utility equation, its equal weighting, the
aggregation level, the common-support rule, empty-region and non-finite handling, the subject-level
bootstrap unit, seed, 10,000 resamples, percentile interval and baseline-wins-ties rule are all
stated. A separate paragraph explains the fail-closed rule: a candidate whose expected evaluation is
missing, errored or membership-mismatched is ineligible, and common support never conceals an
incomplete candidate execution.

**Margins are labelled operational.** The four frozen noninferiority margins are given with their
rationale and identified explicitly as operational, not externally validated clinical thresholds.

**The architecture–metric mismatch is reported honestly, and it is not favourable.** Architecture
selection used DSC/HD95 while the challenge ranks DSC/NSD. We searched the committed record and
rescored the same fold-0 predictions under DSC/NSD at τ=0.5. **The point ordering reverses**:
ResEnc-L is better on four of six official components. The reversal is not robust — the interval
includes a tie — so no architecture review was triggered and no retraining was performed. The
diagnostic was never repeated at τ=1. The paper therefore no longer claims the conclusion is
unchanged under the ranking metric, and records the mismatch as an open limitation.

**Per-component numbers are published.** The main paper carries a compact table of individual
ET/TC/WT DSC and NSD deltas for both subsets at both tolerances. The complete 18- and 23-check
decision matrices, per-component means and the lesion-safety detail are in [`evidence/`](evidence/).

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
