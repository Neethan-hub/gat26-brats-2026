# Data provenance — GAT-26 (BraTS 2026 Task 3 / BraTS-GoAT)

This is the **public, sanitized** provenance statement for the GAT-26 source release. Every claim
below is drawn from the project's committed evidence. Statements that cannot be verified from that
evidence are not made here.

This document deliberately contains **no** credentials, tokens, endpoints, network addresses,
compute-resource identifiers, infrastructure paths, real case identifiers, split membership,
dataset filenames, per-case metrics, or private checksums.

---

## 1. What the model was trained on

- The submitted model was trained using **only the official BraTS-GoAT training distribution supplied
  for Task 3**, obtained through the 2026 BraTS-GoAT subchallenge under the organizers' access terms.
- **No external, independently acquired training dataset was used.** No dataset was acquired
  separately from another BraTS task, from an earlier BraTS release, from an institution, or from any
  other external public source.
- The organizers may compose the GoAT distribution from cohorts associated with relevant BraTS tasks.
  **This project makes no claim to the contrary**; the statement above is about **what this project
  acquired and trained on**, not about how the organizers assembled their own release.
- **At the time model development, model selection, and inference-policy freezing were completed, no
  organizer validation data and no test data had been used** — not for training, not for tuning, not
  for model or policy selection, and not for any post-hoc adjustment.
- Validation and test ground-truth labels are **withheld by the organizers** and were never available
  to this project. Any later authorized access to validation **images**, obtained solely to produce
  predictions for official evaluation, **does not change, tune, retrain, or reopen the frozen model**:
  the architecture, the trained weights, and the inference policy are fixed by the completed
  development described here.

**Corpus as audited:** 1,351 labeled cases, each with four co-registered MRI modalities
(T1n, T1c, T2w, T2f) and one segmentation. All cases audited complete and geometry-valid, with the
label set `{0, 1, 2, 3}` (`0` background, `1` necrotic core, `2` peritumoral edema, `3` enhancing
tumor) and zero label anomalies. The evaluated regions are the standard nested triple
`ET = [3]`, `TC = [1, 3]`, `WT = [1, 2, 3]`.

**Cross-validation design:** a deterministic group-level five-fold split with a fixed, recorded
seed, frozen before any training and never modified. Fold sizes are `[271, 270, 270, 270, 270]`, and
every fold contains ET-absent and TC-absent cases so the empty-region edge cases are exercised. The
split was reproducible: two independent runs produced byte-identical assignments. Concrete split
membership is **not** published here.

## 2. Model initialization

- **All trainable parameters began from a recorded random initialization.**
- **No externally trained pretrained, foundation, or self-supervised checkpoint was used**, and no
  weights were carried in from any source outside this project.
- Each fold was trained independently, with no cross-fold resume and no warm starting.
- The pipeline treats an unexpected weight file as a release-blocking error rather than loading it.

This is supported by the project's committed pre-training authorization audit, its per-fold
completion audits, and its five-fold cross-validation report.

## 3. What this repository does and does not contain

**This repository does not redistribute any challenge data or derived artifact.** Specifically, it
contains **no**:

- MRI images or any imaging payload;
- ground-truth labels or segmentation masks;
- model checkpoints or trained weights;
- predictions or inference outputs;
- dataset archives, preprocessed tensors, or validation data;
- per-case metrics, case identifiers, or split membership;
- protected challenge data of any kind.

What it does contain is **source code and documentation**: the inference runner and its release
container definition, the training/evaluation and audit tooling, the experiment configuration, the
test suite, and the paper source.

## 4. Obtaining the data yourself

The challenge data is **not** available from this repository and is **not** ours to distribute.

To reproduce this work you must **obtain the BraTS-GoAT data independently, directly from the
challenge organizers, under their own registration, eligibility, and data-usage terms.** That
normally requires registering for the challenge, agreeing to the data-usage agreement, and meeting
any access-certification requirements the organizers impose. Access is granted by the organizers,
not by this project, and nothing in this repository grants, implies, or substitutes for it.

Once you hold the data under your own access grant, the code here expects the standard BraTS-GoAT
layout: one folder per case, each containing the four modality volumes and (for training data) the
segmentation, with case folder names ending in the case identifier used by the challenge.

## 5. Attribution required by the organizers

Work using this data must acknowledge the challenge as the organizers require:

> Data used in this publication were obtained as part of the Challenge project through Synapse ID
> (syn74274097).

See `paper/references.bib` for the citation set the challenge rules mandate.

## 6. Scope and honesty notes

- The cross-validation numbers reported in the paper are **out-of-fold cross-validation results on
  training data**. They are **not** validation-leaderboard placements and **not** hidden-test
  results.
- Per-case cohort labels (tumor type / acquisition source) were **not** reliably derivable from the
  released metadata and were never inferred or guessed, so no cohort-stratified analysis is claimed.
- Group-level splitting mitigates but cannot provably eliminate residual same-subject leakage; this
  limitation is stated rather than assumed away.
- No pseudo-labeling, transductive adaptation, or leaderboard-driven tuning was performed.
