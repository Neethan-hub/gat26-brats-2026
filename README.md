# GAT-26 — BraTS 2026 Task 3 (BraTS-GoAT)

**Generalization-Aware Tri-region 3D nnU-Net ensemble for pre-operative, cross-tumor brain tumor
segmentation.** This repository holds the **source code, configuration, tests, and paper source**
for the GAT-26 submission to the BraTS 2026 Generalizability Across Tumors (BraTS-GoAT) subchallenge.

GAT-26 is a competition system and protocol built on the official nnU-Net v2.8.1
`ResidualEncoderUNet` with region-based training. **It is not a claim of a novel backbone.**

## Licence

**This project's own source code is licensed under the Apache License, Version 2.0.**
The complete licence text is in [`LICENSE`](LICENSE).

```
SPDX-License-Identifier: Apache-2.0
```

Two scoping points, stated plainly:

- The Apache-2.0 grant covers **the code in this repository**. It does **not** extend to the
  third-party dependencies the container installs at build time, each of which remains under its own
  upstream licence (see [Dependencies](#dependencies)).
- It does **not** extend to the BraTS challenge data, which this repository does not contain and
  does not redistribute. Challenge data is governed solely by the organizers' own terms — see
  [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).

## What this repository is not

It contains **no** MRI images, **no** ground-truth labels, **no** model checkpoints or trained
weights, **no** predictions, **no** dataset archives, and **no** case identifiers, per-case metrics,
or split membership. To reproduce this work you must obtain the challenge data **independently from
the organizers under their own access terms** — see [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).

## Method summary

- **Backbone.** Official nnU-Net v2.8.1 `ResidualEncoderUNet` (ResEnc-M preset), `3d_fullres`,
  region-based training predicting ET/TC/WT directly. All parameters start from a recorded **random
  initialization**; no pretrained, foundation, or self-supervised weights are used.
- **Data.** BraTS-GoAT training corpus only: 1,351 labeled cases, four modalities
  (T1n, T1c, T2w, T2f), labels `{0,1,2,3}`, regions `ET=[3]`, `TC=[1,3]`, `WT=[1,2,3]`.
- **Split.** Deterministic group-level five-fold split with a fixed recorded seed, frozen before any
  training; fold sizes `[271, 270, 270, 270, 270]`; byte-identical across independent runs.
- **Inference (frozen policy).** Sequential five-checkpoint mean-probability ensemble — one fold
  model resident at a time, so peak memory is that of a single model — with a fixed `0.5` threshold,
  tile step `0.5`, Gaussian weighting, **no** test-time augmentation, **no** connected-component
  filtering, **no** enhancing-tumor cleanup, and a hierarchy-safe reconstruction enforcing
  `WT ⊇ TC ⊇ ET`. Outputs restore the exact source geometry and are integer-labeled `{0,1,2,3}`.
- **Discipline.** The inference decisions we audited — checkpoint choice, mirroring test-time
  augmentation, weight averaging and connected-component post-processing — were each placed behind a
  candidate, utility and gate set **commit-frozen before the specified comparison** was computed.
  This is not a claim that *every* tunable in the pipeline was audited: the audited set is bounded
  and listed in the paper, and thresholds, tile overlap and ensemble weighting were fixed by default
  rather than selected. Three audits ran; all three **retained the baseline**. A same-corpus policy
  holdout limits direct post-result tuning of a given comparison, but it is not nested validation and
  not an external cohort.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/release_infer.py` | The release inference runner — the entrypoint of the submission container |
| `scripts/release_build_preflight.py` | Deterministic build-context preflight (no Docker required) |
| `configs/release/` | `Dockerfile`, pinned `requirements.lock.txt`, and the release runbook |
| `scripts/` | Split construction, selection policy, training/evaluation drivers, audit tooling |
| `configs/` | Experiment and selection-policy configuration |
| `tests/` | Test suite for the above |
| `preflight/` | Static architecture/release contract suite |
| `paper/` | LaTeX source for the camera-ready paper (the compiled PDF is not committed) |
| `evidence/` | Sanitized aggregate policy-audit evidence and the exact official validation scores |
| `docs/` | Method and rule documentation |

## Container contract

The submission container is Linux/AMD64 and runs with **zero network access**. Every dependency and
every weight is baked at build time; the entrypoint never downloads anything.

- `/input` is mounted **read-only**, one folder per case.
- `/output` is **flat** — exactly one `.nii.gz` per case, no sub-folders.
- **Output filenames preserve the complete input case-folder basename**, byte-for-byte:
  `<complete basename>` → `<complete basename>.nii.gz`. The basename is treated as an opaque
  identifier — no case identifier is parsed out of it, no trailing-digit count is required, and no
  cohort prefix is hardcoded. Under this rule the organizers' published placeholder input folder
  `BraTS-MET-12345-100/` — an example in their instructions, not a real case — maps to the
  example output `BraTS-MET-12345-100.nii.gz`; a folder with a three-digit, seven-digit or
  non-numeric ending is handled the same way.
- Input validation **fails closed** before any output is written: missing, duplicate, unknown or
  unreadable modalities are rejected, as are case-folder names that are genuinely unsafe or
  structurally invalid (empty or whitespace-only, `.`, `..` or any hidden leading-dot entry, an
  embedded path separator, NUL or another control character, a symlink whose target escapes the
  input root, or two folders that would collide on one output name). A rejected run produces
  **zero partial output**.
- The container expects a **fresh writable `/output`**, which is what the official execution contract
  supplies and what our qualification testing uses. Overwriting a pre-existing output file is **not**
  a rejected condition: measured behaviour of the submitted runner is that it proceeds. No
  output-collision guarantee is claimed.
- The runner refuses to start unless exactly **five distinct** fold checkpoints are present.

Weights and dataset/plans JSON are supplied through a **private build context** and are **not** in
this repository. Building the image therefore requires checkpoints you have trained yourself.

## Running the tests

Every test file in `tests/` executes from a clean clone of this repository:

```
python3 tests/test_release_infer.py        # or run them all:
for t in tests/test_*.py; do python3 "$t"; done
```

They need only the pinned Python dependencies — no GPU, no challenge data, no model checkpoints.

**Two narrowly scoped subgroups are intentionally skipped here**, because they assert against
internal evidence records that this project does not redistribute:

| Test | Skipped subgroup | Why |
|---|---|---|
| `tests/test_paper_scaffold.py` | `SKIP_PUBLIC_EXPORT_PRIVATE_EVIDENCE` — the numeric-claim-vs-evidence comparisons | They read the private fold-0 aggregate evaluation summaries |
| `tests/test_g77_official_metric.py` | `SKIP_PUBLIC_EXPORT_PRIVATE_GOVERNANCE` — the current-state and persistence-audit checks | They read the private internal governance record |

Everything else in those two files still runs here: the LNCS structure, bibliography, placeholder,
sanitization, publication-state, release-state, page-count and unsupported-claim guards; and the
scientific, configuration, candidate-restriction, metric-definition, fold-isolation, bootstrap and
historical-policy checks. Nothing is stubbed, hardcoded, or approximated in place of a skipped
check — the subgroup is simply not run.

The skip is deliberately hard to reach: it activates only when this tree's `EXPORT_MANIFEST.json`
declares the sanitized Apache-2.0 export *and* the corresponding private files are genuinely
absent. In the development repository those files are mandatory, and the complete versions of both
subgroups run there on every commit.

## Dependencies

Dependencies are **installed from their upstream sources at build time and are not vendored into
this repository.** Each remains under its own licence; the Apache-2.0 grant in `LICENSE` applies
only to this project's own code.

The pinned runtime set (see `configs/release/requirements.lock.txt`, plus `torch` pinned in the
`Dockerfile`) is `nnunetv2`, `acvl_utils`, `dynamic_network_architectures`, `batchgenerators`,
`batchgeneratorsv2`, `SimpleITK` (Apache-2.0); `einops`, `nibabel` (MIT); and `numpy`, `scipy`,
`scikit-image`, `scikit-learn`, `pandas`, `imagecodecs`, `tifffile`, `torch` (BSD-3-Clause). All are
permissive and compatible with distributing this project's code under Apache-2.0.

## Status

- **Docker submission: made, corrected, and resubmitted.** A frozen release container was submitted
  once to the Task-3 Docker queue. Image identities are recorded with the challenge submissions
  themselves and are deliberately not published here. **The organizers' execution of that first
  image failed**: its runner required the input case-folder basename to end in exactly five digits,
  and the hidden test folders end in a three-digit run, so it aborted on the naming check before
  writing any prediction. It produced **no** prediction, metric or ranking evidence. A corrected
  image was then rebuilt from the same five checkpoints and the same frozen inference policy — with
  no scientific change, only the naming repair — and **that corrected image was submitted**. Its
  naming contract is the one described under **Container contract**: the output name is the complete
  input-folder basename.
- **Outcome of the corrected submission: unknown.** The queue recorded the corrected image as
  received. Receipt is not execution: we have **no** evidence that the organizers executed it
  successfully, **no** hidden-test performance, and **no** rank. Nothing in this repository or the
  paper claims otherwise, and no such claim should be inferred from the submission having been
  accepted into the queue.
- **Official validation performance: known.** The frozen released ensemble was submitted once to the
  Task-3 validation-prediction queue and scored DSC 0.772 / 0.824 / 0.879 and NSD 0.540 / 0.500 /
  0.483 for ET / TC / WT. The platform exposed **no rank**, and none is inferred. HD95 is diagnostic
  only; the organizers did not expose the surface tolerance behind the returned NSD values, so none is
  attached to them. Full qualifications are in the paper.
- **Hidden-test performance and final rank: unknown.** The organizers score containers after the
  submission deadline. No hidden-test number exists, and none is reported or projected here.
- **Official target GPU: the corrected image was never measured on it.** We hold **no** A10G
  measurement of the corrected, resubmitted image.
  [`A10G_QUALIFICATION_SUPERSEDED_IMAGE.md`](A10G_QUALIFICATION_SUPERSEDED_IMAGE.md) records an A10G
  exercise of the **earlier, superseded pre-correction image only**, on synthetic fixtures whose
  folder names were fixed-width — so it never exercised the variable-length folder-name condition
  that caused the organizer failure and that the correction addresses. It must not be read as
  qualifying the corrected image. The strongest runtime evidence for the release policy is a
  full-cohort run on a single NVIDIA A40: 451/451 cases, exit code 0, zero inference errors,
  2.48 GiB peak reserved VRAM, 3 h 35 m.
- Internal cross-validation figures were produced with eightfold mirroring test-time augmentation
  while the released container uses none, so internal and official figures are **not** comparable.
  Cross-validation results must never be read as a leaderboard placement.
- This repository does **not** reproduce training on its own: the challenge data must be obtained
  separately from the organizers, and the trained weights are not redistributed here.

## Citation

See `paper/references.bib` for the citation set the challenge rules mandate, and
[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) §5 for the acknowledgement sentence the organizers
require.
