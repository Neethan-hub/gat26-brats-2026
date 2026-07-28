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
- **Discipline.** Every tunable inference decision was placed behind a **pre-registered**,
  confirmation-gated policy fixed before results were seen. Two independent bounded audits
  (checkpoint / TTA / post-processing, and checkpoint-weight averaging) both **retained the
  baseline**, so the released configuration is pre-registered rather than post-hoc.

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
| `paper/` | LaTeX source for the short paper (the compiled PDF is not committed) |
| `docs/` | Method and rule documentation |

## Container contract

The submission container is Linux/AMD64 and runs with **zero network access**. Every dependency and
every weight is baked at build time; the entrypoint never downloads anything.

- `/input` is mounted **read-only**, one folder per case.
- `/output` is **flat** — exactly one `.nii.gz` per case, no sub-folders, with the output name ending
  in the challenge case identifier.
- Input validation **fails closed** before any output is written: missing, duplicate, unknown, or
  unreadable modalities; invalid or ambiguous case-folder names; and output-name collisions are all
  rejected, and a rejected run produces **zero partial output**.
- The runner refuses to start unless exactly **five distinct** fold checkpoints are present.

Weights and dataset/plans JSON are supplied through a **private build context** and are **not** in
this repository. Building the image therefore requires checkpoints you have trained yourself.

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

The trained models and the frozen inference policy exist. **Not** yet available, and reported as
pending rather than estimated: the final container image identity, its runtime and memory acceptance
on the official target GPU, official validation-leaderboard performance, and hidden-test performance.
Cross-validation results must never be read as a leaderboard placement.

## Citation

See `paper/references.bib` for the citation set the challenge rules mandate, and
[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) §5 for the acknowledgement sentence the organizers
require.
