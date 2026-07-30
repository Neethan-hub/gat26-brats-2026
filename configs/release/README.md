# GAT-26 release container (Task 3 / BraTS-GoAT) — build & test

> **Final status (2026-07-30).** The released container was built from this Dockerfile with the five
> distinct final fold checkpoints and submitted once to the Task-3 Docker queue. The `torchvision`
> pin and the build-time version assertions in the Dockerfile are load-bearing: without them the
> lock install resolves `torchvision` from PyPI and upgrades `torch` off the CUDA-12.8 build the rest
> of the evidence was produced on. The single-checkpoint proxy mode described below is historical and
> was **not** used for the submission. The container expects a fresh writable `/output`; overwriting a
> pre-existing output file is not a rejected condition.

**Selected plan: ResEnc-M** (frozen G5 fold-0 selection). Clean-room, zero-network,
A10G 24 GB target. This directory is committed source only — **no weights, no data**.

## Private build context (NOT in git)
Assemble on the authorized worker under an ignored path, e.g. `build_context/`:

```
build_context/
  requirements.lock.txt                 # copy of configs/release/requirements.lock.txt
  scripts/…                             # release_infer.py + helpers (from the tracked repo)
  plans/nnUNetResEncUNetMPlans.json     # baked M plans   (private)
  plans/dataset.json                    # baked channel order / labels (private)
  weights/fold_0/checkpoint_final.pth   # M checkpoint(s) (private)
  …fold_1..fold_4 after G7 (five DISTINCT M checkpoints)
```

Preliminary candidate (pre-G7): a single M0 checkpoint replicated into five runtime slots via
`--allow-duplicate-proxy` — `runtime_only_duplicate_weight_proxy`, **no_accuracy_claim**,
**not_final_ensemble**. The final runner accepts five genuinely distinct M checkpoints with **no
code change** (drop them into `weights/fold_{0..4}/`).

## Build (host with Docker + NVIDIA runtime)
```
docker build --platform=linux/amd64 -f configs/release/Dockerfile -t gat26-goat-m:CANDIDATE ./build_context
```

## Organizer-equivalent local test
```
docker run --rm --network none --runtime=nvidia --memory=48G --shm-size=16G \
  -v /ABS/INPUT:/input:ro -v /ABS/OUTPUT:/output:rw gat26-goat-m:CANDIDATE
```

## Build-context preflight (run BEFORE `docker build`)
```
python3 scripts/release_build_preflight.py --context ./build_context [--proxy]   # REQUIRE 0 problems
```
Verifies every Dockerfile COPY source + plans/dataset.json + the checkpoint layout (five distinct
for A10G-2, or `--proxy` fold_0-only for A10G-1) are present, and that no Synapse config / SSH
material / credentials / validation archive / raw images / `.git` are in the context.

## Contract (enforced by `scripts/release_infer.py`, fail-closed)
- ResEnc-M only; random-init-trained weights only; no external weights.
- Final acceptance (A10G-2) requires **five DISTINCT-content** fold_0–4 checkpoints (SHA-256 verified
  privately); identical copies fail closed unless `--allow-duplicate-proxy` (A10G-1 smoke) is set.
- Genuine-A10G runs pass `--gpus all` and `-e GAT26_REQUIRE_GPU_NAME="NVIDIA A10G"`; the runner aborts
  before inference unless torch sees exactly one CUDA GPU with that exact name.
- Sequential five-checkpoint mean-probability ensemble (one model resident at a time).
- Frozen inference: threshold 0.5, hierarchy-safe WT/TC/ET, **no TTA**, no CC filtering, no
  presence gate, no learned threshold/weight; primary `checkpoint_final`.
- Strict modality discovery by suffix (order-independent); missing/duplicate/unknown → nonzero
  exit BEFORE any output.
- One flat `.nii.gz` per case, name echoes the input case folder (ends in the case ID); exact
  source geometry restored; integer labels {0,1,2,3}; ET⊆TC⊆WT.
- Deterministic; explicit nonzero exit on invalid input or incomplete output.

**Limits (live-verified 2026-07-23, wiki/639582):** A10G 24 GB, 16 vCPU, 200 GB storage,
`--memory` 48 GiB, `--shm-size` 16 GiB, CUDA ≤ 13.0, **12 h total** inference; **zero network**;
`/input` read-only; `/output` flat. Final deadline **30 Jul 2026 23:59 UTC**.
