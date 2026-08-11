# GAT-26 preflight test package

Version 1.2 (21 July 2026). This package checks inexpensive contracts before
protected-data training. It does **not** prove accuracy, reliability on the
hidden test set, or likelihood of winning.

## 1. CPU-only static contracts

Requirement: Python 3.10+ and NumPy.

```bash
python run_static_preflight.py --output static_results.json
```

The 24-test suite covers:

- exhaustive and randomized `{0,1,2,3}` ↔ WT/TC/ET round trips;
- rejection of illegal, fractional, nonfinite, malformed, and nonbinary data;
- hierarchy projection, idempotence, unequal-threshold counterexamples, and
  1,000 randomized nested-mask trials;
- numerical edge cases in a clearly labeled NumPy loss surrogate;
- exact reference-patch stage arithmetic and encoder feature lower bound;
- the pinned ResEnc topology sentinel and GoAT reconstruction order;
- the official award-metric contract (DSC and HD95) kept separate from NSD and
  other diagnostic metrics;
- Task 3's flat filename ending in a five-digit case ID with no timepoint.

`static_results.json` is a reproducible result generated from this exact script.

## 2. Synthetic CUDA patch preflight

Run this only inside a locked CUDA environment after installing
`nnunetv2==2.8.1` and its resolved dependencies. The script fails on a different
nnU-Net version. Use the `batch_dice` value from the generated plan; the default
is `true` only for the synthetic check.

```bash
python run_gpu_preflight.py \
  --patch 160 160 128 \
  --batch-size 2 \
  --mode train \
  --batch-dice \
  --warmup-iterations 1 \
  --iterations 2 \
  --output gpu_train_patch_profile.json

python run_gpu_preflight.py \
  --patch 160 160 128 \
  --batch-size 1 \
  --mode inference \
  --warmup-iterations 2 \
  --iterations 5 \
  --output gpu_inference_patch_profile.json
```

The training mode imports nnU-Net's actual `DC_and_BCE_loss` with
`MemoryEfficientSoftDiceLoss`, mirrors nnU-Net v2.8.1's non-DDP
deep-supervision weights, uses autocast plus a dynamic gradient scaler, checks
finite outputs/loss/gradients, and verifies a strict bitwise checkpoint reload.
Inference mode does not allocate targets or an optimizer.

The reported memory is **patch-forward or patch-train memory only**. Do not call
it A10G release evidence. The complete release graph—preprocessing,
sliding-window inference, all ensemble members, post-processing, NIfTI I/O, and
the Docker process—must later be measured on an actual A10/A10G 24 GB GPU with
network disabled. The release target remains peak reserved VRAM below 21 GiB
and projected total runtime below 9 hours, inside the official 12-hour limit.

## 3. Required evidence sequence

1. Static suite: 24/24.
2. Synthetic CUDA patch training and inference: PASS.
3. Two labeled GoAT cases: complete train-step → checkpoint → full-case
   prediction → inverse geometry → official evaluator.
4. One real fold: stable optimization and official DSC/HD95; NSD reported as a
   diagnostic.
5. Second-fold confirmation before full cross-validation spending.
6. Actual A10G clean-room, no-network Docker test before submission.

Stop at the first failed gate. A passing earlier gate never waives a later one.
