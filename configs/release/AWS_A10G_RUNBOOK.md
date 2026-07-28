# AWS g5.4xlarge — genuine A10G validation runbook (two mandatory gates, fail-closed)

Owner-executed. RunPod has **no** genuine A10G; the genuine A10G lives in the AWS EC2 **g5** family.
Every step fails closed — **abort if any check does not pass.**

## Two separate mandatory A10G gates (HISTORICAL DESIGN — do NOT conflate)

The original two-gate design is preserved below as governance history, and it had
**no circular dependency**: release-feasibility on genuine A10G was to be proven *before* G7 using a
**duplicate-weight proxy** (A10G-1); final acceptance happens *after* G7 with the **five distinct
checkpoints** (A10G-2).

| Gate | When | Checkpoints | Purpose | Authorizes |
|---|---|---|---|---|
| **A10G-1** | **PRE-G7** (**NOT_RUN** — see current status) | M fold-0 in all 5 sequential slots — `runtime_only_duplicate_weight_proxy`, `no_accuracy_claim`, `not_final_ensemble` | container / CUDA-deps / offline / I/O / **VRAM <21 GiB** / determinism / **runtime <9 h projection** | **consideration** of G7 only — NOT G7 itself (needs a separate owner prompt) |
| **A10G-2** | **POST-G7** (**sole actionable gate**) | five genuinely distinct selected-M fold_0–4 `checkpoint_final.pth` | rebuild + test the **exact final submission image**; repeat every memory/runtime/offline/I-O check | mandatory precondition for final submission |

## CURRENT STATUS (authoritative)

- Architecture selection is **final: ResEnc-M**. **M fold-0 was not retrained.**
- **G7 is COMPLETE.** ResEnc-M folds 0–4 are complete and audited PASS; **five genuinely distinct**
  fold_0–4 `checkpoint_final.pth` exist; OOF reconciliation covers 1,351 cases exactly once with the
  frozen evaluator at n=1,351 / 0 errors.
- **G7.5 is COMPLETE** → `G75_RETAIN_BASELINE`; the frozen inference policy is **C0**
  (`checkpoint_final`, equal five-fold sequential ensemble, threshold 0.5, tile step 0.5, Gaussian,
  **no TTA**, **no connected-component filtering**, **no ET cleanup**, hierarchy-safe WT/TC/ET
  reconstruction, deterministic execution).
- **A10G-1 is `NOT_RUN` — it has never been executed and must never be recorded as PASS.** Because it
  was defined as a *pre-G7* feasibility gate and G7 completed under separate owner authorization, it
  **cannot be retroactively satisfied**. It is retained here as historical governance only and never
  substitutes for A10G-2.
- **A10G-2 is now the SOLE actionable A10G gate.** It is **NOT started and NOT authorized**. It
  requires the **five genuinely distinct** fold_0–4 checkpoints and **must NOT use the duplicate-weight
  proxy** (`--allow-duplicate-proxy` is forbidden; `discover_checkpoints` fails closed without all five).
- **The five-checkpoint model and its frozen C0 policy EXIST. The final release Docker image does NOT
  exist**, and **genuine-A10G runtime/VRAM remains UNMEASURED** — worker RTX-PRO-6000 figures are not
  A10G parity.
- **Owner-managed AWS quota/provisioning is PENDING.** No AWS access has been made; no approval is claimed.
- Identical/duplicated checkpoint content must **never** be treated as final-ensemble or accuracy evidence.

---

## Shared prerequisites (both gates)

### P0. Validation input data (no training data, no credentials to AWS)
- An organizer validation **input** artifact is available + authorized for the registered team:
  role = validation images (**labels withheld**), **1 archive, ≈6.29 GiB** (recorded privately in the
  G2 inventory — no entity IDs / filenames / case IDs / hashes here).
- **Acquisition (owner-controlled, secure):** authenticate to Synapse **interactively** using an
  **ephemeral config location OUTSIDE the build context** (e.g. `SYNAPSE_CONFIG=$(mktemp -d)/.synapseConfig`
  or an env token in the shell only), download the validation archive to an input dir **outside the
  build context** (e.g. `./val_input/`), then **delete the ephemeral credential immediately after
  acquisition** (`shred -u "$SYNAPSE_CONFIG"`). Only after registration/team requirements are met.
- **FORBIDDEN:** copying the protected GoAT **training** corpus (`nnUNet_raw` / `nnUNet_preprocessed`)
  or **any Git/Synapse credential** to AWS, and placing the validation archive, any credential, SSH
  material, or `.git` **inside the build context**. The container needs only baked weights +
  validation images; the validation images are mounted at `/input` at run time, never baked.

### P1. Provision (interactive, owner-only)
AWS EC2 **g5.4xlarge** (1× A10G 24 GB, 16 vCPU, 64 GiB RAM), region with g5 stock, ~$1.6/h on-demand
(**verify current pricing**); ~100 GB gp3 EBS. Connect via SSH (owner key) or SSM.

### P2. Hardware / environment gates (ABORT on any failure)
```
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
#   REQUIRE name == "NVIDIA A10G"            (RTX/A40/L4/A10/MIG/emulation -> ABORT: not parity)
#   REQUIRE memory.total ~ 23000–24576 MiB   (24 GB)
nproc                                         # REQUIRE >= 16 vCPU
free -g                                       # REQUIRE host RAM >= 48 GiB usable (g5.4xlarge = 64)
docker --version && docker info               # REQUIRE Docker daemon healthy
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi   # NVIDIA Container Toolkit
#   REQUIRE host driver supports CUDA <= 13.0 (image is CUDA 12.8.1)
```
Record the `nvidia-smi` GPU name verbatim as the parity gate; if not exactly **NVIDIA A10G**, STOP.

### P3. Build-context preflight + build + image content inspection (ABORT on credential/data leak)
```
# Deterministic preflight FIRST (no Docker): every COPY source present, checkpoint layout correct,
# and NO Synapse config / SSH material / credentials / validation archive / raw images / .git in
# the context. --proxy for the A10G-1 layout (fold_0 only); omit it for A10G-2 (five distinct).
python3 scripts/release_build_preflight.py --context ./build_context [--proxy]   # REQUIRE 0 problems
docker build --platform=linux/amd64 -f configs/release/Dockerfile -t gat26-goat-m:TAG ./build_context
docker run --rm --entrypoint /bin/sh gat26-goat-m:TAG -c '
  find / -xdev \( -name "*.pat" -o -name "*.token" -o -name ".synapseConfig" -o -name "id_*" \
    -o -name "*.pem" -o -name ".netrc" -o -name ".git" \) 2>/dev/null; echo "---"; ls -R /opt/gat26/weights'
#   REQUIRE: no credentials, no .git, no training corpus — only baked weights + plans + scripts.
```

### P4. Organizer-equivalent run (fail-closed container contract)
```
docker run --rm --gpus all --network none --runtime=nvidia --memory=48G --shm-size=16G \
  -e GAT26_REQUIRE_GPU_NAME="NVIDIA A10G" \
  -v $PWD/val_input:/input:ro -v $PWD/val_output:/output:rw gat26-goat-m:TAG
#   --gpus all: GPU visible to the container. The runner's GPU gate aborts BEFORE inference unless
#   torch sees EXACTLY ONE CUDA GPU whose name is EXACTLY "NVIDIA A10G".
#   --network none: zero network. /input read-only; /output flat. Exit MUST be 0, one .nii.gz per case.
```
Measure from **inside the real container**: peak VRAM (`nvidia-smi memory.used` sampled + runner
`peak_reserved_gib`) and **seconds/case** (runner `case_seconds`). The genuine-A10G run **must** pass
`--gpus all` and `-e GAT26_REQUIRE_GPU_NAME="NVIDIA A10G"`; a wrong/absent GPU aborts pre-inference.

### P5. Projection (documented, conservative — no assumed hidden-test count)
Project total = (max observed seconds/case) × (a **documented conservative** case-count bound — the
largest plausible hidden-test size stated by organizers, or an explicit conservative multiple of the
validation count; state which). **Never** assume a hidden-test case count.

---

## Gate A10G-1 — PRE-G7 release-feasibility SMOKE — **STATUS: `NOT_RUN` (HISTORICAL; NOT EXECUTABLE)**

> **This gate was never run and cannot be retroactively satisfied.** Its position in the ladder was
> *before* G7; G7 is complete, so the duplicate-weight proxy no longer has a purpose and the stronger
> A10G-2 supersedes it. The procedure below is retained verbatim as governance history. **Do not run
> it, do not record a PASS for it, and never accept it in place of A10G-2.**

1. Prerequisites P0–P2.
2. Build context with the **M fold-0 checkpoint replicated into all five sequential slots** — run the
   container **with `--allow-duplicate-proxy`**. Label every artifact/record
   `runtime_only_duplicate_weight_proxy` / `no_accuracy_claim` / `not_final_ensemble`.
   (`build_context/weights/fold_0/checkpoint_final.pth` present; the proxy path supplies five slots.)
3. Run P3–P5 on genuine A10G.
4. **PASS A10G-1** iff: GPU is genuine **NVIDIA A10G**; exit 0; `--network none`; `/input` read-only;
   `/output` flat, one `.nii.gz`/case; **peak reserved VRAM < 21 GiB**; deterministic; valid outputs
   (labels {0,1,2,3}, ET⊆TC⊆WT, exact geometry); **projected runtime < 9 h**.
5. **Meaning:** A10G-1 proves the real Docker container, CUDA/dependency compatibility, offline
   operation, I/O contract, memory bound, and runtime headroom. It **authorizes consideration of G7**;
   it does **not** authorize G7 by itself (a separate owner prompt is required). Its numbers are
   pipeline/graph/VRAM evidence only — **never** final-ensemble or accuracy evidence.

## Gate A10G-2 — POST-G7 FINAL ACCEPTANCE — **SOLE ACTIONABLE GATE; STATUS: `NOT_STARTED_NOT_AUTHORIZED`**

> Preconditions are **met** (G7 complete; five genuinely distinct checkpoints verified; frozen C0
> policy; deterministic runner). Execution still requires an explicit owner authorization **and**
> owner-managed AWS quota/provisioning. **No proxy is permitted at this gate.**

1. Precondition: **G7 complete** → five genuinely distinct selected-M fold checkpoints
   `weights/fold_{0..4}/checkpoint_final.pth` exist. ✅ satisfied.
2. Assemble the build context with those **five distinct** checkpoints and build the **exact final
   submission image WITHOUT `--allow-duplicate-proxy`** (so `discover_checkpoints` fails closed unless
   all five distinct checkpoints are present).
3. Repeat P3–P5 on genuine A10G — **all** memory/runtime/offline/`/input`/`/output` checks.
4. **PASS A10G-2** iff: genuine A10G; five distinct checkpoints; peak reserved VRAM < 21 GiB;
   projected runtime < 9 h; all output assertions pass; deterministic; image content clean.
5. **A10G-2 is mandatory before final submission.** This is the acceptance of the final submission
   image; A10G-1 does not substitute for it.

---

## Teardown & cost (DOCUMENT; do NOT execute here)
```
docker image rm gat26-goat-m:TAG
shred -u build_context/weights/fold_*/checkpoint_final.pth   # remove weights from AWS
# terminate the instance in the AWS console/CLI; confirm the EBS volume is deleted.
```
Record in `COST_LEDGER.csv`: gate (A10G-1 / A10G-2), instance type, region, hourly price, wall hours,
actual cost, and the verbatim `nvidia-smi` GPU name. Recommended per-gate cost ceiling **≤ $10**.

**All steps here are owner-only and interactive; Claude does not provision, download, build, run, or
tear down any AWS resource, and does not start G7 or training.**
