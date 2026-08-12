# GAT-26
## Final pre-training architecture and operations audit

> [!IMPORTANT]
> **HISTORICAL SNAPSHOT — not a current release validator and not current instructions.**
> This document is a pre-training / pre-submission record from the stage dated in its own header. It is
> retained unmodified as governance history and is **superseded** for every current claim. Do not
> follow it as current guidance, and do not read its constants, metric statements or feasibility
> language as the present contract.
>
> Current truth, as of the r11 camera-ready:
>
> * Official ranking uses **DSC and NSD at `τ=1`**; **HD95 is diagnostic only** and is not ranked.
> * Output naming preserves the **complete opaque input case-folder basename**. There is **no
>   five-digit rule** and no assumed cohort prefix.
> * Any `[160,160,128]` patch size appearing here is **synthetic / reference-only**; the **final
>   trained plan is `[128,160,112]`**.
> * The **corrected** container image was submitted, but **no organizer execution log, no hidden-test
>   result, no rank and no A10G measurement exist for that corrected image**.
> * The **camera-ready paper and the current top-level `README.md` are authoritative** wherever they
>   differ from this document.


Decision memorandum, version 2.0, 21 July 2026. Scope: GAT-26 architecture version 1.2, its execution guide, operator prompt pack, and preflight-test package before protected-data download or model training.

> [!CRITICAL]
> **Verdict: GO only for the next evidence gates; NO-GO for immediate full training.** The reviewed design is internally consistent, aligned with the current Task 3 award metrics, and based on a maintained high-performing baseline. It is ready for metadata discovery, environment locking, synthetic CUDA tests, and a two-case end-to-end smoke test. Full-fold and multi-fold spending remains conditional on those gates. No document can guarantee first place or eliminate hidden-test, optimization, deadline, and operational risk.

## 1. What this audit establishes

This audit can detect specification contradictions, wrong label/ranking semantics, unsupported implementation claims, weak tests, data leakage, stale cloud assumptions, and spending without evidence. It cannot establish segmentation accuracy without the protected GoAT data, training, out-of-fold predictions, and the official evaluator. It also cannot certify release feasibility without the complete container on an actual A10/A10G 24 GB GPU.

The correct funding question is not “can failure be made impossible?” It is: “are the cheapest failure modes removed, and is each expensive action blocked until its prerequisite evidence exists?” Architecture 1.2 now meets that standard.

## 2. Material findings and corrections

| ID | Finding in the previous files | Severity | Version-1.2 correction |
|---|---|---:|---|
| F1 | Candidate selection equally ranked DSC, NSD, and HD95 even though the live Task 3 page says monetary-award ranking uses DSC and HD95. | Critical | Primary utility is now the six global parser columns `ET/TC/WT × DSC/HD95`. NSD and connected-component/sensitivity/specificity/precision results are diagnostics/noninferiority gates unless organizers publish a newer rule. |
| F2 | MedNeXt was treated as an equally ready nnU-Net-v2 fold-0 candidate. Its official repository says the training pipeline is nnU-Net v1 and v2 preprocessing must be adopted independently. | High | Removed from the funded baseline. It is deferred until ResEnc CV and a release prototype pass, with a separate v2 integration gate. |
| F3 | The guide and prompts used a superseded workspace root, assumed a network volume, and included an already-completed bootstrap workflow. | Critical operational | Canonical repository is `/workspace/brats-2026-gat26`; the operator session already runs there remotely on the A40 controller. The 30 GB pod volume is described accurately and data download is blocked until measured sizing and durable training storage exist. |
| F4 | Broken scratch-only image directives would fail after download/upload. | High | Removed all nonportable image directives from the Markdown release. |
| F5 | The GPU harness used a custom Dice+BCE approximation, incorrect deep-supervision weighting, no dynamic gradient scaler, and allocated training objects in inference mode. | Critical test validity | The new harness imports nnU-Net 2.8.1's native `DC_and_BCE_loss` and `MemoryEfficientSoftDiceLoss`, mirrors non-DDP deep supervision, uses autocast plus `GradScaler`, verifies finite tensors/gradients and bitwise checkpoint reload, and keeps inference free of targets/optimizer. |
| F6 | The 12-test static suite contained tautological/weak checks and did not validate malformed inputs, metric separation, or Task 3 filenames. | High | Replaced by 24 deterministic tests, including 1,000 randomized hierarchy trials, negative tests, exact arithmetic/sentinel checks, award-metric separation, and filename/flat-output rules. |
| F7 | Synthetic patch memory was at risk of being interpreted as A10G release parity. | High | Every result and README now labels it `synthetic_patch_only`; the complete sliding-window ensemble Docker test remains mandatory on actual A10/A10G hardware. |
| F8 | The four-way standard/M/L/MedNeXt screen was too costly and integration-heavy for the remaining deadline. | High | The funded screen is M versus L only, after a two-case smoke test; a second fold confirms close results before full CV. |
| F9 | “Original model” could be read as a claim that the residual backbone was invented for GAT-26. | High scientific integrity | The architecture explicitly separates original system contributions from cited reused backbone code and confirms that no external trained weights are used. |
| F10 | Stale cutoff times created false failures and did not reserve enough release time. | High | Guide uses refreshed UTC/Taipei gates from 21 July, prioritizes a valid baseline, and freezes speculative work before the release window. |

## 3. Scientific architecture decision

### 3.1 Funded primary system

- Four MRI channels; random initialization; 2026 GoAT-only training bytes.
- Official nnU-Net v2.8.1 `ResidualEncoderUNet` through ResEnc M/L planners.
- Native region supervision in order `WT, TC, ET` with `regions_class_order=[2,1,3]` for GoAT labels `{0,1,2,3}`.
- Native Dice+BCE region loss, optimizer, schedule, sampling, augmentation, and deep supervision as the frozen baseline.
- Explicit cumulative-max probability projection plus mask intersections so `ET ⊆ TC ⊆ WT` even when thresholds differ.
- Subject-grouped, cohort-aware OOF evaluation; every threshold, presence rule, component cutoff, calibration map, and ensemble weight is cross-fitted.
- Primary selection on the six global DSC/HD95 parser columns, with NSD, connected-component, cohort, small-lesion, empty-region, calibration, tail-risk, runtime, and memory rejection gates.
- Actual A10G clean-room release gate below 21 GiB reserved VRAM and below nine projected hours, leaving margin within the official 12-hour limit.

### 3.2 Originality assessment

The neural backbone is an exact cited implementation, not a newly invented network. This is intentional risk control. GAT-26's original work is the GoAT-specific system: rule-locked labels and outputs, hierarchy-safe reconstruction, cross-domain validation, cross-fitted calibration and post-processing, tail-risk gates, ensemble membership selection, and accuracy–runtime release policy. It is neither a copied checkpoint nor a claim of a novel residual block.

Inventing unvalidated layers under a nine-day deadline would not make success more likely. A custom trainable component can enter only after it beats the exact baseline on paired GoAT evidence and passes deployment limits.

### 3.3 Deferred candidates

MedNeXt, hierarchy loss, surface loss, cohort sampling changes, and additional artifact augmentation are hypotheses. None is authorized merely because a paper reported a benefit elsewhere. MedNeXt additionally requires a separately tested nnU-Net-v2 integration. Optional candidates stop when release time or budget reserve is threatened.

## 4. Executed tests

The corrected CPU-only suite was executed from the rebuilt package and passed **24/24**.

| Test family | Tests | Result |
|---|---:|---:|
| Legal label/region round trips, exhaustive and randomized | 2 | PASS |
| Illegal, fractional, nonfinite, malformed, and nonbinary inputs | 4 | PASS |
| Hierarchy projection, exactness, idempotence, counterexample, and repair | 6 | PASS |
| Randomized nested masks and monotonic ET threshold (1,000 trials each) | 2 | PASS |
| Numerical loss surrogate and hierarchy-penalty direction | 3 | PASS |
| Patch/stride arithmetic, invalid topology inputs, exact feature lower bound | 3 | PASS |
| Pinned ResEnc contract and GoAT class order | 2 | PASS |
| Official award-metric separation and Task 3 filename/flat-output contract | 2 | PASS |

Additional release-package checks performed:

- both Python files parse and byte-compile;
- static results reproduce from the packaged script;
- the ZIP has no absolute paths, parent traversal, symlinks, encrypted entries, duplicate names, or unexpected executables;
- Markdown code fences and local references are checked;
- superseded operational paths/bootstrap instructions, scratch-only image directives, unavailable-format dependencies, and the previous misaligned award score are absent;
- SHA-256 hashes are generated after the final build.

The NumPy loss in the static suite is explicitly a numerical surrogate. It is not used to claim equivalence to nnU-Net. Native-loss equivalence is tested only by the CUDA harness importing the pinned package.

## 5. Tests that remain mandatory

| Gate | Authorized scope | Required evidence | Spending blocked after failure |
|---|---|---|---|
| G1 — controller CUDA patch | Minutes on current A40 | nnU-Net 2.8.1 import; native-loss AMP step; finite outputs/loss/gradients; strict checkpoint reload; measured memory | Any protected-data training |
| G2 — metadata and storage | Read-only Synapse metadata | exact entity/version/file-byte inventory; current 30 GB volume not used for unsafe partial download; durable training-storage plan | Dataset download |
| G3 — protected-data audit | Download only after G2 | cases/modalities/labels/geometry/duplicates/provenance/split integrity | Preprocessing and training |
| G4 — two-case smoke | Two labeled training cases | preprocessing → native train step → checkpoint → full-case inference → inverse geometry → output validator → official evaluator | Full fold |
| G5 — M/L fold 0 | One common fold per plan | stable optimization, official DSC/HD95, NSD diagnostics, strata, cost, checkpoint reproducibility | Second fold/full CV for failed plan |
| G6 — fold 1 confirmation | Close leaders only | directional repeat or cheaper plan wins uncertainty tie | Full CV expansion |
| G7 — selected CV | One selected ResEnc plan | complete OOF, cross-fitted rules, cohort/tail/empty-region report | Optional candidates/final model |
| G8 — release | Frozen model only | actual A10G, no-network, clean cache, complete multi-case Docker; <21 GiB and <9 h projected | Submission |

The current local audit environment has no CUDA/PyTorch and no protected data. Therefore, the CUDA harness is syntax-checked but not falsely marked as executed. G1 should be the first inexpensive technical test after the reviewed files enter the controller repository.

## 6. Current operational boundary

- Repository: `/workspace/brats-2026-gat26` on the dedicated A40 controller.
- Controller volume: 30 GB pod-attached at `/workspace`; it persists across stop/start but is deleted with Pod termination and is probably too small for the full training corpus/cache.
- The controller is suitable for interactive operator tooling, Git, public-source review, metadata discovery, code, and synthetic tests.
- No challenge dataset may be downloaded until the metadata-only byte inventory proves capacity and the owner approves durable training storage.
- Total budget ceiling is $650 and maximum concurrency is four GPUs. Every paid launch needs a pre-run cost estimate, explicit stop condition, and post-run actual cost entry.
- Existing unrelated RunPod resources are out of scope and must never be queried or mutated.

## 7. Remaining unknowns

- Actual GoAT file count, bytes, fingerprint, class balance, cohort balance, duplicates, and lesion-size distribution.
- Whether ResEnc M or L wins on this dataset; published averages cannot decide it.
- Absolute OOF accuracy and hidden-test generalization.
- Whether any optional loss/sampler/calibration rule adds a cross-fitted gain.
- Exact test-case count and complete A10G seconds per case.
- GPU availability, training duration, stochastic failures, and competitor performance.
- Final copyright-form delivery mechanism, which official text says organizers will provide.

Any agent that converts one of these unknowns into an assumed PASS must stop.

## 8. Final recommendation

Release version 1.2 of the four documents and preflight ZIP to the private repository. Then execute only G1 plus governance/document ingestion, followed by read-only Synapse authentication and metadata sizing. Do not start a dataset download or training fold from these documents alone.

This design has a credible path to a strong submission because it combines a high-performing maintained baseline with GoAT-specific reliability and deployment controls. It is not proven accurate yet, and it cannot honestly be guaranteed to win. The gates are the mechanism for discovering failure while it is still cheap.

## 9. Primary evidence reviewed

1. Live Task 3 description, labels, and regions: https://www.synapse.org/Synapse:syn74274097/wiki/639579
2. Live Task 3 data boundary and split composition: https://www.synapse.org/Synapse:syn74274097/wiki/639591
3. Live Task 3 evaluation (award aggregation on DSC and HD95): https://www.synapse.org/Synapse:syn74274097/wiki/639592
4. Live submission instructions and A10G container contract: https://www.synapse.org/Synapse:syn74274097/wiki/639582
5. Live timeline (30 July 2026, 23:59 UTC): https://www.synapse.org/Synapse:syn74274097/wiki/639587
6. Live challenge rules and GoAT-only override: https://www.synapse.org/Synapse:syn74274097/wiki/639585
7. Official deadline forum notice: https://challenges.synapse.org/Challenges/DetailsPage/Community?id=syn74274097&__forum_threadId=14013
8. Official nnU-Net v2.8.1 source and residual presets: https://github.com/MIC-DKFZ/nnUNet/tree/v2.8.1
9. Official nnU-Net region-based training contract: https://github.com/MIC-DKFZ/nnUNet/blob/v2.8.1/documentation/region_based_training.md
10. Official BraTS evaluator v0.0.8 and GoAT config: https://github.com/BraTS/BraTS_evaluation/tree/v0.0.8
11. Official MedNeXt repository and nnU-Net-v1 compatibility notice: https://github.com/MIC-DKFZ/MedNeXt
12. nnU-Net Revisited: https://arxiv.org/abs/2404.09556
