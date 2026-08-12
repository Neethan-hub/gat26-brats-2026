# GAT-26
## Model architecture and experimental protocol for BraTS 2026 GoAT

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


Scientific design specification, version 1.2. Re-audited against the live challenge pages and pinned primary implementations on 21 July 2026. Target: BraTS 2026 Task 3, pre-operative cross-tumor MRI segmentation.

> [!CRITICAL]
> This is a high-probability competition design, not a promise to win. A winner cannot be specified in advance because hidden test composition, competing systems, stochastic training, and operational failures are unknown. The design minimizes avoidable risk by building on rigorously validated 3D CNNs, region-based BraTS supervision, cross-domain validation, conservative ensembling, and a release gate on the actual A10G constraint.

## 1. Executive design decision

The proposed system is **GAT-26: a Generalization-Aware Tri-region 3D ensemble**. GAT-26 is the name of the competition system and validation protocol; it is not a claim that the residual U-Net backbone was invented here.

The funded baseline uses the official nnU-Net v2 `ResidualEncoderUNet` implementation and region-based training, initialized from random weights. The ResEnc M and ResEnc L planners are screened on identical splits: M is the inexpensive control, while L is the official recommended default and becomes primary only if its GoAT gain and A10G deployment profile justify the extra compute. No hand-written approximation of the official residual encoder is allowed.

MedNeXt is **not** part of the funded baseline path. Its maintained training pipeline is based on nnU-Net v1, and its documented 1 mm/128³ recipe is not a drop-in nnU-Net v2.8.1 candidate. Integrating it safely would require an additional preprocessing/trainer validation program that the remaining deadline does not support. It is a deferred research option only after a passing ResEnc release candidate exists, and it receives no compute merely to make the system look novel.

The design rests on six decisions:

1. **Use the exact strong CNN baseline before novelty.** Rigorous comparisons found scaled CNN U-Nets, residual encoders, and ConvNeXt variants highly competitive when trained in the same framework; silently altering the reference preset defeats that protection.
2. **Train only on 2026 GoAT data.** No external images and no external pretrained weights are used.
3. **Represent the label hierarchy explicitly.** ET is contained in TC, which is contained in WT.
4. **Generalize by sampling and perturbing domains, not by guessing the hidden entity.** Cohort metadata may balance training and validation but is not an inference requirement.
5. **Select on the official award metrics, then apply reliability gates.** The primary utility uses the segmentation parser's global DSC and HD95 only. NSD, connected-component behavior, worst-cohort results, empty subregions, calibration, and confidence intervals remain essential diagnostics and noninferiority gates, but NSD is not silently inserted into the award score.
6. **Treat the container as part of the model.** Accuracy that misses the 12-hour or 24 GB limits has zero competitive value. Final funding is blocked until a measured A10G profile exists.

### 1.1 What is original and what is reused

GAT-26 is not a renamed copy of one checkpoint, and it does not reuse anyone else's trained weights. It is an original competition **system and experimental protocol** that combines: the 2026 GoAT-specific label/reconstruction contract; hierarchy-safe probability and mask construction; subject/cohort-aware validation and stress tests; cross-fitted threshold, presence, post-processing, and ensemble selection; tail-risk/empty-region rejection gates; and an A10G accuracy–runtime release frontier. Those choices and their integration are the project-specific contribution.

The convolutional backbone and native training recipe are deliberately reused from the maintained nnU-Net implementation and cited. Claiming that backbone as newly invented would be false. Under this deadline, changing proven layers merely to create superficial novelty would increase the probability of failure. Any new trainable component must outperform the exact backbone on paired GoAT evidence before it can enter the release.

## 2. Rule boundary and provenance

The live Task 3 rule says that only data supplied through BraTS-GoAT may train the final model. The following are prohibited in this design:

- ImageNet, medical-foundation, self-supervised, or supervised weights trained outside GoAT;
- data from a different 2026 BraTS task or an earlier BraTS release;
- institutional/private data;
- synthetic images generated from an externally trained model;
- a Vericerno checkpoint whose full training provenance is not GoAT-only;
- inference-time network calls.

All trainable parameters begin from a recorded random initialization. Code libraries and architecture implementations are allowed, but pretrained tensors must be disabled and audited. Before training, recursively inspect checkpoint-loading paths and cache directories; fail if an unexpected weight file is accessed.

Self-supervised, pseudo-label, semi-supervised, and transductive stages are disabled for the funded sprint. They may be reconsidered only after explicit written organizer authorization and completion of the supervised release candidate.

## 3. Task and output semantics

Each case provides four co-registered 3D MRI modalities:

`X ∈ R^(4×D×H×W) = [T1n, T1c, T2w, T2f]`.

The model predicts three voxelwise probabilities:

- `q_WT`: whole tumor;
- `q_TC`: tumor core;
- `q_ET`: enhancing tumor.

The target regions are derived from the integer annotation:

- `Y_ET = [Y = 3]`
- `Y_TC = [Y ∈ {1,3}]`
- `Y_WT = [Y ∈ {1,2,3}]`

For nnU-Net region-based training, preserve the region declaration order `WT, TC, ET` and set `regions_class_order=[2,1,3]`. WT is written as label 2 first, TC overwrites it with label 1, and ET overwrites TC with label 3. Alphabetically sorting the label dictionary is prohibited. A round-trip unit test must prove that every legal label in `{0,1,2,3}` is reconstructed exactly.

The final integer mask is reconstructed as:

- label 3 for `ET`;
- label 1 for `TC \ ET`;
- label 2 for `WT \ TC`;
- label 0 elsewhere.

The task-page convention is `{0,1,2,3}`. A data or output value of 4 is a release-blocking error.

## 4. Evaluation semantics

The official GoAT evaluator defines ET=`[3]`, TC=`[1,3]`, and WT=`[1,2,3]`. Its configuration computes global Dice similarity coefficient (DSC), normalized surface distance (NSD), and 95th-percentile Hausdorff distance (HD95), plus connected-component metrics.

The live Task 3 evaluation page states that aggregate award ranking uses **DSC and HD95**. The official segmentation parser (`parse_seg_results`) emits the global DSC/NSD/HD95 columns, while the evaluator also computes connected-component details. GAT-26 therefore reports the full output, but the predeclared primary model-selection utility uses only the six global award pairs `ET/TC/WT × DSC/HD95`. NSD and connected-component behavior are diagnostic/noninferiority evidence unless the organizers publish a newer ranking rule before model freeze.

Empty/missed regions are high-risk. The official evaluator returns zero overlap and infinite HD95 in relevant zero-true-positive cases, and its parser maps infinite/missing HD95 penalties to 373. Consequently:

- a tiny false ET component can be worse than a small boundary improvement elsewhere;
- missed small regions damage both overlap and distance metrics;
- output/filename omissions are catastrophic;
- region-presence calibration is a first-class experiment.

## 5. System pipeline

The pipeline is deterministic at validation and release:

1. Discover and validate the four modalities.
2. Preserve a reference image object containing shape, affine, spacing, origin, orientation, and filename identity.
3. Crop only to a safe nonzero union bounding box with recorded padding.
4. Apply the frozen spacing and intensity transform.
5. Run sliding-window inference for each selected model.
6. Restore logits to the original grid.
7. Fuse region logits with fixed weights.
8. Enforce WT ⊇ TC ⊇ ET, threshold, and apply only validated conservative post-processing.
9. Reconstruct `{0,1,2,3}` labels and save with cloned reference geometry.
10. Run an output validator before the process exits.

## 6. Preprocessing

### 6.1 Geometry

Do not re-register challenge data. The downloaded modalities are expected to be aligned, but the auditor verifies this case by case. The planner fingerprints voxel spacing, shape, anisotropy, and foreground extent and chooses a target spacing using the same data-driven principles as nnU-Net.

If all data are already on the challenge’s standard 1 mm isotropic grid, keep that grid. Otherwise, resample images with third-order interpolation and labels with nearest-neighbor interpolation for training, then invert the transform exactly for output. The final NIfTI header and affine are cloned from the input reference, not synthesized from rounded metadata.

### 6.2 Safe crop

The baseline uses the pinned nnU-Net `crop_to_nonzero` behavior: compute the union of nonzero voxels across modalities, crop to its exact bounding box, record that box, and restore the original canvas through the framework's export path. Do not add a custom margin or learned localizer in the baseline. An independent round-trip test must prove exact shape/geometry restoration and fail if any finite tumor-label voxel would be excluded.

### 6.3 Intensity normalization

The baseline is the current nnU-Net MRI normalization: per-case, per-modality z-score normalization inside the nonzero/valid-image mask, with an epsilon guard and zero background preserved. Reject or explicitly repair nonfinite values according to the audited manifest.

Percentile clipping is disabled in the baseline because MRI intensities are not standardized across scanners and unnecessary clipping can remove lesion signal. It becomes a candidate only if the GoAT audit identifies reproducible extreme artifacts and cross-fitted OOF testing shows benefit. Do not apply histogram templates or harmonization learned outside GoAT.

### 6.4 Missing modality policy

If the official dataset contract guarantees four modalities, an absent modality is a hard input error. Modality dropout is a robustness augmentation only if the data audit or organizer guidance suggests missing/corrupted channels may occur. It is not enabled merely because it sounds robust.

## 7. Primary backbone tournament: official nnU-Net ResEnc

GAT-26 does not reimplement the residual U-Net. It imports the official `dynamic_network_architectures.architectures.unet.ResidualEncoderUNet` used by the pinned nnU-Net environment. The candidate starting environment is `nnunetv2==2.8.1`; the final wheel hashes, dependency lock, source commit, CUDA/PyTorch versions, and container digest are frozen only after preprocessing, training, and inference smoke tests pass.

The official isotropic reference topology is:

| Stage | Spatial scale | Channels | Encoder blocks | Main operation |
|---:|---:|---:|---:|---|
| 0 | 1× | 32 | 1 | official residual-encoder stem/stage |
| 1 | 1/2 | 64 | 3 | stride-2 residual stage |
| 2 | 1/4 | 128 | 4 | residual feature extraction |
| 3 | 1/8 | 256 | 6 | residual feature extraction |
| 4 | 1/16 | 320 | 6 | high-context residual stage |
| 5 | 1/32 | 320 | 6 | bottleneck |

The decoder uses the official skip-connected `UNetDecoder` with `n_conv_per_stage_decoder=[1,1,1,1,1]`, not a custom two-block decoder. The frozen contract is `conv_bias=True`, `InstanceNorm3d(eps=1e-5, affine=True)`, LeakyReLU, no dropout, He initialization, and the official zero-initialization behavior before residual addition. Planner-generated kernels and strides may become anisotropic when the GoAT fingerprint requires it.

### 7.1 M/L screen and input plan

ResEnc M and L use the same network family; their planners target different memory budgets and therefore may choose different patch and batch plans. Official reference measurements place M near 9–11 GB and about 12 hours per fold on an A100, while L targets about 24 GB and 35 hours and is the nnU-Net authors' recommended default. Those figures are reference measurements, not GoAT guarantees.

- Input channels: 4.
- Reference synthetic patch: `160×160×128`, which is divisible through five isotropic downsamplings.
- Actual patch, batch, kernels, and strides: generated from the audited GoAT fingerprint and saved in immutable plans.
- Training batch: at least 2 as produced by the planner; do not add gradient accumulation or change batch Dice silently.
- Inference patch: frozen from the winning plan and separately proven below 21 GiB reserved memory on A10G.
- Sliding-window overlap: official 0.5/Gaussian baseline; any faster setting must pass paired OOF and runtime testing.

Run M first through the two-case smoke test and a short measured throughput benchmark. Run M and L on the same first screening fold only after those gates pass. Confirm a close result on a second fold before committing full-CV compute. L advances only if its paired GoAT result justifies its cost and its exact inference graph passes the A10G gate. M remains the release fallback. A standard nnU-Net or MedNeXt four-way tournament is not authorized under the current deadline.

### 7.2 Region logits, projection, and exact reconstruction

nnU-Net region-based training produces three independent WT, TC, and ET logits with sigmoid supervision and deep supervision. The native region decoder at threshold 0.5 is the baseline. Calibration is a later cross-fitted stage.

If hierarchy projection advances, apply the following cumulative maximum after model fusion:

- `p_ET = q_ET`
- `p_TC = max(q_TC, p_ET)`
- `p_WT = max(q_WT, p_TC)`

Ordered probabilities alone do **not** guarantee nested binary masks when region thresholds differ. The final masks must therefore be constructed explicitly:

- `M_WT = p_WT ≥ τ_WT`
- `M_TC = (p_TC ≥ τ_TC) ∩ M_WT`
- `M_ET = (p_ET ≥ τ_ET) ∩ M_TC`

This corrected construction passed randomized property testing and an explicit unequal-threshold counterexample. Reconstruct labels as WT-only→2, TC-only→1, and ET→3. A hierarchy loss remains optional because the reconstruction already guarantees valid masks.

## 8. Deferred diversity model: MedNeXt (not funded initially)

MedNeXt is a 3D ConvNeXt-style encoder-decoder with large-kernel depthwise convolutions, residual up/down blocks, and compound scaling. Its architectural diversity could reduce correlated errors relative to the residual U-Net, but that is a hypothesis, not evidence for this dataset.

The official repository states that its current training framework is built on nnU-Net v1 and that using nnU-Net v2 requires adopting the preprocessor independently. It also warns that the published MedNeXt v1 recipe used 1 mm isotropic spacing and that alternative nnU-Net-style spacing was untested. Therefore, no MedNeXt fold may start until the selected ResEnc full CV and a passing release-container prototype exist. If it is ever activated, use the official architecture code, initialize from scratch, create a separately tested nnU-Net-v2-compatible trainer, and begin with Small/Medium 3×3×3. External weights and UpKern weights trained outside GoAT are prohibited.

MedNeXt can enter a later research branch only if:

- it improves mean OOF rank on at least two folds;
- the improvement is not confined to one large cohort;
- worst-cohort ET and HD95 do not regress materially;
- its incremental A10G time is justified by incremental ensemble benefit;
- it can be exported and loaded without hidden external assets.

The default and expected submission is the selected ResEnc ensemble. Omitting MedNeXt is the scientifically safer decision unless all integration and evidence gates pass before model freeze.

## 9. Supervised objective

### 9.1 Baseline loss

Use nnU-Net's native region loss implementation—memory-efficient soft Dice plus `BCEWithLogits`—with its pinned batch-Dice and deep-supervision behavior. Do not replace it with the version-1.0 custom `0.6/0.4` weighting: even a constant rescaling changes the effective learning-rate recipe, and no GoAT evidence justified that divergence.

WT, TC, and ET are supervised as three foreground regions with no fourth background output. Unit tests must cover empty targets, empty predictions, extreme logits, and deep-supervision target resizing. Record the exact trainer class and loss source hash.

### 9.2 Hierarchy term

Only after the exact native baseline is stable, define the candidate on sigmoid region probabilities as:

`L_hier = mean(ReLU(q_TC - q_WT) + ReLU(q_ET - q_TC))`.

`L = L_region + λ_hier × L_hier`, initially `λ_hier = 0.05`.

Advance only if hierarchy violations fall and official metrics do not regress. Inference projection already guarantees valid masks, so the training term must earn its cost.

### 9.3 Surface-aware term

DSC does not directly constrain the boundary tail measured by HD95. A distance-transform or generalized surface loss is therefore an evidence-gated candidate:

`L_total = L_region + λ_hier L_hier + λ_s(t) L_surface`.

Ramp `λ_s` from 0 to at most 0.10 after the baseline’s early convergence. Precompute distance maps deterministically. Monitor gradients and ET stability; surface objectives can overemphasize noisy or tiny boundaries. Keep the term only if OOF HD95 improves without a credible Dice/NSD or worst-cohort penalty.

Do not simultaneously add focal, Tversky, boundary, Hausdorff, topology, and uncertainty losses. That would prevent attribution and raise optimization risk.

## 10. Sampling and augmentation

### 10.1 Patch sampling

The baseline is nnU-Net's native foreground oversampling and data-loader behavior. Do not begin with a custom `50/25/25` mixture. A small-lesion or boundary-aware sampler becomes a single-variable candidate only after the baseline OOF report demonstrates a specific ET/metastasis recall failure. It advances only if the smallest-volume strata improve without raising empty-region false positives.

### 10.2 Cohort balancing

Native case sampling is the baseline. Where the official training tree or manifest identifies cohorts, inverse-square-root cohort weighting with capped weights is a candidate, not an assumption. Use patient-level sampling and do not duplicate volumes on disk.

Cohort identity is never required by the inference graph. A hidden unseen tumor entity should pass through the same model.

### 10.3 Spatial augmentation

Use the pinned nnU-Net spatial augmentation pipeline unchanged for the baseline, applied consistently to all modalities and labels. Export every sampled parameter range from the trainer configuration. Any change to mirroring axes, rotation, scale, deformation, or low-resolution simulation is one ablation and must preserve plausible anatomy.

### 10.4 Intensity and acquisition augmentation

Use the pinned nnU-Net intensity augmentation pipeline as the baseline. A GoAT-specific acquisition bundle—bias field, Gibbs-like artifacts, or additional modality-specific perturbations—is permitted only as a separately logged candidate. Never add several artifact simulators at once merely because they appear realistic.

### 10.5 MixStyle status

MixStyle is removed from the funded sprint. Its evidence is indirect for this 3D cross-tumor setting, integration changes the official backbone, and the higher-value M/L selection and ResEnc release must finish first. It may be reconsidered only after the selected full CV and release candidate are complete.

## 11. Optimization

The primary baseline follows the pinned nnU-Net trainer exactly unless a measured constraint requires a separately named experiment:

- optimizer: SGD with Nesterov momentum 0.99;
- initial learning rate: 0.01, scaled only under documented batch/DDP rules;
- weight decay: 3×10^-5;
- polynomial decay to zero;
- automatic mixed precision with dynamic loss scaling;
- gradient clipping only if a reproducible instability is observed;
- reference budget: 1,000 epochs × 250 iterations.

Do not shorten training merely because a training-loss curve appears flat; validate a shortened schedule as a new trainer on paired folds. EMA is not in the initial screen and may not displace the native final/best checkpoint without cross-fitted evidence. Never select a checkpoint on training loss alone.

For parallel folds, run one independent training process per GPU. Distributed data parallelism is useful only if one fold cannot finish in time and its changed effective batch is revalidated. Independent folds yield evidence and reduce coordination risk.

## 12. Cross-validation that reflects the hidden test

### 12.1 Split construction

Use five subject-grouped folds when compute permits. Stratify by official cohort/source, ET presence, tumor-volume bins, and any pre/post or timepoint identifier discovered in the data. All scans from a subject belong to one fold.

Export `splits_final.json`, a summary table, and checksums. Never regenerate splits silently. A deterministic seed is necessary but not sufficient; inspect fold balance manually.

### 12.2 Stress tests

For each labeled source with enough cases:

- leave that source out of training and evaluate it as a domain-shift stress test;
- report region metrics, volume strata, and absent-region behavior;
- inspect confidence intervals rather than overinterpreting a tiny sample.

The hidden test composition is undisclosed. A candidate that improves pooled adult glioma while worsening meningioma/metastasis stress tests is not a generalization improvement.

### 12.3 OOF artifacts

Every fold produces full-resolution probability maps before thresholding, hard masks, official metrics, runtime, and QC records. These OOF probabilities are the only data used to select thresholds, post-processing, and ensemble weights.

### 12.4 Statistical reporting

For every candidate report:

- mean, median, standard deviation, and 95% bootstrap CI per region/metric;
- paired subject-level difference from the frozen baseline;
- cohort and tumor-volume strata;
- 5th-percentile DSC and 95th-percentile HD95;
- counts of empty reference, empty prediction, false-positive-only, and missed-region cases;
- calibration/reliability summaries for region presence;
- training and inference cost.

### 12.5 Cross-fitting every learned inference rule

Thresholds, presence gates, post-processing cutoffs, calibration mappings, and ensemble weights are trainable choices even though they are applied after the network. Reporting their performance on the same pooled OOF cases used to choose them is optimistically biased.

For each held-out fold, fit every inference rule using OOF predictions from the other four folds only, then evaluate the untouched fold. Concatenate these five cross-fitted predictions for the unbiased development report. After the method is frozen, refit the rule once on all OOF predictions for final deployment. Store both the cross-fitted evidence and final fitted parameters.

## 13. Candidate selection

Use a predeclared comparison rather than arbitrary raw-unit penalty constants:

1. Compute an equal-weight rank utility for the six global award pairs `ET/TC/WT × DSC/HD95`, reversing HD95, from cross-fitted case-level outputs generated by the pinned official segmentation parser. NSD and connected-component metrics are reported separately and may block a candidate under predeclared noninferiority margins, but they are not part of the primary award utility.
2. Compute paired subject-level differences from the frozen baseline with stratified bootstrap confidence intervals.
3. Require directional improvement on at least two folds and confirmation on a second fold before expanding an architecture screen to full CV.
4. Reject a candidate if a clinically important cohort, the smallest-volume stratum, empty-region behavior, or 95th-percentile HD95 shows a credible material regression.
5. Treat differences inside uncertainty as ties and select the cheaper, simpler model.

Freeze the utility and noninferiority margins before opening candidate results. Public validation may confirm a frozen choice but cannot rescue a failed internal gate.

## 14. Semi-supervised stage: excluded from the funded sprint

No pseudo-label, Mean Teacher, validation-set adaptation, or transductive stage is authorized in the release plan. The current live-rule interpretation is not sufficiently safe, and the short deadline prevents a leakage-proof nested evaluation from being completed responsibly. The fully supervised GoAT-only system is the only release path.

This decision can change only after a written organizer clarification explicitly authorizes the intended images and procedure, the selected supervised full CV is complete, and a held-out pseudo-label experiment can be executed without using its labels. A public-leaderboard gain is never sufficient evidence.

## 15. Thresholding and region-presence calibration

### 15.1 Voxel thresholds

Search fixed thresholds for ET, TC, and WT, initially 0.30–0.70 in increments of 0.05 and then locally. The reported result must use the five-way cross-fitting procedure in §12.5; tuning and reporting on the same pooled OOF predictions is prohibited. Evaluate the full official metric matrix at each triplet and do not optimize Dice alone.

### 15.2 Nested masks

After probability projection:

- `M_WT = p_WT ≥ τ_WT`
- `M_TC = (p_TC ≥ τ_TC) ∩ M_WT`
- `M_ET = (p_ET ≥ τ_ET) ∩ M_TC`

### 15.3 Presence gate

For ET and, if needed, TC, compute case-level features from OOF probabilities: maximum probability, high-probability volume, largest component, and ensemble disagreement. A simple calibrated rule may suppress a clearly spurious region in reference-empty cases.

The presence gate must use the five-way cross-fitting procedure, not merely pooled OOF data. Prefer a transparent threshold or logistic model over a complex classifier. Record false suppression of real small tumors; the gate is rejected if recall loss outweighs empty-case benefit.

## 16. Post-processing

Baseline post-processing is only hierarchy projection and integer reconstruction.

Candidate operations:

- remove components smaller than a very low physical-volume threshold;
- replace an implausible tiny ET component with label 1 rather than background when supported by TC;
- fill only one-voxel holes if OOF evidence improves surface metrics;
- eliminate components outside the valid-brain mask.

Never keep a threshold because it “looks cleaner.” Search in cubic millimeters, not voxels, and validate per tumor entity. Metastases can be small and multifocal; aggressive largest-component rules are especially unsafe.

## 17. Ensemble and deployment policy

### 17.1 Fusion

Average calibrated logits or probabilities, not hard masks. Equal weighting is the baseline. Any learned weight search must be cross-fitted under §12.5 and use a small discrete grid.

### 17.2 Ensemble membership

The primary release candidate is the equal-weight five-fold ensemble of the selected ResEnc plan because every member and its OOF behavior are already part of the validated pipeline. A smaller subset is allowed only if the A10G runtime frontier shows that the full ensemble cannot finish with margin. MedNeXt fusion is outside the baseline release plan and requires its own fully passing integration, CV-diversity, and A10G gates.

Do not replace the proven CV ensemble with a conveniently chosen full-data seed: a full-data seed has no held-out prediction for direct selection. A predeclared single full-data model may be trained only as the runtime fallback if the five-fold release cannot fit the A10G deadline. The choice between CV ensemble and single-model fallback is made by the clean-room runtime gate, not by seed shopping.

The target is a complete release under nine projected hours for the expected case count, leaving at least three hours of operational margin inside the 12-hour limit. Accuracy/runtime points are measured on A10G with the exact container.

### 17.3 Test-time augmentation

Evaluate identity, left-right flip, and the framework's full mirror set as separate runtime/accuracy points using cross-fitted predictions. No TTA is the release baseline. Add flips only after the full five-fold ensemble fits with margin and the gain per A10G minute exceeds the alternative architecture gain.

### 17.4 Adaptive runtime failsafe

At startup the container may count cases and estimate pre-profiled time. It may disable optional TTA from a fixed priority list, but it must not silently drop core fold models. The primary release configuration must already fit the expected workload; the failsafe is for unexpected case counts, not an excuse for poor profiling. Log the deterministic policy.

## 18. Reliability and uncertainty

The challenge accepts a hard mask, not an uncertainty map. Use uncertainty internally for:

- identifying OOF failure clusters;
- region-presence calibration;
- qualitative QC;
- deciding whether an ensemble member adds diversity.

Preferred uncertainty signals are ensemble variance, flip disagreement, and entropy. An evidential head is not part of the baseline: it changes optimization, is not directly ranked, and needs substantial calibration validation.

Reliability claims in the paper must be limited to the challenge population and measured validation protocol. The model is research software and is not validated for clinical use.

## 19. Experiment ladder and burn controls

| ID | Experiment | Dependency | Advancement criterion |
|---|---|---|---|
| E00 | Static contract suite | architecture v1.2 | 24/24 tests pass; completed locally |
| E01 | Synthetic CUDA patch preflight | locked environment | native-loss AMP step, finite gradients, checkpoint, correct shapes; explicitly not A10G release evidence |
| E02 | Two-case GoAT smoke test | data audit + E01 | geometry, loss, checkpoint, evaluator pass |
| E03 | Fold-0 M/L screen | E02 | stable runs and paired global DSC/HD95 parser metrics; NSD/connected-component diagnostics complete |
| E04 | Fold-1 confirmation of close leaders | E03 | direction repeats or simpler model wins tie |
| E05 | Selected ResEnc full CV | E04 | cross-fitted OOF and stress report complete |
| E06 | Deferred MedNeXt integration/CV | E05 + passing release prototype + time reserve | separate v2-compatible integration passes before any fold; incremental cross-fitted ensemble gain |
| E07 | Cohort-balanced sampler | E05 | paired gain, no tail regression |
| E08 | Hierarchy/surface candidate | E05 | official metric gain without cohort harm |
| E09 | Cross-fitted thresholds/presence/post-process | E05/06 | empty-case benefit exceeds small-lesion recall cost |
| E10 | Ensemble/no-TTA runtime frontier | frozen models | Pareto-optimal cross-fitted accuracy/runtime point |
| E11 | A10G clean-room release | E10 | ≤21 GiB, projected ≤9 h, every output assertion passes |

One primary variable changes per experiment. E06, semi-supervision, and MixStyle are outside the baseline sprint. Stop optional E06–E09 at model freeze; never borrow release time.

## 20. Failure-mode matrix

| Failure mode | Signal | Mitigation | Release test |
|---|---|---|---|
| Wrong label convention | value 4 or bad region nesting | strict data/output schema and reconstruction tests | exhaustive unique-value test |
| Subject leakage | duplicate hash or patient in two folds | group-aware split and duplicate audit | split integrity test |
| Adult-glioma domination | poor minority-cohort stress scores | cohort-aware sampler and report | per-cohort paired metrics |
| Tiny ET false positives | reference-empty ET penalties | OOF presence calibration; conservative threshold | empty-region confusion table |
| Small tumor suppression | recall drop after component filtering | no aggressive removal; volume-stratified validation | smallest-volume decile report |
| Scanner shift | style/intensity sensitivity | robust normalization and tested augmentation | source-held-out stress test |
| Hierarchy violation | ET outside TC or TC outside WT | probability projection and reconstruction | zero-violation unit test |
| Geometry corruption | invalid submission despite good mask | clone reference geometry and compare exactly | affine/spacing/orientation assertion |
| Runtime overflow | container killed before all cases | A10G profiling, small ensemble, margin, failsafe | expected-case-count simulation |
| Hidden dependency/network | clean Pod or `--network none` fails | bake weights/dependencies; clean-room build | no-network end-to-end run |
| Stochastic mismatch | reruns differ unexpectedly | deterministic inference and pinned versions | repeated mask hash |

## 21. Required tests

### Unit tests

- modality discovery under shuffled file order;
- task-specific case-ID parsing;
- normalization with zero variance and nonfinite voxels;
- crop/inverse-crop identity;
- region conversion and inverse label reconstruction;
- exact `regions_class_order=[2,1,3]` and insertion-order preservation;
- hierarchy projection on adversarial probabilities and unequal thresholds;
- NIfTI save/load with exact geometry;
- output-name and flat-directory contract;
- no pretrained checkpoint path.

### Integration tests

- one optimizer step and checkpoint round trip;
- exact official topology/config hash comparison against generated plans;
- one full case through preprocessing/inference/post-processing;
- official evaluator on a perfect prediction and controlled errors;
- multi-case container with input order changed;
- missing modality failure with actionable log;
- empty ET, empty TC, and empty prediction cases;
- no-network container execution;
- A10G memory and runtime profiler.

The framework-independent static suite in the preflight package passed 24/24 checks. This does not replace the PyTorch GPU, protected-data, full-case, or A10G container tests.

### Release assertions

The release script must fail unless output count equals case count, every file is readable, geometry matches, values are valid, files are flat under `/output`, no temporary files remain, and the process exits zero.

## 22. Experiment registry schema

`EXPERIMENT_REGISTRY.csv` contains at least:

| Field | Purpose |
|---|---|
| experiment_id / parent_id | lineage |
| hypothesis | predeclared reason |
| git_commit | exact code |
| config_hash | immutable configuration |
| data_manifest_hash / split_hash | data and leakage control |
| architecture / seed / fold | model identity |
| start/end UTC / GPU / cost estimate | operational accounting |
| best checkpoint hash | reproducibility |
| ET/TC/WT DSC, NSD, HD95 | official outcomes |
| worst-cohort and tail metrics | reliability |
| A10G seconds/case and peak VRAM | deployment value |
| decision / decision_reason | evidence trail |

An experiment without a config hash and data/split hash is exploratory and cannot supply final evidence.

## 23. What not to build during this sprint

- a brand-new transformer or diffusion segmentation architecture;
- a hand-written approximation of the official nnU-Net ResEnc preset;
- a cascaded detector that can crop out unseen tumors;
- a large external foundation model;
- a source-conditioned mixture of experts that needs a reliable hidden entity label;
- a complicated evidential/open-set head not used by the rank metric;
- a pseudo-label or transductive stage without explicit written authorization;
- an ensemble selected solely from public-leaderboard scores;
- an aggressive “largest component only” post-processor;
- a final image that downloads weights at startup.

These ideas may be research projects, but they are not justified under the current deadline and rule constraints.

## 24. Scientific basis

| Design element | Primary evidence | Use in GAT-26 |
|---|---|---|
| self-configured 3D U-Net | Isensee et al., nnU-Net, Nature Methods 2021 | preprocessing, patch planning, baseline training |
| official residual-encoder presets | nnU-Net ResEnc preset documentation | exact M/L implementation and measured compute targets |
| scaled residual/ConvNeXt CNN validation | Isensee et al., “nnU-Net Revisited,” 2024 | M/L screen and rigor-first selection |
| region-based training | official nnU-Net documentation | WT/TC/ET targets and ordered reconstruction |
| MedNeXt | Roy et al., MICCAI 2023 | conditional architecture diversity |
| GoAT MedNeXt/ensemble experience | Maani et al., BraTS-GoAT 2024 | ensemble and post-processing prior, not reused weights |
| cross-validated ensemble/post-processing | Jiang et al., ISBI 2024 | OOF thresholding and diverse model fusion |
| surface/HD loss | Karimi and Salcudean, IEEE TMI 2020 | gated HD95-oriented loss candidate |
| generalized surface loss | Celaya et al., 2023 | numerically stable surface candidate |

## 25. References and official implementations

1. Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods. 2021. https://doi.org/10.1038/s41592-020-01008-z
2. Isensee F, Wald T, Ulrich C, et al. nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation. 2024. https://arxiv.org/abs/2404.09556
3. Official nnU-Net v2.8.1 implementation and ResEnc presets. https://github.com/MIC-DKFZ/nnUNet/tree/v2.8.1 and https://github.com/MIC-DKFZ/nnUNet/blob/v2.8.1/documentation/resenc_presets.md
4. Official nnU-Net v2.8.1 region-based training documentation. https://github.com/MIC-DKFZ/nnUNet/blob/v2.8.1/documentation/region_based_training.md
5. nnU-Net v2.8.1 candidate package release, 1 July 2026. https://pypi.org/project/nnunetv2/2.8.1/
6. Roy S, Koehler G, Ulrich C, et al. MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation. MICCAI 2023. https://arxiv.org/abs/2303.09975
7. Official MedNeXt implementation. https://github.com/MIC-DKFZ/MedNeXt
8. Maani FA, et al. On Enhancing Brain Tumor Segmentation Across Diverse Populations with Convolutional Neural Networks. 2024. https://arxiv.org/abs/2405.02852
9. Jiang Z, et al. Enhancing Generalizability in Brain Tumor Segmentation. ISBI 2024. https://www2.die.upm.es/im/papers/ISBI24-1618.pdf
10. Karimi D, Salcudean SE. Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural Networks. IEEE Transactions on Medical Imaging. 2020. https://arxiv.org/abs/1904.10030
11. Celaya A, Riviere B, Fuentes D. A Generalized Surface Loss for Reducing the Hausdorff Distance in Medical Imaging Segmentation. 2023. https://arxiv.org/abs/2302.03868
12. BraTS 2026 Task 3 live page. https://challenges.synapse.org/Challenges/DetailsPage/Task3?id=syn74274097
13. Official BraTS evaluator, verified tag v0.0.8. https://github.com/BraTS/BraTS_evaluation/tree/v0.0.8
14. BraTS 2026 challenge design. https://doi.org/10.5281/zenodo.19714728

Research sources justify candidates; they do not guarantee that a candidate improves this year’s hidden test. The OOF and release gates remain authoritative.
