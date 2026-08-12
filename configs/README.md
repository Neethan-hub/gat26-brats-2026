# `configs/` — stage chronology and what supersedes what

This directory holds **frozen, dated records**. Each preregistration, plan and lock file was written
at a particular stage and then never edited, because editing it would destroy the evidence that the
rule really was fixed *before* the result it governs. That is deliberate, and it has a consequence
readers must not get wrong:

> **Three records named in this table are private-only.** `g79v_tau1_sensitivity.json`,
> `g79vr_verification_spec.json` and `g87_a10g_preregistration.json` exist in the private
> development repository but are deliberately **not** exported, so they are absent from this
> directory in the public tree. They are listed because the chronology is incomplete without them,
> and each row says where the corresponding public evidence lives. Nothing else in this table is
> absent.

> **An earlier record here is not a current claim.** Where a frozen record and the camera-ready paper
> disagree, **the paper and the current top-level `README.md` are authoritative.**

## 1. Stage and freeze chronology

Records are grouped by the gate that produced them. Within the list, later entries supersede earlier
ones wherever they overlap.

| Stage | Records | What it froze |
|---|---|---|
| Pre-training preflight | `../preflight/`, `g45_pretraining_plan.json` | CPU-only static contracts; the pre-training plan. **Historical** — see the banner in `preflight/README.md`. |
| G4.5 / G5 architecture screen | `g45_config_M_fold0.json`, `g45_config_L_fold0.json`, `g45_selection_policy.json` | The ResEnc-M vs ResEnc-L fold-0 screen and its advancement rule, fixed before any fold result was seen. The screen used **DSC and HD95**. |
| G7.7 official-metric alignment | `g77_official_metric_alignment.json` | The organizer-confirmed ranking metrics. **This record supersedes the earlier metric understanding** used by the architecture screen. |
| G79-V / G79-VR tolerance work | `g79v_tau1_sensitivity.json`, `g79vr_verification_spec.json` — **private-only, intentionally not exported** | The `τ=1` analysis and its verification spec. They assert against internal per-case reconciliation evidence that is not redistributable, so they are absent from this directory in the public tree. Their *results* are public: the τ=1 component means and denominators for every screened policy are in `evidence/supplement_inputs.json` and are tabulated in the supplement. |
| G82 / G83 / G84 audits | `g82_preregistration.json`, `g83_dense_overlap_preregistration.json`, `g84_release_tta_preregistration.json` | Audits A, D25 (never executed) and C, each frozen before its comparison. |
| G85 confirmation follow-up | `g85_confirmation_preregistration.json` | The separate, development-informed reference-lesion miss-rate and folds-3–4 follow-up, commit-frozen **after** Audit C stopped and **before** those folds were opened. It is **not** advancement under Audit C. |
| G87 A10G qualification | `g87_a10g_preregistration.json` — **private-only, intentionally not exported**; `release/AWS_A10G_RUNBOOK.md` | The original two-gate A10G design. **Historical** — see below. The preregistration is withheld because it embeds provider resource identifiers; its outcome is stated in full in `A10G_QUALIFICATION_SUPERSEDED_IMAGE.md` and in `configs/release/README.md`. |
| Release | `release/Dockerfile`, `release/requirements.lock.txt`, `release/README.md` | The submitted container contract. |

## 2. Which later records supersede earlier metric understanding

* The architecture screen (`g45_selection_policy.json`) ranks on **DSC and HD95**. That was the
  metric understanding at G4.5/G5.
* `g77_official_metric_alignment.json` records the organizer-confirmed position: the **final ranking
  uses DSC and NSD**, computes final-ranking **NSD at `τ=1`**, and **excludes HD95**. HD95 is
  retained only as a diagnostic.
* The architecture screen was therefore run under a metric pair the challenge does not rank on. The
  paper reports this mismatch openly and rescores the same fold-0 predictions under DSC/NSD as a
  diagnostic; the point ordering reverses without a robust advantage. **The screen record was not
  rewritten** — doing so would have hidden the mismatch.
* `τ=0.5` is Panoptica's default and is retained throughout as a **prespecified sensitivity
  analysis** only. It does not carry equal official standing with `τ=1`.

## 3. Historical tests reproduce historical decisions

Several tests under `../tests/` and the whole of `../preflight/` pin values that are no longer the
current contract — a five-digit basename rule, a `[160,160,128]` reference patch size, DSC/HD95
ranking, the two-gate A10G design. **They are correct as history.** Each such test exists to prove
that a past decision really was made the way the record says, not to reassert an obsolete rule as
current policy. Do not "fix" them to match today's contract, and do not read them as today's
contract.

## 4. Current state

| Item | Current truth |
|---|---|
| Selected policy | **C0** — the frozen baseline. All three audits stopped on the development subset; no candidate was adopted. |
| Patch size | **`[128,160,112]`**. `[160,160,128]` appears only as a synthetic / reference-only figure in historical records. |
| Output naming | The **complete opaque input case-folder basename**, preserved byte-for-byte. **No five-digit rule**, no assumed cohort prefix. |
| Ranking metrics | **DSC and NSD at `τ=1`**. HD95 is diagnostic only. |
| Container images | An earlier **pre-correction** image failed organizer execution on the five-digit assumption. A **corrected** image was rebuilt and resubmitted. |
| Corrected-image evidence | **None beyond the build and submission itself.** No organizer execution log, no hidden-test result, no rank, and **no A10G measurement** for the corrected image. |
| Memory | One fold model is resident at a time, so simultaneous model residency does not scale with five folds. The running probability accumulator and current per-region probabilities are also resident, so total process peak memory is **not** that of a bare single-model run. |
| Authority | The **camera-ready paper** and the **current top-level `README.md`**. |
