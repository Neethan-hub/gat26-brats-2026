# Aggregate evidence for the camera-ready paper

Everything here is **aggregate**. There are no per-case values, no case identifiers, no split
membership, no images, labels or predictions, and no submission, registry or cloud identifiers. The
files are generated from the committed audit records by `scripts/g91_public_evidence.py`, which
fails closed if a restricted string would be published.

Reviewers asked for two things the original paper did not show: the individual ET/TC/WT DSC **and**
NSD components behind each decision, and the complete decision-check matrices. Both are here.

| File | Contents |
|---|---|
| `policy_audit_components.csv` | Per-component baseline C0, candidate M8 and delta, for all three evaluation subsets at **both** NSD tolerances (36 rows) |
| `policy_audit_summary.json` | Aggregate ΔU, common-support size, bootstrap positive fraction, percentile interval and per-fold ΔU per subset |
| `policy_audit_checks.json` | The complete **18-check** development matrix and **23-check** holdout matrix, plus the supportive pooled matrix, each with pass/fail per named check |
| `lesion_noninferiority.json` | Reference-lesion miss-rate noninferiority on the holdout: counts, rates, per-region deltas, margins and outcome |
| `official_validation_scores.json` | The six official validation scores at full precision, with denominators and the row-count discrepancy recorded |
| `supplement_inputs.json` | The whitelisted aggregate projection the supplement generator reads: component means and deltas, common-support sizes, bootstrap summaries, per-fold deltas, the three gate matrices and lesion component **counts**, at full float precision. It exists so `scripts/g92_build_supplement.py` reproduces the committed `paper/supplement.tex` byte-for-byte from this repository alone, with no private input |

## How to read these

**Subsets.** `development_folds_0_2` is the subset on which candidates were compared and where
Audits A and B stopped. `policy_selection_holdout_folds_3_4` is the same-corpus holdout that the
mirroring candidate advanced to and failed. `pooled_all_folds` reuses the folds that produced the
decision, so it is **supportive and post-selection**, never a basis for a decision.

**Utility.** `U_tau(P; S_tau)` is the unweighted arithmetic mean of the six **raw** component means
— ET/TC/WT × DSC/NSD(tau) — over the common subject set `S_tau`. Therefore

    delta U_tau = U_tau(candidate) - U_tau(baseline)
                = mean of the six component deltas

which is exactly how `policy_audit_components.csv` reconstructs every `delta_U` in
`policy_audit_summary.json`; the generator asserts that identity and refuses to write if it fails.
DSC and NSD carry equal weight, no region is up-weighted, and HD95 never enters `U_tau`.

`U_tau` is **not a rank statistic**: no ranking, fractional ranking or tie-breaking is involved. The
separate fold-0 architecture screen *does* use a fractional-rank statistic, `R`. `R` and `U_tau` are
different objects, are never combined, and neither is the challenge's own ranking procedure.

**Common support.** A region-case contributes only if *both* policies yield a finite value, so a
change in the number of scorable cases cannot by itself manufacture a gain. Common support is a
comparability device, not an error-tolerance device: a candidate whose expected evaluation is
missing, errored or membership-mismatched is ineligible outright.

**Tolerances are not equal.** The organizers confirmed that the final challenge ranking uses DSC and
NSD, excludes HD95, and computes final-ranking NSD at **τ=1**. τ=1 is therefore the
official-ranking-aligned analysis. τ=0.5 is Panoptica's default — what our earlier runs used — and is
reported as a prespecified **sensitivity analysis** at the reviewers' request; it does not carry
equal official standing. The two tell different stories: the holdout retained 28.7 % of the
development gain at τ=1 and 59.3 % at τ=0.5. Neither ratio is an effect size.

Separately, and more narrowly: the organizers did **not** disclose which tolerance produced the
returned participant-visible *validation* scores, so no tolerance is attached to those numbers in
`official_validation_scores.json`.

**Missing bootstraps are null, never inferred.** The committed record contains no development-subset
bootstrap at τ=0.5. That entry is `null` with `bootstrap_available: false`, and no value is computed
after the fact to fill it. A regression test fails if it is ever populated.

**Margins are operational.** The lesion-safety margins were chosen to be strict relative to observed
between-fold variation of the quantity they bound. They are **not** externally validated clinical
thresholds and carry no clinical interpretation.

## What is deliberately absent

Per-case metrics, case identifiers, fold membership, zero-DSC case lists, patient-level qualitative
examples, model weights, and any organizer log. The paper's aggregate claims are reproducible from
these files; the per-case data behind them is not redistributable.
