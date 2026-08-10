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

## How to read these

**Subsets.** `development_folds_0_2` is the subset on which candidates were compared and where
Audits A and B stopped. `policy_selection_holdout_folds_3_4` is the same-corpus holdout that the
mirroring candidate advanced to and failed. `pooled_all_folds` reuses the folds that produced the
decision, so it is **supportive and post-selection**, never a basis for a decision.

**Utility.** `U(P)` is the unweighted mean of six fractional ranks over ET/TC/WT × DSC/NSD. DSC and
NSD carry equal weight, no region is up-weighted, and HD95 never enters `U`. Ties resolve to the
baseline.

**Common support.** A region-case contributes only if *both* policies yield a finite value, so a
change in the number of scorable cases cannot by itself manufacture a gain. Common support is a
comparability device, not an error-tolerance device: a candidate whose expected evaluation is
missing, errored or membership-mismatched is ineligible outright.

**Both tolerances.** The official NSD surface tolerance was never exposed to participants, so every
NSD quantity is reported at τ=1 and τ=0.5 with equal standing. The two tell different stories — the
holdout retained 28.7 % of the development gain at τ=1 and 59.3 % at τ=0.5 — and neither ratio
should be read as the effect size.

**Margins are operational.** The lesion-safety margins were chosen to be strict relative to observed
between-fold variation of the quantity they bound. They are **not** externally validated clinical
thresholds and carry no clinical interpretation.

## What is deliberately absent

Per-case metrics, case identifiers, fold membership, zero-DSC case lists, patient-level qualitative
examples, model weights, and any organizer log. The paper's aggregate claims are reproducible from
these files; the per-case data behind them is not redistributable.
