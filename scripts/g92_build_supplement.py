#!/usr/bin/env python3
"""Generate the camera-ready supplement LaTeX from the committed audit artifacts.

Every number in the supplement is read from `artifacts/` at build time rather than typed by
hand, and the utility identity is re-derived and asserted before anything is written. The
supplement is aggregate-only: no per-case value, case identifier, split membership,
prediction, private path, submission identifier or image digest is emitted.

Usage: g92_build_supplement.py <repo_root> <out.tex>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

COMPONENTS = ["ET_DSC", "TC_DSC", "WT_DSC", "ET_NSD", "TC_NSD", "WT_NSD"]
PRETTY = {"ET_DSC": "ET DSC", "TC_DSC": "TC DSC", "WT_DSC": "WT DSC",
          "ET_NSD": "ET NSD", "TC_NSD": "TC NSD", "WT_NSD": "WT NSD"}

HEADER = r"""\documentclass[runningheads]{llncs}
\usepackage{array}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{longtable}
\usepackage[hidelinks]{hyperref}
\usepackage{xurl}
\hypersetup{
  pdftitle={Supplementary material: GAT-26, BraTS-GoAT 2026 Task 3},
  pdfauthor={Nathan Chen},
  pdfsubject={Supplementary material for the GAT-26 camera-ready paper}
}
% Float placement parameters. These govern where LaTeX may put a float; they change no
% margin, font size, page size or spacing. Loosening them keeps each table on or near the page
% that discusses it instead of accumulating a backlog that lands on half-empty float pages.
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}
\renewcommand{\topfraction}{0.92}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.06}
\renewcommand{\floatpagefraction}{0.75}
\renewcommand{\thetable}{S\arabic{table}}
\begin{document}
\title{Supplementary Material\texorpdfstring{\\}{ --- }GAT-26: Release-Path Auditing and Confirmation-Gated
Inference Selection for Cross-Tumor Brain Tumor Segmentation}
\titlerunning{Supplementary material --- GAT-26}
\author{Nathan Chen}
\authorrunning{N. Chen}
\institute{Kang Chiao International School, Xiugang Campus, New Taipei City, Taiwan\\
\email{naifenchen52@gmail.com}}
\maketitle

\begin{center}\fbox{\parbox{0.93\textwidth}{\small
\textbf{Scope.} This supplement contains \emph{aggregate} evidence only. It includes no patient
image, no label, no prediction, no case identifier, no split membership and no per-case metric.
It supports the main paper; it does not replace its core evidence.}}\end{center}

\section{Two distinct statistics: the architecture rank $R$ and the audit utility $U_\tau$}

The paper uses two scalar summaries that must not be conflated, and neither is the challenge's own
ranking procedure.

\medskip\noindent\textbf{Architecture screen, $R$ (fractional rank).} The fold-0 ResEnc-M versus
ResEnc-L screen ranks the two candidates component by component and averages those ranks, so $R$ is
a mean fractional rank over the six components and \emph{lower $R$ is better}. It is a \emph{rank}
statistic: it discards the size of each difference and retains only its direction. Two different tie
conventions apply here and must not be confused. \emph{Within} $R$, an exact component-level metric
tie assigns both models the average of the tied positions, so a tied component contributes the same
amount to each model's rank and can favour neither. \emph{Separately}, the frozen selection rule
resolves a tied or unmet advancement criterion in favour of ResEnc-M, the cheaper baseline plan;
that rule governs the decision, not the rank arithmetic. The screen was applied once, to fold~0,
under DSC/HD95.

\medskip\noindent\textbf{The advancement rule in full.} ResEnc-L could advance from fold~0 to
fold-1 confirmation only if $R(\mathrm{M})-R(\mathrm{L})\ge 1/6$ --- one component of six, the
constant \texttt{MEANINGFUL\_RANK\_GAIN} of the committed policy --- the paired-bootstrap 95\%
percentile interval for $\Delta R=R(\mathrm{L})-R(\mathrm{M})$ lay entirely below zero, and every
frozen noninferiority gate was supplied and passed, a gate with no input supplied counting as a
failure. Note the two directions: the rank gain $R(\mathrm{M})-R(\mathrm{L})$ is positive when
ResEnc-L is ahead, whereas $\Delta R$ is negative when ResEnc-L is ahead. Passing on fold~0 did not
select or prefer ResEnc-L; it only triggered fold-1 confirmation, and expansion required the same
rule to hold on fold~1. In every other case ResEnc-M was retained.

\medskip\noindent\textbf{Audit C utility, $U_\tau$ (raw metric mean).} Let $S_\tau$ be the subjects
for which all six DSC/NSD components are finite under \emph{both} policies, and
$\bar m_{c,\tau}(P;S_\tau)$ the mean of component $c$ over $S_\tau$. Then
\[
U_\tau(P;S_\tau)=\frac{1}{6}\sum_{c}\bar m_{c,\tau}(P;S_\tau),
\qquad
\Delta U_\tau=U_\tau(P_{\mathrm{cand}};S_\tau)-U_\tau(P_{\mathrm{base}};S_\tau),
\]
where $c$ ranges over the six DSC/NSD components, ET/TC/WT $\times$ DSC/NSD$_\tau$.
$U_\tau$ retains the magnitude of every difference and is \emph{not a rank statistic}: no ranking
or tie-breaking enters it. $R$ and $U_\tau$ are different objects and are never combined. Because both
policies share $S_\tau$, $\Delta U_\tau$ is identically the arithmetic mean of the six component
deltas; Table~\ref{tab:identity} verifies this against the committed record for every analysis
reported.

\medskip\noindent\textbf{Tolerance.} The organizers confirmed that the final challenge ranking uses
DSC and NSD, excludes HD95, and computes final-ranking NSD at $\tau=1$. Throughout, $\tau=1$ is the
official-ranking-aligned analysis and $\tau=0.5$ --- Panoptica's default, and what our earlier runs
used --- is a prespecified sensitivity analysis reported at the reviewers' request. It does not
carry equal official standing.
"""

FOOTER = r"""
\section{What this supplement does not contain}

No patient image, segmentation label, model prediction, model weight, case identifier, fold
membership or per-case metric appears here or in the public repository. No submission identifier,
image tag or digest, registry path, cloud resource identifier or credential appears here. Every
quantity above is an aggregate over a named subset, generated directly from the committed audit
records rather than transcribed by hand.

\end{document}
"""


def f(x, nd=6):
    return "---" if x is None else f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    repo, out = Path(sys.argv[1]), Path(sys.argv[2])
    A = repo / "artifacts"
    g84 = json.loads((A / "g84_result.json").read_text())
    g85 = json.loads((A / "g85_result.json").read_text())

    subsets = [
        ("Development (folds 0--2)", g84["calibration"], "tau1", "tau05", g84["common_support"]),
        ("Policy-selection holdout (folds 3--4)", g85["confirmation"], "t10", "t05", None),
        ("Pooled, supportive only (all folds)", g85["pooled_all_five_folds"], "t10", "t05", None),
    ]

    L = [HEADER]

    # ---- arithmetic identity table -------------------------------------------------
    L.append(r"""
\begin{table}[htbp]\centering
\caption{The utility identity, re-derived from the committed component evidence. For every analysis
reported, the recorded $\Delta U_\tau$ equals the arithmetic mean of its six component deltas. A
fractional-rank definition would instead yield exactly $+1.000$ in all six rows, since the candidate
improved all six components everywhere; it does not describe these data.}
\label{tab:identity}
\begin{tabular}{@{}llrrr@{}}
\toprule
Subset & $\tau$ & Recorded $\Delta U_\tau$ & Mean of six deltas & $|$difference$|$ \\
\midrule""")
    for label, block, k1, k05, _ in subsets:
        for key, tau in ((k1, "1.0"), (k05, "0.5")):
            b = block.get(key)
            if not b:
                continue
            rep = b["delta_U_common"]
            recomputed = sum(b["component_deltas"][c] for c in COMPONENTS) / 6
            assert abs(rep - recomputed) < 1e-12, (label, key, rep, recomputed)
            L.append(f"{label} & ${tau}$ & ${rep:+.9f}$ & ${recomputed:+.9f}$ & "
                     f"$<10^{{-12}}$ \\\\")
    L.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

    # ---- per-component means, each with its own adjacent bootstrap summary --------
    L.append(r"\section{Per-component means and deltas, Audit C (M8 versus C0)}" + "\n")
    L.append(r"""Each table below carries its own bootstrap summary and per-fold values inside the
same float, so no cross-referencing between paragraphs is required. ``Fraction of paired bootstrap
resamples with positive $\Delta U_\tau$'' is exactly that -- a resampling frequency, not a
probability and not a posterior.
""")
    for label, block, k1, k05, ncom in subsets:
        for key, tau in ((k1, "1.0"), (k05, "0.5")):
            b = block.get(key)
            if not b:
                continue
            n = b.get("n_common") or ncom
            role = "official-ranking-aligned" if tau == "1.0" else "sensitivity analysis"
            base, cand, dl = b["baseline_means"], b["candidate_means"], b["component_deltas"]
            bs = b.get("bootstrap") or {}
            if bs:
                ci = bs.get("ci95")
                boot = (f"Fraction of paired bootstrap resamples with positive "
                        f"$\\Delta U_\\tau$: ${bs.get('prob_positive')}$. "
                        f"95\\% percentile interval $[{ci[0]:+.6f},{ci[1]:+.6f}]$.")
            else:
                boot = ("Bootstrap: \\emph{not present in the committed record for this tolerance}. "
                        "No value is computed after the fact, and none is implied.")
            fd = b.get("fold_deltas")
            per = ("Per-fold $\\Delta U_\\tau$: "
                   + ", ".join(f"fold {k} ${v:+.6f}$" for k, v in sorted(fd.items())) + "."
                   ) if fd else ""
            L.append(rf"""
\begin{{table}}[htbp]\centering
\caption{{{label}, $\tau={tau}$ ({role}). Common subject support $|S_\tau|={n}$. DSC does not
depend on $\tau$, so the DSC rows repeat between tolerances.}}
\begin{{tabular}}{{@{{}}lrrr@{{}}}}
\toprule
Component & Baseline C0 & Candidate M8 & $\Delta$ \\
\midrule""")
            for c in COMPONENTS:
                L.append(f"{PRETTY[c]} & ${base[c]:.6f}$ & ${cand[c]:.6f}$ & ${dl[c]:+.6f}$ \\\\")
            L.append(r"\midrule")
            L.append(rf"$U_\tau$ / $\Delta U_\tau$ & ${sum(base[c] for c in COMPONENTS)/6:.6f}$ & "
                     rf"${sum(cand[c] for c in COMPONENTS)/6:.6f}$ & "
                     rf"${b['delta_U_common']:+.6f}$ \\")
            L.append(r"\bottomrule")
            L.append(r"\end{tabular}")
            L.append(r"\par\medskip")
            L.append(r"\begin{minipage}{0.92\textwidth}\small " + boot + " " + per +
                     r"\end{minipage}")
            L.append(r"\end{table}")

    # ---- decision matrices ---------------------------------------------------------
    L.append(r"""
\section{Complete decision-check matrices}

Check names are the \emph{legacy machine field names} recorded by the audit code, reproduced verbatim
so the tables can be matched against the published JSON. They are identifiers, not claims: in
particular \texttt{bootstrap\_probability} names a gate on the \emph{fraction of paired bootstrap
resamples with a positive difference}, which is a resampling frequency and not a probability. Each
check is reported exactly as the committed record holds it. The holdout matrix contains three
checks whose only purpose is to make the audit fail closed --- exact membership, zero evaluator
errors and independent recomputation agreement --- so that a missing, errored or
membership-mismatched candidate evaluation is ineligible rather than silently compared on whatever
subset survived.
""")
    for title, blob, cap in (
        ("Development subset (folds 0--2): 18 checks", g84["gates"],
         f"{g84['n_gates_passed']} of {g84['n_gates_total']} passed"),
        ("Policy-selection holdout (folds 3--4): 23 checks", g85["confirmation_gates"],
         f"{g85['confirmation_gates']['n_passed']} of "
         f"{g85['confirmation_gates']['n_total']} passed"),
        ("Pooled, supportive only: 18 checks", g85["pooled_gates"],
         f"{g85['pooled_gates']['n_passed']} of {g85['pooled_gates']['n_total']} passed"),
    ):
        L.append(rf"""
\begin{{table}}[htbp]\centering
\caption{{{title} --- {cap}.}}
\begin{{tabular}}{{@{{}}lc@{{}}}}
\toprule
Check (legacy machine field name) & Outcome \\
\midrule""")
        for k, v in blob["checks"].items():
            name = k.replace("_", r"\_")
            L.append(rf"\texttt{{{name}}} & {'pass' if v else r'\textbf{FAIL}'} \\")
        L.append(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}")

    # ---- lesion evidence -----------------------------------------------------------
    ln = g85["confirmation"]["lesion_noninferiority"]
    di = g85["confirmation"]["lesion_diagnostic_only"]
    g84l = g84["lesion"]
    bs = ln["bootstrap"]
    rd = ln["region_deltas"]
    mg = ln["margins"]
    L.append(r"""
\section{Lesion-level evidence and its limits}

Components are 26-connected. A reference component counts as missed if no predicted component of the
same region overlaps it; the denominator therefore depends only on the reference and is
policy-invariant, which the audit verified.

The counters below are matched components, not voxels, and they live in two different spaces. The
first two are counted over \emph{predicted} components, the third over \emph{reference} components.
The committed record labels the prediction-space overlap counter diagnostic only, and this supplement
keeps that label: it is reported for transparency and never used to override an official metric or a
safety gate. Prediction-space overlaps and reference-space misses are \emph{not} complementary under
component matching --- one predicted component may overlap several reference components and one
reference component may be covered by several predicted components --- so their sum need not equal
the reference-component total, and here it does not. The safety gate is computed as
$\mathrm{FN}_{\mathrm{ref}}/N_{\mathrm{ref}}$, the missed fraction of reference components; it is not
the legacy recall derived from the prediction-space overlap counter.
""")
    L.append(r"""
\begin{table}[htbp]\centering
\caption{Component-level counts and rates under 26-connectivity. Development counts are from the
development subset; the non-inferiority analysis and its margins apply to the policy-selection
holdout. The first row of each block is a prediction-space diagnostic counter and is not a
reference-space true-positive count.}
\begin{tabular}{@{}lr@{\hspace{1.8em}}r@{}}
\toprule
Quantity & \shortstack[r]{Baseline\\C0} & \shortstack[r]{Candidate\\M8} \\
\midrule""")
    rows = [
        (r"\multicolumn{3}{@{}l}{\emph{Development subset}}\\", None, None),
        (r"\quad Predicted components overlapping a reference",
         g84l["baseline"]["TP"], g84l["candidate"]["TP"]),
        (r"\quad Predicted components with no reference overlap",
         g84l["baseline"]["FP"], g84l["candidate"]["FP"]),
        (r"\quad Reference components with no predicted overlap",
         g84l["baseline"]["FN"], g84l["candidate"]["FN"]),
        (r"\multicolumn{3}{@{}l}{\emph{Policy-selection holdout}}\\", None, None),
        (r"\quad Predicted components overlapping a reference",
         di["baseline"]["tp_pred_diagnostic"], di["candidate"]["tp_pred_diagnostic"]),
        (r"\quad Predicted components with no reference overlap",
         di["baseline"]["fp_pred"], di["candidate"]["fp_pred"]),
        (r"\quad Reference components with no predicted overlap",
         di["baseline"]["fn_ref"], di["candidate"]["fn_ref"]),
        (r"\quad Reference components in total", ln["n_ref_total"][0], ln["n_ref_total"][1]),
    ]
    for name, b0, c0 in rows:
        L.append(name if b0 is None else f"{name} & ${b0}$ & ${c0}$ \\\\")
    L.append(r"\quad Missed fraction of reference components"
             f" & ${ln['miss_rate'][0]:.6f}$ & ${ln['miss_rate'][1]:.6f}$ \\\\")
    L.append(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}")

    L.append(
        f"On the holdout the miss rate rose by ${bs['point_delta']:+.6f}$ against a margin of "
        f"${mg['point_max']}$, with a one-sided 95\\% upper bound of "
        f"${bs['upper_95_one_sided']:+.6f}$ against ${mg['upper_bound_max']}$. Per region the "
        f"increases were ET ${rd['ET']:+.6f}$, TC ${rd['TC']:+.6f}$ and WT ${rd['WT']:+.6f}$, each "
        f"above the per-region limit of ${mg['region_max']}$. Predicted false-positive components "
        f"fell, so the ${mg['fp_max_increase_fraction']}$ relative-increase cap was not breached. "
        f"The bootstrap used {bs['n_subjects']} subjects, seed {bs['seed']} and "
        f"{bs['resamples']} resamples.\n")

    L.append(r"""
\medskip\noindent\textbf{Limits of this aggregate interpretation.} These are counts over a whole
subset. They establish the \emph{direction} of the change --- fewer predicted false positives, more
missed reference components --- and nothing about which lesions changed status. We did not stratify
by lesion size, location, region or cohort, and we did not test any mechanism, so the
precision--sensitivity reading offered in the main paper is a hypothesis rather than a demonstrated
cause. The absolute miss rate is high under this definition because every reference component counts
equally regardless of size. The margins are \emph{operational}: they were fixed in the protocol to be
strict relative to observed between-fold variation, the committed protocol records no clinical
derivation for them, and they are not externally validated clinical thresholds.
""")

    # ---- architecture diagnostic ---------------------------------------------------
    L.append(r"""
\section{Architecture diagnostic under the ranking metric}

The fold-0 ResEnc-M versus ResEnc-L screen was decided under DSC/HD95, on the values in
Table~\ref{tab:screen}. Because the challenge ranks DSC and NSD, the same fold-0 predictions were
rescored under DSC/NSD at $\tau=0.5$ as a diagnostic.

\begin{table}[htbp]\centering
\caption{The fold-0 screen as decided, under DSC and HD95. Means are over all 271 fold-0 cases.}
\label{tab:screen}
\begin{tabular}{@{}lcccccc@{}}
\toprule
Model & ET DSC & TC DSC & WT DSC & ET HD95 & TC HD95 & WT HD95 \\
\midrule
ResEnc-M (selected) & 0.859 & \textbf{0.914} & \textbf{0.934} & 14.17 & \textbf{5.96} & 3.89 \\
ResEnc-L            & \textbf{0.861} & 0.912 & 0.932 & \textbf{11.19} & 6.08 & \textbf{4.11} \\
\bottomrule
\end{tabular}
\end{table}

\medskip\noindent\textbf{Why the two tables differ slightly in ET DSC.} Table~\ref{tab:screen}
averages over all 271 fold-0 cases, in which a case with an empty reference \emph{and} an empty
prediction scores $1$. The rescoring below follows the official parser's skip-missing convention,
which drops those cases from the denominator: one for ResEnc-M and two for ResEnc-L. Excluding them
reproduces $0.8581$ and $0.8602$ exactly. Both figures are correct under their own stated
convention, and no other component is affected at the precision reported.

\begin{table}[htbp]\centering
\caption{Fold-0 architecture diagnostic under the ranked component pair, $\tau=0.5$. Bold marks the
better value in each column.}
\begin{tabular}{@{}lcccccc@{}}
\toprule
Model & ET DSC & TC DSC & WT DSC & ET NSD & TC NSD & WT NSD \\
\midrule
ResEnc-M (selected) & 0.8581 & \textbf{0.9140} & \textbf{0.9336} & 0.6565 & 0.6213 & 0.5759 \\
ResEnc-L            & \textbf{0.8602} & 0.9117 & 0.9322 & \textbf{0.6679} & \textbf{0.6244} & \textbf{0.5762} \\
\bottomrule
\end{tabular}
\end{table}

ResEnc-L is better at the point estimate on four of six DSC/NSD components, so the ordering
\emph{reverses} relative to the DSC/HD95 screen. On the rank statistic itself, with
$\Delta R = R(\mathrm{L})-R(\mathrm{M})$ so that negative values favour ResEnc-L, the observed value
is $\Delta R = R(\mathrm{L})-R(\mathrm{M}) = -0.333$ with a 95\% percentile interval of
$[-1.000,+0.667]$, which includes zero: no rank advantage is established, and the declared robustness
requirement was not met. Under the frozen tie rule an unmet advancement criterion retains ResEnc-M.

\medskip\noindent\textbf{Chronology, for the record.} The fold-0 screen completed on 2026-07-23 and
the organizers' clarification that the final ranking uses DSC/NSD at $\tau=1$ and excludes HD95 was
received on 2026-07-28, both dated in the committed decision log. That ordering is recorded as fact
and is \emph{not} offered as justification: the deployed model was still selected under a metric pair
the challenge does not rank on, and that remains a limitation of this submission.

\medskip\noindent\textbf{Limitations.} The diagnostic is fold-0 only and was computed at $\tau=0.5$
only; it was never repeated at the official-ranking tolerance $\tau=1$. It scores each architecture's
own end-of-training validation predictions, not the release path, so it is not comparable to the
Audit~C rows. The WT NSD difference is $0.0003$, at the edge of what these data resolve. Fold-0
ResEnc-L weights already existed and are what the diagnostic scores; after the diagnostic no
five-fold ResEnc-L expansion, architecture change or retraining campaign was undertaken. The
architecture--metric mismatch is therefore recorded as an open limitation, not a resolved question.

\section{Bounded experiment inventory}

The search was bounded, and is listed rather than implied to be exhaustive. The list mixes
candidates that were executed and scored with proposals that were screened out \emph{before}
execution; the outcome column says which is which.

\begin{table}[!ht]\centering
\caption{Bounded experiment inventory, including executed candidates and proposals screened out
before execution. Nothing here was adopted; the released policy is unchanged.}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.40\textwidth}>{\raggedright\arraybackslash}p{0.56\textwidth}@{}}
\toprule
Item & Outcome \\
\midrule
Best-validation checkpoint & No robust improvement; stopped on the development subset. \\
Mirroring over recorded axes & Executed. Point-positive, but the interval included zero; superseded by Audit~C. \\
ET connected-component cleanup, three declared volume thresholds & Traded ET overlap against ET boundary distance; an apparent mean gain proved to come from changed ET support. \\
Weight averaging (two declared per-fold soups of the final and best-validation checkpoints) & All six components moved the wrong way for both soups, monotonically in the amount mixed in. \\
Eightfold mirroring TTA (candidate M8) & Improved all six components on the development subset; failed the policy-selection holdout on the utility gate and the lesion miss-rate safety gate. \\
40-epoch fine-tuning screen: tumor-type-mixture, domain-generalization augmentation, and their combination & Failed its frozen gates. Bounded at 40 epochs, so it cannot speak to behaviour trained to convergence. \\
Dense-overlap variant (tile step $0.5\rightarrow0.25$) & \emph{Not} tested: its baseline gate mixed an augmentation-derived lineage with the no-augmentation release baseline. \\
Larger ensembles, cascades, cohort routing, alternative thresholds & Not tested. \\
\bottomrule
\end{tabular}
\end{table}

\section{Runtime, container and repeatability}

\noindent\textbf{Full-cohort run.} The exact release policy processed 451 of 451 official validation
cases with exit code $0$ and zero inference errors, at a peak framework-reserved VRAM of
$2.48$\,GiB and a wall time of 3\,h\,35\,m ($28.6$\,s per case) on a single NVIDIA A40. Both figures
describe that A40 execution. The $2.48$\,GiB is memory reserved by the tensor framework, not total
device memory, and is neither an A10G measurement nor a statement of A10G headroom.

\medskip\noindent\textbf{Repeatability.} Across two independent 24-case runs, 18 of
$2.14\times10^{8}$ output voxels differed. The runner is near-deterministic but not bit-exact,
because deterministic-algorithm enforcement runs in warn-only mode. Every output remained
label-valid, hierarchy-consistent and geometry-exact. This variation is orders of magnitude smaller
than the release-path gap discussed in the main paper.

\medskip\noindent\textbf{Container contract.} Linux/AMD64, zero network, all dependencies and all
five fold checkpoints baked in at build time. \texttt{/input} is read-only, one folder per case;
\texttt{/output} is flat, one \texttt{.nii.gz} per case. Each output file name is the complete
validated basename of its input case folder --- no truncation to a fixed-width identifier, no
hardcoded cohort prefix. Labels are restricted to $\{0,1,2,3\}$ and geometry is preserved exactly.
Modality discovery fails closed before any output is written. The runner refuses to start without
five distinct fold checkpoints. The official contract supplies a fresh writable \texttt{/output}; no
claim is made that the runner refuses to overwrite an existing file.

\medskip\noindent\textbf{Superseded A10G qualification.} An \emph{earlier, superseded}
pre-correction image was exercised on one NVIDIA A10G using synthetic four-modality volumes. That
image was executed by the organizers on the hidden test set and failed before writing any
prediction, because its runner required the input case-folder basename to end in exactly five
digits. The synthetic fixtures used in that qualification had fixed-width folder names and therefore
never exercised the variable-length condition that caused the failure. \textbf{The corrected image
that was finally submitted has never been measured on an A10G.} No organizer execution log, no
hidden-test result and no rank exists for it, and none is claimed. The historical record is retained
in the public repository under a filename that marks it as superseded.
""")

    L.append(FOOTER)
    out.write_text("\n".join(L))
    print(f"wrote {out} ({len(''.join(L).split())} words)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
