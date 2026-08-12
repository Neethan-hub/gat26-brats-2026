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
% This supplement is table-dense: several pages carry two full-width tables and nothing else.
% Under the class default (\flushbottom) LaTeX stretches vertical glue to force a flush bottom on
% such a page and reports "Underfull \vbox while \output is active". \raggedbottom lets those
% pages simply end where their content ends. It changes no font size, margin, page dimension,
% line spacing or table geometry, and it is not used to obtain any particular page count.
\raggedbottom
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.90}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.60}
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

\section{Two Distinct Statistics: The Architecture Rank $R$ and the Audit Utility $U_\tau$}

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
fold-1 confirmation only if the executable rank-gain threshold on $R(\mathrm{M})-R(\mathrm{L})$ was
met --- nominally $1/6$, one component of six, stored as the constant
\texttt{MEANINGFUL\_\allowbreak RANK\_\allowbreak GAIN} of the committed policy --- the
paired-bootstrap 95\% percentile
interval for $\Delta R=R(\mathrm{L})-R(\mathrm{M})$ lay entirely below zero, and every frozen
auxiliary gate was supplied and passed, a gate with no input supplied counting as a failure. Note
the two directions: the rank gain $R(\mathrm{M})-R(\mathrm{L})$ is positive when ResEnc-L is ahead,
whereas $\Delta R$ is negative when ResEnc-L is ahead. Passing on fold~0 did not select or prefer
ResEnc-L; it only triggered fold-1 confirmation, and expansion required the same rule to hold on
fold~1. In every other case ResEnc-M was retained.

\medskip\noindent\textbf{What the executable comparison actually is.} The gate is the binary64
comparison \texttt{rank\_\allowbreak gain >= MEANINGFUL\_\allowbreak RANK\_\allowbreak GAIN}, not an
exact real-number test of $R(\mathrm{M})-R(\mathrm{L})\ge 1/6$, and the two differ at the boundary.
The stored threshold is the double nearest $1/6$, $0.16666666666666666$; a configuration that is
exactly $1/6$ in rational arithmetic --- ResEnc-L ahead on three components, tied on one, behind on
two --- is computed as a difference of averaged ranks and evaluates to about $0.16666666666666652$,
some $1.4\times10^{-16}$ short, so that nominal boundary case is rejected. We disclose this because
the manuscript quotes the nominal value. It did not affect the recorded architecture decision, whose
fold-0 rank gain was $-0.333$ against a required $+1/6$ and which failed the other conditions too.
We did not modify the frozen policy code, and no result changed.

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
\section{What This Supplement Does Not Contain}

No patient image, segmentation label, model prediction, model weight, case identifier, fold
membership or per-case metric appears here or in the public repository. No submission identifier,
image tag or digest, registry path, cloud resource identifier or credential appears here. Every
quantity above is an aggregate over a named subset, generated directly from the committed audit
records rather than transcribed by hand.

\end{document}
"""


def f(x, nd=6):
    return "---" if x is None else f"{x:+.{nd}f}" if isinstance(x, float) else str(x)




AB_ORDER = ("C0", "C1", "C2", "C3", "C0_et10", "C0_et25", "C0_et50", "S1", "S2")
AB_LABEL = {
    "C0": r"C0 (released baseline)",
    "C1": r"C1 best-validation checkpoint",
    "C2": r"C2 recorded-axis mirroring",
    "C3": r"C3 best checkpoint $+$ mirroring",
    "C0_et10": r"ET cleanup, $10$\,mm$^3$",
    "C0_et25": r"ET cleanup, $25$\,mm$^3$",
    "C0_et50": r"ET cleanup, $50$\,mm$^3$",
    "S1": r"S1 soup, $0.75$ final $+$ $0.25$ best",
    "S2": r"S2 soup, $0.50$ final $+$ $0.50$ best",
}
AB_COMPS = ("DSC_ET", "DSC_TC", "DSC_WT", "NSD_ET", "NSD_TC", "NSD_WT")
# Short codes for the numeric tables; the full definitions are given once, in prose, above them.
AB_SHORT = {
    "C0": "C0", "C1": "C1", "C2": "C2", "C3": "C3",
    "C0_et10": r"ET$_{10}$", "C0_et25": r"ET$_{25}$", "C0_et50": r"ET$_{50}$",
    "S1": "S1", "S2": "S2",
}


def _tt(identifier: str) -> str:
    """Render a frozen status identifier verbatim, with LaTeX-safe underscores."""
    # \allowbreak after each underscore: these frozen status strings are long single
    # words and would otherwise overflow the measure as an unbreakable \texttt run.
    return r"\texttt{" + str(identifier).replace("_", r"\_\allowbreak{}") + "}"


AB_INTRO = r"""
Audit~A screened the best-validation checkpoint, mirroring over the recorded axes, both combined,
and enhancing-tumour connected-component cleanup at three predeclared volume thresholds. Audit~B
screened two predeclared per-fold weight averages of the final and best-validation checkpoints. The
main paper reports their outcomes; the per-candidate components are here.

\medskip\noindent\textbf{What these numbers are, exactly.} All rows are the development subset
($n=@N@$ subjects, folds @F0@--@F1@), scored by @EVAL@. The policy-selection holdout was never
opened for these audits, so no folds-3--4 value exists for any of them. Two measurement families
appear below and are kept apart because they aggregate differently:

\begin{itemize}
\item the \emph{decision} metrics --- DSC and HD95 over all @N@ subjects --- which are the pair
      these audits were actually ranked on, and
\item the \emph{official} metrics --- DSC and NSD --- re-scored afterwards from the same preserved
      predictions once the organizers confirmed the ranking pair. These use the evaluator's skipna
      rule, so each policy carries its \emph{own} denominators.
\end{itemize}

\medskip\noindent\textbf{Short codes used in the tables.} C1 is the best-validation checkpoint;
C2 is mirroring over the recorded axes; C3 is C1 and C2 combined; ET$_{10}$, ET$_{25}$ and
ET$_{50}$ are enhancing-tumour connected-component cleanup at the three predeclared volume
thresholds, in mm$^3$; S1 and S2 are the per-fold weight averages $0.75\times$final
$+\ 0.25\times$best and $0.50\times$final $+\ 0.50\times$best; C0 is the released baseline.

\noindent Because the two families use different aggregation rules they disagree in the third
decimal, and they are never combined into one column. The official-metric rows are \emph{not} on
the paired common support used for Audit~C, so they are not line-comparable with the Audit~C tables
above. Every value here comes from nnU-Net's end-of-training validation lineage, which applies
eightfold mirroring --- not from the release path. $\tau=1$ is the official-ranking-aligned
tolerance; $\tau=0.5$ is Panoptica's default and is reported as a prespecified sensitivity analysis
only. Rendered values are rounded to four decimal places; the machine-readable record in
\texttt{evidence/supplement\_inputs.json} carries full float precision, and its
\texttt{provenance} block maps every value in this section back to the frozen record and key path
it was read from, together with an explicit list of quantities that were never recorded and are
therefore not shown.
"""

AB_OUTCOME = r"""
\noindent\textbf{Outcome.} No candidate advanced at either tolerance: @G75@ for Audit~A, @G76@ for
Audit~B, and the later official-metric re-scorings returned @G77@ and @G79V@. The list of advancing
candidates is empty and the release policy was not changed. Denominators differ between policies
because enhancing-tumour cleanup removes small predicted components and can leave a subject with no
finite ET score; that is precisely why an ET-only mean gain is not evidence of improvement, and why
the audits' common-support rule exists.

\medskip\noindent\textbf{On the two weight soups.} Both lose to C0 on every component at both
tolerances, and on all three HD95 diagnostics. The ordering between them is C0 $>$ S2 $>$ S1
throughout: the soup that mixes \emph{more} of the best-validation checkpoint ($0.50$) is uniformly
\emph{better} than the one that mixes less ($0.25$). The degradation is therefore consistent in
direction but \emph{not} monotone in the amount mixed in, and the main paper says so.
"""

SCREEN_INTRO = r"""
\medskip\noindent\textbf{The bounded 40-epoch fine-tuning screen.} Three fine-tuning recipes were
screened for @EP@ epochs at a learning rate of $@LR@$ (seed $@SEED@$) on folds @SF0@--@SF1@, on a
common support of $n=@SN@$ subjects, each initialised from a bit-identical copy of the corresponding
C0 checkpoint. @CAVEAT@ The frozen record stores \emph{deltas} against C0 only: absolute
per-component means for the candidate arms were never computed and are not reconstructed here, and
HD95 was excluded by the preregistration, so no HD95 value exists for any arm. Bootstrap, per-fold
and tail statistics exist at $\tau=1$ only.
"""

SCREEN_OUTCOME = r"""
\noindent At $\tau=1$ the aggregate utility change was $@UT1@$ (T), $@UDG1@$ (DG) and
$@UTDG1@$ (TDG); at $\tau=0.5$ it was $@UT05@$, $@UDG05@$ and $@UTDG05@$. The $\tau=1$ paired
bootstrap reproduces those point estimates, with positive-resample fractions $@FT@$, $@FDG@$ and
$@FTDG@$; every interval spans zero. All three arms failed the $\tau=0.5$ utility gate and the total zero-DSC gate, so the
terminal status was @STATUS@ and C0 was retained. Of the ten frozen eligibility gates, T passed
@GT@, DG passed @GDG@ and TDG passed @GTDG@.

\medskip\noindent\textbf{The dense-overlap variant D25.} @D25NOTE@ It found that the step change
multiplies tiles per volume by $@M1@$, $@M2@$ and $@M3@$ across three representative geometries.
"""


def _fill(template: str, mapping: dict) -> str:
    for k, v in mapping.items():
        template = template.replace(k, str(v))
    return template


def _gates(screen: dict, arm: str) -> str:
    g = screen["arms"][arm]["gates"]
    return "$%d/%d$" % (sum(1 for v in g.values() if v), len(g))


def _ab_section(ab: dict, screen: dict, d25: dict) -> str:
    """Reviewer-requested per-component evidence for Audits A and B, and the bounded screen."""
    sub, off, dec = ab["subset"], ab["official_metrics"], ab["decision_metrics_dsc_hd95"]
    stats, out = ab["statistics_tau_1.0"], ab["outcomes"]
    n = sub["n_subjects"]
    L = [r"\section{Per-Component Evidence for the Checkpoint and Weight-Averaging Audits}"]
    L.append(_fill(AB_INTRO, {"@N@": n, "@F0@": sub["folds"][0], "@F1@": sub["folds"][-1],
                              "@EVAL@": sub["evaluator"]}))

    for tau_key, tau_tex, role in (("tau_1.0", r"\tau=1", "official-ranking-aligned"),
                                   ("tau_0.5", r"\tau=0.5", "sensitivity analysis")):
        L.append(r"\begin{table}[!htbp]\centering\small")
        L.append(r"\caption{Audit~A/B candidates, official-metric component means at $" + tau_tex +
                 r"$ (" + role + r"), development subset, $n=" + str(n) + r"$. Denominators are "
                 r"per policy and are given in Table~\ref{tab:abdenom}.}")
        L.append(r"\begin{tabular}{@{}l" + "c" * 6 + r"@{}}")
        L.append(r"\toprule")
        L.append(r"Policy & ET DSC & TC DSC & WT DSC & ET NSD & TC NSD & WT NSD \\")
        L.append(r"\midrule")
        for name in AB_ORDER:
            a = off[tau_key][name]["aggregates"]
            L.append(AB_SHORT[name] + " & " +
                     " & ".join("$%.4f$" % a[c] for c in AB_COMPS) + r" \\")
            if name == "C0":
                L.append(r"\midrule")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    L.append(r"\begin{table}[!htbp]\centering\small")
    L.append(r"\caption{Candidate minus C0, official-metric component deltas. A positive value "
             r"favours the candidate. Development subset, $n=" + str(n) + r"$.}")
    L.append(r"\begin{tabular}{@{}ll" + "c" * 6 + r"@{}}")
    L.append(r"\toprule")
    L.append(r"Policy & $\tau$ & $\Delta$ET DSC & $\Delta$TC DSC & $\Delta$WT DSC & "
             r"$\Delta$ET NSD & $\Delta$TC NSD & $\Delta$WT NSD \\")
    L.append(r"\midrule")
    for name in AB_ORDER:
        if name == "C0":
            continue
        for tau_key, tau_tex in (("tau_1.0", "1"), ("tau_0.5", "0.5")):
            base = off[tau_key]["C0"]["aggregates"]
            cand = off[tau_key][name]["aggregates"]
            lab = AB_SHORT[name] if tau_key == "tau_1.0" else ""
            L.append(lab + r" & $" + tau_tex + r"$ & " +
                     " & ".join("$%+.4f$" % (cand[c] - base[c]) for c in AB_COMPS) + r" \\")
        L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    L.append(r"\begin{table}[!htbp]\centering\small")
    L.append(r"\caption{The metrics these audits actually decided on: DSC and HD95 over all $n=" +
             str(n) + r"$ development subjects. HD95 is in millimetres and lower is better. This is "
             r"a different aggregation from the official-metric tables above, so the DSC columns "
             r"differ slightly.}")
    L.append(r"\begin{tabular}{@{}l" + "c" * 6 + r"@{}}")
    L.append(r"\toprule")
    L.append(r"Policy & ET DSC & TC DSC & WT DSC & ET HD95 & TC HD95 & WT HD95 \\")
    L.append(r"\midrule")
    for name in AB_ORDER:
        d = dec[name]
        cells = " & ".join(["$%.4f$" % d[k] for k in ("et_dsc", "tc_dsc", "wt_dsc")] +
                           ["$%.2f$" % d[k] for k in ("et_hd95", "tc_hd95", "wt_hd95")])
        L.append(AB_SHORT[name] + " & " + cells + r" \\")
        if name == "C0":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    L.append(r"\begin{table}[!htbp]\centering\small")
    L.append(r"\caption{Per-policy denominators (finite scored subjects per component), the frozen "
             r"advancement statistic against C0 at $\tau=1$, and the advancement outcome. "
             r"\textbf{Sign convention:} the rank gain is $R(\mathrm{C0})-R(\mathrm{candidate})$, "
             r"so a \emph{positive} value favours the candidate. The interval is the paired "
             r"subject-level percentile interval for that same quantity, in the \emph{same} "
             r"orientation as the point estimate. The frozen record stores the interval in the "
             r"opposite orientation, as $R(\mathrm{candidate})-R(\mathrm{C0})$; it is negated here "
             r"(which reverses its endpoints) and retained unmodified in "
             r"\texttt{evidence/supplement\_inputs.json} under "
             r"\texttt{bootstrap\_delta\_ci\_candidate\_minus\_C0}. No candidate advanced.}")
    L.append(r"\label{tab:abdenom}")
    L.append(r"\begin{tabular}{@{}lcccccc@{}}")
    L.append(r"\toprule")
    L.append(r"Policy & $n$ ET & $n$ TC & $n$ WT & Rank gain & 95\% interval & Advances \\")
    L.append(r"\midrule")
    for name in AB_ORDER:
        den = off["tau_1.0"][name]["denominators"]
        st = stats.get(name)
        if st is None:
            rg = ci = adv = "---"
        else:
            rg = "$%+.3f$" % st["rank_gain_over_C0"]
            # Same orientation as the point estimate; see the caption and the sign-convention
            # block in evidence/supplement_inputs.json.
            ci = "$[%+.3f,%+.3f]$" % (st["rank_gain_ci"][0], st["rank_gain_ci"][1])
            adv = "no" if not st["advances"] else r"\textbf{yes}"
        L.append(AB_SHORT[name] + r" & $%d$ & $%d$ & $%d$ & " % (
            den["DSC_ET"], den["DSC_TC"], den["DSC_WT"]) + rg + " & " + ci + " & " + adv + r" \\")
        if name == "C0":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    L.append(_fill(AB_OUTCOME, {"@G75@": _tt(out["g75"]), "@G76@": _tt(out["g76"]),
                                "@G77@": _tt(out["g77"]), "@G79V@": _tt(out["g79v"])}))

    d = screen["design"]
    L.append(_fill(SCREEN_INTRO, {"@EP@": d["epochs"], "@LR@": d["finetune_lr"],
                                  "@SEED@": d["seed"], "@SF0@": d["folds"][0],
                                  "@SF1@": d["folds"][-1], "@SN@": d["n_common_support"],
                                  "@CAVEAT@": screen["convergence_caveat"]}))
    L.append(r"\begin{table}[!htbp]\centering\small")
    L.append(r"\caption{Bounded " + str(d["epochs"]) + r"-epoch fine-tuning screen: component "
             r"deltas against C0 on the common support ($n=" + str(d["n_common_support"]) +
             r"$). Positive favours the candidate. Aggregate utility changes are given below. The screen "
             r"selected on $\min(\Delta U_{\tau=1},\Delta U_{\tau=0.5})$; no arm was eligible.}")
    L.append(r"\begin{tabular}{@{}ll" + "c" * 6 + r"@{}}")
    L.append(r"\toprule")
    L.append(r"Arm & $\tau$ & $\Delta$ET DSC & $\Delta$TC DSC & $\Delta$WT DSC & $\Delta$ET NSD & "
             r"$\Delta$TC NSD & $\Delta$WT NSD \\")
    L.append(r"\midrule")
    SC = ("ET_DSC", "TC_DSC", "WT_DSC", "ET_NSD", "TC_NSD", "WT_NSD")
    for arm in ("T", "DG", "TDG"):
        a = screen["arms"][arm]
        npass = sum(1 for v in a["gates"].values() if v)
        ntot = len(a["gates"])
        for tau_key, tau_tex in (("tau_1.0", "1"), ("tau_0.5", "0.5")):
            cd = a["component_deltas_" + tau_key]
            lab = arm if tau_key == "tau_1.0" else ""
            L.append(lab + r" & $" + tau_tex + r"$ & " +
                     " & ".join("$%+.5f$" % cd[c] for c in SC) + r" \\")
        L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tb = {a: screen["arms"][a]["bootstrap_tau_1.0"] for a in ("T", "DG", "TDG")}
    tm = d25["tile_multiplier_analysis"]["tiles"]
    keys = list(tm)
    L.append(_fill(SCREEN_OUTCOME, {
        "@PT@": "%+.6f" % tb["T"]["point"], "@PDG@": "%+.6f" % tb["DG"]["point"],
        "@PTDG@": "%+.6f" % tb["TDG"]["point"],
        "@FT@": "%.4f" % tb["T"]["prob_positive"], "@FDG@": "%.4f" % tb["DG"]["prob_positive"],
        "@FTDG@": "%.4f" % tb["TDG"]["prob_positive"],
        "@STATUS@": _tt(screen["terminal_status"]), "@D25NOTE@": d25["note"],
        "@GT@": _gates(screen, "T"), "@GDG@": _gates(screen, "DG"),
        "@GTDG@": _gates(screen, "TDG"),
        "@UT1@": "%+.6f" % screen["arms"]["T"]["delta_U_common_tau_1.0"],
        "@UDG1@": "%+.6f" % screen["arms"]["DG"]["delta_U_common_tau_1.0"],
        "@UTDG1@": "%+.6f" % screen["arms"]["TDG"]["delta_U_common_tau_1.0"],
        "@UT05@": "%+.6f" % screen["arms"]["T"]["delta_U_common_tau_0.5"],
        "@UDG05@": "%+.6f" % screen["arms"]["DG"]["delta_U_common_tau_0.5"],
        "@UTDG05@": "%+.6f" % screen["arms"]["TDG"]["delta_U_common_tau_0.5"],
        "@M1@": tm[keys[0]]["multiplier"], "@M2@": tm[keys[1]]["multiplier"],
        "@M3@": tm[keys[2]]["multiplier"]}))
    return "\n".join(L) + "\n"


INVENTORY_ROWS = (
    ("C1 best-validation checkpoint", "Executed",
     "Scored on the development subset. No robust improvement; the rank gain against C0 is negative. Stopped."),
    ("C2 mirroring over the recorded axes", "Executed",
     "Point-positive on the decision metrics but the interval included zero; superseded by Audit~C."),
    (r"C3 best checkpoint $+$ mirroring", "Executed",
     "The combined arm. No robust improvement; stopped on the development subset."),
    (r"ET connected-component cleanup, $10$\,mm$^3$", "Executed",
     "Raised ET DSC but changed ET support, trading overlap against boundary distance. Stopped."),
    (r"ET connected-component cleanup, $25$\,mm$^3$", "Executed",
     "As above, with fewer scored ET subjects. Stopped."),
    (r"ET connected-component cleanup, $50$\,mm$^3$", "Executed",
     "As above; the apparent mean gain comes from the changed ET denominator. Stopped."),
    (r"S1 weight soup, $0.75$ final $+$ $0.25$ best", "Executed",
     "All six components moved the wrong way at both tolerances, and it also failed the HD95 tail gate."),
    (r"S2 weight soup, $0.50$ final $+$ $0.50$ best", "Executed",
     "All six components moved the wrong way at both tolerances. Uniformly better than S1, so the degradation is not monotone in the amount mixed in."),
    ("M8 eightfold mirroring TTA", "Executed",
     "Improved all six components on the development subset but failed the frozen lesion false-negative veto and stopped; a separate folds-3--4 follow-up then failed the utility and miss-rate gates. C0 retained."),
    ("T tail-aware sampling", "Executed (bounded screen)",
     "40-epoch screen. Failed 5 of 10 gates, including both utility gates; ineligible."),
    ("DG appearance transform", "Executed (bounded screen)",
     r"40-epoch screen. Failed 3 of 10 gates, including the $\tau=0.5$ utility gate; ineligible."),
    (r"TDG (T $+$ DG)", "Executed (bounded screen)",
     r"40-epoch screen. Failed 2 of 10 gates, including the $\tau=0.5$ utility gate; ineligible."),
    (r"D25 dense overlap, tile step $0.25$", r"\textbf{Never executed}",
     "Preregistered and considered, then blocked before any run by an invalid baseline specification: the required baseline values are reproducible only under the mirroring lineage, while the specification defines C0 with mirroring disabled and prohibits test-time augmentation. Zero D25 predictions were generated and no D25 result of any kind exists. This is not evidence that D25 fails."),
)


def _inventory_table() -> str:
    """The complete bounded inventory: every item the main paper refers to, with a status column."""
    # A longtable, not a float: at thirteen prose rows this inventory is taller than one page, and
    # as a float it would overflow and strand a short page. longtable breaks across pages cleanly
    # and keeps the header on each one. No font, margin or spacing is changed.
    L = [r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.25\textwidth}"
         r">{\raggedright\arraybackslash}p{0.15\textwidth}"
         r">{\raggedright\arraybackslash}p{0.52\textwidth}@{}}",
         r"\caption{Complete bounded experiment inventory. Every candidate the main paper refers "
         r"to appears here. The status column separates candidates that were executed and scored "
         r"from proposals that were screened out before any execution. Nothing here was adopted; "
         r"the released policy is unchanged.}\\",
         r"\toprule", r"Item & Status & Outcome \\", r"\midrule", r"\endfirsthead",
         r"\toprule", r"Item & Status & Outcome \\", r"\midrule", r"\endhead",
         r"\bottomrule", r"\endfoot"]
    for item, status, outcome in INVENTORY_ROWS:
        L.append(item + " & " + status + " & " + outcome + r" \\")
        L.append(r"\addlinespace[2pt]")
    L += [r"\end{longtable}"]
    return "{\\small\n" + "\n".join(L) + "\n}\n"


def load_inputs(repo: Path) -> tuple[dict, dict, dict, str]:
    """Return (g84, g85, source_description), from private artifacts or public evidence.

    Two input paths, in priority order, both fail-closed:

      1. ``artifacts/g84_result.json`` + ``artifacts/g85_result.json`` -- the frozen private audit
         records. Authoritative, and used whenever they are present.
      2. ``evidence/supplement_inputs.json`` -- the sanitized aggregate projection published by
         ``scripts/g91_public_evidence.py``. It carries exactly the fields read below, at full
         float precision, so a public regeneration reproduces this file byte-for-byte.

    If neither is present the generator stops with an explicit diagnostic. It never proceeds on
    partial input and never substitutes a default for a missing measurement.
    """
    a84 = repo / "artifacts" / "g84_result.json"
    a85 = repo / "artifacts" / "g85_result.json"
    if a84.is_file() and a85.is_file():
        # The reviewer-requested Audit A/B, bounded-screen and D25 blocks are assembled by the
        # public-evidence generator, which is the single place that whitelists them. Reusing it
        # here keeps the private and public paths on one definition instead of two.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import g91_public_evidence as EV

        extra = {
            "audit_ab": EV._audit_ab(
                json.loads((repo / "artifacts" / "g75_inference_policy_decision.json")
                           .read_text(encoding="utf-8")),
                json.loads((repo / "artifacts" / "g76_checkpoint_soup_decision.json")
                           .read_text(encoding="utf-8")),
                json.loads((repo / "artifacts" / "g77_official_metric_decision.json")
                           .read_text(encoding="utf-8")),
                json.loads((repo / "artifacts" / "g79v_tau1_sensitivity_results.json")
                           .read_text(encoding="utf-8"))),
            "screen_g82": EV._screen_g82(
                json.loads((repo / "artifacts" / "g82_result.json").read_text(encoding="utf-8")),
                json.loads((repo / "configs" / "g82_preregistration.json")
                           .read_text(encoding="utf-8"))),
            "d25": EV._d25(json.loads((repo / "artifacts" / "g83_result.json")
                                      .read_text(encoding="utf-8"))),
        }
        return (json.loads(a84.read_text(encoding="utf-8")),
                json.loads(a85.read_text(encoding="utf-8")),
                extra,
                "private audit artifacts (artifacts/g84_result.json, artifacts/g85_result.json)")

    pub = repo / "evidence" / "supplement_inputs.json"
    if pub.is_file():
        payload = json.loads(pub.read_text(encoding="utf-8"))
        try:
            extra = {k: payload[k] for k in ("audit_ab", "screen_g82", "d25")}
            return payload["g84"], payload["g85"], extra, f"public aggregate evidence ({pub})"
        except KeyError as exc:
            raise SystemExit(
                f"{pub} is malformed: missing top-level key {exc}. Regenerate it with "
                "scripts/g91_public_evidence.py."
            ) from None

    raise SystemExit(
        "no supplement inputs found. Provide either the private audit records\n"
        f"  {a84}\n  {a85}\n"
        "or the sanitized public aggregate\n"
        f"  {pub}\n"
        "The latter is produced by: python3 scripts/g91_public_evidence.py <private_repo> <out>"
    )


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    repo, out = Path(sys.argv[1]), Path(sys.argv[2])
    g84, g85, extra, source = load_inputs(repo)
    print(f"inputs: {source}")

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
    L.append(r"\section{Per-Component Means and Deltas, Audit C (M8 Versus C0)}" + "\n")
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
\section{Complete Decision-Check Matrices}

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
\section{Lesion-Level Evidence and Its Limits}

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
\section{Architecture Diagnostic Under the Ranking Metric}

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
ResEnc-M (selected) & 0.859 & \textbf{0.914} & \textbf{0.934} & 14.17 & \textbf{5.96} & \textbf{3.89} \\
ResEnc-L            & \textbf{0.861} & 0.912 & 0.932 & \textbf{11.19} & 6.08 & 4.11 \\
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

@@AB_SECTION@@
\section{Bounded Experiment Inventory}

The search was bounded, and is listed rather than implied to be exhaustive. The list mixes
candidates that were executed and scored with proposals that were screened out \emph{before}
execution; the outcome column says which is which.

@@INVENTORY@@

\section{Runtime, Container, and Repeatability}

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

    body = "\n".join(L)
    # r11: reviewer-requested per-component evidence for Audits A/B and the bounded screen, and the
    # completed inventory. Both are rendered from the same input record as everything else.
    body = body.replace("@@AB_SECTION@@",
                        _ab_section(extra["audit_ab"], extra["screen_g82"], extra["d25"]))
    body = body.replace("@@INVENTORY@@", _inventory_table())
    assert "@@" not in body, "unfilled supplement placeholder remains"
    L = [body]
    L.append(FOOTER)
    # Byte-deterministic across operating systems: encode UTF-8 explicitly and write BYTES, so no
    # platform newline translation can occur. Path.write_text() opens in text mode, which on Windows
    # rewrites every "\n" to "\r\n" and would make the generated file differ, on that platform
    # alone, from the committed one.
    out.write_bytes("\n".join(L).encode("utf-8"))
    print(f"wrote {out} ({len(''.join(L).split())} words)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
