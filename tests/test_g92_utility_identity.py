#!/usr/bin/env python3
"""G92 — utility-definition and unsupported-claim regression tests.

Run: python3 tests/test_g92_utility_identity.py

The camera-ready manuscript once defined the Audit C utility as a mean of fractional ranks.
That definition is arithmetically incompatible with the reported values: the candidate improved
all six components in every analysis, so a rank utility would give exactly +1.000 everywhere,
whereas the recorded deltas are on the order of 1e-3. The committed evidence shows the utility is
the unweighted arithmetic mean of the six RAW component means on common subject support.

These tests pin that, keep the architecture rank statistic R distinct from the audit utility
U_tau, and refuse any bootstrap number the committed record does not contain.

They read only tracked repository files. No GPU, no challenge data, no network.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMPONENTS = ["ET_DSC", "TC_DSC", "WT_DSC", "ET_NSD", "TC_NSD", "WT_NSD"]
TOL = 1e-12

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{'' if ok else ' -- ' + detail}")
    if not ok:
        failures.append(name)


def _subsets() -> list[tuple[str, dict]]:
    """Every (label, per-tolerance block) pair recorded in the private audit artifacts."""
    out = []
    g84 = REPO / "artifacts" / "g84_result.json"
    g85 = REPO / "artifacts" / "g85_result.json"
    if g84.is_file():
        cal = json.loads(g84.read_text(encoding="utf-8"))["calibration"]
        for k in ("tau1", "tau05"):
            if cal.get(k):
                out.append((f"development/{k}", cal[k]))
    if g85.is_file():
        d = json.loads(g85.read_text(encoding="utf-8"))
        for name, key in (("holdout", "confirmation"), ("pooled", "pooled_all_five_folds")):
            for k in ("t10", "t05"):
                if d[key].get(k):
                    out.append((f"{name}/{k}", d[key][k]))
    return out


# ---------------------------------------------------------------- 1. arithmetic identity
def test_delta_u_is_mean_of_component_deltas() -> None:
    """Each recorded delta-U must equal the arithmetic mean of its six component deltas."""
    blocks = _subsets()
    if not blocks:
        print("  skip  private artifacts absent (public export); JSON identity still checked")
    for label, b in blocks:
        reported = b["delta_U_common"]
        deltas = b["component_deltas"]
        recomputed = sum(deltas[c] for c in COMPONENTS) / 6
        check(f"delta_U identity [{label}]", abs(reported - recomputed) < TOL,
              f"reported {reported!r} vs mean-of-deltas {recomputed!r}")
        # And the deltas themselves must be candidate-minus-baseline of the raw means.
        base, cand = b["baseline_means"], b["candidate_means"]
        worst = max(abs((cand[c] - base[c]) - deltas[c]) for c in COMPONENTS)
        check(f"component deltas are cand-minus-base [{label}]", worst < TOL, f"worst {worst!r}")


def test_rank_definition_would_not_reproduce_the_data() -> None:
    """A fractional-rank utility is falsifiable here: it would give +1.000 everywhere."""
    for label, b in _subsets():
        base, cand = b["baseline_means"], b["candidate_means"]
        wins = sum(1 for c in COMPONENTS if cand[c] > base[c])
        rank_delta = (wins - (6 - wins)) / 6
        check(f"rank utility is refuted [{label}]",
              abs(rank_delta - b["delta_U_common"]) > 1e-6,
              "a rank definition happens to match; re-derive the definition")


def test_public_json_carries_the_identity() -> None:
    """The published summary must state the raw-mean definition and carry the recomputation."""
    path = REPO / "evidence" / "policy_audit_summary.json"
    if not path.is_file():
        print("  skip  evidence/policy_audit_summary.json not present in this tree")
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    utility = d.get("utility", "").lower()
    check("public summary says raw component means",
          "raw" in utility and "mean" in utility, utility[:90])
    check("public summary denies a rank statistic",
          "not a rank statistic" in utility, utility[:90])
    for label, sub in d["subsets"].items():
        for tau, e in sub.items():
            got, recomputed = e["delta_U_common_support"], \
                e["delta_U_recomputed_as_mean_of_component_deltas"]
            check(f"public identity [{label}/{tau}]", abs(got - recomputed) < TOL,
                  f"{got!r} vs {recomputed!r}")


def test_component_csv_reproduces_delta_u() -> None:
    """delta-U must be reconstructible from the published per-component CSV alone."""
    path = REPO / "evidence" / "policy_audit_components.csv"
    summary = REPO / "evidence" / "policy_audit_summary.json"
    if not (path.is_file() and summary.is_file()):
        print("  skip  public component CSV not present in this tree")
        return
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    by: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        by.setdefault((r["subset"], r["nsd_tolerance"]), []).append(float(r["delta"]))
    d = json.loads(summary.read_text(encoding="utf-8"))["subsets"]
    for (subset, tau), deltas in by.items():
        check(f"csv has six components [{subset}/{tau}]", len(deltas) == 6, str(len(deltas)))
        entry = d[subset][f"tau_{tau}"]
        check(f"csv reproduces delta_U [{subset}/{tau}]",
              abs(sum(deltas) / 6 - entry["delta_U_common_support"]) < 1e-9)


# ------------------------------------------------- 2. no fractional-rank description of Audit C
def test_no_fractional_rank_description_of_audit_c() -> None:
    """Audit C must never again be described as a fractional-rank utility."""
    targets = [REPO / "paper" / "main.tex", REPO / "paper" / "supplement.tex",
               REPO / "evidence" / "README.md", REPO / "CAMERA_READY_REVISION.md",
               REPO / "public" / "CAMERA_READY_REVISION.md"]
    bad = re.compile(
        r"U(?:_?\\?tau)?\s*(?:\([^)]*\))?\s*is\s+the\s+(?:unweighted\s+)?mean\s+of\s+"
        r"(?:the\s+)?six\s+fractional\s+ranks", re.I)
    for path in targets:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        check(f"no rank-utility wording in {path.name}", not bad.search(text),
              (bad.search(text).group(0) if bad.search(text) else ""))
        # The audit utility must be introduced as a raw-metric mean somewhere in the science text.
        if path.name in ("main.tex", "supplement.tex"):
            check(f"{path.name} states the raw-mean definition",
                  ("raw" in text and re.search(r"U_\\tau", text) is not None))


def test_rank_statistic_kept_separate() -> None:
    """The architecture statistic must be named R and kept distinct from U_tau."""
    for name in ("main.tex", "supplement.tex"):
        path = REPO / "paper" / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        check(f"{name} names the architecture statistic R", re.search(r"\$R\$", text) is not None)
        check(f"{name} separates R from U_tau",
              re.search(r"(different object|never combined|not a rank statistic)", text) is not None)


# ------------------------------------------------- 3. no unsupported bootstrap values
def test_development_tau05_bootstrap_is_absent() -> None:
    """The committed record holds no development bootstrap at tau=0.5; nothing may invent one."""
    g84 = REPO / "artifacts" / "g84_result.json"
    if g84.is_file():
        cal = json.loads(g84.read_text(encoding="utf-8"))["calibration"]
        check("g84 development tau=0.5 has no bootstrap",
              not cal.get("tau05", {}).get("bootstrap"))
    summary = REPO / "evidence" / "policy_audit_summary.json"
    if summary.is_file():
        dev = json.loads(summary.read_text(encoding="utf-8"))["subsets"]["development_folds_0_2"]["tau_0.5"]
        check("public dev tau=0.5 positive fraction is null",
              dev["bootstrap_positive_fraction"] is None, repr(dev["bootstrap_positive_fraction"]))
        check("public dev tau=0.5 interval is null",
              dev["bootstrap_ci95_percentile"] is None, repr(dev["bootstrap_ci95_percentile"]))
        check("public dev tau=0.5 flagged unavailable", dev["bootstrap_available"] is False)


def test_manuscript_does_not_claim_a_dev_tau05_bootstrap() -> None:
    """No prose or table may report a development tau=0.5 positive fraction or interval."""
    for name in ("main.tex", "supplement.tex"):
        path = REPO / "paper" / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        check(f"{name} has no '1.000 / 1.000' development fraction",
              "$1.000$ / $1.000$" not in text and "1.000 / 1.000" not in text)
        # An unqualified "every bootstrap resample was positive" would overstate the tau=0.5 case.
        for m in re.finditer(r"every[^.]{0,40}bootstrap resample[^.]{0,40}positive", text, re.I):
            window = text[max(0, m.start() - 160):m.end() + 60]
            check(f"{name} scopes 'every resample positive' to tau=1",
                  r"\tau=1" in window, m.group(0))


# ------------------------------------------------- 4. tolerance roles
def test_tau1_is_described_as_official_ranking_aligned() -> None:
    """tau=1 is the final-ranking tolerance; tau=0.5 is a sensitivity analysis."""
    for name in ("main.tex", "supplement.tex"):
        path = REPO / "paper" / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").lower()
        check(f"{name} names tau=1 as the ranking tolerance",
              "official-ranking" in text or "final ranking uses" in text)
        check(f"{name} calls tau=0.5 a sensitivity analysis", "sensitivity analysis" in text)
        check(f"{name} drops the 'tolerance never exposed' claim",
              "never exposed" not in text)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{'FAIL' if failures else 'PASS'} -- {len(failures)} failing check(s)")
    if failures:
        for f in failures:
            print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
