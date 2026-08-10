#!/usr/bin/env python3
"""G96 — the executable rank-gain threshold is a float comparison, and the paper says so.

Run: python3 tests/test_g96_threshold_disclosure.py

The frozen policy gates ResEnc-L advancement on `rank_gain >= MEANINGFUL_RANK_GAIN`, a binary64
comparison. That is not the same predicate as the real-number inequality R(M)-R(L) >= 1/6: a rank
configuration that is exactly 1/6 in rational arithmetic is computed as a difference of averaged
ranks and lands about 1.4e-16 below the stored threshold, so the nominal boundary case is rejected.

Earlier drafts asserted the exact real-number form. These tests (a) reproduce the discrepancy from
the committed module, (b) require the supplement to disclose it with the two concrete values, (c)
require the disclosure to say the recorded decision was unaffected, and (d) reject any renewed claim
of exact equivalence anywhere in the published text.

Nothing here modifies the policy: the module is imported read-only. No GPU, no challenge data, no
network.
"""
from __future__ import annotations

import re
import sys
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import g45_selection_policy as SP  # noqa: E402

failures: list[str] = []

STORED = "0.16666666666666666"      # repr of the double nearest 1/6
BOUNDARY = "0.16666666666666652"    # what the exactly-1/6 rank configuration computes to


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{'' if ok else ' — ' + detail}")
    if not ok:
        failures.append(name)


def _documents() -> list[Path]:
    return [p for p in (REPO / "paper" / "main.tex",
                        REPO / "paper" / "supplement.tex",
                        REPO / "public" / "CAMERA_READY_REVISION.md",
                        REPO / "CAMERA_READY_REVISION.md") if p.is_file()]


# ------------------------------------------------- 1. reproduce the discrepancy from the module
def test_the_stored_threshold_is_the_double_nearest_one_sixth() -> None:
    check("MEANINGFUL_RANK_GAIN is nominally 1/6",
          Fraction(SP.MEANINGFUL_RANK_GAIN).limit_denominator(1000) == Fraction(1, 6))
    check(f"its stored repr is {STORED}", repr(SP.MEANINGFUL_RANK_GAIN) == STORED,
          repr(SP.MEANINGFUL_RANK_GAIN))


def test_the_exactly_one_sixth_configuration_falls_below_the_stored_threshold() -> None:
    """L ahead on three components, tied on one, behind on two: exactly 1/6 in exact arithmetic."""
    aggM = {("ET", "DSC"): 0.80, ("TC", "DSC"): 0.80, ("WT", "DSC"): 0.90,
            ("ET", "HD95"): 9.0, ("TC", "HD95"): 7.0, ("WT", "HD95"): 5.0}
    aggL = {("ET", "DSC"): 0.90, ("TC", "DSC"): 0.90, ("WT", "DSC"): 0.90,
            ("ET", "HD95"): 5.0, ("TC", "HD95"): 9.0, ("WT", "HD95"): 9.0}
    r = SP.award_ranks({"M": aggM, "L": aggL})
    gain = r["M"] - r["L"]
    check("the configuration is 1/6 in exact arithmetic", abs(gain - 1 / 6) < 1e-12, repr(gain))
    check(f"it computes to {BOUNDARY}", repr(gain) == BOUNDARY, repr(gain))
    check("the executable comparison rejects it", not (gain >= SP.MEANINGFUL_RANK_GAIN))
    check("the shortfall is about 1.4e-16",
          0 < SP.MEANINGFUL_RANK_GAIN - gain < 1e-15, repr(SP.MEANINGFUL_RANK_GAIN - gain))
    check("so the executable gate is NOT the real-number inequality",
          (gain >= SP.MEANINGFUL_RANK_GAIN) != (Fraction(1, 6) >= Fraction(1, 6)))


def test_the_recorded_decision_was_nowhere_near_the_boundary() -> None:
    """The fold-0 screen recorded a rank gain of -0.333: the wrong side by ~0.5, not by an ulp."""
    import json
    rs = REPO / "RUN_STATE.json"
    if not rs.is_file():
        print("  skip  RUN_STATE.json absent (public export); prose disclosure still checked")
        return
    hits = re.findall(r'"rank_gain_L_over_M":\s*(-?[\d.]+)', rs.read_text())
    check("a recorded fold-0 rank gain exists", bool(hits), "no rank_gain_L_over_M recorded")
    for v in hits:
        g = float(v)
        check(f"recorded gain {v} is far below the threshold",
              SP.MEANINGFUL_RANK_GAIN - g > 1e-3, v)


# ------------------------------------------------- 2. the disclosure is published
def test_supplement_discloses_the_binary64_comparison() -> None:
    path = REPO / "paper" / "supplement.tex"
    if not path.is_file():
        print("  skip  paper/supplement.tex not present in this tree")
        return
    text = " ".join(path.read_text().split())
    check("supplement names the executable comparison",
          "rank\\_gain >= MEANINGFUL\\_RANK\\_GAIN" in text or
          "rank\\_gain >= MEANINGFUL\\_RANK\\_GAIN" in text.replace("  ", " "),
          "executable comparison not quoted")
    check("supplement says binary64", "binary64" in text)
    check(f"supplement gives the stored threshold {STORED}", STORED in text)
    check(f"supplement gives the boundary value {BOUNDARY}", BOUNDARY in text)
    check("supplement denies exact equivalence",
          re.search(r"not\s+(?:an\s+)?exact\s+real-number", text) is not None
          or "not equivalent" in text, "no explicit denial of equivalence")
    check("supplement says the recorded decision was unaffected",
          re.search(r"did not affect the recorded architecture decision", text) is not None)
    check("supplement says the frozen policy code was not modified",
          re.search(r"did not modify the frozen policy code", text) is not None)


def test_no_document_claims_exact_real_number_equivalence() -> None:
    """No published text may reassert the exact inequality as the gate."""
    bad = re.compile(r"R\(\\?mathrm\{?M\}?\)\s*-\s*R\(\\?mathrm\{?L\}?\)\s*(?:\\ge|>=|≥)\s*1/6")
    for path in _documents():
        text = " ".join(path.read_text().split())
        rel = path.relative_to(REPO)
        for m in bad.finditer(text):
            window = text[max(0, m.start() - 220):m.end() + 220]
            # The supplement may quote the form only to say it is NOT what the code does.
            excused = ("not an exact real-number" in window) or ("not equivalent" in window)
            check(f"{rel} does not present the exact inequality as the gate", excused,
                  window[:200])
        else:
            check(f"{rel} scanned for the exact-inequality claim", True)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{'FAIL' if failures else 'PASS'} — {len(failures)} failing check(s)")
    for f in failures:
        print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
