#!/usr/bin/env python3
"""G94 — architecture rank statistic, tie conventions and supplement regeneration.

Run: python3 tests/test_g94_rank_statistic.py

The camera-ready supplement once described the architecture screen as resolving ties "to the smaller
model" and reported "the rank gain is $R=+0.333$" alongside the 95% interval of the *opposite-signed*
quantity. Both statements were wrong, and they were wrong in a way that is easy to reintroduce,
because the committed record genuinely carries two opposite sign conventions for the same result:

    rank_gain_L_over_M = R(M) - R(L) = +0.333        (companion field)
    bootstrap delta    = R(L) - R(M) = -0.333        (what the implementation computes)

`scripts/g45_selection_policy.py` is authoritative. These tests pin what it actually does, keep the
two distinct tie conventions apart, and require the published text to use one sign convention only.

They read only tracked repository files and import the committed policy module. No GPU, no challenge
data, no network.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import g45_selection_policy as SP  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{'' if ok else ' — ' + detail}")
    if not ok:
        failures.append(name)


def _agg(dsc: dict[str, float], hd95: dict[str, float]) -> dict:
    out = {(r, "DSC"): dsc[r] for r in ("ET", "TC", "WT")}
    out.update({(r, "HD95"): hd95[r] for r in ("ET", "TC", "WT")})
    return out


def _norm(text: str) -> str:
    """Collapse the LaTeX and Unicode spellings of the same statement onto one form."""
    text = text.replace("ΔR", r"\Delta R").replace("−", "-")
    text = text.replace(r"\Delta R", "DeltaR")
    text = text.replace(r"\mathrm{L}", "L").replace(r"\mathrm{M}", "M")
    text = text.replace("$", "")
    return re.sub(r"\s+", "", text)


def _definition_sites() -> list[Path]:
    """Files that define or restate the architecture statistic $R$."""
    return [p for p in (REPO / "paper" / "main.tex",
                        REPO / "paper" / "supplement.tex",
                        REPO / "scripts" / "g92_build_supplement.py") if p.is_file()]


def _revision_records() -> list[Path]:
    """The camera-ready revision record, at its private path and its exported public path."""
    return [p for p in (REPO / "public" / "CAMERA_READY_REVISION.md",
                        REPO / "CAMERA_READY_REVISION.md") if p.is_file()]


# ------------------------------------------------- 1. component-level ties average tied positions
def test_exact_component_tie_gives_both_models_the_average_tied_rank() -> None:
    """An exactly tied component must contribute 1.5 to each of the two models, never 1 and 2."""
    # Both WT components are exactly equal; the other four are won outright, two each, so any
    # tie-break that favoured a model would pull the two mean ranks apart.
    aggM = _agg({"ET": 0.90, "TC": 0.80, "WT": 0.93}, {"ET": 5.0, "TC": 9.0, "WT": 7.0})
    aggL = _agg({"ET": 0.80, "TC": 0.90, "WT": 0.93}, {"ET": 9.0, "TC": 5.0, "WT": 7.0})
    r = SP.award_ranks({"M": aggM, "L": aggL})

    # Isolate the tied component: rank it on its own and require 1.5 for both models.
    tied_only = {m: {("WT", "DSC"): a[("WT", "DSC")]} for m, a in (("M", aggM), ("L", aggL))}
    saved = SP.COMPONENTS
    try:
        SP.COMPONENTS = [("WT", "DSC")]
        tied = SP.award_ranks(tied_only)
    finally:
        SP.COMPONENTS = saved
    check("tied component ranks M at the average position", tied["M"] == 1.5, repr(tied["M"]))
    check("tied component ranks L at the average position", tied["L"] == 1.5, repr(tied["L"]))
    check("a tie cannot favour either model", tied["M"] == tied["L"])

    # And with three outright wins each plus the tie, the two mean ranks must come out equal.
    check("balanced wins plus one tie leaves the mean ranks equal",
          abs(r["M"] - r["L"]) < 1e-12, f"M={r['M']!r} L={r['L']!r}")
    check("the tie is not silently broken toward the smaller model", not r["M"] < r["L"],
          f"M={r['M']!r} L={r['L']!r}")


# ------------------------------------------------- 2. a final decision tie retains ResEnc-M
def test_final_decision_tie_retains_resenc_m() -> None:
    """Rank arithmetic averages ties; the *decision* rule resolves a tie to M. Different rules."""
    policy = SP.build_policy()
    check("the frozen tie rule names M", "select_M" in policy["tie_rule"]["rule"],
          policy["tie_rule"]["rule"])
    check("M is declared the tie winner", policy["m_decision_rule"]["M_is_baseline_control_and_tie_winner"] is True)
    check("'tie' is a listed select_M condition", "tie" in policy["m_decision_rule"]["select_M_if"])

    # Exercise the decision engine on a genuine tie: identical metrics for both models.
    same = {"ET": 0.90, "TC": 0.85, "WT": 0.92}
    hd = {"ET": 6.0, "TC": 7.0, "WT": 8.0}

    def recs(model):
        return [{"subject": f"s{i:03d}", "evaluated_ok": True, "metrics": _agg(same, hd)}
                for i in range(12)]

    extra = {f"{k}_{m}": v for k, v in (("smallest_volume_dsc", 0.80), ("dsc_p05", 0.70),
                                        ("hd95_p95", 12.0), ("empty_ref_fp", 0.01),
                                        ("missed_region", 0.02), ("runtime", 100.0),
                                        ("cost", 10.0)) for m, v in (("M", v), ("L", v))}
    decision = SP.decide_after_fold0({"M": recs("M"), "L": recs("L"), "extra": extra})
    check("an exact decision tie selects M", decision["decision"] == "select_M",
          decision["decision"])
    check("a tie is not a meaningful L benefit",
          decision["fold0"]["meaningful_L_benefit"] is False)


# ------------------------------------------------- 3. one sign convention in the published text
def test_delta_r_sign_and_interval_are_stated_consistently() -> None:
    """Every published restatement must carry DeltaR = R(L)-R(M) = -0.333 and [-1.000,+0.667]."""
    want_point = "DeltaR=R(L)-R(M)=-0.333"
    want_ci = "[-1.000,+0.667]"
    for path in _definition_sites() + _revision_records():
        text = _norm(path.read_text())
        rel = path.relative_to(REPO)
        check(f"{rel} states the signed point estimate", want_point in text)
        check(f"{rel} states the percentile interval", want_ci in text)
        # The opposite convention must never appear attached to the same interval.
        check(f"{rel} does not restate the point as +0.333",
              "DeltaR=R(L)-R(M)=+0.333" not in text)


def test_implementation_confirms_the_reported_sign() -> None:
    """The engine's delta really is rank_L - rank_M, so negative favours L."""
    policy = SP.build_policy()
    check("the policy declares delta = rank_L - rank_M",
          policy["bootstrap"]["delta_definition"] == "delta = rank_L - rank_M (negative favors L)",
          policy["bootstrap"]["delta_definition"])
    # L strictly better on every component -> its mean rank is lower -> delta is negative.
    aggM = _agg({"ET": 0.80, "TC": 0.80, "WT": 0.80}, {"ET": 9.0, "TC": 9.0, "WT": 9.0})
    aggL = _agg({"ET": 0.90, "TC": 0.90, "WT": 0.90}, {"ET": 5.0, "TC": 5.0, "WT": 5.0})
    r = SP.award_ranks({"M": aggM, "L": aggL})
    check("lower R is better", r["L"] < r["M"], f"M={r['M']!r} L={r['L']!r}")
    check("delta is negative when L wins", (r["L"] - r["M"]) < 0)


# ------------------------------------------------- 4. the stale wording never returns
def test_stale_rank_wording_is_rejected() -> None:
    """The two withdrawn formulations must not reappear where $R$ is defined or restated."""
    stale = ("rank gain is $R=+0.333", "ties resolved to the smaller model")
    for path in _definition_sites():
        text = path.read_text()
        for phrase in stale:
            check(f"{path.relative_to(REPO)} is free of {phrase!r}", phrase not in text)


def test_the_two_tie_conventions_are_distinguished_in_the_supplement() -> None:
    """The supplement must separate rank-arithmetic ties from the selection tie rule."""
    path = REPO / "paper" / "supplement.tex"
    if not path.is_file():
        print("  skip  paper/supplement.tex not present in this tree")
        return
    text = path.read_text()
    check("supplement states the averaged tied positions",
          "average of the tied positions" in text)
    check("supplement states the separate selection tie rule",
          re.search(r"tied or unmet advancement criterion in favour of ResEnc-M", text) is not None)
    check("supplement says lower R is better", r"lower $R$ is better" in text)


# ------------------------------------------------- 5. the supplement is exactly what the script emits
def test_committed_supplement_is_byte_identical_to_a_fresh_regeneration() -> None:
    """The generated file must never be hand-edited: regenerating it must reproduce it exactly."""
    gen = REPO / "scripts" / "g92_build_supplement.py"
    committed = REPO / "paper" / "supplement.tex"
    if not (REPO / "artifacts" / "g84_result.json").is_file():
        print("  skip  artifacts/ absent (public export); regeneration cannot be exercised here")
        return
    if not (gen.is_file() and committed.is_file()):
        print("  skip  generator or supplement absent in this tree")
        return
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "supplement.tex"
        proc = subprocess.run([sys.executable, str(gen), str(REPO), str(out)],
                              capture_output=True, text=True)
        check("the generator runs cleanly", proc.returncode == 0, proc.stderr[-300:])
        if proc.returncode != 0:
            return
        fresh, have = out.read_bytes(), committed.read_bytes()
        check("regenerated supplement is byte-identical to the committed file",
              fresh == have, f"{len(fresh)} B fresh vs {len(have)} B committed")


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
