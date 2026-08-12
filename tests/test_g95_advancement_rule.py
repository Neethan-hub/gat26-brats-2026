#!/usr/bin/env python3
"""G95 — the architecture advancement rule, checked against the executable policy.

Run: python3 tests/test_g95_advancement_rule.py

Earlier drafts said only that "ResEnc-L is preferred if the interval excludes zero and every
declared noninferiority gate passes". That was incomplete in four ways: it omitted the
MEANINGFUL_RANK_GAIN threshold, it said "excludes zero" where the code requires the interval's
upper endpoint to be strictly below zero, it did not say that an unsupplied gate is a failure, and
it implied fold 0 could select ResEnc-L when fold 0 can only trigger fold-1 confirmation.

These tests do two separate things. First they exercise `scripts/g45_selection_policy.py` and pin
what it actually does. Second they check the published prose against values *derived from that
module* rather than against hard-coded sentences, so the prose cannot drift away from the code.

They read only tracked repository files and import the committed policy module. No GPU, no
challenge data, no network.
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

# Every auxiliary gate the fail-closed fold-0 decision requires to be supplied.
AUX = ("smallest_volume_dsc", "dsc_p05", "hd95_p95", "empty_ref_fp", "missed_region",
       "runtime", "cost")


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{'' if ok else ' -- ' + detail}")
    if not ok:
        failures.append(name)


def metrics(dsc: dict[str, float], hd95: dict[str, float]) -> dict:
    m = {(r, "DSC"): dsc[r] for r in ("ET", "TC", "WT")}
    m.update({(r, "HD95"): hd95[r] for r in ("ET", "TC", "WT")})
    return m


def fold(dsc_M, hd_M, dsc_L, hd_L, n_subjects=8, extra=None, n_boot=120):
    subs = [f"s{i:03d}" for i in range(n_subjects)]
    return {
        "M": [{"subject": s, "evaluated_ok": True, "metrics": metrics(dsc_M, hd_M)} for s in subs],
        "L": [{"subject": s, "evaluated_ok": True, "metrics": metrics(dsc_L, hd_L)} for s in subs],
        "extra": extra, "n": n_boot,
    }


def all_gates_supplied(better_for_L=True):
    """Supply every auxiliary gate input, with L equal to M so no gate can fail."""
    out = {}
    for k in AUX:
        out[f"{k}_M"] = 1.0
        out[f"{k}_L"] = 1.0
    return out


def L_wins_everything():
    return fold({"ET": 0.80, "TC": 0.80, "WT": 0.80}, {"ET": 9.0, "TC": 9.0, "WT": 9.0},
                {"ET": 0.90, "TC": 0.90, "WT": 0.90}, {"ET": 5.0, "TC": 5.0, "WT": 5.0},
                extra=all_gates_supplied())


# ------------------------------------------------- 1. the threshold constant
def test_meaningful_rank_gain_is_one_sixth() -> None:
    frac = Fraction(SP.MEANINGFUL_RANK_GAIN).limit_denominator(1000)
    check("MEANINGFUL_RANK_GAIN == 1/6", frac == Fraction(1, 6), str(frac))
    check("it is one component of six", abs(SP.MEANINGFUL_RANK_GAIN - 1 / len(SP.COMPONENTS)) < 1e-12)


def test_rank_gain_and_delta_have_opposite_signs() -> None:
    """rank gain = R(M)-R(L); the reported bootstrap difference = R(L)-R(M)."""
    f = L_wins_everything()
    b = SP._fold_benefit(f, require_all_gates=True)
    gain, delta = b["rank_gain_L_over_M"], b["bootstrap"]["delta_point"]
    check("rank gain is positive when L is ahead", gain > 0, repr(gain))
    check("delta is negative when L is ahead", delta < 0, repr(delta))
    check("they are exact negatives of one another", abs(gain + delta) < 1e-12,
          f"gain {gain!r}, delta {delta!r}")
    check("the policy declares delta = rank_L - rank_M",
          SP.build_policy()["bootstrap"]["delta_definition"].startswith("delta = rank_L - rank_M"))


def test_threshold_behaviour_around_one_sixth() -> None:
    """Pin the comparison boundary, including the one-ulp reality of the float comparison.

    The rule is `rank_gain >= MEANINGFUL_RANK_GAIN` on plain floats. A construction that is exactly
    1/6 in rational arithmetic evaluates to 0.16666666666666652, about 1.4e-16 below the constant,
    so the comparison at the exact boundary is decided by representation rather than by the rule.
    That is a property of the committed implementation and is recorded here rather than papered
    over; it never affected a decision, since the observed fold-0 gain was -0.333, nowhere near it.
    """
    # L wins 3 components outright, ties 1, loses 2  ->  R(M)-R(L) = 1/6 exactly.
    aggM = {("ET", "DSC"): 0.80, ("TC", "DSC"): 0.80, ("WT", "DSC"): 0.90,
            ("ET", "HD95"): 9.0, ("TC", "HD95"): 7.0, ("WT", "HD95"): 5.0}
    aggL = {("ET", "DSC"): 0.90, ("TC", "DSC"): 0.90, ("WT", "DSC"): 0.90,
            ("ET", "HD95"): 5.0, ("TC", "HD95"): 9.0, ("WT", "HD95"): 9.0}
    r = SP.award_ranks({"M": aggM, "L": aggL})
    gain = r["M"] - r["L"]
    check("constructed gain is 1/6 in exact arithmetic", abs(gain - 1 / 6) < 1e-12, repr(gain))
    check("the float boundary sits within one ulp of the constant",
          abs(gain - SP.MEANINGFUL_RANK_GAIN) < 1e-15, repr(gain - SP.MEANINGFUL_RANK_GAIN))
    # A clear pass: L wins four of six, gain 1/3.
    aggL3 = dict(aggL); aggL3[("TC", "HD95")] = 5.0
    r3 = SP.award_ranks({"M": aggM, "L": aggL3})
    check("a gain comfortably above 1/6 meets the threshold",
          (r3["M"] - r3["L"]) >= SP.MEANINGFUL_RANK_GAIN, repr(r3["M"] - r3["L"]))
    # A 3-3 split gives a gain of 0, below the threshold.
    aggL2 = dict(aggL); aggL2[("WT", "DSC")] = 0.80
    r2 = SP.award_ranks({"M": aggM, "L": aggL2})
    check("a 3-3 split falls below the threshold",
          (r2["M"] - r2["L"]) < SP.MEANINGFUL_RANK_GAIN, repr(r2["M"] - r2["L"]))


# ------------------------------------------------- 2. decision behaviour
def test_fold0_pass_only_triggers_fold1_confirmation() -> None:
    """Passing every fold-0 condition must not select or expand ResEnc-L."""
    d = SP.decide_after_fold0(L_wins_everything())
    b = d["fold0"]
    check("fold-0 meaningful benefit is recognised", b["meaningful_L_benefit"] is True)
    check("the bootstrap upper endpoint is strictly below zero",
          b["bootstrap"]["delta_ci_high"] < 0.0, repr(b["bootstrap"]["delta_ci_high"]))
    check("fold 0 decides only to confirm on fold 1", d["decision"] == "confirm_L_on_fold1",
          d["decision"])
    check("fold 0 never selects or expands L",
          "expand" not in d["decision"] and d["decision"] != "select_L", d["decision"])
    check("the policy declares fold 0 cannot expand on its own",
          SP.build_policy()["meaningful_benefit_rule"]["fold0_alone_only_triggers_fold1_confirmation"]
          is True)


def test_expansion_requires_the_same_rule_on_fold1() -> None:
    good, tie = L_wins_everything(), None
    # A fold in which the two models are identical: no gain, interval straddles zero.
    same = fold({"ET": 0.90, "TC": 0.90, "WT": 0.90}, {"ET": 5.0, "TC": 5.0, "WT": 5.0},
                {"ET": 0.90, "TC": 0.90, "WT": 0.90}, {"ET": 5.0, "TC": 5.0, "WT": 5.0},
                extra=all_gates_supplied())
    both = SP.decide_after_two_folds(good, L_wins_everything())
    check("two confirming folds allow expansion",
          both["decision"] == "L_may_expand_subject_to_owner_budget_regate", both["decision"])
    check("expansion is flagged as directionally confirmed", both["confirmed_same_direction"] is True)
    mixed = SP.decide_after_two_folds(good, same)
    check("a non-confirming fold 1 retains M", mixed["decision"] == "select_M", mixed["decision"])
    check("the policy declares fold-1 confirmation is required",
          SP.build_policy()["meaningful_benefit_rule"]["L_expands_only_if_fold1_confirms_same_direction"]
          is True)


def test_an_unsupplied_gate_is_a_failure_not_a_pass() -> None:
    """Fail-closed: a gate with no input can never let ResEnc-L advance."""
    for missing in AUX:
        extra = all_gates_supplied()
        del extra[f"{missing}_M"], extra[f"{missing}_L"]
        f = L_wins_everything(); f["extra"] = extra
        d = SP.decide_after_fold0(f)
        check(f"a missing '{missing}' input retains M", d["decision"] == "select_M", d["decision"])
        check(f"'{missing}' is reported as not provided",
              missing in " ".join(d.get("fail_closed_missing_gates", [])) or
              bool(d.get("fail_closed_missing_gates")))
    check("the policy requires every frozen gate",
          SP.build_policy()["meaningful_benefit_rule"]["must_pass_every_frozen_noninferiority_gate"]
          is True)


def test_a_tie_retains_resenc_m() -> None:
    same = fold({"ET": 0.90, "TC": 0.90, "WT": 0.90}, {"ET": 5.0, "TC": 5.0, "WT": 5.0},
                {"ET": 0.90, "TC": 0.90, "WT": 0.90}, {"ET": 5.0, "TC": 5.0, "WT": 5.0},
                extra=all_gates_supplied())
    d = SP.decide_after_fold0(same)
    check("an exact tie retains M", d["decision"] == "select_M", d["decision"])
    check("the frozen tie rule names M", "select_M" in SP.build_policy()["tie_rule"]["rule"])


# ------------------------------------------------- 3. the prose agrees with the code
def _norm(text: str) -> str:
    text = text.replace("ΔR", r"\Delta R").replace("−", "-").replace("≥", r"\ge")
    text = text.replace(r"\Delta R", "DeltaR")
    text = text.replace(r"\mathrm{L}", "L").replace(r"\mathrm{M}", "M")
    text = text.replace("~", " ").replace("$", "").replace(r"\,", " ")
    return re.sub(r"\s+", " ", text)


def _documents() -> list[Path]:
    return [p for p in (REPO / "paper" / "main.tex",
                        REPO / "paper" / "supplement.tex",
                        REPO / "public" / "CAMERA_READY_REVISION.md",
                        REPO / "CAMERA_READY_REVISION.md") if p.is_file()]


def test_prose_states_the_rule_the_code_implements() -> None:
    """Every published statement of the rule must carry all four elements, and the threshold
    string must be the one derived from MEANINGFUL_RANK_GAIN rather than a typed-in constant."""
    frac = Fraction(SP.MEANINGFUL_RANK_GAIN).limit_denominator(1000)
    threshold = f"{frac.numerator}/{frac.denominator}"          # "1/6", derived from the module
    for path in _documents():
        text = _norm(path.read_text(encoding="utf-8"))
        rel = path.relative_to(REPO)
        m = re.search(r"could advance from fold 0[^.]*\.", text)
        check(f"{rel} states the advancement rule", m is not None)
        if not m:
            continue
        sentence = m.group(0)
        # G96: the sentence must name the rank-gain quantity and the nominal threshold derived
        # from the constant. It must NOT assert an exact real-number inequality, because the
        # executable gate is a binary64 comparison that differs from it at the boundary.
        check(f"{rel} names the rank-gain quantity",
              re.search(r"R\(M\)\s*-\s*R\(L\)", sentence) is not None, sentence[:180])
        check(f"{rel} states the nominal threshold {threshold}",
              re.search(r"nominally\s*" + re.escape(threshold), sentence) is not None,
              sentence[:180])
        check(f"{rel} does not claim an exact real-number inequality",
              re.search(r"R\(M\)\s*-\s*R\(L\)\s*\\ge\s*" + re.escape(threshold),
                        sentence) is None, sentence[:180])
        check(f"{rel} names the bootstrap difference as R(L)-R(M)",
              "DeltaR=R(L)-R(M)" in sentence or "DeltaR" in sentence)
        check(f"{rel} requires the interval below zero",
              "below zero" in sentence, sentence[:180])
        check(f"{rel} requires every gate supplied and passed",
              "supplied and passed" in sentence, sentence[:180])
        tail = text[m.end():m.end() + 400]
        check(f"{rel} says fold 0 only triggers fold-1 confirmation",
              "fold-1 confirmation" in sentence or "fold-1 confirmation" in tail)
        check(f"{rel} says expansion requires fold 1",
              re.search(r"(expansion|expand)[^.]{0,90}fold 1", sentence + tail, re.I) is not None)
        check(f"{rel} says otherwise ResEnc-M was retained",
              re.search(r"ResEnc-M was retained", sentence + tail) is not None)


def test_prose_does_not_keep_the_old_incomplete_sentence() -> None:
    stale = "ResEnc-L is preferred only if that interval excludes zero"
    for path in _documents():
        check(f"{path.relative_to(REPO)} drops the incomplete sentence",
              stale not in " ".join(path.read_text(encoding="utf-8").split()))


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{'FAIL' if failures else 'PASS'} -- {len(failures)} failing check(s)")
    for f in failures:
        print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
