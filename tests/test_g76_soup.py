#!/usr/bin/env python3
"""GAT-26 G7.6 checkpoint-soup regression tests (`python3 tests/test_g76_soup.py`).

Covers, without needing torch, a GPU, checkpoints, or protected data:
  * soup construction arithmetic for the two (and only two) predeclared ratios;
  * `_orig_mod.` key normalization and collision detection;
  * fail-closed behaviour on key-set / shape / dtype-class mismatch;
  * non-floating buffers are never averaged and must match exactly;
  * optimizer / scheduler / scaler / logger / epoch state never enters a soup checkpoint;
  * determinism (same inputs -> identical outputs, and order independence);
  * calibration/confirmation isolation and fail-closed selection in the G7.6 decision record.

A tiny tensor stub mirrors the torch API surface `g76_soup` uses, so the pure logic is testable in
the stdlib-only environment the rest of the governance suite runs in.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import g76_soup as S  # noqa: E402

FAILS = 0


def check(name, cond):
    global FAILS
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILS += 1


class T:
    """Minimal tensor stub: enough of the torch API for build_soup()."""

    def __init__(self, values, shape=None, dtype="float32"):
        self.values = list(values)
        self.shape = tuple(shape) if shape is not None else (len(self.values),)
        self.dtype = dtype

    def is_floating_point(self):
        return self.dtype.startswith("float")

    def double(self):
        return T([float(v) for v in self.values], self.shape, "float64")

    def to(self, dtype):
        if dtype.startswith("int") or dtype == "bool":
            return T([int(v) for v in self.values], self.shape, dtype)
        return T([float(v) for v in self.values], self.shape, dtype)

    def clone(self):
        return T(list(self.values), self.shape, self.dtype)

    def __mul__(self, k):
        return T([v * k for v in self.values], self.shape, self.dtype)

    def __add__(self, o):
        return T([a + b for a, b in zip(self.values, o.values)], self.shape, self.dtype)

    def __eq__(self, o):
        return isinstance(o, T) and self.values == o.values and self.dtype == o.dtype

    def __repr__(self):
        return f"T({self.values},{self.dtype})"


def approx(xs, ys, tol=1e-12):
    return len(xs) == len(ys) and all(abs(a - b) <= tol for a, b in zip(xs, ys))


def expect_fail(fn, *a, **k):
    try:
        fn(*a, **k)
    except S.SoupFailure:
        return True
    except Exception:
        return False
    return False


def main():
    print("test_g76_soup:")

    # --- ratios: exactly the two predeclared candidates, nothing else ------------------
    check("only_two_ratios", sorted(S.RATIOS) == ["S1", "S2"])
    check("S1_is_75_25", S.RATIOS["S1"] == (0.75, 0.25))
    check("S2_is_50_50", S.RATIOS["S2"] == (0.50, 0.50))
    check("ratios_sum_to_one", all(abs(a + b - 1.0) < 1e-12 for a, b in S.RATIOS.values()))

    fin = {"w": T([1.0, 2.0, 3.0]), "b": T([10.0])}
    bes = {"w": T([3.0, 4.0, 7.0]), "b": T([20.0])}

    s1 = S.build_soup(fin, bes, "S1")
    s2 = S.build_soup(fin, bes, "S2")
    check("S1_arithmetic", approx(s1["w"].values, [1.5, 2.5, 4.0]))
    check("S2_arithmetic", approx(s2["w"].values, [2.0, 3.0, 5.0]))
    check("S1_differs_from_S2", s1["w"].values != s2["w"].values)
    check("soup_differs_from_both_parents",
          s1["w"].values != fin["w"].values and s1["w"].values != bes["w"].values)
    check("dtype_restored_to_source", s1["w"].dtype == "float32")
    check("inputs_not_mutated",
          fin["w"].values == [1.0, 2.0, 3.0] and bes["w"].values == [3.0, 4.0, 7.0])

    # --- determinism -------------------------------------------------------------------
    check("deterministic_repeat", S.build_soup(fin, bes, "S1")["w"].values == s1["w"].values)
    rev_fin = {k: fin[k] for k in reversed(list(fin))}
    check("key_order_independent", S.build_soup(rev_fin, bes, "S1")["w"].values == s1["w"].values)

    # --- _orig_mod. normalization ------------------------------------------------------
    comp = {"_orig_mod.w": T([1.0, 2.0, 3.0]), "_orig_mod.b": T([10.0])}
    check("orig_mod_normalized", S.build_soup(comp, bes, "S1")["w"].values == s1["w"].values)
    check("orig_mod_collision_fails",
          expect_fail(S.normalize_keys, {"_orig_mod.w": T([1.0]), "w": T([2.0])}))

    # --- fail-closed structural guards -------------------------------------------------
    check("unknown_ratio_fails", expect_fail(S.build_soup, fin, bes, "S3"))
    check("interpolated_ratio_fails", expect_fail(S.build_soup, fin, bes, "S1.5"))
    check("key_set_mismatch_fails",
          expect_fail(S.build_soup, fin, {"w": T([3.0, 4.0, 7.0])}, "S1"))
    check("extra_key_fails",
          expect_fail(S.build_soup, fin, dict(bes, extra=T([1.0])), "S1"))
    check("shape_mismatch_fails",
          expect_fail(S.build_soup, {"w": T([1.0, 2.0])}, {"w": T([1.0, 2.0, 3.0])}, "S1"))
    check("dtype_class_mismatch_fails",
          expect_fail(S.build_soup, {"w": T([1.0])}, {"w": T([1], dtype="int64")}, "S1"))

    # --- non-floating buffers: identical only, never averaged --------------------------
    same_int = ({"n": T([7], dtype="int64"), "w": T([1.0])},
                {"n": T([7], dtype="int64"), "w": T([3.0])})
    got = S.build_soup(*same_int, "S2")
    check("identical_int_buffer_passes_through", got["n"].values == [7] and got["n"].dtype == "int64")
    check("int_buffer_not_averaged", got["n"].values == [7])
    check("differing_int_buffer_fails_closed",
          expect_fail(S.build_soup, {"n": T([7], dtype="int64"), "w": T([1.0])},
                      {"n": T([9], dtype="int64"), "w": T([3.0])}, "S2"))
    check("differing_bool_buffer_fails_closed",
          expect_fail(S.build_soup, {"f": T([1], dtype="bool")}, {"f": T([0], dtype="bool")}, "S1"))

    # --- no training state in the soup checkpoint --------------------------------------
    ck = S.soup_checkpoint(s1, "S1", {"fold": 0})
    check("checkpoint_has_network_weights", "network_weights" in ck)
    check("checkpoint_records_ratio", ck["gat26_soup"]["ratio_name"] == "S1")
    check("checkpoint_has_no_training_state",
          not any(k in ck for k in S.FORBIDDEN_CHECKPOINT_KEYS))
    check("forbidden_list_covers_optimizer", "optimizer_state" in S.FORBIDDEN_CHECKPOINT_KEYS)
    check("forbidden_list_covers_scheduler", "lr_scheduler_state" in S.FORBIDDEN_CHECKPOINT_KEYS)
    check("forbidden_list_covers_scaler", "grad_scaler_state" in S.FORBIDDEN_CHECKPOINT_KEYS)
    check("forbidden_list_covers_logger", "logging" in S.FORBIDDEN_CHECKPOINT_KEYS)
    check("forbidden_list_covers_epoch", "current_epoch" in S.FORBIDDEN_CHECKPOINT_KEYS)

    # --- strict loading is mandatory in every soup-consuming code path ------------------
    src = (REPO / "scripts" / "g76_soup.py").read_text()
    check("soup_module_never_uses_strict_false", "strict=False" not in src)
    infer = (REPO / "scripts" / "release_infer.py").read_text()
    check("release_runner_never_uses_strict_false", "strict=False" not in infer)

    # --- decision record: calibration/confirmation isolation + fail-closed selection ----
    dec_path = REPO / "artifacts" / "g76_checkpoint_soup_decision.json"
    if dec_path.exists():
        d = json.loads(dec_path.read_text())
        design = d.get("design", {})
        check("decision_is_one_of_three",
              d.get("decision") in ("G76_RETAIN_C0", "G76_SELECT_S1", "G76_SELECT_S2"))
        check("calibration_folds_are_0_1_2", design.get("calibration_folds") == [0, 1, 2])
        check("confirmation_folds_are_3_4", design.get("confirmation_folds") == [3, 4])
        check("calibration_and_confirmation_disjoint",
              not set(design.get("calibration_folds", [])) & set(design.get("confirmation_folds", [])))
        check("calibration_n_is_811", design.get("calibration_n") == 811)
        check("candidates_are_exactly_s1_s2", design.get("candidates") == ["S1", "S2"])
        check("bootstrap_seed_frozen", design.get("bootstrap_seed") == 21072026)
        check("bootstrap_resamples_frozen", design.get("bootstrap_resamples") == 10000)
        check("baseline_wins_ties", design.get("baseline_wins_ties") is True)
        check("policy_frozen_before_results", design.get("frozen_before_results") is True)
        # fail-closed: retaining C0 must not claim a confirmation it never ran
        if d.get("decision") == "G76_RETAIN_C0":
            check("retain_c0_declares_frozen_policy", d.get("final_policy") == "C0")
            check("retain_c0_no_unrun_confirmation_claim",
                  d.get("confirmation_run") in (True, False))
            if d.get("confirmation_run") is False:
                check("unrun_confirmation_has_no_results",
                      not d.get("confirmation", {}).get("results"))
        else:
            check("selected_candidate_ran_confirmation", d.get("confirmation_run") is True)
            check("selected_candidate_matches_decision",
                  d.get("decision", "").endswith(d.get("final_policy", "\x00")))
        check("no_case_ids_in_decision",
              not any(k in json.dumps(d) for k in ("BraTS-GoAT-", "BraTS-GLI-")))
    else:
        print("  ..   decision record not present yet (pre-decision run)")

    print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
