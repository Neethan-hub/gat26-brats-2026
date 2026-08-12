#!/usr/bin/env python3
"""r11: the documented supplement-regeneration command must work in a PUBLIC tree.

Run as `python3 tests/test_r11_supplement_regeneration.py`. Exit 0 on success, 1 on any failure.

`paper/README.md` documents:

    python3 scripts/g92_build_supplement.py . paper/supplement.tex

Before r11 that command only worked inside the private repository: the generator read
`artifacts/g84_result.json` and `artifacts/g85_result.json`, and `artifacts/` is never exported, so
in a sanitized public export the command died with an uncaught `FileNotFoundError`. The committed
`paper/supplement.tex` shipped as an unreproducible artifact.

`scripts/g91_public_evidence.py` now also emits `evidence/supplement_inputs.json`, a whitelisted
aggregate projection of exactly the fields the generator reads, at full float precision. This suite
pins the three properties that make the documented command honest:

  1. From a tree containing ONLY `evidence/` and `scripts/` -- no `artifacts/` anywhere -- the
     command succeeds and reproduces the committed `paper/supplement.tex` BYTE-FOR-BYTE.
  2. With neither input present the generator fails closed with an explicit diagnostic, rather than
     a traceback or a partial document.
  3. The published aggregate carries no per-case value, case identifier, fold membership,
     prediction, private path, credential or resource identifier.

Property 1 is the one that matters: it is the difference between a reproducibility claim and a
reproducibility fact.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GENERATOR = REPO / "scripts" / "g92_build_supplement.py"
EVIDENCE = REPO / "evidence" / "supplement_inputs.json"
COMMITTED = REPO / "paper" / "supplement.tex"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def _run(root: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(root / "scripts" / "g92_build_supplement.py"), str(root), str(out)],
        capture_output=True, text=True, cwd=str(root),
    )


def test_public_tree_reproduces_the_committed_supplement() -> None:
    """The whole point: a tree with no artifacts/ must rebuild the shipped file exactly."""
    for required in (GENERATOR, EVIDENCE, COMMITTED):
        if not required.is_file():
            check(f"required input present: {required.relative_to(REPO)}", False)
            return
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "public"
        (root / "evidence").mkdir(parents=True)
        (root / "scripts").mkdir(parents=True)
        (root / "paper").mkdir(parents=True)
        shutil.copy2(EVIDENCE, root / "evidence" / EVIDENCE.name)
        shutil.copy2(GENERATOR, root / "scripts" / GENERATOR.name)

        check("public fixture has no artifacts/", not (root / "artifacts").exists())

        out = root / "paper" / "supplement.tex"
        proc = _run(root, out)
        check("generator exits 0 in a public tree", proc.returncode == 0,
              (proc.stderr or proc.stdout)[-400:])
        if proc.returncode != 0:
            return
        check("generator reports the public input source",
              "public aggregate evidence" in proc.stdout, proc.stdout.strip()[:200])
        check("output written", out.is_file())
        if not out.is_file():
            return
        produced = out.read_bytes()
        committed = COMMITTED.read_bytes()
        check("public regeneration is BYTE-IDENTICAL to committed paper/supplement.tex",
              produced == committed,
              f"produced {len(produced)} B, committed {len(committed)} B")


def test_generator_fails_closed_without_inputs() -> None:
    """No inputs at all must produce a diagnostic, not a traceback and not a partial document."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "empty"
        (root / "scripts").mkdir(parents=True)
        (root / "paper").mkdir(parents=True)
        shutil.copy2(GENERATOR, root / "scripts" / GENERATOR.name)
        out = root / "paper" / "supplement.tex"
        proc = _run(root, out)
        check("generator refuses to run with no inputs", proc.returncode != 0)
        combined = proc.stdout + proc.stderr
        check("failure names both accepted input paths",
              "g84_result.json" in combined and "supplement_inputs.json" in combined,
              combined[-300:])
        check("failure is a diagnostic, not a raw traceback",
              "Traceback (most recent call last)" not in combined, combined[-300:])
        check("no partial document is written on failure", not out.exists())


def test_published_aggregate_carries_no_private_data() -> None:
    """The projection is aggregate-only by construction; this pins it against drift."""
    if not EVIDENCE.is_file():
        check("evidence/supplement_inputs.json present", False)
        return
    raw = EVIDENCE.read_text(encoding="utf-8")
    payload = json.loads(raw)

    forbidden = {
        "BraTS case identifier": r"\bBraTS[-_](?:GoAT|GLI|MEN|MET|PED|SSA)[-_]\d{4,}\b",
        "absolute worker path": r"/workspace|/dev/shm|/root/",
        "credential-shaped token": r"ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}",
        "provider endpoint": r"\b[\w.-]+\.(?:runpod\.(?:io|net)|amazonaws\.com)\b",
        "image digest": r"sha256:[0-9a-f]{16,}",
        # Assembled from fragments, exactly as scripts/make_code_export.py does for the same
        # rule: spelling these identifiers out here would publish, in this very file, the values
        # the check exists to keep out of the export.
        "protected resource id": r"\b" + "rezirf" + r"6tzr4vle\b|\b" + "lsxmgd" + r"efp6\b",
    }
    for label, pattern in forbidden.items():
        hits = re.findall(pattern, raw)
        check(f"no {label} in the published aggregate", not hits, str(hits[:3]))

    # Structural: the projection must be a fixed shape, not an accidental full dump.
    check("top-level keys are exactly the documented projection",
          set(payload) == {"note", "g84", "g85", "audit_ab", "screen_g82", "d25", "provenance"},
          str(sorted(payload)))
    g84_keys = set(payload.get("g84", {}))
    check("g84 projection is limited to the documented fields",
          g84_keys == {"common_support", "n_gates_total", "n_gates_passed", "gates",
                       "calibration", "lesion"}, str(sorted(g84_keys)))
    g85_keys = set(payload.get("g85", {}))
    check("g85 projection is limited to the documented fields",
          g85_keys == {"confirmation", "pooled_all_five_folds", "confirmation_gates",
                       "pooled_gates"}, str(sorted(g85_keys)))

    # A per-case array would be far larger than any aggregate this file should carry.
    def widest(node) -> int:
        if isinstance(node, list):
            return max([len(node)] + [widest(v) for v in node], default=len(node))
        if isinstance(node, dict):
            return max([widest(v) for v in node.values()], default=0)
        return 0
    check("no array is long enough to be per-case data", widest(payload) <= 24,
          f"widest array = {widest(payload)}")

    prov = payload.get("provenance", {})
    check("provenance ledger present", bool(prov.get("entries")))
    check("every ledger entry names a source record and key path",
          all(e.get("source_record") and e.get("source_key_path")
              for e in prov.get("entries", [])))
    check("ledger records the quantities that were never measured",
          bool(prov.get("values_that_do_not_exist")))

    # The introductory note is the first thing a reader of the ledger sees. If it names only one
    # class of derived value, a reader scanning the note rather than every entry will take the
    # other for a direct measurement -- which is precisely the confusion the per-entry derived
    # flags exist to prevent.
    note = prov.get("note", "")
    check("provenance note names the component-delta derivation",
          "candidate-minus-C0 component deltas" in note, note[:120])
    check("provenance note names the rank-gain interval derivation",
          "rank_gain_ci = [-raw_high, -raw_low]" in note, note[:120])
    check("provenance note says the frozen interval is the opposite orientation",
          "opposite-orientation" in note, note[:120])
    check("provenance note states both categories are marked derived",
          "derived: true" in note, note[:120])


# The eight displayed rows of supplement Table S18, pinned as literals. These are what a reader
# sees; if the transformation or the frozen record ever drifts, this table is what must fail.
EXPECTED_S18 = {
    "C1":      (-1.000, (-1.000, -0.333)),
    "C2":      (+1.000, (-0.667, +1.000)),
    "C3":      (-0.333, (-1.000, +1.000)),
    "C0_et10": (+0.333, (-0.333, +0.333)),
    "C0_et25": (+0.333, (-0.333, +0.333)),
    "C0_et50": (+0.333, (-0.333, +0.333)),
    "S1":      (-1.000, (-1.000, -0.667)),
    "S2":      (-1.000, (-1.000, -0.333)),
}
TOL = 5e-4


def test_rank_gain_interval_orientation() -> None:
    """The displayed interval must share the point estimate's orientation.

    rank_gain_over_C0 is R(C0) - R(candidate), so positive favours the candidate. The frozen
    bootstrap interval is stored in the OPPOSITE orientation, R(candidate) - R(C0). Publishing the
    two side by side unconverted reads as though a negative point estimate came with a positive
    interval, which is how the defect this test exists to prevent was introduced.

    The conversion is a negation, and negating an interval reverses its endpoints:
        rank_gain_ci = [-hi, -lo]   from   bootstrap_delta_ci_candidate_minus_C0 = [lo, hi]
    """
    if not EVIDENCE.is_file():
        check("evidence/supplement_inputs.json present", False)
        return
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    ab = payload.get("audit_ab", {})
    stats = ab.get("statistics_tau_1.0", {})
    conv = ab.get("statistics_sign_convention", {})

    check("sign convention is documented in the published evidence",
          all(k in conv for k in ("rank_gain_over_C0", "rank_gain_ci",
                                  "bootstrap_delta_ci_candidate_minus_C0")))
    check("the documented convention names the candidate-favouring direction",
          "POSITIVE favours the candidate" in conv.get("rank_gain_over_C0", ""))

    for name, st in sorted(stats.items()):
        raw = st.get("bootstrap_delta_ci_candidate_minus_C0")
        got = st.get("rank_gain_ci")
        check(f"{name}: both orientations published", bool(raw) and bool(got))
        if not (raw and got):
            continue
        check(f"{name}: rank_gain_ci == [-hi, -lo] of the frozen interval",
              abs(got[0] + raw[1]) < 1e-12 and abs(got[1] + raw[0]) < 1e-12,
              f"raw={raw} derived={got}")
        check(f"{name}: derived interval is well-ordered", got[0] <= got[1], str(got))
        # A negation must not silently become the identity when the interval is asymmetric.
        if abs(raw[0] + raw[1]) > 1e-12:
            check(f"{name}: asymmetric interval actually changed orientation", got != raw,
                  f"raw={raw} derived={got}")

    for name, (rg, ci) in EXPECTED_S18.items():
        st = stats.get(name)
        check(f"{name}: displayed values match the pinned Table S18 row", bool(st))
        if not st:
            continue
        check(f"{name}: point estimate {rg:+.3f}",
              abs(st["rank_gain_over_C0"] - rg) < TOL, str(st.get("rank_gain_over_C0")))
        check(f"{name}: interval [{ci[0]:+.3f},{ci[1]:+.3f}]",
              abs(st["rank_gain_ci"][0] - ci[0]) < TOL and abs(st["rank_gain_ci"][1] - ci[1]) < TOL,
              str(st.get("rank_gain_ci")))

    # The provenance ledger must not be able to describe the displayed interval as raw. A reader
    # who trusts "derived: false" would take the shipped interval for a frozen measurement; it is a
    # negation of one, and the ledger is the only place that distinction is recorded.
    prov = payload.get("provenance", {}).get("entries", [])
    displayed = [e for e in prov
                 if e.get("public_projection_field", "").endswith("rank_gain_ci")]
    check("provenance names the displayed interval exactly once", len(displayed) == 1,
          f"{len(displayed)} entries")
    for e in displayed:
        check("the displayed interval is marked derived", e.get("derived") is True,
              str(e.get("derived")))
        check("the derivation formula is stated",
              e.get("derivation") == "rank_gain_ci = [-raw_high, -raw_low]",
              str(e.get("derivation")))
        check("the derivation names the frozen source record",
              e.get("source_record") == "artifacts/g79v_tau1_sensitivity_results.json",
              str(e.get("source_record")))
        check("the derivation names the frozen source field",
              "bootstrap_ci" in e.get("source_key_path", ""), str(e.get("source_key_path")))
    raw_entries = [e for e in prov
                   if "rank_gain_over_C0" in e.get("public_projection_field", "")
                   and not e.get("public_projection_field", "").endswith("rank_gain_ci")]
    check("the raw point estimate, frozen interval and advancement stay non-derived",
          bool(raw_entries) and all(e.get("derived") is False for e in raw_entries),
          str([e.get("derived") for e in raw_entries]))

    # The correction is presentational: every candidate must still fail to advance.
    check("no candidate advances", all(st.get("advances") is False for st in stats.values()),
          str({k: v.get("advances") for k, v in stats.items()}))


def test_table_s18_renders_the_converted_interval() -> None:
    """The rendered supplement must carry the converted interval and define the convention."""
    if not COMMITTED.is_file():
        check("paper/supplement.tex present", False)
        return
    tex = COMMITTED.read_text(encoding="utf-8")
    check("Table S18 caption defines the sign convention",
          "Sign convention:" in tex and
          r"R(\mathrm{C0})-R(\mathrm{candidate})" in tex)
    check("caption states the point and interval share an orientation",
          r"\emph{same} orientation as the point estimate" in tex)
    check("caption names where the unconverted interval is retained",
          r"bootstrap\_delta\_ci\_candidate\_minus\_C0" in tex)
    for name, (rg, ci) in EXPECTED_S18.items():
        row = "$%+.3f$ & $[%+.3f,%+.3f]$" % (rg, ci[0], ci[1])
        check(f"{name}: rendered row present", row in tex, row)


def test_generated_bytes_are_platform_independent() -> None:
    """The generator must emit bytes, not platform-translated text.

    Path.write_text() opens in text mode; on Windows that rewrites every LF to CRLF, so the same
    generator would produce a file that differs from the committed one on that platform alone.
    Writing encoded bytes removes the platform from the equation, which is what makes the
    byte-identity assertion above meaningful anywhere it is run.
    """
    if not GENERATOR.is_file():
        check("generator present", False)
        return
    src = GENERATOR.read_text(encoding="utf-8")
    check("generator writes bytes, not text", "out.write_bytes(" in src)
    check("generator does not write the document via write_text",
          'out.write_text(' not in src)
    check("generator encodes UTF-8 explicitly", '.encode("utf-8")' in src)
    if COMMITTED.is_file():
        raw = COMMITTED.read_bytes()
        check("committed supplement contains no CRLF", b"\r\n" not in raw,
              f"{raw.count(bytes([13, 10]))} CRLF sequences")
        check("committed supplement contains no bare CR", bytes([13]) not in raw,
              f"{raw.count(bytes([13]))} CR bytes")
        check("committed supplement decodes as UTF-8", _decodes_utf8(raw))


def test_injected_prose_is_latex_safe() -> None:
    """Prose copied from the evidence record into the .tex must not contain an unescaped percent.

    A bare "%" is a LaTeX comment: it silently swallows the rest of the source line and joins the
    next one, which truncated the convergence caveat mid-sentence ("40 epochs at 5" ... ) without
    producing any error, warning or overfull box. The build looked perfectly clean.
    """
    if not EVIDENCE.is_file() or not COMMITTED.is_file():
        check("inputs present for LaTeX-safety check", False)
        return
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    # Exactly the strings the generator injects VERBATIM into the .tex. Other fields in the
    # evidence record are rendered through formatting or are not rendered at all, and legitimately
    # contain underscores and carets; flagging those would be noise, and noise is how a real
    # finding gets ignored.
    prose = {
        "audit_ab.subset.evaluator": payload["audit_ab"]["subset"]["evaluator"],
        "screen_g82.convergence_caveat": payload["screen_g82"]["convergence_caveat"],
        "d25.note": payload["d25"]["note"],
    }
    tex_all = COMMITTED.read_text(encoding="utf-8")
    flat_tex = " ".join(tex_all.split())
    for name, text in prose.items():
        probe = " ".join(text.split())[:60]
        check(f"{name}: is in fact injected verbatim (guard stays honest)", probe in flat_tex)
    for name, text in prose.items():
        for ch, why in (("%", "comment"), ("&", "alignment tab"), ("#", "parameter")):
            bare = [m.start() for m in re.finditer(r"(?<!\\)" + re.escape(ch), text)]
            check(f"{name}: no unescaped {ch!r} ({why})", not bare, f"offsets {bare}")

    # End-to-end: the caveat must survive into the rendered source intact, not truncated.
    tex = COMMITTED.read_text(encoding="utf-8")
    tail = "the screen did not pass."
    check("convergence caveat reaches the .tex complete", tail in tex)
    check("caveat percentage renders escaped", r"5\% of the original learning rate" in tex)


def _decodes_utf8(raw: bytes) -> bool:
    try:
        raw.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def main() -> int:
    for name, fn in (
        ("public-tree regeneration", test_public_tree_reproduces_the_committed_supplement),
        ("fail-closed without inputs", test_generator_fails_closed_without_inputs),
        ("published aggregate is sanitized", test_published_aggregate_carries_no_private_data),
        ("rank-gain interval orientation", test_rank_gain_interval_orientation),
        ("Table S18 renders the converted interval", test_table_s18_renders_the_converted_interval),
        ("generated bytes are platform-independent", test_generated_bytes_are_platform_independent),
        ("injected prose is LaTeX-safe", test_injected_prose_is_latex_safe),
    ):
        print(f"\n{name}")
        fn()
    print(f"\n{'FAIL' if FAILURES else 'PASS'} -- {len(FAILURES)} failing check(s)")
    for f in FAILURES:
        print(f"  - {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
