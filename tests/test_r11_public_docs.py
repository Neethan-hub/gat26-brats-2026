#!/usr/bin/env python3
"""r11 public-documentation consistency tests (stdlib only).

Run as `python3 tests/test_r11_public_docs.py`. Exit 0 on success, 1 on any failure.

These are FAIL-CLOSED regression tests for the public-facing tree. They pin the six things the r11
documentation audit found wrong or unguarded:

  1. Memory wording. The runner bounds *model residency*, not total process peak. The overbroad
     phrasings must never come back to the public-facing README or to the release runner.
  2. Historical banners. Five dated pre-training / pre-submission records must each carry a
     conspicuous banner saying so, and must each restate the current truth.
  3. Current truth. The authoritative statements -- DSC and NSD at tau=1, HD95 diagnostic, complete
     opaque basename with no five-digit rule -- must be present where a reader looks first.
  4. Historical-string scoping. Legitimate historical DSC/HD95, five-digit, provider and A10G
     strings are ALLOWED, but only inside a record that is clearly labelled as archival. This test
     deliberately does NOT ban those strings globally; it bans them in unlabelled documents.
  5. Worktree exclusion. The exporter must enumerate tracked files only, so a registered harness
     worktree checked out under the hidden harness directory can never be traversed into an
     export.
  6. No harness-directory or nested-worktree member may ever appear in an export manifest or an
     export tree.

A note on scope, because the distinction matters: this suite governs the *public-facing* tree. It
does not police `DECISIONS.md`, `RUN_STATE.json`, `COST_LEDGER.csv` or `artifacts/`, which are
private operational records excluded from the sanitized export by `scripts/make_code_export.py`.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# --- 1. memory wording ------------------------------------------------------------------------
# Public-facing files whose memory claims are load-bearing for a reader sizing a deployment.
MEMORY_FILES = ("README.md", "scripts/release_infer.py", "configs/release/README.md")

# Overbroad claims. Each would tell a reader that total process peak equals one model, which is
# false: the running probability accumulator and the current per-region probabilities are also held.
BANNED_MEMORY_PHRASES = (
    "peak memory is that of a single model",
    "peak VRAM is a single model",
    "peak VRAM is that of a single model",
    "memory-safe ensemble",
    "(memory-safe)",
)

# The accurate claim, in the two halves that must both be present.
REQUIRED_MEMORY_HALVES = (
    "does not scale with",       # ... simultaneous model residency does not scale with five folds
    "not that of a bare",        # ... total process peak memory is not that of a bare single-model run
)

# --- 2. historical banners --------------------------------------------------------------------
BANNER_FILES = (
    "preflight/README.md",
    "preflight/run_static_preflight.py",
    "docs/BraTS_2026_GoAT_Model_Architecture.md",
    "docs/BraTS_2026_GAT26_Preflight_Audit.md",
    "configs/release/AWS_A10G_RUNBOOK.md",
)
BANNER_MARK = "HISTORICAL SNAPSHOT"
# Every banner must restate the current truth, so the label can never drift from the facts.
BANNER_REQUIRED_SUBSTRINGS = (
    "tau=1",                      # normalised below so the Unicode tau also matches
    "diagnostic",
    "opaque",
    "five-digit rule",
    "[128,160,112]",
    "no A10G measurement",
    "authoritative",
)
# A banner must appear near the top of the file, not buried at the end.
BANNER_MAX_LINE = 40

# --- 3. current truth in the authoritative documents ------------------------------------------
CURRENT_TRUTH_FILES = ("README.md", "configs/README.md")
CURRENT_TRUTH_SUBSTRINGS = (
    "dsc and nsd at",
    "tau=1",
    "no five-digit rule",
    "[128,160,112]",
)

# --- 4. historical-string scoping -------------------------------------------------------------
# Strings that describe a SUPERSEDED contract. They are legitimate history, so they are permitted --
# but only in a file that tells the reader it is history.
HISTORICAL_MARKERS = ("[160,160,128]", "five digits", "five-digit")
# Any of these makes a document self-labelling as archival.
ARCHIVAL_LABELS = (
    BANNER_MARK, "historical", "superseded", "archived", "archival",
    "no five-digit rule", "chronology",
)
# Public-facing documentation swept for rule 4. Code and tests carry their own labelled context and
# are checked by the dedicated suites; this rule is about prose a reader could mistake for current.
SCOPED_DOC_GLOBS = ("*.md", "docs/*.md", "configs/*.md", "configs/release/*.md",
                    "preflight/*.md", "public/*.md")

# --- 4b. machine-absolute paths in code that ships ---------------------------------------------
# A tracked file may mention the absolute worker root ONLY in one of two ways:
#   (b) inside a record that carries the HISTORICAL SNAPSHOT banner, or
#   (d) as a RULE LITERAL -- a pattern or assertion whose whole purpose is to spell the path out in
#       order to forbid it.
# Anything else is operational code that would break on a checkout rooted anywhere else, which is
# how scripts/g84_eval.py, scripts/g85_eval.py, scripts/g5_runner.py and tests/test_g83_science.py
# came to carry a private absolute path -- and how check 8e came to vanish silently from every
# public run. Each allowlisted site is named with its file and the reason, so a NEW occurrence
# fails this test instead of blending in.
ABS_ROOT = "/" + "work" + "space"          # assembled so this rule is not its own finding
RULE_LITERAL_SITES = {
    "scripts/g91_public_evidence.py": "FORBIDDEN_SUBSTRINGS: the exporter's own reject list",
    "tests/test_paper_scaffold.py": "no_private_paths: asserts the path is absent from paper/",
    "tests/test_release_infer.py": "weights_dir_not_worker_path: asserts the runner default is not a worker path",
    "tests/test_r11_supplement_regeneration.py": "sanitization scan over the published aggregate",
    "tests/test_r11_public_docs.py": "this rule",
}

# --- 5/6. export hygiene ----------------------------------------------------------------------
# The hidden harness directory that holds registered agent worktrees. Assembled from fragments,
# exactly as `scripts/make_code_export.py` does for its own resource identifiers, so that naming
# the thing this test forbids does not itself become an export finding.
HARNESS_DIR = "." + "cla" + "ude"
FORBIDDEN_EXPORT_SUBSTRINGS = (HARNESS_DIR + "/", HARNESS_DIR + "\\", "/worktrees/")


def _in_validated_export(must_be_absent: str) -> bool:
    """True ONLY inside a genuine sanitized export. Fail-closed by construction.

    Mirrors `tests/test_paper_scaffold.py`: a well-formed root EXPORT_MANIFEST.json must declare
    the Apache-2.0 export, must actually describe THIS tree (it lists this test file and LICENSE),
    and the named path must be genuinely absent. The development repository never contains an
    EXPORT_MANIFEST.json -- the exporter writes it into an export directory -- so deleting files
    inside the private repository can never activate this.
    """
    try:
        man = json.loads((REPO / "EXPORT_MANIFEST.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if man.get("declared_license") != "Apache-2.0":
        return False
    listed = {e.get("path") for e in man.get("files", []) if isinstance(e, dict)}
    if not listed or "LICENSE" not in listed:
        return False
    if Path(__file__).resolve().relative_to(REPO).as_posix() not in listed:
        return False
    return not (REPO / must_be_absent).exists() and must_be_absent not in listed


def _norm(text: str) -> str:
    """Lowercase and fold the two tau spellings, so a test never depends on the encoding used."""
    return (text.replace("τ", "tau").replace("$\\tau$", "tau").replace("\\tau", "tau")
            .replace("`tau=1`", "tau=1").replace("tau = 1", "tau=1").lower())


def test_memory_wording():
    errs = []
    for rel in MEMORY_FILES:
        p = REPO / rel
        if not p.exists():
            errs.append(f"{rel}: missing")
            continue
        text = p.read_text(encoding="utf-8")
        low = text.lower()
        for phrase in BANNED_MEMORY_PHRASES:
            if phrase.lower() in low:
                errs.append(f"{rel}: overbroad memory claim present: {phrase!r}")
        # Strip Markdown emphasis and code ticks so the claim is matched on prose, not on markup.
        flat = " ".join(low.replace("*", "").replace("`", "").split())
        for half in REQUIRED_MEMORY_HALVES:
            if half not in flat:
                errs.append(f"{rel}: corrected memory qualification missing: {half!r}")
    return errs


def test_historical_banners():
    errs = []
    for rel in BANNER_FILES:
        p = REPO / rel
        if not p.exists():
            errs.append(f"{rel}: missing")
            continue
        lines = p.read_text(encoding="utf-8").splitlines()
        idx = next((i for i, l in enumerate(lines) if BANNER_MARK in l), None)
        if idx is None:
            errs.append(f"{rel}: no {BANNER_MARK!r} banner")
            continue
        if idx > BANNER_MAX_LINE:
            errs.append(f"{rel}: banner at line {idx + 1}, must be within {BANNER_MAX_LINE}")
        head = _norm(" ".join(lines[idx:idx + 40]))
        for sub in BANNER_REQUIRED_SUBSTRINGS:
            if _norm(sub) not in head:
                errs.append(f"{rel}: banner does not restate current truth: {sub!r}")
    return errs


def test_current_truth_documents():
    errs = []
    for rel in CURRENT_TRUTH_FILES:
        p = REPO / rel
        if not p.exists():
            errs.append(f"{rel}: missing")
            continue
        flat = " ".join(_norm(p.read_text(encoding="utf-8")).split())
        for sub in CURRENT_TRUTH_SUBSTRINGS:
            if _norm(sub) not in flat:
                errs.append(f"{rel}: current-truth statement missing: {sub!r}")
    return errs


def test_historical_strings_are_scoped():
    """Superseded-contract strings are allowed, but only in a self-labelled archival document."""
    errs = []
    seen = set()
    for pat in SCOPED_DOC_GLOBS:
        for p in sorted(REPO.glob(pat)):
            if not p.is_file() or p in seen:
                continue
            seen.add(p)
            rel = p.relative_to(REPO).as_posix()
            text = p.read_text(encoding="utf-8")
            low = text.lower()
            hits = [m for m in HISTORICAL_MARKERS if m.lower() in low]
            if not hits:
                continue
            if not any(lbl.lower() in low for lbl in ARCHIVAL_LABELS):
                errs.append(f"{rel}: superseded-contract string(s) {hits} with no archival label")
    return errs


def test_absolute_paths_only_in_history_or_rules():
    """No shipping operational file may carry the absolute worker root.

    Scope is deliberately the PUBLIC surface: every tracked file the exporter selects. Frozen JSON
    evidence and the private operational logs are out of scope -- they are never exported, and
    rewriting a frozen record to satisfy a lint would destroy the evidence it exists to preserve.
    """
    errs = []
    # Enumerate the public surface from disk, then narrow with the exporter's own selected() when
    # it is importable. Disk enumeration keeps this check alive everywhere it must run: inside a
    # sanitized export (no exporter) and in a git-less working copy (no `git ls-files`).
    PUBLIC_DIRS = ("scripts", "tests", "configs", "preflight", "docs", "evidence", "paper", "public")
    exported = [q.relative_to(REPO).as_posix()
                for d in PUBLIC_DIRS for q in (REPO / d).rglob("*")
                if q.is_file() and "__pycache__" not in q.parts]
    exported += [q.name for q in REPO.glob("*.md") if q.is_file()]
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_mce", REPO / "scripts" / "make_code_export.py")
        mce = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mce)
        exported = [f for f in exported if mce.selected(f)]
    except Exception:
        pass  # keep the wider disk-derived set; a superset can only make this check stricter

    for rel in sorted(exported):
        path = REPO / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if ABS_ROOT not in text:
            continue
        if rel in RULE_LITERAL_SITES:
            continue
        head = "\n".join(text.splitlines()[:BANNER_MAX_LINE])
        if BANNER_MARK in head:
            continue
        lines = [i for i, l in enumerate(text.splitlines(), 1) if ABS_ROOT in l]
        errs.append(f"{rel}: machine-absolute path on line(s) {lines} — neither a banner-marked "
                    f"historical record nor an allowlisted rule literal")
    return errs


def test_exporter_enumerates_tracked_files_only():
    """The exporter must never walk the filesystem from the repo root.

    A registered harness worktree lives under the hidden harness directory's `worktrees/`
    subdirectory and contains a second, older `paper/main.tex`. `git ls-files` / `git archive`
    cannot see it; `os.walk`, `shutil.copytree` and `zip -r` can. This test pins the safe
    enumeration and pins the exclusion in both directions.
    """
    errs = []
    p = REPO / "scripts" / "make_code_export.py"
    # The exporter excludes itself from the sanitized export (it is in EXCLUDE_EXACT), so in a
    # validated export there is no source to inspect. Fail closed: only a genuine export may skip
    # that half of this check, and the tracked-path half below always runs.
    if p.exists():
        src = p.read_text(encoding="utf-8")
    elif _in_validated_export("scripts/make_code_export.py"):
        src = ""
    else:
        return [f"{p}: missing"]
    if src:
        if "ls-files" not in src:
            errs.append("make_code_export.py: does not enumerate via `git ls-files`")
        for unsafe in ("os.walk(", "shutil.copytree(", "rglob("):
            if unsafe in src:
                errs.append(f"make_code_export.py: unsafe recursive enumeration {unsafe!r}")
    # The repository must not track anything under the harness directory, in any worktree.
    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                                 text=True, check=False).stdout.splitlines()
    except OSError:
        tracked = []
    for t in tracked:
        if t.startswith(HARNESS_DIR + "/") or "/worktrees/" in t:
            errs.append(f"tracked worktree/harness path would enter the export: {t}")
    return errs


def test_no_harness_members_in_export_manifest():
    """No export manifest may declare a harness-directory or nested-worktree member."""
    errs = []
    for man in sorted(REPO.glob("**/EXPORT_MANIFEST.json")):
        try:
            data = json.loads(man.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errs.append(f"{man}: unreadable manifest ({exc})")
            continue
        for entry in data.get("files", []):
            path = entry.get("path", "") if isinstance(entry, dict) else str(entry)
            for bad in FORBIDDEN_EXPORT_SUBSTRINGS:
                if bad in path:
                    errs.append(f"{man}: forbidden export member {path!r}")
    # Present on disk is fine -- the harness owns that directory. It must simply never be
    # tracked, which test_exporter_enumerates_tracked_files_only already enforces.
    return errs


def main() -> int:
    all_errors = []
    for name, fn in (
        ("memory wording", test_memory_wording),
        ("historical-snapshot banners", test_historical_banners),
        ("current-truth documents", test_current_truth_documents),
        ("historical-string scoping", test_historical_strings_are_scoped),
        ("absolute paths only in history or rule literals",
         test_absolute_paths_only_in_history_or_rules),
        ("exporter tracked-file enumeration / worktree exclusion",
         test_exporter_enumerates_tracked_files_only),
        ("no harness-directory export members", test_no_harness_members_in_export_manifest),
    ):
        errs = fn()
        print(f"[{'PASS' if not errs else 'FAIL'}] {name}")
        for e in errs:
            print(f"    - {e}")
        all_errors.extend(errs)
    print(f"\nR11 PUBLIC-DOC TESTS: {'PASS' if not all_errors else 'FAIL'} "
          f"({len(all_errors)} issue(s))")
    return 0 if not all_errors else 1


if __name__ == "__main__":
    sys.exit(main())
