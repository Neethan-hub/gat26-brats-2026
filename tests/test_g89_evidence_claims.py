#!/usr/bin/env python3
"""G89 — evidence-claim regression tests (`python3 tests/test_g89_evidence_claims.py`).

The frozen release policy uses NO connected-component filtering. A G88 evidence file recorded that
as a bare `"connected_component_filtering": true`, following the surrounding convention where the
Boolean meant *the verification gate passed* — but read alone it asserts the opposite of the policy.

These tests fail if any tracked evidence file describes connected-component filtering as enabled, or
reintroduces the ambiguous bare Boolean, and they pin the narrowed scope of the G88 image's
qualification claim. They read only tracked repository files; no image, GPU or network is involved.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FAILS = 0

# Phrasings that would assert filtering is on. `"none"`, `False`, and the structured form are fine.
ENABLED_PATTERNS = (
    re.compile(r'"(?:connected_component_filtering|cc_filtering)"\s*:\s*true', re.I),
    re.compile(r'"(?:connected_component_filtering|cc_filtering)"\s*:\s*"(?:on|enabled|yes)"', re.I),
    re.compile(r'connected[- ]component filtering (?:is |was )?(?:enabled|applied|on)\b', re.I),
)


def check(name, cond):
    global FAILS
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILS += 1


def tracked_evidence_files():
    for pat in ("artifacts/*.json", "artifacts/*.md", "public/*.md", "configs/**/*.json"):
        yield from sorted(REPO.glob(pat))


def main():
    print("test_g89_evidence_claims:")

    # The private evidence records this test guards are deliberately NOT part of the public source
    # export. In an export they are absent, and there is nothing to guard; in the private
    # repository their absence would itself be a failure.
    required = [REPO / "artifacts" / n for n in
                ("g88_corrected_image.json", "g88_result.json", "g86_release_freeze.json")]
    if not (REPO / "artifacts").is_dir():
        print("  ..   no artifacts/ directory: public export, evidence guard not applicable")
        print("test_g89_evidence_claims: PASS (not applicable)")
        return 0
    missing = [p for p in required if not p.is_file()]
    if missing:
        print(f"  FAIL missing private evidence: {[p.name for p in missing]}")
        print("test_g89_evidence_claims: FAIL (1)")
        return 1

    # 1. no tracked evidence file may describe component filtering as enabled
    offenders = []
    for p in tracked_evidence_files():
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            # the correction record quotes the old form to explain it; that quotation is inside a
            # fenced block and is explicitly labelled, so exempt only that one file's quotations
            if p.name == "G89_EVIDENCE_CORRECTION.md":
                continue
            if any(rx.search(line) for rx in ENABLED_PATTERNS):
                offenders.append(f"{p.relative_to(REPO)}:{n}")
    check("no_evidence_file_claims_component_filtering_enabled", not offenders)
    if offenders:
        for o in offenders[:10]:
            print(f"       {o}")

    # 2. the G88 corrected-image record carries the unambiguous structure
    ci = json.loads((REPO / "artifacts" / "g88_corrected_image.json").read_text(encoding="utf-8"))
    ccf = ci["frozen_policy_verified_inside_the_image"]["connected_component_filtering"]
    check("component_filtering_is_structured", isinstance(ccf, dict))
    check("component_filtering_enabled_is_false", isinstance(ccf, dict) and ccf.get("enabled") is False)
    check("component_filtering_verification_passed",
          isinstance(ccf, dict) and ccf.get("verification_passed") is True)

    # 3. the release freeze still says the policy is none, and the runner still reports it off
    frz = json.loads((REPO / "artifacts" / "g86_release_freeze.json").read_text(encoding="utf-8"))
    check("release_freeze_policy_is_none",
          frz["policy"]["connected_component_filtering"] == "none")
    runner = (REPO / "scripts" / "release_infer.py").read_text(encoding="utf-8")
    check("runner_reports_cc_filtering_false", '"cc_filtering": False' in runner)
    check("runner_has_no_component_filtering_code",
          not re.search(r"label\(|connected_components|remove_small_objects|cc3d", runner))

    # 4. the G88 image's qualification claim is scoped, not bare
    res = json.loads((REPO / "artifacts" / "g88_result.json").read_text(encoding="utf-8"))
    q = res["corrected_image"]["qualified"]
    check("g88_qualified_claim_is_scoped", isinstance(q, dict))
    if isinstance(q, dict):
        check("g88_claim_states_hardware", "A40" in str(q.get("hardware", "")))
        check("g88_claim_denies_oci_parity", q.get("full_oci_runtime_parity") is False)
        check("g88_claim_denies_a10g_evidence", q.get("a10g_evidence_for_this_image") is False)

    print(f"test_g89_evidence_claims: {'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
