#!/usr/bin/env python3
"""GAT-26 deterministic release build-context preflight (NO Docker, NO build).

Before `docker build`, verify that a private build context is COMPLETE and CLEAN:
  * every `COPY <src>` source in configs/release/Dockerfile exists in the context;
  * the required plans/dataset.json are present;
  * the checkpoint layout matches the mode — final (A10G-2): weights/fold_0..4/checkpoint_final.pth;
    proxy (A10G-1): weights/fold_0/checkpoint_final.pth;
  * NO forbidden material is anywhere in the context — Synapse config, shell history, SSH material,
    credentials/tokens, validation archives, raw images, predictions, `.git`, coverage/venv/caches.
Fail closed (nonzero) on any problem. The check logic is a pure function for unit testing.
"""
from __future__ import annotations

import argparse
import re
import sys
from fnmatch import fnmatch
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO / "configs" / "release" / "Dockerfile"

REQUIRED_FILES = ("plans/nnUNetResEncUNetMPlans.json", "plans/dataset.json")
FORBIDDEN_NAMES = {".git", ".synapseConfig", ".netrc", ".bash_history", ".zsh_history",
                   ".python_history", ".ssh", ".aws", ".gitconfig"}
FORBIDDEN_DIR_PARTS = {"nnUNet_raw", "nnUNet_preprocessed", "predictions", "oof", ".git",
                       "__pycache__", ".venv", ".venv-eval"}
FORBIDDEN_GLOBS = ("*.pat", "*.token", "*token*", "id_rsa*", "id_ed25519*", "*.pem", "*.p12",
                   "*.pfx", "credentials*", "*.synapseconfig", "*.key")
FORBIDDEN_FILE_EXT = (".zip", ".tar", ".tar.gz", ".tgz", ".nii", ".nii.gz", ".npy", ".npz")


def dockerfile_copy_sources(dockerfile_text):
    """Return the source args of every `COPY <src...> <dest>` in the Dockerfile (dest dropped)."""
    joined = dockerfile_text.replace("\\\n", " ")
    srcs = []
    for line in joined.splitlines():
        m = re.match(r"\s*COPY\s+(.+)$", line)
        if not m:
            continue
        parts = [t for t in m.group(1).split() if not t.startswith("--")]
        if len(parts) >= 2:
            srcs += parts[:-1]        # everything except the destination
    return srcs


def check_context(context_dir, dockerfile_text, proxy=False):
    """Return a sorted list of problems (empty == clean & complete). Pure — no side effects."""
    ctx = Path(context_dir)
    errors = []

    # 1. every Dockerfile COPY source must exist in the context
    for s in dockerfile_copy_sources(dockerfile_text):
        if not (ctx / s.rstrip("/")).exists():
            errors.append(f"missing COPY source: {s}")

    # 2. required plans/dataset
    for f in REQUIRED_FILES:
        if not (ctx / f).is_file():
            errors.append(f"missing required file: {f}")

    # 3. checkpoint layout for the declared mode
    if proxy:
        if not (ctx / "weights" / "fold_0" / "checkpoint_final.pth").is_file():
            errors.append("proxy layout requires weights/fold_0/checkpoint_final.pth")
    else:
        for i in range(5):
            if not (ctx / "weights" / f"fold_{i}" / "checkpoint_final.pth").is_file():
                errors.append(f"final layout requires weights/fold_{i}/checkpoint_final.pth")

    # 4. no forbidden material anywhere in the context
    if ctx.exists():
        for p in ctx.rglob("*"):
            rel = p.relative_to(ctx)
            low = p.name.lower()
            if p.name in FORBIDDEN_NAMES:
                errors.append(f"forbidden entry in context: {rel}")
            if any(part in FORBIDDEN_DIR_PARTS for part in rel.parts):
                errors.append(f"forbidden path in context: {rel}")
            if any(fnmatch(low, g) for g in FORBIDDEN_GLOBS):
                errors.append(f"forbidden credential-like file: {rel}")
            if p.is_file() and low.endswith(FORBIDDEN_FILE_EXT):
                # checkpoints (*.pth) are allowed weights; images/archives/arrays are not
                errors.append(f"forbidden data/archive in context: {rel}")
    return sorted(set(errors))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True, help="path to the private docker build context")
    ap.add_argument("--proxy", action="store_true", help="A10G-1 proxy layout (fold_0 only)")
    args = ap.parse_args()
    problems = check_context(args.context, DOCKERFILE.read_text(encoding="utf-8"), proxy=args.proxy)
    mode = "A10G-1_proxy" if args.proxy else "A10G-2_final"
    print(f"build_context_preflight mode={mode} problems={len(problems)}")
    for p in problems:
        print(f"  - {p}")
    return 0 if not problems else 9


if __name__ == "__main__":
    sys.exit(main())
