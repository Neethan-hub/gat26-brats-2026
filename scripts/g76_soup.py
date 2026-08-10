#!/usr/bin/env python3
"""GAT-26 G7.6 checkpoint-soup construction (fail-closed, testable).

A "soup" is a per-fold weighted average of that fold's own `checkpoint_final` and `checkpoint_best`
network weights. Only the two ratios predeclared in the G7.6 stage prompt are permitted:

    S1 = 0.75 * final + 0.25 * best
    S2 = 0.50 * final + 0.50 * best

Hard guarantees (every violation raises SoupFailure -- never a silent fallback):
  * identical state-dict key sets after `_orig_mod.` normalization;
  * identical shapes for every key;
  * ONLY floating-point tensors are averaged (in float64, cast back to the source dtype);
  * every non-floating tensor/buffer (e.g. `num_batches_tracked`, integer/bool buffers) must be
    BITWISE IDENTICAL between the two checkpoints, otherwise the soup fails closed;
  * optimizer / lr-scheduler / grad-scaler / logging / epoch / training state is never averaged and
    never carried into the soup -- the soup checkpoint holds network weights plus provenance only;
  * the source checkpoints are opened read-only and are never modified.

The resulting state dict is intended to be loaded with `strict=True`; the non-strict loading mode is
never used anywhere in the soup or release paths (enforced by tests/test_g76_soup.py).
This module is pure and importable so it can be unit-tested without a GPU, checkpoints, or data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

RATIOS = {"S1": (0.75, 0.25), "S2": (0.50, 0.50)}
ORIG_MOD_PREFIX = "_orig_mod."


class SoupFailure(Exception):
    """Any structural or numerical precondition violation. Always fail closed."""


def normalize_keys(state_dict):
    """Strip the torch.compile `_orig_mod.` prefix. Collisions are a hard failure."""
    out = {}
    for k, v in state_dict.items():
        nk = k[len(ORIG_MOD_PREFIX):] if k.startswith(ORIG_MOD_PREFIX) else k
        if nk in out:
            raise SoupFailure(f"key collision after _orig_mod. normalization: {nk}")
        out[nk] = v
    return out


def _is_float(t):
    # torch tensors expose is_floating_point(); the tests use a tiny stub with the same API.
    return bool(t.is_floating_point())


def build_soup(sd_final, sd_best, ratio_name):
    """Return the souped state dict for `ratio_name`. Inputs are NOT mutated."""
    if ratio_name not in RATIOS:
        raise SoupFailure(f"unknown ratio {ratio_name!r}; permitted: {sorted(RATIOS)}")
    w_final, w_best = RATIOS[ratio_name]

    a = normalize_keys(sd_final)
    b = normalize_keys(sd_best)
    if set(a) != set(b):
        missing = sorted(set(a) ^ set(b))[:5]
        raise SoupFailure(f"state-dict key sets differ (e.g. {missing})")

    out = {}
    for k in a:
        ta, tb = a[k], b[k]
        if tuple(ta.shape) != tuple(tb.shape):
            raise SoupFailure(f"shape mismatch for {k}: {tuple(ta.shape)} vs {tuple(tb.shape)}")
        if _is_float(ta) != _is_float(tb):
            raise SoupFailure(f"dtype class mismatch for {k}: float vs non-float")
        if _is_float(ta):
            # Average in float64 for reproducibility, then restore the source dtype.
            out[k] = (ta.double() * w_final + tb.double() * w_best).to(ta.dtype)
        else:
            # Non-float buffers are NEVER averaged: they must already agree exactly.
            if not _tensors_equal(ta, tb):
                raise SoupFailure(f"non-floating tensor {k} differs between checkpoints; refusing to average")
            out[k] = ta.clone()
    return out


def _tensors_equal(x, y):
    """Bitwise equality. Uses torch.equal for real tensors; falls back to `==` otherwise so the
    pure logic stays unit-testable without torch."""
    try:
        import torch
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            return bool(torch.equal(x, y))
    except ImportError:
        pass
    return bool(x == y)


# Training state that must never leak into a soup checkpoint.
FORBIDDEN_CHECKPOINT_KEYS = (
    "optimizer_state", "grad_scaler_state", "lr_scheduler_state", "logging",
    "current_epoch", "_best_ema", "trainer_name_history",
)


def soup_checkpoint(soup_sd, ratio_name, provenance):
    """Wrap souped weights in a minimal checkpoint carrying no training state."""
    ck = {
        "network_weights": soup_sd,
        "gat26_soup": {
            "ratio_name": ratio_name,
            "weights": {"checkpoint_final": RATIOS[ratio_name][0],
                        "checkpoint_best": RATIOS[ratio_name][1]},
            "provenance": provenance,
            "note": "network weights only; no optimizer/scheduler/scaler/logger/epoch state",
        },
    }
    leaked = [k for k in FORBIDDEN_CHECKPOINT_KEYS if k in ck]
    if leaked:
        raise SoupFailure(f"training state leaked into soup checkpoint: {leaked}")
    return ck


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path):
    import torch
    for mod in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            torch.serialization.add_safe_globals([__import__(mod, fromlist=["scalar"]).scalar])
        except Exception:
            pass
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def main():
    ap = argparse.ArgumentParser(description="Build a G7.6 checkpoint soup for one fold.")
    ap.add_argument("--results-dir", required=True, help="nnU-Net ResEnc-M results dir")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--ratio", choices=sorted(RATIOS), required=True)
    ap.add_argument("--out", required=True, help="output soup checkpoint path (private/ignored)")
    ap.add_argument("--provenance-out", default="", help="optional private provenance JSON path")
    a = ap.parse_args()

    import torch
    fp = Path(a.results_dir) / f"fold_{a.fold}" / "checkpoint_final.pth"
    bp = Path(a.results_dir) / f"fold_{a.fold}" / "checkpoint_best.pth"
    for p in (fp, bp):
        if not p.is_file():
            raise SoupFailure(f"missing source checkpoint: {p.name}")

    mt_before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in (fp, bp)}
    ck_f, ck_b = _load(fp), _load(bp)
    for tag, ck in (("final", ck_f), ("best", ck_b)):
        if "network_weights" not in ck:
            raise SoupFailure(f"{tag} checkpoint has no network_weights")

    prov = {
        "fold": a.fold,
        "ratio": a.ratio,
        "source_final_sha256": sha256_file(fp),   # private provenance only; never printed/committed
        "source_best_sha256": sha256_file(bp),
        "source_final_epoch": ck_f.get("current_epoch"),
        "source_best_epoch": ck_b.get("current_epoch"),
    }
    soup = build_soup(ck_f["network_weights"], ck_b["network_weights"], a.ratio)
    os.makedirs(Path(a.out).parent, exist_ok=True)
    torch.save(soup_checkpoint(soup, a.ratio, prov), a.out)

    mt_after = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in (fp, bp)}
    if mt_before != mt_after:
        raise SoupFailure("source checkpoints were modified during soup construction")

    if a.provenance_out:
        os.makedirs(Path(a.provenance_out).parent, exist_ok=True)
        with open(a.provenance_out, "w") as fh:
            json.dump(prov, fh, indent=1)

    # Sanitized stdout only: never print a hash.
    print(json.dumps({"fold": a.fold, "ratio": a.ratio, "n_tensors": len(soup),
                      "sources_unmodified": True, "wrote_soup": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
