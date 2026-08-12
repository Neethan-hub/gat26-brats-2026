# Camera-ready paper source

LaTeX source for the BraTS-GoAT 2026 Task-3 submission and its supplement. Compiled PDFs are not
committed.

| File | What it is |
|---|---|
| `main.tex` | The main paper |
| `supplement.tex` | Supplementary material — **generated**, do not hand-edit |
| `references.bib` | Bibliography, shared by both documents |
| `CITATION_LEDGER.md` | Provenance of every reference |

## Review model

The manuscript is **not anonymized**: the organizers confirmed on the challenge discussion thread
that review for this track is single-blind, so author identity is carried in the front matter.

## Building

`llncs.cls` and `splncs04.bst` are not vendored here; take them from the CTAN `llncs` package
(v2.26, 25-Feb-2025) and place them beside the sources. Both documents then build with:

```
pdflatex -interaction=nonstopmode main
bibtex   main
pdflatex -interaction=nonstopmode main
pdflatex -interaction=nonstopmode main          # -> main.pdf

pdflatex -interaction=nonstopmode supplement
pdflatex -interaction=nonstopmode supplement    # -> supplement.pdf  (no bibliography)
```

The only figure is drawn in TikZ inside `main.tex`, so there are no external image assets.

Repeated builds are byte-identical within one pinned toolchain when `SOURCE_DATE_EPOCH` and
`FORCE_SOURCE_DATE` are exported, which fixes the timestamps pdfTeX embeds. Byte identity is
not claimed across TeX distributions or pdfTeX versions, which legitimately differ in object layout.
Without those variables, two builds from the same toolchain differ only in `/CreationDate`,
`/ModDate` and the file `/ID`. The camera-ready PDFs were produced
with pdfTeX 3.14159265-2.6-1.40.20 (TeX Live 2019/Debian) and BibTeX 0.99d.

## Regenerating the supplement

`supplement.tex` is generated, never hand-edited. Every reported measurement or result in it is
read from a committed record at build time -- none is typed in -- and the generator re-derives and
asserts the utility identity before writing anything. Quantities the supplement presents as
*derived* rather than measured are labelled as such where they appear, and the `provenance` block of
`evidence/supplement_inputs.json` records, per value, the frozen source field and any derivation
formula:

```
python3 scripts/g92_build_supplement.py . paper/supplement.tex
```

That command works in **both** trees, and reproduces the committed `paper/supplement.tex`
byte-for-byte in each:

| Tree | Input the generator reads |
|---|---|
| This public repository | `evidence/supplement_inputs.json` -- a whitelisted aggregate projection of the two frozen audit records, published at full float precision by `scripts/g91_public_evidence.py`. It carries no per-case value, case identifier, fold membership, prediction or path. |
| The private development repository | `artifacts/g84_result.json` and `artifacts/g85_result.json` -- the frozen audit records themselves, which take priority when present. |

With neither input available the generator stops with an explicit diagnostic naming both accepted
paths; it never emits a partial document and never substitutes a default for a missing measurement.
`tests/test_r11_supplement_regeneration.py` exercises exactly this: it rebuilds the supplement from
a tree containing only `evidence/` and `scripts/`, and fails unless the result is byte-identical to
the committed file.

## Third-party files

`llncs.cls` and `splncs04.bst` come from the CTAN `llncs` package, which its README distributes under
**CC BY 4.0**. That covers the package as distributed; the two files carry different upstream
histories and their own notices, which are preserved verbatim and should be read directly:

- `llncs.cls` — Copyright (c) 1996–2025 Springer.
- `splncs04.bst` — derived from `merlin.mbs`, Copyright 1994–2007 Patrick W. Daly, distributed under
  the **LaTeX Project Public License**, and adapted by Maurizio "Titto" Patrignani. It is *not*
  solely Springer-copyrighted, and this repository makes no such claim.

Both are included unmodified in the camera-ready source archive because the submission instructions
require the archive to contain every file needed to compile the paper.
