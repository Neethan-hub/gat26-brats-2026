# GAT-26 paper (BraTS 2026 Task 3 / BraTS-GoAT) — scaffold

Publication-quality **Springer LNCS** short-paper scaffold, populated only with **verified committed
evidence**. Everything not yet measured/decided is a conspicuous fail-closed placeholder
(`\ownerinput{...}` or `\pending{...}`) — never fabricated.

## Files
- `main.tex` — the paper (8–10 pp excl. references; all required LNCS sections).
- `references.bib` — citation ledger (organizer-mandated primary sources + nnU-Net).
- `CITATION_LEDGER.md` — reference provenance + numeric-claim → committed-evidence map.
- `SUBMISSION_CHECKLIST.md` — owner-facing submission administration.

## Compile — **RUN AND PASSED** (Stage G78, 2026-07-28)

The paper has now been **compiled against the official Springer LNCS class** in an isolated,
git-ignored TeX environment under `/tmp` (nothing was installed globally on the controller).

| Item | Result |
|---|---|
| Class files | official Springer `llncs.cls` **v2.26 (25-Feb-2025)** + `splncs04.bst`, from the CTAN `llncs` distribution |
| Engine | isolated Tectonic 0.15.0 under `/tmp` (full LaTeX + BibTeX passes to convergence) |
| Undefined citations / references | **0** |
| Bibliography entries typeset | **12 / 12** |
| Total typeset length | **10 pages** |
| **Length excluding references** | **9 pages** — References heading begins on page 9 · **within the required 8–10** |
| Overfull boxes | 2 minor (7.7 pt in body; 13.1 pt caused by the long `\ownerinput{acknowledgements}` placeholder, which resolves when the owner supplies the text) |
| Visual review | every page rendered and inspected — no clipping, no broken references, no malformed tables, no missing citations, no unfinished prose |

`llncs.cls`, `splncs04.bst`, `main.pdf`, and all LaTeX build artifacts remain **git-ignored** and are
**not committed** (Springer copyright / build output).

To reproduce, place `llncs.cls` and `splncs04.bst` in this directory and run:

```
cd paper
pdflatex main
bibtex   main
pdflatex main
pdflatex main   # -> main.pdf
```

**Compiling cleanly is not the same as being submission-ready.** Every `\ownerinput{}` / `\pending{}`
placeholder below is still open, so the manuscript must not be submitted as it stands.

## Verified content (populated from committed artifacts)
Dataset/split statistics (1,351 cases; seed 21072026; folds [271,270,270,270,270]; labels {0,1,2,3};
ET/TC/WT); the **fold-0 M-vs-L architecture-selection screen**; the **complete five-fold
cross-validated OOF result** (1,351 cases each predicted once by a model that did not train on it;
official evaluator n=1,351 / 0 errors; ET 0.865/13.39, TC 0.915/6.75, WT 0.927/6.34, dsc_p05 0.666,
hd95_p95 17.72); both **frozen inference-policy audits** (G7.5 and G7.6) and their negative results;
random initialization / no external or pretrained weights; evaluator (BraTS-evaluation 0.0.8, GoAT);
inference ensemble + hierarchy-safe reconstruction + enforced-determinism settings (measured as
**near-deterministic, not bit-exact**: 18/2.14e8 voxels differ across two 24-case runs); container contract.
See `CITATION_LEDGER.md` for the value→evidence map.

**Official-metric note:** the paper states the organizers' Task-3 ranking metrics correctly as
**DSC + NSD** (`wiki/639592`, mod 2026-07-24) and is explicit that GAT-26's pre-registered *internal*
selection utility used DSC + HD95 as a proxy. See `docs/RULE_SNAPSHOT.md` §2026-07-25.

## Pending (conspicuous placeholders — do NOT fill by extrapolation)
`\pending{}`: genuine-A10G peak reserved VRAM and seconds/case (A10G-2); final image tag and digest;
official validation score and rank (GAT-26 is currently **unranked — no official validation
submission**); hidden-test results.
`\ownerinput{}`: final title; author list, affiliations, emails, OpenReview profiles; acknowledgements;
public source-code URL; team name; OpenReview and Synapse submission IDs; confirmation of the final
organizer metric definition at camera-ready.

## Live rule basis (retrieved 2026-07-23T07:23:22Z; re-verified 2026-07-25T12:52Z and **2026-07-28T01:15–01:17Z**)
8–10 pp excl. refs, Springer LNCS (wiki/639582, unchanged, mod 2026-07-12); OpenReview submission +
mandatory short paper linked to the Docker submission via the Synapse team name; copyright form at
camera-ready; final deadline 30 Jul 2026 23:59 UTC (wiki/639587, unchanged, mod 2026-07-22).
**No rule conflicts observed; no page-limit or template change.**

**Recorded 2026-07-28 — source-code link is MANDATORY IN THE PAPER.** The Timeline wiki (639587)
states verbatim: *"Short paper must report a) source code (GitHub link), b) method description, and
c) results on training and validation data."* This makes the `\ownerinput{public source-code URL}`
placeholder a **hard submission blocker**, not an optional field. Note the separate Challenge-Rules
provision (639585): *"Organizers will be irrevocably permitted to make the submitted container
publicly accessible on an Apache v.2.0 license, unless another license is otherwise indicated"* —
so declining to indicate a licence is itself a licence decision. The licence choice is the owner's.

**Anonymity: no rule was found, which is NOT confirmation — the OpenReview venue anonymity setting
is an owner-verification blocker.**

## Submission blockers

**Cleared by Stage G78 (2026-07-28):**
- Compilation with the official Springer LNCS class to a PDF — **done**.
- Rendered page-by-page visual review — **done**.
- Verified 8–10-page length excluding references — **done (9 pages excluding references)**.
- BibTeX metadata: every entry's title and leading author list re-verified against its **primary
  source** (arXiv landing pages / publisher DOIs, 2026-07-28); the organizational-author placeholders
  `{BraTS Challenge Organizers}` are **gone**. Large consortium author lists are truncated with
  `and others` (rendered "et al." by `splncs04`), which is a deliberate, verified truncation.

**Still open (hard blockers):**
- Every `\ownerinput{}` placeholder — title, authors/affiliations/emails, acknowledgements, the
  public source-code URL, and the camera-ready NSD-tolerance confirmation.
- Every `\pending{}` placeholder — genuine-A10G VRAM/runtime (A10G-2), final image tag/digest,
  official validation score/rank, hidden-test results.
- OpenReview venue anonymity setting (owner verification; absence of a rule is not confirmation).

The final container image and genuine-A10G validation remain pending; the paper describes a release
**scaffold and design target**, not a completed container.
