# GAT-26 paper — source & citation ledger

Primary sources only. BraTS citations are the ones **mandated by the live Challenge Rules**
(Synapse `wiki/639585`, retrieved 2026-07-23T07:23:22Z); arXiv IDs / DOIs are quoted verbatim from
that page. Method citations are the standard primary sources. **No invented references.**

## Reference provenance

| BibTeX key | Source of the identifier | Status |
|---|---|---|
| `baid2021brats` | Challenge Rules 639585 (flagship, arXiv:2107.02314) | id verified; author list verified vs arXiv 2026-07-28 |
| `menze2015brats` | Challenge Rules 639585 (DOI 10.1109/TMI.2014.2377694) | verified id |
| `bakas2017advancing` | Challenge Rules 639585 (DOI 10.1038/sdata.2017.117) | verified id |
| `bakas2017gbm` | Challenge Rules 639585 (DOI 10.7937/K9/TCIA.2017.KLXWJJ1Q) | verified id (optional data citation) |
| `bakas2017lgg` | Challenge Rules 639585 (DOI 10.7937/K9/TCIA.2017.GJQ7R0EF) | verified id (optional data citation) |
| `bratsmen2023` | Challenge Rules 639585 (arXiv:2305.07642) | id verified; author list + title verified vs arXiv 2026-07-28 |
| `bratsmet2023` | Challenge Rules 639585 (arXiv:2306.00838) | id verified; author list + title verified vs arXiv 2026-07-28 |
| `bratsped2023` | Challenge Rules 639585 (arXiv:2305.17033) | id verified; author list + title verified vs arXiv 2026-07-28 |
| `bratsssa2023` | Challenge Rules 639585 (arXiv:2305.19369) | id verified; author list + title verified vs arXiv 2026-07-28 |
| `karargyris2023medperf` | Challenge Rules 639585 (DOI 10.1038/s42256-023-00652-2) | verified id |
| `isensee2021nnunet` | Method backbone — nnU-Net (Nature Methods, DOI 10.1038/s41592-020-01008-z) | standard primary source |
| `isensee2024nnunetrevisited` | ResEnc presets — nnU-Net Revisited (arXiv:2404.09556) | standard primary source |
| `wortsman2022modelsoups` | Weight-averaging audit — Model Soups (ICML 2022, PMLR vol. 162, pp. 23965–23998) | verified vs the PMLR proceedings record 2026-07-30 |
| `damour2022underspecification` | Underspecification motivation (JMLR 23(226), 1–61, arXiv:2011.03395) | verified vs the JMLR record 2026-07-30 |
| `reinke2024metricsreloaded` | Metric-choice rationale — Metrics Reloaded (Nature Methods 21, 195–212, DOI 10.1038/s41592-023-02151-z) | verified vs Crossref 2026-07-30; first-authored by Maier-Hein, Reinke second |
| `kofler2023panoptica` | Lesion-wise evaluation — Panoptica (arXiv:2312.02608) | verified vs the arXiv record 2026-07-30 |

**Bibliography completeness.** Every cited work below has a verified primary record.
Every entry's exact title and leading author list has now been verified against the **primary
source** — the arXiv landing page for each arXiv-identified manuscript, and the publisher DOI record
for the journal entries — and the BibTeX was rewritten accordingly. The organizational-author
placeholders (`{BraTS Challenge Organizers}`) have been **replaced with verified author lists** and
no longer appear anywhere in `references.bib`.

Verified 2026-07-28 against the arXiv primary source:

| Key | Verified leading author | Verified title (as recorded) |
|---|---|---|
| `baid2021brats` | Baid, U.; Ghodasara, S.; Mohan, S.; … | The RSNA-ASNR-MICCAI BraTS 2021 Benchmark … |
| `bratsmen2023` | LaBella, D.; Adewole, M.; Alonso-Basanta, M.; … | The ASNR-MICCAI BraTS Challenge 2023: Intracranial Meningioma |
| `bratsmet2023` | Moawad, A.W.; Janas, A.; Baid, U.; … | The BraTS-METS Challenge 2023: Brain Metastasis Segmentation on Pre-treatment MRI |
| `bratsped2023` | Fathi Kazerooni, A.; Khalili, N.; Liu, X.; … | The BraTS Challenge 2023: Focus on Pediatrics (CBTN-CONNECT-DIPGR-ASNR-MICCAI BraTS-PEDs) |
| `bratsssa2023` | Adewole, M.; Rudie, J.D.; Gbadamosi, A.; … | The BraTS Challenge 2023: Glioma Segmentation in Sub-Saharan Africa Patient Population (BraTS-Africa) |
| `isensee2024nnunetrevisited` | Isensee, F.; Wald, T.; Ulrich, C.; … | nnU-Net Revisited (accepted at MICCAI 2024) |

**Deliberate, verified truncation:** the BraTS consortium manuscripts carry very large author lists.
Leading authors are recorded verbatim from the primary source and truncated with `and others`, which
`splncs04.bst` typesets as "et al." This is a recorded editorial choice, **not** missing metadata.

The mandated acknowledgement sentence is present in `main.tex` ("Data used in this publication
were obtained as part of the Challenge project through Synapse ID (syn74274097).").

## Numeric-claim verification (paper value → committed evidence)

| Paper claim | Value | Committed evidence source |
|---|---|---|
| Labeled cases | 1,351 | **private-only:** `RELEASE_CHECKLIST.md` (G3 audit), `RUN_STATE.json` g45_pretraining. **Public:** stated in `paper/main.tex` §2.1 and in `public/DATA_PROVENANCE.md` |
| Split seed | 21072026 | `scripts/g45_selection_policy.py` SEED; `RELEASE_CHECKLIST.md` |
| Fold sizes | [271,270,270,270,270] | **private-only:** `RELEASE_CHECKLIST.md` (G4.5 split). **Public:** `scripts/g45_group_split.py` reproduces them from the recorded seed; also stated in `paper/main.tex` §2.2 |
| Fold-0 validation cases | 271 | **private-only:** `artifacts/g5_{m,l}_fold0_official_eval_summary.json` (n). **Public:** the fold sizes above |
| Labels / regions | {0,1,2,3}; ET=[3],TC=[1,3],WT=[1,2,3] | `docs/RULE_SNAPSHOT.md`; wiki/639579 |
| M fold-0 ET/TC/WT DSC | 0.859 / 0.914 / 0.934 | **private-only:** `artifacts/g5_m_fold0_official_eval_summary.json`. **Public:** tabulated in `paper/supplement.tex` (architecture diagnostic) |
| M fold-0 ET/TC/WT HD95 (mm) | 14.17 / 5.96 / 3.89 | same |
| M dsc_p05 / hd95_p95 | 0.649 / 15.96 | same |
| L fold-0 ET/TC/WT DSC | 0.861 / 0.912 / 0.932 | **private-only:** `artifacts/g5_l_fold0_official_eval_summary.json`. **Public:** tabulated in `paper/supplement.tex` (architecture diagnostic) |
| L fold-0 ET/TC/WT HD95 (mm) | 11.19 / 6.08 / 4.11 | same |
| L dsc_p05 / hd95_p95 | 0.637 / 18.06 | same |
| Selection | select_M; M wins 4/6; rank_gain −0.333 | **private-only:** `artifacts/g5_fold0_selection_decision.json`. **Public:** the rule is executable in `scripts/g45_selection_policy.py` and stated in `paper/main.tex` §2.3 |
| Bootstrap | seed 21072026, 10,000 resamples, CI [−0.667,+0.667], not favoring L | same |
| Tail noninferiority | dsc_p05 fail; hd95_p95 fail | same |
| Evaluator | BraTS-evaluation 0.0.8, GoAT config | `docs/RULE_SNAPSHOT.md`; `configs/evaluator_environment.lock.txt` |
| HD95 zero-TP penalty | 373 mm | `scripts/g45_selection_policy.py` HD95_PENALTY; wiki (evaluator) |
| Container limits | A10G 24 GB, 16 vCPU, 48 GiB mem, 16 GiB shm, CUDA ≤13.0, 12 h | wiki/639582 (2026-07-12); `docs/RULE_SNAPSHOT.md` |
| Inference | threshold 0.5, no TTA/CC/presence-gate, hierarchy-safe WT⊇TC⊇ET | `scripts/release_infer.py` |

The table above covers the fold-0 architecture screen, the split and the container/inference
contract. It is **not** a complete map of every number in `main.tex`: the Audit C development and
follow-up families (per-component means and deltas at both tolerances), the pooled supportive
analysis, the lesion noninferiority analysis, the official-validation values, the runtime and
repeatability figures, and the binary64 rank-gain threshold are evidenced in the **supplement** and
in the published aggregate records under `evidence/`, not in this table. No placeholder remains in
the manuscript.

**Reading the source column.** Every row that rests on a record which is deliberately **not**
redistributed is marked **private-only** inline, and names a public equivalent wherever one exists —
an executable script, a committed config, a published aggregate under `evidence/`, or the manuscript
itself. Rows with no such marking cite files that ship in this repository and can be opened directly.
The private records are named rather than hidden, so a reader can see exactly what backs each value;
their absence from the export is intentional, not an omission. This replaces the single blanket
disclaimer that previously stood here, which did not say which rows it applied to.
