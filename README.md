# Explainable Multilingual Legal Meaning Preservation Using JUDGEBERT

**CSET 346 — Natural Language Processing — Semester Project**

An NLP system that predicts whether a simplified or reformulated legal sentence
**preserves the legal meaning** of the original statutory sentence, on a 1–10
Legal Meaning Preservation (LMP) scale, in **English and Hindi**, across three
Indian legal domains — with an explanation of *why*.

Inspired by *JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences*
(Grabar et al.). See [Naming and honesty](#naming-and-honesty) for exactly what
"inspired by" means here.

---

## Research question

> How effectively can a JUDGEBERT-based model assess legal meaning preservation
> in a new language and legal domain, and does it outperform generic
> semantic-similarity approaches?

---

## Project status

This project is **partially complete**. There is exactly one rule in force
throughout: nothing is reported that was not actually produced.

| Phase | Deliverable | Status |
| --- | --- | --- |
| 1 | Project skeleton, config, environment | ✅ complete |
| 2 | Corpus (340 pairs), validation, splitting | ✅ complete |
| 2 | Annotation tool, guidelines, agreement | ✅ built — **awaiting human annotators** |
| 2 | Test suite (325 tests) | ✅ complete |
| — | **Human LMP annotation** | ⛔ **not done — blocks every experiment** |
| 3 | Baselines (TF-IDF, Sentence-BERT) | ⬜ not started |
| 4 | JUDGEBERT-inspired model | ⬜ not started |
| 5 | Training + evaluation harness | ⬜ not started |
| 6 | Experiments 1–5 | ⬜ not started |
| 7 | Explainability layer | ⬜ not started |
| 8 | Streamlit demo | ⬜ not started |
| 9 | Paper, presentation, viva bank | ⬜ not started |

**There are no model results in this repository, and there will not be any
until the corpus is annotated.** Every row's `human_lmp_score` is empty, and
`src/data/load_data.py:assert_gold()` raises rather than let an experiment run
on machine-drafted labels. That is by design — see
[The integrity rule](#the-integrity-rule).

---

## Architecture

Solid arrows are implemented and runnable today. Dotted arrows are planned.

```mermaid
flowchart TD
    subgraph inputs["📥 Raw inputs"]
        PDF["8 official statute PDFs<br/>4 English · 4 Hindi"]
        AUTH["7 authored JSONL batches<br/>hand-written pairs"]
    end

    subgraph dataprep["🗂️ Data layer — src/data/"]
        EXTRACT["extract_sources.py<br/><i>English only — Hindi PDFs are corrupt</i>"]
        BUILD["build_dataset.py<br/>merge · validate keys · assign provenance"]
        VALID["validate_data.py<br/>schema · vocabulary · range · leakage"]
        SPLIT["split_data.py<br/>grouped, stratified splits"]
        MASTER[/"lmp_pairs.csv<br/>340 pairs × 23 cols<br/><b>0 human labels</b>"/]
    end

    subgraph annot["✍️ Annotation — src/annotation/"]
        APP["annotation_app.py<br/>blind-mode Streamlit tool"]
        GUIDE["guidelines.md<br/>scale anchors · error taxonomy"]
        AGREE["agreement.py<br/>κ · α · ICC(2,1)"]
        MERGE["merge_annotations.py<br/><i>only writer of gold columns</i>"]
        CUR["curator_review.py<br/><i>curator-only: drafts vs human</i>"]
        GOLD[/"gold corpus<br/>human_lmp_score filled"/]
    end

    subgraph models["🤖 Models"]
        PRE["preprocess.py<br/>legally conservative"]
        CUES["legal_cues.py<br/>modals · negation · conditions<br/>exceptions · quantities"]
        TFIDF["tfidf_baseline.py<br/>+ TRAIN-only calibration"]
        SBERT["sbert_baseline.py<br/>+ TRAIN-only calibration"]
        JB["judgebert_model.py<br/>[CLS] orig [SEP] simp [SEP]<br/>MuRIL cross-encoder"]
    end

    subgraph eval["📊 Evaluation & experiments"]
        METRICS["metrics.py<br/>Pearson · Spearman · RMSE · MAE"]
        EXP["experiments 1–5"]
        EXPL["explainability/<br/>cue alignment (primary)<br/>attention (supporting)"]
        DASH["streamlit_app.py<br/>Legal Meaning Analyzer"]
    end

    PDF --> EXTRACT --> BUILD
    AUTH --> BUILD
    BUILD --> VALID --> SPLIT --> MASTER
    MASTER --> APP
    GUIDE -.-> APP
    APP --> AGREE --> MERGE --> GOLD
    MASTER --> MERGE
    GOLD -.-> CUR

    GOLD --> PRE --> TFIDF
    GOLD --> PRE --> SBERT
    GOLD --> PRE --> JB
    CUES --> JB
    CUES -.-> EXPL

    TFIDF --> METRICS
    SBERT --> METRICS
    JB --> METRICS
    METRICS --> EXP --> EXPL --> DASH

    classDef done fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724
    classDef waiting fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#856404
    classDef todo fill:#f8f9fa,stroke:#adb5bd,stroke-width:2px,stroke-dasharray:5 5,color:#6c757d

    class EXTRACT,BUILD,VALID,SPLIT,MASTER,APP,GUIDE,AGREE,MERGE,CUR,PRE,CUES done
    class GOLD waiting
    class TFIDF,SBERT,JB,METRICS,EXP,EXPL,DASH todo
```

---

## Repository layout

```
NLP_Project/
├── configs/config.yaml          # single source of every setting; nothing hard-coded
├── data/                        # see data/README.md
│   ├── raw/{sources,sections,authored}/
│   ├── processed/lmp_pairs.csv  # THE master corpus
│   │   └── curator_drafts.csv   # CURATOR-ONLY drafts — never shown to annotators
│   └── annotations/             # annotator files + adjudications (no drafts here)
├── src/
│   ├── data/                    # schema, loading, building, splitting, validation
│   │   ├── schema.py            # the single source of truth for the dataset schema
│   │   ├── build_dataset.py     # authored JSONL -> master CSV
│   │   ├── preprocess.py        # deliberately conservative, legally aware
│   │   ├── legal_cues.py        # modals, negation, conditions, quantities, entities
│   │   └── {validate,split,extract}_*.py
│   ├── annotation/              # guidelines.md, annotation_app.py, agreement.py,
│   │                            # merge_annotations.py, curator_review.py (curator-only)
│   └── utils/                   # config, IO with provenance, logging
├── reports/                     # generated + written analyses
│   ├── data_quality.md
│   ├── split_diagnostics.md
│   ├── agreement.md             # RESULT_PENDING until annotators exist
│   ├── curator_draft_review.md  # CURATOR-ONLY: drafts vs human labels
│   └── source_provenance.md     # ← why the Hindi half is author-rendered
├── results/                     # machine-readable outputs, stamped with provenance
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip freeze > requirements.lock.txt      # pin exact versions for the paper

# 2. Rebuild the corpus from the authored batches
python -m src.data.build_dataset         # -> data/processed/lmp_pairs.csv
python -m src.data.validate_data --write-report
python -m src.data.split_data --report

# 3. Annotate (the blocking step — see below)
streamlit run src/annotation/annotation_app.py
```

Expect `validate_data` to report **0 errors and 2 warnings**. Both warnings are
correct and expected while the corpus is unannotated (`gold_coverage`,
`no_human_annotations`); if either disappears before annotation is done,
something has fabricated labels and the build should be investigated.

---

## The corpus at a glance

| Property | Value |
| --- | --- |
| Pairs | **340** = 170 English + 170 Hindi |
| Legal domains | 3 — contract (202), cyber (72), sale_of_goods (66) |
| Source provisions | 159 distinct `source_group`s, each with both languages |
| Splits | 234 train / 50 validation / 56 test |
| Groups spanning splits | **0** (leakage-free by construction) |
| LMP score range | 1–10, all ten values populated |
| Error phenomena | all 11 categories the brief requires, incl. Date/Number/Entity |
| **Human labels** | **0 — annotation pending** |

Full documentation: [`data/README.md`](data/README.md) and
[`data/raw/README.md`](data/raw/README.md).

---

## The integrity rule

The project's central discipline is that **a machine-drafted score is never a
human label**, and the data structure enforces it rather than relying on care:

| Column | Filled by | Rows filled |
| --- | --- | --- |
| `draft_lmp_score`, `draft_error_type` | the corpus author | 340 — *suggestions, not labels* |
| `human_lmp_score`, `error_type`, `decision` | a human annotator | **0** |

Enforcement is in code, at three layers:

1. **`schema.is_gold()`** — a row is ground truth only if `annotation_status`
   is a `human_*` value *and* it actually carries a score. The second condition
   catches a row marked `human_single` whose score was accidentally cleared.
2. **`load_data.assert_gold()`** — every experiment calls this first, and on the
   current corpus it raises. There are two messages, depending on how the data
   was loaded, and both are correct:

   ```
   >>> assert_gold(load_master(subset="gold"), "Experiment 1")
   ValueError: No rows available for Experiment 1. TODO_HUMAN_ANNOTATION —
   annotate the corpus before evaluating.

   >>> assert_gold(load_master(subset="all"), "Experiment 1")
   ValueError: Refusing to evaluate Experiment 1: 340 row(s) are not
   human-annotated (statuses present: ['draft']). Only ['human_consensus',
   'human_multi_pending', 'human_single'] count as ground truth.
   ```
3. **`build_dataset.assert_no_premature_judgements()`** — the build *refuses to
   write* a master file containing any populated gold column, so a mistake
   upstream cannot ship a contaminated corpus.

Baseline models additionally require an explicit, documented linear calibration
fitted on the **train split only**. Cosine similarity is never reported as an
LMP score — see `configs/config.yaml` → `models.tfidf.calibration`.

---

## Annotation workflow

Annotation is the blocking step. The infrastructure is complete and waiting.

```bash
streamlit run src/annotation/annotation_app.py
```

The tool has three tabs:

* **Annotate** — 1–10 scoring. Strictly **blind**: the curator's draft is never
  rendered, before *or* after a score is saved. There is no reveal control.
  The tool allows a score to be revised, so a draft shown after the first save
  would anchor the revision just as a draft shown before it anchors the
  original — a partially-anchored label is not an independent label. The queue
  offers no draft-based ordering either, since sorting by a hidden field leaks
  it through the position of the pair.
* **Review** — your own progress and your own score distribution, plus the
  corpus-wide agreement coefficients (Cohen's κ quadratic, Krippendorff's α
  ordinal, ICC(2,1), % within ±1 point), or an explicit `RESULT_PENDING`
  explaining why they cannot be computed yet. Per-pair scores from *other*
  annotators are withheld here and appear only in the Adjudicate tab.
* **Adjudicate** — pairs where two annotators differ by ≥2 points. The
  adjudicator must be a different person from both annotators, and the tool
  refuses the record if they are not.

```
annotators score independently  →  agreement.py measures it
        ↓                                    ↓
  <id>.csv per annotator          spread ≥ 2 → adjudication (a third person)
        ↓                                    ↓
        └──────→ merge_annotations.py ←──────┘
                         ↓
        human_single / human_multi_pending / human_consensus
```

Annotations are written to a temporary file and moved into place with
`os.replace`, so an interrupted save cannot destroy the scores already
recorded. Those rows are hand-entered human judgement and cannot be
regenerated; a truncated file would be an afternoon's work lost.

Agreement values are **computed only from annotation files that exist** and are
never estimated, and only from the annotator files — never from the
adjudication file, which would make the coefficient partly a measure of the
adjudicator's own choices. The module returns `RESULT_PENDING` rather than a
number when fewer than 20 pairs have been scored twice, because a coefficient
computed on a handful of rows is noise and reporting it would be worse than
reporting nothing.

The curator's draft scores are compared against the human labels by
`src/annotation/curator_review.py`, which is **curator-only** — it prints the
draft beside the human score for every pair, so showing it to an annotator
would hand them the expected label. It reports `RESULT_PENDING` until human
labels exist.

Read [`src/annotation/guidelines.md`](src/annotation/guidelines.md) before
scoring anything — §2 holds the scale anchors, §4 the error taxonomy, and §7
the confusions that cost the most agreement.

---

## Planned experiments

From the project brief; none has been run.

| # | Experiment | Design | Status |
| --- | --- | --- | --- |
| 1 | Baselines vs JUDGEBERT | TF-IDF, SBERT, JUDGEBERT on the test split | `TODO_RUN_EXPERIMENT` |
| 2 | English vs Hindi | same three models, split by language | `TODO_RUN_EXPERIMENT` |
| 3 | Generalisation | train on contract + sale_of_goods, test on held-out **cyber** | `TODO_RUN_EXPERIMENT` |
| 4 | Sanity checks | identical pairs, unrelated pairs, modal-swap pairs | `TODO_RUN_EXPERIMENT` |
| 5 | Qualitative error analysis | ≥15 hard cases across all 11 error phenomena | `TODO_RUN_EXPERIMENT` |

Experiment 3 uses `cyber` as the held-out domain — it is the only domain with
no representation in training, and `configs/config.yaml` asserts at load time
that `generalization_domain != primary_domain`, so the experiment cannot
silently degrade into interpolation.

---

## Reproducibility

Every artefact in `results/` carries a provenance header: UTC timestamp, git
revision, Python version, platform, config fingerprint and seed. The config
fingerprint is a SHA-256 of the config with machine-specific absolute paths
stripped, so two machines running the same experiment produce the same
fingerprint.

A single seed in `configs/config.yaml` (`project.seed`) drives splitting,
sampling and training. `set_global_seed()` additionally pins
`torch.use_deterministic_algorithms` where available.

This project directory is its own git repository. Until it has a first commit,
`git_revision()` reports `"unknown"` rather than borrowing a revision from some
other repository — a provenance field that names the wrong commit is worse than
one that admits it does not know. Commit once and every subsequent results file
stamps the real hash.

```bash
python -m src.data.build_dataset      # deterministic
python -m src.data.split_data --report # deterministic; re-running is a no-op
python -m src.annotation.agreement --report
python -m src.annotation.merge_annotations   # dry run by default
python -m src.annotation.curator_review      # CURATOR-ONLY; never run for an annotator
```

`merge_annotations` writes in place and is therefore destructive, so it prints
a full change summary and **writes nothing without `--apply`**. Use
`--apply --backup` to keep a timestamped copy of the pre-merge master.

Cite `requirements.lock.txt`, not `requirements.txt`, in the paper — only the
lock file pins exact versions.

### Hardware

Developed on Apple Silicon (macOS, MPS backend). `torch` installs the
MPS-enabled build automatically. For CUDA, install torch from the appropriate
wheel index first, then the rest of `requirements.txt`. No GPU is required to
run the data layer, the annotation tool, or the baselines.

---

## Testing

```bash
.venv/bin/python -m pytest tests/ -q      # 325 tests, ~2s, no GPU, no network
```

No test downloads a model, calls an API, or needs a GPU, so the suite runs in
about a second and can gate every commit.

| Module | Tests | Covers |
| --- | --- | --- |
| `test_schema.py` | 36 | column contract, dtype coercion, the gold gate, CSV round-trip |
| `test_load_data.py` | 27 | loading, derived columns, `assert_gold` refusal messages |
| `test_split_data.py` | 18 | group integrity, reproducibility, stratification |
| `test_preprocess.py` | 39 | normalisation, Devanagari tokenisation, the legal-content guard |
| `test_legal_cues.py` | 67 | modals, negation, conditions, exceptions, quantities |
| `test_agreement.py` | 48 | κ / α / ICC(2,1) against known-answer cases, the `RESULT_PENDING` gate |
| `test_annotation_merge.py` | 27 | status assignment, adjudication precedence, error-type conflicts |
| `test_annotation_app.py` | 63 | blind-mode guarantee, score range, atomic saves, queue walk, adjudicator independence |

Three habits are deliberate:

* **Known-answer cases.** The agreement coefficients are checked against
  hand-computed values (perfect agreement → 1.0, systematic opposition → κ = 0.0,
  zero variance → undefined), not merely against their own previous output.
* **Regression tests are named after the bug.** Each one states the symptom and
  why it mattered, so the test explains itself to whoever trips it. The
  tokenisation and quantity regressions in `test_preprocess.py` and
  `test_legal_cues.py` are the two worth reading.
* **The workflow tests are mutation-checked.** A test that passes the first
  time deserves suspicion, so the annotation tests were re-run against a
  deliberately broken app — the queue cursor reverted to its skipping version,
  and a post-save draft reveal reinstated. Both break the suite. That check
  caught a weak assertion of my own: an end-to-end test asserting the queue was
  *ascending* passed happily while every other pair was being skipped, because
  a skipping queue is still ascending. It now asserts the exact pairs the
  default queue must present.

A test that asserts a guarantee the code does not actually make is worse than
no test, so several tests here also pin behaviour that looks like a defect and
is not — cross-category cue overlap, and trusting a file's declared
`annotator_id` over its filename are both load-bearing (see the docstrings).

---

## Known limitations

Stated here rather than buried, because each one constrains what the paper can
claim.

1. **The Hindi half is author-rendered, not quoted.** Every official Hindi
   statute PDF obtained uses a legacy non-Unicode font whose glyph mapping is
   broken; the Hindi word for "Act" appears zero times in a 41-page Hindi Act.
   All 170 Hindi rows carry `verbatim_quote: "false"`. The two language halves
   are therefore **not equivalent provenance categories**, so Experiment 2
   mixes a language effect with a provenance effect. Full evidence:
   [`reports/source_provenance.md`](reports/source_provenance.md).
2. **The corpus is small** (340 pairs), which is why splits are grouped by
   `source_group` to prevent sibling leakage, and why bootstrap confidence
   intervals are reported alongside every metric.
3. **The corpus is unaudited by humans.** Every score currently in the data is
   the author's draft.
4. **"JUDGEBERT-inspired" is a deliberate hedge.** The original work's exact
   architecture and training recipe are not fully specified in the public
   paper, so this project implements a cross-encoder regression model in the
   same spirit and does not claim to reproduce it. See below.

### Naming and honesty

* The model is **JUDGEBERT-*inspired***, not a reproduction.
* **Attention weights are not a legal explanation.** The explainability layer
  leads with auditable cue alignment, and reports attention as weak supporting
  evidence only, with an explicit disclaimer in the UI.
* **No fabricated numbers.** Every metric in the paper must come from an actual
  run. Placeholders `TODO_RUN_EXPERIMENT`, `TODO_HUMAN_ANNOTATION` and
  `RESULT_PENDING` mark the gaps rather than papering over them.
* **Agreement values are measured, never asserted.**
* **No paid external API** is required anywhere in the pipeline.

---

## Documentation index

| Document | What it covers |
| --- | --- |
| [`src/annotation/guidelines.md`](src/annotation/guidelines.md) | The annotation procedure, scale anchors, error taxonomy, agreement protocol |
| [`data/README.md`](data/README.md) | Schema, integrity property, annotation workflow, rebuild instructions |
| [`data/raw/README.md`](data/raw/README.md) | Source documents, authored batch format, deliberate corpus design |
| [`reports/source_provenance.md`](reports/source_provenance.md) | Source provenance and the Hindi PDF corruption evidence |
| [`reports/data_quality.md`](reports/data_quality.md) | Latest validation output |
| [`reports/split_diagnostics.md`](reports/split_diagnostics.md) | Split balance and leakage checks |
| [`reports/agreement.md`](reports/agreement.md) | Inter-annotator agreement (currently `RESULT_PENDING`) |

---

## Licence

Source statute texts are Government of India works, not subject to copyright
under s.52(1)(q) of the Indian Copyright Act, 1957. See [`LICENSE`](LICENSE)
for the terms covering this repository's own code and data.
