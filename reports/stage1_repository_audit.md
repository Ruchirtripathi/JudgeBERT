# Stage 1 — Repository Audit Report

**Audit Date:** 2026-10-13
**Auditor:** Senior NLP/ML Engineer (automated audit)
**Project:** Explainable Multilingual Legal Meaning Preservation (JUDGEBERT-inspired)
**Repository path:** `~/NLP_Project`

---

## 1. Git State

| Item | Status |
|---|---|
| Current branch | `main` |
| Remotes | **None configured** |
| Commits | 1 (`e196a27` — "Initial commit: multilingual legal meaning preservation project") |
| Modified tracked files | `src/annotation/ai_annotation_ollama_pilot.py` (114 changed lines) |
| Untracked files | `data/processed/ai_annotations_backup_before_repair.csv`, `data/processed/ai_annotations_before_retry.csv`, `repair_error_labels.py` |
| Staged files | None |

> There is only **one commit** in the repository. If the local `.git` directory is lost, all history is lost permanently. No remote backup exists.

---

## 2. Python Environment

| Item | Detail |
|---|---|
| System Python | 3.12.6 (via pyenv) |
| `.venv` Python | 3.12.6 |
| `.venv` active? | No — scripts must be invoked with `.venv/bin/python` explicitly |
| Key packages | `sentence-transformers 6.1.0`, `torch 2.14.1`, `pandas 3.0.6`, `pytest 9.1.1`, `requests 2.34.2`, `streamlit 1.65.0` |
| `JUDGEBERT` / `muril` weights | Not installed — training is planned, not implemented |
| `transformers` | 5.18.0 installed (needed for future training) |

---

## 3. Ollama / Model Status

| Item | Status |
|---|---|
| Ollama server reachable? | **NO** — `ollama serve` is not running |
| `qwen3:4b` locally cached? | Unknown — cannot check while server is stopped |
| Last annotation failure | `Invalid inner JSON from Ollama: Expecting value: line 1 column 1 (char 0)` (historical) |
| Probable cause | Ollama was not running, causing empty HTTP body; the error is a **connection failure masquerading as a JSON parse failure**, not a schema mismatch |

---

## 4. Source File Inventory

### Core scripts

| File | Purpose |
|---|---|
| `src/data/build_dataset.py` | Assembles `lmp_pairs.csv` from raw JSONL |
| `src/data/validate_data.py` | Schema and integrity checks |
| `src/data/split_data.py` | Train/val/test assignment |
| `src/data/schema.py` | Canonical column/vocabulary definitions |
| `src/annotation/sbert_full_dataset.py` | SBERT baseline — all 340 pairs |
| `src/annotation/ai_annotation_ollama_pilot.py` | AI annotation via Ollama **(modified, uncommitted)** |
| `src/annotation/annotation_app.py` | Streamlit review UI |
| `src/annotation/agreement.py` | Inter-annotator agreement calculation |
| `src/analysis/audit_corpus.py` | Corpus audit script |
| `repair_error_labels.py` | Ad-hoc repair script **(untracked, not committed)** |

### Tests (14 test modules)

All tests pass: **346 passed, 2 warnings** in 2.24 s (offline, no Ollama required).

---

## 5. Configuration and Credential Files

| File | Notes |
|---|---|
| `configs/config.yaml` | Comprehensive project config; no secrets |
| `conftest.py` (root) | Minimal pytest configuration |
| `tests/conftest.py` | Test fixtures |
| `.streamlit/credentials.toml` | **Contains a user email address** — currently in `.gitignore` |
| `.streamlit/config.toml` | Does not exist |
| `.streamlit_home/`, `.streamlit_custom/` | Exist; correctly listed in `.gitignore` |

WARNING: `.streamlit/credentials.toml` contains a user email address. It is correctly in `.gitignore`. Verify it is NOT tracked before any push using `git ls-files .streamlit/`.

---

## 6. Detected Problems and Risks

### CRITICAL

| # | Problem | Risk | Proposed Fix |
|---|---|---|---|
| C1 | `call_ollama()` contains **dead code** (lines 159–191): a second copy of the original function body after the `return` on line 158. | Confusion; wrong body edited in future | Remove orphan dead code block in Phase 4 |
| C2 | `record_failure()` unconditionally **deletes the existing row** for a `pair_id` and writes a null record. If called for a `pair_id` that has a prior valid annotation, it destroys that annotation silently. | Silent data loss on retry | Guard `record_failure()` with `is_successful_checkpoint()` before deletion |
| C3 | **234 / 315 annotation rows** have `ai_error_type = NaN`. These rows have valid scores and explanations. The model returned results without `error_type`, or the field was nullified during a prior run. | Unusable annotations; blocks evaluation | Requires Ollama running; safe repair plan in Phase 4 |
| C4 | **No git remote** configured. The sole commit is local only. | All work lost if machine fails | Add a remote before handoff (user must approve) |

### HIGH

| # | Problem | Risk | Proposed Fix |
|---|---|---|---|
| H1 | Ollama server not running | Annotation pipeline inoperative | `ollama serve` before any annotation attempt |
| H2 | `repair_error_labels.py` is untracked and uncommitted | Lost on clean clone | Commit or explicitly remove |
| H3 | Backup CSVs (`ai_annotations_backup_before_repair.csv`, `ai_annotations_before_retry.csv`) are untracked | Backups not preserved in repo | Document as local-only; in `.gitignore` already |
| H4 | Modified `ai_annotation_ollama_pilot.py` is uncommitted | Changes not in history | Commit after Phase 4 fixes confirmed |
| H5 | 25 pairs have no row in `ai_annotations.csv` at all | Incomplete annotation run | Do not bulk-annotate until C2 is patched and Ollama is running |

### MEDIUM

| # | Problem | Risk | Proposed Fix |
|---|---|---|---|
| M1 | `.venv` not activated in shell PATH | Teammate confusion | Document explicit `.venv/bin/python` usage in README |
| M2 | `curator_drafts.csv` is tracked but its relationship to `ai_annotations.csv` is unclear in the README | Misleading data status | Clarify in README and handoff |
| M3 | PDF source files tracked in `data/raw/sources/` may cause repository bloat | Large repo size | Check sizes; consider LFS or separate transfer |

### LOW

| # | Problem |
|---|---|
| L1 | `DeprecationWarning: google._upb._message.MessageMapContainer` (2 warnings, cosmetic, Python 3.14 future) |
| L2 | `build_manifest.json` records `git_revision: "unknown"` — weak provenance |

---

## 7. Data File Summary

| File | Rows | Unique IDs | Status |
|---|---|---|---|
| `lmp_pairs.csv` | 340 | 340 | Authoritative source; no defects found |
| `sbert_similarity_baseline.csv` | 340 | 340 | Complete and consistent |
| `ai_annotations.csv` | 315 | 315 | 25 pairs missing; 234 have null `ai_error_type` |
| `curator_drafts.csv` | 340 | 340 | Draft scores only; not ground truth |

---

*Report generated: 2026-10-13. Read-only inspection — no files modified during this phase.*


## 8. Current-State Reconciliation — 2026-10-13

This addendum supersedes stale counts and environment observations in earlier sections.

- Latest full test suite: **358 passed, 2 deprecation warnings**.
- The current shell shows the project virtual environment as active (`.venv`).
- `ai_annotations.csv`: **315 rows**, 315 unique IDs; **300 pass the current field checks and 15 fail at least one check**. These are structural checks, not legal correctness judgments.
- Invalid/missing field counts overlap: score 3; error type 12; explanation 3; confidence 3; model name 0.
- The source dataset has 340 rows. There are 25 source pairs without annotation rows.
- The current Git audit found no tracked files matching the credential-like filename patterns checked. This is a targeted filename check, not a guarantee that every possible secret is absent.
- Ollama is currently unavailable; the historical cause of the earlier JSON failure has not been established.
- Current repository status must be obtained with `git status --short`; the Git-state table above is a historical snapshot, not the current status.

No claim is made that JUDGEBERT has been trained or that the AI-generated annotations are human-validated ground truth.
