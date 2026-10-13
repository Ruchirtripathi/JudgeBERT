# Stage 1 — Data Integrity Report

**Audit Date:** 2026-10-13
**Auditor:** Automated audit (read-only)

---

## 1. `lmp_pairs.csv` — Source Pair Dataset

### Schema Validation

| Column | Present | Non-null | Notes |
|---|---|---|---|
| `pair_id` | Yes | 340/340 | 340 unique — no duplicates |
| `language` | Yes | 340/340 | 170 `en`, 170 `hi` |
| `legal_domain` | Yes | 340/340 | `contract` 202, `cyber` 72, `sale_of_goods` 66 |
| `original_text` | Yes | 340/340 | All non-empty |
| `simplified_text` | Yes | 340/340 | All non-empty |
| `human_lmp_score` | Yes | **0/340** | No human annotations exist yet — by design |
| `decision` | Yes | **0/340** | No human decisions yet — by design |
| `error_type` | Yes | **0/340** | No human error labels yet — by design |
| `split` | Yes | 340/340 | `train` 234, `validation` 50, `test` 56 |
| `source_group` | Yes | 340/340 | 159 unique groups |
| `annotation_status` | Yes | 340/340 | All `"draft"` (no `human_consensus` or `human_single`) |
| `draft_lmp_score` | Yes | 340/340 | Curator draft scores 1–10 |
| `draft_error_type` | Yes | 340/340 | Curator draft labels |
| `verbatim_quote` | Yes | 340/340 | `True` for English, `False` for Hindi |

### Split Distribution

| Split | Count | English | Hindi |
|---|---|---|---|
| train | 234 | 117 | 117 |
| validation | 50 | 25 | 25 |
| test | 56 | 28 | 28 |
| **Total** | **340** | **170** | **170** |

**Split leakage:** None — source groups do not span multiple splits (confirmed by `build_manifest.json`).

### Domain Distribution

| Domain | Count | % |
|---|---|---|
| contract | 202 | 59.4% |
| cyber | 72 | 21.2% |
| sale_of_goods | 66 | 19.4% |

**Note:** The config designates `sale_of_goods` as the out-of-domain generalisation test domain. This is appropriate given its smaller size.

### Annotation Status

**No human ground-truth annotations exist.** All 340 rows have:
- `annotation_status = "draft"`
- `human_lmp_score = NaN` (empty)
- `decision = NaN` (empty)
- `error_type = NaN` (empty)

`draft_lmp_score` and `draft_error_type` are curator-assigned drafts and must **not** be treated as human gold labels. This is correctly enforced by `src/data/schema.py::is_gold()`.

### Draft Score Distribution (not gold labels)

| Score | Count |
|---|---|
| 1 | 6 |
| 2 | 20 |
| 3 | 20 |
| 4 | 54 |
| 5 | 78 |
| 6 | 30 |
| 7 | 20 |
| 8 | 48 |
| 9 | 56 |
| 10 | 8 |

### Source Provenance and Language Notes

- **English pairs (170):** `verbatim_quote = True` — sourced from official public-domain PDF statutes (Indian Contract Act 1872, IT Act 2000, Sale of Goods Act 1930, Specific Relief Act 1963). Source PDFs tracked in `data/raw/sources/`.
- **Hindi pairs (170):** `verbatim_quote = False` — **author-rendered** Devanagari translations, not verbatim official Hindi legal text. This is a documented limitation. The corpus does not claim to contain official Hindi statutory text.
- **Duplicate original_text values:** A small number of English source sentences appear as originals for both English and Hindi pairs (expected for source-group cross-language testing; documented in the final corpus audit).

### Duplicate Check

- Duplicate `pair_id`: **0**
- Duplicate simplified sentences: **0** (within language)
- Source groups spanning splits: **0** (anti-leakage constraint holds)

---

## 2. `ai_annotations.csv` — AI Annotation File

### Current Structural and Field Validation

- Rows: **315**
- Unique pair IDs: **315**
- Duplicate pair IDs: **0**
- Fully valid under the current field checks: **300**
- Incomplete or invalid under at least one field check: **15**

| Validation check | Invalid/missing rows |
|---|---:|
| Score outside 1–10 or missing | 3 |
| Error type outside the allowed schema or missing | 12 |
| Empty explanation | 3 |
| Confidence outside 0–1 or missing | 3 |
| Empty model name | 0 |

These counts overlap. A row can fail multiple checks. A structurally valid annotation is **not necessarily legally correct**.

### Score Distribution

| Score | Count |
|---|---:|
| 1.0 | 164 |
| 5.0 | 2 |
| 7.0 | 16 |
| 8.0 | 31 |
| 9.0 | 6 |
| 10.0 | 93 |
| Missing | 3 |

The distribution is concentrated at scores 1 and 10. This warrants investigation; it does not establish whether the scores are accurate or inaccurate.

### Annotation Quality Warnings

Manual review is needed for annotations whose explanations contradict their labels, describe a legal error while assigning `None`, or contain self-corrections and uncertainty. The automated field checks do not detect all such cases.

- No human gold labels are available in the source dataset.
- AI annotations remain provisional and must not be presented as human-validated judgments.
- Do not train or evaluate a legal meaning-preservation model against these annotations as if they were gold labels.
- The current CSV contains 315 annotations for 340 source pairs; **25 source pairs are not represented**, assuming all annotation IDs refer to the source dataset.
- Preserve local backups as an audit trail. Do not automatically repair or relabel questionable annotations.

---

## 3. `sbert_similarity_baseline.csv` — Baseline Output

### Schema

| Column | Present | Non-null | Notes |
|---|---|---|---|
| `pair_id` | Yes | 340/340 | 340 unique — no duplicates |
| `language` | Yes | 340/340 | Consistent with source |
| `legal_domain` | Yes | 340/340 | Consistent with source |
| `split` | Yes | 340/340 | Consistent with source |
| `cosine_similarity` | Yes | 340/340 | All finite, range [0.695, 0.999] |

### Similarity Statistics

| Statistic | Value |
|---|---|
| Mean | 0.950 |
| Std | 0.046 |
| Min | 0.695 |
| 25th pct | 0.933 |
| Median | 0.963 |
| 75th pct | 0.981 |
| Max | 0.999 |

All values finite and within [-1, 1]. High mean cosine similarity (0.95) is expected — simplified legal text closely mirrors the original.

### Model Used

`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

Referential integrity: all 340 `pair_id` values match `lmp_pairs.csv`. ✅

---

## 4. Overall Integrity Findings

| Check | Result |
|---|---|
| No duplicate pair IDs in any file | PASS |
| All annotation IDs exist in source | PASS |
| Baseline covers all 340 pairs | PASS |
| No split leakage | PASS |
| Human gold labels present | **FAIL — 0 human labels; all are drafts** |
| AI annotation field validity | **CAUTION — current CSV audit required** |
| Score distribution is unimodal/expected | **CAUTION — bimodal at 1 and 10** |
| Hindi provenance is verbatim official text | **LIMITATION — author-rendered; documented** |

---

## 5. Required Actions (Data Layer)

1. **Do not treat AI annotations as gold labels.** They are provisional and incomplete.
2. **Do not train any model** on `ai_lmp_score` as ground truth until human annotation is complete.
3. **Repair `ai_error_type`** only when Ollama is running and Phase 4 safety fixes are applied.
4. **Investigate the bimodal score distribution** — determine whether scores from the earlier run used a different prompt or temperature before relying on them.
5. **Retain the backup files** (`ai_annotations_backup_before_repair.csv`, `ai_annotations_before_retry.csv`) locally as audit trail.

---

*Report generated: 2026-10-13. No files modified during this phase.*
