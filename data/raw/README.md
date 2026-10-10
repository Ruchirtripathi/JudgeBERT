# Raw Inputs

Nothing in this directory is generated. Every file here is either a retrieved
source document or an authored record. Files here are treated as immutable: if
something is wrong, fix it upstream and re-run the build rather than editing a
derived file.

```
raw/
├── sources/   8 official statute PDFs — 4 English, 4 Hindi
├── sections/  per-section JSONL parsed from the 4 English PDFs
└── authored/  7 JSONL batches — the corpus source of truth
```

---

## 1. `sources/` — the retrieved documents

| File | Instrument | Act No. | Pages | Extraction |
| --- | --- | --- | --- | --- |
| `indian_contract_act_1872_eng.pdf` | Indian Contract Act, 1872 | 9 of 1872 | 53 | ✅ clean |
| `indian_contract_act_1872_hin.pdf` | Indian Contract Act, 1872 | 9 of 1872 | 41 | ❌ corrupt |
| `it_act_2000_eng.pdf` | Information Technology Act, 2000 | 21 of 2000 | 41 | ✅ clean |
| `it_act_2000_hin.pdf` | Information Technology Act, 2000 | 21 of 2000 | 37 | ❌ corrupt |
| `sale_of_goods_1930_eng.pdf` | Sale of Goods Act, 1930 | 3 of 1930 | 16 | ✅ clean |
| `sale_of_goods_1930_hin.pdf` | Sale of Goods Act, 1930 | 3 of 1930 | 13 | ❌ corrupt |
| `specific_relief_eng.pdf` | Specific Relief Act, 1963 | 47 of 1963 | 17 | ✅ clean |
| `specific_relief_hin.pdf` | Specific Relief Act, 1963 | 47 of 1963 | 15 | ❌ corrupt |

All eight were retrieved from
`https://huggingface.co/datasets/d-riti/Dataset-For-Indian-legal-knowledge-base`,
which redistributes the official Government of India texts. The instruments
themselves are Government of India works, not subject to copyright under
s.52(1)(q) of the Indian Copyright Act, 1957.

**Why the four Hindi files are marked corrupt.** They use legacy non-Unicode
Devanagari fonts with broken glyph mappings. The extracted codepoints are in
the Devanagari block but they are the *wrong* codepoints, and the same source
character decodes differently in different words — so no lookup table can
reverse it. The strongest single piece of evidence:

> In `it_act_2000_hin.pdf`, page 1, the word अधिनियम (Act) extracts as
> अगिगनयम and धाराओं (sections) extracts as िाराओं — both on the same page,
> 24 characters apart. The `ध` became `ग` in one word and `ि` in the other.

Consequently the Hindi word for "Act" appears **zero times** in a 41-page
Hindi Act, and the Specific Relief Act's own subject word अनुतोष appears
**zero times** in a PDF entirely about अनुतोष.

Full evidence, including the English control and the ZWJ counts:
[`reports/source_provenance.md`](../../reports/source_provenance.md).

**`specific_relief_*` is registered but unused as a corpus domain.** It is
retained because Experiment 5's "Entity" probes cite Specific Relief Act
version years (1877 vs 1963). It is deliberately absent from
`dataset.domains` in `configs/config.yaml`.

---

## 2. `sections/` — parsed sections

One JSONL file per **English** instrument. Each file begins with a `_meta`
record and is followed by one record per section:

```json
{"_meta": {"path": "data/raw/sources/it_act_2000_eng.pdf",
           "title": "The Information Technology Act, 2000",
           "act_number": "Act No. 21 of 2000", "language": "en",
           "legal_domain": "cyber",
           "retrieved_from": "https://huggingface.co/datasets/...",
           "licence": "Apache-2.0 (dataset packaging); ...",
           "encoding_ok": true,
           "notes": "Official English text. Extracts cleanly with pypdf."}}
```

**There are no Hindi section files, and their absence is deliberate.** There is
no trustworthy Hindi section text to write, because the Hindi PDFs do not
extract. An empty directory would be misleading; a generated one would be
fabrication.

Regenerate with:

```bash
.venv/bin/python -m src.data.extract_sources
```

---

## 3. `authored/` — the corpus source of truth

Seven hand-authored batches. These, not the PDFs, are what `build_dataset`
reads, and they are the authoritative record of every pair in the corpus.

| Batch | Rows | Domain | Source statute | Provisions |
| --- | --- | --- | --- | --- |
| `contract_batch01.jsonl` | 50 | contract | Indian Contract Act, 1872 | 25 × 2 langs |
| `contract_batch02.jsonl` | 50 | contract | Indian Contract Act, 1872 | 25 × 2 langs |
| `contract_batch03.jsonl` | 50 | contract | Indian Contract Act, 1872 | 25 × 2 langs |
| `contract_batch04.jsonl` | 50 | contract | Indian Contract Act, 1872 | 25 × 2 langs |
| `cyber_batch01.jsonl` | 60 | cyber | Information Technology Act, 2000 | 30 × 2 langs |
| `sale_of_goods_batch01.jsonl` | 64 | sale_of_goods | Sale of Goods Act, 1930 | 32 × 2 langs |
| `targeted_error_batch01.jsonl` | 16 | mixed | — (probes) | 8 × 2 langs |
| **Total** | **340** | | | **159 distinct groups** |

Each batch is one row per (provision × language), so a batch of 30 provisions
yields 60 rows: 30 `ENG_*` and 30 `HIN_*` sharing the same `source_group`.

**The group count needs one line of explanation, because the arithmetic does
not add up on its own.** Summing the per-batch group counts gives 170, but the
corpus has **159** distinct `source_group` values. Eleven groups appear in more
than one batch:

| Shared group | Appears in | Why |
| --- | --- | --- |
| `ICA1872::10`, `::37` | contract batches 01 and 02 | the same provision was independently authored twice |
| `ICA1872::126` | contract batches 02 and 04 | as above |
| `ICA1872::1`, `SOGA1930::58`, `ITA2000::23/24/44/45/62/74` | a domain batch + the probe batch | deliberate — see below |

The three contract-internal overlaps were **not** planned; they were found by
the grouping check. They are left in place rather than removed, because they
demonstrate that the anti-leakage mechanism is doing real work on the
naturally-authored corpus and not merely on the synthetic probes. Removing them
would make the corpus tidier and the leakage test weaker.

### Record format

Exactly thirteen keys, enforced by `build_dataset.read_authored`, which
hard-fails on any missing or unexpected key rather than silently ignoring it:

| Key | Type | Notes |
| --- | --- | --- |
| `pair_id` | str | `<LANG>_<DOMAIN>_<NNN>`, unique across the corpus |
| `language` | `"en"` \| `"hi"` | |
| `legal_domain` | str | must match the statute's registered domain |
| `source_section` | str | section number in the cited instrument |
| `source_group` | str | `<STATUTE_KEY>::<section>` — the anti-leakage key |
| `verbatim_quote` | `"true"` \| `"false"` | `"true"` for English only; the build rejects a Hindi `"true"` |
| `simplification_type` | str | how the rewrite was produced (see below) |
| `original_text` | str | the statutory sentence |
| `simplified_text` | str | the rewrite being judged |
| `context` | str | neighbouring text; `""` when not supplied |
| `draft_lmp_score` | int 1–10 | the author's provisional score |
| `draft_error_type` | str | the author's provisional error label |
| `curator_notes` | str | why the pair exists and what defect it carries |

```json
{"pair_id": "ENG_CON_101", "language": "en", "legal_domain": "contract",
 "source_section": "1", "source_group": "ICA1872::1", "verbatim_quote": "true",
 "simplification_type": "hallucination",
 "original_text": "... of the value or amount of five hundred rupees or upwards ...",
 "simplified_text": "... of the value or amount of one thousand rupees or upwards ...",
 "context": "", "draft_lmp_score": 3, "draft_error_type": "Number",
 "curator_notes": "The statutory monetary threshold is silently doubled ... shares
                   source_group ICA1872::1 with ENG_CON_076, so the pair also
                   exercises the anti-leakage grouping."}
```

### Deliberate design of the corpus

`draft_lmp_score` spans the whole scale, and the spread is intentional rather
than incidental:

| Score | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rows | 6 | 20 | 20 | 54 | 78 | 30 | 20 | 48 | 56 | 8 |

| `simplification_type` | Rows | | `draft_error_type` | Rows |
| --- | --- | --- | --- | --- |
| `faithful_paraphrase` | 112 | | `None` | 112 |
| `omission` | 58 | | `Omission` | 58 |
| `scope_narrowing` | 42 | | `Scope` | 44 |
| `exception_drop` | 42 | | `Exception` | 42 |
| `condition_drop` | 40 | | `Condition` | 40 |
| `hallucination` | 22 | | `Obligation/Permission` | 12 |
| `modal_weakening` | 10 | | `Contradiction` | 10 |
| `negation_flip` | 6 | | `Number` | 8 |
| `contradiction` | 4 | | `Hallucination` | 6 |
| `quantifier_weakening` | 2 | | `Date` | 4 |
| `modal_strengthening` | 2 | | `Entity` | 4 |

Three things this table shows:

1. **Every error phenomenon the brief's Experiment 5 must report on is
   present**, including the ones that are hard to find in real simplification
   data (negation flips, quantifier weakening, modal strengthening).
2. **`targeted_error_batch01.jsonl` exists to fill Date / Number / Entity.**
   Those categories were absent from the naturally-authored batches, so eight
   provisions were re-used as single-defect probes — each one changing exactly
   one number, date or entity so the defect is clean and attributable.
3. **Sibling groups are deliberate.** `targeted_error_batch01.jsonl` reuses
   `source_group` values of rows in other batches (e.g. `ICA1872::1` appears in
   both a contract batch and the probe batch). That is intentional: it makes the
   anti-leakage grouping do real work rather than being trivially satisfied.

### A note on `ENG_SOG_033`

One probe substitutes "Specific Relief Act, 1877" with "1963" and scores it 4.
Its `curator_notes` flag it as **contested**: 1963 is the Act actually in force
today, so a reader could argue the rewrite is an improvement rather than an
error. It is retained because that ambiguity is worth analysing, and the paper
should treat it as a boundary case rather than a clean defect.

---

## 4. Rebuilding

```bash
.venv/bin/python -m src.data.extract_sources      # sources/ -> sections/
.venv/bin/python -m src.data.build_dataset        # authored/ -> processed/
.venv/bin/python -m src.data.validate_data --write-report
.venv/bin/python -m src.data.split_data --report
```

`build_dataset` validates key sets, cross-checks each record's
`legal_domain` and `language` against the statute registry, and refuses to
write if any gold column would be populated. See
[`../README.md`](../README.md) for the full data-layer documentation.
