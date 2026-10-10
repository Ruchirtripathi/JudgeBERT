# Source Provenance and Text-Quality Report

**Purpose.** Every sentence in this corpus is traceable to a named public
source, and every place where the corpus departs from that source is recorded
here. This report is the evidence base for the corpus-integrity claims made in
the paper, and it documents one limitation that materially affects how the
Hindi half of the corpus should be read.

---

## 1. Sources obtained

| # | Instrument | Act No. | Language | Local file | Domain |
| --- | --- | --- | --- | --- | --- |
| 1 | The Indian Contract Act, 1872 | 9 of 1872 | English | `indian_contract_act_1872_eng.pdf` | contract |
| 2 | The Indian Contract Act, 1872 | 9 of 1872 | Hindi | `indian_contract_act_1872_hin.pdf` | contract |
| 3 | The Information Technology Act, 2000 | 21 of 2000 | English | `it_act_2000_eng.pdf` | cyber |
| 4 | The Information Technology Act, 2000 | 21 of 2000 | Hindi | `it_act_2000_hin.pdf` | cyber |
| 5 | The Sale of Goods Act, 1930 | 3 of 1930 | English | `sale_of_goods_1930_eng.pdf` | sale_of_goods |
| 6 | The Sale of Goods Act, 1930 | 3 of 1930 | Hindi | `sale_of_goods_1930_hin.pdf` | sale_of_goods |
| 7 | The Specific Relief Act, 1963 | 47 of 1963 | English | `specific_relief_eng.pdf` | _(registered, unused)_ |
| 8 | The Specific Relief Act, 1963 | 47 of 1963 | Hindi | `specific_relief_hin.pdf` | _(registered, unused)_ |

All eight files were retrieved from the packaging at

> `https://huggingface.co/datasets/d-riti/Dataset-For-Indian-legal-knowledge-base`

which redistributes the official Government of India texts under an Apache-2.0
dataset licence. The underlying instruments are Government of India works and
are not subject to copyright under s.52(1)(q) of the Indian Copyright Act,
1957. Files 7 and 8 are retained for provenance and for the Experiment 5
"Entity" probes (which cite Specific Relief Act version years); they are not
used as a corpus domain, and `specific_relief` is deliberately absent from
`dataset.domains` in `configs/config.yaml`.

**Note on the choice of domains.** Two domains originally considered for this
project — consumer protection and insurance — were dropped. No public-domain
source could be obtained for either that provided **both** languages with
usable text. Rather than substitute machine-translated or paraphrased
"statutory" text and present it as authentic, the project moved to the three
statutes above, which is why `dataset.domains` does not match the illustrative
list in the original plan.

---

## 2. English extraction: verified clean

The English PDFs extract correctly and were used to populate
`data/raw/sections/*_eng.jsonl`. As a control, all four titles were checked
against their expected strings:

| File | Extracted opening (p.1) | Verdict |
| --- | --- | --- |
| `indian_contract_act_1872_eng.pdf` | `THE INDIAN CONTRACT ACT, 1872` | correct |
| `it_act_2000_eng.pdf` | `THE INFORMATION TECHNOLOGY ACT, 2000` | correct |
| `sale_of_goods_1930_eng.pdf` | `THE SALE OF GOODS ACT, 1930` | correct |
| `specific_relief_eng.pdf` | `THE SPECIFIC RELIEF ACT, 1963` | correct |

None of the four contains a single Devanagari codepoint or zero-width joiner.
This control matters: it establishes that the extractor and the PDFs'
typography are sound, so the Hindi failures in §3 are attributable to the
Hindi fonts and not to the extraction toolchain.

---

## 3. Hindi extraction: unusable, and irreversibly so

**All four Hindi PDFs fail.** The failure is not that text is missing — the
Devanagari codepoints *are* produced (33,000–114,000 per file). The failure is
that the glyphs are mapped to the **wrong** codepoints, so what comes out is
not Hindi.

### 3.1 The titles are corrupted

| File | Correct Hindi title | Extracted |
| --- | --- | --- |
| `indian_contract_act_1872_hin.pdf` | भारतीय संविदा **अधिनियम**, 1872 | भारतीय संविदा **अविवियम**, 1872 |
| `it_act_2000_hin.pdf` | सूचना प्रौद्योगिकी **अधिनियम**, 2000 | सूचना प्रौद्योगिकी **अगिगनयम**, 2000 |
| `sale_of_goods_1930_hin.pdf` | माल-विक्रय **अधिनियम**, 1930 | माल-विक्रय **अविवियम**, 1930 |
| `specific_relief_hin.pdf` | विनिर्दिष्ट अनुतोष **अधिनियम**, 1963 | विविर्दिष्ट अनुतोष **अविवियम**, 1963 |

The Hindi word for "Act" — अधिनियम — is the most frequent content word in any
Indian statute. It appears in **none** of the four files in its correct form.

### 3.2 The decisive finding: the corruption is not a substitution cipher

This is the point that rules out repair. If a legacy font mapped each glyph to
one consistently-wrong codepoint, a lookup table would reverse it. It does not.

Within **page 1 of `it_act_2000_hin.pdf` alone**, the single source character
`ध` is decoded two different ways:

| Correct | Extracted | The `ध` became |
| --- | --- | --- |
| अ**धि**नियम (Act) | अ**गि**गनयम | `ग` |
| **धा**राओं (sections) | **िा**राओं | `ि` (the i-matra) |

These two words sit 24 characters apart on the same page. The same input
character has produced two different outputs, so no character-level mapping
exists that could recover the source. The corruption is a property of the
**ligature glyphs**, not of the characters, and it is therefore not invertible.

Weakening the substitution idea further: in the other three files the same
अधिनियम comes out as अविवियम — a third decoding of the same word (`ध` → `व`).
So there are at least **two distinct font families failing in two distinct
ways**:

* **Failure A** — Contract Act 1872, Sale of Goods 1930, Specific Relief 1963:
  `ध`/`न` → `व`
* **Failure B** — IT Act 2000: `ध` → `ग` in one ligature and `ि` in another

### 3.3 Evidence from word-level presence

The cleanest test is simply whether correct Hindi legal vocabulary survives
extraction at all:

| Word | Meaning | Contract | IT Act | Sale of Goods | Specific Relief |
| --- | --- | --- | --- | --- | --- |
| अधिनियम | Act | **0** | **0** | **0** | **0** |
| धारा | section | **0** | **0** | **0** | **0** |
| अधिकार | right | **0** | **0** | **0** | **0** |
| नहीं | not | **0** | 88 | **0** | **0** |
| अनुतोष | specific relief | **0** | **0** | **0** | **0** |
| संविदा | contract | 359 | 0 | 135 | 139 |
| विक्रय | sale | 39 | 0 | 127 | 7 |

The pattern is exact. Words whose spelling happens to avoid the corrupted
ligatures (संविदा, विक्रय) survive in the hundreds. Words containing them
(धारा, अधिनियम, अधिकार) are at **zero**.

The `specific_relief_hin.pdf` row is the most damning: that Act's entire
subject matter is अनुतोष — specific relief — and the word occurs **zero
times** in a PDF devoted to it.

### 3.4 Frequency profile

The extracted IT Act consonant distribution is not a Hindi distribution. `क`
ranks first at 13.4% and `ग` third at 8.1% (over 54,919 characters), while `ध`
does not reach the top twelve at all — in a document that is one continuous
sequence of धाराएं (sections).

### 3.5 Zero-width joiner contamination

| File | U+200D (ZWJ) count | English counterpart |
| --- | --- | --- |
| `indian_contract_act_1872_hin.pdf` | 21 | 0 |
| `sale_of_goods_1930_hin.pdf` | 120 | 0 |
| `it_act_2000_hin.pdf` | 702 | 0 |
| `specific_relief_hin.pdf` | **9,562** | 0 |

The Specific Relief file carries a ZWJ between nearly every pair of words,
which would break tokenisation independently of the glyph problem.

---

## 4. Consequence for the corpus — what was done

Because Hindi `original_text` could not be sourced from these PDFs, the Hindi
rows in this corpus are **author-rendered**: written by the corpus author
directly in Hindi, on the basis of the official English text (which extracts
cleanly and is verified in §2), rather than machine-read from the Hindi PDFs.

This is recorded in the data itself, not only in this report:

* every Hindi row carries `verbatim_quote: "false"`;
* every English row carries `verbatim_quote: "true"`;
* the `source` column for Hindi rows reads
  `<statute> (author-rendered Hindi)`;
* `src/data/build_dataset.py` **hard-fails** a build in which a Hindi row
  claims `verbatim_quote: "true"`, so the distinction cannot be lost by a
  later edit.

`data/raw/sections/` therefore contains section files for the four English
instruments and **no Hindi section files**, because there is no trustworthy
Hindi section text to write.

### What this does and does not affect

**It does not affect** the meaning-preservation task as posed. Annotators
judge the Hindi sentences they are shown, and every Hindi pair was authored as
a deliberate transformation of a verified English original with a documented
`simplification_type`. The English half of the corpus is verbatim-quoted and
independently checkable against the PDFs.

**It does affect** three claims, which the paper must state plainly:

1. **Cross-lingual comparability.** The English and Hindi halves are not
   equivalent *provenance* categories — one is quoted, the other rendered. A
   difference in model performance between the two languages therefore mixes a
   language effect with a provenance effect, and Experiment 2 must say so
   rather than reporting a clean cross-lingual comparison.
2. **`verbatim_quote: "false"` is not a quality judgement.** It records how the
   text was obtained, not whether it is correct Hindi.
3. **This is a reproducibility limit.** A reader cannot regenerate the Hindi
   rows from the PDFs in `data/raw/sources/`. They are reproducible from the
   authored JSONL files, which are committed, but the ultimate source of the
   Hindi wording is the author.

---

## 5. Confirmed non-issues

Two candidate problems were investigated and found **not** to be present:

* **English PDFs** — all four extract cleanly, titles verified (§2). No
  Devanagari, no ZWJ, no replacement characters.
* **No replacement characters anywhere.** All eight files extract with
  `U+FFFD` count zero, so nothing has been silently dropped by a failed decode;
  the Hindi problem is mis-mapping, not loss.

---

## 6. How to reproduce this report

```bash
# extract text from every source PDF and re-run the checks above
.venv/bin/python - <<'EOF'
from pypdf import PdfReader
from pathlib import Path
for p in sorted(Path("data/raw/sources").glob("*.pdf")):
    r = PdfReader(str(p))
    txt = "".join((pg.extract_text() or "") for pg in r.pages)
    print(f"{p.name:38s} ZWJ={txt.count(chr(0x200d)):6d} "
          f"U+FFFD={txt.count(chr(0xFFFD)):4d} "
          f"has_अधिनियम={txt.count('अधिनियम')}")
EOF
```

Expected: every English file reports `ZWJ=0`, `U+FFFD=0`, `has_अधिनियम=0`
(the last trivially, being English); every Hindi file reports a non-zero ZWJ
count and `has_अधिनियम=0` despite being a Hindi statute.
