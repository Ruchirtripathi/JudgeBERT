"""Extract and segment statutory source text from the raw source documents.

This is the first stage of the data pipeline:

    data/raw/sources/*.pdf   --extract-->   data/raw/sections/*.jsonl
                                             |  one record per statutory
                                             |  section, with page-level
                                             |  provenance

The output is an *intermediate* artefact.  It is not the dataset: it is the
pool of authentic legal sentences from which the corpus curator selects
originals and writes simplifications.

Source-encoding caveat
----------------------
The English source PDF extracts cleanly.  The official Hindi PDF for the same
Act does **not**: it was typeset with a legacy non-Unicode Devanagari font,
and glyph-to-Unicode mapping collapses distinct ligatures onto the same
codepoint (``अधिनियम`` extracts as ``अविवियम``; both ``धि`` and ``नि`` map to
``वि``).  Text recovered from it is therefore *not* usable as legal text and
this module refuses to emit it by default.  See
``reports/source_provenance.md`` and the Limitations section of the paper.

Usage
-----
::

    python -m src.data.extract_sources --source contract_act_1872_eng
    python -m src.data.extract_sources --list
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------
#: Every external document the corpus draws on.  ``encoding_ok`` records
#: whether automated text extraction from this file yields trustworthy text.
#: A source with ``encoding_ok=False`` may be cited for provenance but its
#: extracted text must not be fed into the corpus.
SOURCE_REGISTRY: dict[str, dict] = {
    "contract_act_1872_eng": {
        "path": "data/raw/sources/indian_contract_act_1872_eng.pdf",
        "title": "The Indian Contract Act, 1872",
        "act_number": "Act No. 9 of 1872",
        "language": "en",
        "legal_domain": "contract",
        "source_name": "Indian Contract Act 1872",
        "retrieved_from": (
            "https://huggingface.co/datasets/d-riti/"
            "Dataset-For-Indian-legal-knowledge-base"
        ),
        "licence": "Apache-2.0 (dataset packaging); the Act itself is a "
                   "Government of India work, not subject to copyright under "
                   "s.52(1)(q) of the Indian Copyright Act, 1957.",
        "encoding_ok": True,
        "notes": "Official English text. Extracts cleanly with pypdf.",
    },
    "contract_act_1872_hin": {
        "path": "data/raw/sources/indian_contract_act_1872_hin.pdf",
        "title": "भारतीय संविदा अधिनियम, 1872",
        "act_number": "1872 का अधिनियम संख्या 9",
        "language": "hi",
        "legal_domain": "contract",
        "source_name": "Indian Contract Act 1872 (official Hindi version)",
        "retrieved_from": (
            "https://huggingface.co/datasets/d-riti/"
            "Dataset-For-Indian-legal-knowledge-base"
        ),
        "licence": "Apache-2.0 (dataset packaging); Government of India work.",
        "encoding_ok": False,
        "notes": (
            "Legacy non-Unicode Devanagari font. Glyph mapping is lossy: "
            "distinct ligatures collapse onto one codepoint, e.g. 'अधिनियम' "
            "extracts as 'अविवियम'. Text extraction DISABLED for this source; "
            "the document is retained as provenance evidence only."
        ),
    },
    "it_act_2000_eng": {
        "path": "data/raw/sources/it_act_2000_eng.pdf",
        "title": "The Information Technology Act, 2000",
        "act_number": "Act No. 21 of 2000",
        "language": "en",
        "legal_domain": "cyber",
        "source_name": "Information Technology Act 2000",
        "retrieved_from": (
            "https://huggingface.co/datasets/d-riti/"
            "Dataset-For-Indian-legal-knowledge-base"
        ),
        "licence": "Apache-2.0 (dataset packaging); Government of India work, "
                   "not subject to copyright under s.52(1)(q) of the Indian "
                   "Copyright Act, 1957.",
        "encoding_ok": True,
        "notes": "Official English text. Extracts cleanly with pypdf.",
    },
    "it_act_2000_hin": {
        "path": "data/raw/sources/it_act_2000_hin.pdf",
        "title": "सूचना प्रौद्योगिकी अधिनियम, 2000",
        "act_number": "2000 का अधिनियम संख्या 21",
        "language": "hi",
        "legal_domain": "cyber",
        "source_name": "Information Technology Act 2000 (official Hindi version)",
        "retrieved_from": (
            "https://huggingface.co/datasets/d-riti/"
            "Dataset-For-Indian-legal-knowledge-base"
        ),
        "licence": "Apache-2.0 (dataset packaging); Government of India work.",
        "encoding_ok": False,
        "notes": (
            "Legacy non-Unicode Devanagari font, corrupting by a DIFFERENT "
            "mapping than the Contract Act PDF: 'ध' maps to 'ि' and 'वि' maps "
            "to 'गि', so 'अधिनियम' extracts as 'अगिगनयम', 'विधि' as 'विवि', "
            "'धाराएं' as 'िाराएं', 'लागू' as 'लािू'. Two official Hindi "
            "statutes corrupting in two different ways is the evidence that "
            "this is systemic to the Government's Devanagari typesetting, "
            "not a one-off bad file. Extraction DISABLED."
        ),
    },
    "sale_of_goods_1930_eng": {
        "path": "data/raw/sources/sale_of_goods_1930_eng.pdf",
        "title": "The Sale of Goods Act, 1930",
        "act_number": "Act No. 3 of 1930",
        "language": "en",
        "legal_domain": "sale_of_goods",
        "source_name": "Sale of Goods Act 1930",
        "retrieved_from": (
            "https://huggingface.co/datasets/d-riti/"
            "Dataset-For-Indian-legal-knowledge-base"
        ),
        "licence": "Apache-2.0 (dataset packaging); Government of India work.",
        "encoding_ok": True,
        "notes": "Official English text. Extracts cleanly with pypdf.",
    },
    "sale_of_goods_1930_hin": {
        "path": "data/raw/sources/sale_of_goods_1930_hin.pdf",
        "title": "माल-विक्रय अधिनियम, 1930",
        "act_number": "1930 का अधिनियम संख्या 3",
        "language": "hi",
        "legal_domain": "sale_of_goods",
        "source_name": "Sale of Goods Act 1930 (official Hindi version)",
        "retrieved_from": (
            "https://huggingface.co/datasets/d-riti/"
            "Dataset-For-Indian-legal-knowledge-base"
        ),
        "licence": "Apache-2.0 (dataset packaging); Government of India work.",
        "encoding_ok": False,
        "notes": (
            "Legacy non-Unicode Devanagari font; corrupts identically to the "
            "Contract Act PDF ('अधिनियम' -> 'अविवियम'). Extraction DISABLED."
        ),
    },
    "specific_relief_1963_eng": {
        "path": "data/raw/sources/specific_relief_eng.pdf",
        "title": "The Specific Relief Act, 1963",
        "act_number": "Act No. 47 of 1963",
        "language": "en",
        "legal_domain": "specific_relief",
        "source_name": "Specific Relief Act 1963",
        "retrieved_from": (
            "https://huggingface.co/datasets/d-riti/"
            "Dataset-For-Indian-legal-knowledge-base"
        ),
        "licence": "Apache-2.0 (dataset packaging); Government of India work.",
        "encoding_ok": True,
        "notes": (
            "Official English text, extracts cleanly. Registered for "
            "provenance; not currently drawn on by any authored batch."
        ),
    },
    "specific_relief_1963_hin": {
        "path": "data/raw/sources/specific_relief_hin.pdf",
        "title": "विनिर्दिष्ट अनुतोष अधिनियम, 1963",
        "act_number": "1963 का अधिनियम संख्या 47",
        "language": "hi",
        "legal_domain": "specific_relief",
        "source_name": "Specific Relief Act 1963 (official Hindi version)",
        "retrieved_from": (
            "https://huggingface.co/datasets/d-riti/"
            "Dataset-For-Indian-legal-knowledge-base"
        ),
        "licence": "Apache-2.0 (dataset packaging); Government of India work.",
        "encoding_ok": False,
        "notes": (
            "Legacy non-Unicode Devanagari font. Extraction DISABLED. "
            "Registered for provenance."
        ),
    },
}

#: A section header looks like ``10. What agreements are contracts.—All ...``
#: The em-dash separator is what distinguishes a real section body from an
#: entry in the table of contents, which lists the same numbering.
_SECTION_START_RE = re.compile(
    r"(?m)^\s*(?P<num>\d{1,3})\.\s+(?P<title>[^\n—]{3,120}?)\s*[—\-–]{1,2}\s*"
)

_DEVA_RE = re.compile(r"[\u0900-\u097F]")

#: Known-bad extraction fingerprints for the legacy-font Hindi PDFs.  Two
#: official Hindi statutes from the same publisher corrupt by *different*
#: glyph mappings, so the guard cannot rely on a single signature.  Used as a
#: runtime check so that a future re-download cannot silently reintroduce
#: corrupted text into the corpus.
_CORRUPTION_MARKERS = (
    "अविवियम",  # अधिनियम, Contract Act / Sale of Goods Act mapping
    "कवतपय",    # कर्तव्य, same mapping
    "अगिगनयम",  # अधिनियम, IT Act mapping
    "विवि",      # विधि, IT Act mapping
    "िाराएं",   # धाराएं, IT Act mapping
)

#: Words a genuine Hindi statute text will contain with correct spelling.
#: Corruption of the kind observed here mangles these first, so their total
#: absence is a reliable signal that the font mapping is broken.
_HINDI_PROBE_WORDS = ("अधिनियम", "धारा", "विधि", "अधिकार", "व्यक्ति")


@dataclass
class Section:
    """One statutory section recovered from a source document."""

    source_key: str
    source_name: str
    language: str
    legal_domain: str
    section_number: int
    section_title: str
    text: str
    page_numbers: list[int] = field(default_factory=list)
    verbatim: bool = True
    encoding_verified: bool = True

    @property
    def citation(self) -> str:
        return f"{self.source_name}, s.{self.section_number}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citation"] = self.citation
        return d


class SourceEncodingError(RuntimeError):
    """Raised when extracted text is known to be unreliable."""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def read_pdf_pages(path: str | Path) -> list[str]:
    """Return the extracted text of each page, in order."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pypdf is required for source extraction. Install it with:\n"
            "    pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


def _devanagari_ratio(text: str) -> float:
    """Fraction of non-space characters that are Devanagari."""
    meaningful = [c for c in text if not c.isspace()]
    if not meaningful:
        return 0.0
    return len(_DEVA_RE.findall(text)) / len(meaningful)


def check_encoding(page_texts: list[str], source: dict) -> None:
    """Refuse to emit text from a source known to extract unreliably.

    Two independent checks:
      1. The registry flag ``encoding_ok``.
      2. A live scan for known corruption fingerprints, in case the file was
         swapped for a differently-encoded copy.
    """
    if not source.get("encoding_ok", True):
        raise SourceEncodingError(
            f"Source '{source['title']}' is flagged encoding_ok=False.\n"
            f"Reason: {source.get('notes', 'no reason recorded')}\n"
            f"Extraction refused — using this text would put corrupted legal "
            f"language into the corpus."
        )

    blob = "\n".join(page_texts)
    if source.get("language") == "hi":
        for marker in _CORRUPTION_MARKERS:
            if marker in blob:
                raise SourceEncodingError(
                    f"Corruption fingerprint {marker!r} found in "
                    f"'{source['title']}'. Extraction refused."
                )
        if _devanagari_ratio(blob) < 0.3:
            raise SourceEncodingError(
                f"'{source['title']}' was expected to be Devanagari but only "
                f"{_devanagari_ratio(blob):.1%} of characters are. "
                f"The file is probably not the document it claims to be."
            )
        # A correct Hindi statute will use at least one of these words with
        # correct spelling.  If none appear, the font mapping has mangled them
        # all — the exact failure observed in both official Hindi PDFs.
        present = [w for w in _HINDI_PROBE_WORDS if w in blob]
        if not present:
            raise SourceEncodingError(
                f"None of the probe words {list(_HINDI_PROBE_WORDS)} appear "
                f"correctly spelled in '{source['title']}'. A genuine Hindi "
                f"statute would use several. The font mapping is broken; "
                f"extraction refused."
            )


def _char_page_index(page_texts: list[str]) -> tuple[str, list[int]]:
    """Concatenate pages, returning the blob and a char-offset -> page map."""
    blob_parts: list[str] = []
    offsets: list[int] = []
    cursor = 0
    for page in page_texts:
        offsets.append(cursor)
        blob_parts.append(page)
        cursor += len(page) + 1  # +1 for the newline we join with
    return "\n".join(blob_parts), offsets


def _page_for_offset(offset: int, offsets: list[int]) -> int:
    """1-based page number containing a character offset."""
    page = 1
    for i, start in enumerate(offsets):
        if offset >= start:
            page = i + 1
        else:
            break
    return page


def segment_sections(
    page_texts: list[str], source: dict, min_body_chars: int = 40
) -> list[Section]:
    """Split a statute into sections using the ``N. Title.—body`` pattern.

    The table of contents reuses the same numbering, so matches with a very
    short or missing body are discarded.  Repeated section numbers keep the
    *longest* body, which is the real section rather than a TOC entry.
    """
    blob, offsets = _char_page_index(page_texts)
    blob = re.sub(r"[ \t]+", " ", blob)
    # Repair words split across a line break by PDF extraction ("sha ll").
    blob = re.sub(r"\b([a-z]{2,})\s+(ll|ing|ed|tion|ment)\b", r"\1\2", blob)

    matches = list(_SECTION_START_RE.finditer(blob))
    logger.info("Found %d candidate section headers", len(matches))

    by_number: dict[int, Section] = {}
    for i, match in enumerate(matches):
        number = int(match.group("num"))
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(blob)
        body = blob[body_start:body_end].strip()

        if len(body) < min_body_chars:
            continue  # almost certainly a table-of-contents line

        title = match.group("title").strip().rstrip(".")
        page_start = _page_for_offset(match.start(), offsets)
        page_end = _page_for_offset(body_end, offsets)
        pages = list(range(page_start, page_end + 1))

        existing = by_number.get(number)
        if existing is not None and len(existing.text) >= len(body):
            continue

        by_number[number] = Section(
            source_key=source["key"],
            source_name=source["source_name"],
            language=source["language"],
            legal_domain=source["legal_domain"],
            section_number=number,
            section_title=title,
            text=re.sub(r"\s+", " ", body).strip(),
            page_numbers=pages,
            verbatim=True,
            encoding_verified=True,
        )

    return [by_number[k] for k in sorted(by_number)]


def extract_source(
    source_key: str,
    output_dir: Path,
    force: bool = False,
) -> list[Section]:
    """Extract one registered source into ``output_dir/<key>.jsonl``."""
    if source_key not in SOURCE_REGISTRY:
        raise KeyError(
            f"Unknown source {source_key!r}. Known: {sorted(SOURCE_REGISTRY)}"
        )

    source = dict(SOURCE_REGISTRY[source_key])
    source["key"] = source_key
    pdf_path = Path(source["path"])
    if not pdf_path.is_absolute():
        pdf_path = Path(__file__).resolve().parents[2] / pdf_path

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Source file missing: {pdf_path}\n"
            f"See data/raw/README.md for how to obtain it."
        )

    page_texts = read_pdf_pages(pdf_path)
    logger.info("Read %d pages from %s", len(page_texts), pdf_path.name)

    # Raises for sources we know we cannot trust.
    check_encoding(page_texts, source)

    sections = segment_sections(page_texts, source)
    logger.info("Segmented %d sections", len(sections))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{source_key}.jsonl"
    if out_path.exists() and not force:
        logger.warning("%s exists; use --force to overwrite", out_path)
        return sections

    with out_path.open("w", encoding="utf-8") as fh:
        meta = {k: v for k, v in source.items() if k != "key"}
        fh.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for section in sections:
            fh.write(json.dumps(section.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Wrote %s", out_path)

    return sections


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract statutory sections from raw source documents."
    )
    parser.add_argument("--source", action="append", default=None,
                        help="Source key to extract (repeatable). Default: all "
                             "sources with encoding_ok=True.")
    parser.add_argument("--out", default="data/raw/sections",
                        help="Output directory for section JSONL files")
    parser.add_argument("--list", action="store_true", help="List registered sources")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args(argv)

    setup_logging("INFO")

    if args.list:
        for key, meta in SOURCE_REGISTRY.items():
            flag = "OK      " if meta.get("encoding_ok") else "UNUSABLE"
            print(f"{flag} {key:26s} {meta['title']}")
            print(f"         {meta['path']}")
        return 0

    keys = args.source or [
        k for k, v in SOURCE_REGISTRY.items() if v.get("encoding_ok")
    ]
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parents[2] / out_dir

    failures = 0
    for key in keys:
        try:
            sections = extract_source(key, out_dir, force=args.force)
            print(f"{key}: {len(sections)} sections -> {out_dir / (key + '.jsonl')}")
        except SourceEncodingError as exc:
            failures += 1
            logger.error("Refused to extract %s:\n%s", key, exc)
        except (FileNotFoundError, KeyError) as exc:
            failures += 1
            logger.error("%s", exc)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
