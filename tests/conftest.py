"""Shared pytest fixtures.

The fixtures here are deliberately small and synthetic where possible. A test
that depends on the 340-row corpus breaks the day someone edits an authored
batch; a test that builds the three rows it needs keeps testing the property it
was written for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import schema  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def master_frame() -> pd.DataFrame:
    """A small, well-formed master frame with correct dtypes and no gold labels.

    Four pairs over two languages and two domains, with sibling reformulations
    sharing ``source_group`` values so that grouping behaviour is testable.
    """
    rows = [
        # two siblings of the same provision -> must never split apart
        ("ENG_CON_001", "en", "contract", "ICA1872::1", "train", 9),
        ("ENG_CON_002", "en", "contract", "ICA1872::1", "train", 3),
        # both languages of one provision
        ("ENG_CON_003", "en", "contract", "ICA1872::2", "train", 5),
        ("HIN_CON_003", "hi", "contract", "ICA1872::2", "train", 5),
        ("ENG_CYB_001", "en", "cyber", "ITA2000::43", "validation", 8),
        ("HIN_CYB_001", "hi", "cyber", "ITA2000::43", "validation", 8),
    ]
    df = pd.DataFrame(
        {
            "pair_id": [r[0] for r in rows],
            "language": [r[1] for r in rows],
            "legal_domain": [r[2] for r in rows],
            "source_group": [r[3] for r in rows],
            "split": [r[4] for r in rows],
            "draft_lmp_score": [r[5] for r in rows],
            "original_text": [f"Original text for {r[0]}." for r in rows],
            "simplified_text": [f"Simplified text for {r[0]}." for r in rows],
            "source": [
                "Indian Contract Act 1872" if r[1] == "en" else "Indian Contract Act 1872 (author-rendered Hindi)"
                for r in rows
            ],
            "source_section": ["1", "1", "2", "2", "43", "43"],
            "simplification_type": ["faithful_paraphrase"] * 6,
            "draft_error_type": ["None", "Omission", "Scope", "Scope", "None", "None"],
            "verbatim_quote": ["true" if r[1] == "en" else "false" for r in rows],
            "annotation_status": ["draft"] * 6,
            "human_lmp_score": [pd.NA] * 6,
            "error_type": [pd.NA] * 6,
            "decision": [pd.NA] * 6,
            "annotator_id": [pd.NA] * 6,
            "annotated_at": [pd.NA] * 6,
            "notes": [""] * 6,
            "context": [""] * 6,
            "source_url": ["https://example.invalid"] * 6,
        }
    )
    return schema.coerce_types(schema.ensure_columns(df))


@pytest.fixture
def annotation_dir(tmp_path: Path) -> Path:
    """An empty directory to hold per-annotator CSV files."""
    directory = tmp_path / "annotators"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_annotations(directory: Path, annotator_id: str, scores: dict[str, int],
                       error_type: str = "None") -> Path:
    """Write one annotator's file, in the format the annotation tool produces."""
    frame = pd.DataFrame(
        [
            {
                "pair_id": pair_id,
                "human_lmp_score": score,
                "error_type": error_type,
                "notes": f"{annotator_id} note for {pair_id}",
                "annotator_id": annotator_id,
                "annotated_at": "2026-01-01T00:00:00Z",
            }
            for pair_id, score in scores.items()
        ]
    )
    path = directory / f"{annotator_id}.csv"
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


@pytest.fixture
def write_annotations():
    """The annotation-file writer, exposed as a fixture.

    Exposed as a fixture rather than imported from this module so that the
    suite does not depend on ``tests`` being importable as a package, which
    fails under ``--import-mode=importlib``.
    """
    return _write_annotations
