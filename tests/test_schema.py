"""Tests for the dataset schema, and for the gold/non-gold gate.

The gate is the project's central integrity mechanism: a machine-drafted score
must never be usable as a human label. These tests exist so that a future
refactor of :mod:`src.data.schema` cannot quietly weaken it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import schema
from src.utils.io import read_csv, write_csv


# ---------------------------------------------------------------------------
# Vocabulary and structure
# ---------------------------------------------------------------------------
def test_schema_has_no_duplicate_columns():
    names = [c.name for c in schema.SCHEMA]
    assert len(names) == len(set(names))


def test_declared_columns_are_unique_and_partitioned():
    assert set(schema.REQUIRED_COLUMNS).isdisjoint(schema.OPTIONAL_COLUMNS)
    assert set(schema.REQUIRED_COLUMNS) | set(schema.OPTIONAL_COLUMNS) == set(
        schema.ALL_COLUMNS
    )


def test_gold_and_non_gold_statuses_are_disjoint_and_known():
    assert schema.GOLD_STATUSES.isdisjoint(schema.NON_GOLD_STATUSES)
    assert schema.GOLD_STATUSES | schema.NON_GOLD_STATUSES == set(
        schema.ANNOTATION_STATUSES
    )


def test_every_column_has_a_description():
    for column in schema.SCHEMA:
        assert column.description.strip(), f"{column.name} has no description"


# ---------------------------------------------------------------------------
# empty_frame / ensure_columns
# ---------------------------------------------------------------------------
def test_empty_frame_is_empty_but_well_formed():
    df = schema.empty_frame()
    assert len(df) == 0
    assert list(df.columns) == schema.ALL_COLUMNS


def test_ensure_columns_adds_missing_optionals_and_preserves_extras():
    df = pd.DataFrame({"pair_id": ["X"], "mystery_column": [1]})
    out = schema.ensure_columns(df)
    for name in schema.OPTIONAL_COLUMNS:
        assert name in out.columns
    # An unrecognised column is kept, not silently dropped, so validation can
    # report the typo rather than swallowing it.
    assert "mystery_column" in out.columns


# ---------------------------------------------------------------------------
# coerce_types
# ---------------------------------------------------------------------------
def test_coerce_types_scores_become_float_with_empty_becoming_nan():
    df = pd.DataFrame(
        {"human_lmp_score": ["7", "", "3.5"], "draft_lmp_score": ["1", "10", ""]}
    )
    out = schema.coerce_types(df)
    assert out["human_lmp_score"].dtype == "float64"
    # An empty cell is missing, NOT a score of zero.
    assert out["human_lmp_score"].isna().tolist() == [False, True, False]
    assert out["human_lmp_score"].iloc[2] == 3.5


def test_coerce_types_does_not_invent_error_type_for_unannotated_rows():
    """The bug this test pins: blank error_type must not become 'None'.

    'None' is a claim that a human looked and found no legal error. Making that
    claim on an unannotated row fabricates a judgement.
    """
    df = pd.DataFrame(
        {
            "error_type": [None, None, None],
            "annotation_status": ["draft", "unannotated", "draft"],
        }
    )
    out = schema.coerce_types(df)
    assert out["error_type"].isna().all()


def test_coerce_types_fills_none_for_annotated_blank_error_type():
    df = pd.DataFrame(
        {
            "error_type": [None, "Omission"],
            "annotation_status": ["human_single", "human_single"],
        }
    )
    out = schema.coerce_types(df)
    assert out["error_type"].tolist() == ["None", "Omission"]


def test_coerce_types_normalises_verbatim_quote_booleans_to_strings():
    """A column read back from CSV as all-true/all-false becomes bool.

    Left alone it would give the corpus two spellings of the same value
    depending on whether it came from CSV or from the authored JSONL.
    """
    df = pd.DataFrame({"verbatim_quote": [True, False, np.True_]})
    out = schema.coerce_types(df)
    assert out["verbatim_quote"].tolist() == ["true", "false", "true"]


def test_coerce_types_strips_text_whitespace():
    df = pd.DataFrame({"original_text": ["  spaced  ", "fine"]})
    out = schema.coerce_types(df)
    assert out["original_text"].tolist() == ["spaced", "fine"]


def test_coerce_types_defaults_missing_annotation_status():
    df = pd.DataFrame({"pair_id": ["X"], "annotation_status": [None]})
    out = schema.coerce_types(df)
    assert out["annotation_status"].iloc[0] == "unannotated"


# ---------------------------------------------------------------------------
# The gold gate
# ---------------------------------------------------------------------------
def test_is_gold_requires_human_status(master_frame):
    assert not schema.is_gold(master_frame).any()


def test_is_gold_true_for_human_status_with_a_score(master_frame):
    df = master_frame.copy()
    df.loc[0, "annotation_status"] = "human_single"
    df.loc[0, "human_lmp_score"] = 7.0
    mask = schema.is_gold(df)
    assert mask.iloc[0]
    assert mask.sum() == 1


def test_is_gold_false_for_human_status_without_a_score(master_frame):
    """A row claiming human_single but carrying no score is not usable truth."""
    df = master_frame.copy()
    df.loc[0, "annotation_status"] = "human_single"
    df.loc[0, "human_lmp_score"] = np.nan
    assert not schema.is_gold(df).any()


def test_gold_subset_returns_only_gold_rows(master_frame):
    df = master_frame.copy()
    df.loc[0, "annotation_status"] = "human_consensus"
    df.loc[0, "human_lmp_score"] = 9.0
    gold = schema.gold_subset(df)
    assert len(gold) == 1
    assert gold["pair_id"].iloc[0] == "ENG_CON_001"


def test_draft_status_is_never_gold(master_frame):
    """The specific value 'draft' means 'a suggestion exists', never truth."""
    df = master_frame.copy()
    df["human_lmp_score"] = 5.0  # even if scores were somehow present
    assert not schema.is_gold(df).any()


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "score,expected",
    [(1, "low"), (3, "low"), (4, "mid"), (7, "mid"), (8, "high"), (10, "high")],
)
def test_score_bucket_boundaries(score, expected):
    assert schema.score_bucket(score) == expected


def test_score_bucket_handles_missing_values():
    for missing in (None, np.nan, pd.NA):
        assert schema.score_bucket(missing) is None


@pytest.mark.parametrize(
    "score,expected",
    [
        (10, "Preserved"),
        (8, "Preserved"),
        (7.9, "Partially Preserved"),
        (5, "Partially Preserved"),
        (4.9, "Not Preserved"),
        (1, "Not Preserved"),
    ],
)
def test_score_to_decision_default_thresholds(score, expected):
    assert schema.score_to_decision(score) == expected


def test_score_to_decision_respects_custom_thresholds():
    assert schema.score_to_decision(6, preserved_min=6.0, partial_min=3.0) == "Preserved"


def test_score_to_decision_returns_none_when_unscored():
    assert schema.score_to_decision(np.nan) is None


def test_validate_vocabulary_reports_only_unknown_values():
    unknown = schema.validate_vocabulary(["en", "hi", "fr"], schema.LANGUAGES.keys())
    assert unknown == ["fr"]


# ---------------------------------------------------------------------------
# CSV round-trip
# ---------------------------------------------------------------------------
def test_csv_round_trip_preserves_the_none_error_label(tmp_path, master_frame):
    """'None' is a legitimate category meaning 'no error', not a missing value.

    Pandas' default NA list includes the literal string "None", so without the
    explicit settings in src.utils.io.read_csv this value is silently destroyed
    and 'no error' becomes indistinguishable from 'not yet judged'.
    """
    df = master_frame.copy()
    df["error_type"] = "None"
    df["annotation_status"] = "human_single"
    df["human_lmp_score"] = 8.0

    path = tmp_path / "round_trip.csv"
    write_csv(df, path)
    back = schema.coerce_types(schema.ensure_columns(read_csv(path)))

    assert back["error_type"].tolist() == ["None"] * len(df)
    assert back["error_type"].notna().all()


def test_csv_round_trip_keeps_empty_scores_missing(tmp_path, master_frame):
    path = tmp_path / "empty_scores.csv"
    write_csv(master_frame, path)
    back = schema.coerce_types(schema.ensure_columns(read_csv(path)))
    assert back["human_lmp_score"].isna().all()


def test_csv_round_trip_is_stable_for_verbatim_quote(tmp_path, master_frame):
    path = tmp_path / "verbatim.csv"
    write_csv(master_frame, path)
    back = schema.coerce_types(schema.ensure_columns(read_csv(path)))
    assert set(back["verbatim_quote"]) == {"true", "false"}
