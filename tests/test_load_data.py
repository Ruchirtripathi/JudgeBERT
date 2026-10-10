"""Tests for corpus loading, derived columns and the gold gate.

The gold gate is the project's central integrity mechanism, so the tests that
matter most here are the ones asserting that un-annotated data cannot reach an
experiment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import load_data, schema
from src.utils.io import write_csv


@pytest.fixture
def master_csv(tmp_path, master_frame):
    """The fixture frame written to disk, as the real pipeline would have it."""
    path = tmp_path / "lmp_pairs.csv"
    write_csv(master_frame, path)
    return path


# ---------------------------------------------------------------------------
# load_master
# ---------------------------------------------------------------------------
def test_load_master_reads_the_file_and_returns_every_row(master_csv, master_frame):
    loaded = load_data.load_master(path=master_csv)
    assert len(loaded) == len(master_frame)
    assert "pair_id" in loaded.columns


def test_load_master_returns_a_copy_the_caller_can_mutate(master_csv):
    first = load_data.load_master(path=master_csv)
    first.loc[0, "pair_id"] = "MUTATED"
    second = load_data.load_master(path=master_csv)
    assert second["pair_id"].iloc[0] != "MUTATED"


def test_load_master_rejects_an_unknown_subset(master_csv):
    with pytest.raises(ValueError, match="Unknown subset"):
        load_data.load_master(path=master_csv, subset="everything")


def test_load_master_gold_subset_is_empty_before_annotation(master_csv):
    """The current corpus state: no human labels exist, so gold is empty."""
    assert load_data.load_master(path=master_csv, subset="gold").empty


def test_load_master_draft_subset_returns_the_draft_rows(master_csv, master_frame):
    loaded = load_data.load_master(path=master_csv, subset="draft")
    assert len(loaded) == len(master_frame)
    assert (loaded["annotation_status"] == "draft").all()


def test_load_master_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data.load_master(path=tmp_path / "absent.csv")


def test_master_path_points_at_the_processed_directory(repo_root):
    cfg = load_data.load_config()
    assert load_data.master_path(cfg).name == "lmp_pairs.csv"


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------
def test_decision_is_derived_from_the_human_score_when_present(master_frame):
    df = master_frame.copy()
    df.loc[0, "human_lmp_score"] = 9.0
    df.loc[1, "human_lmp_score"] = 2.0
    out = load_data.add_derived_columns(df)
    assert out.loc[0, "decision"] == "Preserved"
    assert out.loc[1, "decision"] == "Not Preserved"


def test_decision_stays_empty_when_no_human_score_exists(master_frame):
    """An unannotated row must not acquire a decision.

    A decision on a row nobody has judged is a fabricated legal finding, and
    it would also let ``assert_gold``'s row count look healthier than it is.
    """
    out = load_data.add_derived_columns(master_frame)
    assert out["decision"].isna().all()


def test_a_human_decision_is_never_overwritten_by_the_derived_one(master_frame):
    """A human adjudicator's decision outranks the threshold arithmetic."""
    df = master_frame.copy()
    df.loc[0, "human_lmp_score"] = 9.0
    df.loc[0, "decision"] = "Not Preserved"  # deliberate human override
    out = load_data.add_derived_columns(df)
    assert out.loc[0, "decision"] == "Not Preserved"


def test_score_bucket_falls_back_to_the_draft_score_when_unannotated(master_frame):
    """Buckets are needed to stratify the annotation batches, pre-annotation."""
    out = load_data.add_derived_columns(master_frame)
    assert out["score_bucket"].notna().all()
    assert out.loc[master_frame["draft_lmp_score"] == 9, "score_bucket"].iloc[0] == "high"


def test_source_group_is_left_alone_when_already_present(master_frame):
    out = load_data.add_derived_columns(master_frame)
    assert out["source_group"].tolist() == master_frame["source_group"].tolist()


def test_missing_source_group_is_derived_from_source_and_section():
    df = pd.DataFrame(
        {
            "pair_id": ["X1", "X2"],
            "source": ["Indian Contract Act 1872", "Indian Contract Act 1872"],
            "source_section": ["10", "10"],
            "human_lmp_score": [np.nan, np.nan],
            "draft_lmp_score": [5, 5],
        }
    )
    out = load_data.add_derived_columns(df)
    # Two rows of the same provision must land in the same group, or the
    # anti-leakage guarantee silently stops holding for derived corpora.
    assert out["source_group"].nunique() == 1


def test_group_without_a_section_number_falls_back_to_the_row_id():
    """Conservative: never merge two provisions that cannot be shown to match."""
    df = pd.DataFrame(
        {
            "pair_id": ["X1", "X2"],
            "source": ["Some Act", "Some Act"],
            "source_section": ["", ""],
            "human_lmp_score": [np.nan, np.nan],
        }
    )
    out = load_data.add_derived_columns(df)
    assert out["source_group"].nunique() == 2


# ---------------------------------------------------------------------------
# The gold gate
# ---------------------------------------------------------------------------
def test_assert_gold_raises_on_an_empty_frame():
    with pytest.raises(ValueError, match="TODO_HUMAN_ANNOTATION"):
        load_data.assert_gold(pd.DataFrame(), "Experiment 1")


def test_assert_gold_raises_on_draft_rows(master_frame):
    with pytest.raises(ValueError, match="not human-annotated"):
        load_data.assert_gold(master_frame, "Experiment 1")


def test_assert_gold_names_the_offending_statuses(master_frame):
    """The message must say *why*, not just that it refused."""
    with pytest.raises(ValueError) as excinfo:
        load_data.assert_gold(master_frame, "Experiment 3")
    assert "draft" in str(excinfo.value)
    assert "Experiment 3" in str(excinfo.value)


def test_assert_gold_passes_for_fully_annotated_rows(master_frame):
    df = master_frame.copy()
    df["annotation_status"] = "human_consensus"
    df["human_lmp_score"] = 7.0
    load_data.assert_gold(df, "Experiment 1")  # must not raise


def test_assert_gold_rejects_a_partially_annotated_frame(master_frame):
    """One draft row among annotated ones is enough to invalidate the run."""
    df = master_frame.copy()
    df["annotation_status"] = "human_consensus"
    df["human_lmp_score"] = 7.0
    df.loc[0, "annotation_status"] = "draft"
    with pytest.raises(ValueError):
        load_data.assert_gold(df, "Experiment 1")


def test_assert_gold_accepts_every_gold_status(master_frame):
    for status in sorted(schema.GOLD_STATUSES):
        df = master_frame.copy()
        df["annotation_status"] = status
        df["human_lmp_score"] = 6.0
        load_data.assert_gold(df, "Experiment 1")


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------
def test_filter_rows_applies_every_supplied_filter(master_frame):
    out = load_data.filter_rows(master_frame, languages=["en"], domains=["contract"])
    assert set(out["language"]) == {"en"}
    assert set(out["legal_domain"]) == {"contract"}


def test_filter_rows_with_no_arguments_returns_everything(master_frame):
    assert len(load_data.filter_rows(master_frame)) == len(master_frame)


def test_split_frames_returns_all_three_splits(master_frame):
    frames = load_data.split_frames(master_frame)
    assert set(frames) == set(schema.SPLITS)
    assert sum(len(f) for f in frames.values()) == len(master_frame)


def test_summary_counts_rows_languages_and_gold(master_frame):
    summary = load_data.summary(master_frame)
    assert summary["n_rows"] == len(master_frame)
    assert summary["n_gold"] == 0  # nothing is annotated yet
    assert summary["languages"] == {"en": 4, "hi": 2}


# ---------------------------------------------------------------------------
# Model input projection
# ---------------------------------------------------------------------------
def test_model_input_frame_renames_the_label_column(master_frame):
    out = load_data.model_input_frame(master_frame)
    assert "label" in out.columns
    assert "human_lmp_score" not in out.columns


def test_model_input_frame_excludes_the_draft_score():
    """The draft must not be readable as a feature by any model.

    ``draft_lmp_score`` is the author's guess at the target. Leaving it in the
    model input would give the model a near-copy of the label it is supposed to
    predict, and every metric would be meaningless.
    """
    df = pd.DataFrame(
        {
            "pair_id": ["X"], "language": ["en"], "legal_domain": ["contract"],
            "split": ["train"], "original_text": ["a"], "simplified_text": ["b"],
            "human_lmp_score": [np.nan], "draft_lmp_score": [7],
            "draft_error_type": ["Omission"],
        }
    )
    out = load_data.model_input_frame(df)
    assert "draft_lmp_score" not in out.columns
    assert "draft_error_type" not in out.columns


def test_model_input_frame_tolerates_absent_optional_columns():
    """Missing optional columns are dropped, not back-filled with blanks.

    In particular no empty ``label`` column is invented. This projection is
    used at inference time too, where there is legitimately no label; a caller
    assembling a *training* frame without ``human_lmp_score`` should get a
    KeyError on the missing column rather than silently train on all-NaN
    targets.
    """
    df = pd.DataFrame({"pair_id": ["X"], "original_text": ["a"], "simplified_text": ["b"]})
    out = load_data.model_input_frame(df)
    assert list(out.columns) == ["pair_id", "original_text", "simplified_text"]
    assert "label" not in out.columns
