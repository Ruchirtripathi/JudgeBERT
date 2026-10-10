"""Tests for the annotation merge.

This module is the single writer of the gold columns, so it is the point at
which the corpus can become contaminated. The tests below are mostly about
what it refuses to do: it must not invent scores, must not mark a row as human
truth without a human score, and must not write anything during a dry run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.annotation import agreement as ag
from src.annotation import merge_annotations as mm


def _adjudications(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    return frame if not frame.empty else pd.DataFrame(columns=ag.ADJUDICATION_COLUMNS)


# ---------------------------------------------------------------------------
# The untouched case
# ---------------------------------------------------------------------------
def test_unannotated_rows_are_left_exactly_as_they_were(master_frame):
    merged, summary = mm.plan_merge(master_frame, pd.DataFrame(columns=ag.ANNOTATION_COLUMNS),
                                    _adjudications([]))
    assert summary["n_rows_annotated"] == 0
    assert summary["n_rows_untouched"] == len(master_frame)
    assert merged["human_lmp_score"].isna().all()
    assert merged["error_type"].isna().all()
    assert merged["decision"].isna().all()
    assert (merged["annotation_status"] == "draft").all()


def test_an_annotator_with_no_matching_pair_changes_nothing(master_frame):
    long_df = pd.DataFrame(
        [{"pair_id": "DOES_NOT_EXIST", "human_lmp_score": 7, "error_type": "None",
          "notes": "", "annotator_id": "alice", "annotated_at": "2026-01-01T00:00:00Z"}]
    )
    merged, summary = mm.plan_merge(master_frame, long_df, _adjudications([]))
    assert summary["n_rows_annotated"] == 0
    assert merged["human_lmp_score"].isna().all()


# ---------------------------------------------------------------------------
# Single annotator
# ---------------------------------------------------------------------------
def test_a_single_annotation_produces_human_single(master_frame):
    long_df = pd.DataFrame(
        [{"pair_id": "ENG_CON_001", "human_lmp_score": 9, "error_type": "None",
          "notes": "clear", "annotator_id": "alice", "annotated_at": "2026-01-02T00:00:00Z"}]
    )
    merged, summary = mm.plan_merge(master_frame, long_df, _adjudications([]))

    row = merged[merged["pair_id"] == "ENG_CON_001"].iloc[0]
    assert row["annotation_status"] == "human_single"
    assert row["human_lmp_score"] == 9.0
    assert row["decision"] == "Preserved"
    assert row["error_type"] == "None"
    assert row["annotator_id"] == "alice"
    assert summary["status_counts"]["human_single"] == 1


def test_a_single_annotation_leaves_the_other_rows_untouched(master_frame):
    long_df = pd.DataFrame(
        [{"pair_id": "ENG_CON_001", "human_lmp_score": 9, "error_type": "None",
          "notes": "", "annotator_id": "alice", "annotated_at": "2026-01-02T00:00:00Z"}]
    )
    merged, _ = mm.plan_merge(master_frame, long_df, _adjudications([]))
    others = merged[merged["pair_id"] != "ENG_CON_001"]
    assert others["human_lmp_score"].isna().all()
    assert (others["annotation_status"] == "draft").all()


def test_a_nan_score_is_not_treated_as_an_annotation(master_frame):
    """A blank score is an abstention, not a judgement of zero."""
    long_df = pd.DataFrame(
        [{"pair_id": "ENG_CON_001", "human_lmp_score": np.nan, "error_type": "None",
          "notes": "", "annotator_id": "alice", "annotated_at": "2026-01-02T00:00:00Z"}]
    )
    merged, summary = mm.plan_merge(master_frame, long_df, _adjudications([]))
    assert summary["n_rows_annotated"] == 0
    assert merged["human_lmp_score"].isna().all()


# ---------------------------------------------------------------------------
# Two annotators: agreement, mean, and the adjudication boundary
# ---------------------------------------------------------------------------
def test_two_close_annotations_average_to_consensus(master_frame):
    long_df = pd.DataFrame(
        [
            {"pair_id": "ENG_CON_001", "human_lmp_score": 8, "error_type": "None", "notes": "a",
             "annotator_id": "alice", "annotated_at": "2026-01-01T00:00:00Z"},
            {"pair_id": "ENG_CON_001", "human_lmp_score": 9, "error_type": "None", "notes": "b",
             "annotator_id": "bob", "annotated_at": "2026-01-02T00:00:00Z"},
        ]
    )
    merged, _ = mm.plan_merge(master_frame, long_df, _adjudications([]))
    row = merged[merged["pair_id"] == "ENG_CON_001"].iloc[0]
    # Spread of 1 is below the threshold of 2, so the mean stands as consensus.
    assert row["annotation_status"] == "human_consensus"
    assert row["human_lmp_score"] == 8.5
    assert set(row["annotator_id"].split(",")) == {"alice", "bob"}


def test_a_spread_at_the_threshold_awaits_adjudication(master_frame):
    """A 2-point spread must not be silently averaged away."""
    long_df = pd.DataFrame(
        [
            {"pair_id": "ENG_CON_001", "human_lmp_score": 7, "error_type": "None", "notes": "a",
             "annotator_id": "alice", "annotated_at": "2026-01-01T00:00:00Z"},
            {"pair_id": "ENG_CON_001", "human_lmp_score": 9, "error_type": "None", "notes": "b",
             "annotator_id": "bob", "annotated_at": "2026-01-02T00:00:00Z"},
        ]
    )
    merged, summary = mm.plan_merge(master_frame, long_df, _adjudications([]))
    row = merged[merged["pair_id"] == "ENG_CON_001"].iloc[0]
    assert row["annotation_status"] == "human_multi_pending"
    # The provisional mean is recorded, and the disagreement is stated in the
    # notes so an adjudicator can find it.
    assert row["human_lmp_score"] == 8.0
    assert "awaiting adjudication" in row["notes"]
    assert "alice=7" in row["notes"] and "bob=9" in row["notes"]


def test_identical_scores_from_two_annotators_give_consensus(master_frame):
    long_df = pd.DataFrame(
        [
            {"pair_id": "ENG_CON_001", "human_lmp_score": 6, "error_type": "Omission", "notes": "",
             "annotator_id": "alice", "annotated_at": "2026-01-01T00:00:00Z"},
            {"pair_id": "ENG_CON_001", "human_lmp_score": 6, "error_type": "Omission", "notes": "",
             "annotator_id": "bob", "annotated_at": "2026-01-02T00:00:00Z"},
        ]
    )
    merged, _ = mm.plan_merge(master_frame, long_df, _adjudications([]))
    row = merged[merged["pair_id"] == "ENG_CON_001"].iloc[0]
    assert row["annotation_status"] == "human_consensus"
    assert row["human_lmp_score"] == 6.0


# ---------------------------------------------------------------------------
# Adjudication outranks the annotators
# ---------------------------------------------------------------------------
def test_an_adjudication_overrides_the_annotator_scores(master_frame):
    long_df = pd.DataFrame(
        [
            {"pair_id": "ENG_CON_001", "human_lmp_score": 2, "error_type": "None", "notes": "a",
             "annotator_id": "alice", "annotated_at": "2026-01-01T00:00:00Z"},
            {"pair_id": "ENG_CON_001", "human_lmp_score": 9, "error_type": "None", "notes": "b",
             "annotator_id": "bob", "annotated_at": "2026-01-02T00:00:00Z"},
        ]
    )
    adjudications = _adjudications(
        [{"pair_id": "ENG_CON_001", "consensus_score": 5, "error_type": "Scope",
          "adjudicator_id": "carol", "notes": "split the difference", "adjudicated_at": "2026-01-03T00:00:00Z"}]
    )
    merged, _ = mm.plan_merge(master_frame, long_df, adjudications)
    row = merged[merged["pair_id"] == "ENG_CON_001"].iloc[0]
    assert row["annotation_status"] == "human_consensus"
    assert row["human_lmp_score"] == 5.0
    assert row["error_type"] == "Scope"
    assert row["annotator_id"] == "carol"
    assert row["decision"] == "Partially Preserved"


def test_an_adjudication_with_a_blank_score_falls_back_to_the_annotators(master_frame):
    """An empty adjudication row is not a decision; it is an unstarted one."""
    long_df = pd.DataFrame(
        [{"pair_id": "ENG_CON_001", "human_lmp_score": 8, "error_type": "None", "notes": "",
          "annotator_id": "alice", "annotated_at": "2026-01-01T00:00:00Z"}]
    )
    adjudications = _adjudications(
        [{"pair_id": "ENG_CON_001", "consensus_score": np.nan, "error_type": "None",
          "adjudicator_id": "carol", "notes": "", "adjudicated_at": "2026-01-03T00:00:00Z"}]
    )
    merged, _ = mm.plan_merge(master_frame, long_df, adjudications)
    row = merged[merged["pair_id"] == "ENG_CON_001"].iloc[0]
    assert row["annotation_status"] == "human_single"
    assert row["human_lmp_score"] == 8.0


# ---------------------------------------------------------------------------
# Error type handling
# ---------------------------------------------------------------------------
def test_agreeing_annotators_keep_their_error_type(master_frame):
    long_df = pd.DataFrame(
        [
            {"pair_id": "ENG_CON_001", "human_lmp_score": 4, "error_type": "Omission", "notes": "",
             "annotator_id": "alice", "annotated_at": "2026-01-01T00:00:00Z"},
            {"pair_id": "ENG_CON_001", "human_lmp_score": 4, "error_type": "Omission", "notes": "",
             "annotator_id": "bob", "annotated_at": "2026-01-02T00:00:00Z"},
        ]
    )
    merged, _ = mm.plan_merge(master_frame, long_df, _adjudications([]))
    assert merged[merged["pair_id"] == "ENG_CON_001"].iloc[0]["error_type"] == "Omission"


def test_disagreeing_error_types_are_reported_as_a_conflict(master_frame):
    """The chosen label must be traceable, and the dissent must not vanish."""
    long_df = pd.DataFrame(
        [
            {"pair_id": "ENG_CON_001", "human_lmp_score": 4, "error_type": "Omission", "notes": "",
             "annotator_id": "alice", "annotated_at": "2026-01-01T00:00:00Z"},
            {"pair_id": "ENG_CON_001", "human_lmp_score": 4, "error_type": "Scope", "notes": "",
             "annotator_id": "bob", "annotated_at": "2026-01-02T00:00:00Z"},
        ]
    )
    merged, summary = mm.plan_merge(master_frame, long_df, _adjudications([]))
    assert summary["n_error_type_conflicts"] == 1
    conflict = summary["error_type_conflicts"][0]
    assert conflict["pair_id"] == "ENG_CON_001"
    assert {conflict["chosen"], *conflict["dissent"].split(", ")} == {"Omission", "Scope"}
    # A tie still has to produce exactly one value in a single-valued column.
    assert merged[merged["pair_id"] == "ENG_CON_001"].iloc[0]["error_type"] in {"Omission", "Scope"}


def test_a_blank_error_type_from_an_annotator_becomes_none(master_frame):
    """'None' is the taxonomy's label for 'no legal error introduced'."""
    long_df = pd.DataFrame(
        [{"pair_id": "ENG_CON_001", "human_lmp_score": 10, "error_type": "", "notes": "",
          "annotator_id": "alice", "annotated_at": "2026-01-01T00:00:00Z"}]
    )
    merged, _ = mm.plan_merge(master_frame, long_df, _adjudications([]))
    assert merged[merged["pair_id"] == "ENG_CON_001"].iloc[0]["error_type"] == "None"


def test_an_unrecognised_error_type_is_relabelled_other_and_kept_in_the_notes(master_frame):
    long_df = pd.DataFrame(
        [{"pair_id": "ENG_CON_001", "human_lmp_score": 4, "error_type": "Typo", "notes": "",
          "annotator_id": "alice", "annotated_at": "2026-01-01T00:00:00Z"}]
    )
    merged, _ = mm.plan_merge(master_frame, long_df, _adjudications([]))
    row = merged[merged["pair_id"] == "ENG_CON_001"].iloc[0]
    assert row["error_type"] == "Other"
    # The annotator's own wording must survive; silently replacing it would
    # destroy information the adjudicator needs.
    assert "Typo" in row["notes"]


def test_notes_from_every_annotator_are_combined(master_frame):
    long_df = pd.DataFrame(
        [
            {"pair_id": "ENG_CON_001", "human_lmp_score": 4, "error_type": "None", "notes": "alice note",
             "annotator_id": "alice", "annotated_at": "2026-01-01T00:00:00Z"},
            {"pair_id": "ENG_CON_001", "human_lmp_score": 4, "error_type": "None", "notes": "bob note",
             "annotator_id": "bob", "annotated_at": "2026-01-02T00:00:00Z"},
        ]
    )
    merged, _ = mm.plan_merge(master_frame, long_df, _adjudications([]))
    notes = merged[merged["pair_id"] == "ENG_CON_001"].iloc[0]["notes"]
    assert "alice note" in notes and "bob note" in notes


# ---------------------------------------------------------------------------
# Decisions and the gold gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "score,expected",
    [(10, "Preserved"), (8, "Preserved"), (7, "Partially Preserved"),
     (5, "Partially Preserved"), (4, "Not Preserved"), (1, "Not Preserved")],
)
def test_decision_follows_the_configured_thresholds(master_frame, score, expected):
    long_df = pd.DataFrame(
        [{"pair_id": "ENG_CON_001", "human_lmp_score": score, "error_type": "None", "notes": "",
          "annotator_id": "alice", "annotated_at": "2026-01-01T00:00:00Z"}]
    )
    merged, _ = mm.plan_merge(master_frame, long_df, _adjudications([]))
    assert merged[merged["pair_id"] == "ENG_CON_001"].iloc[0]["decision"] == expected


def test_merged_rows_satisfy_the_gold_gate(master_frame):
    """The whole point: a merged row is usable as ground truth, and a
    non-merged row still is not."""
    from src.data import load_data, schema

    long_df = pd.DataFrame(
        [{"pair_id": "ENG_CON_001", "human_lmp_score": 9, "error_type": "None", "notes": "",
          "annotator_id": "alice", "annotated_at": "2026-01-01T00:00:00Z"}]
    )
    merged, _ = mm.plan_merge(master_frame, long_df, _adjudications([]))

    gold = schema.is_gold(merged)
    assert gold.sum() == 1
    assert gold[merged["pair_id"] == "ENG_CON_001"].iloc[0]

    # Evaluating on the full frame must still fail, because the other rows
    # remain unannotated.
    with pytest.raises(ValueError, match="not human-annotated"):
        load_data.assert_gold(merged, "Experiment 1")


def test_the_merge_never_invents_a_score_for_an_unannotated_row(master_frame):
    """No annotation in, no annotation out — for every row, every time."""
    merged, _ = mm.plan_merge(master_frame, pd.DataFrame(columns=ag.ANNOTATION_COLUMNS),
                              _adjudications([]))
    assert merged["human_lmp_score"].notna().sum() == 0
    assert merged["error_type"].notna().sum() == 0
    assert set(merged["annotation_status"]) == {"draft"}


# ---------------------------------------------------------------------------
# Dry-run semantics and the file round trip
# ---------------------------------------------------------------------------
def test_plan_merge_does_not_mutate_the_input_frame(master_frame):
    before = master_frame.copy()
    mm.plan_merge(master_frame, pd.DataFrame(columns=ag.ANNOTATION_COLUMNS), _adjudications([]))
    pd.testing.assert_frame_equal(master_frame, before)


def test_render_summary_says_nothing_was_written_on_a_dry_run(master_frame):
    _, summary = mm.plan_merge(master_frame, pd.DataFrame(columns=ag.ANNOTATION_COLUMNS),
                               _adjudications([]))
    text = mm.render_summary(summary, applied=False)
    assert "DRY RUN" in text
    assert "nothing was written" in text


def test_render_summary_says_applied_when_applied(master_frame):
    _, summary = mm.plan_merge(master_frame, pd.DataFrame(columns=ag.ANNOTATION_COLUMNS),
                               _adjudications([]))
    assert "APPLIED" in mm.render_summary(summary, applied=True)


def test_end_to_end_from_annotation_files_on_disk(master_frame, annotation_dir, tmp_path, write_annotations):
    """The whole chain: files written by the tool -> agreement -> merge."""
    from src.utils.io import write_csv

    write_annotations(annotation_dir, "alice", {"ENG_CON_001": 8, "ENG_CON_003": 4})
    write_annotations(annotation_dir, "bob", {"ENG_CON_001": 9, "ENG_CON_003": 4})

    master_path = tmp_path / "lmp_pairs.csv"
    write_csv(master_frame, master_path)

    long_df = ag.load_annotations(directory=annotation_dir)
    merged, summary = mm.plan_merge(master_frame, long_df, _adjudications([]))

    assert summary["n_rows_annotated"] == 2
    by_id = merged.set_index("pair_id")

    # 8 vs 9: within tolerance, averaged, consensus.
    assert by_id.loc["ENG_CON_001", "human_lmp_score"] == 8.5
    assert by_id.loc["ENG_CON_001", "annotation_status"] == "human_consensus"
    # 4 vs 4: exact agreement.
    assert by_id.loc["ENG_CON_003", "annotation_status"] == "human_consensus"
    # Everything else is untouched.
    assert by_id.loc["ENG_CYB_001", "annotation_status"] == "draft"
    assert pd.isna(by_id.loc["ENG_CYB_001", "human_lmp_score"])
