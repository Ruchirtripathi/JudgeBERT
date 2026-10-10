"""Tests for inter-annotator agreement.

Agreement numbers end up in the paper, so the properties tested here are that
the coefficients are actually correct on cases with a known answer, that a
genuine zero is reported as zero rather than as "not computed", and that the
module refuses to produce a number when there is not enough data to support
one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.annotation import agreement as ag


def _matrix(scores: dict[str, list]) -> pd.DataFrame:
    n = len(next(iter(scores.values())))
    return pd.DataFrame(scores, index=[f"P{i:02d}" for i in range(n)])


PERFECT = _matrix({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
OFFSET = _matrix({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "b": [2, 3, 4, 5, 6, 7, 8, 9, 10, 10]})
FLAT = _matrix({"a": [5, 5, 5, 5], "b": [5, 5, 5, 5]})
OPPOSED = _matrix({"a": [1, 1, 1, 1], "b": [10, 10, 10, 10]})


# ---------------------------------------------------------------------------
# Cohen's kappa (quadratic weights)
# ---------------------------------------------------------------------------
def test_kappa_is_one_for_perfect_agreement():
    assert ag.cohen_kappa_quadratic(PERFECT["a"].to_numpy(), PERFECT["b"].to_numpy()) == pytest.approx(1.0)


def test_kappa_for_a_systematic_one_point_offset():
    """Near-perfect, but not perfect: the weight is quadratic, not absolute."""
    kappa = ag.cohen_kappa_quadratic(OFFSET["a"].to_numpy(), OFFSET["b"].to_numpy())
    assert kappa == pytest.approx(0.9454545454545454)


def test_kappa_is_none_when_a_rater_has_no_variance():
    """Chance agreement is undefined, not perfect.

    Returning 1.0 here would be the worst possible error: two annotators who
    both wrote "5" on every pair would be reported as in complete agreement.
    """
    assert ag.cohen_kappa_quadratic(FLAT["a"].to_numpy(), FLAT["b"].to_numpy()) is None


def test_complete_disagreement_gives_exactly_zero():
    """kappa == 0.0 is falsy, and must not be mistaken for 'not computed'."""
    kappa = ag.cohen_kappa_quadratic(OPPOSED["a"].to_numpy(), OPPOSED["b"].to_numpy())
    assert kappa == 0.0
    assert kappa is not None


def test_kappa_ignores_pairs_where_either_rater_abstained():
    a = pd.Series([1.0, 2.0, None, 4.0]).to_numpy(dtype=object)
    b = pd.Series([1.0, None, 3.0, 5.0]).to_numpy(dtype=object)
    # Only rows 0 and 3 have two ratings.
    assert ag.cohen_kappa_quadratic(a, b) is not None


def test_kappa_is_symmetric_in_its_arguments():
    forward = ag.cohen_kappa_quadratic(OFFSET["a"].to_numpy(), OFFSET["b"].to_numpy())
    backward = ag.cohen_kappa_quadratic(OFFSET["b"].to_numpy(), OFFSET["a"].to_numpy())
    assert forward == pytest.approx(backward)


def test_kappa_returns_none_with_fewer_than_two_shared_ratings():
    assert ag.cohen_kappa_quadratic([5.0], [5.0]) is None


# ---------------------------------------------------------------------------
# Krippendorff's alpha (ordinal)
# ---------------------------------------------------------------------------
def test_alpha_is_one_for_perfect_agreement():
    assert ag.krippendorff_alpha_ordinal(PERFECT) == pytest.approx(1.0)


def test_alpha_is_negative_for_systematic_disagreement():
    """Alpha is not bounded below by zero, unlike kappa.

    Systematic opposition is worse than chance, and alpha says so.
    """
    assert ag.krippendorff_alpha_ordinal(OPPOSED) == pytest.approx(-0.75)


def test_alpha_is_none_when_no_pair_was_rated_twice():
    single = _matrix({"a": [1, 2, 3], "b": [None, None, None]})
    assert ag.krippendorff_alpha_ordinal(single) is None


def test_alpha_handles_three_raters():
    """Kappa is undefined for three raters; alpha is not, which is why both exist."""
    three = _matrix({"a": [1, 2, 3, 4, 5, 6], "b": [1, 2, 3, 4, 5, 6], "c": [1, 2, 3, 4, 5, 6]})
    assert ag.krippendorff_alpha_ordinal(three) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ICC(2,1)
# ---------------------------------------------------------------------------
def test_icc_is_one_for_perfect_agreement():
    assert ag.icc_2_1(PERFECT) == pytest.approx(1.0)


def test_icc_for_a_systematic_one_point_offset():
    assert ag.icc_2_1(OFFSET) == pytest.approx(0.9506398537477148)


def test_icc_is_none_without_between_target_variance():
    assert ag.icc_2_1(FLAT) is None


def test_icc_is_not_defined_for_a_single_rater():
    assert ag.icc_2_1(_matrix({"a": [1, 2, 3]})) is None


# ---------------------------------------------------------------------------
# Descriptive agreement
# ---------------------------------------------------------------------------
def test_percent_within_tolerance_counts_boundary_differences_as_agreement():
    assert ag.percent_within(OFFSET, 1.0) == 1.0
    assert ag.percent_within(OPPOSED, 1.0) == 0.0


def test_mean_absolute_difference_is_zero_for_perfect_agreement():
    assert ag.mean_absolute_difference(PERFECT) == 0.0


def test_mean_absolute_difference_for_the_offset_case():
    assert ag.mean_absolute_difference(OFFSET) == pytest.approx(0.9)


def test_percent_within_and_mad_ignore_unrated_pairs():
    matrix = _matrix({"a": [1.0, 5.0, None], "b": [2.0, None, 7.0]})
    # Only pair 0 has two ratings, and they differ by exactly 1.
    assert ag.percent_within(matrix, 1.0) == 1.0
    assert ag.mean_absolute_difference(matrix) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Kappa bands
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, "slight"),
        (0.19, "slight"),
        (0.25, "fair"),
        (0.5, "moderate"),
        (0.7, "substantial"),
        (0.9, "almost perfect"),
        (1.0, "almost perfect"),
    ],
)
def test_kappa_band_boundaries(value, expected):
    assert ag.kappa_band(value) == expected


def test_kappa_band_returns_pending_for_a_missing_value():
    """A band label must never be invented for an uncomputed coefficient."""
    assert ag.kappa_band(None) == ag.RESULT_PENDING


# ---------------------------------------------------------------------------
# Rating matrix
# ---------------------------------------------------------------------------
def test_rating_matrix_is_pairs_by_annotators(tmp_path, annotation_dir, write_annotations):
    write_annotations(annotation_dir, "alice", {"P1": 5, "P2": 6})
    write_annotations(annotation_dir, "bob", {"P1": 5, "P3": 7})
    long_df = ag.load_annotations(directory=annotation_dir)

    matrix = ag.rating_matrix(long_df)
    assert set(matrix.columns) == {"alice", "bob"}
    assert set(matrix.index) == {"P1", "P2", "P3"}
    # An abstention is missing, not zero.
    assert pd.isna(matrix.loc["P2", "bob"])
    assert not pd.isna(matrix.loc["P2", "alice"])


def test_rating_matrix_is_empty_for_an_empty_frame():
    assert ag.rating_matrix(pd.DataFrame()).empty


def test_out_of_range_scores_are_dropped_at_load(annotation_dir, caplog, write_annotations):
    """A score of 11 must not reach the coefficient.

    Quadratic weights and the ordinal metric are defined by distance on the
    1-10 scale, so an out-of-range value does not just add noise, it yields a
    coefficient that is off the scale and looks like a finding. Agreement is a
    reported number, so the row is dropped and the drop is logged.
    """
    write_annotations(annotation_dir, "a", {"P1": 11, "P2": 0, "P3": 5, "P4": 7})
    long_df = ag.load_annotations(directory=annotation_dir)

    assert set(long_df["pair_id"]) == {"P3", "P4"}
    assert any("outside" in record.message for record in caplog.records)


def test_in_range_boundary_scores_are_kept(annotation_dir, write_annotations):
    write_annotations(annotation_dir, "a", {"P1": 1, "P2": 10})
    assert set(ag.load_annotations(directory=annotation_dir)["pair_id"]) == {"P1", "P2"}


# ---------------------------------------------------------------------------
# Loading annotation files
# ---------------------------------------------------------------------------
def test_load_annotations_returns_an_empty_frame_for_an_empty_directory(annotation_dir):
    empty = ag.load_annotations(directory=annotation_dir)
    assert empty.empty
    assert list(empty.columns) == ag.ANNOTATION_COLUMNS


def test_load_annotations_reads_one_row_per_score(tmp_path, annotation_dir, write_annotations):
    write_annotations(annotation_dir, "alice", {"P1": 5, "P2": 6})
    write_annotations(annotation_dir, "bob", {"P1": 5})
    long_df = ag.load_annotations(directory=annotation_dir)
    assert len(long_df) == 3
    assert set(long_df["annotator_id"]) == {"alice", "bob"}


def test_load_annotations_trusts_the_declared_id_over_the_filename(annotation_dir, caplog):
    """The *declared* id wins, and that is the safe direction.

    Trusting the filename would let a copy of one annotator's file masquerade
    as a second independent annotator and manufacture perfect agreement.
    Trusting the declared id collapses the copy back onto the original, so
    ``n_annotators`` stays 1 and the module correctly reports
    ``RESULT_PENDING``. The mismatch is logged rather than hidden.
    """
    path = annotation_dir / "carol.csv"
    pd.DataFrame(
        [{"pair_id": "P1", "human_lmp_score": 4, "error_type": "None",
          "notes": "", "annotator_id": "someone_else", "annotated_at": "2026-01-01T00:00:00Z"}]
    ).to_csv(path, index=False, encoding="utf-8", lineterminator="\n")

    long_df = ag.load_annotations(directory=annotation_dir)
    assert set(long_df["annotator_id"]) == {"someone_else"}
    assert any("trusting the" in record.message for record in caplog.records)


def test_a_duplicated_annotation_file_cannot_fake_a_second_annotator(annotation_dir, write_annotations):
    """The integrity property the test above is really protecting.

    Copying alice.csv must not produce two annotators in perfect agreement —
    that would be the easiest possible way to fabricate an agreement figure.
    """
    scores = {f"P{i}": (i % 10) + 1 for i in range(25)}
    original = write_annotations(annotation_dir, "alice", scores)
    (annotation_dir / "alice_copy.csv").write_bytes(original.read_bytes())

    result = ag.compute_agreement(ag.load_annotations(directory=annotation_dir))
    assert result["status"] == ag.RESULT_PENDING
    assert result["n_annotators"] == 1


def test_a_file_declaring_two_annotator_ids_is_rejected(annotation_dir, caplog):
    """One file is one annotator; a mixed file cannot be attributed."""
    path = annotation_dir / "mixed.csv"
    pd.DataFrame(
        [
            {"pair_id": "P1", "human_lmp_score": 4, "error_type": "None", "notes": "",
             "annotator_id": "alice", "annotated_at": "2026-01-01T00:00:00Z"},
            {"pair_id": "P2", "human_lmp_score": 6, "error_type": "None", "notes": "",
             "annotator_id": "bob", "annotated_at": "2026-01-01T00:00:00Z"},
        ]
    ).to_csv(path, index=False, encoding="utf-8", lineterminator="\n")

    assert ag.load_annotations(directory=annotation_dir).empty
    assert any("several annotator_ids" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# compute_agreement: the top-level gate
# ---------------------------------------------------------------------------
def test_compute_agreement_is_pending_with_no_annotations(annotation_dir):
    result = ag.compute_agreement(ag.load_annotations(directory=annotation_dir))
    assert result["status"] == ag.RESULT_PENDING
    assert "No annotation files" in result["reason"]


def test_compute_agreement_is_pending_with_one_annotator(annotation_dir, write_annotations):
    write_annotations(annotation_dir, "alice", {f"P{i}": 5 for i in range(30)})
    result = ag.compute_agreement(ag.load_annotations(directory=annotation_dir))
    assert result["status"] == ag.RESULT_PENDING
    assert "Only 1 annotator" in result["reason"]


def test_compute_agreement_is_pending_below_the_overlap_threshold(annotation_dir, write_annotations):
    """Six shared pairs cannot support a coefficient, and it must not pretend."""
    scores = {f"P{i}": (i % 10) + 1 for i in range(6)}
    write_annotations(annotation_dir, "alice", scores)
    write_annotations(annotation_dir, "bob", scores)

    result = ag.compute_agreement(ag.load_annotations(directory=annotation_dir), min_overlap=20)
    assert result["status"] == ag.RESULT_PENDING
    assert "at least 20" in result["reason"]
    assert result["cohen_kappa_quadratic"] == ag.RESULT_PENDING
    # Row-level spread is still reported, because it is informative even
    # when the coefficient is not.
    assert "disagreements" in result


def test_compute_agreement_computes_once_there_is_enough_overlap(annotation_dir, write_annotations):
    scores = {f"P{i}": (i % 10) + 1 for i in range(40)}
    write_annotations(annotation_dir, "alice", scores)
    write_annotations(annotation_dir, "bob", scores)

    result = ag.compute_agreement(ag.load_annotations(directory=annotation_dir), n_resamples=50)
    assert result["status"] == "computed"
    assert result["cohen_kappa_quadratic"] == pytest.approx(1.0)
    assert result["krippendorff_alpha_ordinal"] == pytest.approx(1.0)
    assert result["percent_within_1"] == 1.0


def test_compute_agreement_reports_a_genuine_zero_as_zero(annotation_dir, write_annotations):
    """The falsy-zero trap, at the level a caller actually sees.

    Complete disagreement is a real and interesting result. If the top-level
    function used ``value or RESULT_PENDING`` it would report "not computed",
    which is both wrong and the opposite of informative.
    """
    scores_a = {f"P{i}": 1 for i in range(25)}
    scores_b = {f"P{i}": 10 for i in range(25)}
    write_annotations(annotation_dir, "alice", scores_a)
    write_annotations(annotation_dir, "bob", scores_b)

    result = ag.compute_agreement(ag.load_annotations(directory=annotation_dir), n_resamples=20)
    assert result["status"] == "computed"
    assert result["cohen_kappa_quadratic"] == 0.0
    assert result["cohen_kappa_quadratic"] != ag.RESULT_PENDING
    assert result["cohen_kappa_band"] == "slight"


def test_compute_agreement_with_three_annotators_omits_kappa_but_gives_alpha(annotation_dir, write_annotations):
    """Weighted kappa is a two-rater statistic; reporting one would be wrong."""
    scores = {f"P{i}": (i % 10) + 1 for i in range(30)}
    for name in ("alice", "bob", "carol"):
        write_annotations(annotation_dir, name, scores)

    result = ag.compute_agreement(ag.load_annotations(directory=annotation_dir), n_resamples=20)
    assert result["status"] == "computed"
    assert result["cohen_kappa_quadratic"] == ag.RESULT_PENDING
    assert result["krippendorff_alpha_ordinal"] == pytest.approx(1.0)
    assert result["mean_pairwise_kappa"] == pytest.approx(1.0)


def test_compute_agreement_counts_pairs_needing_adjudication(annotation_dir, write_annotations):
    scores = {f"P{i}": 5 for i in range(25)}
    spread = dict(scores)
    spread["P0"] = 9  # a 4-point spread, well past the threshold of 2
    write_annotations(annotation_dir, "alice", scores)
    write_annotations(annotation_dir, "bob", spread)

    result = ag.compute_agreement(ag.load_annotations(directory=annotation_dir), n_resamples=20)
    assert result["n_pairs_needing_adjudication"] == 1


# ---------------------------------------------------------------------------
# Disagreement table and bootstrap
# ---------------------------------------------------------------------------
def test_disagreement_table_lists_only_pairs_at_or_above_the_threshold():
    matrix = _matrix({"a": [5, 5, 5, 5], "b": [5, 6, 7, 9]})
    table = ag.disagreement_table(matrix, threshold=2)
    assert len(table) == 2  # differences of 2 and 4


def test_bootstrap_ci_brackets_the_point_estimate():
    matrix = _matrix(
        {"a": [1, 3, 5, 7, 9, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
         "b": [1, 3, 5, 7, 9, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10]}
    )
    low, high = ag.bootstrap_ci(
        matrix,
        lambda m: ag.cohen_kappa_quadratic(m.iloc[:, 0].to_numpy(), m.iloc[:, 1].to_numpy()),
        n_resamples=200,
        seed=1,
    )
    assert low <= 1.0 <= high


def test_bootstrap_ci_is_reproducible_for_a_given_seed():
    a = ag.bootstrap_ci(OPPOSED, lambda m: 0.5, n_resamples=10, seed=3)
    b = ag.bootstrap_ci(OPPOSED, lambda m: 0.5, n_resamples=10, seed=3)
    assert a == b


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def test_render_report_explains_why_it_is_pending(annotation_dir):
    text = ag.render_report(ag.compute_agreement(ag.load_annotations(directory=annotation_dir)))
    assert ag.RESULT_PENDING in text
    assert "No annotation files" in text


def test_render_report_includes_the_computed_values(annotation_dir, write_annotations):
    scores = {f"P{i}": (i % 10) + 1 for i in range(40)}
    write_annotations(annotation_dir, "alice", scores)
    write_annotations(annotation_dir, "bob", scores)
    text = ag.render_report(
        ag.compute_agreement(ag.load_annotations(directory=annotation_dir), n_resamples=20)
    )
    assert "Cohen" in text
    assert "Krippendorff" in text
    assert "ICC" in text
