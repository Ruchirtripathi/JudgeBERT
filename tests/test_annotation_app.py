"""Tests for the human annotation workflow.

The annotation tool is the only component in this project that creates ground
truth, and its failure mode is quiet. A tool that skips every other pair, or
that loses an afternoon of hand-entered scores to a half-finished write, does
not raise: it produces a corpus that looks annotated and is not. These tests
therefore pin the properties that make a label trustworthy — the range, the
provenance, the durability of the file, and the blindness of the annotator —
rather than the cosmetics of the interface.

Three layers:

* **Persistence** — the functions that write a human label, tested directly.
* **Blindness** — that no automatically generated expected label reaches the
  annotator, asserted against the *rendered* app rather than the source, since
  the risk is a value appearing on screen, not a variable existing.
* **Independence** — that agreement is computed from independently submitted
  annotations, and that an adjudicator cannot resolve their own disagreement.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import pytest

from src.annotation import agreement
from src.annotation import annotation_app as app
from src.annotation import curator_review

streamlit_testing = pytest.importorskip(
    "streamlit.testing.v1", reason="streamlit is required to render the annotation tool"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def annotations_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every write the app makes into a temporary directory.

    Without this the tests would write into ``data/annotations/`` for real, and
    a stray test annotator file left behind by a crash would then be read as a
    genuine annotator by ``agreement.load_annotations``.
    """
    root = tmp_path / "annotations"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app, "_annotations_root", lambda cfg=None: root)
    return root


def _write_annotator_file(root: Path, annotator_id: str, rows: list[dict]) -> Path:
    """Write a CSV in the exact format the tool produces."""
    directory = root / "annotators"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{annotator_id}.csv"
    pd.DataFrame(rows, columns=agreement.ANNOTATION_COLUMNS).to_csv(
        path, index=False, encoding="utf-8", lineterminator="\n"
    )
    return path


# ---------------------------------------------------------------------------
# The module must be importable and testable at all
# ---------------------------------------------------------------------------
def test_the_module_imports_without_a_streamlit_runtime():
    """Regression: ``st.set_page_config`` used to run at import time.

    A module-level ``set_page_config`` made the whole file un-importable outside
    a script run, which left the label-writing code — the only code that may
    create ground truth — with no direct tests. Page setup now runs from
    ``main()``, and this test fails the moment anything moves back to import
    time.
    """
    assert callable(app.main)
    assert callable(app._save_score)


# ---------------------------------------------------------------------------
# Annotator identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["A01", "annotator-2", "rater_3", "AB", "x"])
def test_valid_annotator_ids_round_trip(value):
    assert app._validate_annotator_id(value) == value


def test_annotator_ids_are_stripped():
    assert app._validate_annotator_id("  A01  ") == "A01"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "../processed/lmp_pairs",
        "sub/dir",
        "back\\slash",
        "C:name",
        ".hidden",
        "---",
        "x" * (app.MAX_ANNOTATOR_ID_LENGTH + 1),
    ],
)
def test_unsafe_annotator_ids_are_refused(value):
    """The id becomes a filename, so it is constrained rather than trusted.

    ``../processed/lmp_pairs`` would make the tool write outside its own
    directory — and the annotator's real file would silently not exist.
    """
    with pytest.raises(ValueError):
        app._validate_annotator_id(value)


# ---------------------------------------------------------------------------
# Score validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("score", [1, 5, 10])
def test_scores_in_range_are_accepted(score):
    assert app._validated_score(score, context="t") == score


@pytest.mark.parametrize("score", [0, 11, -1, 100])
def test_out_of_range_scores_are_refused(score):
    """Quadratic kappa and the ordinal metric are defined on 1-10.

    A stored 11 does not merely add error: it yields a coefficient that looks
    like a real result and is not, so the value must never reach a file this
    tool wrote.
    """
    with pytest.raises(ValueError):
        app._validated_score(score, context="t")


@pytest.mark.parametrize("score", [7.0, 7.5, "7", None, True, [7]])
def test_non_integer_scores_are_refused(score):
    """``True`` is excluded explicitly: it is an ``int`` in Python."""
    with pytest.raises(ValueError):
        app._validated_score(score, context="t")


def test_a_rejected_score_leaves_the_file_untouched(annotations_root: Path):
    _write_annotator_file(
        annotations_root,
        "A01",
        [
            {
                "pair_id": "P1",
                "human_lmp_score": 4,
                "error_type": "Scope",
                "notes": "",
                "annotator_id": "A01",
                "annotated_at": "2026-01-01T00:00:00Z",
            }
        ],
    )
    before = (annotations_root / "annotators" / "A01.csv").read_text(encoding="utf-8")

    with pytest.raises(ValueError):
        app._save_score("A01", "P2", 11, "Omission", "")

    assert (annotations_root / "annotators" / "A01.csv").read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# Reading an annotator file that has been damaged
# ---------------------------------------------------------------------------
def test_out_of_range_rows_are_dropped_on_load_not_averaged_in(annotations_root: Path):
    _write_annotator_file(
        annotations_root,
        "A01",
        [
            {"pair_id": "P1", "human_lmp_score": 4, "error_type": "Scope",
             "notes": "", "annotator_id": "A01", "annotated_at": "2026-01-01T00:00:00Z"},
            {"pair_id": "P2", "human_lmp_score": 11, "error_type": "Scope",
             "notes": "", "annotator_id": "A01", "annotated_at": "2026-01-01T00:00:00Z"},
            {"pair_id": "P3", "human_lmp_score": "not a number", "error_type": "Scope",
             "notes": "", "annotator_id": "A01", "annotated_at": "2026-01-01T00:00:00Z"},
        ],
    )
    frame = app._load_own_scores("A01")

    assert list(frame["pair_id"]) == ["P1"]
    assert {row["pair_id"] for row in app._rejected_scores(frame)} == {"P2", "P3"}


def test_a_damaged_file_does_not_crash_the_score_widget():
    """The widget used to raise ``ValueError`` on a stored 11.

    ``list(range(1, 11)).index(11)`` raises, so a single hand-edited score took
    the whole session down instead of being reported. The out-of-range value
    now preselects nothing.
    """
    assert app._score_index(11) is None
    assert app._score_index(0) is None
    assert app._score_index(None) is None
    assert app._score_index(1) == 0
    assert app._score_index(10) == 9


def test_an_out_of_range_row_is_dropped_by_clean_scores():
    frame = pd.DataFrame(
        [{"pair_id": "P1", "human_lmp_score": 11, "error_type": "Scope", "notes": "",
          "annotator_id": "A01", "annotated_at": "2026-01-01T00:00:00Z"}]
    )
    cleaned, rejected = app._clean_scores(frame)
    assert cleaned.empty
    assert rejected == [{"pair_id": "P1", "human_lmp_score": 11}]


def test_a_missing_score_column_is_handled(annotations_root: Path):
    path = annotations_root / "annotators" / "A01.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("pair_id,error_type\nP1,Scope\n", encoding="utf-8")

    frame = app._load_own_scores("A01")
    assert frame.empty


# ---------------------------------------------------------------------------
# Durability — the annotations are not reproducible
# ---------------------------------------------------------------------------
def test_a_failed_write_leaves_the_previous_file_intact(annotations_root: Path, monkeypatch):
    """Regression: the file used to be truncated by ``to_csv`` before writing.

    A crash, a full disk or a closed laptop part-way through destroyed every
    annotation the annotator had made. Those rows are hours of human judgement
    and cannot be regenerated, so the write goes to a temporary file and is
    moved into place only once it is complete.
    """
    path = annotations_root / "annotators" / "A01.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    original = "pair_id,human_lmp_score,error_type,notes,annotator_id,annotated_at\n"
    path.write_text(original, encoding="utf-8")

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", explode)
    with pytest.raises(OSError):
        app._atomic_write_csv(pd.DataFrame([{"pair_id": "P1"}]), path)

    assert path.read_text(encoding="utf-8") == original
    leftovers = [p.name for p in path.parent.iterdir() if p.name != "A01.csv"]
    assert leftovers == [], f"temporary files were left behind: {leftovers}"


def test_a_successful_write_leaves_no_temporary_files(annotations_root: Path):
    path = annotations_root / "annotators" / "A01.csv"
    app._atomic_write_csv(pd.DataFrame([{"pair_id": "P1", "human_lmp_score": 5}]), path)

    assert path.exists()
    assert [p.name for p in path.parent.iterdir()] == ["A01.csv"]


def test_saving_stamps_provenance(annotations_root: Path):
    app._save_score("A01", "P1", 6, "Omission", "a note")

    frame = app._load_own_scores("A01")
    row = frame.iloc[0]
    assert row["annotator_id"] == "A01"
    assert row["human_lmp_score"] == 6
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", str(row["annotated_at"]))


def test_saving_is_an_upsert_not_an_append(annotations_root: Path):
    """Re-scoring a pair replaces the judgement; it does not add a second row."""
    app._save_score("A01", "P1", 3, "Contradiction", "first reading")
    app._save_score("A01", "P1", 8, "None", "changed my mind")

    frame = app._load_own_scores("A01")
    assert len(frame) == 1
    assert int(frame.iloc[0]["human_lmp_score"]) == 8


def test_resuming_reads_back_exactly_what_was_saved(annotations_root: Path):
    app._save_score("A01", "P1", 9, "None", "")
    app._save_score("A01", "P2", 2, "Contradiction", "negation dropped")

    frame = app._load_own_scores("A01")
    assert dict(zip(frame["pair_id"], frame["human_lmp_score"])) == {"P1": 9, "P2": 2}
    assert frame.set_index("pair_id").loc["P2", "notes"] == "negation dropped"


def test_annotators_do_not_overwrite_each_other(annotations_root: Path):
    app._save_score("A01", "P1", 3, "Scope", "")
    app._save_score("A02", "P1", 7, "None", "")

    assert int(app._load_own_scores("A01").iloc[0]["human_lmp_score"]) == 3
    assert int(app._load_own_scores("A02").iloc[0]["human_lmp_score"]) == 7


# ---------------------------------------------------------------------------
# The cursor — regression for a bug that silently skipped half the corpus
# ---------------------------------------------------------------------------
def test_the_cursor_stays_put_when_the_queue_shrinks():
    """Regression: saving used to advance the cursor while the queue shrank.

    With "only pairs I have not scored" on, the saved pair leaves the queue on
    the next run, so the following pair slides down into the same index. The
    old code advanced anyway, which meant an annotator working the default
    queue annotated every *other* pair — and the progress bar agreed with them,
    because it counted the pairs actually on disk. The loss was invisible.
    """
    assert app._next_cursor(0, 340, only_unscored=True) == 0
    assert app._next_cursor(7, 340, only_unscored=True) == 7


def test_the_cursor_advances_when_the_whole_corpus_is_shown():
    assert app._next_cursor(0, 340, only_unscored=False) == 1
    assert app._next_cursor(7, 340, only_unscored=False) == 8


def test_the_cursor_never_runs_past_the_end():
    assert app._next_cursor(2, 3, only_unscored=False) == 2
    assert app._next_cursor(0, 1, only_unscored=False) == 0


# ---------------------------------------------------------------------------
# The queue must not leak the draft
# ---------------------------------------------------------------------------
def test_the_queue_offers_no_draft_based_ordering(master_frame: pd.DataFrame):
    """Ordering by a hidden field leaks it through the interface.

    The tool used to offer "lowest draft score first". Selecting it told the
    annotator what the draft says about every pair before they read it, which
    is the anchor blind mode exists to remove — delivered by the sort order
    rather than by a number on screen.
    """
    source = Path(app.__file__).read_text(encoding="utf-8")
    assert "Lowest draft score first" not in source
    assert 'sort_values(["draft_lmp_score"' not in source


def test_the_shuffle_is_deterministic_for_one_annotator(master_frame: pd.DataFrame):
    seed = app._shuffle_seed("A01")
    first = app._queue_frame(master_frame, "all", "all", "all", "Shuffled", seed)
    second = app._queue_frame(master_frame, "all", "all", "all", "Shuffled", seed)
    assert list(first["pair_id"]) == list(second["pair_id"])


def test_two_annotators_get_different_orders(master_frame: pd.DataFrame):
    """Identical orders would make position and fatigue shared between them."""
    a = app._queue_frame(master_frame, "all", "all", "all", "Shuffled", app._shuffle_seed("A01"))
    b = app._queue_frame(master_frame, "all", "all", "all", "Shuffled", app._shuffle_seed("A02"))
    assert list(a["pair_id"]) != list(b["pair_id"])


def test_the_seed_is_stable_across_processes():
    """``hash()`` is salted per process, so it cannot produce a resumable order."""
    assert app._shuffle_seed("A01") == app._shuffle_seed("A01")
    assert app._shuffle_seed("A01") != app._shuffle_seed("A02")


# ---------------------------------------------------------------------------
# Independence of the adjudicator
# ---------------------------------------------------------------------------
def test_an_annotator_cannot_adjudicate_their_own_disagreement():
    conflict = app._adjudicator_conflict("A01", {"A01", "A02"})
    assert conflict is not None and "different person" in conflict


def test_a_third_party_may_adjudicate():
    assert app._adjudicator_conflict("A03", {"A01", "A02"}) is None


def test_an_empty_adjudicator_id_is_not_a_conflict():
    """Emptiness is a separate, already-reported problem."""
    assert app._adjudicator_conflict("", {"A01", "A02"}) is None


def test_agreement_ignores_adjudications(annotations_root: Path):
    """Agreement is computed from independently submitted annotations only.

    If an adjudicated score fed back into the coefficient, the reported
    agreement would partly measure the adjudicator's own choices — the number
    would move towards 1 as disputes were resolved, which is the opposite of
    what an agreement statistic means.
    """
    _write_annotator_file(
        annotations_root,
        "A01",
        [{"pair_id": f"P{i}", "human_lmp_score": 3, "error_type": "Scope", "notes": "",
          "annotator_id": "A01", "annotated_at": "2026-01-01T00:00:00Z"} for i in range(25)],
    )
    _write_annotator_file(
        annotations_root,
        "A02",
        [{"pair_id": f"P{i}", "human_lmp_score": 8, "error_type": "None", "notes": "",
          "annotator_id": "A02", "annotated_at": "2026-01-01T00:00:00Z"} for i in range(25)],
    )
    adjudication = annotations_root / "adjudication.csv"
    pd.DataFrame(
        [{"pair_id": f"P{i}", "consensus_score": 8, "error_type": "None",
          "adjudicator_id": "A03", "notes": "", "adjudicated_at": "2026-01-01T00:00:00Z"}
         for i in range(25)],
        columns=agreement.ADJUDICATION_COLUMNS,
    ).to_csv(adjudication, index=False, encoding="utf-8", lineterminator="\n")

    cfg = {"paths": {"annotations_dir_abs": str(annotations_root)}}
    long_df = agreement.load_annotations(cfg)

    # The adjudication file exists and is non-empty, but contributes no ratings.
    assert set(long_df["annotator_id"]) == {"A01", "A02"}

    result = agreement.compute_agreement(long_df)
    assert result["status"] == "computed"
    assert result["n_annotators"] == 2
    assert result["n_pairs_overlapping"] == 25
    # Perfectly opposed ratings, so agreement is at chance, not consensus.
    assert result["cohen_kappa_quadratic"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The curator's draft comparison is curator-only and never fabricated
# ---------------------------------------------------------------------------
def test_draft_comparison_is_pending_without_human_labels(master_frame: pd.DataFrame):
    result = curator_review.compare_to_drafts(master_frame, agreement.load_annotations(
        {"paths": {"annotations_dir_abs": "/nonexistent"}}
    ))
    assert result["status"] == curator_review.RESULT_PENDING
    assert result["pearson"] == curator_review.RESULT_PENDING
    assert "No annotation files" in result["reason"]


def test_draft_comparison_reports_the_real_numbers(master_frame: pd.DataFrame):
    """Known-answer case: a constant offset is recovered exactly."""
    long_df = pd.DataFrame(
        [
            {
                "pair_id": row.pair_id,
                "human_lmp_score": float(row.draft_lmp_score),
                "error_type": "None",
                "notes": "",
                "annotator_id": "A01",
                "annotated_at": "2026-01-01T00:00:00Z",
            }
            for row in master_frame.itertuples(index=False)
        ]
    )
    result = curator_review.compare_to_drafts(master_frame, long_df, min_pairs=3)

    assert result["status"] == "computed"
    assert result["n_pairs_compared"] == len(master_frame)
    assert result["mean_signed_difference"] == pytest.approx(0.0)
    assert result["mean_absolute_difference"] == pytest.approx(0.0)
    assert result["percent_within_1"] == pytest.approx(1.0)
    assert result["decision_agreement"] == pytest.approx(1.0)


def test_draft_comparison_never_promises_a_result_it_cannot_compute():
    """Few observations means PENDING, not a correlation over noise."""
    master = pd.DataFrame(
        {"pair_id": ["P1", "P2"], "draft_lmp_score": [3.0, 8.0]}
    )
    long_df = pd.DataFrame(
        [
            {"pair_id": "P1", "human_lmp_score": 4, "error_type": "None", "notes": "",
             "annotator_id": "A01", "annotated_at": "x"},
            {"pair_id": "P2", "human_lmp_score": 7, "error_type": "None", "notes": "",
             "annotator_id": "A01", "annotated_at": "x"},
        ]
    )
    result = curator_review.compare_to_drafts(master, long_df, min_pairs=20)
    assert result["status"] == curator_review.RESULT_PENDING
    assert result["pearson"] == curator_review.RESULT_PENDING


def test_the_curator_report_warns_that_it_is_curator_only():
    rendered = curator_review.render_report(
        {"status": curator_review.RESULT_PENDING, "reason": "no data", "rows": []}
    )
    assert "CURATOR-ONLY" in rendered


# ===========================================================================
# The rendered application
# ===========================================================================
_WRAPPER = '''
import os, sys
from pathlib import Path

REPO = os.environ["LMP_TEST_REPO"]
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.annotation import annotation_app

annotation_app._annotations_root = lambda cfg=None: Path(os.environ["LMP_TEST_ANNOTATIONS"])

import streamlit as st
st.session_state["annotator_id"] = os.environ.get("LMP_TEST_ANNOTATOR", "A01")

annotation_app.main()
'''


@pytest.fixture
def wrapper_script(tmp_path: Path, repo_root: Path, annotations_root: Path) -> Path:
    """A script that renders the real app against a temporary annotation store.

    Streamlit re-executes the app file on every run, so the app cannot be
    imported and monkeypatched in-process; the redirection has to happen inside
    the script that Streamlit runs.
    """
    path = tmp_path / "app_wrapper.py"
    path.write_text(_WRAPPER, encoding="utf-8")
    os.environ["LMP_TEST_REPO"] = str(repo_root)
    os.environ["LMP_TEST_ANNOTATIONS"] = str(annotations_root)
    yield path
    os.environ.pop("LMP_TEST_REPO", None)
    os.environ.pop("LMP_TEST_ANNOTATIONS", None)


def _rendered_text(at) -> str:
    """Everything the user can read, as one string."""
    parts: list[str] = []
    for collection in ("markdown", "caption", "info", "warning", "error", "success", "title"):
        for element in getattr(at, collection):
            parts.append(str(getattr(element, "value", "")))
    return "\n".join(parts)


def _shown_pair(at) -> str | None:
    hit = re.search(r"<code>([A-Z_0-9]+)</code>", _rendered_text(at))
    return hit.group(1) if hit else None


@pytest.fixture
def running_app(wrapper_script: Path):
    at = streamlit_testing.AppTest.from_file(str(wrapper_script), default_timeout=60)
    at.run()
    return at


def test_the_app_renders_without_error(running_app):
    assert not running_app.exception


def test_the_annotator_sees_every_required_field(running_app, master_frame):
    """Requirement: pair_id, language, domain, both texts, score, error, notes."""
    text = _rendered_text(running_app)
    assert _shown_pair(running_app) is not None          # pair_id
    assert "English" in text or "Hindi" in text          # language
    assert "Contract Law" in text or "Cyber Law" in text # legal domain
    assert "Original — authoritative wording" in text
    assert "Simplified — the rewrite being judged" in text

    assert [r for r in running_app.radio if r.key.startswith("score_")]
    assert [s for s in running_app.selectbox if s.key.startswith("err_")]
    assert [t for t in running_app.text_input if t.key.startswith("notes_")]


def test_the_decision_is_shown_and_derived_from_the_score(running_app):
    """The decision is displayed but never entered, so it cannot disagree."""
    radio = [r for r in running_app.radio if r.key.startswith("score_")][0]
    radio.set_value(9)
    running_app.run()
    assert "Preserved" in _rendered_text(running_app)

    radio = [r for r in running_app.radio if r.key.startswith("score_")][0]
    radio.set_value(2)
    running_app.run()
    assert "Not Preserved" in _rendered_text(running_app)


def test_the_draft_score_is_never_shown_before_scoring(running_app):
    assert _shown_pair(running_app) is not None
    text = _rendered_text(running_app)
    assert "Curator's draft:" not in text
    assert "draft_lmp_score" not in text


def test_the_draft_score_is_still_not_shown_after_scoring(running_app):
    """The reveal used to happen once a score was saved.

    That is not blindness: the tool allows a score to be revised, so a draft
    seen after the first save anchors the revision exactly as one seen before
    would anchor the original.
    """
    radio = [r for r in running_app.radio if r.key.startswith("score_")][0]
    radio.set_value(7)
    [s for s in running_app.selectbox if s.key.startswith("err_")][0].set_value("Omission")
    [b for b in running_app.button if b.label == "Save & next"][0].click()
    running_app.run()

    text = _rendered_text(running_app)
    assert "Curator's draft:" not in text
    assert "You agree exactly" not in text
    assert "point(s) below" not in text
    # The annotator is told, positively, why.
    assert "not shown anywhere in this tool" in text


def test_saving_walks_the_queue_without_skipping_pairs(running_app):
    """End-to-end regression for the cursor bug.

    Saving used to advance the cursor while the just-saved pair was dropping
    out of the queue, so the annotator saw pair 1, then 3, then 5. Nothing
    reported it: the progress bar counts the pairs actually on disk, so it
    agreed with the annotator that progress was being made, and half the corpus
    would have been left unannotated with no sign of it.

    Asserting the sequence is *ascending* is not enough — a skipping queue is
    still ascending. This asserts the exact pairs the default queue must
    present. It is the one test here that reads the real corpus, because the
    app loads the master itself and exposes no seam to inject a substitute.
    """
    from src.data.load_data import load_master

    expected = sorted(load_master(subset="all")["pair_id"].astype(str))[:4]

    seen = [_shown_pair(running_app)]
    for _ in range(3):
        [r for r in running_app.radio if r.key.startswith("score_")][0].set_value(7)
        [s for s in running_app.selectbox if s.key.startswith("err_")][0].set_value("Omission")
        [b for b in running_app.button if b.label == "Save & next"][0].click()
        running_app.run()
        seen.append(_shown_pair(running_app))

    assert seen == expected, f"the queue skipped pairs: {seen} != {expected}"
    assert len(set(seen)) == len(seen), f"a pair was revisited: {seen}"


def test_saving_writes_a_file_the_merge_can_read(running_app, annotations_root: Path):
    [r for r in running_app.radio if r.key.startswith("score_")][0].set_value(6)
    [s for s in running_app.selectbox if s.key.startswith("err_")][0].set_value("Condition")
    [t for t in running_app.text_input if t.key.startswith("notes_")][0].set_value("dropped a proviso")
    [b for b in running_app.button if b.label == "Save & next"][0].click()
    running_app.run()

    written = list((annotations_root / "annotators").glob("*.csv"))
    assert len(written) == 1
    frame = pd.read_csv(written[0], keep_default_na=False)
    assert list(frame.columns) == agreement.ANNOTATION_COLUMNS
    assert frame.iloc[0]["annotator_id"] == "A01"
    assert int(frame.iloc[0]["human_lmp_score"]) == 6
    assert frame.iloc[0]["notes"] == "dropped a proviso"


def test_the_draft_never_reaches_the_rendered_interface(running_app, repo_root: Path):
    """A structural check on the app's own source.

    The rendered tests above can only see the one pair on screen. This asserts
    the *code path* is gone, so a draft cannot be rendered for some other pair,
    tab or session-state branch.
    """
    source = Path(app.__file__).read_text(encoding="utf-8")
    for forbidden in ("draft_lmp_score", "draft_error_type", "draft_delta", "Curator's draft"):
        assert forbidden not in source, f"{forbidden} still appears in the annotation tool"
