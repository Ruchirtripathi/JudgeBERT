#!/usr/bin/env python3
"""Regression tests for the annotation checkpointing safety fixes (Phase 4).

Covers:
1. Valid checkpoint is skipped (no overwrite).
2. Blank/invalid annotation is retried.
3. Failed retry does NOT destroy a prior valid annotation (record_failure guard).
4. Successful upsert replaces an invalid row without creating duplicates.
5. Pair IDs remain unique after multiple writes.
6. CSV read/write round-trip preserves valid labels.
7. Invalid model JSON is handled safely (no false success).
8. A failed request does not result in a false success message.
9. Interrupted/empty model output is detected before writing.
10. record_failure guard emits the correct warning.
"""

import io
import json
import pathlib
import sys

import pandas as pd
import pytest
import requests

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation import ai_annotation_ollama_pilot as pilot


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_row(pair_id: str = "TEST_001") -> pd.Series:
    return pd.Series({
        "pair_id": pair_id,
        "language": "en",
        "legal_domain": "contract",
        "original_text": "The agent shall perform all duties.",
        "simplified_text": "The agent must do all tasks.",
    })


def valid_csv_row(pair_id: str = "TEST_001") -> dict:
    return {
        "pair_id": pair_id,
        "language": "en",
        "legal_domain": "contract",
        "ai_lmp_score": 8,
        "ai_error_type": "Scope",
        "ai_explanation": "Meaning is well preserved.",
        "ai_confidence": 0.9,
        "model_name": "qwen3:4b",
    }


def write_csv(path: pathlib.Path, rows: list) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Test 1 — Valid checkpoint is skipped; call_ollama is never invoked
# ---------------------------------------------------------------------------
def test_valid_checkpoint_is_skipped(tmp_path, monkeypatch):
    out_csv = tmp_path / "ai_annotations.csv"
    write_csv(out_csv, [valid_csv_row()])
    monkeypatch.setattr(pilot, "OUTPUT_CSV", out_csv)

    ollama_called = []
    monkeypatch.setattr(pilot, "call_ollama", lambda p: ollama_called.append(1))

    row = make_row()
    assert pilot.is_successful_checkpoint(row) is True
    pilot.process_pair(row)
    # call_ollama must NOT have been invoked
    assert ollama_called == [], "call_ollama was invoked despite valid checkpoint"


# ---------------------------------------------------------------------------
# Test 2 — Blank annotation row is retried (not considered valid)
# ---------------------------------------------------------------------------
def test_blank_annotation_triggers_retry(tmp_path, monkeypatch):
    out_csv = tmp_path / "ai_annotations.csv"
    write_csv(out_csv, [{
        "pair_id": "TEST_001",
        "language": "en",
        "legal_domain": "contract",
        "ai_lmp_score": None,
        "ai_error_type": None,
        "ai_explanation": None,
        "ai_confidence": None,
        "model_name": "qwen3:4b",
    }])
    monkeypatch.setattr(pilot, "OUTPUT_CSV", out_csv)

    row = make_row()
    # A blank record must NOT be considered a successful checkpoint
    assert pilot.is_successful_checkpoint(row) is False


# ---------------------------------------------------------------------------
# Test 3 — record_failure does NOT destroy a prior valid annotation
# ---------------------------------------------------------------------------
def test_record_failure_guards_valid_annotation(tmp_path, monkeypatch, capsys):
    out_csv = tmp_path / "ai_annotations.csv"
    write_csv(out_csv, [valid_csv_row()])
    monkeypatch.setattr(pilot, "OUTPUT_CSV", out_csv)

    row = make_row()
    # Attempt to record a failure on a pair that already has a valid annotation
    pilot.record_failure(row)

    # The CSV must still contain the original valid row, not a null row
    after = pd.read_csv(out_csv, keep_default_na=False)
    assert len(after) == 1
    assert str(after.iloc[0]["ai_lmp_score"]) == "8"
    assert after.iloc[0]["ai_error_type"] == "Scope"
    assert after.iloc[0]["ai_explanation"] == "Meaning is well preserved."

    # A warning must have been printed to stderr
    captured = capsys.readouterr()
    assert "guard" in captured.err or "refused" in captured.err


# ---------------------------------------------------------------------------
# Test 4 — Successful write replaces invalid row without creating duplicates
# ---------------------------------------------------------------------------
def test_write_result_replaces_invalid_without_duplicates(tmp_path, monkeypatch):
    out_csv = tmp_path / "ai_annotations.csv"
    # Start with an invalid (failure) record
    write_csv(out_csv, [{
        "pair_id": "TEST_001",
        "language": "en",
        "legal_domain": "contract",
        "ai_lmp_score": None,
        "ai_error_type": None,
        "ai_explanation": None,
        "ai_confidence": None,
        "model_name": "qwen3:4b",
    }])
    monkeypatch.setattr(pilot, "OUTPUT_CSV", out_csv)

    row = make_row()
    prediction = {
        "ai_lmp_score": 7,
        "ai_error_type": "Omission",
        "ai_explanation": "A clause was dropped.",
        "ai_confidence": 0.85,
        "model_name": "qwen3:4b",
    }
    pilot.write_result(row, prediction)

    after = pd.read_csv(out_csv, keep_default_na=False)
    # Exactly one row — no duplicate
    assert len(after) == 1
    assert after.iloc[0]["ai_lmp_score"] in (7, "7", 7.0)
    assert after.iloc[0]["ai_error_type"] == "Omission"


# ---------------------------------------------------------------------------
# Test 5 — Pair IDs remain unique after multiple writes
# ---------------------------------------------------------------------------
def test_pair_ids_unique_after_multiple_writes(tmp_path, monkeypatch):
    out_csv = tmp_path / "ai_annotations.csv"
    monkeypatch.setattr(pilot, "OUTPUT_CSV", out_csv)

    for attempt in range(3):
        prediction = {
            "ai_lmp_score": 5 + attempt,
            "ai_error_type": "None",
            "ai_explanation": f"Attempt {attempt}",
            "ai_confidence": 0.8,
            "model_name": "qwen3:4b",
        }
        pilot.write_result(make_row(), prediction)

    after = pd.read_csv(out_csv)
    assert after["pair_id"].nunique() == 1
    assert len(after) == 1


# ---------------------------------------------------------------------------
# Test 6 — CSV round-trip preserves valid labels exactly
# ---------------------------------------------------------------------------
def test_csv_round_trip_preserves_labels(tmp_path, monkeypatch):
    out_csv = tmp_path / "ai_annotations.csv"
    monkeypatch.setattr(pilot, "OUTPUT_CSV", out_csv)

    original_prediction = {
        "ai_lmp_score": 9,
        "ai_error_type": "Obligation/Permission",
        "ai_explanation": "Modal verb changed from 'shall' to 'may'.",
        "ai_confidence": 0.95,
        "model_name": "qwen3:4b",
    }
    pilot.write_result(make_row(), original_prediction)

    # Re-read and check checkpoint recognition
    row = make_row()
    assert pilot.is_successful_checkpoint(row) is True

    after = pd.read_csv(out_csv, keep_default_na=False)
    rec = after[after["pair_id"] == "TEST_001"].iloc[0]
    assert rec["ai_error_type"] == "Obligation/Permission"
    assert rec["ai_explanation"] == "Modal verb changed from 'shall' to 'may'."


# ---------------------------------------------------------------------------
# Test 7 — Invalid model JSON is rejected by parse_and_validate
# ---------------------------------------------------------------------------
def test_invalid_model_json_rejected():
    bad_jsons = [
        "{}",                        # missing all required keys
        '{"ai_lmp_score": 0}',       # score out of range
        '{"ai_lmp_score": 5, "ai_error_type": "UNKNOWN_LABEL", '
        '"ai_explanation": "x", "ai_confidence": 0.5, "model_name": "m"}',
        '{"ai_lmp_score": 5, "ai_error_type": "Scope", '
        '"ai_explanation": "", "ai_confidence": 0.5, "model_name": "m"}',  # empty explanation
    ]
    for bad in bad_jsons:
        with pytest.raises(ValueError):
            pilot.parse_and_validate(bad)


# ---------------------------------------------------------------------------
# Test 8 — A failed request does not produce a false success message
# ---------------------------------------------------------------------------
def test_failed_request_no_false_success(tmp_path, monkeypatch, capsys):
    out_csv = tmp_path / "ai_annotations.csv"
    monkeypatch.setattr(pilot, "OUTPUT_CSV", out_csv)

    # Simulate call_ollama always raising (e.g., connection error)
    def always_fail(prompt):
        raise requests.ConnectionError("simulated failure")

    monkeypatch.setattr(pilot, "call_ollama", always_fail)

    row = make_row()
    pilot.process_pair(row)

    captured = capsys.readouterr()
    # Must report failure, not success
    assert "[ok]" not in captured.out
    assert "[failed]" in captured.err or "failed" in captured.err

    # The CSV should contain a failure record (all nulls for this pair)
    if out_csv.exists():
        after = pd.read_csv(out_csv, keep_default_na=False)
        if len(after) > 0:
            rec = after[after["pair_id"] == "TEST_001"]
            if len(rec) > 0:
                # Score should be empty/null for a failure record
                assert str(rec.iloc[0]["ai_lmp_score"]) in ("", "nan", "None", "NaN")


# ---------------------------------------------------------------------------
# Test 9 — Interrupted/empty model output is detected before writing
# ---------------------------------------------------------------------------
def test_empty_ollama_response_raises_before_write(tmp_path, monkeypatch):
    out_csv = tmp_path / "ai_annotations.csv"
    monkeypatch.setattr(pilot, "OUTPUT_CSV", out_csv)

    # Simulate Ollama returning an empty response field
    from types import SimpleNamespace

    def fake_post(*args, **kwargs):
        resp = SimpleNamespace()
        resp.status_code = 200
        resp.raise_for_status = lambda: None
        resp.json = lambda: {"response": "", "model": "qwen3:4b"}
        return resp

    monkeypatch.setattr(pilot.requests, "post", fake_post)

    with pytest.raises(ValueError, match="Empty Ollama response"):
        pilot.call_ollama("dummy prompt")

    # No output file should have been written
    assert not out_csv.exists() or pd.read_csv(out_csv).empty


# ---------------------------------------------------------------------------
# Test 10 — record_failure on a new pair (no prior record) writes the failure row
# ---------------------------------------------------------------------------
def test_record_failure_on_new_pair_writes_failure(tmp_path, monkeypatch):
    out_csv = tmp_path / "ai_annotations.csv"
    monkeypatch.setattr(pilot, "OUTPUT_CSV", out_csv)

    row = make_row("NEW_001")
    pilot.record_failure(row)

    after = pd.read_csv(out_csv, keep_default_na=False)
    assert len(after) == 1
    assert after.iloc[0]["pair_id"] == "NEW_001"
    assert str(after.iloc[0]["ai_lmp_score"]) in ("", "nan", "None", "NaN")


# ---------------------------------------------------------------------------
# Test 11 — Failed retry preserves an existing valid annotation
# ---------------------------------------------------------------------------
def test_failed_retry_preserves_existing_valid_annotation(
    tmp_path, monkeypatch, capsys
):
    out_csv = tmp_path / "ai_annotations.csv"
    write_csv(out_csv, [valid_csv_row("TEST_001")])
    monkeypatch.setattr(pilot, "OUTPUT_CSV", out_csv)
    monkeypatch.setattr(pilot.time, "sleep", lambda _: None)

    def always_fail(prompt):
        raise requests.ConnectionError("simulated Ollama failure")

    monkeypatch.setattr(pilot, "call_ollama", always_fail)

    pilot.process_pair(make_row("TEST_001"))

    after = pd.read_csv(out_csv, keep_default_na=False)
    assert len(after) == 1
    assert after.iloc[0]["pair_id"] == "TEST_001"
    assert str(after.iloc[0]["ai_lmp_score"]) == "8"
    assert after.iloc[0]["ai_error_type"] == "Scope"
    assert after.iloc[0]["ai_explanation"] == "Meaning is well preserved."

    captured = capsys.readouterr()
    assert "[skip] TEST_001 already completed" in captured.out
    assert "[failed]" not in captured.err


# ---------------------------------------------------------------------------
# Test 12 — Prompt example must not anchor the model to the minimum score
# ---------------------------------------------------------------------------
def test_build_prompt_uses_neutral_score_example():
    prompt = pilot.build_prompt(make_row("PROMPT_TEST"))

    assert '"score": 5' in prompt
    assert '"score": 1' not in prompt
    assert "score must be an integer from 1 to 10" in prompt
    assert "explanation must be a non-empty string" in prompt
