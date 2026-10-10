#!/usr/bin/env python3
"""Regression tests for the Ollama pilot checkpoint logic.

These tests verify the exact cause of the previous "Running pilot on 0 pairs"
behaviour and ensure that:
1. A fully valid checkpoint row is recognised as completed and the pilot
   reports that it is already complete.
2. A row with empty/invalid fields is *not* considered completed, causing the
   pilot to process the pair.
3. The pilot selects five eligible pairs *independently* of any existing
   checkpoint state.
"""

import io
import pathlib
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

# Make the project root importable
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.annotation import ai_annotation_ollama_pilot as pilot

# Helper to create a minimal pair DataFrame for testing
def make_pairs(pair_ids):
    rows = []
    for pid in pair_ids:
        rows.append({
            "pair_id": pid,
            "language": "en",
            "legal_domain": "contract",
            "original_text": "Orig",
            "simplified_text": "Simpl",
            "error_type": "Scope",
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Test 1 – pilot reports already complete when all selected rows are valid
# ---------------------------------------------------------------------------
def test_pilot_already_complete(monkeypatch, capsys, tmp_path):
    # Force selection to exactly two known pairs
    selected = make_pairs(["A", "B"])
    monkeypatch.setattr(pilot, "select_five_pairs", lambda df: selected)

    # Create a valid checkpoint CSV containing those two rows with proper data
    out_csv = tmp_path / "ai_annotations.csv"
    valid_rows = pd.DataFrame([
        {
            "pair_id": "A",
            "language": "en",
            "legal_domain": "contract",
            "ai_lmp_score": 8,
            "ai_error_type": "Scope",
            "ai_explanation": "ok",
            "ai_confidence": 0.9,
            "model_name": "qwen3:4b",
        },
        {
            "pair_id": "B",
            "language": "en",
            "legal_domain": "contract",
            "ai_lmp_score": 7,
            "ai_error_type": "None",
            "ai_explanation": "perfect",
            "ai_confidence": 1.0,
            "model_name": "qwen3:4b",
        },
    ])
    valid_rows.to_csv(out_csv, index=False)
    monkeypatch.setattr(pilot, "OUTPUT_CSV", out_csv)

    # Run main – it should detect completion and print the specific message
    pilot.main()
    captured = capsys.readouterr()
    assert "Pilot already complete" in captured.out

# ---------------------------------------------------------------------------
# Test 2 – incomplete (empty) rows are treated as pending
# ---------------------------------------------------------------------------
def test_pilot_pending_due_to_empty_fields(monkeypatch, capsys, tmp_path):
    selected = make_pairs(["C", "D"])
    monkeypatch.setattr(pilot, "select_five_pairs", lambda df: selected)

    # Create a checkpoint CSV where one row is fully valid and the other is empty
    out_csv = tmp_path / "ai_annotations.csv"
    mixed_rows = pd.DataFrame([
        {
            "pair_id": "C",
            "language": "en",
            "legal_domain": "contract",
            "ai_lmp_score": 9,
            "ai_error_type": "Scope",
            "ai_explanation": "good",
            "ai_confidence": 0.95,
            "model_name": "qwen3:4b",
        },
        {
            "pair_id": "D",
            "language": "en",
            "legal_domain": "contract",
            "ai_lmp_score": None,
            "ai_error_type": None,
            "ai_explanation": None,
            "ai_confidence": None,
            "model_name": "qwen3:4b",
        },
    ])
    mixed_rows.to_csv(out_csv, index=False)
    monkeypatch.setattr(pilot, "OUTPUT_CSV", out_csv)

    # Monkeypatch ``process_pair`` to avoid real network calls – just record calls
    called = []
    def fake_process(row):
        called.append(row["pair_id"])
    monkeypatch.setattr(pilot, "process_pair", fake_process)

    pilot.main()
    captured = capsys.readouterr()
    # Should report running pilot on 1 pair (the incomplete one)
    assert "Running pilot on 1 pairs" in captured.out
    # Ensure the incomplete pair "D" was processed, not the completed "C"
    assert called == ["D"]

# ---------------------------------------------------------------------------
# Test 3 – selection of five pairs works regardless of existing checkpoint rows
# ---------------------------------------------------------------------------
def test_select_five_independent_of_checkpoint(monkeypatch):
    # Load the real CSV (small subset is fine) but we will monkeypatch the
    # OUTPUT_CSV to point to a non‑existent file so that ``should_skip`` always
    # returns False.
    dummy_out = pathlib.Path("nonexistent.csv")
    monkeypatch.setattr(pilot, "OUTPUT_CSV", dummy_out)

    # Run select_five_pairs on the actual dataset
    df = pilot.load_pairs()
    selected = pilot.select_five_pairs(df)
    assert len(selected) == 5
    # Ensure we have at least two languages represented
    langs = set(selected["language"].unique())
    assert "en" in langs
    assert "hi" in langs

"""End of regression test suite"""
