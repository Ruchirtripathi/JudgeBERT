#!/usr/bin/env python3
"""Tests for the Ollama pilot annotation script.

These tests use ``pytest`` and ``monkeypatch`` to mock the HTTP calls to the
local Ollama server.  No real network traffic is performed.

The test suite covers:
1. Successful parsing of a valid JSON response.
2. Detection of malformed JSON.
3. Proper propagation of a request timeout.
4. Checkpoint‑recovery – a pair that already exists in the output CSV is
   skipped.
"""

import json
import pathlib
import sys
import requests
from types import SimpleNamespace

import pandas as pd
import pytest

# Add the project root to ``sys.path`` so we can import the pilot module.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.annotation import ai_annotation_ollama_pilot as pilot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dummy_row():
    """Return a minimal pandas Series representing a pair.
    Only the fields accessed by the pilot script are needed.
    """
    data = {
        "pair_id": "TEST_001",
        "language": "en",
        "legal_domain": "contract",
        "original_text": "Original sentence.",
        "simplified_text": "Simplified sentence.",
        "error_type": "Scope",
    }
    return pd.Series(data)

# ---------------------------------------------------------------------------
# Test 1 – valid JSON response
# ---------------------------------------------------------------------------
def test_parse_valid_json():
    valid_json = json.dumps(
        {
            "ai_lmp_score": 7,
            "ai_error_type": "Scope",
            "ai_explanation": "Threshold removed.",
            "ai_confidence": 0.92,
            "model_name": "qwen3:4b",
        }
    )
    parsed = pilot.parse_and_validate(valid_json)
    assert parsed["ai_lmp_score"] == 7
    assert parsed["ai_error_type"] == "Scope"
    assert isinstance(parsed["ai_confidence"], float)

# ---------------------------------------------------------------------------
# Test 2 – malformed JSON should raise ValueError
# ---------------------------------------------------------------------------
def test_parse_malformed_json():
    bad_json = "{ 'not': 'valid' }"  # single quotes are invalid JSON
    with pytest.raises(ValueError, match="Invalid JSON"):
        pilot.parse_and_validate(bad_json)

# ---------------------------------------------------------------------------
# Test 3 – HTTP timeout handling
# ---------------------------------------------------------------------------
def test_ollama_timeout(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.Timeout("simulated timeout")

    # Replace the ``requests`` module inside the pilot with a SimpleNamespace that
    # contains the fake ``post`` function and the ``Timeout`` exception class.
    monkeypatch.setattr(pilot, "requests", SimpleNamespace(post=fake_post, Timeout=requests.Timeout))
    with pytest.raises(requests.Timeout):
        pilot.call_ollama("dummy prompt")

# ---------------------------------------------------------------------------
# Test 4 – checkpoint recovery (skip already processed pair)
# ---------------------------------------------------------------------------
def test_checkpoint_skip(tmp_path, monkeypatch):
    # Create a temporary output CSV containing the dummy pair_id
    out_csv = tmp_path / "ai_annotations.csv"
    df = pd.DataFrame(
        [
            {
                "pair_id": "TEST_001",
                "language": "en",
                "legal_domain": "contract",
                "ai_lmp_score": 8,
                "ai_error_type": "None",
                "ai_explanation": "perfect",
                "ai_confidence": 1.0,
                "model_name": "qwen3:4b",
            }
        ]
    )
    df.to_csv(out_csv, index=False)

    # Patch the module constant to point at the temporary CSV
    monkeypatch.setattr(pilot, "OUTPUT_CSV", out_csv)

    # Monkeypatch ``call_ollama`` to ensure it would NOT be called if skipping works
    called = {"count": 0}

    def fake_call(prompt):
        called["count"] += 1
        return json.dumps(
            {
                "ai_lmp_score": 5,
                "ai_error_type": "Scope",
                "ai_explanation": "test",
                "ai_confidence": 0.5,
                "model_name": "qwen3:4b",
            }
        )

    monkeypatch.setattr(pilot, "call_ollama", fake_call)

    # Process the row – should be skipped and ``call_ollama`` never invoked
    pilot.process_pair(dummy_row())
    assert called["count"] == 0, "Ollama should not be called for already processed pair"

"""End of test suite"""
