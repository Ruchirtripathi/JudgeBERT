#!/usr/bin/env python3
"""AI‑assisted legal meaning annotation pilot using Ollama (qwen3:4b).

This script reads five representative sentence pairs from
`data/processed/lmp_pairs.csv`, sends a JSON‑only prompt to the local Ollama
HTTP API, validates the response, and writes the predictions to
`data/processed/ai_annotations.csv`.  It includes robust timeout handling,
checkpointing, and deterministic settings (temperature=0, JSON format).

Only the five‑pair pilot is executed – the full dataset is not processed.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration -------------------------------------------------------------
# ---------------------------------------------------------------------------
# Resolve paths relative to the repository root, irrespective of the current
# working directory. This makes the script robust in tests where the CWD may be
# a temporary directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data/processed/lmp_pairs.csv"
OUTPUT_CSV = PROJECT_ROOT / "data/processed/ai_annotations.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:4b"
# Increased HTTP read timeout (in seconds)
HTTP_TIMEOUT = 180
# Maximum number of tokens we allow Ollama to emit – conservative
MAX_TOKENS = 512
# Temperature set to 0 for reproducibility
TEMPERATURE = 0

ALLOWED_ERRORS = [
    "None",
    "Omission",
    "Hallucination",
    "Contradiction",
    "Obligation/Permission",
    "Condition",
    "Exception",
    "Scope",
    "Date",
    "Number",
    "Entity",
    "Other",
]

# ---------------------------------------------------------------------------
# Helper functions ----------------------------------------------------------
# ---------------------------------------------------------------------------

def load_pairs() -> pd.DataFrame:
    """Load the source CSV and keep only columns needed for the prompt."""
    df = pd.read_csv(INPUT_CSV)
    required = ["pair_id", "language", "legal_domain", "original_text", "simplified_text"]
    missing = set(required) - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {INPUT_CSV}: {missing}")
    return df[required]


def select_five_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Select five diverse pairs.

    The original script relied on an ``error_type`` column that is not present.
    We simply take the first five rows, preferring a mix of English and Hindi
    when possible.
    """
    # Prefer a mix of languages if they exist.
    lang_filtered = df[df["language"].isin(["en", "hi"])]
    if len(lang_filtered) >= 5:
        return lang_filtered.head(5)
    return df.head(5)

def build_prompt(row: pd.Series) -> str:
    """Ask for a complete, schema-compliant legal meaning annotation."""
    return (
        "You are an expert legal analyst evaluating preservation of legal meaning.\n"
        "Compare the original statutory sentence with its simplified rewrite.\n\n"
        "Return exactly one JSON object with these four keys:\n"
        '{"score": 5, "error_type": "None", '
        '"explanation": "Explanation here", "confidence": 0.8}\n\n'
        "Rules:\n"
        "- score must be an integer from 1 to 10.\n"
        f"- error_type MUST be exactly one of: {', '.join(ALLOWED_ERRORS)}.\n"
        '- Use "None" when no meaningful legal error is present.\n'
        "- Never omit error_type, even when the score is high.\n"
        "- explanation must be a non-empty string.\n"
        "- confidence must be a number between 0 and 1.\n"
        "- Do not use null or an empty string for any field.\n"
        "- Output JSON only, without Markdown fences.\n\n"
        f"Original: {row['original_text']}\n"
        f"Simplified: {row['simplified_text']}\n"
    )

def call_ollama(prompt: str) -> str:
    """Call the Ollama generate endpoint with JSON‑only output.
    Parameters are set to disable any internal "thinking" mode and to request JSON output directly.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
        "format": "json",
        "think": False,  # disable Qwen3 thinking mode if supported
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected Ollama response type: {type(data)}; expected dict")
    # Diagnostics (do not dump full prompt)
    print("[debug] Ollama outer keys:", list(data.keys()))
    outer = data.get("response")
    print("[debug] Ollama response field repr:", repr(outer))
    thinking = data.get("thinking", "")
    if isinstance(thinking, str):
        print("[debug] Ollama thinking length:", len(thinking))
    # Empty or whitespace response is an error
    if outer is None or (isinstance(outer, str) and outer.strip() == ""):
        raise ValueError(
            f"Empty Ollama response. Keys present: {list(data.keys())}. "
            f"Thinking field present: {'thinking' in data}."
        )
    try:
        inner = json.loads(outer)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid inner JSON from Ollama: {exc}") from exc
    # Support both possible inner schemas
    if "score" in inner or "error_type" in inner:
        mapped = {
            "ai_lmp_score": inner.get("score"),
            "ai_error_type": inner.get("error_type"),
            "ai_explanation": inner.get("explanation"),
            "ai_confidence": inner.get("confidence"),
            "model_name": data.get("model", MODEL_NAME),
        }
    elif "ai_lmp_score" in inner:
        mapped = {
            "ai_lmp_score": inner.get("ai_lmp_score"),
            "ai_error_type": inner.get("ai_error_type"),
            "ai_explanation": inner.get("ai_explanation"),
            "ai_confidence": inner.get("ai_confidence"),
            "model_name": data.get("model", MODEL_NAME),
        }
    else:
        raise ValueError("Inner JSON does not contain expected score fields")
    return json.dumps(mapped)



def parse_and_validate(json_str: str) -> Dict:
    """Parse the JSON string and enforce schema constraints."""
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    required = ["ai_lmp_score", "ai_error_type", "ai_explanation", "ai_confidence", "model_name"]
    for key in required:
        if key not in result:
            raise ValueError(f"Missing required key: {key}")
    score = result["ai_lmp_score"]
    if not isinstance(score, int) or not (1 <= score <= 10):
        raise ValueError(f"ai_lmp_score must be int 1‑10, got {score}")
    err = result["ai_error_type"]
    if err not in ALLOWED_ERRORS:
        raise ValueError(f"ai_error_type '{err}' not allowed")
    explanation = result["ai_explanation"]
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("ai_explanation must be a non-empty string")
    conf = result["ai_confidence"]
    if not isinstance(conf, (float, int)) or not (0.0 <= float(conf) <= 1.0):
        raise ValueError(f"ai_confidence must be between 0 and 1, got {conf}")
    if not isinstance(result["model_name"], str) or not result["model_name"]:
        raise ValueError("model_name must be a non‑empty string")
    return result


def write_result(row: pd.Series, prediction: Dict) -> None:
    """Append a validated prediction to the output CSV.
    Columns: pair_id, language, legal_domain, ai_lmp_score, ai_error_type,
    ai_explanation, ai_confidence, model_name
    """
    out_row = {
        "pair_id": row["pair_id"],
        "language": row["language"],
        "legal_domain": row["legal_domain"],
        "ai_lmp_score": prediction["ai_lmp_score"],
        "ai_error_type": prediction["ai_error_type"],
        "ai_explanation": prediction["ai_explanation"],
        "ai_confidence": prediction["ai_confidence"],
        "model_name": prediction["model_name"],
    }
    cols = ["pair_id", "language", "legal_domain", "ai_lmp_score", "ai_error_type",
            "ai_explanation", "ai_confidence", "model_name"]
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        existing = existing[existing["pair_id"] != out_row["pair_id"]]
        new_df = pd.concat([existing, pd.DataFrame([out_row])], ignore_index=True)
    else:
        new_df = pd.DataFrame([out_row])
    new_df.to_csv(OUTPUT_CSV, index=False, columns=cols)

def record_failure(row: pd.Series) -> None:
    """Write a failure record (empty fields) only if no valid annotation already exists.

    Safety contract:
    - If a fully valid annotation already exists for this pair_id, this function
      logs a warning and returns without modifying the CSV.
    - A failure record is written only when no valid annotation exists.
    - Guarantees at most one failure entry per pair_id.
    """
    pair_id = row["pair_id"]
    # --- Safety guard: never overwrite a valid annotation ---
    if is_successful_checkpoint(row):
        print(
            f"[guard] record_failure refused to overwrite valid annotation for {pair_id}",
            file=sys.stderr,
        )
        return
    failure = {
        "pair_id": pair_id,
        "language": row["language"],
        "legal_domain": row["legal_domain"],
        "ai_lmp_score": None,
        "ai_error_type": None,
        "ai_explanation": None,
        "ai_confidence": None,
        "model_name": MODEL_NAME,
    }
    cols = ["pair_id", "language", "legal_domain", "ai_lmp_score", "ai_error_type",
            "ai_explanation", "ai_confidence", "model_name"]
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        existing = existing[existing["pair_id"] != pair_id]
        new_df = pd.concat([existing, pd.DataFrame([failure])], ignore_index=True)
    else:
        new_df = pd.DataFrame([failure])
    new_df.to_csv(OUTPUT_CSV, index=False, columns=cols)


def is_successful_checkpoint(row) -> bool:
    """Return True iff a *fully valid* prediction for ``row`` already exists.
    A row is considered successful only when:
    - All required columns are present.
    - ``ai_lmp_score`` is an integer (or numeric string) between 1 and 10.
    - ``ai_error_type`` is one of ``ALLOWED_ERRORS``.
    - ``ai_explanation`` is a non‑empty string.
    - ``ai_confidence`` is a number between 0 and 1 (inclusive).
    - ``model_name`` is a non‑empty string.
    Empty, NaN, ``None`` or whitespace values cause the row to be treated as incomplete.
    """
    # Ensure checkpoint CSV exists
    if not OUTPUT_CSV.exists():
        return False
    # Load CSV without converting empty strings to NaN
    done = pd.read_csv(OUTPUT_CSV, keep_default_na=False)
    # Verify required columns exist
    required_cols = ["pair_id", "ai_lmp_score", "ai_error_type", "ai_explanation", "ai_confidence", "model_name"]
    for col in required_cols:
        if col not in done.columns:
            return False
    pid = str(row["pair_id"]).strip()
    # Find matching record
    match = done[done["pair_id"].astype(str).str.strip() == pid]
    if match.empty:
        return False
    rec = match.iloc[0]

    # Validate score: accept only integer values from 1 through 10.
    try:
        score_val = rec["ai_lmp_score"]
        if isinstance(score_val, str):
            score_val = score_val.strip()
            if not score_val:
                return False
            score_val = float(score_val)

        score = float(score_val)
    except (TypeError, ValueError, OverflowError):
        return False

    if not score.is_integer() or not (1 <= score <= 10):
        return False

    # Validate error_type
    if rec["ai_error_type"] not in ALLOWED_ERRORS:
        return False
    # Validate explanation
    if not isinstance(rec["ai_explanation"], str) or rec["ai_explanation"].strip() == "":
        return False
    # Validate confidence
    try:
        conf_val = rec["ai_confidence"]
        if isinstance(conf_val, str):
            conf_val = float(conf_val)
        conf = float(conf_val)
    except Exception:
        return False
    if not (0.0 <= conf <= 1.0):
        return False
    # Validate model_name
    if not isinstance(rec["model_name"], str) or rec["model_name"].strip() == "":
        return False
    return True






def should_skip(row) -> bool:
    """Legacy wrapper retained for compatibility with existing code/tests."""
    return is_successful_checkpoint(row)


def process_pair(row: pd.Series) -> None:
    """Process a single pair, retrying up to a finite limit (default 3)."""
    MAX_RETRIES = 3
    pair_id = row["pair_id"]
    if should_skip(row):
        print(f"[skip] {pair_id} already completed")
        return
    prompt = build_prompt(row)
    attempt = 0
    while attempt < MAX_RETRIES:
        attempt += 1
        try:
            raw = call_ollama(prompt)
            print(f"[debug] call_ollama returned type={type(raw)} value={raw}")
            prediction = parse_and_validate(raw)
            # Ensure required fields are present and non‑null
            missing = [k for k, v in prediction.items() if v is None]
            if missing:
                raise ValueError(f"Prediction missing fields: {missing}")
            prediction["model_name"] = MODEL_NAME
            write_result(row, prediction)
            print(
                f"[ok] {pair_id} -> score {prediction['ai_lmp_score']}, error {prediction['ai_error_type']} (attempt {attempt})"
            )
            return
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"[retry {attempt}] {pair_id} failed: {exc}",
                file=sys.stderr,
            )
            time.sleep(2)
    # Record failure using upsert logic
    record_failure(row)
    print(f"[failed] {pair_id} recorded with empty fields", file=sys.stderr)


def check_ollama_preflight() -> bool:
    """Perform a minimal connectivity check against the Ollama server.

    Sends a trivial JSON request and verifies that the response field is
    non-empty.  Returns True if Ollama is healthy, False otherwise.
    Errors are printed to stderr; no exception is raised.
    """
    test_payload = {
        "model": MODEL_NAME,
        "prompt": 'Return JSON: {"ok": true}',
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0, "num_predict": 32},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=test_payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        response_field = data.get("response", "")
        if not isinstance(response_field, str) or response_field.strip() == "":
            print(
                f"[preflight] Ollama responded but 'response' field is empty. "
                f"Keys: {list(data.keys())}. "
                f"Thinking present: {'thinking' in data}.",
                file=sys.stderr,
            )
            return False
        print(f"[preflight] Ollama OK. Response sample: {response_field[:80]!r}")
        return True
    except requests.ConnectionError as exc:
        print(f"[preflight] Cannot connect to Ollama at {OLLAMA_URL}: {exc}", file=sys.stderr)
        return False
    except requests.Timeout:
        print("[preflight] Ollama connection timed out (15 s).", file=sys.stderr)
        return False
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[preflight] Ollama preflight failed unexpectedly: {exc}", file=sys.stderr)
        return False


def select_pending_batch(df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    """Select up to batch_size rows that lack a valid annotation checkpoint."""
    pending_rows = []

    for _, row in df.iterrows():
        if not should_skip(row):
            pending_rows.append(row)

        if len(pending_rows) >= batch_size:
            break

    return pd.DataFrame(pending_rows, columns=df.columns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate provisional AI LMP annotations with Ollama."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--all",
        action="store_true",
        help="Process every source pair, skipping successful checkpoints.",
    )
    mode.add_argument(
        "--batch-size",
        type=int,
        help="Process the next N pairs not yet present in the output CSV.",
    )
    args, _ = parser.parse_known_args()

    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    df = load_pairs()

    if args.all:
        candidate_df = df
        label = "full dataset"


    elif args.batch_size is not None:
        candidate_df = select_pending_batch(df, args.batch_size)
        label = "pending-pair batch"

    else:
        candidate_df = select_five_pairs(df)
        label = "pilot"

    pending = [
        row for _, row in candidate_df.iterrows()
        if not should_skip(row)
    ]

    if not pending:
        if label == "pilot":
            print("Pilot already complete; all selected pairs have valid checkpoints.")
        else:
            print(f"No pending pairs for {label}; nothing to process.")
        return

    # --- Pre-flight check: confirm Ollama is reachable before the annotation loop ---
    if not check_ollama_preflight():
        print(
            "[error] Ollama preflight failed. Start Ollama with 'ollama serve' and verify "
            f"that '{MODEL_NAME}' is available via 'ollama list' before retrying.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Running {label} on {len(pending)} pairs...")
    for row in pending:
        process_pair(row)

    print(f"Completed {label}. Results stored in {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
