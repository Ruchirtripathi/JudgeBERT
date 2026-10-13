# Stage 1 — Ollama Annotation Pipeline Diagnostics

**Audit Date:** 2026-10-13
**Scope:** Phase 3 — Diagnose the `Invalid inner JSON from Ollama` failure without retrying.

---

## 1. Current Connectivity Status

**Ollama server is NOT running.** Attempting `http://localhost:11434/api/tags` returns a connection refused error.

The `ollama list` command also returns:
```
Error: could not connect to ollama server, run 'ollama serve' to start it
```

The `qwen3:4b` model availability cannot be confirmed without a running server.

---

## 2. Root Cause Analysis of the Historical Failure

The reported exception was:
```
Invalid inner JSON from Ollama: Expecting value: line 1 column 1 (char 0)
```

This error message originates from `call_ollama()` in `ai_annotation_ollama_pilot.py` at this line:
```python
inner = json.loads(outer)
```

Where `outer = data.get("response")`.

The error `Expecting value: line 1 column 1 (char 0)` means `json.loads` received an **empty string `""`** as input.

### Possible causes (ranked by probability)

| # | Cause | Evidence | How to distinguish |
|---|---|---|---|
| 1 | **Ollama server was not running** — HTTP connection succeeded to a stale server process, which returned an empty body or a 503, causing `response.json()` to return `{"response": ""}` | Most likely given current state; the server is confirmed not running | Check `response.status_code` before `response.json()` |
| 2 | **Model was still loading** — Ollama accepted the request but the model was not ready, returning an empty `response` field | Plausible on first call after cold start | The `call_ollama` function does raise on HTTP errors but an empty `response` field would not raise `raise_for_status()` |
| 3 | **qwen3:4b thinking mode produced output in `thinking` field only** — with `"think": False` disabled, but if the model ignored the flag, the `response` field might be empty while content went to `thinking` | The code now reads `data.get("thinking")` and logs its length | Check debug output: if `thinking length > 0` and `response = ""`, this is the cause |
| 4 | **Output token limit too small** — `num_predict: 200` (original) truncated the JSON mid-output, causing an empty or malformed `response` field | The limit was raised to 512 in the current (uncommitted) version | Compare `num_predict: 200` vs `num_predict: 512` behaviour |
| 5 | **Network/proxy issue** — unlikely on localhost | No evidence |  |

**Most probable root cause:** The Ollama server was not running or not ready when the annotation script was invoked, resulting in an empty `response` field. The error message misleadingly describes this as a JSON parse failure rather than a connectivity failure.

**Secondary contributing factor:** The original `num_predict: 200` token limit may have been insufficient for some legal sentences with long explanations, potentially causing truncated JSON on longer pairs.

---

## 3. Code-Level Findings

### 3.1 Dead code block in `call_ollama()` (lines 159–191)

The function currently has this structure:

```python
def call_ollama(prompt: str) -> str:
    # ... (active code) ...
    return json.dumps(mapped)   # <-- function returns HERE (line 158)
    """Call the Ollama generate endpoint ...   # <-- DEAD: unreachable docstring
    ...
    return json.dumps(mapped)   # <-- DEAD: second return (line 191)
```

Lines 159–191 are **unreachable dead code** — a duplicate of the original function body that was never removed during the refactor. This is a clarity and maintenance risk but does not affect current runtime behaviour.

### 3.2 `record_failure()` safety defect

```python
def record_failure(row: pd.Series) -> None:
    # ...
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        existing = existing[existing["pair_id"] != failure["pair_id"]]  # <-- removes ANY existing row
        new_df = pd.concat([existing, pd.DataFrame([failure])], ignore_index=True)
```

`record_failure()` removes the row for `pair_id` unconditionally, **even if that row contains a valid previous annotation**. There is no guard calling `is_successful_checkpoint()` before the deletion.

In `process_pair()`, `should_skip()` is called at the top of the function, so a pair with an existing valid annotation will be skipped on a re-run. However, if `process_pair()` is invoked directly (bypassing `should_skip`), or if `should_skip` returns `False` due to a partial/invalid row, `record_failure()` will destroy the existing row after all retries fail.

### 3.3 `is_successful_checkpoint()` — correctness

The function correctly validates:
- Score is integer 1–10
- Error type is in `ALLOWED_ERRORS`
- Explanation is non-empty string
- Confidence is float 0–1
- Model name is non-empty string

It uses `keep_default_na=False` to avoid treating empty strings as NaN — correct.

However: it reads the CSV from disk on every call. In a batch run of 340 pairs, this means 340 full CSV reads. This is acceptable for a 315-row file but will become slow at scale.

### 3.4 Empty-body detection

The current code checks:
```python
if outer is None or (isinstance(outer, str) and outer.strip() == ""):
    raise ValueError("Empty Ollama response...")
```

This is correct and would catch an empty `response` field. However, this guard was added in the uncommitted revision. The original committed version did not have it.

---

## 4. Diagnostic Plan (to be executed when Ollama is running)

Execute the following steps **one at a time** to confirm Ollama connectivity and model response before any annotation run:

```bash
# Step 1: Start Ollama
ollama serve &

# Step 2: Confirm model is available
ollama list

# Step 3: Send a minimal connectivity test (not a full annotation prompt)
curl -s -X POST http://localhost:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3:4b","prompt":"Return JSON: {\"ok\": true}","stream":false,"format":"json","think":false,"options":{"temperature":0,"num_predict":32}}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('response field:', repr(d.get('response'))); print('thinking len:', len(d.get('thinking','')))"

# Step 4: Inspect result
# Expected: response field contains '{"ok": true}' or similar
# If response is empty string: model is not responding correctly; do not proceed with annotation
```

Do not run a full annotation batch until Step 3 confirms a non-empty, valid JSON response.

---

## 5. Recommended Fixes (Phase 4)

1. Remove dead code (lines 159–191 of `call_ollama()`).
2. Guard `record_failure()` with `is_successful_checkpoint()`.
3. Add a pre-flight connectivity check function that fails fast before any annotation loop.
4. Ensure the 3-null failure records in `ai_annotations.csv` are investigated before any repair run.

---

*No files modified. Ollama diagnostics completed offline.*


## 6. Current-State Reconciliation — 2026-10-13

This addendum supersedes conflicting historical code findings above.

- The working revision removes the previously reported unreachable duplicate code in `call_ollama()`.
- The working revision adds a guard in `record_failure()` to avoid replacing an already valid stored checkpoint.
- Empty Ollama response detection and a preflight connectivity check are present in the working revision.
- The focused annotation safety tests and full test suite pass in the current working tree. Latest recorded full-suite result: **358 passed, 2 deprecation warnings**.
- Ollama is currently unavailable. No new live annotation run was verified during this reconciliation.
- The cause of the historical empty-JSON failure remains **unconfirmed**; the current connectivity failure is not proof of its historical cause.

The code changes remain uncommitted. Re-run tests after any further changes.
