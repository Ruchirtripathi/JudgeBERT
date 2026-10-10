"""Filesystem IO helpers with provenance stamping.

Centralising reads/writes here means every artefact produced by the project
carries a header recording how it was produced (timestamp, config fingerprint,
git revision), which is what makes the results directory auditable.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

#: JSON-friendly placeholder for a value that must come from a real run.
TODO_RUN_EXPERIMENT = "TODO_RUN_EXPERIMENT"
TODO_HUMAN_ANNOTATION = "TODO_HUMAN_ANNOTATION"
RESULT_PENDING = "RESULT_PENDING"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a trailing ``Z``."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_revision(short: bool = True) -> str:
    """Return the current git commit hash, or ``"unknown"`` outside a repo.

    Never raises: a missing git binary or a detached tarball checkout should
    not break an experiment.
    """
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[2],
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def provenance(cfg: Mapping[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    """Build a provenance block to embed in every results file."""
    from src.utils.config import config_fingerprint

    block: dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "git_revision": git_revision(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if cfg is not None:
        block["config_fingerprint"] = config_fingerprint(cfg)
        block["project_version"] = (cfg.get("project") or {}).get("version", "unknown")
        block["seed"] = (cfg.get("project") or {}).get("seed")
    block.update(extra)
    return block


# ---------------------------------------------------------------------------
# DataFrames
# ---------------------------------------------------------------------------
def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV as UTF-8, treating only genuinely empty cells as missing.

    ``keep_default_na=False`` with ``na_values=[""]`` is deliberate and matters
    more than it looks.  Pandas' default NA list includes the *literal strings*
    ``"None"``, ``"NA"``, ``"N/A"``, ``"null"`` and ``"nan"``.  ``"None"`` is a
    legitimate value in this corpus — it is the error taxonomy's label for "no
    legal error was introduced" — so the default would silently turn it into a
    missing value, making "no error" indistinguishable from "not yet judged".
    That is exactly the distinction the annotation lifecycle exists to keep.

    Under these settings an empty cell is ``NaN`` (and therefore detectable),
    while any non-empty text survives verbatim.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("keep_default_na", False)
    kwargs.setdefault("na_values", [""])
    df = pd.read_csv(path, **kwargs)
    logger.debug("Read %d rows x %d cols from %s", len(df), df.shape[1], path)
    return df


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    """Write a DataFrame to CSV as UTF-8 *without* the index.

    ``lineterminator="\\n"`` keeps diffs clean across platforms.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("index", False)
    kwargs.setdefault("lineterminator", "\n")
    df.to_csv(path, **kwargs)
    logger.info("Wrote %d rows -> %s", len(df), path)
    return path


# ---------------------------------------------------------------------------
# JSON / JSONL
# ---------------------------------------------------------------------------
class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy/pandas scalar types."""

    def default(self, o: Any) -> Any:
        try:
            import numpy as np

            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                value = float(o)
                return value if value == value else None  # NaN -> null
            if isinstance(o, (np.bool_,)):
                return bool(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
        except ImportError:
            pass
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        return super().default(o)


def write_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """Write JSON with numpy/pandas/Path coercion and ``ensure_ascii=False``.

    ``ensure_ascii=False`` keeps Devanagari readable in the output rather than
    escaping it to ``\\uXXXX``, which makes the results files inspectable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False, cls=_NumpyEncoder)
        fh.write("\n")
    logger.info("Wrote JSON -> %s", path)
    return path


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Write a list of dicts as newline-delimited JSON (one object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, cls=_NumpyEncoder))
            fh.write("\n")
    logger.info("Wrote %d JSONL records -> %s", len(records), path)
    return path


# ---------------------------------------------------------------------------
# Markdown tables
# ---------------------------------------------------------------------------
def dataframe_to_markdown(df: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table.

    Implemented directly rather than via ``tabulate`` to avoid a dependency
    and to guarantee the exact formatting used in the reports.
    """
    if df.empty:
        return "_(no rows)_\n"

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            if value != value:  # NaN
                return "n/a"
            return float_format.format(value)
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows]) + "\n"


def append_markdown(text: str, path: str | Path) -> Path:
    """Append a block of Markdown to a report file, creating it if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")
    return path
