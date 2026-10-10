"""Merge per-annotator scores into the master corpus's gold columns.

The annotation tool writes one file per annotator under
``data/annotations/annotators/``.  Nothing else may write the master corpus's
gold columns (``human_lmp_score``, ``error_type``, ``decision``,
``annotation_status``, ``annotator_id``, ``annotated_at``).  This module is the
single point at which those columns are populated, which is what makes the
provenance of every label traceable to a named annotator and a timestamp.

Status assignment
-----------------
For each pair, given the set of human scores on record:

===========================  ==========================  ==========================
Situation                    Status written              Score written
===========================  ==========================  ==========================
an adjudication exists       ``human_consensus``         the adjudicated score
>=2 scores, spread < 2       ``human_consensus``         the mean
>=2 scores, spread >= 2      ``human_multi_pending``     the mean (provisional)
exactly 1 score              ``human_single``            that score
no human score               *unchanged*                 *unchanged*
===========================  ==========================  ==========================

``human_multi_pending`` is a human-derived label, so the schema admits it as
gold, but it is **not** an adjudicated one.  Reported experiments should use
``human_consensus`` rows; the merge prints how many pending rows remain so the
distinction cannot be lost by accident.

Safety
------
Writing is in place and therefore destructive, so the default is a dry run.
Pass ``--apply`` to actually write, and ``--backup`` to keep a timestamped copy
of the pre-merge master alongside it.

Usage
-----
::

    python -m src.annotation.merge_annotations                 # dry run
    python -m src.annotation.merge_annotations --apply --backup
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.annotation import agreement
from src.data import schema
from src.data.load_data import master_path
from src.utils.config import load_config, resolve_path
from src.utils.io import dataframe_to_markdown, provenance, read_csv, write_csv, write_json
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _combine_notes(entries: list[tuple[str, Any]]) -> str:
    """Join ``(annotator_id, note)`` pairs into one attributed string.

    Keeping the attribution inline means the paper's error analysis can quote a
    specific annotator's reasoning without cross-referencing another file.
    """
    parts = []
    for annotator, note in entries:
        text = "" if note is None or pd.isna(note) else str(note).strip()
        if text:
            parts.append(f"[{annotator}] {text}")
    return " | ".join(parts)


def _modal_error(entries: list[tuple[str, Any]]) -> tuple[str | None, list[str]]:
    """Most common error type among annotators, and any dissenters' labels.

    Returns ``(label, conflicting_labels)``.  A conflict is not an error to be
    hidden: it is evidence that the pair is genuinely ambiguous, and it is
    surfaced in the merge summary.
    """
    labels = [
        str(value).strip()
        for _, value in entries
        if value is not None and not pd.isna(value) and str(value).strip()
    ]
    if not labels:
        return None, []
    counts = pd.Series(labels).value_counts()
    top = str(counts.index[0])
    conflicts = sorted(set(labels) - {top})
    return top, conflicts


def plan_merge(
    master: pd.DataFrame,
    long_df: pd.DataFrame,
    adjudications: pd.DataFrame,
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute the merged master frame without writing anything.

    Returns the new frame and a summary describing exactly what changed, so a
    dry run can be reviewed before anything is overwritten.
    """
    cfg = cfg or load_config()
    decision_cfg = cfg.get("decision", {})
    preserved_min = float(decision_cfg.get("preserved_min", 8.0))
    partial_min = float(decision_cfg.get("partial_min", 5.0))

    out = schema.ensure_columns(master.copy())
    out = schema.coerce_types(out)

    # pair_id -> list of (annotator_id, score, error_type, notes, annotated_at)
    by_pair: dict[str, list[tuple[str, float, Any, Any, Any]]] = {}
    for row in long_df.itertuples(index=False):
        score = getattr(row, "human_lmp_score", None)
        if score is None or pd.isna(score):
            continue
        by_pair.setdefault(str(row.pair_id), []).append(
            (
                str(row.annotator_id),
                float(score),
                getattr(row, "error_type", None),
                getattr(row, "notes", None),
                getattr(row, "annotated_at", None),
            )
        )

    adjudicated = {
        str(row.pair_id): row for row in adjudications.itertuples(index=False)
    } if not adjudications.empty else {}

    changes: list[dict[str, Any]] = []
    statuses: dict[str, int] = {}
    error_conflicts: list[dict[str, Any]] = []

    for idx, row in out.iterrows():
        pair_id = str(row["pair_id"])
        entries = by_pair.get(pair_id, [])
        adjudication = adjudicated.get(pair_id)
        previous_status = row.get("annotation_status")

        if adjudication is not None and not pd.isna(getattr(adjudication, "consensus_score", None)):
            score = float(adjudication.consensus_score)
            status = "human_consensus"
            annotator = str(getattr(adjudication, "adjudicator_id", "") or "adjudicator")
            timestamp = str(getattr(adjudication, "adjudicated_at", "") or _utc_now_iso())
            error = getattr(adjudication, "error_type", None)
            notes = _combine_notes(
                [(str(annotator), getattr(adjudication, "notes", None))]
                + [(a, n) for a, _, _, n, _ in entries]
            )
        elif len(entries) >= 2:
            scores = [s for _, s, _, _, _ in entries]
            score = float(sum(scores) / len(scores))
            spread = max(scores) - min(scores)
            status = "human_consensus" if spread < agreement.ADJUDICATION_THRESHOLD else "human_multi_pending"
            annotator = ",".join(a for a, _, _, _, _ in entries)
            timestamp = max(str(t) for _, _, _, _, t in entries if t is not None and not pd.isna(t)) or _utc_now_iso()
            error, conflicts = _modal_error([(a, e) for a, _, e, _, _ in entries])
            if conflicts:
                error_conflicts.append(
                    {"pair_id": pair_id, "chosen": error, "dissent": ", ".join(conflicts)}
                )
            notes = _combine_notes([(a, n) for a, _, _, n, _ in entries])
            if status == "human_multi_pending":
                notes = (notes + " | " if notes else "") + (
                    f"[merge] annotators differ by {spread:g} points "
                    f"({', '.join(f'{a}={s:g}' for a, s, _, _, _ in entries)}); "
                    "awaiting adjudication"
                )
        elif len(entries) == 1:
            annotator, score, error, note, timestamp = entries[0]
            annotator = str(annotator)
            status = "human_single"
            timestamp = str(timestamp) if timestamp is not None and not pd.isna(timestamp) else _utc_now_iso()
            notes = _combine_notes([(annotator, note)])
        else:
            statuses[str(previous_status)] = statuses.get(str(previous_status), 0) + 1
            continue

        if error is None or pd.isna(error) or not str(error).strip():
            error = "None"
        error = str(error).strip()
        if error not in schema.ERROR_LABELS:
            logger.warning(
                "%s: error_type %r is not in the taxonomy; recording it as 'Other' "
                "and keeping the original text in the notes.",
                pair_id,
                error,
            )
            notes = (notes + " | " if notes else "") + f"[merge] unrecognised error_type {error!r}"
            error = "Other"

        out.at[idx, "human_lmp_score"] = score
        out.at[idx, "error_type"] = error
        out.at[idx, "decision"] = schema.score_to_decision(score, preserved_min, partial_min)
        out.at[idx, "annotation_status"] = status
        out.at[idx, "annotator_id"] = annotator
        out.at[idx, "annotated_at"] = timestamp
        out.at[idx, "notes"] = notes

        statuses[status] = statuses.get(status, 0) + 1
        changes.append(
            {
                "pair_id": pair_id,
                "language": row.get("language"),
                "status": status,
                "score": round(score, 2),
                "n_annotators": 0 if adjudication is not None else len(entries),
                "error_type": error,
            }
        )

    out = schema.coerce_types(out)
    summary = {
        "n_rows_total": int(len(out)),
        "n_rows_annotated": len(changes),
        "n_rows_untouched": int(len(out)) - len(changes),
        "status_counts": statuses,
        "error_type_conflicts": error_conflicts,
        "n_error_type_conflicts": len(error_conflicts),
        "changes": changes,
    }
    return out, summary


def render_summary(summary: dict[str, Any], applied: bool) -> str:
    """Render the merge summary as Markdown."""
    mode = "APPLIED" if applied else "DRY RUN — nothing was written"
    lines = [f"# Annotation Merge ({mode})\n"]
    lines.append(
        f"- Rows in master: **{summary['n_rows_total']}**\n"
        f"- Rows carrying a human score after merge: **{summary['n_rows_annotated']}**\n"
        f"- Rows left untouched: **{summary['n_rows_untouched']}**\n"
    )
    lines.append("\n## Statuses\n")
    table = pd.DataFrame(
        sorted(summary["status_counts"].items()), columns=["status", "rows"]
    )
    lines.append(dataframe_to_markdown(table))

    if summary["n_error_type_conflicts"]:
        lines.append(
            f"\n## Error-type disagreement ({summary['n_error_type_conflicts']})\n\n"
            "Annotators chose different primary error types for these pairs. "
            "This is a real finding about the difficulty of the taxonomy and "
            "belongs in the paper's error analysis.\n"
        )
        lines.append(dataframe_to_markdown(pd.DataFrame(summary["error_type_conflicts"])))

    changes = summary.get("changes") or []
    if changes:
        lines.append("\n## Per-pair changes\n")
        lines.append(dataframe_to_markdown(pd.DataFrame(changes)))

    pending = summary["status_counts"].get("human_multi_pending", 0)
    if pending:
        lines.append(
            f"\n> **{pending} pair(s) are `human_multi_pending`.** They carry a "
            f"provisional mean and are NOT adjudicated. Primary results must "
            f"report `human_consensus` only.\n"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge per-annotator scores into the master corpus."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--master", type=str, default=None)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write the master CSV. Without this it is a dry run.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Keep a timestamped copy of the master before overwriting it.",
    )
    parser.add_argument("--report", action="store_true", help="Write reports/annotation_merge.md")
    args = parser.parse_args(argv)

    setup_logging("INFO")
    cfg = load_config(args.config)

    master_file = Path(args.master) if args.master else master_path(cfg)
    master = read_csv(master_file)
    long_df = agreement.load_annotations(cfg)
    adjudications = agreement.load_adjudications(cfg)

    if long_df.empty and adjudications.empty:
        logger.warning(
            "No annotations found under %s. Nothing to merge.",
            agreement.annotators_dir(cfg),
        )

    merged, summary = plan_merge(master, long_df, adjudications, cfg)

    logger.info(
        "Merge plan: %d row(s) annotated, %d untouched. Statuses: %s",
        summary["n_rows_annotated"],
        summary["n_rows_untouched"],
        summary["status_counts"],
    )

    if args.apply:
        if args.backup:
            stamp = _utc_now_iso().replace(":", "").replace("-", "")
            backup = master_file.with_name(f"{master_file.stem}.premerge.{stamp}.csv")
            shutil.copy2(master_file, backup)
            logger.info("Backed up master to %s", backup)
        write_csv(merged, master_file)
        logger.info("Wrote merged master -> %s", master_file)
    else:
        logger.info("Dry run. Re-run with --apply to write.")

    if args.report:
        reports_dir = resolve_path(cfg, "reports_dir", create=True)
        results_dir = resolve_path(cfg, "results_dir", create=True)
        (reports_dir / "annotation_merge.md").write_text(
            render_summary(summary, args.apply), encoding="utf-8"
        )
        write_json(
            {"provenance": provenance(cfg), "applied": bool(args.apply), **summary},
            results_dir / "annotation_merge.json",
        )
        logger.info("Wrote %s", reports_dir / "annotation_merge.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
