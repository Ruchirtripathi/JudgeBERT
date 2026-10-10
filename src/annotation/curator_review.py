"""How the curator's draft scores relate to the human ones. CURATOR-ONLY.

.. warning::

   Nothing produced by this module may be shown to an annotator, in the
   annotation tool or anywhere else.  It prints pairs side by side with both
   the draft and the human score.  An annotator who sees it is looking at the
   expected label, which is exactly what the blind mode of
   :mod:`src.annotation.annotation_app` withholds — see the project
   requirement that no automatically generated expected label be exposed to
   annotators.

Why this exists
---------------
The drafts were written by the curator while the corpus was being assembled,
to steer the score distribution toward the ten-point scale.  That makes them a
useful sanity check *after* annotation: if human annotators systematically
disagree with the drafts, either the scale anchors in
``src/annotation/guidelines.md`` are miscalibrated, or the drafts are, and the
difference is worth reporting.  It is a check on the drafts, not a result about
model quality, and it is never a substitute for the human labels.

This is the only legitimate home for the comparison.  It used to be rendered
inside the annotation tool's Review tab, which meant every annotator could see
the draft for every pair they had scored — and, because the tool lets a score
be revised, could then revise against it.

Usage
-----
::

    python -m src.annotation.curator_review --report
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import pandas as pd

from src.annotation import agreement
from src.data import schema
from src.data.load_data import load_master
from src.utils.config import load_config, resolve_path
from src.utils.io import (
    RESULT_PENDING,
    dataframe_to_markdown,
    provenance,
    write_json,
)
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

#: Below this many paired observations a correlation is not reported.
MIN_PAIRS: int = 20


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation, or ``None`` when it is undefined."""
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Ranks with tied values sharing their average rank.

    Needed for Spearman: assigning tied values arbitrary distinct ranks would
    invent an ordering the data does not have, and on a 1-10 integer scale ties
    are the common case rather than the exception.
    """
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    ordered = values[order]
    start = 0
    while start < len(ordered):
        stop = start
        while stop + 1 < len(ordered) and ordered[stop + 1] == ordered[start]:
            stop += 1
        if stop > start:
            ranks[order[start : stop + 1]] = (start + stop + 2) / 2.0
        start = stop + 1
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2:
        return None
    return _pearson(_rankdata(a), _rankdata(b))


def compare_to_drafts(
    master: pd.DataFrame,
    long_df: pd.DataFrame,
    cfg: dict | None = None,
    *,
    min_pairs: int = MIN_PAIRS,
) -> dict[str, Any]:
    """Compare every human score on record against the curator's draft.

    Returns ``RESULT_PENDING`` rather than a number when there is too little
    overlap, on the same reasoning as
    :func:`src.annotation.agreement.compute_agreement`: a correlation over a
    handful of rows is noise dressed as a finding.
    """
    cfg = cfg or load_config()
    decision_cfg = cfg.get("decision", {})
    preserved_min = float(decision_cfg.get("preserved_min", 8.0))
    partial_min = float(decision_cfg.get("partial_min", 5.0))

    pending: dict[str, Any] = {
        "status": RESULT_PENDING,
        "n_pairs_compared": 0,
        "min_pairs_required": min_pairs,
        "pearson": RESULT_PENDING,
        "spearman": RESULT_PENDING,
        "mean_absolute_difference": RESULT_PENDING,
        "mean_signed_difference": RESULT_PENDING,
        "percent_within_1": RESULT_PENDING,
        "decision_agreement": RESULT_PENDING,
        "per_annotator": {},
    }

    if long_df.empty:
        pending["reason"] = (
            "No annotation files found, so there are no human scores to compare "
            "the drafts against. Run the annotation tool first."
        )
        return pending

    drafts = master[["pair_id", "draft_lmp_score"]].copy()
    drafts["pair_id"] = drafts["pair_id"].astype(str)
    drafts["draft_lmp_score"] = pd.to_numeric(drafts["draft_lmp_score"], errors="coerce")

    joined = long_df.copy()
    joined["pair_id"] = joined["pair_id"].astype(str)
    joined = joined.merge(drafts, on="pair_id", how="left")
    joined["human_lmp_score"] = pd.to_numeric(joined["human_lmp_score"], errors="coerce")
    joined = joined.dropna(subset=["human_lmp_score", "draft_lmp_score"])

    if joined.empty:
        pending["reason"] = (
            "No pair on record carries both a human score and a curator draft."
        )
        return pending

    human = joined["human_lmp_score"].to_numpy(dtype=float)
    draft = joined["draft_lmp_score"].to_numpy(dtype=float)
    difference = human - draft

    result = dict(pending)
    result["n_pairs_compared"] = int(len(joined))
    result["n_annotators"] = int(joined["annotator_id"].nunique())

    if len(joined) < min_pairs:
        result["reason"] = (
            f"Only {len(joined)} pair(s) carry both a human score and a draft; "
            f"at least {min_pairs} are needed before a correlation is worth "
            f"reporting. Row-level differences are listed instead."
        )
        result["rows"] = _row_level(joined, preserved_min, partial_min)
        return result

    result["status"] = "computed"
    result["pearson"] = _pearson(human, draft)
    result["spearman"] = _spearman(human, draft)
    result["mean_absolute_difference"] = float(np.abs(difference).mean())
    result["mean_signed_difference"] = float(difference.mean())
    result["percent_within_1"] = float((np.abs(difference) <= 1).mean())

    if result["pearson"] is None:
        result["pearson"] = RESULT_PENDING
    if result["spearman"] is None:
        result["spearman"] = RESULT_PENDING

    human_decision = joined["human_lmp_score"].map(
        lambda value: schema.score_to_decision(value, preserved_min, partial_min)
    )
    draft_decision = joined["draft_lmp_score"].map(
        lambda value: schema.score_to_decision(value, preserved_min, partial_min)
    )
    result["decision_agreement"] = float((human_decision == draft_decision).mean())

    result["per_annotator"] = {}
    for annotator_id, block in joined.groupby("annotator_id"):
        values = block["human_lmp_score"].to_numpy(dtype=float)
        drafts_block = block["draft_lmp_score"].to_numpy(dtype=float)
        result["per_annotator"][str(annotator_id)] = {
            "n_pairs": int(len(block)),
            "pearson": _pearson(values, drafts_block),
            "mean_signed_difference": float((values - drafts_block).mean()),
            "mean_absolute_difference": float(np.abs(values - drafts_block).mean()),
        }

    result["rows"] = _row_level(joined, preserved_min, partial_min)
    return result


def _row_level(
    joined: pd.DataFrame, preserved_min: float, partial_min: float
) -> list[dict[str, Any]]:
    """One record per compared pair, largest disagreement first."""
    rows: list[dict[str, Any]] = []
    for entry in joined.itertuples(index=False):
        human = float(entry.human_lmp_score)
        draft = float(entry.draft_lmp_score)
        rows.append(
            {
                "pair_id": entry.pair_id,
                "annotator_id": entry.annotator_id,
                "human": human,
                "draft": draft,
                "difference": human - draft,
                "human_decision": schema.score_to_decision(human, preserved_min, partial_min),
                "draft_decision": schema.score_to_decision(draft, preserved_min, partial_min),
            }
        )
    rows.sort(key=lambda row: (-abs(row["difference"]), row["pair_id"]))
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def render_report(result: dict[str, Any]) -> str:
    """Render the comparison as Markdown, banner first."""
    lines = [
        "# Curator Drafts vs Human Labels\n",
        "> **CURATOR-ONLY.** This document shows both the draft and the human "
        "score for each pair. It must not be shown to annotators: it contains "
        "the automatically generated expected label that blind annotation "
        "exists to withhold. Compare it against "
        "`src/annotation/guidelines.md` §2 — a systematic gap means the scale "
        "anchors or the drafts are miscalibrated.\n",
    ]

    if result.get("status") == RESULT_PENDING:
        lines.append(f"**{RESULT_PENDING}** — {result.get('reason')}\n")
        rows = result.get("rows") or []
        if rows:
            lines.append("\n## Pairs compared so far\n")
            lines.append(dataframe_to_markdown(pd.DataFrame(rows)))
        return "\n".join(lines) + "\n"

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines.append(
        f"Compared **{result['n_pairs_compared']}** scored pair(s) across "
        f"**{result['n_annotators']}** annotator(s).\n"
    )
    table = pd.DataFrame(
        [
            {"Statistic": "Pearson r (human vs draft)", "Value": fmt(result["pearson"])},
            {"Statistic": "Spearman rho", "Value": fmt(result["spearman"])},
            {"Statistic": "Decision agreement", "Value": fmt(result["decision_agreement"])},
            {"Statistic": "Within +/-1 point", "Value": fmt(result["percent_within_1"])},
            {"Statistic": "Mean absolute difference", "Value": fmt(result["mean_absolute_difference"])},
            {"Statistic": "Mean signed difference", "Value": fmt(result["mean_signed_difference"])},
        ]
    )
    lines.append(dataframe_to_markdown(table, float_format="{:.3f}"))
    lines.append(
        "\nA positive mean signed difference means annotators scored *above* the "
        "drafts on average; a negative one means the drafts were optimistic.\n"
    )

    per = result.get("per_annotator") or {}
    if per:
        lines.append("\n## Per annotator\n")
        frame = pd.DataFrame(per).T.reset_index().rename(columns={"index": "annotator"})
        lines.append(dataframe_to_markdown(frame, float_format="{:.3f}"))

    rows = result.get("rows") or []
    if rows:
        lines.append("\n## Largest disagreements\n")
        lines.append(dataframe_to_markdown(pd.DataFrame(rows[:40])))

    lines.append(
        "\n> These are measured values from the annotation files on disk. "
        "They say how well the drafts anticipated the annotators, which is a "
        "statement about the drafts — not about any model's performance.\n"
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare curator draft scores against human annotations (curator-only)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--report", action="store_true", help="Write reports/ and results/ artefacts")
    args = parser.parse_args(argv)

    setup_logging("INFO")
    cfg = load_config(args.config)

    master = load_master(subset="all")
    long_df = agreement.load_annotations(cfg)
    logger.info("Loaded %d row(s) of human annotation.", len(long_df))

    result = compare_to_drafts(master, long_df, cfg)
    result = {"provenance": provenance(cfg), **result}

    if result["status"] == RESULT_PENDING:
        logger.warning("Draft comparison: %s — %s", RESULT_PENDING, result.get("reason"))
    else:
        logger.info(
            "Draft comparison over %d pair(s): r=%.3f, rho=%s, mean signed diff=%.3f.",
            result["n_pairs_compared"],
            result["pearson"],
            result["spearman"],
            result["mean_signed_difference"],
        )

    if args.report:
        reports_dir = resolve_path(cfg, "reports_dir", create=True)
        results_dir = resolve_path(cfg, "results_dir", create=True)
        (reports_dir / "curator_draft_review.md").write_text(
            render_report(result), encoding="utf-8"
        )
        write_json(result, results_dir / "curator_draft_review.json")
        logger.info("Wrote %s", reports_dir / "curator_draft_review.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
