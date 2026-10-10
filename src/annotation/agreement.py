"""Inter-annotator agreement for the LMP scale.

Every statistic here is computed from the annotation files that actually exist
on disk.  Nothing is estimated, imputed or smoothed, and when there is not
enough data to compute a statistic honestly the module returns
:data:`RESULT_PENDING` rather than a number.  A kappa computed on four
overlapping rows is noise, and reporting it as a result would be worse than
reporting nothing.

Statistics
----------
``cohen_kappa_quadratic``
    Pairwise agreement on the ordinal 1-10 scale, disagreement weighted by the
    square of the distance between the two scores.
``krippendorff_alpha_ordinal``
    The same idea generalised to more than two annotators and to missing data,
    which is the normal state of a partially-complete annotation effort.
``icc_2_1``
    ICC(2,1): two-way random effects, absolute agreement, single measures.
    Treats the scale as continuous, so it answers a slightly different
    question from kappa - "do annotators give the same number?" rather than
    "do they give the same category?".
``percent_within``
    The practical tolerance.  Two annotators who differ by one point on a
    ten-point scale have not really disagreed; this says how often that holds.

Reading the results
-------------------
Agreement on this task is expected to be moderate rather than high, because
several pairs in the corpus are deliberately hard (a single changed numeral,
a dropped proviso).  A moderate value is a finding about the difficulty of
legal meaning preservation, not a defect to be repaired by re-scoring until
the number looks better.  See ``src/annotation/guidelines.md`` section 6.

Usage
-----
::

    python -m src.annotation.agreement --report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.data import schema
from src.utils.config import load_config, resolve_path
from src.utils.io import (
    RESULT_PENDING,
    dataframe_to_markdown,
    provenance,
    read_csv,
    write_json,
)
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

#: Columns every per-annotator file must carry.
ANNOTATION_COLUMNS: list[str] = [
    "pair_id",
    "human_lmp_score",
    "error_type",
    "notes",
    "annotator_id",
    "annotated_at",
]

#: Columns of the adjudication file.
ADJUDICATION_COLUMNS: list[str] = [
    "pair_id",
    "consensus_score",
    "error_type",
    "adjudicator_id",
    "notes",
    "adjudicated_at",
]

#: Below this many *overlapping* pairs a statistic is not reported.
#:
#: Rationale: with n overlapping units the standard error of kappa is roughly
#: 1/sqrt(n); at n=20 that is already +/-0.22, which is wider than the gap
#: between "moderate" and "substantial" on the Landis & Koch bands.  Reporting
#: a number at that precision would invite a conclusion the data cannot carry.
MIN_OVERLAP_PAIRS: int = 20

#: Difference at or above which a pair is routed to adjudication.
ADJUDICATION_THRESHOLD: int = 2

#: Landis & Koch (1977) bands for kappa.
KAPPA_BANDS: list[tuple[float, float, str]] = [
    (float("-inf"), 0.20, "slight"),
    (0.20, 0.40, "fair"),
    (0.40, 0.60, "moderate"),
    (0.60, 0.80, "substantial"),
    (0.80, float("inf"), "almost perfect"),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def annotators_dir(cfg: dict | None = None) -> Path:
    """Directory holding one CSV per annotator."""
    cfg = cfg or load_config()
    return resolve_path(cfg, "annotations_dir") / "annotators"


def adjudication_path(cfg: dict | None = None) -> Path:
    """Path of the adjudication file."""
    cfg = cfg or load_config()
    return resolve_path(cfg, "annotations_dir") / "adjudication.csv"


def load_annotations(cfg: dict | None = None, directory: Path | None = None) -> pd.DataFrame:
    """Read every per-annotator file into one long frame.

    Returns a frame with :data:`ANNOTATION_COLUMNS`.  The ``annotator_id``
    column is taken from the file's own rows, not from the filename, so a
    mis-named file cannot silently attribute one person's judgements to
    another.  A disagreement between the two is reported as a warning.
    """
    cfg = cfg or load_config()
    directory = directory or annotators_dir(cfg)
    if not directory.exists():
        logger.info("No annotator directory at %s yet.", directory)
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)

    frames: list[pd.DataFrame] = []
    for path in sorted(directory.glob("*.csv")):
        try:
            frame = read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Could not read %s: %s", path, exc)
            continue

        missing = [c for c in ANNOTATION_COLUMNS if c not in frame.columns]
        if missing:
            logger.error(
                "Skipping %s: missing column(s) %s. An annotation file must be "
                "written by src/annotation/annotation_app.py.",
                path.name,
                missing,
            )
            continue

        declared = frame["annotator_id"].dropna().astype(str).unique()
        if len(declared) > 1:
            logger.error("Skipping %s: it declares several annotator_ids %s.", path.name, list(declared))
            continue
        if len(declared) == 1 and declared[0] != path.stem:
            logger.warning(
                "%s declares annotator_id %r but is named %r; trusting the "
                "declared id.",
                path.name,
                declared[0],
                path.stem,
            )
        frames.append(frame[ANNOTATION_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined["human_lmp_score"] = pd.to_numeric(
        combined["human_lmp_score"], errors="coerce"
    )

    # Range-check the scores here, at the single point where annotator files
    # enter the system.  Every statistic downstream assumes the 1-10 scale:
    # quadratic weights and the ordinal metric are both defined in terms of the
    # distance between categories, so a stray 11 or 0 does not merely add a
    # large error, it produces a coefficient that is off the scale and looks
    # like a real result.  Agreement is a reported number in the paper, so a
    # malformed score must be dropped loudly rather than averaged in.
    out_of_range = combined["human_lmp_score"].notna() & ~combined[
        "human_lmp_score"
    ].between(schema.SCORE_MIN, schema.SCORE_MAX)
    if out_of_range.any():
        for annotator_id, block in combined.loc[out_of_range].groupby("annotator_id"):
            logger.error(
                "%s supplied %d score(s) outside %d-%d (%s); dropping those rows.",
                annotator_id,
                len(block),
                schema.SCORE_MIN,
                schema.SCORE_MAX,
                sorted(block["human_lmp_score"].unique().tolist()),
            )
        combined = combined.loc[~out_of_range]

    # A pair scored twice by the same annotator is a bug in the tool or a
    # hand-edit; keep the last write and say so.
    dupes = combined.duplicated(subset=["pair_id", "annotator_id"], keep="last")
    if dupes.any():
        logger.warning(
            "%d duplicate (pair_id, annotator_id) row(s); keeping the latest.",
            int(dupes.sum()),
        )
        combined = combined.loc[~dupes]
    return combined.reset_index(drop=True)


def load_adjudications(cfg: dict | None = None) -> pd.DataFrame:
    """Read the adjudication file, or an empty frame if it does not exist."""
    path = adjudication_path(cfg)
    if not path.exists():
        return pd.DataFrame(columns=ADJUDICATION_COLUMNS)
    frame = read_csv(path)
    missing = [c for c in ADJUDICATION_COLUMNS if c not in frame.columns]
    if missing:
        logger.error("Adjudication file %s is missing %s; ignoring it.", path.name, missing)
        return pd.DataFrame(columns=ADJUDICATION_COLUMNS)
    frame["consensus_score"] = pd.to_numeric(frame["consensus_score"], errors="coerce")
    return frame


def rating_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long annotation frame to ``pair_id x annotator``.

    Missing cells are ``NaN``, which every statistic below treats as "this
    annotator did not score this pair" rather than as a zero.
    """
    if long_df.empty:
        return pd.DataFrame()
    matrix = long_df.pivot_table(
        index="pair_id",
        columns="annotator_id",
        values="human_lmp_score",
        aggfunc="last",
    )
    return matrix.sort_index()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def cohen_kappa_quadratic(
    a: Sequence[float],
    b: Sequence[float],
    lo: int = schema.SCORE_MIN,
    hi: int = schema.SCORE_MAX,
) -> float | None:
    """Quadratic-weighted Cohen's kappa for two raters.

    Weights are ``(i-j)^2 / (hi-lo)^2``, so a 1-point disagreement costs 1/81
    of a 9-point one.  The full declared range (1-10) is used for the
    denominator even when the observed subset is narrower, so that kappa is
    comparable between an English-only and a Hindi-only slice.

    Returns ``None`` when the statistic is undefined - no overlapping pairs,
    or no variance at all in the ratings.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return None

    categories = np.arange(lo, hi + 1)
    size = len(categories)
    observed = np.zeros((size, size))
    for xi, yi in zip(x.astype(int), y.astype(int)):
        if lo <= xi <= hi and lo <= yi <= hi:
            observed[xi - lo, yi - lo] += 1
    n = observed.sum()
    if n == 0:
        return None

    weights = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            weights[i, j] = (categories[i] - categories[j]) ** 2 / (hi - lo) ** 2

    row = observed.sum(axis=1, keepdims=True)
    col = observed.sum(axis=0, keepdims=True)
    expected = row @ col / n

    numerator = float((weights * observed).sum())
    denominator = float((weights * expected).sum())
    if denominator == 0:
        # Every rating sits in one category: agreement is trivially perfect
        # but kappa is 0/0.  Undefined, not 1.0.
        return None
    return 1.0 - numerator / denominator


def krippendorff_alpha_ordinal(matrix: pd.DataFrame) -> float | None:
    """Krippendorff's alpha with the ordinal difference function.

    Generalises kappa to any number of annotators and tolerates missing cells,
    which matters because annotators will not finish the corpus in lockstep.

    The ordinal metric is

        delta^2(c, k) = ( sum_{g=c..k} n_g  -  (n_c + n_k)/2 )^2

    i.e. the disagreement between two categories is the mass of the scale
    lying between them, which is what makes the statistic order-aware rather
    than treating every mis-match as equally wrong.
    """
    if matrix.empty:
        return None
    values = matrix.to_numpy(dtype=float)
    present = values[~np.isnan(values)]
    if present.size == 0:
        return None

    categories = np.unique(present).astype(int)
    index = {int(c): i for i, c in enumerate(categories)}
    size = len(categories)

    coincidence = np.zeros((size, size))
    for row in values:
        rated = row[~np.isnan(row)].astype(int)
        m_u = len(rated)
        if m_u < 2:
            continue
        for i in range(m_u):
            for j in range(m_u):
                if i == j:
                    continue
                coincidence[index[rated[i]], index[rated[j]]] += 1.0 / (m_u - 1)

    n_c = coincidence.sum(axis=1)
    total = coincidence.sum()
    if total <= 1:
        return None

    expected = np.outer(n_c, n_c)
    for i in range(size):
        expected[i, i] = n_c[i] * (n_c[i] - 1)
    expected = expected / (total - 1)

    delta = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            low, high = min(i, j), max(i, j)
            between = n_c[low : high + 1].sum()
            delta[i, j] = (between - (n_c[i] + n_c[j]) / 2.0) ** 2

    denominator = float((expected * delta).sum())
    if denominator == 0:
        return None
    return 1.0 - float((coincidence * delta).sum()) / denominator


def icc_2_1(matrix: pd.DataFrame) -> float | None:
    """ICC(2,1): two-way random effects, absolute agreement, single measures.

    Uses only complete cases, since the mean squares are not defined when a
    unit is missing a rating.  Requires at least three complete units.
    """
    if matrix.empty:
        return None
    values = matrix.to_numpy(dtype=float)
    complete = values[~np.isnan(values).any(axis=1)]
    n, k = complete.shape
    if n < 3 or k < 2:
        return None

    grand = complete.mean()
    row_means = complete.mean(axis=1)
    col_means = complete.mean(axis=0)

    ss_rows = k * float(((row_means - grand) ** 2).sum())
    ss_cols = n * float(((col_means - grand) ** 2).sum())
    ss_error = float(
        ((complete - row_means[:, None] - col_means[None, :] + grand) ** 2).sum()
    )

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if denominator == 0:
        return None
    return (ms_rows - ms_error) / denominator


def percent_within(matrix: pd.DataFrame, tolerance: float = 1.0) -> float | None:
    """Share of within-pair rating comparisons that differ by <= tolerance.

    Computed over every pair of ratings on the same item, so it uses all the
    data rather than only the complete rows that ICC needs.
    """
    if matrix.empty:
        return None
    values = matrix.to_numpy(dtype=float)
    deltas: list[float] = []
    for row in values:
        rated = row[~np.isnan(row)]
        for i in range(len(rated)):
            for j in range(i + 1, len(rated)):
                deltas.append(abs(rated[i] - rated[j]))
    if not deltas:
        return None
    deltas_arr = np.asarray(deltas)
    return float((deltas_arr <= tolerance).mean())


def mean_absolute_difference(matrix: pd.DataFrame) -> float | None:
    """Mean absolute difference between two annotators' scores, in scale points."""
    if matrix.empty:
        return None
    values = matrix.to_numpy(dtype=float)
    deltas: list[float] = []
    for row in values:
        rated = row[~np.isnan(row)]
        for i in range(len(rated)):
            for j in range(i + 1, len(rated)):
                deltas.append(abs(rated[i] - rated[j]))
    if not deltas:
        return None
    return float(np.mean(deltas))


def kappa_band(value: float | None) -> str:
    """Landis & Koch interpretation of a kappa value."""
    if value is None or value != value:
        return RESULT_PENDING
    for low, high, label in KAPPA_BANDS:
        if low < value <= high:
            return label
    return RESULT_PENDING


def bootstrap_ci(
    matrix: pd.DataFrame,
    statistic,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI for a statistic over annotator units.

    Resamples *pairs* (rows) with replacement, which is the unit of
    independence.  Returns ``None`` when the statistic is undefined on the
    original sample or on more than a fifth of the resamples - a CI built on
    a statistic that mostly does not exist would be misleading.
    """
    if matrix.empty or len(matrix) < 3:
        return None
    point = statistic(matrix)
    if point is None:
        return None

    rng = np.random.default_rng(seed)
    values = matrix.to_numpy(dtype=float)
    n = len(values)
    draws: list[float] = []
    failed = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample = pd.DataFrame(values[idx], columns=matrix.columns)
        value = statistic(sample)
        if value is None or value != value:
            failed += 1
            continue
        draws.append(float(value))

    if failed > n_resamples / 5 or len(draws) < 10:
        return None
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(draws, alpha)),
        float(np.quantile(draws, 1.0 - alpha)),
    )


# ---------------------------------------------------------------------------
# Disagreement routing
# ---------------------------------------------------------------------------
def disagreement_table(matrix: pd.DataFrame, threshold: int = ADJUDICATION_THRESHOLD) -> pd.DataFrame:
    """Pairs whose annotators differ by at least ``threshold`` points.

    These are the pairs the guidelines route to adjudication.  The table names
    the annotators and both scores so the adjudicator can see exactly what is
    in dispute.
    """
    # Expected schema columns
    cols = ["pair_id", "min_score", "max_score", "spread", "annotators"]
    if matrix.empty:
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, Any]] = []
    for pair_id, row in matrix.iterrows():
        rated = row.dropna()
        if len(rated) < 2:
            continue
        spread = float(rated.max() - rated.min())
        if spread >= threshold:
            rows.append(
                {
                    "pair_id": pair_id,
                    "min_score": float(rated.min()),
                    "max_score": float(rated.max()),
                    "spread": spread,
                    "annotators": ", ".join(
                        f"{name}={value:g}" for name, value in rated.items()
                    ),
                }
            )
    # Build DataFrame with explicit columns to guarantee schema
    frame = pd.DataFrame(rows, columns=cols)
    if not frame.empty:
        frame = frame.sort_values(["spread", "pair_id"], ascending=[False, True])
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-level computation
# ---------------------------------------------------------------------------
def compute_agreement(
    long_df: pd.DataFrame,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    min_overlap: int = MIN_OVERLAP_PAIRS,
) -> dict[str, Any]:
    """Compute every agreement statistic, or explain why it cannot be.

    The returned dict always carries ``status`` and ``n_annotators``.  When
    ``status`` is :data:`RESULT_PENDING`, every statistic is that same token -
    the caller never has to guess whether a missing number means zero or
    unknown.
    """
    pending: dict[str, Any] = {
        "status": RESULT_PENDING,
        "n_annotators": 0,
        "n_pairs_scored": 0,
        "n_pairs_overlapping": 0,
        "min_overlap_required": min_overlap,
        "cohen_kappa_quadratic": RESULT_PENDING,
        "cohen_kappa_band": RESULT_PENDING,
        "cohen_kappa_ci": RESULT_PENDING,
        "krippendorff_alpha_ordinal": RESULT_PENDING,
        "icc_2_1": RESULT_PENDING,
        "percent_within_1": RESULT_PENDING,
        "mean_absolute_difference": RESULT_PENDING,
        "adjudication_threshold": ADJUDICATION_THRESHOLD,
        "n_pairs_needing_adjudication": RESULT_PENDING,
    }

    if long_df.empty:
        pending["reason"] = (
            "No annotation files found. Run the annotation tool "
            "(src/annotation/annotation_app.py) first."
        )
        return pending

    matrix = rating_matrix(long_df)
    n_annotators = int(matrix.shape[1])
    n_scored = int(matrix.notna().any(axis=1).sum())
    n_overlapping = int((matrix.notna().sum(axis=1) >= 2).sum())

    pending["n_annotators"] = n_annotators
    pending["n_pairs_scored"] = n_scored
    pending["n_pairs_overlapping"] = n_overlapping

    if n_annotators < 2:
        pending["reason"] = (
            f"Only {n_annotators} annotator(s) on record. Agreement is a "
            f"property of two or more independent annotators; there is "
            f"nothing to compute yet."
        )
        return pending
    if n_overlapping < min_overlap:
        pending["reason"] = (
            f"Only {n_overlapping} pair(s) scored by more than one annotator; "
            f"at least {min_overlap} are needed before a coefficient is "
            f"meaningful. Row-level spread is reported instead."
        )
        pending["disagreements"] = disagreement_table(matrix).to_dict("records")
        return pending

    result = dict(pending)
    result["status"] = "computed"

    if n_annotators == 2:
        columns = list(matrix.columns)
        kappa = cohen_kappa_quadratic(
            matrix[columns[0]].to_numpy(), matrix[columns[1]].to_numpy()
        )
        ci = bootstrap_ci(
            matrix,
            lambda m: cohen_kappa_quadratic(m.iloc[:, 0].to_numpy(), m.iloc[:, 1].to_numpy()),
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )
    else:
        # Weighted kappa is defined for exactly two raters; with three or more
        # the faithful statistic is Krippendorff's alpha, which the caller has
        # anyway.  Report the pairwise mean as a descriptive extra.
        kappa = None
        ci = None
        pairwise = [
            cohen_kappa_quadratic(
                matrix[matrix.columns[i]].to_numpy(),
                matrix[matrix.columns[j]].to_numpy(),
            )
            for i in range(n_annotators)
            for j in range(i + 1, n_annotators)
        ]
        pairwise = [p for p in pairwise if p is not None]
        result["mean_pairwise_kappa"] = float(np.mean(pairwise)) if pairwise else RESULT_PENDING

    # NOTE: every one of these must be tested with `is not None`.  A genuine
    # alpha of 0.0 (chance-level agreement) is falsy, and `x or PENDING`
    # would silently report it as "not computed" — hiding the most
    # interesting possible result.
    alpha = krippendorff_alpha_ordinal(matrix)
    icc = icc_2_1(matrix)

    result["cohen_kappa_quadratic"] = kappa if kappa is not None else RESULT_PENDING
    result["cohen_kappa_band"] = kappa_band(kappa)
    result["cohen_kappa_ci"] = list(ci) if ci is not None else RESULT_PENDING
    result["krippendorff_alpha_ordinal"] = alpha if alpha is not None else RESULT_PENDING
    result["icc_2_1"] = icc if icc is not None else RESULT_PENDING
    result["percent_within_1"] = percent_within(matrix, 1.0)
    result["mean_absolute_difference"] = mean_absolute_difference(matrix)
    disagreements = disagreement_table(matrix)
    result["n_pairs_needing_adjudication"] = int(len(disagreements))
    result["disagreements"] = disagreements.to_dict("records")
    result["per_annotator"] = {
        str(name): {
            "n_scored": int(matrix[name].notna().sum()),
            "mean": float(matrix[name].mean()) if matrix[name].notna().any() else RESULT_PENDING,
            "sd": float(matrix[name].std()) if matrix[name].notna().sum() > 1 else RESULT_PENDING,
        }
        for name in matrix.columns
    }
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def render_report(result: dict[str, Any]) -> str:
    """Render the agreement block as Markdown."""
    lines = ["# Inter-Annotator Agreement\n"]

    if result.get("status") == RESULT_PENDING:
        lines.append(f"**{RESULT_PENDING}** — {result.get('reason', 'insufficient data')}\n")
        lines.append(
            f"- Annotators on record: **{result.get('n_annotators', 0)}**\n"
            f"- Pairs scored: **{result.get('n_pairs_scored', 0)}**\n"
            f"- Pairs scored by two or more: **{result.get('n_pairs_overlapping', 0)}** "
            f"(need {result.get('min_overlap_required')})\n"
        )
        rows = result.get("disagreements") or []
        if rows:
            lines.append("\n## Pairs already showing spread\n")
            lines.append(dataframe_to_markdown(pd.DataFrame(rows)))
        return "\n".join(lines) + "\n"

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines.append(
        f"Computed over **{result['n_pairs_overlapping']}** pairs scored by "
        f"**{result['n_annotators']}** annotators "
        f"({result['n_pairs_scored']} pairs scored in total).\n"
    )
    lines.append("## Coefficients\n")
    table = pd.DataFrame(
        [
            {
                "Statistic": "Cohen's kappa (quadratic)",
                "Value": fmt(result["cohen_kappa_quadratic"]),
                "Interpretation": result["cohen_kappa_band"],
            },
            {
                "Statistic": "Krippendorff's alpha (ordinal)",
                "Value": fmt(result["krippendorff_alpha_ordinal"]),
                "Interpretation": "",
            },
            {
                "Statistic": "ICC(2,1)",
                "Value": fmt(result["icc_2_1"]),
                "Interpretation": "",
            },
            {
                "Statistic": "Agreement within +/-1 point",
                "Value": fmt(result["percent_within_1"]),
                "Interpretation": "share of comparisons",
            },
            {
                "Statistic": "Mean absolute difference",
                "Value": fmt(result["mean_absolute_difference"]),
                "Interpretation": "scale points",
            },
        ]
    )
    lines.append(dataframe_to_markdown(table, float_format="{:.3f}"))

    ci = result.get("cohen_kappa_ci")
    if isinstance(ci, list):
        lines.append(
            f"\nBootstrap 95% CI for kappa: **[{ci[0]:.3f}, {ci[1]:.3f}]**. "
            "On a corpus this size the interval is wide, and differences "
            "between models should be read against it rather than against the "
            "point estimate alone.\n"
        )

    per = result.get("per_annotator") or {}
    if per:
        lines.append("\n## Per annotator\n")
        frame = pd.DataFrame(per).T.reset_index().rename(columns={"index": "annotator"})
        lines.append(dataframe_to_markdown(frame, float_format="{:.3f}"))

    rows = result.get("disagreements") or []
    lines.append(
        f"\n## Adjudication queue\n\n"
        f"**{result['n_pairs_needing_adjudication']}** pair(s) differ by "
        f"{result['adjudication_threshold']} or more points and require "
        f"adjudication.\n"
    )
    if rows:
        lines.append(dataframe_to_markdown(pd.DataFrame(rows)))

    lines.append(
        "\n> These are measured values from the annotation files on disk. "
        "They are reported as computed, whatever they show.\n"
    )
    return "\n".join(lines) + "\n"


def _strip_ci(result: dict[str, Any]) -> dict[str, Any]:
    """Drop the bulky disagreement list from the JSON summary, keeping the count."""
    out = {k: v for k, v in result.items() if k != "disagreements"}
    if "disagreements" in result:
        out["disagreements"] = result["disagreements"][:50]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute inter-annotator agreement from the annotation files."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--report", action="store_true", help="Write results/ and reports/ artefacts")
    parser.add_argument("--n-resamples", type=int, default=1000)
    args = parser.parse_args(argv)

    setup_logging("INFO")
    cfg = load_config(args.config)

    long_df = load_annotations(cfg)
    logger.info("Loaded %d annotation row(s).", len(long_df))

    result = compute_agreement(long_df, n_resamples=args.n_resamples, seed=int(cfg["project"]["seed"]))
    result = {"provenance": provenance(cfg), **result}

    if result["status"] == RESULT_PENDING:
        logger.warning("Agreement: %s — %s", RESULT_PENDING, result.get("reason"))
    else:
        logger.info(
            "Agreement: kappa=%.3f (%s), alpha=%s, ICC=%.3f over %d overlapping pairs.",
            result["cohen_kappa_quadratic"],
            result["cohen_kappa_band"],
            result["krippendorff_alpha_ordinal"],
            result["icc_2_1"],
            result["n_pairs_overlapping"],
        )

    if args.report:
        results_dir = resolve_path(cfg, "results_dir", create=True)
        reports_dir = resolve_path(cfg, "reports_dir", create=True)
        write_json(_strip_ci(result), results_dir / "agreement.json")
        (reports_dir / "agreement.md").write_text(render_report(result), encoding="utf-8")
        logger.info("Wrote %s", reports_dir / "agreement.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
