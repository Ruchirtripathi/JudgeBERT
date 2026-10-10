"""Canonical dataset schema for the LMP corpus.

This module is the single source of truth for:
  * which columns exist and what they mean,
  * which values are legal in each categorical column,
  * which annotation statuses count as human ground truth.

Every other module imports from here rather than re-declaring column names,
so a schema change is made in exactly one place.

Dataset integrity contract
--------------------------
``human_lmp_score`` is populated **only** by a human annotator.  A
machine-drafted score lives in ``draft_lmp_score`` and is marked with
``annotation_status == "draft"``.  :func:`is_gold` is the single gate that
evaluation code must pass a row through before treating its score as truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

#: ISO 639-1 code -> human-readable name.
LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
}

#: Legal domains present in the corpus.
#:
#: ``contract``   — Indian Contract Act, 1872.  Primary domain: it is a single
#:                  consolidated statute with numbered sections, official
#:                  public-domain text, and the closest Indian analogue to the
#:                  French insurance-contract setting of the original
#:                  JUDGEBERT work.
#: ``insurance``  — insurance-contract clauses; bridges to the original paper.
#: ``consumer``   — Consumer Protection Act, 2019.  Held out entirely from
#:                  training so that Experiment 3 measures genuine
#:                  cross-domain generalisation rather than interpolation.
DOMAINS: dict[str, str] = {
    "contract": "Contract Law (Indian Contract Act, 1872)",
    "cyber": "Cyber Law (Information Technology Act, 2000)",
    "sale_of_goods": "Sale of Goods (Sale of Goods Act, 1930)",
}

#: Three-way decision derived from the continuous LMP score.
DECISIONS: list[str] = ["Preserved", "Partially Preserved", "Not Preserved"]

#: Legal error taxonomy (single-label).
ERROR_LABELS: list[str] = [
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

#: Annotation lifecycle.  Only ``GOLD_STATUSES`` may serve as ground truth.
ANNOTATION_STATUSES: list[str] = [
    "unannotated",
    "draft",
    "human_single",
    "human_multi_pending",
    "human_consensus",
]
GOLD_STATUSES: frozenset[str] = frozenset(
    {"human_single", "human_multi_pending", "human_consensus"}
)
NON_GOLD_STATUSES: frozenset[str] = frozenset({"unannotated", "draft"})

#: Train/validation/test split names.
SPLITS: list[str] = ["train", "validation", "test"]

#: Score range for human LMP annotation (inclusive).
SCORE_MIN: int = 1
SCORE_MAX: int = 10

#: The three coarse bands used for stratification and imbalance reporting.
SCORE_BUCKETS: list[tuple[int, int]] = [(1, 3), (4, 7), (8, 10)]
SCORE_BUCKET_LABELS: list[str] = ["low", "mid", "high"]

#: ``source_group`` values used to keep sibling reformulations in one split.
#: Computed in :mod:`src.data.load_data`; declared here for documentation.


# ---------------------------------------------------------------------------
# Column specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Column:
    """Specification of a single dataset column."""

    name: str
    dtype: str  # "str" | "int" | "float" | "category"
    required: bool
    description: str
    allowed_values: tuple[str, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    #: Empty string is acceptable (e.g. ``context``, ``error_type`` == "None")
    allow_empty: bool = False
    #: Column may not be null in *gold* rows only (e.g. human_lmp_score).
    required_for_gold: bool = False

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.name


SCHEMA: list[Column] = [
    Column(
        name="pair_id",
        dtype="str",
        required=True,
        description=(
            "Stable unique identifier. Convention: <LANG>_<DOMAIN>_<NNN>, "
            "e.g. ENG_CON_001 / HIN_CON_001. Must be unique across the corpus."
        ),
    ),
    Column(
        name="language",
        dtype="category",
        required=True,
        description="ISO 639-1 language code of both sentences in the pair.",
        allowed_values=tuple(LANGUAGES.keys()),
    ),
    Column(
        name="legal_domain",
        dtype="category",
        required=True,
        description="Legal domain the sentence pair is drawn from.",
        allowed_values=tuple(DOMAINS.keys()),
    ),
    Column(
        name="original_text",
        dtype="str",
        required=True,
        description="The source legal sentence (authoritative wording).",
    ),
    Column(
        name="simplified_text",
        dtype="str",
        required=True,
        description="The simplified / reformulated legal sentence being judged.",
    ),
    Column(
        name="human_lmp_score",
        dtype="float",
        required=True,
        description=(
            "HUMAN-ANNOTATED Legal Meaning Preservation score on a 1-10 scale. "
            "Never populated by a model. Empty until a human annotator has "
            "scored the pair."
        ),
        minimum=SCORE_MIN,
        maximum=SCORE_MAX,
        required_for_gold=True,
    ),
    Column(
        name="decision",
        dtype="category",
        required=True,
        description=(
            "Three-way decision derived from human_lmp_score using the "
            "calibrated thresholds in the config. Empty until annotated."
        ),
        allowed_values=tuple(DECISIONS),
    ),
    Column(
        name="error_type",
        dtype="category",
        required=True,
        description=(
            "Primary legal error introduced by the simplification. "
            "'None' when meaning is fully preserved."
        ),
        allowed_values=tuple(ERROR_LABELS),
    ),
    Column(
        name="source",
        dtype="str",
        required=True,
        description=(
            "Provenance of the original sentence, e.g. "
            "'Indian Contract Act 1872'. Free text but must be non-empty."
        ),
    ),
    Column(
        name="split",
        dtype="category",
        required=True,
        description="Assigned partition: train / validation / test.",
        allowed_values=tuple(SPLITS),
    ),
    # --- optional columns -------------------------------------------------
    Column(
        name="context",
        dtype="str",
        required=False,
        description=(
            "Optional surrounding legal context (neighbouring sub-clauses). "
            "Used by the context-aware variant of Experiment 3."
        ),
        allow_empty=True,
    ),
    Column(
        name="source_section",
        dtype="str",
        required=False,
        description="Section / article number within the source instrument.",
        allow_empty=True,
    ),
    Column(
        name="source_url",
        dtype="str",
        required=False,
        description="URL the original sentence was taken from.",
        allow_empty=True,
    ),
    Column(
        name="simplification_type",
        dtype="str",
        required=False,
        description=(
            "How the simplification was produced, e.g. 'author_rewrite', "
            "'clause_split', 'synonym_substitution', 'adversarial_edit'."
        ),
        allow_empty=True,
    ),
    Column(
        name="source_group",
        dtype="str",
        required=False,
        description=(
            "Anti-leakage grouping key. All rows sharing a source_group are "
            "placed in the same split. Derived from the source instrument and "
            "section if not supplied."
        ),
        allow_empty=True,
    ),
    Column(
        name="annotation_status",
        dtype="category",
        required=False,
        description=(
            "Lifecycle state of the annotation. Only human_* values are gold. "
            "Defaults to 'unannotated' when absent."
        ),
        allowed_values=tuple(ANNOTATION_STATUSES),
    ),
    Column(
        name="draft_lmp_score",
        dtype="float",
        required=False,
        description=(
            "CURATOR-DRAFTED suggestion, recorded while the corpus was built to "
            "steer the score distribution. Never ground truth, and never shown "
            "to an annotator: the annotation tool is blind, so this column is "
            "not rendered before *or* after a human scores the pair. Retained "
            "so the curator can measure post hoc how the drafts relate to the "
            "human labels (src/annotation/curator_review.py)."
        ),
        minimum=SCORE_MIN,
        maximum=SCORE_MAX,
    ),
    Column(
        name="draft_error_type",
        dtype="category",
        required=False,
        description=(
            "CURATOR-DRAFTED primary error type — the counterpart of "
            "draft_lmp_score, and equally not ground truth. It records the "
            "phenomenon the pair was built to exhibit, which is what lets "
            "Experiment 5 sample deliberately by phenomenon (Date, Number and "
            "Entity in particular exist nowhere else until human annotation "
            "is complete). Never confused with error_type."
        ),
        allowed_values=tuple(ERROR_LABELS),
        allow_empty=True,
    ),
    Column(
        name="annotator_id",
        dtype="str",
        required=False,
        description="Pseudonymous identifier of the annotator. Never a real name.",
        allow_empty=True,
    ),
    Column(
        name="annotated_at",
        dtype="str",
        required=False,
        description="ISO-8601 UTC timestamp of annotation.",
        allow_empty=True,
    ),
    Column(
        name="verbatim_quote",
        dtype="str",
        required=False,
        description=(
            "'true' if original_text is a verbatim quotation of the cited "
            "source section, 'false' if it was lightly normalised, "
            "'unverified' if provenance has not yet been checked."
        ),
        allow_empty=True,
    ),
    Column(
        name="notes",
        dtype="str",
        required=False,
        description="Free-text annotator or curator notes.",
        allow_empty=True,
    ),
    Column(
        name="score_bucket",
        dtype="category",
        required=False,
        description=(
            "DERIVED. Coarse band of the LMP score: low (1-3), mid (4-7) or "
            "high (8-10). Recomputed on every load by "
            "src.data.load_data.add_derived_columns from human_lmp_score when "
            "it exists, falling back to draft_lmp_score so that unannotated "
            "rows can still be stratified for annotation batching. Never "
            "edited by hand."
        ),
        allowed_values=tuple(SCORE_BUCKET_LABELS),
        allow_empty=True,
    ),
]

COLUMNS: dict[str, Column] = {c.name: c for c in SCHEMA}
REQUIRED_COLUMNS: list[str] = [c.name for c in SCHEMA if c.required]
OPTIONAL_COLUMNS: list[str] = [c.name for c in SCHEMA if not c.required]
ALL_COLUMNS: list[str] = [c.name for c in SCHEMA]


# ---------------------------------------------------------------------------
# Frame construction and coercion
# ---------------------------------------------------------------------------
def empty_frame(include_optional: bool = True) -> pd.DataFrame:
    """An empty master frame with every column present and correctly typed.

    Useful for the annotation tool (which appends to a template) and for
    tests that need a well-formed but empty input.
    """
    names = ALL_COLUMNS if include_optional else REQUIRED_COLUMNS
    data: dict[str, pd.Series] = {}
    for name in names:
        col = COLUMNS[name]
        if col.dtype in {"float"}:
            data[name] = pd.Series(dtype="float64")
        elif col.dtype in {"int"}:
            data[name] = pd.Series(dtype="Int64")
        else:
            data[name] = pd.Series(dtype="object")
    return pd.DataFrame(data)


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing optional columns as empty, and drop unknown extras.

    Unknown columns are *kept* but reported by validation rather than silently
    dropped, so that a typo in an optional column name is not swallowed.
    """
    df = df.copy()
    for name in OPTIONAL_COLUMNS:
        if name not in df.columns:
            df[name] = pd.NA
    # Put columns in canonical order, leaving unrecognised ones at the end.
    ordered = [c for c in ALL_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in ALL_COLUMNS]
    return df[ordered + extras]


def _normalise_text_value(value: Any) -> Any:
    """Normalise one value of a declared-text column.

    Two round-trip hazards are handled here.

    * Surrounding whitespace is stripped from strings.
    * A text column whose every value happens to be ``"true"`` or ``"false"``
      is read back from CSV as a *boolean* column.  ``verbatim_quote`` is
      exactly that column, and leaving it as ``True``/``False`` would give the
      corpus two spellings of the same value depending on how it was loaded.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    return value


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce each known column to its declared dtype without destroying NaNs.

    Text columns have surrounding whitespace stripped.  ``human_lmp_score``
    and ``draft_lmp_score`` are coerced to float, with empty strings becoming
    ``NaN`` (an unannotated row) rather than ``0.0`` (a wrong score of zero).
    """
    df = df.copy()
    for name, col in COLUMNS.items():
        if name not in df.columns:
            continue
        series = df[name]

        if col.dtype == "float":
            df[name] = pd.to_numeric(series, errors="coerce").astype("float64")
        elif col.dtype == "int":
            df[name] = pd.to_numeric(series, errors="coerce").astype("Int64")
        else:
            # Normalise to string, mapping true nulls to pd.NA (not the
            # literal text "nan", which pandas produces from a float column).
            cleaned = series.astype("object").where(series.notna(), pd.NA)
            cleaned = cleaned.map(_normalise_text_value)
            df[name] = cleaned

    # annotation_status defaults to 'unannotated' when absent or blank.
    if "annotation_status" in df.columns:
        status = df["annotation_status"]
        df["annotation_status"] = status.where(
            status.notna() & (status.astype(str).str.len() > 0), "unannotated"
        )

    if "error_type" in df.columns:
        # A blank error type on a *human-annotated* row means "None" — the
        # annotator looked and found no error.  A blank error type on an
        # unannotated row must stay blank: filling it with "None" would assert
        # that no legal error was introduced, which is a claim no human has
        # made yet.
        err = df["error_type"]
        blank = err.isna() | (err.astype(str).str.strip() == "")
        annotated = df.get("annotation_status")
        if annotated is not None:
            human = annotated.isin(GOLD_STATUSES)
        else:
            human = pd.Series(False, index=df.index)
        df["error_type"] = err.where(~(blank & human), "None")

    return df


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------
def score_bucket(score: Any) -> str | None:
    """Map a 1-10 score to ``low`` / ``mid`` / ``high``; ``None`` if missing."""
    if score is None or (isinstance(score, float) and score != score) or pd.isna(score):
        return None
    value = float(score)
    for (low, high), label in zip(SCORE_BUCKETS, SCORE_BUCKET_LABELS):
        if low <= value <= high:
            return label
    return None


def score_to_decision(
    score: Any, preserved_min: float = 8.0, partial_min: float = 5.0
) -> str | None:
    """Apply the configured thresholds to a score.

    Kept here (rather than in the evaluation package) because the annotation
    tool also uses it to pre-fill the ``decision`` column.
    """
    if score is None or pd.isna(score):
        return None
    value = float(score)
    if value >= preserved_min:
        return "Preserved"
    if value >= partial_min:
        return "Partially Preserved"
    return "Not Preserved"


def is_gold(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows whose score is human ground truth.

    A row is gold when its ``annotation_status`` is one of
    :data:`GOLD_STATUSES` **and** it actually carries a score.  The second
    condition guards against a row marked ``human_single`` whose score column
    was accidentally cleared.
    """
    if "annotation_status" not in df.columns or "human_lmp_score" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    status_ok = df["annotation_status"].isin(GOLD_STATUSES)
    score_ok = df["human_lmp_score"].notna()
    return (status_ok & score_ok).fillna(False)


def gold_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the human-annotated rows (the only evaluable subset)."""
    return df.loc[is_gold(df)].copy()


def describe_schema() -> str:
    """Human-readable schema description, embedded into data/README.md."""
    lines = []
    for col in SCHEMA:
        req = "required" if col.required else "optional"
        rng = ""
        if col.minimum is not None and col.maximum is not None:
            rng = f" (range {col.minimum:g}-{col.maximum:g})"
        elif col.allowed_values:
            rng = " in {" + ", ".join(col.allowed_values) + "}"
        lines.append(f"- **{col.name}** _({req}, {col.dtype}){rng}_ — {col.description}")
    return "\n".join(lines)


def validate_vocabulary(values: Iterable[Any], allowed: Iterable[str]) -> list[str]:
    """Return the values present in ``values`` that are not in ``allowed``."""
    allowed_set = set(allowed)
    return sorted({str(v) for v in values if pd.notna(v)} - allowed_set)
