"""Loading, normalising and slicing the LMP corpus.

Everything that reads the master CSV goes through here, so that derived
columns (``score_bucket``, ``source_group``, ``decision``) are computed
identically everywhere and the gold/non-gold gate is applied in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal

import pandas as pd

from src.data import schema
from src.utils.config import load_config, resolve_path
from src.utils.io import read_csv
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

#: Which rows a caller wants.
#:
#: ``gold``  — human-annotated rows only.  The only valid input to a reported
#:             experiment.
#: ``draft`` — rows carrying a machine-drafted score (for pipeline smoke
#:             tests and for the annotation tool's pre-fill).
#: ``all``   — everything, including unannotated rows (for data statistics).
SubsetName = Literal["gold", "draft", "all"]


def master_path(cfg: dict | None = None) -> Path:
    """Absolute path of the master corpus CSV."""
    cfg = cfg or load_config()
    filename = Path(cfg["dataset"]["master_file"]).name
    return resolve_path(cfg, "processed_dir") / filename


def load_master(
    path: str | Path | None = None,
    cfg: dict | None = None,
    *,
    subset: SubsetName = "all",
    derive: bool = True,
) -> pd.DataFrame:
    """Load the master corpus.

    Parameters
    ----------
    path
        Explicit CSV path.  Defaults to ``data/processed/lmp_pairs.csv``.
    cfg
        Project config.
    subset
        ``"gold"``, ``"draft"`` or ``"all"`` — see :data:`SubsetName`.
    derive
        Compute ``score_bucket``, ``decision`` (where missing) and
        ``source_group``.

    Returns
    -------
    pandas.DataFrame
        A copy, safe for the caller to mutate.
    """
    cfg = cfg or load_config()
    csv_path = Path(path) if path else master_path(cfg)
    df = read_csv(csv_path)
    df = schema.ensure_columns(df)
    df = schema.coerce_types(df)

    if derive:
        df = add_derived_columns(df, cfg)

    if subset == "gold":
        df = schema.gold_subset(df)
    elif subset == "draft":
        df = df[df["annotation_status"] == "draft"].copy()
    elif subset != "all":
        raise ValueError(f"Unknown subset {subset!r}; expected gold/draft/all")

    logger.info(
        "Loaded %d rows (%s) from %s", len(df), subset, csv_path.name
    )
    return df.reset_index(drop=True)


def add_derived_columns(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Add ``score_bucket``, ``source_group`` and fill missing ``decision``.

    ``source_group`` is the anti-leakage key.  When the corpus does not supply
    one, it is derived from ``source`` + ``source_section`` so that all
    reformulations of the same provision stay together.  When neither is
    available, the row becomes its own group — which is safe but weaker, and
    is flagged by a warning.
    """
    cfg = cfg or load_config()
    df = df.copy()

    # -- score bucket ------------------------------------------------------
    df["score_bucket"] = df["human_lmp_score"].map(schema.score_bucket)
    # Fall back to the draft score so that unannotated rows can still be
    # stratified by their expected difficulty during annotation batching.
    if "draft_lmp_score" in df.columns:
        needs = df["score_bucket"].isna()
        df.loc[needs, "score_bucket"] = df.loc[needs, "draft_lmp_score"].map(
            schema.score_bucket
        )

    # -- decision ----------------------------------------------------------
    decision_cfg = cfg.get("decision", {})
    preserved_min = float(decision_cfg.get("preserved_min", 8.0))
    partial_min = float(decision_cfg.get("partial_min", 5.0))
    computed_decision = df["human_lmp_score"].map(
        lambda s: schema.score_to_decision(s, preserved_min, partial_min)
    )
    # Never overwrite a human-entered decision; only fill the gaps.
    if "decision" in df.columns:
        # A column that is entirely empty round-trips through CSV as float64,
        # which cannot hold the string labels we are about to write into it.
        if df["decision"].dtype != object:
            df["decision"] = df["decision"].astype("object")
        blank = df["decision"].isna() | (df["decision"].astype(str).str.strip() == "")
        df.loc[blank, "decision"] = computed_decision[blank]
    else:
        df["decision"] = computed_decision

    # -- source group ------------------------------------------------------
    if "source_group" not in df.columns:
        df["source_group"] = pd.NA

    blank_group = df["source_group"].isna() | (
        df["source_group"].astype(str).str.strip() == ""
    )
    if blank_group.any():
        source = df["source"].fillna("unknown").astype(str).str.strip()
        section = (
            df["source_section"].fillna("").astype(str).str.strip()
            if "source_section" in df.columns
            else pd.Series("", index=df.index)
        )
        derived = source + "::" + section
        # With no section number, fall back to the row's own id: a unique
        # group is conservative (it can never merge two different provisions)
        # but does not protect against un-labelled paraphrases.
        no_section = section == ""
        derived = derived.where(~no_section, source + "::" + df["pair_id"].astype(str))
        df.loc[blank_group, "source_group"] = derived[blank_group]

    return df


def assert_gold(df: pd.DataFrame, context: str = "") -> None:
    """Fail loudly if a caller tries to evaluate on non-human labels.

    This is the enforcement point for the project's central integrity rule:
    a machine-drafted score must never be reported as a human annotation.
    """
    if df.empty:
        raise ValueError(
            f"No rows available{(' for ' + context) if context else ''}. "
            "TODO_HUMAN_ANNOTATION — annotate the corpus before evaluating."
        )
    if "annotation_status" in df.columns:
        bad = ~df["annotation_status"].isin(schema.GOLD_STATUSES)
        if bad.any():
            offenders = sorted(df.loc[bad, "annotation_status"].astype(str).unique())
            raise ValueError(
                f"Refusing to evaluate {context or 'this data'}: "
                f"{int(bad.sum())} row(s) are not human-annotated "
                f"(statuses present: {offenders}). Only "
                f"{sorted(schema.GOLD_STATUSES)} count as ground truth."
            )


# ---------------------------------------------------------------------------
# Slicing helpers
# ---------------------------------------------------------------------------
def filter_rows(
    df: pd.DataFrame,
    languages: Iterable[str] | None = None,
    domains: Iterable[str] | None = None,
    splits: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return the rows matching every supplied filter (``None`` = no filter)."""
    out = df
    if languages is not None:
        wanted = list(languages)
        out = out[out["language"].isin(wanted)]
    if domains is not None:
        wanted = list(domains)
        out = out[out["legal_domain"].isin(wanted)]
    if splits is not None:
        wanted = list(splits)
        out = out[out["split"].isin(wanted)]
    return out.copy()


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return ``{"train": ..., "validation": ..., "test": ...}`` slices."""
    return {
        name: df[df["split"] == name].copy().reset_index(drop=True)
        for name in schema.SPLITS
    }


def summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compact corpus summary used in logs and the dashboard header."""
    return {
        "n_rows": int(len(df)),
        "languages": {
            str(k): int(v) for k, v in df["language"].value_counts().items()
        },
        "domains": {
            str(k): int(v) for k, v in df["legal_domain"].value_counts().items()
        },
        "splits": {
            str(k): int(v) for k, v in df["split"].value_counts().items()
        },
        "n_gold": int(schema.is_gold(df).sum()),
    }


def model_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Project the corpus to the columns the models consume.

    Keeping this projection explicit means a model can never accidentally
    read the label column as a feature.
    """
    columns = ["pair_id", "language", "legal_domain", "split",
               "original_text", "simplified_text", "human_lmp_score"]
    present = [c for c in columns if c in df.columns]
    out = df[present].copy()
    out = out.rename(columns={"human_lmp_score": "label"})
    return out
