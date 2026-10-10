"""Build the master corpus from the authored JSONL batches.

Pipeline position::

    data/raw/authored/*.jsonl  --build_dataset-->  data/processed/lmp_pairs.csv
                                                   data/processed/curator_drafts.csv
                                                   data/processed/build_manifest.json

The authored files are the curator's working notes: they carry the source
sentences, the rewrites, and a **draft** score recorded while the corpus was
being constructed (used to steer the score distribution).  The master corpus is
the artefact the rest of the project reads.

The separation between the two is the whole point of this module
----------------------------------------------------------------
The project's central integrity rule is that a machine- or curator-drafted
score must never be mistaken for a human LMP annotation.  This module enforces
that structurally rather than by convention:

* ``human_lmp_score``, ``error_type`` and ``decision`` are written **empty for
  every single row**.  No non-human judgement ever occupies a gold column, so
  the master corpus cannot be contaminated even by a downstream mistake.
* The draft score lives in ``draft_lmp_score``, a column the schema explicitly
  designates as non-gold, with ``annotation_status == "draft"``.  The schema's
  :func:`~src.data.schema.is_gold` gate excludes it, so
  :func:`~src.data.load_data.assert_gold` refuses to evaluate on it.
* The drafts are also written to a plain ``curator_drafts.csv`` side-car so
  they are inspectable without opening the master.  That side-car lives under
  ``data/processed/``, **not** under ``data/annotations/``: it carries the
  draft score and the designed error type for every pair, so an annotator who
  opened it would see exactly the expected label that the annotation tool
  withholds.  Keeping it out of the annotator-facing directory is the point —
  the file is curator material, and the annotation interface never reads it.

Provenance
----------
Every row is cross-checked against :data:`src.data.extract_sources.SOURCE_REGISTRY`:
the ``legal_domain`` and ``language`` recorded in the authored file must match
the registry entry for the statute the row cites.  A mismatch is a hard error,
because it would mean the corpus misattributes a sentence to the wrong Act.

Usage
-----
::

    python -m src.data.build_dataset                 # build, then split
    python -m src.data.build_dataset --no-split      # build only
    python -m src.data.build_dataset --check         # report, write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import schema
from src.data.extract_sources import SOURCE_REGISTRY
from src.utils.config import load_config, resolve_path, set_global_seed
from src.utils.io import provenance, write_csv, write_json
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

#: Keys every authored record must carry, exactly.
AUTHORED_KEYS: frozenset[str] = frozenset(
    {
        "pair_id",
        "language",
        "legal_domain",
        "source_section",
        "source_group",
        "verbatim_quote",
        "simplification_type",
        "original_text",
        "simplified_text",
        "context",
        "draft_lmp_score",
        "draft_error_type",
        "curator_notes",
    }
)

#: ``source_group`` prefix -> (statute citation, English source key, Hindi key).
#:
#: The Hindi key is recorded for provenance only.  Every official Hindi PDF in
#: the registry has ``encoding_ok=False`` (see reports/source_provenance.md),
#: so no Hindi ``original_text`` in this corpus was machine-extracted from it:
#: each was rendered by hand from the English text and is therefore marked
#: ``verbatim_quote: "false"``.
STATUTE_REGISTRY: dict[str, tuple[str, str, str]] = {
    "ICA1872": ("Indian Contract Act 1872", "contract_act_1872_eng", "contract_act_1872_hin"),
    "ITA2000": ("Information Technology Act 2000", "it_act_2000_eng", "it_act_2000_hin"),
    "SOGA1930": ("Sale of Goods Act 1930", "sale_of_goods_1930_eng", "sale_of_goods_1930_hin"),
}

#: Optional free-text columns where the empty string should become a true null,
#: so that "not supplied" has exactly one representation in the master.
_NULLABLE_TEXT = ("context", "annotator_id", "annotated_at", "notes")


class BuildError(RuntimeError):
    """Raised when the authored batches cannot be assembled into a corpus."""


# ---------------------------------------------------------------------------
# Reading the authored batches
# ---------------------------------------------------------------------------
def read_authored(authored_dir: Path) -> list[dict[str, Any]]:
    """Read and structurally validate every ``*.jsonl`` batch.

    A record with a missing, misspelled or unexpected key is a hard failure:
    a typo such as ``simplified_type`` would otherwise silently drop a column.
    """
    paths = sorted(authored_dir.glob("*.jsonl"))
    if not paths:
        raise BuildError(
            f"No authored batches found in {authored_dir}. Expected one or more "
            f"*.jsonl files with the keys {sorted(AUTHORED_KEYS)}."
        )

    records: list[dict[str, Any]] = []
    for path in paths:
        n_before = len(records)
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BuildError(f"{path.name}:{lineno}: invalid JSON — {exc}") from exc

                keys = set(record)
                if keys != AUTHORED_KEYS:
                    missing = sorted(AUTHORED_KEYS - keys)
                    extra = sorted(keys - AUTHORED_KEYS)
                    detail = []
                    if missing:
                        detail.append(f"missing {missing}")
                    if extra:
                        detail.append(f"unexpected {extra}")
                    raise BuildError(
                        f"{path.name}:{lineno} ({record.get('pair_id', '?')}): "
                        f"key mismatch — {'; '.join(detail)}"
                    )
                records.append(record)
        logger.info("%s: %d rows", path.name, len(records) - n_before)

    return records


# ---------------------------------------------------------------------------
# Mapping to the master schema
# ---------------------------------------------------------------------------
def _statute_for(source_group: str) -> tuple[str, str, str]:
    """Resolve a ``source_group`` to its statute entry."""
    prefix = str(source_group).split("::", 1)[0]
    if prefix not in STATUTE_REGISTRY:
        raise BuildError(
            f"Unknown statute prefix {prefix!r} in source_group {source_group!r}. "
            f"Known prefixes: {sorted(STATUTE_REGISTRY)}"
        )
    return STATUTE_REGISTRY[prefix]


def to_master_rows(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split authored records into (master rows, draft side-car rows).

    Every field that represents a *judgement* about the pair — the LMP score,
    the error type, the three-way decision — is left empty in the master.  The
    draft values go to the side-car.
    """
    master: list[dict[str, Any]] = []
    drafts: list[dict[str, Any]] = []

    for record in records:
        pair_id = str(record["pair_id"])
        language = str(record["language"])
        source_group = str(record["source_group"])
        statute, eng_key, hin_key = _statute_for(source_group)
        source_key = eng_key if language == "en" else hin_key
        source_meta = SOURCE_REGISTRY[source_key]

        # -- provenance cross-check ---------------------------------------
        if source_meta["legal_domain"] != record["legal_domain"]:
            raise BuildError(
                f"{pair_id}: legal_domain is {record['legal_domain']!r} but the "
                f"cited statute {statute!r} is registered as "
                f"{source_meta['legal_domain']!r}."
            )
        if source_meta["language"] != language:
            raise BuildError(
                f"{pair_id}: language is {language!r} but source {source_key!r} "
                f"is registered as {source_meta['language']!r}."
            )

        verbatim = str(record["verbatim_quote"]).lower()
        if verbatim not in {"true", "false", "unverified"}:
            raise BuildError(
                f"{pair_id}: verbatim_quote must be true/false/unverified, "
                f"got {record['verbatim_quote']!r}"
            )
        if language == "hi" and verbatim == "true":
            raise BuildError(
                f"{pair_id}: Hindi original claimed as a verbatim quote, but no "
                f"official Hindi source in the registry is machine-extractable. "
                f"Mark it 'false'."
            )

        source_label = statute if language == "en" else f"{statute} (author-rendered Hindi)"

        row: dict[str, Any] = {
            "pair_id": pair_id,
            "language": language,
            "legal_domain": record["legal_domain"],
            "original_text": record["original_text"],
            "simplified_text": record["simplified_text"],
            # --- gold columns: deliberately empty, for every row ----------
            "human_lmp_score": pd.NA,
            "decision": pd.NA,
            "error_type": pd.NA,
            # --------------------------------------------------------------
            "source": source_label,
            "split": pd.NA,  # assigned by assign_splits()
            "context": record["context"],
            "source_section": str(record["source_section"]),
            "source_url": source_meta["retrieved_from"],
            "simplification_type": record["simplification_type"],
            "source_group": source_group,
            # 'draft' means "a machine/curator suggestion exists; NOT truth".
            "annotation_status": "draft",
            "draft_lmp_score": record["draft_lmp_score"],
            # The curator's structured error type travels with the master, not
            # only into curator_drafts.csv: Experiment 5 samples by phenomenon, and for
            # Date / Number / Entity this is the only such signal that exists
            # before human annotation is complete.
            "draft_error_type": record["draft_error_type"],
            "annotator_id": pd.NA,
            "annotated_at": pd.NA,
            "verbatim_quote": verbatim,
            "notes": record["curator_notes"],
        }
        master.append(row)

        drafts.append(
            {
                "pair_id": pair_id,
                "language": language,
                "legal_domain": record["legal_domain"],
                "source_group": source_group,
                "source_section": record["source_section"],
                "simplification_type": record["simplification_type"],
                "draft_lmp_score": record["draft_lmp_score"],
                "draft_error_type": record["draft_error_type"],
                "verbatim_quote": verbatim,
                "curator_notes": record["curator_notes"],
            }
        )

    master_df = schema.ensure_columns(pd.DataFrame(master))
    drafts_df = pd.DataFrame(drafts)

    # One representation for "not supplied".
    for column in _NULLABLE_TEXT:
        if column in master_df.columns:
            blank = master_df[column].astype("object").map(
                lambda v: isinstance(v, str) and v.strip() == ""
            )
            master_df.loc[blank, column] = pd.NA

    # Deterministic, human-readable ordering.
    master_df = master_df.sort_values(
        ["legal_domain", "source_group", "language", "pair_id"], kind="stable"
    ).reset_index(drop=True)
    drafts_df = drafts_df.sort_values(
        ["legal_domain", "source_group", "language", "pair_id"], kind="stable"
    ).reset_index(drop=True)

    return schema.coerce_types(master_df), drafts_df


# ---------------------------------------------------------------------------
# Integrity assertions on the assembled corpus
# ---------------------------------------------------------------------------
def assert_no_premature_judgements(df: pd.DataFrame) -> None:
    """Refuse to write a corpus containing any non-human judgement.

    This is the check that makes the module's central promise enforceable: if a
    future edit to :func:`to_master_rows` ever populated a gold column from the
    authored file, the build fails here rather than shipping a contaminated
    corpus.
    """
    for column in ("human_lmp_score", "error_type", "decision"):
        populated = df[column].notna() & (df[column].astype(str).str.strip() != "")
        if populated.any():
            offenders = df.loc[populated, "pair_id"].astype(str).head(5).tolist()
            raise BuildError(
                f"'{column}' is populated on {int(populated.sum())} row(s) "
                f"({offenders}). These are human-annotation columns and must be "
                f"empty until a human annotates the corpus. Draft values belong "
                f"in draft_lmp_score / data/processed/curator_drafts.csv."
            )

    bad_status = ~df["annotation_status"].isin(schema.NON_GOLD_STATUSES)
    if bad_status.any():
        offenders = sorted(df.loc[bad_status, "annotation_status"].astype(str).unique())
        raise BuildError(
            f"Freshly built corpus must be entirely non-gold, but found statuses "
            f"{offenders}. Only {sorted(schema.NON_GOLD_STATUSES)} are valid here."
        )


def check_balance(df: pd.DataFrame, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise the corpus and flag anything the brief requires but is missing."""
    languages = Counter(df["language"])
    domains = Counter(df["legal_domain"])
    scores = Counter(df["draft_lmp_score"].astype("Int64").dropna().astype(int))
    error_types = Counter(str(r["draft_error_type"]) for r in records)

    groups = defaultdict(set)
    for row in df.itertuples(index=False):
        groups[str(row.source_group)].add(str(row.language))

    warnings: list[str] = []
    if len(df) < 300:
        warnings.append(f"only {len(df)} pairs; the brief requires at least 300")
    for code in ("en", "hi"):
        if languages.get(code, 0) < 150:
            warnings.append(
                f"language {code!r} has {languages.get(code, 0)} rows; "
                f"the brief targets 150+ per language"
            )
    missing_scores = [s for s in range(1, 11) if s not in scores]
    if missing_scores:
        warnings.append(f"no pairs at LMP score(s) {missing_scores}")
    # The error taxonomy the brief's Experiment 5 must report on.
    required_phenomena = {
        "Omission", "Hallucination", "Contradiction", "Obligation/Permission",
        "Condition", "Exception", "Scope", "Date", "Number", "Entity",
    }
    absent = sorted(required_phenomena - set(error_types))
    if absent:
        warnings.append(f"no pairs exercising error type(s) {absent}")
    unpaired = sorted(k for k, v in groups.items() if v != {"en", "hi"})

    for warning in warnings:
        logger.warning("corpus balance: %s", warning)

    return {
        "n_rows": int(len(df)),
        "n_languages": {str(k): int(v) for k, v in sorted(languages.items())},
        "n_domains": {str(k): int(v) for k, v in sorted(domains.items())},
        "draft_score_histogram": {str(k): int(v) for k, v in sorted(scores.items())},
        "draft_error_type_counts": {
            str(k): int(v) for k, v in sorted(error_types.items(), key=lambda kv: -kv[1])
        },
        "n_source_groups": len(groups),
        "n_verbatim_originals": int((df["verbatim_quote"] == "true").sum()),
        "n_author_rendered_originals": int((df["verbatim_quote"] == "false").sum()),
        "groups_without_both_languages": unpaired,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build(
    cfg: dict[str, Any] | None = None,
    *,
    authored_dir: Path | None = None,
    assign_split: bool = True,
    check_only: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assemble the master corpus and, unless ``check_only``, write it out."""
    cfg = cfg or load_config()
    root = resolve_path(cfg, "root")
    authored_dir = authored_dir or (root / cfg["paths"]["raw_dir"] / "authored")

    records = read_authored(authored_dir)
    master_df, drafts_df = to_master_rows(records)
    assert_no_premature_judgements(master_df)

    if assign_split:
        # Imported here so that the splitter stays independently runnable.
        from src.data.load_data import add_derived_columns
        from src.data.split_data import assign_splits

        master_df = add_derived_columns(master_df, cfg)
        master_df = assign_splits(master_df, cfg)

    manifest: dict[str, Any] = {
        "provenance": provenance(cfg, seed=cfg["project"]["seed"]),
        "authored_dir": str(authored_dir.relative_to(root)),
        "authored_files": sorted(p.name for p in authored_dir.glob("*.jsonl")),
        "split_assigned": bool(assign_split),
        **check_balance(master_df, records),
        "integrity": {
            "human_lmp_score_populated": int(master_df["human_lmp_score"].notna().sum()),
            "error_type_populated": int(master_df["error_type"].notna().sum()),
            "decision_populated": int(master_df["decision"].notna().sum()),
            "note": (
                "All three are zero by construction. The corpus contains no "
                "human annotation yet; every score is a curator draft and is "
                "excluded from ground truth by src.data.schema.is_gold."
            ),
        },
    }

    if check_only:
        return master_df, manifest

    master_path = resolve_path(cfg, "processed_dir", create=True) / Path(
        cfg["dataset"]["master_file"]
    ).name
    write_csv(master_df, master_path)

    # Curator material.  It lives beside the master, not under data/annotations/,
    # because it contains the draft score and designed error type for every pair.
    drafts_path = resolve_path(cfg, "processed_dir", create=True) / "curator_drafts.csv"
    write_csv(drafts_df, drafts_path)

    write_json(
        manifest,
        resolve_path(cfg, "processed_dir", create=True) / "build_manifest.json",
    )
    return master_df, manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build data/processed/lmp_pairs.csv from the authored batches."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--authored-dir", type=str, default=None)
    parser.add_argument("--no-split", action="store_true",
                        help="Do not assign train/validation/test splits")
    parser.add_argument("--check", action="store_true",
                        help="Report on the corpus without writing anything")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging("INFO")
    set_global_seed(cfg["project"]["seed"])

    authored_dir = Path(args.authored_dir) if args.authored_dir else None

    try:
        df, manifest = build(
            cfg,
            authored_dir=authored_dir,
            assign_split=not args.no_split,
            check_only=args.check,
        )
    except BuildError as exc:
        logger.error("Build failed: %s", exc)
        return 1

    print()
    print(f"Rows                 {manifest['n_rows']}")
    print(f"Languages            {manifest['n_languages']}")
    print(f"Domains              {manifest['n_domains']}")
    print(f"Source groups        {manifest['n_source_groups']}")
    print(f"Verbatim originals   {manifest['n_verbatim_originals']}")
    print(f"Author-rendered      {manifest['n_author_rendered_originals']} (Hindi; see reports/source_provenance.md)")
    print(f"Draft score spread   {manifest['draft_score_histogram']}")
    print(f"Draft error types    {manifest['draft_error_type_counts']}")
    if manifest["warnings"]:
        print("\nWARNINGS")
        for warning in manifest["warnings"]:
            print(f"  - {warning}")
    print()
    print("Human annotations present: 0 — the corpus is unannotated. "
          "TODO_HUMAN_ANNOTATION")
    if args.check:
        print("(--check: nothing was written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
