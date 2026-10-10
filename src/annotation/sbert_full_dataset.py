"""
Full-dataset SBERT similarity baseline

This script computes cosine similarity between the original and simplified legal sentences
for every pair in ``data/processed/lmp_pairs.csv`` using the cached model
``sentence-transformers/paraphrase-multilingual-mpnet-base-v2``.

The output is written to ``data/processed/sbert_similarity_baseline.csv`` and
contains the following columns:
- pair_id
- language
- legal_domain
- split (train/validation/test as present in the source CSV)
- cosine_similarity (float in ``[-1, 1]``)

The script processes the data in batches to keep memory usage modest and validates
that every input pair yields a finite similarity score and that IDs are preserved.

Usage (from the project root)::

    .venv/bin/python src/annotation/sbert_full_dataset.py

The model weights are assumed to be already cached; no download will be attempted.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CSV_INPUT = DATA_DIR / "lmp_pairs.csv"
CSV_OUTPUT = DATA_DIR / "sbert_similarity_baseline.csv"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE = 64  # small enough for the 340‑row dataset


def main() -> None:
    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    try:
        df = pd.read_csv(CSV_INPUT)
    except Exception as exc:
        sys.exit(f"Failed to read {CSV_INPUT}: {exc}")

    required_cols = {"pair_id", "language", "legal_domain", "original_text", "simplified_text", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"Input CSV is missing required columns: {missing}")

    # -----------------------------------------------------------------------
    # Load model (cached)
    # -----------------------------------------------------------------------
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as exc:
        sys.exit(f"Failed to load model {MODEL_NAME}: {exc}")

    # -----------------------------------------------------------------------
    # Compute cosine similarities in batches
    # -----------------------------------------------------------------------
    similarities: list[float] = []
    num_rows = len(df)
    for start in range(0, num_rows, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_rows)
        batch = df.iloc[start:end]
        orig_texts = batch["original_text"].tolist()
        simp_texts = batch["simplified_text"].tolist()
        # Encode with normalisation so we can use dot‑product as cosine
        orig_emb = model.encode(orig_texts, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True)
        simp_emb = model.encode(simp_texts, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True)
        # Element‑wise cosine similarity (dot product of unit vectors)
        batch_cos = np.sum(orig_emb * simp_emb, axis=1).tolist()
        similarities.extend(batch_cos)

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------
    if len(similarities) != num_rows:
        sys.exit(f"Number of similarity scores ({len(similarities)}) does not match input rows ({num_rows})")

    sims_arr = np.array(similarities, dtype=float)
    if not np.all(np.isfinite(sims_arr)):
        sys.exit("Found non‑finite similarity values")
    # Allow a tiny tolerance beyond the theoretical [-1, 1] due to floating point rounding
    if not np.all((sims_arr >= -1.000001) & (sims_arr <= 1.000001)):
        sys.exit("Similarity values fall outside the expected range [-1, 1]")

    # Ensure IDs are unique and preserved
    if df["pair_id"].duplicated().any():
        sys.exit("Duplicate pair_id values found in the input CSV")

    # -----------------------------------------------------------------------
    # Write output CSV
    # -----------------------------------------------------------------------
    out_df = pd.DataFrame({
        "pair_id": df["pair_id"],
        "language": df["language"],
        "legal_domain": df["legal_domain"],
        "split": df["split"],
        "cosine_similarity": similarities,
    })
    out_df.to_csv(CSV_OUTPUT, index=False)
    print(f"✅ Completed: {num_rows} pairs written to {CSV_OUTPUT}")


if __name__ == "__main__":
    main()
