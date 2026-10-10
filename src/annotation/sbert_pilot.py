"""sbert_pilot.py

Small pilot script that demonstrates how to compute a cosine similarity between the
original and simplified sentences for a handful of legal sentence pairs using the
multilingual SBERT model `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.

The script:
1. Loads the CSV ``data/processed/lmp_pairs.csv`` and selects five representative
   rows (the first five rows of the file – they contain both English and Hindi
   examples).
2. Loads the SBERT model via ``SentenceTransformer``.
3. Encodes the ``original_text`` and ``simplified_text`` columns.
4. Computes cosine similarity for each pair.
5. Prints the following fields for every selected pair:
   * ``pair_id``
   * ``language``
   * ``legal_domain``
   * ``cosine_similarity`` (rounded to 4 decimal places)
   * ``original_text``
   * ``simplified_text``

If the model weights are not already cached locally the ``SentenceTransformer``
constructor will attempt to download them, which is disallowed in this sandbox.
The script catches such errors, informs the user, and exits without performing
any download.
"""

import csv
import sys
from pathlib import Path
from typing import List

# Import libraries – they are declared in ``requirements.txt``.
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print("ERROR: Could not import SentenceTransformer. Ensure the package is installed in the current environment.")
    print(f"Details: {e}")
    sys.exit(1)

import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two 1‑D vectors."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def load_pairs(csv_path: Path, n: int = 5) -> List[dict]:
    """Read the CSV and return the first *n* rows as dictionaries.

    The CSV header defines the column names; we keep only the columns we need.
    """
    pairs = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            pairs.append(row)
    return pairs

def main() -> None:
    csv_path = Path(__file__).parents[2] / "data" / "processed" / "lmp_pairs.csv"
    if not csv_path.is_file():
        print(f"ERROR: Could not find CSV at {csv_path}")
        sys.exit(1)

    # Load a small sample of pairs.
    pairs = load_pairs(csv_path, n=5)
    if not pairs:
        print("ERROR: CSV appears to be empty.")
        sys.exit(1)

    # Load the SBERT model.
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        # Most likely a download attempt failed.
        print("ERROR: Failed to load the SBERT model. The required weights are not cached locally and downloading is not permitted in this environment.")
        print(f"Details: {e}")
        sys.exit(1)

    # Encode all original and simplified texts in a single batch for efficiency.
    originals = [p["original_text"] for p in pairs]
    simplified = [p["simplified_text"] for p in pairs]
    try:
        orig_emb = model.encode(originals, convert_to_numpy=True, show_progress_bar=False)
        simp_emb = model.encode(simplified, convert_to_numpy=True, show_progress_bar=False)
    except Exception as e:
        print("ERROR: Encoding failed – this may indicate a problem with the model or the runtime environment.")
        print(f"Details: {e}")
        sys.exit(1)

    # Compute and print results.
    for i, pair in enumerate(pairs):
        sim = cosine_similarity(orig_emb[i], simp_emb[i])
        print("---")
        print(f"pair_id: {pair['pair_id']}")
        print(f"language: {pair['language']}")
        print(f"legal_domain: {pair['legal_domain']}")
        print(f"cosine_similarity: {sim:.4f}")
        print(f"original_text: {pair['original_text']}")
        print(f"simplified_text: {pair['simplified_text']}")

if __name__ == "__main__":
    main()
