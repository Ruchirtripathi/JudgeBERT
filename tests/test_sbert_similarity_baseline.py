import pathlib
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
DATA_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_CSV = DATA_DIR / "lmp_pairs.csv"
OUTPUT_CSV = DATA_DIR / "sbert_similarity_baseline.csv"

def test_output_exists_and_schema():
    assert OUTPUT_CSV.exists(), f"Output file {OUTPUT_CSV} not found"
    out_df = pd.read_csv(OUTPUT_CSV)
    # Required columns
    required = {"pair_id", "language", "legal_domain", "split", "cosine_similarity"}
    assert required.issubset(set(out_df.columns)), f"Missing columns: {required - set(out_df.columns)}"
    # Row count matches input
    in_df = pd.read_csv(INPUT_CSV)
    assert len(out_df) == len(in_df), "Row count mismatch between input and output"
    # IDs preserved and unique
    assert out_df["pair_id"].tolist() == in_df["pair_id"].tolist(), "pair_id order changed"
    assert out_df["pair_id"].is_unique, "Duplicate pair_id in output"
    # Similarity values are finite and within [-1, 1] (tolerance)
    sims = out_df["cosine_similarity"].astype(float)
    assert sims.notnull().all(), "Found NaN similarity values"
    assert ((sims >= -1.000001) & (sims <= 1.000001)).all(), "Similarity out of expected range"
