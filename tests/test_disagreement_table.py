import pandas as pd
from src.annotation import agreement

def test_disagreement_table_empty_matrix_returns_schema():
    # Empty matrix (no annotators, no rows)
    empty_matrix = pd.DataFrame()
    df = agreement.disagreement_table(empty_matrix)
    assert list(df.columns) == ["pair_id", "min_score", "max_score", "spread", "annotators"]
    assert df.empty

def test_disagreement_table_no_dispute_returns_schema():
    # Matrix with two annotators but all spreads below threshold (default 2)
    data = {
        "A01": [5, 6, 7],
        "A02": [5, 6, 8],
    }
    matrix = pd.DataFrame(data, index=["p1", "p2", "p3"])
    df = agreement.disagreement_table(matrix, threshold=5)  # high threshold, no disputes
    assert list(df.columns) == ["pair_id", "min_score", "max_score", "spread", "annotators"]
    assert df.empty

def test_disagreement_table_with_dispute():
    data = {
        "A01": [1, 2, 5],
        "A02": [5, 2, 3],
        "A03": [5, 2, 5],
    }
    matrix = pd.DataFrame(data, index=["p1", "p2", "p3"])
    df = agreement.disagreement_table(matrix, threshold=2)
    # Expect at least one dispute (pair p1: spread 4, p3: spread 2)
    assert not df.empty
    assert "pair_id" in df.columns
    # Verify that pair_id values are present and correctly typed
    assert set(df["pair_id"].astype(str)) == {"p1", "p3"}
    # Verify spread values meet threshold
    assert all(df["spread"] >= 2)
