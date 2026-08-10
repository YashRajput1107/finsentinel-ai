import pandas as pd
import pytest


@pytest.fixture
def sample_meta():
    """Tiny metadata frame mimicking chunks_meta — for filter/MMR tests (no real index needed)."""
    return pd.DataFrame({
        "chunk_id": [0, 1, 2, 3],
        "ticker": ["TSLA", "TSLA", "AAPL", "AAPL"],
        "source_type": ["filing", "transcript", "filing", "filing"],
        "text": ["a", "b", "c", "d"],
    })