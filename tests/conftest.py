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

@pytest.fixture(scope="session")
def embed_model():
    """Load MiniLM once for the whole test session (it's slow to construct)."""
    from sentence_transformers import SentenceTransformer
    from src.rag.embed_index import EMBED_MODEL
    return SentenceTransformer(EMBED_MODEL)