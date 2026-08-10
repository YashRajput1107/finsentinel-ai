import pandas as pd
import pytest

from src.utils.config import FEATURES_DIR


def test_stock_target_is_near_a_coin_flip():
    path = FEATURES_DIR / "AAPL.parquet"
    if not path.exists():
        pytest.skip("features not built")
    base_rate = pd.read_parquet(path)["target_1d"].mean()
    # HONEST expectation: next-day direction is ~50/50. It must NOT be high.
    # A base rate far from ~0.5 would mean the label definition changed.
    assert 0.40 <= base_rate <= 0.60


@pytest.mark.slow
def test_sentiment_behavioral_sanity():
    from src.nlp.sentiment import score_texts
    out = score_texts([
        "The company posted record profits and raised its full-year outlook.",
        "The firm reported massive losses and slashed its dividend.",
    ])
    labels = [o["label"].lower() for o in out]
    assert labels[0] == "positive"
    assert labels[1] == "negative"