import numpy as np
import pandas as pd

from src.preprocessing.features import compute_features


def make_prices(n=200, seed=0):
    """Deterministic synthetic OHLCV so we control the 'right answer'."""
    rng = np.random.default_rng(seed)
    close = np.abs(np.cumsum(rng.normal(0, 1, n))) + 100.0   # always positive
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "date": dates, "ticker": "TEST",
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": rng.integers(1_000, 10_000, n),
    })


def test_features_do_not_look_ahead():
    prices = make_prices(200)
    base = compute_features(prices)
    feature_cols = [c for c in base.columns if not c.startswith("target") and c != "ticker"]

    # Change a FUTURE price and recompute.
    P = 150
    perturbed = prices.copy()
    perturbed.loc[P, "close"] *= 1.5
    after = compute_features(perturbed)

    # Every feature row BEFORE the change must be byte-identical.
    # If any feature peeked forward, changing a future price would alter an earlier row.
    cutoff = prices.loc[P, "date"]
    mask = base.index < cutoff
    pd.testing.assert_frame_equal(base.loc[mask, feature_cols], after.loc[mask, feature_cols])


def test_chronological_split_has_no_overlap():
    feats = compute_features(make_prices(300))
    split = int(len(feats) * 0.8)
    train, test = feats.iloc[:split], feats.iloc[split:]
    assert train.index.max() < test.index.min()          # train strictly before test