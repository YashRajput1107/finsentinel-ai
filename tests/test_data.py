import pandas as pd

from src.data_pipeline.stock_data import load_prices


def test_price_schema():
    df = load_prices(["AAPL"])
    expected = {"date", "ticker", "open", "high", "low", "close", "volume"}
    assert expected.issubset(df.columns)


def test_price_types():
    df = load_prices(["AAPL"])
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_numeric_dtype(df["close"])


def test_price_ranges():
    df = load_prices(["AAPL", "TSLA"])
    assert (df["close"] > 0).all()                       # prices are positive
    assert (df["high"] >= df["low"]).all()               # high never below low
    assert (df["volume"] >= 0).all()                     # volume non-negative
    assert df[["open", "high", "low", "close"]].notna().all().all()   # no gaps


def test_no_duplicate_ticker_date():
    df = load_prices(["AAPL", "TSLA"])
    assert not df.duplicated(subset=["ticker", "date"]).any()


def test_no_future_dates():
    df = load_prices(["AAPL"])
    assert df["date"].max() <= pd.Timestamp.today()      # nothing from the future


def test_dates_sorted_per_ticker():
    for _, group in load_prices(["AAPL", "TSLA"]).groupby("ticker"):
        assert group["date"].is_monotonic_increasing