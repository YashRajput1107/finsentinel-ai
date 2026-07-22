import logging

import pandas as pd

from src.data_pipeline.stock_data import load_prices
from src.utils.config import FEATURES_DIR, TICKERS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def compute_rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def compute_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    return (ema12 - ema26) / close

def compute_features(prices: pd.DataFrame)-> pd.DataFrame:
    """One ticker's prices in -> one leakage-safe feature matrix out."""
    df = prices.sort_values("date").set_index("date")
    out = pd.DataFrame(index=df.index)
    out["ticker"] = df["ticker"].iloc[0]

    # --- what happened recently (lagged returns) ---
    for lag in [1,2,3,5,10]:
        out[f"ret_{lag}d"] = df["close"].pct_change(lag)
    # --- how stretched is the rubber band (trend) ---
    ma20 =df["close"].rolling(20).mean()
    ma50 =df["close"].rolling(50).mean()
    out["price_vs_ma20"] = df["close"] / ma20
    out["ma20_vs_ma50"] = ma20 / ma50

    # --- how wild has it been (risk) ---
    out["vol_20d"] = df["close"].pct_change().rolling(20).std()
    # --- who's winning the tug of war (momentum) ---
    out["rsi_14"] = compute_rsi(df["close"])
    out["macd"] = compute_macd(df["close"])

     # --- is the crowd unusually loud (activity) ---
    vol_ma = df["volume"].rolling(20).mean()
    vol_sd = df["volume"].rolling(20).std()
    out["volume_z"] = (df["volume"] - vol_ma) / vol_sd

    future_1d = df["close"].shift(-1)
    future_5d = df["close"].shift(-5)
    out["target_1d"] = (future_1d > df["close"]).where(future_1d.notna())
    out["target_5d"] = (future_5d > df["close"]).where(future_5d.notna())

    return out.dropna()
def build_features() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in TICKERS:
        feats = compute_features(load_prices([ticker]))
        feats.to_parquet(FEATURES_DIR / f"{ticker}.parquet")
        logger.info("%s: %d rows, %d features", ticker, len(feats), feats.shape[1] - 3)

if __name__ == "__main__":
    build_features()