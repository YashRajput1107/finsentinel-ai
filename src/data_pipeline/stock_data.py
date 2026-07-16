import logging

import pandas as pd
import yfinance as yf

from sqlalchemy import create_engine

from src.utils.config import DB_PATH, HISTORY_YEARS, RAW_PRICES_DIR, TICKERS

logging.basicConfig(level=logging.INFO,format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def fetch_prices(tickers=TICKERS,years=HISTORY_YEARS) -> pd.DataFrame:
    frames=[]
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=f"{years}y", auto_adjust=True)
        except Exception as e:
            logger.warning("Download failed for %s: %s", ticker, e)
            continue
        if df.empty:
            logger.warning("No data returned for %s — skipping", ticker)
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df["ticker"] = ticker
        df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]

        before = len(df)
        df = df.dropna(subset=["open", "high", "low", "close"])
        dropped = before - len(df)
        if dropped:
            logger.warning("%s: dropped %d row(s) with missing prices", ticker, dropped)    
        frames.append(df)
        logger.info("%s: %d rows fetched", ticker, len(df))

    if not frames:
        raise RuntimeError("No data fetched for any ticker")
    return pd.concat(frames, ignore_index=True)
        


def save_prices(df: pd.DataFrame) -> None:
    RAW_PRICES_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in df["ticker"].unique():
        ticker_df= df[df["ticker"]==ticker]
        ticker_df.to_parquet(RAW_PRICES_DIR / f"{ticker}.parquet", index=False)
        logger.info("%s: %d rows -> parquet", ticker, len(ticker_df))
    
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df.to_sql("prices", engine, if_exists="replace", index=False)
    logger.info("SQLite: %d total rows -> table 'prices'", len(df))


def load_prices(tickers=None) -> pd.DataFrame:
    if tickers is None:
        tickers = TICKERS
    frames=[]
    for ticker in tickers:
        path=RAW_PRICES_DIR / f"{ticker}.parquet"
        if not path.exists():
            logger.warning('No stored data for %s-run fetch_prices() first', ticker)
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError("No stored data found for any ticker")
    return pd.concat(frames, ignore_index=True)

if __name__ == "__main__":
    prices = fetch_prices()
    save_prices(prices)
    summary = prices.groupby("ticker")["date"].agg(["count", "min", "max"])
    logger.info("Pipeline complete:\n%s", summary)