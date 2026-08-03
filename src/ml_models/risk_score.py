import logging

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine

from src.data_pipeline.stock_data import load_prices
from src.utils.config import DB_PATH, TICKERS
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Weights sum to 1.0 — documented, chosen (not learned). This IS the risk model.
WEIGHTS = {
    "leverage":      0.20,   # debt-to-equity
    "illiquidity":   0.20,   # inverse of current ratio
    "unprofit":      0.20,   # inverse of profit margin
    "beta":          0.15,   # market sensitivity
    "volatility":    0.25,   # our own price-derived
}


def fetch_fundamentals(tickers=TICKERS)-> pd.DataFrame:

    """Pull balance-sheet + market metrics from yfinance. Missing values -> NaN (handled later)."""
    rows=[]
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            rows.append({
                "ticker": t,
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio":  info.get("currentRatio"),
                "profit_margin":  info.get("profitMargins"),
                "beta":           info.get("beta"),
            })
            logger.info(f"Fetched fundamentals for {t}")
        except Exception as e:
            logger.warning("%s: Fundamentals fetch failed:%s",t,e)
            rows.append({"ticker":t})
    return pd.DataFrame(rows).set_index("ticker")

def compute_volatility(tickers=TICKERS)-> pd.DataFrame:
    """Annualized volatility per ticker, from stored prices."""
    prices=load_prices(tickers)
    prices["ret"]=prices.groupby("ticker")["close"].pct_change()
    vol=prices.groupby("ticker")["ret"].std()*np.sqrt(252)
    return vol.rename("volatility")

def compute_risk_score()->pd.DataFrame:
    """Transparent 0-10 FINANCIAL risk score per company (balance-sheet health,
    not price/market risk).

    Factors: leverage, liquidity, profitability, beta, volatility - min-max
    normalized within our 10-company universe and weighted (see WEIGHTS).

    Known limitations (descriptive score, not predictive; document, don't hide):
    - Measures FINANCIAL risk, not price risk: NVDA ranks low-risk (strong margins,
      low debt) despite being highly volatile. Price volatility is shown separately.
    - No sector adjustment: a bank's structurally high leverage isn't normalized.
    - Ratio blind spots: AAPL/AMZN rank high because current ratio (~1.0) and thin
      margins don't capture Apple's cash reserves or Amazon's strategy.
    """
    df=fetch_fundamentals()
    df["volatility"]=compute_volatility()
    #  Orient every factor so that HIGHER = RISKIER
    factors = pd.DataFrame(index=df.index)
    factors["leverage"]   = df["debt_to_equity"]
    factors["illiquidity"] = -df["current_ratio"]   # low liquidity = high risk -> negate
    factors["unprofit"]    = -df["profit_margin"]    # low margin = high risk -> negate
    factors["beta"]        = df["beta"]
    factors["volatility"]  = df["volatility"]
    # Fill missing with the column median (a neutral, non-extreme value) and log it
    missing = factors.isna().sum()
    if missing.any():
        logger.warning("Missing factor values filled with median:\n%s", missing[missing > 0])
    factors = factors.fillna(factors.median())
    # Min-max normalize each factor to 0-1 (relative risk within our 10-company universe)
    norm = (factors - factors.min()) / (factors.max() - factors.min())

    # Weighted sum -> 0-10
    score = sum(norm[f] * w for f, w in WEIGHTS.items()) * 10
    out = pd.DataFrame({"financial_risk_score": score.round(2)}).sort_values("financial_risk_score")

    engine = create_engine(f"sqlite:///{DB_PATH}")
    out.reset_index().to_sql("financial_risk_scores", engine, if_exists="replace", index=False)
    logger.info("Risk scores saved to SQLite:\n%s", out.to_string())
    return out
if __name__ == "__main__":
    scores = compute_risk_score()
    print(scores)

