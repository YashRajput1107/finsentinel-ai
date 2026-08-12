import sys
from pathlib import Path

# streamlit runs THIS file directly, so its own folder is on the path — not the repo root.
# add the repo root so `from src...` imports resolve. (This is the gotcha I warned about.)
# NOTE: everything importing `src` has to come AFTER this line.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

from src.data_pipeline.stock_data import load_prices
from src.utils.config import TICKERS, NEWS_PATH, REDDIT_PATH, DB_PATH

st.set_page_config(page_title="FinSentinel", page_icon="📈", layout="wide")

@st.cache_data
def get_prices(ticker: str):
    """One company's price history from SQLite. Cached, so it runs once per ticker,
    not on every rerun."""
    return load_prices([ticker]).sort_values("date")

@st.cache_data
def get_sentiment(source: str):
    """Stored FinBERT-scored text. We READ scored results - never re-run the model in the UI."""
    return pd.read_parquet(NEWS_PATH if source == "news" else REDDIT_PATH)


@st.cache_data
def get_risk_scores():
    """Financial risk scores computed on Day 16, read from SQLite."""
    engine = create_engine(f"sqlite:///{DB_PATH}")
    return pd.read_sql("SELECT * FROM financial_risk_scores", engine)

st.title("FinSentinel — Financial Intelligence Dashboard")

ticker = st.sidebar.selectbox("Company", TICKERS)
df = get_prices(ticker)
close = df["close"]
ann_vol = close.pct_change().std() * np.sqrt(252) * 100

tab_overview, tab_sentiment, tab_risk, tab_models = st.tabs(
    ["Overview", "Sentiment", "Risk", "Models"]
)

with tab_overview:
    latest = close.iloc[-1]
    period_return = (close.iloc[-1] / close.iloc[0] - 1) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest close", f"${latest:,.2f}")
    c2.metric("Period return", f"{period_return:+.1f}%")
    c3.metric("Annualized volatility", f"{ann_vol:.1f}%")

    st.subheader(f"{ticker} — closing price")
    st.line_chart(df.set_index("date")["close"])
    st.caption(f"{len(df):,} trading days · {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")

with tab_sentiment:
    st.subheader(f"{ticker} — sentiment (FinBERT)")
    source = st.radio("Source", ["news", "reddit"], horizontal=True)

    data = get_sentiment(source)
    subset = data[data["ticker"] == ticker]

    if subset.empty:
        st.warning(f"No {source} data for {ticker}. (MSFT has no news coverage in this dataset.)")
    else:
        counts = subset["sentiment"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("Positive", f"{counts.get('positive', 0) / len(subset) * 100:.0f}%")
        c2.metric("Neutral",  f"{counts.get('neutral', 0)  / len(subset) * 100:.0f}%")
        c3.metric("Negative", f"{counts.get('negative', 0) / len(subset) * 100:.0f}%")

        st.bar_chart(counts)
        st.caption(
            f"{len(subset):,} {source} items · {subset['date'].min():%Y-%m-%d} to "
            f"{subset['date'].max():%Y-%m-%d}. **Historical, not live** — this is not current market mood."
        )

with tab_risk:
    st.subheader(f"{ticker} — financial risk score")
    risk = get_risk_scores().sort_values("financial_risk_score")
    row = risk[risk["ticker"] == ticker]

    if row.empty:
        st.warning("No risk score available for this company.")
    else:
        score = float(row["financial_risk_score"].iloc[0])
        rank = int(risk["ticker"].tolist().index(ticker)) + 1

        c1, c2, c3 = st.columns(3)
        c1.metric("Financial risk (0-10)", f"{score:.2f}")
        c2.metric("Rank (1 = safest)", f"{rank} of {len(risk)}")
        c3.metric("Price volatility (annualized)", f"{ann_vol:.1f}%")

        st.bar_chart(risk.set_index("ticker")["financial_risk_score"])
        st.caption(
            "Measures **financial** risk — balance-sheet health (leverage, liquidity, profitability, "
            "beta, volatility) — **not price risk**. NVDA scores low-risk financially despite being "
            "highly price-volatile, which is why volatility is shown separately. Weights are chosen, "
            "not learned; no sector adjustment."
        )

with tab_models:
    st.subheader("Model performance — measured, not promised")

    st.markdown("""
| Model | Task | Result | Honest read |
|---|---|---|---|
| Logistic Regression / RF / XGBoost / MLP / LSTM | Next-day price direction | **~0.51 accuracy** vs **0.531 baseline** | **Null result** — does not beat always-guessing the majority class |
| FinBERT | Financial sentiment classification | **0.938 macro-F1** | Genuinely strong, but see caveat below |
| TF-IDF + LogReg | Financial sentiment (baseline) | 0.752 macro-F1 | Reference point for FinBERT |
""")

    st.error(
        "**No directional prediction is shown in this dashboard.** Every model family tested "
        "(logistic regression, random forest, XGBoost, MLP, LSTM) performed at or below a majority-class "
        "baseline on next-day direction. Displaying a buy/sell signal from a coin-flip model would be "
        "misleading, so the evaluation is shown instead of a prediction."
    )

    st.warning(
        "**Sentiment caveat — benchmark contamination.** FinBERT's 0.938 was measured on the Financial "
        "PhraseBank, which is part of its own fine-tuning data. It is an optimistic upper bound, not a "
        "measure of real-world performance on news or Reddit text."
    )

    st.caption("Full methodology and results: docs/EVALUATION.md")