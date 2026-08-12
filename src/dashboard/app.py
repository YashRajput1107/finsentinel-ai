import sys
from pathlib import Path

# streamlit runs THIS file directly, so its own folder is on the path — not the repo root.
# add the repo root so `from src...` imports resolve. (This is the gotcha I warned about.)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import streamlit as st

from src.data_pipeline.stock_data import load_prices
from src.utils.config import TICKERS

st.set_page_config(page_title="FinSentinel", page_icon="📈", layout="wide")

@st.cache_data
def get_prices(ticker: str):
    """One company's price history from SQLite. Cached, so it runs once per ticker,
    not on every rerun."""
    return load_prices([ticker]).sort_values("date")

st.title("FinSentinel — Financial Intelligence Dashboard")

ticker = st.sidebar.selectbox("Company", TICKERS)   # returns the current selection this run
df = get_prices(ticker)
close = df["close"]

latest = close.iloc[-1]
period_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
ann_vol = close.pct_change().std() * np.sqrt(252) * 100   # same formula as your risk score

c1, c2, c3 = st.columns(3)
c1.metric("Latest close", f"${latest:,.2f}")
c2.metric("Period return", f"{period_return:+.1f}%")
c3.metric("Annualized volatility", f"{ann_vol:.1f}%")

st.subheader(f"{ticker} — closing price")
st.line_chart(df.set_index("date")["close"])

st.caption(f"{len(df):,} trading days · {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")

