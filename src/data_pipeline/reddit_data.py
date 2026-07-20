import kagglehub
import re
import pandas as pd
from pathlib import Path
import logging
from src.utils.config import REDDIT_PATH
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

COMPANY_NAMES = {
    "AAPL": ["Apple"],
    "MSFT": ["Microsoft"],
    "GOOGL": ["Google", "Alphabet", "GOOG"],
    "AMZN": ["Amazon"],
    "NVDA": ["Nvidia"],
    "TSLA": ["Tesla"],
    "META": ["Facebook", "Meta Platforms"],
    "JPM": ["JPMorgan", "JP Morgan"],
    "JNJ": ["Johnson & Johnson"],
    "XOM": ["Exxon", "ExxonMobil"],
}
def find_tickers(text:str)->list[str]:
    found=[]
    for ticker, names in COMPANY_NAMES.items():
        symbol_hit = re.search(rf"\${ticker}\b|\b{ticker}\b", text)   # case-SENSITIVE
        name_pattern = "|".join(rf"\b{re.escape(n)}\b" for n in names)
        name_hit = re.search(name_pattern, text, re.IGNORECASE)
        if symbol_hit or name_hit:
            found.append(ticker)
    return found
def build_reddit_dataset():
    src = Path(kagglehub.dataset_download("gpreda/reddit-wallstreetsbets-posts"))
    df = pd.read_csv(src / "reddit_wsb.csv")
    logger.info("Reddit raw: %d posts", len(df))

    df["body"] = df["body"].fillna("")
    df = df[~df["body"].isin(["[deleted]", "[removed]"])]
    
    df["text"]= (df["title"].fillna("") + " " + df["body"]).str.strip()
    df["text"] = df["text"].str.replace(r"http\S+", "", regex=True)

    df["date"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed").dt.date
    df = df.dropna(subset=["date"])

    df["ticker"] = df["text"].apply(find_tickers)
    df = df[df["ticker"].str.len() > 0]
    logger.info("Reddit posts mentioning our companies: %d", len(df))

    df = df.explode("ticker")
    df = df.drop_duplicates(subset=["date", "ticker", "title"])

    df = df[["date", "ticker", "title", "text", "score"]]
    df.to_parquet(REDDIT_PATH, index=False)
    logger.info("Reddit saved: %d rows -> %s", len(df), REDDIT_PATH.name)
    logger.info("Per ticker:\n%s", df["ticker"].value_counts().to_string())
    return df