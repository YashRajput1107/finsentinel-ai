import logging
import zipfile

import pandas as pd
from huggingface_hub import hf_hub_download

from src.utils.config import NEWS_PATH, PHRASEBANK_PATH, PROCESSED_DIR, TICKERS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def fetch_phrasebank() -> pd.DataFrame:
    zip_path=hf_hub_download("takala/financial_phrasebank",
                           "data/FinancialPhraseBank-v1.0.zip", repo_type="dataset")

    with zipfile.ZipFile(zip_path) as z:
        with z.open(f"FinancialPhraseBank-v1.0/Sentences_75Agree.txt") as f:
            lines= f.read().decode("latin-1").splitlines()
    rows=[line.rsplit("@",1) for line in lines if "@" in line]
    df= pd.DataFrame(rows,columns=["text","label"])
    df["text"] = df["text"].str.strip()
    df["label"] = df["label"].str.strip()

    before= len(df)
    df=df.drop_duplicates(subset="text")
    if len(df) < before:
       logger.info("PhraseBank: dropped %d duplicate sentences", before - len(df))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PHRASEBANK_PATH, index=False)
    logger.info("PhraseBank saved: %d rows -> %s", len(df), PHRASEBANK_PATH.name)
    return df

def build_news_dataset() -> pd.DataFrame:
    """Download financial news, keep our 10 tickers, standardize, save to parquet."""
    import kagglehub
    from pathlib import Path

    src = Path(kagglehub.dataset_download(
        "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests"))
    frames = []
    for fname in ["raw_analyst_ratings.csv", "raw_partner_headlines.csv"]:
        part = pd.read_csv(src / fname)
        frames.append(part[["headline", "publisher", "date", "stock"]])
    df = pd.concat(frames, ignore_index=True)
    logger.info("News raw (both files): %d rows", len(df))

    # Meta traded as FB until 2022 — translate the old ticker to the current one
    df["stock"] = df["stock"].replace({"FB": "META"})

    df = df.rename(columns={"stock": "ticker"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True, format="mixed").dt.date

    df = df[df["ticker"].isin(TICKERS)]
    logger.info("News after ticker filter: %d rows", len(df))

    before = len(df)
    df = df.dropna(subset=["date", "headline"])
    df = df.drop_duplicates(subset=["date", "ticker", "headline"])
    logger.info("News after cleaning: %d rows (-%d)", len(df), before - len(df))

    df = df[["date", "ticker", "headline", "publisher"]].rename(columns={"publisher": "source"})
    df.to_parquet(NEWS_PATH, index=False)
    logger.info("News saved -> %s", NEWS_PATH.name)
    return df

if __name__ == "__main__":
    pb = fetch_phrasebank()
    news = build_news_dataset()
    logger.info("PhraseBank: %d | News: %d rows, %d tickers, %s -> %s",
                len(pb), len(news), news["ticker"].nunique(),
                news["date"].min(), news["date"].max())