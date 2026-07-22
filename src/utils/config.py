from pathlib import Path

import os
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "JNJ", "XOM"]


SEC_TICKER_TO_CIK = {"XOM": "0000034088"}
HISTORY_YEARS = 5
DATA_DIR = PROJECT_ROOT / "data"
RAW_PRICES_DIR = DATA_DIR / "raw" / "prices"
DB_PATH = DATA_DIR / "finsentinel.db"


SEC_USER_AGENT = os.getenv("SEC_EDGAR_USER_AGENT", "")
SEC_COMPANY, SEC_EMAIL = SEC_USER_AGENT.rsplit(" ", 1)
RAW_SEC_DIR = DATA_DIR / "raw" / "sec"
PROCESSED_SEC_DIR = DATA_DIR / "processed" / "sec"

PROCESSED_DIR = DATA_DIR / "processed"
PHRASEBANK_PATH= PROCESSED_DIR / "phrasebank.parquet"
NEWS_PATH = PROCESSED_DIR / "news.parquet"

REDDIT_PATH = PROCESSED_DIR / "reddit.parquet"
TRANSCRIPTS_DIR = PROCESSED_DIR / "transcripts"

FEATURES_DIR= PROCESSED_DIR / "features"