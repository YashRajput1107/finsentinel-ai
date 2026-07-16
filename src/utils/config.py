from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "JNJ", "XOM"]

HISTORY_YEARS = 5
DATA_DIR = PROJECT_ROOT / "data"
RAW_PRICES_DIR = DATA_DIR / "raw" / "prices"
DB_PATH = DATA_DIR / "finsentinel.db"