import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

from src.utils.config import TRANSCRIPTS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

REPO = "jlh-ibm/earnings_call"
COVERED = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]   # our tickers present in this dataset


def build_transcripts() -> pd.DataFrame:
    """Download earnings-call transcripts for covered tickers, save + index them."""
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for f in list_repo_files(REPO, repo_type="dataset"):
        parts = f.split("/")
        if len(parts) == 4 and parts[1] == "transcripts" and parts[2] in COVERED:
            ticker = parts[2]
            local = hf_hub_download(REPO, f, repo_type="dataset")
            text = Path(local).read_text(encoding="utf-8", errors="ignore")

            date_str = Path(f).stem.rsplit("-", 1)[0]          # "2016-Apr-26-AAPL" -> "2016-Apr-26"
            date = datetime.strptime(date_str, "%Y-%b-%d").date()

            out = TRANSCRIPTS_DIR / f"{ticker}_{date}.txt"
            out.write_text(text, encoding="utf-8")
            records.append({"ticker": ticker, "date": date,
                            "path": str(out), "words": len(text.split())})

    index = pd.DataFrame(records).sort_values(["ticker", "date"])
    index.to_csv(TRANSCRIPTS_DIR / "index.csv", index=False)
    logger.info("Transcripts: %d files across %d companies",
                len(index), index["ticker"].nunique())
    return index