import logging
import re

import pandas as pd
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader

import warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


from src.utils.config import (
    PROCESSED_SEC_DIR,
    RAW_SEC_DIR,
    SEC_COMPANY,
    SEC_EMAIL,
    SEC_TICKER_TO_CIK,
    TICKERS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

FILINGS_TO_GET = {"10-K": 2, "10-Q": 4}



def download_filings(tickers=TICKERS)-> None:
    RAW_SEC_DIR.mkdir(parents=True, exist_ok=True)
    dl= Downloader(SEC_COMPANY, SEC_EMAIL, RAW_SEC_DIR)

    for ticker in tickers:
        for form, limit in FILINGS_TO_GET.items():
            try:
                query_id = SEC_TICKER_TO_CIK.get(ticker, ticker)
                n = dl.get(form, query_id, limit=limit, download_details=True)
                if n < limit:
                    logger.warning("%s: only %d of %d x %s downloaded", ticker, n, limit, form)
                else:
                    logger.info("%s: downloaded %d x %s", ticker, n, form)
            except Exception as e:
                logger.warning("%s: %s download failed: %s", ticker, form, e)
    
def extract_text(html_path)-> str:
    html=html_path.read_text(encoding="utf-8",errors="ignore")
    soup= BeautifulSoup(html, "lxml")

    for tag in soup(["script","style"]):
        tag.decompose()
    for tag in soup.find_all(style=lambda s: s and "display:none" in s.replace(" ", "").lower()):
        tag.decompose()
    text=soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)

def get_filing_date(accession_dir)-> str:
    submission = accession_dir / "full-submission.txt"
    if submission.exists():
        head = submission.read_text(encoding="utf-8", errors="ignore")[:3000]
        match = re.search(r"FILED AS OF DATE:\s*(\d{8})", head)
        if match:
            d = match.group(1)
            return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return "unknown"

def process_filings()-> pd.DataFrame:
    PROCESSED_SEC_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    filings_root = RAW_SEC_DIR / "sec-edgar-filings"
    cik_to_ticker = {cik: t for t, cik in SEC_TICKER_TO_CIK.items()}
    

    for html_path in sorted(filings_root.rglob("primary-document.html")):
        accession_dir = html_path.parent
        form = accession_dir.parent.name          
        ticker = accession_dir.parent.parent.name
        ticker = cik_to_ticker.get(ticker, ticker)

        text = extract_text(html_path)
        date = get_filing_date(accession_dir)

        out_name = f"{ticker}_{form.replace("-",'')}_{date}.txt"
        out_path= PROCESSED_SEC_DIR / out_name
        out_path.write_text(text, encoding="utf-8")

        records.append({
            "ticker": ticker,
            "form": form,
            "date": date,
            "path": str(out_path),
            "words": len(text.split()),
            "has_risk_factors": bool(re.search(r"item\s*1a", text[:50000], re.IGNORECASE)),
        })

        logger.info("%s %s %s: %d words", ticker, form, date, len(text.split()))

    index = pd.DataFrame(records)
    index.to_csv(PROCESSED_SEC_DIR / "index.csv", index=False)
    logger.info("Index written: %d filings", len(index))
    return index

if __name__ == "__main__":
    download_filings()
    index = process_filings()
    logger.info("Summary:\n%s", index.groupby(["ticker", "form"]).size())