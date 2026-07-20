"""Run every data collector — one command rebuilds the entire data layer."""
import logging

from src.data_pipeline.stock_data import fetch_prices, save_prices
from src.data_pipeline.sec_filings import download_filings, process_filings
from src.data_pipeline.text_datasets import fetch_phrasebank, build_news_dataset
from src.data_pipeline.reddit_data import build_reddit_dataset
from src.data_pipeline.transcripts import build_transcripts

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("=== 1/5 stock prices ===")
    save_prices(fetch_prices())
    logger.info("=== 2/5 SEC filings ===")
    download_filings()
    process_filings()
    logger.info("=== 3/5 phrasebank + news ===")
    fetch_phrasebank()
    build_news_dataset()
    logger.info("=== 4/5 reddit ===")
    build_reddit_dataset()
    logger.info("=== 5/5 transcripts ===")
    build_transcripts()
    logger.info("ALL COLLECTORS DONE")