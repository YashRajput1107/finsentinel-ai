import logging
import pandas as pd
from transformers import pipeline
from src.utils.config import NEWS_PATH, REDDIT_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_model = None
def _get_model():
    global _model
    if _model is None:
        _model = pipeline("text-classification", model="ProsusAI/finbert")
    return _model

def score_texts(texts):
    """List of strings -> list of {'label', 'score'}."""
    return _get_model()(list(texts), truncation=True, batch_size=16)

def score_dataset(path, text_col):
    df = pd.read_parquet(path)
    results = score_texts(df[text_col].fillna("").tolist())
    df["sentiment"] = [r["label"].lower() for r in results]
    df["sentiment_score"] = [r["score"] for r in results]
    df.to_parquet(path, index=False)
    logger.info("%s: scored %d rows -> %s", path.name, len(df), df["sentiment"].value_counts().to_dict())
    return df

if __name__ == "__main__":
    score_dataset(NEWS_PATH, "headline")
    score_dataset(REDDIT_PATH, "text")