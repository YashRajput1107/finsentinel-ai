import logging

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.config import FEATURES_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

FEATURES = ["ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d",
            "price_vs_ma20", "ma20_vs_ma50", "vol_20d", "rsi_14", "macd", "volume_z"]


def train_trend_model():
    """Train the chosen trend model on all pre-2025 data and save it."""
    frames = [pd.read_parquet(p) for p in sorted(FEATURES_DIR.glob("*.parquet"))]
    data = pd.concat(frames).sort_index()
    tv = data[data.index < "2025-01-01"]

    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=1000))])
    model.fit(tv[FEATURES], tv["target_1d"].astype(int))

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODELS_DIR / "trend_model.joblib")
    logger.info("Trend model saved -> models/trend_model.joblib")
    return model


if __name__ == "__main__":
    train_trend_model()