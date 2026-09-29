"""
train_question_recommender.py — Train the question recommendation model.

Trains a content-based filtering model to recommend questions based on
candidate weak areas and question topics.

Usage:
    python -m ml.training.train_question_recommender
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"
_MODEL_DIR = Path(__file__).parent.parent / "models"
_PERF_PATH = _DATA_DIR / "synthetic_performance.csv"


def load_data() -> pd.DataFrame:
    """Load synthetic performance data."""
    if not _PERF_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {_PERF_PATH}\n"
            "Run generate_synthetic_data.py first."
        )
    return pd.read_csv(_PERF_PATH)


def train() -> None:
    """Train the question recommender (content-based model)."""
    logger.info("Loading data...")
    df = load_data()
    logger.info(f"Loaded {len(df)} samples")

    # For content-based filtering, we store the feature weights
    # that determine how candidate features map to question relevance

    # Compute feature importance based on correlation with score
    feature_cols = [
        "missions_completed",
        "missions_first_try",
        "commit_days",
        "pass_rate",
        "skip_rate",
        "avg_attempts",
        "years_experience",
    ]

    correlations = {}
    for col in feature_cols:
        corr = df[col].corr(df["score"])
        correlations[col] = abs(corr) if not np.isnan(corr) else 0.0

    # Normalize to sum to 1
    total = sum(correlations.values())
    if total > 0:
        correlations = {k: v / total for k, v in correlations.items()}

    logger.info(f"\n=== Feature Importance ===")
    for feat, imp in sorted(correlations.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {feat}: {imp:.4f}")

    # Save
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MODEL_DIR / "question_recommender.joblib"
    joblib.dump({"feature_weights": correlations}, model_path)

    logger.info(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train()
