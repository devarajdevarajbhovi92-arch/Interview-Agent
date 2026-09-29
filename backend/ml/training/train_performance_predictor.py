"""
train_performance_predictor.py — Train the candidate performance prediction model.

Trains a Ridge regression model to predict overall interview score
based on candidate features.

Usage:
    python -m ml.training.train_performance_predictor
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
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


def prepare_features(df: pd.DataFrame):
    """Prepare features for training."""
    feature_cols = [
        "missions_completed",
        "missions_first_try",
        "commit_days",
        "pass_rate",
        "skip_rate",
        "avg_attempts",
        "years_experience",
    ]

    X = df[feature_cols].values
    y = df["score"].values

    return X, y, feature_cols


def train() -> None:
    """Train the performance predictor."""
    logger.info("Loading data...")
    df = load_data()
    logger.info(f"Loaded {len(df)} samples")

    logger.info("Preparing features...")
    X, y, feature_cols = prepare_features(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    logger.info("Training Ridge regression model...")
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"\n=== Performance Prediction Metrics ===")
    logger.info(f"Mean Absolute Error: {mae:.2f}")
    logger.info(f"R² Score: {r2:.4f}")

    # Save
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MODEL_DIR / "performance_predictor.joblib"
    joblib.dump(model, model_path)

    logger.info(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train()
