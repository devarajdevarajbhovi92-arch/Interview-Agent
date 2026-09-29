"""
train_difficulty_predictor.py — Train the question difficulty prediction model.

Trains a Gradient Boosting classifier to predict question difficulty
based on candidate features and question topic.

Usage:
    python -m ml.training.train_difficulty_predictor
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"
_MODEL_DIR = Path(__file__).parent.parent / "models"
_DIFF_PATH = _DATA_DIR / "synthetic_difficulty.csv"

DIFFICULTY_LABELS = ["easy", "medium", "hard"]


def load_data() -> pd.DataFrame:
    """Load synthetic difficulty data."""
    if not _DIFF_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {_DIFF_PATH}\n"
            "Run generate_synthetic_data.py first."
        )
    return pd.read_csv(_DIFF_PATH)


def prepare_features(df: pd.DataFrame):
    """Prepare features for training."""
    feature_cols = [
        "first_try_ratio",
        "missions_completed",
        "commit_days",
    ]

    X = df[feature_cols].values
    y = df["difficulty"].map(
        {label: i for i, label in enumerate(DIFFICULTY_LABELS)}
    ).values

    return X, y, feature_cols


def train() -> None:
    """Train the difficulty predictor."""
    logger.info("Loading data...")
    df = load_data()
    logger.info(f"Loaded {len(df)} samples")

    logger.info("Preparing features...")
    X, y, feature_cols = prepare_features(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    logger.info("Training Gradient Boosting classifier...")
    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)

    logger.info("\n=== Difficulty Classification Report ===")
    print(classification_report(
        y_test, y_pred, target_names=DIFFICULTY_LABELS
    ))

    # Save
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MODEL_DIR / "difficulty_predictor.joblib"
    joblib.dump(clf, model_path)

    logger.info(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train()
