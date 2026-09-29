"""
train_answer_scorer.py — Train the continuous answer scoring model.

Trains a Gradient Boosting regressor to predict a continuous score (0-100)
for each answer based on text and question features.

Usage:
    python -m ml.training.train_answer_scorer
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from ..feature_engineering import extract_text_features, extract_question_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"
_MODEL_DIR = Path(__file__).parent.parent / "models"
_QA_PATH = _DATA_DIR / "synthetic_qa_pairs.csv"


def load_data() -> pd.DataFrame:
    """Load synthetic Q&A data."""
    if not _QA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {_QA_PATH}\n"
            "Run generate_synthetic_data.py first."
        )
    return pd.read_csv(_QA_PATH)


def prepare_features(df: pd.DataFrame):
    """Prepare features for training."""
    text_feats_list = []
    question_feats_list = []

    for _, row in df.iterrows():
        t_feats = extract_text_features(str(row["answer"]))
        q_feats = extract_question_features(str(row["question"]))
        text_feats_list.append(list(t_feats.values()))
        question_feats_list.append(list(q_feats.values()))

    import numpy as np
    X = np.hstack([np.array(text_feats_list), np.array(question_feats_list)])

    # Generate continuous score from labels
    quality_scores = {"strong": 100, "shallow": 60, "confused": 35, "off_topic": 15, "missing": 0}
    completeness_scores = {"full": 100, "partial": 55, "missing": 0}

    y = []
    for _, row in df.iterrows():
        q_score = quality_scores.get(row["quality"], 55)
        c_score = completeness_scores.get(row["completeness"], 55)
        y.append((q_score + c_score) / 2)

    return X, y


def train() -> None:
    """Train the answer scorer."""
    logger.info("Loading data...")
    df = load_data()
    logger.info(f"Loaded {len(df)} samples")

    logger.info("Preparing features...")
    X, y = prepare_features(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    logger.info("Training Gradient Boosting regressor...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"\n=== Answer Scoring Metrics ===")
    logger.info(f"Mean Absolute Error: {mae:.2f}")
    logger.info(f"R² Score: {r2:.4f}")

    # Save
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MODEL_DIR / "answer_scorer.joblib"
    joblib.dump(model, model_path)

    logger.info(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train()
