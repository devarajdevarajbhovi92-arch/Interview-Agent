"""
train_mastery_estimator.py — Train the topic mastery estimation model.

Trains a Bayesian Knowledge Tracing (BKT) model to estimate topic mastery
based on candidate mission history.

Usage:
    python -m ml.training.train_mastery_estimator
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
_MASTERY_PATH = _DATA_DIR / "synthetic_mastery.csv"


def load_data() -> pd.DataFrame:
    """Load synthetic mastery data."""
    if not _MASTERY_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {_MASTERY_PATH}\n"
            "Run generate_synthetic_data.py first."
        )
    return pd.read_csv(_MASTERY_PATH)


def bkt_predict(
    mission_attempts: int,
    mission_passed: bool,
    mission_skipped: bool,
    p_learn: float = 0.3,
    p_guess: float = 0.2,
    p_slip: float = 0.1,
    p_forget: float = 0.05,
) -> float:
    """Predict mastery using Bayesian Knowledge Tracing."""
    p_know = 0.5  # Prior

    if mission_skipped:
        p_know = p_know * (1 - p_forget)
    elif mission_passed:
        p_know_given_correct = (
            p_know * (1 - p_slip)
        ) / (
            p_know * (1 - p_slip) + (1 - p_know) * p_guess
        )
        p_know = p_know_given_correct + (1 - p_know_given_correct) * p_learn
    else:
        p_know_given_wrong = (
            p_know * p_slip
        ) / (
            p_know * p_slip + (1 - p_know) * (1 - p_guess)
        )
        p_know = p_know_given_wrong * (1 - p_forget)

    return p_know * 100


def train() -> None:
    """Train the mastery estimator (optimize BKT parameters)."""
    logger.info("Loading data...")
    df = load_data()
    logger.info(f"Loaded {len(df)} samples")

    # Grid search for optimal BKT parameters
    logger.info("Optimizing BKT parameters...")

    best_mae = float("inf")
    best_params = {}

    for p_learn in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for p_guess in [0.1, 0.2, 0.3]:
            for p_slip in [0.05, 0.1, 0.15]:
                for p_forget in [0.01, 0.05, 0.1]:
                    predictions = []
                    for _, row in df.iterrows():
                        pred = bkt_predict(
                            int(row["mission_attempts"]),
                            bool(row["mission_passed"]),
                            bool(row["mission_skipped"]),
                            p_learn, p_guess, p_slip, p_forget,
                        )
                        predictions.append(pred)

                    mae = mean_absolute_error(df["mastery"], predictions)
                    if mae < best_mae:
                        best_mae = mae
                        best_params = {
                            "p_learn": p_learn,
                            "p_guess": p_guess,
                            "p_slip": p_slip,
                            "p_forget": p_forget,
                        }

    logger.info(f"\n=== BKT Optimization Results ===")
    logger.info(f"Best MAE: {best_mae:.2f}")
    logger.info(f"Best parameters: {best_params}")

    # Save
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MODEL_DIR / "mastery_estimator.joblib"
    joblib.dump({"params": best_params}, model_path)

    logger.info(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train()
