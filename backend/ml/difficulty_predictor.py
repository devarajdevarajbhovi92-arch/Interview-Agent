"""
difficulty_predictor.py — Question difficulty prediction model.

Predicts how difficult a question will be for a specific candidate based on
their mission history and question topic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .feature_engineering import (
    extract_candidate_features,
    extract_question_features,
)

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent / "models"
_MODEL_PATH = _MODEL_DIR / "difficulty_predictor.joblib"

DIFFICULTY_LABELS = ["easy", "medium", "hard"]


class DifficultyPredictor:
    """ML-based question difficulty predictor."""

    def __init__(self) -> None:
        self._model: GradientBoostingClassifier | None = None
        self._is_loaded = False

    def load(self) -> bool:
        """Load pre-trained model from disk."""
        if _MODEL_PATH.exists():
            try:
                self._model = joblib.load(_MODEL_PATH)
                self._is_loaded = True
                logger.info("Difficulty predictor loaded from disk.")
                return True
            except Exception as e:
                logger.warning(f"Failed to load difficulty predictor: {e}")
        return False

    def predict(self, question: str, candidate: dict) -> dict[str, Any]:
        """
        Predict difficulty of a question for a candidate.

        Returns:
            dict with keys: difficulty, confidence
        """
        if not self._is_loaded or self._model is None:
            return self._heuristic_predict(question, candidate)

        try:
            q_feats = extract_question_features(question)
            c_feats = extract_candidate_features(candidate)

            features = np.array(
                list(q_feats.values()) + list(c_feats.values()),
                dtype=np.float32,
            ).reshape(1, -1)

            prediction = self._model.predict(features)[0]
            probabilities = self._model.predict_proba(features)[0]

            return {
                "difficulty": DIFFICULTY_LABELS[prediction],
                "confidence": round(float(max(probabilities)), 3),
            }
        except Exception as e:
            logger.warning(f"ML prediction failed, using heuristic: {e}")
            return self._heuristic_predict(question, candidate)

    def _heuristic_predict(self, question: str, candidate: dict) -> dict[str, Any]:
        """Fallback heuristic when model is not available."""
        missions = candidate.get("missions", [])
        signals = candidate.get("signals", {})

        # Find relevant mission for this question topic
        first_try_ratio = signals.get("missionsFirstTry", 0) / max(signals.get("missionsCompleted", 1), 1)

        # Simple heuristic based on candidate's overall performance
        if first_try_ratio > 0.7:
            return {"difficulty": "easy", "confidence": 0.5}
        elif first_try_ratio > 0.4:
            return {"difficulty": "medium", "confidence": 0.5}
        else:
            return {"difficulty": "hard", "confidence": 0.5}

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


# Singleton instance
_predictor: DifficultyPredictor | None = None


def get_difficulty_predictor() -> DifficultyPredictor:
    """Get or create the singleton DifficultyPredictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = DifficultyPredictor()
        _predictor.load()
    return _predictor
