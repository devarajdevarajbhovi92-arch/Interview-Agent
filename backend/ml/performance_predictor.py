"""
performance_predictor.py — Candidate overall performance prediction model.

Predicts the candidate's overall interview score (0-100) before the interview
starts, based on their mission history and profile.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from .feature_engineering import extract_candidate_features

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent / "models"
_MODEL_PATH = _MODEL_DIR / "performance_predictor.joblib"


class PerformancePredictor:
    """ML-based candidate performance predictor."""

    def __init__(self) -> None:
        self._model: Ridge | None = None
        self._is_loaded = False

    def load(self) -> bool:
        """Load pre-trained model from disk."""
        if _MODEL_PATH.exists():
            try:
                self._model = joblib.load(_MODEL_PATH)
                self._is_loaded = True
                logger.info("Performance predictor loaded from disk.")
                return True
            except Exception as e:
                logger.warning(f"Failed to load performance predictor: {e}")
        return False

    def predict(self, candidate: dict) -> dict[str, Any]:
        """
        Predict overall interview score for a candidate.

        Returns:
            dict with keys: predicted_score, confidence
        """
        if not self._is_loaded or self._model is None:
            return self._heuristic_predict(candidate)

        try:
            c_feats = extract_candidate_features(candidate)
            features = np.array(list(c_feats.values()), dtype=np.float32).reshape(1, -1)

            score = float(self._model.predict(features)[0])
            score = max(0.0, min(100.0, score))  # Clamp to 0-100

            return {
                "predicted_score": round(score, 1),
                "confidence": 0.6,  # Ridge doesn't give probabilities
            }
        except Exception as e:
            logger.warning(f"ML prediction failed, using heuristic: {e}")
            return self._heuristic_predict(candidate)

    def _heuristic_predict(self, candidate: dict) -> dict[str, Any]:
        """Fallback heuristic when model is not available."""
        signals = candidate.get("signals", {})
        missions = candidate.get("missions", [])

        completed = signals.get("missionsCompleted", 0)
        first_try = signals.get("missionsFirstTry", 0)
        commit_days = signals.get("commitDays", 0)

        # Simple weighted heuristic
        first_try_ratio = first_try / max(completed, 1)
        completion_ratio = completed / max(len(missions), 1)
        activity_ratio = min(commit_days / 30.0, 1.0)

        score = (
            first_try_ratio * 40
            + completion_ratio * 30
            + activity_ratio * 30
        )

        return {
            "predicted_score": round(score, 1),
            "confidence": 0.5,
        }

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


# Singleton instance
_predictor: PerformancePredictor | None = None


def get_performance_predictor() -> PerformancePredictor:
    """Get or create the singleton PerformancePredictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = PerformancePredictor()
        _predictor.load()
    return _predictor
