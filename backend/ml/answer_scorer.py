"""
answer_scorer.py — Continuous answer scoring model.

Provides a continuous score (0-100) for each answer using Gradient Boosting
Regression, enabling more granular feedback than categorical judgment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .feature_engineering import extract_text_features, extract_question_features

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent / "models"
_MODEL_PATH = _MODEL_DIR / "answer_scorer.joblib"


class AnswerScorer:
    """ML-based continuous answer scorer."""

    def __init__(self) -> None:
        self._model: GradientBoostingRegressor | None = None
        self._is_loaded = False

    def load(self) -> bool:
        """Load pre-trained model from disk."""
        if _MODEL_PATH.exists():
            try:
                self._model = joblib.load(_MODEL_PATH)
                self._is_loaded = True
                logger.info("Answer scorer loaded from disk.")
                return True
            except Exception as e:
                logger.warning(f"Failed to load answer scorer: {e}")
        return False

    def score(self, question: str, answer: str) -> dict[str, Any]:
        """
        Score an answer on a continuous 0-100 scale.

        Returns:
            dict with keys: score, confidence
        """
        if not self._is_loaded or self._model is None:
            return self._heuristic_score(question, answer)

        try:
            t_feats = extract_text_features(answer)
            q_feats = extract_question_features(question)

            features = np.array(
                list(t_feats.values()) + list(q_feats.values()),
                dtype=np.float32,
            ).reshape(1, -1)

            score = float(self._model.predict(features)[0])
            score = max(0.0, min(100.0, score))

            return {
                "score": round(score, 1),
                "confidence": 0.7,
            }
        except Exception as e:
            logger.warning(f"ML scoring failed, using heuristic: {e}")
            return self._heuristic_score(question, answer)

    def _heuristic_score(self, question: str, answer: str) -> dict[str, Any]:
        """Fallback heuristic scoring when model is not available."""
        text_lower = answer.lower().strip()

        # Base score
        score = 50.0

        # Length factor
        word_count = len(answer.split())
        if word_count > 50:
            score += 15
        elif word_count > 20:
            score += 10
        elif word_count < 5:
            score -= 20

        # Quality indicators
        strong_phrases = [
            "for example", "specifically", "i implemented", "i designed",
            "the result was", "in production", "i optimized",
        ]
        if any(p in text_lower for p in strong_phrases):
            score += 15

        uncertain_phrases = ["i think", "maybe", "probably", "i guess", "not sure"]
        if any(p in text_lower for p in uncertain_phrases):
            score -= 10

        # Code/numbers bonus
        if "```" in answer or "def " in text_lower or "class " in text_lower:
            score += 10
        if any(c.isdigit() for c in answer):
            score += 5

        # Skip detection
        skip_phrases = {"i don't know", "skip", "pass", "no idea", "n/a", ""}
        if text_lower in skip_phrases:
            score = 0.0

        score = max(0.0, min(100.0, score))

        return {
            "score": round(score, 1),
            "confidence": 0.5,
        }

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


# Singleton instance
_scorer: AnswerScorer | None = None


def get_answer_scorer() -> AnswerScorer:
    """Get or create the singleton AnswerScorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = AnswerScorer()
        _scorer.load()
    return _scorer
