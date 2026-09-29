"""
mastery_estimator.py — Topic mastery estimation using Bayesian Knowledge Tracing.

Estimates the candidate's mastery level (0-100%) for each curriculum topic
based on their mission history, commit patterns, and interview performance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .feature_engineering import TOPIC_TERMS

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent / "models"
_MODEL_PATH = _MODEL_DIR / "mastery_estimator.joblib"


class MasteryEstimator:
    """Bayesian Knowledge Tracing-based topic mastery estimator."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._is_loaded = False
        # BKT parameters: P(learn), P(guess), P(slip), P(forget)
        self._p_learn = 0.3
        self._p_guess = 0.2
        self._p_slip = 0.1
        self._p_forget = 0.05

    def load(self) -> bool:
        """Load pre-trained model from disk."""
        if _MODEL_PATH.exists():
            try:
                data = joblib.load(_MODEL_PATH)
                self._model = data.get("model", None)
                self._p_learn = data.get("p_learn", 0.3)
                self._p_guess = data.get("p_guess", 0.2)
                self._p_slip = data.get("p_slip", 0.1)
                self._p_forget = data.get("p_forget", 0.05)
                self._is_loaded = True
                logger.info("Mastery estimator loaded from disk.")
                return True
            except Exception as e:
                logger.warning(f"Failed to load mastery estimator: {e}")
        return False

    def estimate_mastery(self, candidate: dict, topic: str) -> dict[str, Any]:
        """
        Estimate mastery level for a specific topic.

        Returns:
            dict with keys: mastery (0-100), confidence
        """
        missions = candidate.get("missions", [])
        signals = candidate.get("signals", {})

        # Find relevant missions for this topic
        topic_missions = self._get_topic_missions(missions, topic)

        if not topic_missions:
            return {"mastery": 0.0, "confidence": 0.0}

        # BKT-based estimation
        p_know = 0.5  # Prior probability of knowing the topic

        for mission in topic_missions:
            passed = mission.get("passed", False)
            skipped = mission.get("skipped", False)
            attempts = mission.get("attempts", 1)

            if skipped:
                # Skipped missions slightly decrease mastery
                p_know = p_know * (1 - self._p_forget)
            elif passed:
                # Correct answer: update using Bayes
                p_know_given_correct = (
                    p_know * (1 - self._p_slip)
                ) / (
                    p_know * (1 - self._p_slip) + (1 - p_know) * self._p_guess
                )
                # Apply learning
                p_know = p_know_given_correct + (1 - p_know_given_correct) * self._p_learn
            else:
                # Failed attempt
                p_know_given_wrong = (
                    p_know * self._p_slip
                ) / (
                    p_know * self._p_slip + (1 - p_know) * (1 - self._p_guess)
                )
                p_know = p_know_given_wrong * (1 - self._p_forget)

        mastery = round(p_know * 100, 1)
        confidence = min(len(topic_missions) * 0.3, 1.0)

        return {"mastery": mastery, "confidence": round(confidence, 3)}

    def estimate_all_topics(self, candidate: dict) -> dict[str, dict[str, Any]]:
        """Estimate mastery for all known topics."""
        results = {}
        for topic in TOPIC_TERMS:
            results[topic] = self.estimate_mastery(candidate, topic)
        return results

    def _get_topic_missions(self, missions: list[dict], topic: str) -> list[dict]:
        """Get missions relevant to a topic."""
        topic_terms = TOPIC_TERMS.get(topic, [])
        relevant = []
        for mission in missions:
            mission_title = mission.get("title", "").lower()
            if any(term in mission_title for term in topic_terms):
                relevant.append(mission)
        return relevant

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


# Singleton instance
_estimator: MasteryEstimator | None = None


def get_mastery_estimator() -> MasteryEstimator:
    """Get or create the singleton MasteryEstimator instance."""
    global _estimator
    if _estimator is None:
        _estimator = MasteryEstimator()
        _estimator.load()
    return _estimator
