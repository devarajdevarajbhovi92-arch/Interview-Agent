"""
question_recommender.py — Personalized question recommendation model.

Recommends the next best question to ask based on the candidate's
demonstrated weak areas and learning history using content-based filtering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .feature_engineering import TOPIC_TERMS, extract_candidate_features

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent / "models"
_MODEL_PATH = _MODEL_DIR / "question_recommender.joblib"


class QuestionRecommender:
    """Content-based question recommender."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._is_loaded = False

    def load(self) -> bool:
        """Load pre-trained model from disk."""
        if _MODEL_PATH.exists():
            try:
                self._model = joblib.load(_MODEL_PATH)
                self._is_loaded = True
                logger.info("Question recommender loaded from disk.")
                return True
            except Exception as e:
                logger.warning(f"Failed to load question recommender: {e}")
        return False

    def recommend(
        self,
        candidate: dict,
        available_questions: list[str],
        asked_questions: list[str] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Recommend top-K questions for a candidate.

        Returns:
            list of dicts with keys: question, score, reason
        """
        if not available_questions:
            return []

        asked_questions = asked_questions or []

        # Filter out already asked questions
        remaining = [q for q in available_questions if q not in asked_questions]
        if not remaining:
            remaining = available_questions

        if not self._is_loaded or self._model is None:
            return self._heuristic_recommend(candidate, remaining, top_k)

        try:
            c_feats = extract_candidate_features(candidate)
            c_vector = np.array(list(c_feats.values()), dtype=np.float32)

            scored_questions = []
            for question in remaining:
                # Simple content-based scoring
                score = self._compute_relevance_score(question, candidate, c_vector)
                scored_questions.append({
                    "question": question,
                    "score": round(score, 3),
                    "reason": self._get_recommendation_reason(question, candidate),
                })

            scored_questions.sort(key=lambda x: x["score"], reverse=True)
            return scored_questions[:top_k]
        except Exception as e:
            logger.warning(f"ML recommendation failed, using heuristic: {e}")
            return self._heuristic_recommend(candidate, remaining, top_k)

    def _compute_relevance_score(
        self, question: str, candidate: dict, c_vector: np.ndarray
    ) -> float:
        """Compute relevance score for a question."""
        q_lower = question.lower()
        missions = candidate.get("missions", [])

        score = 0.5  # Base score

        # Boost score for topics where candidate struggled
        for mission in missions:
            mission_title = mission.get("title", "").lower()
            if any(term in q_lower for term in mission_title.split()):
                if not mission.get("passed", True):
                    score += 0.3  # Failed mission
                elif mission.get("skipped", False):
                    score += 0.2  # Skipped mission
                elif mission.get("attempts", 1) > 2:
                    score += 0.1  # Many attempts

        return min(score, 1.0)

    def _get_recommendation_reason(self, question: str, candidate: dict) -> str:
        """Generate human-readable reason for recommendation."""
        q_lower = question.lower()
        missions = candidate.get("missions", [])

        for mission in missions:
            mission_title = mission.get("title", "").lower()
            if any(term in q_lower for term in mission_title.split()):
                if not mission.get("passed", True):
                    return "Topic where candidate previously struggled"
                elif mission.get("skipped", False):
                    return "Topic candidate skipped — needs probing"
                elif mission.get("attempts", 1) > 2:
                    return "Topic with multiple attempts — verify understanding"

        return "General knowledge verification"

    def _heuristic_recommend(
        self, candidate: dict, questions: list[str], top_k: int
    ) -> list[dict[str, Any]]:
        """Fallback heuristic recommendation."""
        scored = []
        for q in questions:
            score = self._compute_relevance_score(q, candidate, np.array([]))
            scored.append({
                "question": q,
                "score": round(score, 3),
                "reason": self._get_recommendation_reason(q, candidate),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


# Singleton instance
_recommender: QuestionRecommender | None = None


def get_question_recommender() -> QuestionRecommender:
    """Get or create the singleton QuestionRecommender instance."""
    global _recommender
    if _recommender is None:
        _recommender = QuestionRecommender()
        _recommender.load()
    return _recommender
