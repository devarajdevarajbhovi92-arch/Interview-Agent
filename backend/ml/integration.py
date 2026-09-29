"""
integration.py — ML integration layer for the main application.

Provides a unified interface for all ML models to be used by the
FastAPI backend. Handles model loading, inference, and fallback logic.
"""

from __future__ import annotations

import logging
from typing import Any

from .quality_classifier import get_quality_classifier
from .difficulty_predictor import get_difficulty_predictor
from .performance_predictor import get_performance_predictor
from .mastery_estimator import get_mastery_estimator
from .answer_scorer import get_answer_scorer
from .question_recommender import get_question_recommender

logger = logging.getLogger(__name__)


class MLIntegration:
    """
    Unified ML integration layer.

    Provides a single interface for all ML models with automatic
    fallback to heuristics when models are not available.
    """

    def __init__(self) -> None:
        self._quality_clf = get_quality_classifier()
        self._difficulty_pred = get_difficulty_predictor()
        self._performance_pred = get_performance_predictor()
        self._mastery_est = get_mastery_estimator()
        self._answer_scorer = get_answer_scorer()
        self._question_rec = get_question_recommender()

        # Track which models are loaded
        self._model_status = {
            "quality_classifier": self._quality_clf.is_loaded,
            "difficulty_predictor": self._difficulty_pred.is_loaded,
            "performance_predictor": self._performance_pred.is_loaded,
            "mastery_estimator": self._mastery_est.is_loaded,
            "answer_scorer": self._answer_scorer.is_loaded,
            "question_recommender": self._question_rec.is_loaded,
        }

        loaded = sum(1 for v in self._model_status.values() if v)
        logger.info(f"ML Integration initialized: {loaded}/{len(self._model_status)} models loaded")

    @property
    def model_status(self) -> dict[str, bool]:
        """Return loading status of all models."""
        return self._model_status.copy()

    def judge_answer(self, question: str, answer: str) -> dict[str, Any]:
        """
        Judge an answer using ML classifier with heuristic fallback.

        Returns:
            dict with keys: completeness, quality, confidence, source
        """
        result = self._quality_clf.predict(question, answer)
        result["source"] = "ml" if self._quality_clf.is_loaded else "heuristic"
        return result

    def score_answer(self, question: str, answer: str) -> dict[str, Any]:
        """
        Score an answer continuously (0-100) using ML regressor.

        Returns:
            dict with keys: score, confidence, source
        """
        result = self._answer_scorer.score(question, answer)
        result["source"] = "ml" if self._answer_scorer.is_loaded else "heuristic"
        return result

    def predict_difficulty(self, question: str, candidate: dict) -> dict[str, Any]:
        """
        Predict question difficulty for a candidate.

        Returns:
            dict with keys: difficulty, confidence, source
        """
        result = self._difficulty_pred.predict(question, candidate)
        result["source"] = "ml" if self._difficulty_pred.is_loaded else "heuristic"
        return result

    def predict_performance(self, candidate: dict) -> dict[str, Any]:
        """
        Predict overall interview performance for a candidate.

        Returns:
            dict with keys: predicted_score, confidence, source
        """
        result = self._performance_pred.predict(candidate)
        result["source"] = "ml" if self._performance_pred.is_loaded else "heuristic"
        return result

    def estimate_mastery(self, candidate: dict, topic: str) -> dict[str, Any]:
        """
        Estimate topic mastery for a candidate.

        Returns:
            dict with keys: mastery, confidence, source
        """
        result = self._mastery_est.estimate_mastery(candidate, topic)
        result["source"] = "ml" if self._mastery_est.is_loaded else "heuristic"
        return result

    def estimate_all_mastery(self, candidate: dict) -> dict[str, dict[str, Any]]:
        """
        Estimate mastery for all topics.

        Returns:
            dict mapping topic -> {mastery, confidence, source}
        """
        results = self._mastery_est.estimate_all_topics(candidate)
        source = "ml" if self._mastery_est.is_loaded else "heuristic"
        for topic in results:
            results[topic]["source"] = source
        return results

    def recommend_questions(
        self,
        candidate: dict,
        available_questions: list[str],
        asked_questions: list[str] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Recommend next questions for a candidate.

        Returns:
            list of dicts with keys: question, score, reason, source
        """
        results = self._question_rec.recommend(
            candidate, available_questions, asked_questions, top_k
        )
        source = "ml" if self._question_rec.is_loaded else "heuristic"
        for r in results:
            r["source"] = source
        return results

    def get_insights(self, candidate: dict) -> dict[str, Any]:
        """
        Get comprehensive ML insights for a candidate.

        Returns:
            dict with keys: performance_prediction, topic_mastery, recommendations
        """
        performance = self.predict_performance(candidate)
        mastery = self.estimate_all_mastery(candidate)

        # Identify weak topics (mastery < 50)
        weak_topics = [
            topic for topic, data in mastery.items()
            if data.get("mastery", 0) < 50
        ]

        # Identify strong topics (mastery > 75)
        strong_topics = [
            topic for topic, data in mastery.items()
            if data.get("mastery", 0) > 75
        ]

        return {
            "performance_prediction": performance,
            "topic_mastery": mastery,
            "weak_topics": weak_topics,
            "strong_topics": strong_topics,
            "model_status": self._model_status,
        }


# Singleton instance
_ml_integration: MLIntegration | None = None


def get_ml_integration() -> MLIntegration:
    """Get or create the singleton MLIntegration instance."""
    global _ml_integration
    if _ml_integration is None:
        _ml_integration = MLIntegration()
    return _ml_integration
