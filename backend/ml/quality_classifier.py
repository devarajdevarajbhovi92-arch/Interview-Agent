"""
quality_classifier.py — Answer quality classification model.

Predicts completeness (full/partial/missing) and quality (strong/shallow/confused/off_topic)
from answer text using a Random Forest classifier with TF-IDF features.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from .feature_engineering import extract_text_features

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent / "models"
_MODEL_PATH = _MODEL_DIR / "quality_classifier.joblib"
_VECTORIZER_PATH = _MODEL_DIR / "quality_tfidf.joblib"

COMPLETENESS_LABELS = ["full", "partial", "missing"]
QUALITY_LABELS = ["strong", "shallow", "confused", "off_topic", "missing"]


class QualityClassifier:
    """ML-based answer quality classifier."""

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._tfidf: TfidfVectorizer | None = None
        self._is_loaded = False

    def load(self) -> bool:
        """Load pre-trained model from disk."""
        if _MODEL_PATH.exists() and _VECTORIZER_PATH.exists():
            try:
                data = joblib.load(_MODEL_PATH)
                self._pipeline = data["pipeline"]
                self._tfidf = joblib.load(_VECTORIZER_PATH)
                self._is_loaded = True
                logger.info("Quality classifier loaded from disk.")
                return True
            except Exception as e:
                logger.warning(f"Failed to load quality classifier: {e}")
        return False

    def predict(self, question: str, answer: str) -> dict[str, str]:
        """
        Predict completeness and quality for an answer.

        Returns:
            dict with keys: completeness, quality, confidence
        """
        if not self._is_loaded or self._pipeline is None:
            return self._heuristic_predict(question, answer)

        try:
            text_feats = extract_text_features(answer)
            tfidf_matrix = self._tfidf.transform([answer])
            manual_feats = np.array(list(text_feats.values()), dtype=np.float32).reshape(1, -1)

            # Combine TF-IDF + manual features
            from scipy.sparse import hstack, csr_matrix
            combined = hstack([tfidf_matrix, csr_matrix(manual_feats)])

            # Get predictions from each estimator separately
            predictions = []
            confidences = []
            for estimator in self._pipeline.estimators_:
                pred = estimator.predict(combined)
                proba = estimator.predict_proba(combined)
                predictions.append(int(pred[0]))
                confidences.append(float(max(proba[0])))

            completeness = COMPLETENESS_LABELS[predictions[0]]
            quality = QUALITY_LABELS[predictions[1]]
            confidence = round(float(np.mean(confidences)), 3)

            return {
                "completeness": completeness,
                "quality": quality,
                "confidence": confidence,
            }
        except Exception as e:
            logger.warning(f"ML prediction failed, using heuristic: {e}")
            return self._heuristic_predict(question, answer)
        except Exception as e:
            logger.warning(f"ML prediction failed, using heuristic: {e}")
            return self._heuristic_predict(question, answer)

    def _heuristic_predict(self, question: str, answer: str) -> dict[str, str]:
        """Fallback heuristic when model is not available."""
        text_lower = answer.lower().strip()

        # Missing detection
        skip_phrases = {"i don't know", "skip", "pass", "no idea", "n/a", ""}
        if text_lower in skip_phrases or len(text_lower) < 10:
            return {"completeness": "missing", "quality": "missing", "confidence": 0.5}

        # Off-topic detection
        off_topic_phrases = ["unrelated", "different topic", "not about", "off topic"]
        if any(p in text_lower for p in off_topic_phrases):
            return {"completeness": "partial", "quality": "off_topic", "confidence": 0.5}

        # Confused detection
        confused_phrases = ["actually", "wait", "no", "i mean", "let me rephrase"]
        if any(p in text_lower for p in confused_phrases):
            return {"completeness": "partial", "quality": "confused", "confidence": 0.5}

        # Strong detection
        strong_phrases = [
            "for example", "specifically", "i implemented", "i designed",
            "the result was", "in production", "i optimized",
        ]
        if any(p in text_lower for p in strong_phrases) and len(answer) > 100:
            return {"completeness": "full", "quality": "strong", "confidence": 0.6}

        # Shallow detection
        uncertain_phrases = ["i think", "maybe", "probably", "i guess", "not sure"]
        if any(p in text_lower for p in uncertain_phrases) or len(answer) < 50:
            return {"completeness": "partial", "quality": "shallow", "confidence": 0.5}

        # Default
        return {"completeness": "partial", "quality": "shallow", "confidence": 0.5}

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


# Singleton instance
_classifier: QualityClassifier | None = None


def get_quality_classifier() -> QualityClassifier:
    """Get or create the singleton QualityClassifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = QualityClassifier()
        _classifier.load()
    return _classifier
