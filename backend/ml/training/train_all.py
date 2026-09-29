"""
train_all.py — Train all ML models in sequence.

Usage:
    python -m ml.training.train_all
"""

from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Train all models."""
    logger.info("=" * 60)
    logger.info("TRAINING ALL ML MODELS")
    logger.info("=" * 60)

    # 1. Generate synthetic data
    logger.info("\n[1/6] Generating synthetic data...")
    from .generate_synthetic_data import main as generate_data
    generate_data()

    # 2. Train quality classifier
    logger.info("\n[2/6] Training quality classifier...")
    from .train_quality_classifier import train as train_quality
    train_quality()

    # 3. Train difficulty predictor
    logger.info("\n[3/6] Training difficulty predictor...")
    from .train_difficulty_predictor import train as train_difficulty
    train_difficulty()

    # 4. Train performance predictor
    logger.info("\n[4/6] Training performance predictor...")
    from .train_performance_predictor import train as train_performance
    train_performance()

    # 5. Train answer scorer
    logger.info("\n[5/6] Training answer scorer...")
    from .train_answer_scorer import train as train_scorer
    train_scorer()

    # 6. Train mastery estimator
    logger.info("\n[6/6] Training mastery estimator...")
    from .train_mastery_estimator import train as train_mastery
    train_mastery()

    logger.info("\n" + "=" * 60)
    logger.info("ALL MODELS TRAINED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
