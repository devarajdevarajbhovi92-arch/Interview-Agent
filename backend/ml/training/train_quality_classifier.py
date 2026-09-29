"""
train_quality_classifier.py — Train the answer quality classification model.

Trains a Random Forest classifier with TF-IDF features to predict
completeness and quality labels from answer text.

Usage:
    python -m ml.training.train_quality_classifier
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from ..feature_engineering import extract_text_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"
_MODEL_DIR = Path(__file__).parent.parent / "models"
_QA_PATH = _DATA_DIR / "synthetic_qa_pairs.csv"

COMPLETENESS_LABELS = ["full", "partial", "missing"]
QUALITY_LABELS = ["strong", "shallow", "confused", "off_topic", "missing"]


def load_data() -> pd.DataFrame:
    """Load synthetic Q&A data."""
    if not _QA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {_QA_PATH}\n"
            "Run generate_synthetic_data.py first."
        )
    return pd.read_csv(_QA_PATH)


def prepare_features(df: pd.DataFrame):
    """Prepare TF-IDF and manual features."""
    # TF-IDF features
    tfidf = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words="english",
    )
    tfidf_matrix = tfidf.fit_transform(df["answer"].fillna(""))

    # Manual features
    manual_features = []
    for _, row in df.iterrows():
        feats = extract_text_features(str(row["answer"]))
        manual_features.append(list(feats.values()))
    manual_matrix = csr_matrix(manual_features)

    # Combine
    X = hstack([tfidf_matrix, manual_matrix])

    # Labels
    y_completeness = df["completeness"].map(
        {label: i for i, label in enumerate(COMPLETENESS_LABELS)}
    ).values
    y_quality = df["quality"].map(
        {label: i for i, label in enumerate(QUALITY_LABELS)}
    ).values
    y = [[c, q] for c, q in zip(y_completeness, y_quality)]

    return X, y, tfidf


def train() -> None:
    """Train the quality classifier."""
    logger.info("Loading data...")
    df = load_data()
    logger.info(f"Loaded {len(df)} samples")

    logger.info("Preparing features...")
    X, y, tfidf = prepare_features(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=[yi[0] for yi in y]
    )

    # Train
    logger.info("Training Random Forest classifier...")
    base_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf = MultiOutputClassifier(base_clf)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)

    logger.info("\n=== Completeness Classification Report ===")
    completeness_labels = [COMPLETENESS_LABELS[i] for i in range(len(COMPLETENESS_LABELS))]
    print(classification_report(
        [yi[0] for yi in y_test],
        [yi[0] for yi in y_pred],
        target_names=completeness_labels,
    ))

    logger.info("\n=== Quality Classification Report ===")
    quality_labels = [QUALITY_LABELS[i] for i in range(len(QUALITY_LABELS))]
    print(classification_report(
        [yi[1] for yi in y_test],
        [yi[1] for yi in y_pred],
        target_names=quality_labels,
    ))

    # Save
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MODEL_DIR / "quality_classifier.joblib"
    tfidf_path = _MODEL_DIR / "quality_tfidf.joblib"

    joblib.dump({"pipeline": clf}, model_path)
    joblib.dump(tfidf, tfidf_path)

    logger.info(f"\nModel saved to {model_path}")
    logger.info(f"Vectorizer saved to {tfidf_path}")


if __name__ == "__main__":
    train()
