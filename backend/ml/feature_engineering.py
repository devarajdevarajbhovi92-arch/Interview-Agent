"""
feature_engineering.py — Shared feature extraction for all ML models.

Extracts numerical and categorical features from text and structured data
for use in scikit-learn pipelines.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Technical term dictionaries per topic ──────────────────────────────────

TOPIC_TERMS: dict[str, list[str]] = {
    "embeddings": ["embedding", "vector", "dimension", "similarity", "cosine", "semantic"],
    "vector_databases": ["vector database", "pinecone", "weaviate", "milvus", "faiss", "chroma", "index"],
    "retrieval": ["retrieval", "search", "ranking", "bm25", "tf-idf", "matching", "query"],
    "prompt_engineering": ["prompt", "token", "context", "instruction", "few-shot", "zero-shot", "chain"],
    "function_calling": ["function", "api", "tool", "schema", "json", "structured output", "call"],
    "chatbot_backend": ["backend", "api", "rest", "fastapi", "flask", "endpoint", "websocket"],
    "streaming": ["stream", "chunk", "sse", "websocket", "real-time", "buffer", "async"],
    "multi_agent": ["agent", "orchestration", "coordination", "workflow", "pipeline", "multi-agent"],
    "mcp": ["mcp", "model context protocol", "protocol", "server", "client", "tool"],
    "docker": ["docker", "container", "image", "dockerfile", "compose", "kubernetes", "k8s", "deploy"],
    "monitoring": ["monitoring", "logging", "observability", "metrics", "tracing", "alerting", "grafana"],
    "capstone": ["capstone", "project", "demo", "presentation", "deployment", "production"],
    "data_foundations": ["pandas", "numpy", "dataframe", "sql", "etl", "data cleaning", "preprocessing"],
    "llm_core": ["llm", "transformer", "attention", "fine-tuning", "lora", "qlora", "pretraining"],
    "evaluation": ["evaluation", "metric", "accuracy", "precision", "recall", "f1", "benchmark"],
    "security": ["security", "auth", "oauth", "jwt", "encryption", "vulnerability", "injection"],
}

# ── Quality indicator phrases ──────────────────────────────────────────────

STRONG_INDICATORS = [
    "for example", "specifically", "in my experience", "i implemented",
    "the result was", "i measured", "the metric", "in production",
    "i designed", "i architected", "the trade-off", "i optimized",
    "the benchmark", "i deployed", "the improvement",
]

SHALLOW_INDICATORS = [
    "i think", "maybe", "probably", "i guess", "not sure",
    "i believe", "sort of", "kind of", "i suppose",
]

CONFUSED_INDICATORS = [
    "actually", "wait", "no", "i mean", "let me rephrase",
    "that's not right", "i confused", "mixed up",
]

OFF_TOPIC_INDICATORS = [
    "unrelated", "different topic", "not about", "off topic",
    "i want to talk about", "let me tell you about something else",
]


def extract_text_features(text: str) -> dict[str, float]:
    """Extract numerical features from answer text."""
    text_lower = text.lower()
    words = text_lower.split()

    features: dict[str, float] = {
        "char_count": float(len(text)),
        "word_count": float(len(words)),
        "sentence_count": float(len(re.split(r"[.!?]+", text))),
        "avg_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
        "has_code": float(bool(re.search(r"```|def |class |import |function |const |let |var ", text_lower))),
        "has_numbers": float(bool(re.search(r"\d+", text))),
        "has_example": float(any(p in text_lower for p in STRONG_INDICATORS)),
        "has_uncertainty": float(any(p in text_lower for p in SHALLOW_INDICATORS)),
        "has_correction": float(any(p in text_lower for p in CONFUSED_INDICATORS)),
        "has_off_topic": float(any(p in text_lower for p in OFF_TOPIC_INDICATORS)),
        "question_marks": float(text.count("?")),
        "exclamation_marks": float(text.count("!")),
        "uppercase_ratio": float(sum(1 for c in text if c.isupper()) / max(len(text), 1)),
    }

    # Topic-specific term counts
    for topic, terms in TOPIC_TERMS.items():
        features[f"topic_{topic}"] = float(sum(1 for t in terms if t in text_lower))

    return features


def extract_question_features(question: str) -> dict[str, float]:
    """Extract features from question text."""
    q_lower = question.lower()
    words = q_lower.split()

    features: dict[str, float] = {
        "q_char_count": float(len(question)),
        "q_word_count": float(len(words)),
        "q_has_how": float("how" in q_lower),
        "q_has_why": float("why" in q_lower),
        "q_has_explain": float("explain" in q_lower or "describe" in q_lower),
        "q_has_compare": float("compare" in q_lower or "difference" in q_lower),
        "q_has_design": float("design" in q_lower or "architect" in q_lower),
        "q_has_implement": float("implement" in q_lower or "build" in q_lower),
        "q_has_troubleshoot": float("troubleshoot" in q_lower or "debug" in q_lower or "fix" in q_lower),
        "q_has_example": float("example" in q_lower or "instance" in q_lower),
    }

    # Detect topic
    for topic, terms in TOPIC_TERMS.items():
        features[f"q_topic_{topic}"] = float(any(t in q_lower for t in terms))

    return features


def extract_candidate_features(candidate: dict) -> dict[str, float]:
    """Extract features from candidate profile."""
    signals = candidate.get("signals", {})
    missions = candidate.get("missions", [])

    total_missions = len(missions)
    passed = sum(1 for m in missions if m.get("passed", False))
    skipped = sum(1 for m in missions if m.get("skipped", False))
    total_attempts = sum(m.get("attempts", 1) for m in missions)

    features: dict[str, float] = {
        "cand_missions_completed": float(signals.get("missionsCompleted", 0)),
        "cand_missions_first_try": float(signals.get("missionsFirstTry", 0)),
        "cand_commit_days": float(signals.get("commitDays", 0)),
        "cand_pass_rate": float(passed / max(total_missions, 1)),
        "cand_skip_rate": float(skipped / max(total_missions, 1)),
        "cand_avg_attempts": float(total_attempts / max(total_missions, 1)),
        "cand_years_exp": float(candidate.get("member", {}).get("yearsExperience", 0)),
    }

    return features


def build_tfidf_vectorizer(max_features: int = 500) -> TfidfVectorizer:
    """Build a TF-IDF vectorizer for text features."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words="english",
    )


def combine_features(
    text_feats: dict[str, float],
    question_feats: dict[str, float] | None = None,
    candidate_feats: dict[str, float] | None = None,
) -> np.ndarray:
    """Combine all feature dicts into a single numpy array."""
    combined: dict[str, float] = {}
    combined.update(text_feats)
    if question_feats:
        combined.update(question_feats)
    if candidate_feats:
        combined.update(candidate_feats)
    return np.array(list(combined.values()), dtype=np.float32)


def get_feature_names(
    text_feats: dict[str, float],
    question_feats: dict[str, float] | None = None,
    candidate_feats: dict[str, float] | None = None,
) -> list[str]:
    """Get feature names in order."""
    names = list(text_feats.keys())
    if question_feats:
        names.extend(question_feats.keys())
    if candidate_feats:
        names.extend(candidate_feats.keys())
    return names
