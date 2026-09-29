"""
generate_synthetic_data.py — Generate synthetic training data for all ML models.

Uses the existing curriculum and candidates JSON files to create realistic
synthetic Q&A pairs with known quality labels for training.

Output: CSV files in ml/data/ directory
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np

# Paths
_ROOT = Path(__file__).parent.parent.parent.parent
_CURRICULUM_PATH = _ROOT / "curriculum (2).json"
_CANDIDATES_PATH = _ROOT / "candidates.json"
_OUTPUT_DIR = Path(__file__).parent.parent / "data"

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# ── Synthetic answer templates ─────────────────────────────────────────────

STRONG_ANSWERS = [
    "I implemented a {tech} solution that reduced latency by 40%. For example, I used {tech2} to optimize the data pipeline, which improved throughput significantly.",
    "In my experience with {tech}, I designed a scalable architecture using {tech2}. The key trade-off was between consistency and availability, and I chose eventual consistency for better performance.",
    "I deployed a {tech} system in production that handles 10K requests per second. I monitored it with Prometheus and Grafana, and set up alerting for p99 latency.",
    "The approach I took was to first understand the problem domain, then design a solution using {tech}. I measured the results using A/B testing and saw a 25% improvement.",
    "I architected a {tech} solution that improved developer productivity by 30%. Specifically, I implemented automated testing and CI/CD pipelines using {tech2}.",
]

SHALLOW_ANSWERS = [
    "I think {tech} is a good technology for this use case. It has many features and is widely used in the industry.",
    "Maybe you could use {tech} for this? I believe it's popular and has good documentation.",
    "I'm not entirely sure, but I think {tech} works by processing data in some way. It's related to machine learning.",
    "Probably the best approach is to use {tech} since it's the standard. I guess it depends on the requirements.",
    "I suppose {tech} could work here. It's a technology I've heard about but haven't used extensively.",
]

CONFUSED_ANSWERS = [
    "Actually, wait. I think {tech} is different from what I said. Let me rephrase. It's more like a database but also a cache? No, that's not right either.",
    "Hmm, I'm a bit confused. Is {tech} the same as {tech2}? I think they're related but I'm not sure how.",
    "No, that's not quite right. I mean, {tech} does involve processing, but it's not exactly machine learning. Or is it? Let me think...",
    "Wait, I mixed things up. {tech} is actually for deployment, not for data processing. Or was I thinking of {tech2}?",
]

OFF_TOPIC_ANSWERS = [
    "I want to talk about something else entirely. Let me tell you about my favorite programming language instead.",
    "This is unrelated to the question, but I think the weather today is quite nice for a walk.",
    "Not about {tech}, but have you seen the latest movie? It was really interesting.",
    "Off topic, but I think the best part of the cohort was the community and networking.",
]

MISSING_ANSWERS = [
    "I don't know.",
    "Skip",
    "Pass",
    "No idea",
    "N/A",
    "",
]

# ── Topic extraction from curriculum ───────────────────────────────────────

def load_curriculum_topics() -> list[dict]:
    """Load topics from curriculum JSON."""
    with open(_CURRICULUM_PATH, encoding="utf-8") as f:
        data = json.load(f)

    topics = []
    for day in data.get("days", []):
        if day.get("type") != "SETUP":
            topics.append({
                "day": day["day"],
                "title": day["title"],
                "tools": day.get("tools", []),
                "objectives": day.get("objectives", []),
            })
    return topics


def load_candidates() -> list[dict]:
    """Load candidates from JSON."""
    with open(_CANDIDATES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("candidates", [])


# ── Synthetic data generators ──────────────────────────────────────────────

def generate_qa_pairs(topics: list[dict], n_samples: int = 5000) -> list[dict]:
    """Generate synthetic Q&A pairs with quality labels."""
    samples = []

    for i in range(n_samples):
        topic = random.choice(topics)
        tools = topic.get("tools", ["Python", "Docker"])
        tech = random.choice(tools) if tools else "Python"
        tech2 = random.choice(tools) if len(tools) > 1 else "Kubernetes"

        # Generate question
        question_templates = [
            f"Can you explain your approach to {topic['title']}?",
            f"How would you implement {tech} in a production environment?",
            f"What are the key challenges you faced with {topic['title']}?",
            f"Describe a project where you used {tech}.",
            f"How does {tech} compare to other tools you've used?",
            f"What is your experience with {tech} and {tech2}?",
            f"Can you walk me through your {topic['title']} project?",
            f"How do you handle errors when using {tech}?",
        ]
        question = random.choice(question_templates)

        # Generate answer with label
        quality_roll = random.random()
        if quality_roll < 0.15:
            answer = random.choice(MISSING_ANSWERS)
            completeness = "missing"
            quality = "missing"
        elif quality_roll < 0.30:
            answer = random.choice(OFF_TOPIC_ANSWERS)
            completeness = "partial"
            quality = "off_topic"
        elif quality_roll < 0.45:
            answer = random.choice(CONFUSED_ANSWERS).format(tech=tech, tech2=tech2)
            completeness = "partial"
            quality = "confused"
        elif quality_roll < 0.70:
            answer = random.choice(SHALLOW_ANSWERS).format(tech=tech, tech2=tech2)
            completeness = "partial"
            quality = "shallow"
        else:
            answer = random.choice(STRONG_ANSWERS).format(tech=tech, tech2=tech2)
            completeness = "full"
            quality = "strong"

        samples.append({
            "question": question,
            "answer": answer,
            "topic": topic["title"],
            "completeness": completeness,
            "quality": quality,
        })

    return samples


def generate_performance_data(candidates: list[dict], n_samples: int = 2000) -> list[dict]:
    """Generate synthetic performance prediction data."""
    samples = []

    for _ in range(n_samples):
        candidate = random.choice(candidates)
        signals = candidate.get("signals", {})
        missions = candidate.get("missions", [])

        completed = signals.get("missionsCompleted", 0)
        first_try = signals.get("missionsFirstTry", 0)
        commit_days = signals.get("commitDays", 0)

        # Generate score based on features with noise
        first_try_ratio = first_try / max(completed, 1)
        completion_ratio = completed / max(len(missions), 1)
        activity_ratio = min(commit_days / 30.0, 1.0)

        base_score = (
            first_try_ratio * 40
            + completion_ratio * 30
            + activity_ratio * 30
        )
        noise = np.random.normal(0, 5)
        score = max(0, min(100, base_score + noise))

        samples.append({
            "candidate_id": candidate["member"]["id"],
            "missions_completed": completed,
            "missions_first_try": first_try,
            "commit_days": commit_days,
            "pass_rate": completion_ratio,
            "skip_rate": sum(1 for m in missions if m.get("skipped", False)) / max(len(missions), 1),
            "avg_attempts": sum(m.get("attempts", 1) for m in missions) / max(len(missions), 1),
            "years_experience": candidate["member"].get("yearsExperience", 0),
            "score": round(score, 1),
        })

    return samples


def generate_difficulty_data(topics: list[dict], candidates: list[dict], n_samples: int = 3000) -> list[dict]:
    """Generate synthetic difficulty prediction data."""
    samples = []

    for _ in range(n_samples):
        topic = random.choice(topics)
        candidate = random.choice(candidates)
        signals = candidate.get("signals", {})

        first_try_ratio = signals.get("missionsFirstTry", 0) / max(signals.get("missionsCompleted", 1), 1)

        # Difficulty based on candidate performance and topic complexity
        if first_try_ratio > 0.7:
            difficulty = "easy"
        elif first_try_ratio > 0.4:
            difficulty = "medium"
        else:
            difficulty = "hard"

        # Add some noise
        if random.random() < 0.1:
            difficulty = random.choice(["easy", "medium", "hard"])

        samples.append({
            "topic": topic["title"],
            "candidate_id": candidate["member"]["id"],
            "first_try_ratio": round(first_try_ratio, 3),
            "missions_completed": signals.get("missionsCompleted", 0),
            "commit_days": signals.get("commitDays", 0),
            "difficulty": difficulty,
        })

    return samples


def generate_mastery_data(candidates: list[dict], topics: list[dict], n_samples: int = 2000) -> list[dict]:
    """Generate synthetic mastery estimation data."""
    samples = []

    for _ in range(n_samples):
        candidate = random.choice(candidates)
        topic = random.choice(topics)
        missions = candidate.get("missions", [])

        # Find relevant missions
        relevant = [m for m in missions if topic["title"].lower() in m.get("title", "").lower()]

        if not relevant:
            mastery = random.uniform(0, 30)
        else:
            passed = sum(1 for m in relevant if m.get("passed", False))
            total = len(relevant)
            mastery = (passed / total) * 100 + np.random.normal(0, 10)
            mastery = max(0, min(100, mastery))

        samples.append({
            "candidate_id": candidate["member"]["id"],
            "topic": topic["title"],
            "mission_attempts": sum(m.get("attempts", 1) for m in relevant),
            "mission_passed": any(m.get("passed", False) for m in relevant),
            "mission_skipped": any(m.get("skipped", False) for m in relevant),
            "mastery": round(mastery, 1),
        })

    return samples


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    """Generate all synthetic datasets."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading curriculum and candidates...")
    topics = load_curriculum_topics()
    candidates = load_candidates()
    print(f"  Found {len(topics)} topics and {len(candidates)} candidates")

    # Generate Q&A pairs
    print("Generating synthetic Q&A pairs...")
    qa_data = generate_qa_pairs(topics, n_samples=5000)
    qa_path = _OUTPUT_DIR / "synthetic_qa_pairs.csv"
    with open(qa_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "topic", "completeness", "quality"])
        writer.writeheader()
        writer.writerows(qa_data)
    print(f"  Saved {len(qa_data)} samples to {qa_path}")

    # Generate performance data
    print("Generating synthetic performance data...")
    perf_data = generate_performance_data(candidates, n_samples=2000)
    perf_path = _OUTPUT_DIR / "synthetic_performance.csv"
    with open(perf_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "candidate_id", "missions_completed", "missions_first_try", "commit_days",
            "pass_rate", "skip_rate", "avg_attempts", "years_experience", "score"
        ])
        writer.writeheader()
        writer.writerows(perf_data)
    print(f"  Saved {len(perf_data)} samples to {perf_path}")

    # Generate difficulty data
    print("Generating synthetic difficulty data...")
    diff_data = generate_difficulty_data(topics, candidates, n_samples=3000)
    diff_path = _OUTPUT_DIR / "synthetic_difficulty.csv"
    with open(diff_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "topic", "candidate_id", "first_try_ratio", "missions_completed", "commit_days", "difficulty"
        ])
        writer.writeheader()
        writer.writerows(diff_data)
    print(f"  Saved {len(diff_data)} samples to {diff_path}")

    # Generate mastery data
    print("Generating synthetic mastery data...")
    mastery_data = generate_mastery_data(candidates, topics, n_samples=2000)
    mastery_path = _OUTPUT_DIR / "synthetic_mastery.csv"
    with open(mastery_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "candidate_id", "topic", "mission_attempts", "mission_passed", "mission_skipped", "mastery"
        ])
        writer.writeheader()
        writer.writerows(mastery_data)
    print(f"  Saved {len(mastery_data)} samples to {mastery_path}")

    print("\nAll synthetic datasets generated successfully!")
    print(f"Output directory: {_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
