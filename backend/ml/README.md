# ML Module — Machine Learning for MAESTER AI Interview Agent

## Overview

This module adds machine learning capabilities to the MAESTER AI Interview Agent,
enabling data-driven decision making alongside the existing LLM-based approach.

## Architecture

```
ml/
├── __init__.py                    # Module exports
├── feature_engineering.py         # Shared feature extraction
├── quality_classifier.py          # Answer quality classification
├── difficulty_predictor.py        # Question difficulty prediction
├── performance_predictor.py       # Candidate performance prediction
├── mastery_estimator.py           # Topic mastery estimation (BKT)
├── answer_scorer.py               # Continuous answer scoring
├── question_recommender.py        # Personalized question recommendation
├── integration.py                 # Unified ML integration layer
├── models/                        # Trained model files (.joblib)
├── training/                      # Training scripts
│   ├── generate_synthetic_data.py
│   ├── train_quality_classifier.py
│   ├── train_difficulty_predictor.py
│   ├── train_performance_predictor.py
│   ├── train_answer_scorer.py
│   ├── train_mastery_estimator.py
│   ├── train_question_recommender.py
│   └── train_all.py
└── data/                          # Training datasets (CSV)
    ├── synthetic_qa_pairs.csv
    ├── synthetic_performance.csv
    ├── synthetic_difficulty.csv
    └── synthetic_mastery.csv
```

## Models

### 1. Quality Classifier
- **Type:** Random Forest + TF-IDF
- **Input:** Question text, Answer text
- **Output:** Completeness (full/partial/missing), Quality (strong/shallow/confused/off_topic)
- **Training Data:** 5,000 synthetic Q&A pairs
- **Usage:** `ml.judge_answer(question, answer)`

### 2. Difficulty Predictor
- **Type:** Gradient Boosting Classifier
- **Input:** Question features, Candidate features
- **Output:** Difficulty (easy/medium/hard)
- **Training Data:** 3,000 synthetic samples
- **Usage:** `ml.predict_difficulty(question, candidate)`

### 3. Performance Predictor
- **Type:** Ridge Regression
- **Input:** Candidate profile features
- **Output:** Predicted score (0-100)
- **Training Data:** 2,000 synthetic samples
- **Usage:** `ml.predict_performance(candidate)`

### 4. Mastery Estimator
- **Type:** Bayesian Knowledge Tracing (BKT)
- **Input:** Candidate mission history, Topic
- **Output:** Mastery level (0-100%)
- **Training Data:** 2,000 synthetic samples
- **Usage:** `ml.estimate_mastery(candidate, topic)`

### 5. Answer Scorer
- **Type:** Gradient Boosting Regressor
- **Input:** Question text, Answer text
- **Output:** Continuous score (0-100)
- **Training Data:** 5,000 synthetic Q&A pairs
- **Usage:** `ml.score_answer(question, answer)`

### 6. Question Recommender
- **Type:** Content-Based Filtering
- **Input:** Candidate profile, Available questions
- **Output:** Top-K recommended questions with scores
- **Training Data:** 2,000 synthetic samples
- **Usage:** `ml.recommend_questions(candidate, questions)`

## Training Pipeline

### Step 1: Generate Synthetic Data
```bash
cd backend
python -m ml.training.generate_synthetic_data
```

### Step 2: Train All Models
```bash
python -m ml.training.train_all
```

### Step 3: Verify Models
```bash
python -c "from ml.integration import get_ml_integration; ml = get_ml_integration(); print(ml.model_status)"
```

## Integration

The ML models are integrated into the main application through `ml/integration.py`:

```python
from ml.integration import get_ml_integration

ml = get_ml_integration()

# Judge an answer (ML with heuristic fallback)
judgment = ml.judge_answer(question, answer)

# Score an answer continuously
score = ml.score_answer(question, answer)

# Get comprehensive insights
insights = ml.get_insights(candidate)
```

## Fallback Strategy

All ML models include heuristic fallbacks:
- If a model file is not found, heuristics are used
- If ML prediction fails, heuristics are used
- The `source` field in responses indicates whether ML or heuristic was used

## Data Flow

```
Candidate Profile → Performance Predictor → Initial Difficulty
                                          ↓
Question → Difficulty Predictor → Adaptive Question Selection
                                          ↓
Answer → Quality Classifier → Judgment → Follow-up Decision
                                          ↓
Answer → Answer Scorer → Continuous Score → Feedback
                                          ↓
Mission History → Mastery Estimator → Topic Mastery → Personalization
```
