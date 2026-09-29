# AI-Powered Interview Preparation System — Complete ML-Enhanced Project

## Project Overview

**MAESTER AI Interview Agent** is a machine learning-enhanced technical interview system that conducts adaptive, multi-turn technical interviews. It combines LLM intelligence (Groq) with trained ML models to deliver personalized, data-driven interview experiences.

---

## Complete Project Structure

```
C:\Users\devaraj\interview agent\Interview-Agent\
│
├── backend\
│   ├── main.py                          # FastAPI app (ML-integrated)
│   ├── llm.py                           # Groq API calls (fallback layer)
│   ├── session.py                       # InterviewSession & Turn dataclasses
│   ├── planner.py                       # Question plan builder
│   ├── data.py                          # JSON data loader
│   ├── prompts.py                       # LLM prompt templates
│   ├── requirements.txt                 # Python dependencies (updated)
│   │
│   ├── ml\                              # ★ NEW: ML Module
│   │   ├── __init__.py                  # Module exports
│   │   ├── feature_engineering.py       # Shared feature extraction
│   │   ├── quality_classifier.py        # Answer quality classification
│   │   ├── difficulty_predictor.py      # Question difficulty prediction
│   │   ├── performance_predictor.py     # Candidate performance prediction
│   │   ├── mastery_estimator.py         # Topic mastery estimation (BKT)
│   │   ├── answer_scorer.py             # Continuous answer scoring
│   │   ├── question_recommender.py      # Personalized question recommendation
│   │   ├── integration.py               # Unified ML integration layer
│   │   ├── README.md                    # ML module documentation
│   │   │
│   │   ├── models\                      # Trained model files
│   │   │   ├── quality_classifier.joblib
│   │   │   ├── quality_tfidf.joblib
│   │   │   ├── difficulty_predictor.joblib
│   │   │   ├── performance_predictor.joblib
│   │   │   ├── mastery_estimator.joblib
│   │   │   ├── answer_scorer.joblib
│   │   │   └── question_recommender.joblib
│   │   │
│   │   ├── training\                    # Training scripts
│   │   │   ├── __init__.py
│   │   │   ├── generate_synthetic_data.py
│   │   │   ├── train_quality_classifier.py
│   │   │   ├── train_difficulty_predictor.py
│   │   │   ├── train_performance_predictor.py
│   │   │   ├── train_answer_scorer.py
│   │   │   ├── train_mastery_estimator.py
│   │   │   ├── train_question_recommender.py
│   │   │   └── train_all.py
│   │   │
│   │   └── data\                        # Training datasets
│   │       ├── synthetic_qa_pairs.csv
│   │       ├── synthetic_performance.csv
│   │       ├── synthetic_difficulty.csv
│   │       └── synthetic_mastery.csv
│   │
│   └── ...
│
├── frontend\
│   ├── src\
│   │   ├── components\
│   │   │   ├── MLPoweredInsights.tsx    # ★ NEW: ML insights display
│   │   │   └── ...
│   │   └── ...
│   └── ...
│
├── candidates.json                      # Candidate data (20 candidates)
├── curriculum (2).json                  # Curriculum data (31 days, 8 modules)
└── ...
```

---

## Dataset Description

### 1. Synthetic Q&A Pairs Dataset
**File:** `C:\Users\devaraj\interview agent\Interview-Agent\backend\ml\data\synthetic_qa_pairs.csv`
**Records:** 5,000 samples

| Column | Type | Description |
|--------|------|-------------|
| `question` | text | Interview question |
| `answer` | text | Candidate's answer |
| `topic` | categorical | Topic category |
| `completeness` | categorical | `full`, `partial`, `missing` |
| `quality` | categorical | `strong`, `shallow`, `confused`, `off_topic` |

**Distribution:**
- Strong answers: ~30%
- Shallow answers: ~25%
- Confused answers: ~15%
- Off-topic answers: ~15%
- Missing answers: ~15%

### 2. Synthetic Performance Dataset
**File:** `C:\Users\devaraj\interview agent\Interview-Agent\backend\ml\data\synthetic_performance.csv`
**Records:** 2,000 samples

| Column | Type | Description |
|--------|------|-------------|
| `candidate_id` | categorical | Candidate identifier |
| `missions_completed` | int | Total missions completed |
| `missions_first_try` | int | Missions passed on first try |
| `commit_days` | int | Active days in cohort |
| `pass_rate` | float | Proportion of passed missions |
| `skip_rate` | float | Proportion of skipped missions |
| `avg_attempts` | float | Average attempts per mission |
| `years_experience` | float | Years of experience |
| `score` | float | Target score (0-100) |

### 3. Synthetic Difficulty Dataset
**File:** `C:\Users\devaraj\interview agent\Interview-Agent\backend\ml\data\synthetic_difficulty.csv`
**Records:** 3,000 samples

| Column | Type | Description |
|--------|------|-------------|
| `topic` | categorical | Question topic |
| `candidate_id` | categorical | Candidate identifier |
| `first_try_ratio` | float | Candidate's first-try pass rate |
| `missions_completed` | int | Total missions completed |
| `commit_days` | int | Active days |
| `difficulty` | categorical | `easy`, `medium`, `hard` |

### 4. Synthetic Mastery Dataset
**File:** `C:\Users\devaraj\interview agent\Interview-Agent\backend\ml\data\synthetic_mastery.csv`
**Records:** 2,000 samples

| Column | Type | Description |
|--------|------|-------------|
| `candidate_id` | categorical | Candidate identifier |
| `topic` | categorical | Curriculum topic |
| `mission_attempts` | int | Attempts on this topic |
| `mission_passed` | binary | Pass/fail |
| `mission_skipped` | binary | Skipped |
| `mastery` | float | Mastery level (0-100) |

---

## ML Models Summary

| Model | Type | Input | Output | Training Data |
|-------|------|-------|--------|---------------|
| Quality Classifier | Random Forest + TF-IDF | Question + Answer | Completeness + Quality | 5,000 Q&A pairs |
| Difficulty Predictor | Gradient Boosting | Question + Candidate | Difficulty (easy/medium/hard) | 3,000 samples |
| Performance Predictor | Ridge Regression | Candidate profile | Score (0-100) | 2,000 samples |
| Mastery Estimator | Bayesian Knowledge Tracing | Mission history + Topic | Mastery (0-100%) | 2,000 samples |
| Answer Scorer | Gradient Boosting Regressor | Question + Answer | Score (0-100) | 5,000 Q&A pairs |
| Question Recommender | Content-Based Filtering | Candidate + Questions | Top-K recommendations | 2,000 samples |

---

## How to Run

### 1. Install Dependencies
```bash
cd C:\Users\devaraj\interview agent\Interview-Agent\backend
pip install -r requirements.txt
```

### 2. Generate Synthetic Data
```bash
python -m ml.training.generate_synthetic_data
```

### 3. Train All Models
```bash
python -m ml.training.train_all
```

### 4. Start the Application
```bash
uvicorn main:app --reload
```

### 5. Access the Application
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/init-interview` | POST | Initialize interview with resume |
| `/api/interview` | POST | Send message / get response |
| `/api/session/{id}/questions` | GET | Get session questions |
| `/api/ml/insights/{candidate_id}` | GET | Get ML insights for candidate |
| `/api/ml/recommend` | POST | Get question recommendations |

---

## Key Features

### 1. Adaptive Questioning
- Questions are selected based on candidate's weak areas
- Difficulty adapts in real-time based on performance
- Follow-up questions are generated based on answer quality

### 2. ML-Powered Judgments
- Answer quality is classified by trained ML model
- Continuous scoring provides granular feedback
- Confidence scores indicate prediction reliability

### 3. Personalized Feedback
- Performance prediction sets expectations
- Topic mastery estimation identifies knowledge gaps
- Recommendations focus on areas needing improvement

### 4. Fallback Strategy
- All ML models have heuristic fallbacks
- System works even without trained models
- Source tracking indicates ML vs heuristic decisions

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         CANDIDATE                                │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │   Resume    │  │  Profile     │  │  Mission History      │  │
│  └──────┬──────┘  └──────┬───────┘  └───────────┬───────────┘  │
└─────────┼─────────────────┼──────────────────────┼──────────────┘
          │                 │                      │
          ▼                 ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ML LAYER                                    │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ Performance      │  │ Mastery          │                    │
│  │ Predictor        │  │ Estimator        │                    │
│  │ → Initial Score  │  │ → Topic Mastery  │                    │
│  └──────────────────┘  └──────────────────┘                    │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ Difficulty       │  │ Question         │                    │
│  │ Predictor        │  │ Recommender      │                    │
│  │ → Adaptive Level │  │ → Personalized   │                    │
│  └──────────────────┘  └──────────────────┘                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INTERVIEW LOOP                                │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │   Question  │→ │   Answer     │→ │   ML Judgment         │  │
│  │   Selection │  │   Collection │  │   + Scoring           │  │
│  └─────────────┘  └──────────────┘  └───────────────────────┘  │
│         ▲                                    │                   │
│         │                                    ▼                   │
│         │                           ┌──────────────────┐        │
│         └───────────────────────────│  Follow-up       │        │
│                                     │  Decision        │        │
│                                     └──────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FEEDBACK                                    │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │   Summary   │  │   Score      │  │   Recommendations     │  │
│  └─────────────┘  └──────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technical Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.11, FastAPI, Uvicorn |
| ML | scikit-learn, NumPy, pandas, joblib, SciPy |
| LLM | Groq API (openai/gpt-oss-120b) |
| Frontend | React, TypeScript, Vite, Tailwind CSS |
| Data Storage | JSON files, in-memory dicts, browser localStorage |
| Model Storage | joblib serialized files |

---

## Future Enhancements

1. **Real Data Collection:** Replace synthetic data with real interview transcripts
2. **Deep Learning:** Fine-tune BERT for answer quality classification
3. **Reinforcement Learning:** Train RL agent for optimal question selection
4. **Multi-Modal:** Add voice analysis for communication skills assessment
5. **Production Deployment:** Add Redis for session persistence, Docker for containerization
