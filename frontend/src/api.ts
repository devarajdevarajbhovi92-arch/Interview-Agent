export const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export interface Feedback {
  summary: string;
  strong_sections: string[];
  weak_sections: string[];
  areas_to_improve: string[];
  strengths?: string[];
  gaps?: string[];
  next?: string[];
  questions_answered?: number;
  total_questions?: number;
  completion_rate?: string;
  score?: number;
}

export interface InterviewResponse {
  reply: string;
  done: boolean;
  feedback?: Feedback;
}

export interface SessionQuestion {
  question: string;
  answer: string | null;
  judgment: {
    completeness: string;
    quality: string;
    reasoning: string;
  } | null;
}

export interface SessionQuestionsResponse {
  questions: SessionQuestion[];
  total_questions: number;
  questions_answered: number;
}

export async function startInterview(
  sessionId: string,
  candidate: object
): Promise<InterviewResponse> {
  const res = await fetch(`${API_BASE}/api/interview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId, candidate }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Server error on Turn 1");
  }
  return res.json();
}

export async function sendMessage(
  sessionId: string,
  message: string,
  action?: string
): Promise<InterviewResponse> {
  const res = await fetch(`${API_BASE}/api/interview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId, message, action }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Server error on message turn");
  }
  return res.json();
}

export async function finishInterview(sessionId: string): Promise<InterviewResponse> {
  return sendMessage(sessionId, "__END_INTERVIEW__", "finish");
}

export async function getSessionQuestions(sessionId: string): Promise<SessionQuestionsResponse> {
  const res = await fetch(`${API_BASE}/api/session/${sessionId}/questions`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Failed to fetch session questions");
  }
  return res.json();
}

// ── ML API Functions ──────────────────────────────────────────────────────

export interface MLModelStatus {
  models: Record<string, boolean>;
  loaded_count: number;
  total_count: number;
}

export interface MLScoreResult {
  score: number;
  confidence: number;
  source: string;
}

export interface MLTopicMastery {
  mastery: number;
  confidence: number;
  source: string;
}

export interface MLInsights {
  performance_prediction: {
    predicted_score: number;
    confidence: number;
    source: string;
  };
  topic_mastery: Record<string, MLTopicMastery>;
  weak_topics: string[];
  strong_topics: string[];
  model_status: Record<string, boolean>;
}

export async function getMLStatus(): Promise<MLModelStatus> {
  const res = await fetch(`${API_BASE}/api/ml/status`);
  if (!res.ok) throw new Error("Failed to fetch ML status");
  return res.json();
}

export async function scoreAnswer(question: string, answer: string): Promise<MLScoreResult> {
  const res = await fetch(`${API_BASE}/api/ml/score?question=${encodeURIComponent(question)}&answer=${encodeURIComponent(answer)}`);
  if (!res.ok) throw new Error("Failed to score answer");
  return res.json();
}

export async function getMLInsights(candidateId: string): Promise<MLInsights> {
  const res = await fetch(`${API_BASE}/api/ml/insights/${candidateId}`);
  if (!res.ok) throw new Error("Failed to fetch ML insights");
  return res.json();
}
