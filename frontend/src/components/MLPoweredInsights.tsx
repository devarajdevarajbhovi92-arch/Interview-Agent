/**
 * MLPoweredInsights.tsx — Display ML-powered insights for a candidate.
 */

import { useEffect, useState } from "react";
import { API_BASE } from "../api";

interface MLInsights {
  performance_prediction: {
    predicted_score: number;
    confidence: number;
    source: string;
  };
  topic_mastery: Record<string, { mastery: number; confidence: number; source: string }>;
  weak_topics: string[];
  strong_topics: string[];
  model_status: Record<string, boolean>;
}

interface Props {
  candidateId: string;
}

export default function MLPoweredInsights({ candidateId }: Props) {
  const [insights, setInsights] = useState<MLInsights | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchInsights() {
      try {
        setLoading(true);
        const res = await fetch(`${API_BASE}/api/ml/insights/${candidateId}`);
        if (!res.ok) throw new Error("Failed to fetch insights");
        const data = await res.json();
        setInsights(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    }
    fetchInsights();
  }, [candidateId]);

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 mb-4">
        <div className="animate-pulse space-y-3">
          <div className="h-4 bg-gray-700 rounded w-1/2"></div>
          <div className="h-4 bg-gray-700 rounded w-3/4"></div>
          <div className="h-4 bg-gray-700 rounded w-2/3"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 mb-4">
        <p className="text-red-400 text-sm">ML Insights unavailable: {error}</p>
      </div>
    );
  }

  if (!insights) return null;

  const loadedModels = Object.values(insights.model_status).filter(Boolean).length;
  const totalModels = Object.keys(insights.model_status).length;

  return (
    <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-5 mb-4 border border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <span className="text-2xl">🤖</span> ML-Powered Insights
        </h3>
        <span className="text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded-full">
          {loadedModels}/{totalModels} models loaded
        </span>
      </div>

      {/* Performance Prediction */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-1">
          <span className="text-sm text-gray-400">Predicted Score</span>
          <span className="text-sm font-mono text-emerald-400">
            {insights.performance_prediction.predicted_score}/100
          </span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div
            className="bg-emerald-500 h-2 rounded-full transition-all"
            style={{ width: `${insights.performance_prediction.predicted_score}%` }}
          ></div>
        </div>
      </div>

      {/* Weak Topics */}
      {insights.weak_topics.length > 0 && (
        <div className="mb-3">
          <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">Weak Areas</p>
          <div className="flex flex-wrap gap-1">
            {insights.weak_topics.slice(0, 5).map((topic) => (
              <span
                key={topic}
                className="text-xs bg-red-900/40 text-red-300 px-2 py-0.5 rounded"
              >
                {topic}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Strong Topics */}
      {insights.strong_topics.length > 0 && (
        <div className="mb-3">
          <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">Strong Areas</p>
          <div className="flex flex-wrap gap-1">
            {insights.strong_topics.slice(0, 5).map((topic) => (
              <span
                key={topic}
                className="text-xs bg-emerald-900/40 text-emerald-300 px-2 py-0.5 rounded"
              >
                {topic}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Model Status */}
      <div className="mt-3 pt-3 border-t border-gray-700">
        <p className="text-xs text-gray-500 uppercase tracking-wide mb-2">Model Status</p>
        <div className="grid grid-cols-2 gap-1">
          {Object.entries(insights.model_status).map(([name, loaded]) => (
            <div key={name} className="flex items-center gap-1.5">
              <span
                className={`w-1.5 h-1.5 rounded-full ${loaded ? "bg-emerald-400" : "bg-gray-600"}`}
              ></span>
              <span className="text-xs text-gray-400">
                {name.replace(/_/g, " ")}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
