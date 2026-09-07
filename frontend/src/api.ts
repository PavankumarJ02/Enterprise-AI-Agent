/**
 * HTTP API client for Enterprise AI Agent backend.
 */

import type {
  HealthResponse,
  HybridSearchRequest,
  HybridSearchResponse,
  PerformanceStatsResponse,
  RAGQueryRequest,
  RAGQueryResponse,
  SQLQueryRequest,
  SQLQueryResponse,
} from "./types";

const API_BASE = "";

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let errorDetail = `HTTP ${response.status} ${response.statusText}`;
    try {
      const errorJson = await response.json();
      errorDetail = errorJson.detail || errorJson.message || JSON.stringify(errorJson);
    } catch {
      // ignore JSON parse failure
    }
    throw new Error(errorDetail);
  }

  return response.json() as Promise<T>;
}

export const api = {
  async getHealth(): Promise<HealthResponse> {
    return request<HealthResponse>("/api/v1/health");
  },

  async sendRAGQuery(query: string, topK: number = 3): Promise<RAGQueryResponse> {
    const payload: RAGQueryRequest = {
      query,
      top_k: topK,
      retrieval_strategy: "hybrid",
    };
    return request<RAGQueryResponse>("/api/v1/rag/query", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  async sendDirectChat(query: string): Promise<{ answer: string; model: string; latency_ms: number }> {
    return request<{ answer: string; model: string; latency_ms: number }>("/api/v1/chat", {
      method: "POST",
      body: JSON.stringify({ query }),
    });
  },

  async executeHybridSearch(
    query: string,
    topK: number = 5,
    denseWeight: number = 0.5
  ): Promise<HybridSearchResponse> {
    const payload: HybridSearchRequest = {
      query,
      top_k: topK,
      dense_weight: denseWeight,
      sparse_weight: Math.round((1.0 - denseWeight) * 100) / 100,
      fusion_method: "weighted",
    };
    return request<HybridSearchResponse>("/api/v1/search/hybrid", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  async executeSQL(query: string): Promise<SQLQueryResponse> {
    const payload: SQLQueryRequest = { query };
    return request<SQLQueryResponse>("/api/v1/sql/query", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  async validateGuardrails(text: string): Promise<{
    passed: boolean;
    action_taken: string;
    sanitized_text: string;
    violations: Array<{
      violation_type: string;
      severity: string;
      message: string;
      action_taken: string;
    }>;
  }> {
    return request<{
      passed: boolean;
      action_taken: string;
      sanitized_text: string;
      violations: Array<{
        violation_type: string;
        severity: string;
        message: string;
        action_taken: string;
      }>;
    }>("/api/v1/guardrails/validate", {
      method: "POST",
      body: JSON.stringify({
        text,
        check_input: true,
        check_pii: true,
        check_injection: true,
      }),
    });
  },

  async redactPII(text: string): Promise<{
    redacted_text: string;
    entities_found: Record<string, number>;
    total_entities: number;
  }> {
    return request<{
      redacted_text: string;
      entities_found: Record<string, number>;
      total_entities: number;
    }>("/api/v1/guardrails/redact", {
      method: "POST",
      body: JSON.stringify({ text }),
    });
  },

  async getPerformanceStats(): Promise<PerformanceStatsResponse> {
    return request<PerformanceStatsResponse>("/api/v1/performance/stats");
  },

  async resetPerformanceCache(): Promise<{ status: string; message: string }> {
    return request<{ status: string; message: string }>("/api/v1/performance/reset", {
      method: "POST",
    });
  },
};
