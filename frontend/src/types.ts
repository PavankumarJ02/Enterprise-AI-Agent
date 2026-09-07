/**
 * Strict TypeScript domain models and API contracts for Enterprise AI Agent.
 */

export type WorkspaceTab = "chat" | "search" | "sql" | "guardrails" | "telemetry";

export interface CitationItem {
  id?: string;
  doc_id?: string;
  title: string;
  source?: string;
  excerpt: string;
  score?: number;
  metadata?: Record<string, unknown>;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string;
  latency_ms?: number;
  model?: string;
  citations?: CitationItem[];
  intent?: string;
  isError?: boolean;
}

export interface RAGQueryRequest {
  query: string;
  top_k?: number;
  retrieval_strategy?: "hybrid" | "dense" | "sparse";
  system_prompt?: string;
}

export interface RAGQueryResponse {
  answer: string;
  query: string;
  sources: CitationItem[];
  model: string;
  latency_ms: number;
}

export interface HybridSearchRequest {
  query: string;
  top_k?: number;
  dense_weight?: number;
  sparse_weight?: number;
  fusion_method?: "rrf" | "weighted";
}

export interface SearchResultItem {
  chunk_id: string;
  document_id: string;
  title: string;
  text: string;
  score: number;
  dense_score?: number;
  sparse_score?: number;
  metadata?: Record<string, unknown>;
}

export interface HybridSearchResponse {
  query: string;
  results: SearchResultItem[];
  total_results: number;
  fusion_method: string;
  latency_ms?: number;
}

export interface SQLQueryRequest {
  query: string;
}

export interface SQLQueryResponse {
  columns: string[];
  rows: Array<Record<string, unknown>>;
  row_count: number;
  execution_time_ms: number;
  is_error?: boolean;
  error_message?: string;
}

export interface GuardrailsViolation {
  violation_type: string;
  severity: "low" | "medium" | "high" | "critical";
  message: string;
  action_taken: "block" | "sanitize" | "flag";
}

export interface GuardrailsValidationResponse {
  is_safe: boolean;
  sanitized_text: string;
  violations: GuardrailsViolation[];
  pii_detected: boolean;
  injection_detected: boolean;
  entities_redacted?: string[];
}

export interface CacheMetrics {
  hits: number;
  misses: number;
  evictions: number;
  size: number;
  hit_ratio: number;
  enabled: boolean;
}

export interface PerformanceStatsResponse {
  status: string;
  overall_hit_ratio_pct: number;
  total_requests: number;
  total_hits: number;
  total_misses: number;
  embedding_cache: CacheMetrics;
  retrieval_cache: CacheMetrics;
}

export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  environment: string;
  llm_healthy: boolean;
}
