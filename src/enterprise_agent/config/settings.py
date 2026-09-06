"""Application configuration management using Pydantic Settings.

Adheres to 12-Factor App principles by extracting configuration from environment variables.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration object loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -------------------------------------------------------------------------
    # Core Application Configuration
    # -------------------------------------------------------------------------
    app_name: str = Field(
        default="Enterprise AI Knowledge & Decision Agent",
        description="Public display name for the service.",
    )
    app_env: Literal["development", "staging", "production", "testing"] = Field(
        default="development",
        description="Current deployment environment.",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application semantic version.",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode and verbose logs.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Standard log level filter.",
    )
    host: str = Field(
        default="0.0.0.0",
        description="HTTP server bind host.",
    )
    port: int = Field(
        default=8000,
        description="HTTP server bind port.",
    )

    # -------------------------------------------------------------------------
    # LLM Service Configuration
    # -------------------------------------------------------------------------
    llm_provider: Literal["mock", "openai", "groq", "ollama", "azure", "gemini"] = Field(
        default="mock",
        description="LLM provider implementation to instantiate.",
    )
    llm_model: str = Field(
        default="gemini-2.5-flash",
        description="LLM model identifier to invoke.",
    )
    llm_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key for the external LLM provider.",
    )
    gemini_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Google Gemini API key (defaults to llm_api_key if left blank).",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible base API URL.",
    )
    llm_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM generation.",
    )
    llm_max_tokens: int = Field(
        default=1024,
        gt=0,
        le=32768,
        description="Maximum tokens allowed in generation response.",
    )
    llm_request_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Timeout in seconds for outbound LLM API requests.",
    )
    llm_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts on transient network or rate-limit errors.",
    )

    # -------------------------------------------------------------------------
    # Embedding Service Configuration
    # -------------------------------------------------------------------------
    embedding_provider: Literal["mock", "gemini", "openai"] = Field(
        default="mock",
        description="Embedding provider implementation to instantiate.",
    )
    embedding_model: str = Field(
        default="text-embedding-004",
        description="Embedding model identifier (e.g. text-embedding-004).",
    )
    embedding_dimensions: int = Field(
        default=768,
        gt=0,
        le=4096,
        description="Dimensionality of generated dense embedding vectors.",
    )
    embedding_batch_size: int = Field(
        default=64,
        gt=0,
        le=512,
        description="Maximum number of text chunks to embed in a single batch request.",
    )

    # -------------------------------------------------------------------------
    # Qdrant Vector Store Configuration
    # -------------------------------------------------------------------------
    qdrant_url: str = Field(
        default="",
        description="Qdrant vector DB URL (leave empty or ':memory:' for local instance).",
    )
    qdrant_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key for authenticated Qdrant Cloud or protected cluster.",
    )
    qdrant_collection_name: str = Field(
        default="enterprise_knowledge",
        description="Default Qdrant collection name for document embeddings.",
    )
    qdrant_prefer_grpc: bool = Field(
        default=False,
        description="Whether to use gRPC protocol instead of REST for Qdrant operations.",
    )

    # -------------------------------------------------------------------------
    # Sparse Search (BM25 / Elasticsearch) & Hybrid Fusion Configuration
    # -------------------------------------------------------------------------
    sparse_search_provider: Literal["bm25", "elasticsearch"] = Field(
        default="bm25",
        description="Sparse search provider implementation ('bm25' or 'elasticsearch').",
    )
    elasticsearch_url: str = Field(
        default="http://localhost:9200",
        description="Elasticsearch cluster HTTP URL.",
    )
    elasticsearch_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key for authenticated Elasticsearch access.",
    )
    elasticsearch_index_name: str = Field(
        default="enterprise_knowledge",
        description="Elasticsearch index name for keyword retrieval.",
    )
    hybrid_fusion_method: Literal["rrf", "linear"] = Field(
        default="rrf",
        description="Rank fusion strategy: 'rrf' (Reciprocal Rank Fusion) or 'linear'.",
    )
    hybrid_dense_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relative weight assigned to dense vector retrieval in hybrid search.",
    )
    hybrid_sparse_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relative weight assigned to sparse BM25 retrieval in hybrid search.",
    )
    rrf_k: int = Field(
        default=60,
        gt=0,
        le=500,
        description="Smoothing constant k for Reciprocal Rank Fusion.",
    )

    # -------------------------------------------------------------------------
    # Cross-Encoder Reranker Configuration
    # -------------------------------------------------------------------------
    reranker_provider: Literal["mock", "flashrank"] = Field(
        default="flashrank",
        description="Cross-encoder reranker provider ('flashrank' or 'mock').",
    )
    reranker_model: str = Field(
        default="ms-marco-TinyBERT-L-2-v2",
        description="Cross-encoder model identifier.",
    )
    reranker_batch_size: int = Field(
        default=32,
        gt=0,
        le=256,
        description="Batch size for cross-encoder inference.",
    )
    reranker_top_k: int = Field(
        default=5,
        gt=0,
        le=100,
        description="Default number of reranked results to return.",
    )
    reranker_candidate_k: int = Field(
        default=25,
        gt=0,
        le=200,
        description="Default number of initial candidates to retrieve before reranking.",
    )

    # -------------------------------------------------------------------------
    # Query Transformation Configuration (HyDE, Multi-Query, Step-Back)
    # -------------------------------------------------------------------------
    hyde_num_hypotheses: int = Field(
        default=1,
        gt=0,
        le=5,
        description="Number of hypothetical documents to generate for HyDE retrieval.",
    )
    multi_query_num_variations: int = Field(
        default=3,
        gt=0,
        le=10,
        description="Number of query perspectives/variations to generate for Multi-Query.",
    )
    step_back_enabled: bool = Field(
        default=True,
        description="Whether Step-Back prompting is enabled for abstract conceptual retrieval.",
    )

    # -------------------------------------------------------------------------
    # Tool-Augmented Agent Engine Configuration
    # -------------------------------------------------------------------------
    agent_max_iterations: int = Field(
        default=6,
        gt=0,
        le=20,
        description="Maximum ReAct reasoning steps before forced termination.",
    )
    agent_timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        le=300.0,
        description="Maximum allowed execution duration for agent task.",
    )
    agent_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for deterministic tool-calling reasoning.",
    )

    # -------------------------------------------------------------------------
    # SQL Database Tool Configuration
    # -------------------------------------------------------------------------
    sql_db_path: str = Field(
        default="data/enterprise.db",
        description="Path to SQLite enterprise database file.",
    )
    sql_max_rows: int = Field(
        default=50,
        gt=0,
        le=500,
        description="Maximum rows returned per SQL query to prevent context overflow.",
    )
    sql_query_timeout_seconds: float = Field(
        default=5.0,
        gt=0.0,
        le=60.0,
        description="Execution timeout in seconds for database queries.",
    )

    # -------------------------------------------------------------------------
    # Semantic Query Router Configuration
    # -------------------------------------------------------------------------
    router_strategy: str = Field(
        default="cascade",
        description="Strategy for routing queries (cascade, heuristic, semantic, llm).",
    )
    router_semantic_threshold: float = Field(
        default=0.78,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity threshold for semantic embedding routing.",
    )
    router_llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Temperature for zero-shot LLM intent classification.",
    )

    # -------------------------------------------------------------------------
    # Enterprise Guardrails & Prompt-Injection Defense Configuration
    # -------------------------------------------------------------------------
    guardrails_enabled: bool = Field(
        default=True,
        description="Whether input and output guardrails verification is globally enabled.",
    )
    guardrails_block_injections: bool = Field(
        default=True,
        description="Whether to block detected prompt injections and jailbreak attacks.",
    )
    guardrails_injection_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for blocking prompt injection and jailbreaks.",
    )
    guardrails_redact_pii: bool = Field(
        default=True,
        description="Whether to automatically redact detected PII entities.",
    )
    guardrails_canary_detection: bool = Field(
        default=True,
        description="Whether to inspect model outputs for canary token prompt leakage.",
    )
    guardrails_canary_secret: str = Field(
        default="CANARY_ENT_SEC_TOKEN_89213",
        description="System prompt canary token used to detect prompt exfiltration.",
    )

    # -------------------------------------------------------------------------
    # RAG Evaluation & Benchmark Configuration
    # -------------------------------------------------------------------------
    eval_faithfulness_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum threshold for answer faithfulness to retrieved context.",
    )
    eval_answer_relevance_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum threshold for semantic answer relevance to user query.",
    )
    eval_context_precision_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum threshold for retrieved context precision.",
    )
    eval_context_recall_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum threshold for ground truth context recall.",
    )

    # -------------------------------------------------------------------------
    # Experiment Tracking & Benchmarking Configuration
    # -------------------------------------------------------------------------
    experiments_db_path: str = Field(
        default="data/experiments.db",
        description="Path to SQLite database for persisting benchmark experiment runs.",
    )
    experiments_default_name: str = Field(
        default="rag_optimization",
        description="Default experiment name group for logging benchmark runs.",
    )

    @property
    def is_production(self) -> bool:
        """Helper to check if running in production."""
        return self.app_env == "production"

    @property
    def is_testing(self) -> bool:
        """Helper to check if running under test suite."""
        return self.app_env == "testing"

    @property
    def effective_gemini_api_key(self) -> str:
        """Resolve Gemini API key from gemini_api_key or fallback to llm_api_key."""
        key = self.gemini_api_key.get_secret_value()
        if key:
            return key
        return self.llm_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    """Provide a cached singleton instance of Settings.

    Can be overridden in tests via FastAPI's `app.dependency_overrides`.
    """
    return Settings()
