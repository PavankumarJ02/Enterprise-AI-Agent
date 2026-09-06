"""FastAPI dependency injection providers."""

from collections.abc import Generator

from fastapi import Depends

from enterprise_agent.agent.service import AgentService
from enterprise_agent.config.settings import Settings, get_settings
from enterprise_agent.embeddings.factory import get_embeddings_service
from enterprise_agent.embeddings.service import EmbeddingsService
from enterprise_agent.evaluation.service import RAGEvaluationService
from enterprise_agent.experiments.service import ExperimentTrackingService
from enterprise_agent.guardrails.service import GuardrailsService
from enterprise_agent.hybrid.service import HybridSearchService
from enterprise_agent.ingestion.service import IngestionService
from enterprise_agent.llm.base import LLMProvider
from enterprise_agent.llm.factory import get_llm_provider
from enterprise_agent.middleware.rate_limit import SlidingWindowRateLimiter
from enterprise_agent.observability.service import ObservabilityService
from enterprise_agent.observability.tracer import Tracer
from enterprise_agent.rag.service import RAGService
from enterprise_agent.reranking.base import Reranker
from enterprise_agent.reranking.factory import create_reranker
from enterprise_agent.reranking.service import TwoStageRetrievalService
from enterprise_agent.router.heuristics import HeuristicRouter
from enterprise_agent.router.llm import LLMRouter
from enterprise_agent.router.semantic import SemanticEmbeddingRouter
from enterprise_agent.router.service import QueryRouterService
from enterprise_agent.sparse.base import SparseStore
from enterprise_agent.sparse.factory import create_sparse_store
from enterprise_agent.sql.connection import SQLiteManager
from enterprise_agent.sql.service import SQLDatabaseService
from enterprise_agent.tools.calculator import CalculatorTool
from enterprise_agent.tools.clock import CurrentTimeTool
from enterprise_agent.tools.registry import ToolRegistry
from enterprise_agent.tools.search import KnowledgeSearchTool
from enterprise_agent.tools.sql import SQLQueryTool, SQLSchemaTool
from enterprise_agent.transformation.hyde import HyDETransformer
from enterprise_agent.transformation.multi_query import MultiQueryTransformer
from enterprise_agent.transformation.service import QueryTransformationService
from enterprise_agent.transformation.step_back import StepBackTransformer
from enterprise_agent.vectorstore.base import VectorStore
from enterprise_agent.vectorstore.factory import create_vector_store
from enterprise_agent.vectorstore.service import VectorSearchService

# Module-level singleton instance of IngestionService
_ingestion_service_instance = IngestionService()

# Module-level singleton instance of VectorStore (preserves in-memory state across requests)
_vector_store_instance: VectorStore | None = None

# Module-level singleton instance of SparseStore
_sparse_store_instance: SparseStore | None = None

# Module-level singleton instance of Reranker
_reranker_instance: Reranker | None = None

# Module-level singleton instance of SQLDatabaseService
_sql_service_instance: SQLDatabaseService | None = None

# Module-level singleton instance of GuardrailsService
_guardrails_service_instance: GuardrailsService | None = None

# Module-level singleton instance of RAGEvaluationService
_evaluation_service_instance: RAGEvaluationService | None = None

# Module-level singleton instance of ExperimentTrackingService
_experiment_service_instance: ExperimentTrackingService | None = None

# Module-level singleton instance of ObservabilityService
_observability_service_instance: ObservabilityService | None = None

# Module-level singleton instance of SlidingWindowRateLimiter
_rate_limiter_instance: SlidingWindowRateLimiter | None = None


def reset_rate_limiter() -> None:
    """Reset singleton SlidingWindowRateLimiter for test isolation."""
    global _rate_limiter_instance
    _rate_limiter_instance = None


def reset_observability_service() -> None:
    """Reset singleton ObservabilityService for test isolation."""
    global _observability_service_instance
    _observability_service_instance = None


def reset_sql_service() -> None:
    """Reset singleton SQLDatabaseService for test isolation."""
    global _sql_service_instance
    _sql_service_instance = None


def reset_guardrails_service() -> None:
    """Reset singleton GuardrailsService for test isolation."""
    global _guardrails_service_instance
    _guardrails_service_instance = None


def reset_evaluation_service() -> None:
    """Reset singleton RAGEvaluationService for test isolation."""
    global _evaluation_service_instance
    _evaluation_service_instance = None


def reset_experiment_service() -> None:
    """Reset singleton ExperimentTrackingService for test isolation."""
    global _experiment_service_instance
    _experiment_service_instance = None


def get_app_settings() -> Settings:
    """Dependency provider for application settings."""
    return get_settings()


def get_observability_service(
    settings: Settings = Depends(get_app_settings),
) -> ObservabilityService:
    """Dependency provider yielding singleton ObservabilityService."""
    global _observability_service_instance
    if _observability_service_instance is None:
        _observability_service_instance = ObservabilityService(settings=settings)
    return _observability_service_instance


def get_tracer(
    observability_service: ObservabilityService = Depends(get_observability_service),
) -> Tracer:
    """Dependency provider yielding active Tracer engine."""
    return observability_service.tracer


def get_rate_limiter(
    settings: Settings = Depends(get_app_settings),
) -> SlidingWindowRateLimiter:
    """Dependency provider yielding singleton SlidingWindowRateLimiter."""
    global _rate_limiter_instance
    if _rate_limiter_instance is None:
        _rate_limiter_instance = SlidingWindowRateLimiter(
            requests_per_minute=settings.rate_limit_requests_per_minute,
            burst_limit=settings.rate_limit_burst_limit,
        )
    return _rate_limiter_instance


def get_llm(
    settings: Settings = Depends(get_app_settings),
) -> Generator[LLMProvider, None, None]:
    """Dependency provider yielding the configured LLM provider instance."""
    provider = get_llm_provider(settings)
    yield provider


def get_ingestion_service() -> IngestionService:
    """Dependency provider for the IngestionService."""
    return _ingestion_service_instance


def get_embeddings(
    settings: Settings = Depends(get_app_settings),
) -> EmbeddingsService:
    """Dependency provider yielding the configured EmbeddingsService."""
    return get_embeddings_service(settings)


def get_vector_store(
    settings: Settings = Depends(get_app_settings),
) -> VectorStore:
    """Dependency provider yielding the VectorStore singleton instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = create_vector_store(settings)
    return _vector_store_instance


def reset_vector_store() -> None:
    """Helper for test suites to reset vector store state."""
    global _vector_store_instance
    _vector_store_instance = None


def get_sparse_store(
    settings: Settings = Depends(get_app_settings),
) -> SparseStore:
    """Dependency provider yielding the SparseStore singleton instance."""
    global _sparse_store_instance
    if _sparse_store_instance is None:
        _sparse_store_instance = create_sparse_store(settings)
    return _sparse_store_instance


def reset_sparse_store() -> None:
    """Helper for test suites to reset sparse store state."""
    global _sparse_store_instance
    _sparse_store_instance = None


def get_vector_search_service(
    vector_store: VectorStore = Depends(get_vector_store),
    embeddings_service: EmbeddingsService = Depends(get_embeddings),
) -> VectorSearchService:
    """Dependency provider yielding the VectorSearchService."""
    return VectorSearchService(
        vector_store=vector_store,
        embeddings_service=embeddings_service,
    )


def get_hybrid_search_service(
    vector_service: VectorSearchService = Depends(get_vector_search_service),
    sparse_store: SparseStore = Depends(get_sparse_store),
    settings: Settings = Depends(get_app_settings),
) -> HybridSearchService:
    """Dependency provider yielding the HybridSearchService."""
    return HybridSearchService(
        vector_service=vector_service,
        sparse_store=sparse_store,
        default_fusion_method=settings.hybrid_fusion_method,
        default_dense_weight=settings.hybrid_dense_weight,
        default_sparse_weight=settings.hybrid_sparse_weight,
        default_rrf_k=settings.rrf_k,
    )


def get_rag_service(
    vector_service: VectorSearchService = Depends(get_vector_search_service),
    llm: LLMProvider = Depends(get_llm),
    tracer: Tracer = Depends(get_tracer),
) -> RAGService:
    """Dependency provider yielding the RAGService."""
    return RAGService(
        vector_service=vector_service,
        llm=llm,
        tracer=tracer,
    )


def get_reranker(
    settings: Settings = Depends(get_app_settings),
) -> Reranker:
    """Dependency provider yielding the Reranker singleton instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = create_reranker(settings)
    return _reranker_instance


def reset_reranker() -> None:
    """Helper for test suites to reset reranker state."""
    global _reranker_instance
    _reranker_instance = None


def get_two_stage_retrieval_service(
    hybrid_service: HybridSearchService = Depends(get_hybrid_search_service),
    vector_service: VectorSearchService = Depends(get_vector_search_service),
    reranker: Reranker = Depends(get_reranker),
) -> TwoStageRetrievalService:
    """Dependency provider yielding the TwoStageRetrievalService."""
    return TwoStageRetrievalService(
        hybrid_service=hybrid_service,
        vector_service=vector_service,
        reranker=reranker,
    )


def get_hyde_transformer(
    llm: LLMProvider = Depends(get_llm),
) -> HyDETransformer:
    """Dependency provider yielding HyDETransformer."""
    return HyDETransformer(llm=llm)


def get_multi_query_transformer(
    llm: LLMProvider = Depends(get_llm),
) -> MultiQueryTransformer:
    """Dependency provider yielding MultiQueryTransformer."""
    return MultiQueryTransformer(llm=llm)


def get_step_back_transformer(
    llm: LLMProvider = Depends(get_llm),
) -> StepBackTransformer:
    """Dependency provider yielding StepBackTransformer."""
    return StepBackTransformer(llm=llm)


def get_query_transformation_service(
    hyde_transformer: HyDETransformer = Depends(get_hyde_transformer),
    multi_query_transformer: MultiQueryTransformer = Depends(get_multi_query_transformer),
    step_back_transformer: StepBackTransformer = Depends(get_step_back_transformer),
    two_stage_service: TwoStageRetrievalService = Depends(get_two_stage_retrieval_service),
    hybrid_service: HybridSearchService = Depends(get_hybrid_search_service),
    vector_service: VectorSearchService = Depends(get_vector_search_service),
) -> QueryTransformationService:
    """Dependency provider yielding QueryTransformationService."""
    return QueryTransformationService(
        hyde_transformer=hyde_transformer,
        multi_query_transformer=multi_query_transformer,
        step_back_transformer=step_back_transformer,
        two_stage_service=two_stage_service,
        hybrid_service=hybrid_service,
        vector_service=vector_service,
    )


def get_sql_service(
    settings: Settings = Depends(get_app_settings),
) -> SQLDatabaseService:
    """Dependency provider yielding singleton SQLDatabaseService."""
    global _sql_service_instance
    if _sql_service_instance is None:
        manager = SQLiteManager(db_path=settings.sql_db_path)
        _sql_service_instance = SQLDatabaseService(
            manager=manager,
            default_max_rows=settings.sql_max_rows,
            query_timeout_seconds=settings.sql_query_timeout_seconds,
        )
    return _sql_service_instance


def get_tool_registry(
    two_stage_service: TwoStageRetrievalService = Depends(get_two_stage_retrieval_service),
    sql_service: SQLDatabaseService = Depends(get_sql_service),
) -> ToolRegistry:
    """Dependency provider yielding ToolRegistry with default enterprise tools."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(CurrentTimeTool())
    registry.register(KnowledgeSearchTool(two_stage_service=two_stage_service))
    registry.register(SQLSchemaTool(sql_service=sql_service))
    registry.register(SQLQueryTool(sql_service=sql_service))
    return registry


def get_agent_service(
    llm: LLMProvider = Depends(get_llm),
    tool_registry: ToolRegistry = Depends(get_tool_registry),
    settings: Settings = Depends(get_app_settings),
) -> AgentService:
    """Dependency provider yielding the AgentService."""
    return AgentService(
        llm=llm,
        tool_registry=tool_registry,
        default_max_iterations=settings.agent_max_iterations,
        default_timeout_seconds=settings.agent_timeout_seconds,
        default_temperature=settings.agent_temperature,
    )


def get_query_router_service(
    llm: LLMProvider = Depends(get_llm),
    embeddings_service: EmbeddingsService = Depends(get_embeddings),
    rag_service: RAGService = Depends(get_rag_service),
    sql_service: SQLDatabaseService = Depends(get_sql_service),
    agent_service: AgentService = Depends(get_agent_service),
    settings: Settings = Depends(get_app_settings),
) -> QueryRouterService:
    """Dependency provider yielding configured QueryRouterService."""
    heuristic_router = HeuristicRouter()
    semantic_router = SemanticEmbeddingRouter(
        embedding_provider=embeddings_service.provider,
        threshold=settings.router_semantic_threshold,
    )
    llm_router = LLMRouter(
        llm=llm,
        temperature=settings.router_llm_temperature,
    )
    return QueryRouterService(
        heuristic_router=heuristic_router,
        semantic_router=semantic_router,
        llm_router=llm_router,
        llm=llm,
        rag_service=rag_service,
        sql_service=sql_service,
        agent_service=agent_service,
        strategy=settings.router_strategy,
    )


def get_guardrails_service(
    settings: Settings = Depends(get_app_settings),
) -> GuardrailsService:
    """Dependency provider yielding singleton GuardrailsService."""
    global _guardrails_service_instance
    if _guardrails_service_instance is None:
        _guardrails_service_instance = GuardrailsService(settings=settings)
    return _guardrails_service_instance


def get_evaluation_service(
    embeddings_service: EmbeddingsService = Depends(get_embeddings),
    rag_service: RAGService = Depends(get_rag_service),
    settings: Settings = Depends(get_app_settings),
) -> RAGEvaluationService:
    """Dependency provider yielding singleton RAGEvaluationService."""
    global _evaluation_service_instance
    if _evaluation_service_instance is None:
        _evaluation_service_instance = RAGEvaluationService(
            embedding_provider=embeddings_service.provider,
            rag_service=rag_service,
            settings=settings,
        )
    return _evaluation_service_instance


def get_experiment_service(
    evaluation_service: RAGEvaluationService = Depends(get_evaluation_service),
    settings: Settings = Depends(get_app_settings),
) -> ExperimentTrackingService:
    """Dependency provider yielding singleton ExperimentTrackingService."""
    global _experiment_service_instance
    if _experiment_service_instance is None:
        _experiment_service_instance = ExperimentTrackingService(
            evaluation_service=evaluation_service,
            settings=settings,
        )
    return _experiment_service_instance
