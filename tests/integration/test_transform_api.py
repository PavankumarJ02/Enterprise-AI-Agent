"""Integration tests for Query Transformation API endpoints."""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from enterprise_agent.api.deps import (
    get_embeddings,
    get_hybrid_search_service,
    get_ingestion_service,
    get_llm,
    get_query_transformation_service,
    get_reranker,
    get_sparse_store,
    get_two_stage_retrieval_service,
    get_vector_search_service,
    get_vector_store,
    reset_reranker,
    reset_sparse_store,
    reset_vector_store,
)
from enterprise_agent.config.settings import Settings
from enterprise_agent.embeddings.factory import get_embeddings_service
from enterprise_agent.hybrid.service import HybridSearchService
from enterprise_agent.ingestion.service import IngestionService
from enterprise_agent.llm.mock_client import MockLLMProvider
from enterprise_agent.main import create_application
from enterprise_agent.reranking.mock import MockReranker
from enterprise_agent.reranking.service import TwoStageRetrievalService
from enterprise_agent.sparse.bm25 import InMemoryBM25Store
from enterprise_agent.transformation.hyde import HyDETransformer
from enterprise_agent.transformation.multi_query import MultiQueryTransformer
from enterprise_agent.transformation.service import QueryTransformationService
from enterprise_agent.transformation.step_back import StepBackTransformer
from enterprise_agent.vectorstore.factory import create_vector_store
from enterprise_agent.vectorstore.service import VectorSearchService


@pytest.fixture
def transform_client() -> Generator[TestClient, None, None]:
    """Test client configured with mock LLM and isolated storage."""
    settings = Settings(
        app_env="testing",
        llm_provider="mock",
        embedding_provider="mock",
        embedding_dimensions=16,
        qdrant_url=":memory:",
        qdrant_collection_name="test_transform_col",
        sparse_search_provider="bm25",
        reranker_provider="mock",
    )
    app = create_application(settings)

    mock_llm = MockLLMProvider()
    ingestion_service = IngestionService()
    embeddings_service = get_embeddings_service(settings)
    vector_store = create_vector_store(settings)
    sparse_store = InMemoryBM25Store()
    mock_reranker = MockReranker()

    vector_service = VectorSearchService(
        vector_store=vector_store,
        embeddings_service=embeddings_service,
    )
    hybrid_service = HybridSearchService(
        vector_service=vector_service,
        sparse_store=sparse_store,
        default_fusion_method="rrf",
    )
    two_stage_service = TwoStageRetrievalService(
        hybrid_service=hybrid_service,
        vector_service=vector_service,
        reranker=mock_reranker,
    )

    hyde_transformer = HyDETransformer(llm=mock_llm)
    multi_query_transformer = MultiQueryTransformer(llm=mock_llm)
    step_back_transformer = StepBackTransformer(llm=mock_llm)

    transform_service = QueryTransformationService(
        hyde_transformer=hyde_transformer,
        multi_query_transformer=multi_query_transformer,
        step_back_transformer=step_back_transformer,
        two_stage_service=two_stage_service,
        hybrid_service=hybrid_service,
        vector_service=vector_service,
    )

    app.dependency_overrides[get_llm] = lambda: mock_llm
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion_service
    app.dependency_overrides[get_embeddings] = lambda: embeddings_service
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_sparse_store] = lambda: sparse_store
    app.dependency_overrides[get_vector_search_service] = lambda: vector_service
    app.dependency_overrides[get_hybrid_search_service] = lambda: hybrid_service
    app.dependency_overrides[get_reranker] = lambda: mock_reranker
    app.dependency_overrides[get_two_stage_retrieval_service] = lambda: two_stage_service
    app.dependency_overrides[get_query_transformation_service] = lambda: transform_service

    reset_vector_store()
    reset_sparse_store()
    reset_reranker()

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


def test_hyde_endpoint(transform_client: TestClient) -> None:
    """Verify POST /api/v1/transform/hyde generates synthetic document passages."""
    response = transform_client.post(
        "/api/v1/transform/hyde",
        json={"query": "What is the remote work policy?", "num_hypotheses": 1},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["original_query"] == "What is the remote work policy?"
    assert len(data["hypothetical_documents"]) == 1
    assert data["latency_ms"] >= 0.0


def test_multi_query_endpoint(transform_client: TestClient) -> None:
    """Verify POST /api/v1/transform/multi-query returns decomposed perspectives."""
    response = transform_client.post(
        "/api/v1/transform/multi-query",
        json={"query": "Kubernetes pod crashloopbackoff debugging", "num_variations": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["original_query"] == "Kubernetes pod crashloopbackoff debugging"
    assert len(data["variations"]) >= 1


def test_step_back_endpoint(transform_client: TestClient) -> None:
    """Verify POST /api/v1/transform/step-back derives high-level questions."""
    response = transform_client.post(
        "/api/v1/transform/step-back",
        json={"query": "Why did PostgreSQL table lock timeout on partition 3?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["original_query"] == "Why did PostgreSQL table lock timeout on partition 3?"
    assert "step_back_query" in data


def test_transformed_search_multi_query(transform_client: TestClient) -> None:
    """Verify POST /api/v1/transform/search with multi_query executes expansion & retrieval."""
    transform_client.post(
        "/api/v1/documents/ingest/text?auto_index=true",
        json={
            "title": "Expense Policy",
            "content": "Business expenses for client dinners are covered up to $100 per person.",
            "source": "expense_policy.pdf",
        },
    )

    response = transform_client.post(
        "/api/v1/transform/search",
        json={
            "query": "client dinner expense limits",
            "strategy": "multi_query",
            "top_k": 3,
            "enable_reranking": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"] == "multi_query"
    assert len(data["transformed_queries"]) >= 1
    assert data["total_results"] >= 1
    assert "client dinners are covered" in data["results"][0]["content"]


def test_transformed_search_hyde_and_step_back(transform_client: TestClient) -> None:
    """Verify POST /api/v1/transform/search with hyde and step_back strategies."""
    transform_client.post(
        "/api/v1/documents/ingest/text?auto_index=true",
        json={
            "title": "Database Architecture",
            "content": "WAL logging guarantees atomicity and durability in ACID engines.",
            "source": "database_internals.md",
        },
    )

    # HyDE search
    res_hyde = transform_client.post(
        "/api/v1/transform/search",
        json={"query": "How does WAL provide durability?", "strategy": "hyde", "top_k": 2},
    )
    assert res_hyde.status_code == 200
    assert res_hyde.json()["strategy"] == "hyde"
    assert len(res_hyde.json()["results"]) >= 1

    # Step-Back search
    res_sb = transform_client.post(
        "/api/v1/transform/search",
        json={"query": "WAL corruption error 404", "strategy": "step_back", "top_k": 2},
    )
    assert res_sb.status_code == 200
    assert res_sb.json()["strategy"] == "step_back"
    assert len(res_sb.json()["results"]) >= 1


def test_transform_validation_errors(transform_client: TestClient) -> None:
    """Verify input validation on empty query strings."""
    res = transform_client.post(
        "/api/v1/transform/hyde",
        json={"query": ""},
    )
    assert res.status_code == 422
