"""Integration tests for Cross-Encoder Rerank API endpoints (/search/rerank)."""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from enterprise_agent.api.deps import (
    get_embeddings,
    get_hybrid_search_service,
    get_ingestion_service,
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
from enterprise_agent.main import create_application
from enterprise_agent.reranking.mock import MockReranker
from enterprise_agent.reranking.service import TwoStageRetrievalService
from enterprise_agent.sparse.bm25 import InMemoryBM25Store
from enterprise_agent.vectorstore.factory import create_vector_store
from enterprise_agent.vectorstore.service import VectorSearchService


@pytest.fixture
def rerank_test_client() -> Generator[TestClient, None, None]:
    """Create test client configured with isolated in-memory stores and mock reranker."""
    settings = Settings(
        app_env="testing",
        llm_provider="mock",
        embedding_provider="mock",
        embedding_dimensions=16,
        qdrant_url=":memory:",
        qdrant_collection_name="test_rerank_api_col",
        sparse_search_provider="bm25",
        reranker_provider="mock",
    )
    app = create_application(settings)

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

    app.dependency_overrides[get_ingestion_service] = lambda: ingestion_service
    app.dependency_overrides[get_embeddings] = lambda: embeddings_service
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_sparse_store] = lambda: sparse_store
    app.dependency_overrides[get_vector_search_service] = lambda: vector_service
    app.dependency_overrides[get_hybrid_search_service] = lambda: hybrid_service
    app.dependency_overrides[get_reranker] = lambda: mock_reranker
    app.dependency_overrides[get_two_stage_retrieval_service] = lambda: two_stage_service

    reset_vector_store()
    reset_sparse_store()
    reset_reranker()

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


def test_two_stage_rerank_pipeline(rerank_test_client: TestClient) -> None:
    """Verify POST /api/v1/search/rerank retrieves candidates and reorders via cross-encoder."""
    # 1. Ingest distinct knowledge documents
    doc1 = {
        "title": "Travel Guidelines",
        "content": "Reimbursed up to $75 daily for meals during company business trips.",
        "source": "travel_guidelines.md",
    }
    doc2 = {
        "title": "Server Maintenance",
        "content": "Database backups are replicated nightly across multiple availability zones.",
        "source": "server_ops.md",
    }
    rerank_test_client.post("/api/v1/documents/ingest/text?auto_index=true", json=doc1)
    rerank_test_client.post("/api/v1/documents/ingest/text?auto_index=true", json=doc2)

    # 2. Execute two-stage reranking query
    response = rerank_test_client.post(
        "/api/v1/search/rerank",
        json={
            "query": "daily meal allowance travel reimbursement",
            "top_k": 5,
            "candidate_k": 10,
            "retrieval_strategy": "hybrid",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "daily meal allowance travel reimbursement"
    assert data["total_candidates"] >= 2
    assert data["total_results"] >= 1
    assert data["retrieval_strategy"] == "hybrid"
    assert "model" in data

    top_chunk = data["results"][0]
    assert "Reimbursed up to $75 daily" in top_chunk["content"]
    assert top_chunk["final_rank"] == 1
    assert "rerank_score" in top_chunk
    assert "initial_rank" in top_chunk
    assert "initial_score" in top_chunk


def test_two_stage_rerank_semantic_and_sparse_strategies(
    rerank_test_client: TestClient,
) -> None:
    """Verify two-stage rerank supports alternative recall strategies (semantic and sparse)."""
    rerank_test_client.post(
        "/api/v1/documents/ingest/text?auto_index=true",
        json={
            "title": "Device Security",
            "content": "Hardware token YubiKey-5C is required for remote zero-trust VPN login.",
            "source": "security.txt",
        },
    )

    # Test semantic strategy
    res_sem = rerank_test_client.post(
        "/api/v1/search/rerank",
        json={
            "query": "remote VPN token",
            "retrieval_strategy": "semantic",
            "top_k": 2,
        },
    )
    assert res_sem.status_code == 200
    assert res_sem.json()["retrieval_strategy"] == "semantic"
    assert len(res_sem.json()["results"]) >= 1

    # Test sparse strategy
    res_sparse = rerank_test_client.post(
        "/api/v1/search/rerank",
        json={
            "query": "YubiKey-5C",
            "retrieval_strategy": "sparse",
            "top_k": 2,
        },
    )
    assert res_sparse.status_code == 200
    assert res_sparse.json()["retrieval_strategy"] == "sparse"
    assert len(res_sparse.json()["results"]) >= 1
    assert "YubiKey-5C" in res_sparse.json()["results"][0]["content"]


def test_direct_rerank_api(rerank_test_client: TestClient) -> None:
    """Verify POST /api/v1/search/rerank/direct rescores explicitly provided items."""
    candidate_items = [
        {
            "chunk_id": "item-1",
            "document_id": "doc-1",
            "content": "Banana splits and sundaes are delicious desserts.",
            "score": 0.85,
            "chunk_index": 0,
            "metadata": {},
        },
        {
            "chunk_id": "item-2",
            "document_id": "doc-2",
            "content": "The company expense policy covers travel meal receipts up to $75.",
            "score": 0.35,
            "chunk_index": 1,
            "metadata": {},
        },
    ]

    response = rerank_test_client.post(
        "/api/v1/search/rerank/direct",
        json={
            "query": "travel expense meal policy",
            "items": candidate_items,
            "top_k": 2,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_candidates"] == 2
    assert data["total_results"] == 2

    # item-2 should be reranked to rank 1 because of strong query keyword overlap
    top_result = data["results"][0]
    assert top_result["chunk_id"] == "item-2"
    assert top_result["final_rank"] == 1
    assert top_result["initial_rank"] == 2
    assert top_result["initial_score"] == 0.35


def test_rerank_empty_store_returns_zero(rerank_test_client: TestClient) -> None:
    """Verify rerank against an empty store safely returns zero candidates and results."""
    response = rerank_test_client.post(
        "/api/v1/search/rerank",
        json={"query": "non-existent document", "top_k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_candidates"] == 0
    assert data["total_results"] == 0
    assert data["results"] == []


def test_rerank_validation_errors(rerank_test_client: TestClient) -> None:
    """Verify Pydantic input validation on invalid rerank payloads."""
    # Empty query
    res_empty_q = rerank_test_client.post(
        "/api/v1/search/rerank",
        json={"query": "", "top_k": 5},
    )
    assert res_empty_q.status_code == 422

    # Negative top_k
    res_neg_k = rerank_test_client.post(
        "/api/v1/search/rerank",
        json={"query": "valid query", "top_k": -1},
    )
    assert res_neg_k.status_code == 422

    # Direct rerank with empty items list
    res_empty_items = rerank_test_client.post(
        "/api/v1/search/rerank/direct",
        json={"query": "valid query", "items": []},
    )
    assert res_empty_items.status_code == 422
