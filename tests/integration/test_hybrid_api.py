"""Integration tests for hybrid search and sparse BM25 endpoints."""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from enterprise_agent.api.deps import (
    get_embeddings,
    get_hybrid_search_service,
    get_ingestion_service,
    get_sparse_store,
    get_vector_search_service,
    get_vector_store,
    reset_sparse_store,
    reset_vector_store,
)
from enterprise_agent.config.settings import Settings
from enterprise_agent.embeddings.factory import get_embeddings_service
from enterprise_agent.hybrid.service import HybridSearchService
from enterprise_agent.ingestion.service import IngestionService
from enterprise_agent.main import create_application
from enterprise_agent.sparse.bm25 import InMemoryBM25Store
from enterprise_agent.vectorstore.factory import create_vector_store
from enterprise_agent.vectorstore.service import VectorSearchService


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """Create test client with isolated in-memory stores and mock services."""
    settings = Settings(
        app_env="testing",
        llm_provider="mock",
        embedding_provider="mock",
        embedding_dimensions=16,
        qdrant_url=":memory:",
        qdrant_collection_name="test_hybrid_api_col",
        sparse_search_provider="bm25",
    )
    app = create_application(settings)

    # Isolated service instances
    ingestion_service = IngestionService()
    embeddings_service = get_embeddings_service(settings)
    vector_store = create_vector_store(settings)
    sparse_store = InMemoryBM25Store()

    vector_service = VectorSearchService(
        vector_store=vector_store,
        embeddings_service=embeddings_service,
    )
    hybrid_service = HybridSearchService(
        vector_service=vector_service,
        sparse_store=sparse_store,
        default_fusion_method="rrf",
    )

    app.dependency_overrides[get_ingestion_service] = lambda: ingestion_service
    app.dependency_overrides[get_embeddings] = lambda: embeddings_service
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_sparse_store] = lambda: sparse_store
    app.dependency_overrides[get_vector_search_service] = lambda: vector_service
    app.dependency_overrides[get_hybrid_search_service] = lambda: hybrid_service

    reset_vector_store()
    reset_sparse_store()

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


def test_sparse_search_api_exact_keyword(test_client: TestClient) -> None:
    """Verify POST /api/v1/search/sparse retrieves exact acronyms and product codes."""
    test_client.post(
        "/api/v1/documents/ingest/text?auto_index=true",
        json={
            "title": "Hardware Catalog",
            "content": "Workstation model SKU-PRO-9942 is approved for engineering teams.",
            "source": "hardware.md",
        },
    )

    response = test_client.post(
        "/api/v1/search/sparse",
        json={"query": "SKU-PRO-9942", "top_k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] >= 1
    assert "SKU-PRO-9942" in data["results"][0]["content"]
    assert data["results"][0]["score"] > 0.0


def test_hybrid_search_api_rrf(test_client: TestClient) -> None:
    """Verify POST /api/v1/search/hybrid executes RRF fusion across dense and sparse."""
    test_client.post(
        "/api/v1/documents/ingest/text?auto_index=true",
        json={
            "title": "Security Guidelines",
            "content": "Protocol ISO-27001 requires multi-factor authentication on all machines.",
            "source": "iso_security.txt",
            "metadata": {"department": "Security"},
        },
    )

    response = test_client.post(
        "/api/v1/search/hybrid",
        json={
            "query": "ISO-27001 multi-factor authentication",
            "top_k": 3,
            "fusion_method": "rrf",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] >= 1
    assert data["fusion_method"] == "rrf"

    top_item = data["results"][0]
    assert "combined_score" in top_item
    assert top_item["combined_score"] > 0
    assert "ISO-27001" in top_item["content"]


def test_hybrid_search_api_linear_and_filtering(test_client: TestClient) -> None:
    """Verify linear fusion method and metadata filtering on hybrid search."""
    test_client.post(
        "/api/v1/documents/ingest/text?auto_index=true",
        json={
            "title": "Legal Policy",
            "content": "Compliance audit rules for legal records.",
            "metadata": {"department": "Legal"},
        },
    )
    test_client.post(
        "/api/v1/documents/ingest/text?auto_index=true",
        json={
            "title": "Finance Policy",
            "content": "Compliance audit rules for expense reports.",
            "metadata": {"department": "Finance"},
        },
    )

    response = test_client.post(
        "/api/v1/search/hybrid",
        json={
            "query": "Compliance audit rules",
            "top_k": 5,
            "fusion_method": "linear",
            "filters": {"department": "Legal"},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["fusion_method"] == "linear"
    assert len(data["results"]) == 1
    assert data["results"][0]["metadata"]["department"] == "Legal"


def test_hybrid_search_validation_error(test_client: TestClient) -> None:
    """Invalid query returns 422 Unprocessable Entity."""
    response = test_client.post("/api/v1/search/hybrid", json={"query": ""})
    assert response.status_code == 422
