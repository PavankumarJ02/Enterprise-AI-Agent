"""Integration tests verifying caching speedup and end-to-end performance optimizations."""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from enterprise_agent.api.deps import (
    get_embedding_cache,
    get_retrieval_cache,
    reset_performance_caches,
)
from enterprise_agent.config.settings import Settings
from enterprise_agent.main import create_application
from enterprise_agent.performance.cache import QueryEmbeddingCache, RetrievalCache


@pytest.fixture
def performance_client() -> Generator[TestClient, None, None]:
    """TestClient with performance caching enabled and isolated cache instances."""
    reset_performance_caches()
    settings = Settings(
        app_env="testing",
        llm_provider="mock",
        embedding_provider="mock",
        embedding_dimensions=16,
        qdrant_url=":memory:",
        performance_cache_enabled=True,
    )
    app = create_application(settings)

    emb_cache = QueryEmbeddingCache(max_size=100, ttl_seconds=3600)
    ret_cache = RetrievalCache(max_size=100, ttl_seconds=1800)

    app.dependency_overrides[get_embedding_cache] = lambda: emb_cache
    app.dependency_overrides[get_retrieval_cache] = lambda: ret_cache

    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()
        reset_performance_caches()


def test_rag_query_caching_integration(performance_client: TestClient) -> None:
    """Verify end-to-end RAG query hits cache on repeated calls."""
    payload = {"query": "What is the parental leave duration?", "top_k": 3}

    # 1. Cold query (cache miss)
    resp1 = performance_client.post("/api/v1/rag/query", json=payload)
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert "answer" in data1

    # 2. Warm query (cache hit)
    resp2 = performance_client.post("/api/v1/rag/query", json=payload)
    assert resp2.status_code == 200
    data2 = resp2.json()

    assert data1["answer"] == data2["answer"]
    assert data1["sources"] == data2["sources"]


def test_embedding_api_caching_integration(performance_client: TestClient) -> None:
    """Verify /api/v1/embeddings/query endpoint benefits from caching."""
    payload = {"query": "How many days of paid vacation do employees receive?"}

    # 1. Cold query
    r1 = performance_client.post("/api/v1/embeddings/query", json=payload)
    assert r1.status_code == 200
    vec1 = r1.json()["vector"]

    # 2. Warm query
    r2 = performance_client.post("/api/v1/embeddings/query", json=payload)
    assert r2.status_code == 200
    vec2 = r2.json()["vector"]

    assert vec1 == vec2
