"""Unit tests for TTLCache, QueryEmbeddingCache, RetrievalCache, and concurrent tool dispatch."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from qdrant_client import AsyncQdrantClient

from enterprise_agent.config.settings import Settings
from enterprise_agent.embeddings.mock import MockEmbeddingProvider
from enterprise_agent.embeddings.service import EmbeddingsService
from enterprise_agent.hybrid.service import HybridSearchService
from enterprise_agent.performance.cache import QueryEmbeddingCache, RetrievalCache, TTLCache
from enterprise_agent.reranking.factory import create_reranker
from enterprise_agent.reranking.service import TwoStageRetrievalService
from enterprise_agent.schemas.search import RerankRequest
from enterprise_agent.sparse.bm25 import InMemoryBM25Store
from enterprise_agent.tools.calculator import CalculatorTool
from enterprise_agent.tools.clock import CurrentTimeTool
from enterprise_agent.tools.registry import ToolRegistry
from enterprise_agent.vectorstore.qdrant import QdrantVectorStore
from enterprise_agent.vectorstore.service import VectorSearchService


def test_ttl_cache_basic_set_get() -> None:
    """Verify basic set, get, hit count, and miss count behaviors."""
    cache: TTLCache[str, str] = TTLCache(max_size=10, default_ttl_seconds=60.0)

    assert cache.get("missing") is None
    assert cache.misses == 1
    assert cache.hits == 0

    cache.set("alpha", "value_a")
    assert cache.get("alpha") == "value_a"
    assert cache.hits == 1
    assert cache.size == 1

    # Invalidation
    assert cache.invalidate("alpha") is True
    assert cache.get("alpha") is None
    assert cache.size == 0


def test_ttl_cache_lru_eviction() -> None:
    """Verify that exceeding max_size evicts the least recently accessed item."""
    cache: TTLCache[str, int] = TTLCache(max_size=3, default_ttl_seconds=60.0)

    cache.set("k1", 1)
    cache.set("k2", 2)
    cache.set("k3", 3)

    # Access k1 so k2 becomes the least recently used
    assert cache.get("k1") == 1

    # Adding 4th item should evict k2
    cache.set("k4", 4)

    assert cache.get("k2") is None
    assert cache.get("k1") == 1
    assert cache.get("k3") == 3
    assert cache.get("k4") == 4
    assert cache.evictions == 1


def test_ttl_cache_ttl_expiration() -> None:
    """Verify entries expire and return None after TTL passes."""
    cache: TTLCache[str, str] = TTLCache(max_size=10, default_ttl_seconds=0.1)

    cache.set("transient", "ephemeral_data", ttl_seconds=0.1)
    assert cache.get("transient") == "ephemeral_data"

    # Wait for TTL expiration
    time.sleep(0.15)

    assert cache.get("transient") is None
    assert cache.size == 0


def test_ttl_cache_thread_safety() -> None:
    """Verify concurrent reads and writes maintain cache consistency."""
    cache: TTLCache[int, int] = TTLCache(max_size=50, default_ttl_seconds=10.0)

    def worker(worker_id: int) -> None:
        for i in range(100):
            key = (worker_id * 100 + i) % 70
            cache.set(key, i)
            _ = cache.get(key)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(8)]
        for f in futures:
            f.result()

    assert cache.size <= 50
    assert cache.hits > 0
    assert cache.evictions > 0


def test_query_embedding_cache_normalization() -> None:
    """Verify case-insensitivity and whitespace trimming on embedding cache keys."""
    cache = QueryEmbeddingCache(max_size=10, ttl_seconds=60.0)

    vec = [0.1, 0.2, 0.3]
    cache.set_embedding("  Hello World  ", "mock-model", 3, vec)

    # Must hit with different casing and spacing
    assert cache.get_embedding("hello world", "mock-model", 3) == vec
    assert cache.get_embedding("HELLO WORLD", "mock-model", 3) == vec
    # Different dimensions or model must miss
    assert cache.get_embedding("hello world", "mock-model", 4) is None
    assert cache.get_embedding("hello world", "other-model", 3) is None


@pytest.mark.asyncio
async def test_embeddings_service_caching_integration() -> None:
    """Verify EmbeddingsService bypasses provider on repeat queries."""
    cache = QueryEmbeddingCache(max_size=10)
    provider = MockEmbeddingProvider(dimensions=16)
    service = EmbeddingsService(provider=provider, cache=cache)

    query = "What is our company parental leave policy?"

    # First call (cold - cache miss)
    vec1 = await service.embed_query(query)
    stats1 = cache.get_stats()
    assert stats1["misses"] == 1
    assert stats1["hits"] == 0

    # Second call (warm - cache hit)
    vec2 = await service.embed_query(query)
    stats2 = cache.get_stats()
    assert stats2["hits"] == 1
    assert vec1 == vec2

    # Batch call with cached and uncached texts
    batch_res = await service.embed_texts([query, "new uncached text"])
    assert len(batch_res.vectors) == 2
    assert batch_res.vectors[0] == vec1
    stats3 = cache.get_stats()
    assert stats3["hits"] == 2


@pytest.mark.asyncio
async def test_retrieval_cache_integration() -> None:
    """Verify TwoStageRetrievalService caches reranked results."""
    settings = Settings(
        app_env="testing",
        embedding_dimensions=16,
        qdrant_url=":memory:",
    )
    cache = RetrievalCache(max_size=10, ttl_seconds=60.0)

    provider = MockEmbeddingProvider(dimensions=16)
    emb_service = EmbeddingsService(provider=provider)
    qdrant_client = AsyncQdrantClient(location=":memory:")
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name="test_perf_vectors")
    vector_service = VectorSearchService(
        vector_store=vector_store,
        embeddings_service=emb_service,
    )
    sparse_store = InMemoryBM25Store()

    hybrid_service = HybridSearchService(
        vector_service=vector_service,
        sparse_store=sparse_store,
    )
    reranker = create_reranker(settings)
    retrieval_service = TwoStageRetrievalService(
        hybrid_service=hybrid_service,
        vector_service=vector_service,
        reranker=reranker,
        cache=cache,
    )

    req = RerankRequest(query="SOC2 compliance policy", top_k=2)

    # First call: cold
    res1 = await retrieval_service.retrieve_and_rerank(req)
    assert cache.get_stats()["misses"] == 1

    # Second call: warm
    res2 = await retrieval_service.retrieve_and_rerank(req)
    assert cache.get_stats()["hits"] == 1
    assert res1.query == res2.query
    assert res1.total_results == res2.total_results


@pytest.mark.asyncio
async def test_tool_registry_concurrent_dispatch() -> None:
    """Verify execute_many runs multiple tool calls concurrently and preserves order."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(CurrentTimeTool())

    tool_calls = [
        ("calculator", {"expression": "100 + 200"}),
        ("current_time", {}),
        ("calculator", {"expression": "50 * 4"}),
    ]

    results = await registry.execute_many(tool_calls)

    assert len(results) == 3
    assert "300" in results[0].output
    assert "UTC" in results[1].output
    assert "200" in results[2].output
    assert all(not r.is_error for r in results)
