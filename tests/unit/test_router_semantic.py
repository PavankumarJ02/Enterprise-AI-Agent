"""Unit tests for SemanticEmbeddingRouter."""

import pytest

from enterprise_agent.embeddings.mock import MockEmbeddingProvider
from enterprise_agent.router.semantic import SemanticEmbeddingRouter
from enterprise_agent.schemas.router import QueryIntent


@pytest.fixture
def mock_embedder() -> MockEmbeddingProvider:
    """Fixture providing deterministic mock embedding provider."""
    return MockEmbeddingProvider(dimensions=16)


@pytest.mark.asyncio
async def test_semantic_router_exact_match(mock_embedder: MockEmbeddingProvider) -> None:
    """Verify semantic router matches query identical or close to exemplar."""
    custom_exemplars = {
        QueryIntent.DIRECT_CHAT: ["Hello, how are you today?"],
        QueryIntent.RAG_SEARCH: ["What is the company vacation policy?"],
        QueryIntent.SQL_DATABASE: ["How many employees work here?"],
        QueryIntent.AUTONOMOUS_AGENT: ["Search policy and calculate bonus."],
    }
    router = SemanticEmbeddingRouter(
        embedding_provider=mock_embedder,
        threshold=0.50,
        exemplars=custom_exemplars,
    )

    # Identical query should yield maximum similarity (close to 1.0)
    result = await router.classify("What is the company vacation policy?")
    assert result is not None
    intent, conf, reasoning = result
    assert intent == QueryIntent.RAG_SEARCH
    assert conf >= 0.80
    assert "rag_search" in reasoning


@pytest.mark.asyncio
async def test_semantic_router_threshold_fallback(mock_embedder: MockEmbeddingProvider) -> None:
    """Verify router returns None when similarity score is below strict threshold."""
    custom_exemplars = {
        QueryIntent.DIRECT_CHAT: ["Greetings friend"],
        QueryIntent.SQL_DATABASE: ["Count sales orders"],
    }
    # Unattainable threshold 1.01
    router = SemanticEmbeddingRouter(
        embedding_provider=mock_embedder,
        threshold=1.01,
        exemplars=custom_exemplars,
    )

    result = await router.classify("Hello")
    assert result is None


@pytest.mark.asyncio
async def test_semantic_router_empty_query(mock_embedder: MockEmbeddingProvider) -> None:
    """Verify empty string returns None without error."""
    router = SemanticEmbeddingRouter(embedding_provider=mock_embedder)
    assert await router.classify("   ") is None
