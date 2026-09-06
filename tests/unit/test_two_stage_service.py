"""Unit tests for TwoStageRetrievalService orchestration and strategy routing."""

from unittest.mock import AsyncMock

import pytest

from enterprise_agent.hybrid.service import HybridSearchService
from enterprise_agent.reranking.mock import MockReranker
from enterprise_agent.reranking.service import TwoStageRetrievalService
from enterprise_agent.schemas.search import (
    DirectRerankRequest,
    HybridSearchResponse,
    HybridSearchResultItem,
    RerankRequest,
    SearchResultItem,
    SemanticSearchResponse,
)
from enterprise_agent.vectorstore.service import VectorSearchService


def _dummy_search_item(chunk_id: str, content: str, score: float = 0.5) -> SearchResultItem:
    return SearchResultItem(
        chunk_id=chunk_id,
        document_id="doc-unit-1",
        content=content,
        score=score,
        chunk_index=0,
        metadata={"source": "test.txt"},
    )


def _dummy_hybrid_item(chunk_id: str, content: str, score: float = 0.5) -> HybridSearchResultItem:
    return HybridSearchResultItem(
        chunk_id=chunk_id,
        document_id="doc-unit-1",
        content=content,
        score=score,
        chunk_index=0,
        metadata={"source": "test.txt"},
        combined_score=score,
    )


@pytest.mark.asyncio
async def test_two_stage_service_hybrid_strategy() -> None:
    """TwoStageRetrievalService delegates to HybridSearchService for 'hybrid' strategy."""
    mock_hybrid = AsyncMock(spec=HybridSearchService)
    mock_vector = AsyncMock(spec=VectorSearchService)
    reranker = MockReranker(forced_scores={"h2": 0.95, "h1": 0.30})

    h1 = _dummy_hybrid_item("h1", "Hybrid candidate 1", score=0.8)
    h2 = _dummy_hybrid_item("h2", "Hybrid candidate 2", score=0.6)
    mock_hybrid.search.return_value = HybridSearchResponse(
        query="test query",
        total_results=2,
        fusion_method="rrf",
        results=[h1, h2],
    )

    service = TwoStageRetrievalService(
        hybrid_service=mock_hybrid,
        vector_service=mock_vector,
        reranker=reranker,
    )

    req = RerankRequest(
        query="test query",
        top_k=2,
        candidate_k=10,
        retrieval_strategy="hybrid",
    )
    res = await service.retrieve_and_rerank(req)

    mock_hybrid.search.assert_awaited_once()
    mock_vector.semantic_search.assert_not_awaited()
    assert res.total_candidates == 2
    assert res.total_results == 2
    # h2 should be rank 1 due to forced score 0.95
    assert res.results[0].chunk_id == "h2"
    assert res.results[0].final_rank == 1
    assert res.results[0].initial_rank == 2


@pytest.mark.asyncio
async def test_two_stage_service_semantic_strategy() -> None:
    """TwoStageRetrievalService delegates to VectorSearchService for 'semantic' strategy."""
    mock_hybrid = AsyncMock(spec=HybridSearchService)
    mock_vector = AsyncMock(spec=VectorSearchService)
    reranker = MockReranker()

    s1 = _dummy_search_item("s1", "Semantic match", score=0.9)
    mock_vector.semantic_search.return_value = SemanticSearchResponse(
        query="semantic query",
        total_results=1,
        results=[s1],
    )

    service = TwoStageRetrievalService(
        hybrid_service=mock_hybrid,
        vector_service=mock_vector,
        reranker=reranker,
    )

    req = RerankRequest(
        query="semantic query",
        top_k=5,
        candidate_k=10,
        retrieval_strategy="semantic",
    )
    res = await service.retrieve_and_rerank(req)

    mock_vector.semantic_search.assert_awaited_once()
    mock_hybrid.search.assert_not_awaited()
    assert res.total_candidates == 1
    assert res.results[0].chunk_id == "s1"


@pytest.mark.asyncio
async def test_two_stage_service_sparse_strategy() -> None:
    """TwoStageRetrievalService delegates to search_sparse_only for 'sparse' strategy."""
    mock_hybrid = AsyncMock(spec=HybridSearchService)
    mock_vector = AsyncMock(spec=VectorSearchService)
    reranker = MockReranker()

    sp1 = _dummy_search_item("sp1", "Keyword match", score=0.7)
    mock_hybrid.search_sparse_only.return_value = [sp1]

    service = TwoStageRetrievalService(
        hybrid_service=mock_hybrid,
        vector_service=mock_vector,
        reranker=reranker,
    )

    req = RerankRequest(
        query="keyword",
        top_k=5,
        candidate_k=10,
        retrieval_strategy="sparse",
    )
    res = await service.retrieve_and_rerank(req)

    mock_hybrid.search_sparse_only.assert_awaited_once()
    assert res.total_candidates == 1
    assert res.results[0].chunk_id == "sp1"


@pytest.mark.asyncio
async def test_two_stage_service_empty_candidates_short_circuits() -> None:
    """When Stage 1 yields zero candidates, service short-circuits with empty results."""
    mock_hybrid = AsyncMock(spec=HybridSearchService)
    mock_vector = AsyncMock(spec=VectorSearchService)
    reranker = MockReranker()

    mock_hybrid.search.return_value = HybridSearchResponse(
        query="empty",
        total_results=0,
        fusion_method="rrf",
        results=[],
    )

    service = TwoStageRetrievalService(
        hybrid_service=mock_hybrid,
        vector_service=mock_vector,
        reranker=reranker,
    )

    req = RerankRequest(query="empty", top_k=5, candidate_k=20)
    res = await service.retrieve_and_rerank(req)

    assert res.total_candidates == 0
    assert res.total_results == 0
    assert res.results == []


@pytest.mark.asyncio
async def test_two_stage_service_direct_rerank() -> None:
    """Direct rerank bypasses Stage 1 retrieval and directly re-scores supplied candidates."""
    mock_hybrid = AsyncMock(spec=HybridSearchService)
    mock_vector = AsyncMock(spec=VectorSearchService)
    reranker = MockReranker(forced_scores={"d2": 0.88, "d1": 0.44})

    d1 = _dummy_search_item("d1", "Direct item 1", score=0.1)
    d2 = _dummy_search_item("d2", "Direct item 2", score=0.2)

    service = TwoStageRetrievalService(
        hybrid_service=mock_hybrid,
        vector_service=mock_vector,
        reranker=reranker,
    )

    req = DirectRerankRequest(query="test", items=[d1, d2], top_k=2)
    res = await service.direct_rerank(req)

    assert res.total_candidates == 2
    assert res.total_results == 2
    assert res.results[0].chunk_id == "d2"
    assert res.results[0].final_rank == 1
    assert res.results[1].chunk_id == "d1"
    assert res.results[1].final_rank == 2
