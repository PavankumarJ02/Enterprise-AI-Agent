"""Unit tests for query transformation components (HyDE, Multi-Query, Step-Back)."""

from unittest.mock import AsyncMock

import pytest

from enterprise_agent.hybrid.service import HybridSearchService
from enterprise_agent.llm.base import LLMProvider, LLMResponse, TokenUsage
from enterprise_agent.reranking.mock import MockReranker
from enterprise_agent.reranking.service import TwoStageRetrievalService
from enterprise_agent.schemas.search import (
    HybridSearchResponse,
    HybridSearchResultItem,
    SearchResultItem,
    SemanticSearchResponse,
)
from enterprise_agent.schemas.transformation import (
    TransformedSearchRequest,
)
from enterprise_agent.transformation.hyde import HyDETransformer
from enterprise_agent.transformation.multi_query import MultiQueryTransformer
from enterprise_agent.transformation.service import QueryTransformationService
from enterprise_agent.transformation.step_back import StepBackTransformer
from enterprise_agent.vectorstore.service import VectorSearchService


def _mock_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="mock-llm-v1",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
        finish_reason="stop",
    )


@pytest.mark.asyncio
async def test_hyde_transformer() -> None:
    """Verify HyDE transformer generates cleaned hypothetical documents."""
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = _mock_llm_response(
        '"Corporate policy allows $75 daily meal reimbursement during business travel."'
    )

    transformer = HyDETransformer(llm=mock_llm)
    hypotheses = await transformer.transform(query="travel meal limit", num_hypotheses=1)

    assert len(hypotheses) == 1
    assert "Corporate policy allows $75 daily meal reimbursement" in hypotheses[0]
    assert not hypotheses[0].startswith('"')
    mock_llm.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_multi_query_transformer_parsing_and_cleaning() -> None:
    """Verify MultiQuery transformer strips numbering, bullets, and retains original query."""
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = _mock_llm_response(
        "1. Kafka consumer group rebalance lag causes\n"
        "2) Partition assignment latency during rebalance\n"
        "- Heartbeat thread timeout in Kafka consumers\n"
        "1. Kafka consumer group rebalance lag causes\n"  # duplicate
    )

    transformer = MultiQueryTransformer(llm=mock_llm)
    variations = await transformer.transform(
        query="Kafka rebalance lag",
        num_variations=3,
    )

    # Must contain original query first
    assert variations[0] == "Kafka rebalance lag"
    # Numbering and bullets stripped
    assert "Kafka consumer group rebalance lag causes" in variations
    assert "Partition assignment latency during rebalance" in variations
    assert "Heartbeat thread timeout in Kafka consumers" in variations
    # No bullets or numbers in output
    for v in variations:
        assert not v.startswith(("1.", "2)", "-"))


@pytest.mark.asyncio
async def test_step_back_transformer_structured_parsing() -> None:
    """Verify StepBack transformer extracts structured question and rationale."""
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = _mock_llm_response(
        "Step-Back Question: What is the Kafka consumer group rebalancing protocol?\n"
        "Rationale: Understanding consumer group protocol explains why lag spikes occur."
    )

    transformer = StepBackTransformer(llm=mock_llm)
    sb_q, rationale = await transformer.generate_step_back(
        query="Why does consumer lag spike during rebalance?"
    )

    assert sb_q == "What is the Kafka consumer group rebalancing protocol?"
    assert rationale == "Understanding consumer group protocol explains why lag spikes occur."


@pytest.mark.asyncio
async def test_step_back_transformer_fallback() -> None:
    """Verify StepBack transformer falls back gracefully when output lacks labels."""
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = _mock_llm_response(
        "What are the foundational principles of distributed consensus?"
    )

    transformer = StepBackTransformer(llm=mock_llm)
    sb_q, rationale = await transformer.generate_step_back(query="Raft election timeout")

    assert sb_q == "What are the foundational principles of distributed consensus?"
    assert rationale is None


@pytest.mark.asyncio
async def test_query_transformation_service_hyde_pipeline() -> None:
    """Verify HyDE search pipeline uses hypothetical passage for vector retrieval."""
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = _mock_llm_response(
        "Hypothetical excerpt on corporate travel policy."
    )

    hyde_tf = HyDETransformer(llm=mock_llm)
    mq_tf = MultiQueryTransformer(llm=mock_llm)
    sb_tf = StepBackTransformer(llm=mock_llm)

    mock_hybrid = AsyncMock(spec=HybridSearchService)
    mock_vector = AsyncMock(spec=VectorSearchService)
    mock_reranker = MockReranker()
    two_stage = TwoStageRetrievalService(
        hybrid_service=mock_hybrid,
        vector_service=mock_vector,
        reranker=mock_reranker,
    )

    dummy_item = SearchResultItem(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="Actual travel guidelines document.",
        score=0.85,
        chunk_index=0,
        metadata={},
    )
    mock_vector.semantic_search.return_value = SemanticSearchResponse(
        query="Hypothetical excerpt on corporate travel policy.",
        total_results=1,
        results=[dummy_item],
    )

    service = QueryTransformationService(
        hyde_transformer=hyde_tf,
        multi_query_transformer=mq_tf,
        step_back_transformer=sb_tf,
        two_stage_service=two_stage,
        hybrid_service=mock_hybrid,
        vector_service=mock_vector,
    )

    res = await service.search_with_transformation(
        TransformedSearchRequest(
            query="travel policy limits",
            strategy="hyde",
            top_k=5,
            enable_reranking=True,
        )
    )

    assert res.strategy == "hyde"
    assert len(res.transformed_queries) >= 1
    assert res.total_results == 1
    assert res.results[0].chunk_id == "chunk-1"
    # Verified vector search was called with the hypothetical passage, NOT raw query
    mock_vector.semantic_search.assert_awaited_once()
    assert (
        mock_vector.semantic_search.call_args[1]["query"]
        == "Hypothetical excerpt on corporate travel policy."
    )


@pytest.mark.asyncio
async def test_query_transformation_service_multi_query_pipeline() -> None:
    """Verify MultiQuery search runs concurrent queries and deduplicates chunks."""
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.generate.return_value = _mock_llm_response("1. Syn 1\n2. Syn 2\n")

    hyde_tf = HyDETransformer(llm=mock_llm)
    mq_tf = MultiQueryTransformer(llm=mock_llm)
    sb_tf = StepBackTransformer(llm=mock_llm)

    mock_hybrid = AsyncMock(spec=HybridSearchService)
    mock_vector = AsyncMock(spec=VectorSearchService)
    mock_reranker = MockReranker()
    two_stage = TwoStageRetrievalService(
        hybrid_service=mock_hybrid,
        vector_service=mock_vector,
        reranker=mock_reranker,
    )

    item1 = HybridSearchResultItem(
        chunk_id="chunk-shared",
        document_id="doc-1",
        content="Shared chunk matching multiple queries.",
        score=0.8,
        chunk_index=0,
        metadata={},
        combined_score=0.8,
    )
    mock_hybrid.search.return_value = HybridSearchResponse(
        query="q",
        total_results=1,
        fusion_method="rrf",
        results=[item1],
    )

    service = QueryTransformationService(
        hyde_transformer=hyde_tf,
        multi_query_transformer=mq_tf,
        step_back_transformer=sb_tf,
        two_stage_service=two_stage,
        hybrid_service=mock_hybrid,
        vector_service=mock_vector,
    )

    res = await service.search_with_transformation(
        TransformedSearchRequest(
            query="original query",
            strategy="multi_query",
            top_k=5,
            enable_reranking=False,
        )
    )

    assert res.strategy == "multi_query"
    # Despite 3 queries returning chunk-shared, result set is deduplicated to 1 item
    assert res.total_results == 1
    assert res.results[0].chunk_id == "chunk-shared"
