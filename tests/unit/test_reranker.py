"""Unit tests for Cross-Encoder Rerankers (Mock and FlashRank) and factory resolution."""

import pytest

from enterprise_agent.config.settings import Settings
from enterprise_agent.core.exceptions import ConfigurationError
from enterprise_agent.reranking.factory import create_reranker
from enterprise_agent.reranking.flashrank import FlashRankReranker
from enterprise_agent.reranking.mock import MockReranker
from enterprise_agent.schemas.search import SearchResultItem


def _make_candidate(
    chunk_id: str,
    content: str,
    score: float = 0.5,
    chunk_index: int = 0,
) -> SearchResultItem:
    """Helper to construct a SearchResultItem candidate."""
    return SearchResultItem(
        chunk_id=chunk_id,
        document_id="doc-test-1",
        content=content,
        score=score,
        chunk_index=chunk_index,
        metadata={"source": "test.txt"},
    )


@pytest.mark.asyncio
async def test_mock_reranker_empty_items() -> None:
    """MockReranker returns empty list when given empty items."""
    reranker = MockReranker()
    results = await reranker.rerank(query="anything", items=[], top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_mock_reranker_lexical_overlap_and_ranking() -> None:
    """MockReranker ranks item with higher query word overlap first."""
    reranker = MockReranker()

    c1 = _make_candidate("c1", "Apples and bananas are nutritious fruits.", score=0.8)
    c2 = _make_candidate("c2", "The company travel policy requires receipts.", score=0.4)

    # Initial order: c1 first, c2 second
    results = await reranker.rerank(query="travel policy receipts", items=[c1, c2], top_k=5)

    assert len(results) == 2
    # c2 should have re-ranked to top position because of high word overlap
    top = results[0]
    assert top.chunk_id == "c2"
    assert top.initial_rank == 2
    assert top.final_rank == 1
    assert top.rerank_score > results[1].rerank_score
    assert top.initial_score == 0.4


@pytest.mark.asyncio
async def test_mock_reranker_forced_scores_and_top_k() -> None:
    """MockReranker respects forced scores and truncates to top_k."""
    reranker = MockReranker(forced_scores={"c1": 0.99, "c2": 0.12, "c3": 0.55})

    items = [
        _make_candidate("c2", "item 2", score=0.2),
        _make_candidate("c3", "item 3", score=0.3),
        _make_candidate("c1", "item 1", score=0.1),
    ]

    results = await reranker.rerank(query="test", items=items, top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[0].rerank_score == 0.99
    assert results[0].final_rank == 1
    assert results[0].initial_rank == 3

    assert results[1].chunk_id == "c3"
    assert results[1].rerank_score == 0.55
    assert results[1].final_rank == 2
    assert results[1].initial_rank == 2


@pytest.mark.asyncio
async def test_flashrank_reranker_empty_items() -> None:
    """FlashRankReranker returns empty list without error when given no items."""
    reranker = FlashRankReranker(model_name="ms-marco-TinyBERT-L-2-v2")
    results = await reranker.rerank(query="France capital", items=[], top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_flashrank_reranker_semantic_reordering() -> None:
    """FlashRank TinyBERT correctly ranks the semantically relevant answer at rank 1."""
    reranker = FlashRankReranker(model_name="ms-marco-TinyBERT-L-2-v2")

    # Candidate 1: completely irrelevant
    c1 = _make_candidate(
        "chunk-distractor",
        "Oranges, lemons, and grapefruits are citrus fruits that grow in warm climates.",
        score=0.9,
    )
    # Candidate 2: exact factual answer
    c2 = _make_candidate(
        "chunk-answer",
        "Paris is the official capital and most populous city of France.",
        score=0.3,
    )

    # Coarse Stage 1 ranked distractor #1 (initial_score 0.9) and answer #2 (initial_score 0.3)
    results = await reranker.rerank(
        query="What is the capital of France?",
        items=[c1, c2],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "chunk-answer"
    assert results[0].final_rank == 1
    assert results[0].initial_rank == 2
    assert results[0].initial_score == 0.3

    assert results[1].chunk_id == "chunk-distractor"
    assert results[1].final_rank == 2
    assert results[1].initial_rank == 1
    assert results[0].rerank_score > results[1].rerank_score


def test_reranker_factory_resolution() -> None:
    """Factory correctly resolves mock and flashrank providers based on settings."""
    mock_settings = Settings(reranker_provider="mock", reranker_model="test-mock")
    mock_r = create_reranker(mock_settings)
    assert isinstance(mock_r, MockReranker)
    assert mock_r.model_name == "test-mock"

    flash_settings = Settings(
        reranker_provider="flashrank",
        reranker_model="ms-marco-TinyBERT-L-2-v2",
    )
    flash_r = create_reranker(flash_settings)
    assert isinstance(flash_r, FlashRankReranker)
    assert flash_r.model_name == "ms-marco-TinyBERT-L-2-v2"

    # Unsupported provider raises ConfigurationError
    invalid_settings = Settings.model_construct(reranker_provider="unsupported")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="Unsupported reranker provider"):
        create_reranker(invalid_settings)
