"""Unit tests for RAGEvaluationService."""

from unittest.mock import AsyncMock

import pytest

from enterprise_agent.config.settings import Settings
from enterprise_agent.embeddings.mock import MockEmbeddingProvider
from enterprise_agent.evaluation.service import RAGEvaluationService
from enterprise_agent.schemas.chat import TokenUsageResponse
from enterprise_agent.schemas.evaluation import EvaluationSample
from enterprise_agent.schemas.rag import RAGQueryResponse, RetrievedSourceChunk


@pytest.mark.asyncio
async def test_evaluation_service_single_sample() -> None:
    """Verify single sample evaluation returns all 4 metrics."""
    provider = MockEmbeddingProvider(dimensions=16)
    service = RAGEvaluationService(embedding_provider=provider)

    sample = EvaluationSample(
        query="What is the internet allowance?",
        ground_truth="Internet allowance is $100 per month.",
        golden_contexts=["Section 3.1: Remote employees receive $100 per month for internet."],
    )
    res = await service.evaluate_sample(sample)
    assert res.query == sample.query
    assert res.faithfulness >= 0.90
    assert 0.0 <= res.answer_relevance <= 1.0
    assert res.context_precision == 1.0
    assert res.context_recall == 1.0


@pytest.mark.asyncio
async def test_evaluation_service_dataset_offline() -> None:
    """Verify full benchmark dataset evaluation completes and passes configured thresholds."""
    provider = MockEmbeddingProvider(dimensions=16)
    settings = Settings(
        eval_faithfulness_threshold=0.75,
        eval_answer_relevance_threshold=0.60,
        eval_context_precision_threshold=0.70,
        eval_context_recall_threshold=0.70,
    )
    service = RAGEvaluationService(embedding_provider=provider, settings=settings)

    report = await service.evaluate_dataset(use_live_rag=False)
    assert report.total_samples == 10
    assert "faithfulness" in report.metrics
    assert "answer_relevance" in report.metrics
    assert "context_precision" in report.metrics
    assert "context_recall" in report.metrics

    # On golden data with ground truth as answers, all metrics should pass
    assert report.overall_passed is True
    assert report.total_latency_ms >= 0.0


@pytest.mark.asyncio
async def test_evaluation_service_dataset_strict_threshold_fail() -> None:
    """Verify overall_passed is False when thresholds are unmet."""
    provider = MockEmbeddingProvider(dimensions=16)
    settings = Settings(
        eval_faithfulness_threshold=0.99,  # Unmet strict threshold
    )
    service = RAGEvaluationService(embedding_provider=provider, settings=settings)

    report = await service.evaluate_dataset(use_live_rag=False)
    assert report.metrics["faithfulness"].passed is False
    assert report.overall_passed is False


@pytest.mark.asyncio
async def test_evaluation_service_with_live_rag_mock() -> None:
    """Verify live RAG mode invokes rag_service.query and calculates metrics."""
    provider = MockEmbeddingProvider(dimensions=16)
    mock_rag = AsyncMock()
    mock_rag.query.return_value = RAGQueryResponse(
        query="What is the internet allowance?",
        answer="The internet allowance is $100 per month.",
        sources=[
            RetrievedSourceChunk(
                chunk_id="chk-1",
                document_id="doc-1",
                chunk_index=0,
                source="handbook.md",
                content="Section 3.1: Remote employees receive $100 per month for internet.",
                score=0.92,
            )
        ],
        model="mock-model",
        usage=TokenUsageResponse(prompt_tokens=50, completion_tokens=15, total_tokens=65),
        latency_ms=12.5,
    )

    service = RAGEvaluationService(
        embedding_provider=provider,
        rag_service=mock_rag,
    )

    custom_samples = [
        EvaluationSample(
            query="What is the internet allowance?",
            ground_truth="Internet allowance is $100 per month.",
            golden_contexts=["Section 3.1: Remote employees receive $100 per month for internet."],
        )
    ]

    report = await service.evaluate_dataset(samples=custom_samples, use_live_rag=True, top_k=2)
    assert report.total_samples == 1
    assert mock_rag.query.await_count == 1
    assert report.samples[0].faithfulness >= 0.90
