"""Unit tests for dense vector embedding providers, batching service, and factory."""

import math
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import errors
from pydantic import SecretStr

from enterprise_agent.config.settings import Settings
from enterprise_agent.core.exceptions import (
    ConfigurationError,
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
)
from enterprise_agent.embeddings.factory import get_embedding_provider, get_embeddings_service
from enterprise_agent.embeddings.gemini import GeminiEmbeddingProvider
from enterprise_agent.embeddings.mock import MockEmbeddingProvider
from enterprise_agent.embeddings.service import EmbeddingsService
from enterprise_agent.ingestion.models import DocumentChunk


@pytest.mark.asyncio
async def test_mock_embedding_provider_properties() -> None:
    """Verify MockEmbeddingProvider generates deterministic L2-normalized vectors."""
    provider = MockEmbeddingProvider(dimensions=768)

    # 1. Determinism
    vec1 = await provider.embed_query("Enterprise policy document")
    vec2 = await provider.embed_query("Enterprise policy document")
    assert vec1 == vec2
    assert len(vec1) == 768

    # 2. L2 Unit Normalization: sum(v_i^2) == 1.0
    norm_sq = sum(x * x for x in vec1)
    assert math.isclose(norm_sq, 1.0, rel_tol=1e-3)

    # 3. Batch texts
    res = await provider.embed_texts(["Chunk A", "Chunk B", "Chunk C"])
    assert len(res.vectors) == 3
    assert res.dimensions == 768
    assert res.token_count > 0

    # 4. Health check
    assert await provider.health_check() is True


@pytest.mark.asyncio
async def test_embeddings_service_batch_slicing() -> None:
    """Verify EmbeddingsService correctly breaks large inputs into batch slices."""
    provider = MockEmbeddingProvider(dimensions=64)
    # Configure batch_size = 3
    service = EmbeddingsService(provider=provider, batch_size=3)

    texts = [f"Text item {i}" for i in range(10)]
    result = await service.embed_texts(texts)

    assert len(result.vectors) == 10
    assert result.dimensions == 64
    # 10 items with batch_size 3 -> 4 calls (3, 3, 3, 1)
    assert len(provider.invocations) == 4
    assert len(provider.invocations[0]) == 3
    assert len(provider.invocations[-1]) == 1


@pytest.mark.asyncio
async def test_embeddings_service_chunk_vectorization() -> None:
    """Verify embed_document_chunks attaches embeddings to DocumentChunk objects."""
    provider = MockEmbeddingProvider(dimensions=128)
    service = EmbeddingsService(provider=provider)

    chunks = [
        DocumentChunk(
            id="doc1:chunk_0",
            document_id="doc1",
            content="First paragraph of customer policy.",
            chunk_index=0,
        ),
        DocumentChunk(
            id="doc1:chunk_1",
            document_id="doc1",
            content="Second paragraph of customer policy.",
            chunk_index=1,
        ),
    ]

    out_chunks, vectors = await service.embed_document_chunks(chunks)

    assert len(out_chunks) == 2
    assert len(vectors) == 2
    assert len(vectors[0]) == 128


def test_factory_resolves_mock() -> None:
    """Factory should instantiate MockEmbeddingProvider when configured."""
    settings = Settings(embedding_provider="mock", embedding_dimensions=768)
    provider = get_embedding_provider(settings)
    assert isinstance(provider, MockEmbeddingProvider)
    assert provider.dimensions == 768


def test_factory_gemini_missing_key() -> None:
    """Factory should raise ConfigurationError when Gemini key is absent."""
    settings = Settings(
        embedding_provider="gemini",
        gemini_api_key=SecretStr(""),
        llm_api_key=SecretStr(""),
    )
    with pytest.raises(ConfigurationError) as exc_info:
        get_embedding_provider(settings)
    assert "GEMINI_API_KEY" in str(exc_info.value)


def test_factory_openai_missing_key() -> None:
    """Factory should raise ConfigurationError when OpenAI key is absent."""
    settings = Settings(
        embedding_provider="openai",
        llm_api_key=SecretStr(""),
    )
    with pytest.raises(ConfigurationError) as exc_info:
        get_embedding_provider(settings)
    assert "LLM_API_KEY" in str(exc_info.value)


def test_factory_service_creation() -> None:
    """Factory creates EmbeddingsService configured with batch size."""
    settings = Settings(
        embedding_provider="mock",
        embedding_batch_size=32,
        embedding_dimensions=512,
    )
    service = get_embeddings_service(settings)
    assert isinstance(service, EmbeddingsService)
    assert service.batch_size == 32
    assert service.dimensions == 512


@pytest.mark.asyncio
async def test_gemini_embedding_provider_mocked() -> None:
    """Verify GeminiEmbeddingProvider sends RETRIEVAL_DOCUMENT and RETRIEVAL_QUERY."""
    provider = GeminiEmbeddingProvider(api_key="fake-gemini-key", dimensions=768)

    mock_emb1 = MagicMock()
    mock_emb1.values = [0.1] * 768
    mock_resp = MagicMock()
    mock_resp.embeddings = [mock_emb1]

    provider._client = MagicMock()
    provider._client.aio.models.embed_content = AsyncMock(return_value=mock_resp)

    # 1. Embed query
    query_vec = await provider.embed_query("What is the refund window?")
    assert len(query_vec) == 768
    call_args_query = provider._client.aio.models.embed_content.call_args[1]
    assert call_args_query["config"].task_type == "RETRIEVAL_QUERY"

    # 2. Embed document texts
    texts_res = await provider.embed_texts(["Document passage"])
    assert len(texts_res.vectors) == 1
    call_args_doc = provider._client.aio.models.embed_content.call_args[1]
    assert call_args_doc["config"].task_type == "RETRIEVAL_DOCUMENT"


def test_gemini_embedding_exception_mapping() -> None:
    """Verify Gemini errors map to domain exceptions."""
    provider = GeminiEmbeddingProvider(api_key="fake-key")

    auth_err = errors.APIError(401, "API_KEY_INVALID")
    assert isinstance(provider._map_exception(auth_err), LLMAuthenticationError)

    quota_err = errors.APIError(429, "Quota exhausted")
    assert isinstance(provider._map_exception(quota_err), LLMRateLimitError)

    srv_err = errors.APIError(503, "Service unavailable")
    assert isinstance(provider._map_exception(srv_err), LLMProviderError)
