"""Unit tests for HybridSearchService coordinating dense and sparse retrieval."""

from collections.abc import AsyncGenerator

import pytest
from qdrant_client import AsyncQdrantClient

from enterprise_agent.embeddings.mock import MockEmbeddingProvider
from enterprise_agent.embeddings.service import EmbeddingsService
from enterprise_agent.hybrid.service import HybridSearchService
from enterprise_agent.ingestion.models import DocumentChunk
from enterprise_agent.sparse.bm25 import InMemoryBM25Store
from enterprise_agent.vectorstore.qdrant import QdrantVectorStore
from enterprise_agent.vectorstore.service import VectorSearchService


@pytest.fixture
async def hybrid_service_env() -> AsyncGenerator[
    tuple[HybridSearchService, AsyncQdrantClient], None
]:
    """Provide initialized HybridSearchService connected to in-memory stores."""
    client = AsyncQdrantClient(location=":memory:")
    vector_store = QdrantVectorStore(client=client, collection_name="test_hybrid_col")
    provider = MockEmbeddingProvider(dimensions=8)
    embeddings_service = EmbeddingsService(provider=provider)
    vector_service = VectorSearchService(
        vector_store=vector_store,
        embeddings_service=embeddings_service,
    )
    sparse_store = InMemoryBM25Store()

    service = HybridSearchService(
        vector_service=vector_service,
        sparse_store=sparse_store,
        default_fusion_method="rrf",
    )

    yield service, client
    await client.close()


@pytest.mark.asyncio
async def test_hybrid_service_dual_indexing_and_search(
    hybrid_service_env: tuple[HybridSearchService, AsyncQdrantClient],
) -> None:
    """Verify dual indexing stores chunks in both dense and sparse indices."""
    service, _ = hybrid_service_env

    chunks = [
        DocumentChunk(
            id="c1",
            document_id="doc_policy",
            content="Section 10.4: SOC2 Compliance requires annual audits for all cloud systems.",
            chunk_index=0,
            metadata={"department": "Compliance"},
        ),
        DocumentChunk(
            id="c2",
            document_id="doc_finance",
            content="Section 2.1: Travel reimbursement guidelines and per diem allowances.",
            chunk_index=0,
            metadata={"department": "Finance"},
        ),
    ]

    dense_count, sparse_count = await service.index_chunks(chunks)
    assert dense_count == 2
    assert sparse_count == 2

    # 1. Search via RRF
    rrf_resp = await service.search(query="SOC2 Compliance", top_k=2)
    assert rrf_resp.total_results >= 1
    assert rrf_resp.fusion_method == "rrf"
    assert rrf_resp.results[0].chunk_id == "c1"
    assert rrf_resp.results[0].combined_score > 0

    # 2. Search via Linear
    linear_resp = await service.search(query="SOC2 Compliance", top_k=2, fusion_method="linear")
    assert linear_resp.fusion_method == "linear"
    assert linear_resp.results[0].chunk_id == "c1"


@pytest.mark.asyncio
async def test_hybrid_service_dual_deletion(
    hybrid_service_env: tuple[HybridSearchService, AsyncQdrantClient],
) -> None:
    """Verify deleting a document removes entries from both Qdrant and BM25."""
    service, _ = hybrid_service_env

    chunks = [
        DocumentChunk(
            id="c1",
            document_id="doc_to_delete",
            content="Temporary confidential draft.",
            chunk_index=0,
        )
    ]
    await service.index_chunks(chunks)

    # Delete
    dense_del, sparse_del = await service.delete_by_document_id("doc_to_delete")
    assert dense_del == 1
    assert sparse_del == 1

    # Verify search yields 0
    res = await service.search(query="confidential draft")
    assert len(res.results) == 0
