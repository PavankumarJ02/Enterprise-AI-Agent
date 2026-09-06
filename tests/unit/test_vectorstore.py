"""Unit tests for Qdrant Vector Store and VectorSearchService."""

import uuid
from collections.abc import AsyncGenerator

import pytest
from qdrant_client import AsyncQdrantClient

from enterprise_agent.core.exceptions import VectorStoreError
from enterprise_agent.embeddings.mock import MockEmbeddingProvider
from enterprise_agent.embeddings.service import EmbeddingsService
from enterprise_agent.ingestion.models import DocumentChunk
from enterprise_agent.vectorstore.qdrant import QdrantVectorStore, chunk_id_to_uuid
from enterprise_agent.vectorstore.service import VectorSearchService


def test_chunk_id_to_uuid_is_deterministic() -> None:
    """Verify chunk_id_to_uuid generates a valid, deterministic UUIDv5."""
    chunk_id = "doc_alpha_123:chunk_0"
    u1 = chunk_id_to_uuid(chunk_id)
    u2 = chunk_id_to_uuid(chunk_id)

    assert u1 == u2
    # Verify it parses as a valid UUID
    parsed = uuid.UUID(u1)
    assert str(parsed) == u1

    # Distinct chunks must yield distinct UUIDs
    u3 = chunk_id_to_uuid("doc_alpha_123:chunk_1")
    assert u1 != u3


@pytest.fixture
async def in_memory_qdrant() -> AsyncGenerator[QdrantVectorStore, None]:
    """Provide an in-memory QdrantVectorStore fixture."""
    client = AsyncQdrantClient(location=":memory:")
    store = QdrantVectorStore(client=client, collection_name="test_knowledge")
    yield store
    await client.close()


@pytest.mark.asyncio
async def test_collection_lifecycle(in_memory_qdrant: QdrantVectorStore) -> None:
    """Verify collection creation and idempotency."""
    assert not await in_memory_qdrant.collection_exists()

    await in_memory_qdrant.create_collection_if_not_exists(dimensions=4)
    assert await in_memory_qdrant.collection_exists()

    # Idempotent call should succeed without error
    await in_memory_qdrant.create_collection_if_not_exists(dimensions=4)
    assert await in_memory_qdrant.collection_exists()


@pytest.mark.asyncio
async def test_upsert_and_count(in_memory_qdrant: QdrantVectorStore) -> None:
    """Verify chunk upsert and vector count tracking."""
    await in_memory_qdrant.create_collection_if_not_exists(dimensions=3)

    chunks = [
        DocumentChunk(
            id="doc1:chunk_0",
            document_id="doc1",
            content="Alpha chunk content",
            chunk_index=0,
            metadata={"source": "manual.pdf", "page": 1},
        ),
        DocumentChunk(
            id="doc1:chunk_1",
            document_id="doc1",
            content="Beta chunk content",
            chunk_index=1,
            metadata={"source": "manual.pdf", "page": 2},
        ),
    ]
    vectors = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    upserted = await in_memory_qdrant.upsert(chunks, vectors)
    assert upserted == 2

    total = await in_memory_qdrant.count()
    assert total == 2


@pytest.mark.asyncio
async def test_upsert_empty_and_mismatch(in_memory_qdrant: QdrantVectorStore) -> None:
    """Verify empty upsert returns 0 and mismatch raises VectorStoreError."""
    await in_memory_qdrant.create_collection_if_not_exists(dimensions=3)

    # Empty list
    assert await in_memory_qdrant.upsert([], []) == 0

    chunk = DocumentChunk(
        id="doc1:chunk_0",
        document_id="doc1",
        content="Test content",
        chunk_index=0,
    )
    # Mismatched lengths
    with pytest.raises(VectorStoreError, match="Mismatch"):
        await in_memory_qdrant.upsert([chunk], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


@pytest.mark.asyncio
async def test_semantic_search_and_ranking(in_memory_qdrant: QdrantVectorStore) -> None:
    """Verify cosine similarity search returns sorted results."""
    await in_memory_qdrant.create_collection_if_not_exists(dimensions=3)

    chunks = [
        DocumentChunk(
            id="docA:chunk_0",
            document_id="docA",
            content="High similarity chunk",
            chunk_index=0,
        ),
        DocumentChunk(
            id="docB:chunk_0",
            document_id="docB",
            content="Opposite similarity chunk",
            chunk_index=0,
        ),
    ]
    # Unit vectors
    vectors = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]
    await in_memory_qdrant.upsert(chunks, vectors)

    # Search with query vector identical to docA
    query = [1.0, 0.0, 0.0]
    results = await in_memory_qdrant.search(query_vector=query, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "docA:chunk_0"
    assert pytest.approx(results[0].score, 0.001) == 1.0
    assert results[1].chunk_id == "docB:chunk_0"
    assert pytest.approx(results[1].score, 0.001) == -1.0


@pytest.mark.asyncio
async def test_search_with_metadata_filters(in_memory_qdrant: QdrantVectorStore) -> None:
    """Verify payload filtering restricts search to matching chunks."""
    await in_memory_qdrant.create_collection_if_not_exists(dimensions=2)

    chunks = [
        DocumentChunk(
            id="c1",
            document_id="doc_hr",
            content="HR handbook",
            chunk_index=0,
            metadata={"department": "HR"},
        ),
        DocumentChunk(
            id="c2",
            document_id="doc_fin",
            content="Finance ledger",
            chunk_index=0,
            metadata={"department": "Finance"},
        ),
    ]
    vectors = [
        [1.0, 0.0],
        [1.0, 0.0],
    ]
    await in_memory_qdrant.upsert(chunks, vectors)

    # Filter by document_id == "doc_hr"
    results = await in_memory_qdrant.search(
        query_vector=[1.0, 0.0],
        top_k=5,
        filters={"document_id": "doc_hr"},
    )
    assert len(results) == 1
    assert results[0].document_id == "doc_hr"

    # Filter by non-existent document
    empty_results = await in_memory_qdrant.search(
        query_vector=[1.0, 0.0],
        top_k=5,
        filters={"document_id": "non_existent"},
    )
    assert len(empty_results) == 0


@pytest.mark.asyncio
async def test_search_min_score_threshold(in_memory_qdrant: QdrantVectorStore) -> None:
    """Verify min_score filters out matches below similarity cutoff."""
    await in_memory_qdrant.create_collection_if_not_exists(dimensions=2)

    chunks = [
        DocumentChunk(id="c1", document_id="d1", content="Chunk 1", chunk_index=0),
        DocumentChunk(id="c2", document_id="d2", content="Chunk 2", chunk_index=0),
    ]
    vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    await in_memory_qdrant.upsert(chunks, vectors)

    # Search with query [1.0, 0.0] and min_score 0.8 -> only c1 matches
    results = await in_memory_qdrant.search(
        query_vector=[1.0, 0.0],
        top_k=5,
        min_score=0.8,
    )
    assert len(results) == 1
    assert results[0].chunk_id == "c1"


@pytest.mark.asyncio
async def test_delete_by_document_id(in_memory_qdrant: QdrantVectorStore) -> None:
    """Verify deleting by document ID purges points."""
    await in_memory_qdrant.create_collection_if_not_exists(dimensions=2)

    chunks = [
        DocumentChunk(id="c1", document_id="doc_del", content="To delete", chunk_index=0),
        DocumentChunk(id="c2", document_id="doc_keep", content="To keep", chunk_index=0),
    ]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    await in_memory_qdrant.upsert(chunks, vectors)
    assert await in_memory_qdrant.count() == 2

    # Delete doc_del
    await in_memory_qdrant.delete_by_document_id("doc_del")
    assert await in_memory_qdrant.count() == 1

    # Verify only doc_keep remains
    remaining = await in_memory_qdrant.search(query_vector=[0.0, 1.0], top_k=5)
    assert len(remaining) == 1
    assert remaining[0].document_id == "doc_keep"


@pytest.mark.asyncio
async def test_health_check(in_memory_qdrant: QdrantVectorStore) -> None:
    """Verify health check returns True for active client."""
    assert await in_memory_qdrant.health_check() is True


@pytest.mark.asyncio
async def test_vector_search_service_workflow() -> None:
    """Verify VectorSearchService end-to-end orchestration with mock embeddings."""
    client = AsyncQdrantClient(location=":memory:")
    vector_store = QdrantVectorStore(client=client, collection_name="service_test")
    provider = MockEmbeddingProvider(dimensions=8)
    embeddings_service = EmbeddingsService(provider=provider)

    service = VectorSearchService(
        vector_store=vector_store,
        embeddings_service=embeddings_service,
    )

    chunks = [
        DocumentChunk(
            id="svc_doc:chunk_0",
            document_id="svc_doc",
            content="Enterprise compliance and governance standards.",
            chunk_index=0,
            metadata={"source": "compliance.txt"},
        ),
        DocumentChunk(
            id="svc_doc:chunk_1",
            document_id="svc_doc",
            content="Kitchen snacks policy and coffee machine usage.",
            chunk_index=1,
            metadata={"source": "compliance.txt"},
        ),
    ]

    indexed_count = await service.index_chunks(chunks)
    assert indexed_count == 2

    response = await service.semantic_search(
        query="governance and compliance",
        top_k=2,
    )
    assert response.query == "governance and compliance"
    assert response.total_results == 2
    assert len(response.results) == 2
    assert isinstance(response.results[0].score, float)

    await client.close()
