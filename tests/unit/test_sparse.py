"""Unit tests for sparse keyword search and InMemoryBM25Store."""

import pytest

from enterprise_agent.ingestion.models import DocumentChunk
from enterprise_agent.sparse.bm25 import InMemoryBM25Store, tokenize_for_bm25


def test_tokenize_for_bm25() -> None:
    """Verify tokenizer preserves codes with hyphens, underscores, and dollar values."""
    text = "Security flaw CVE-2024-1234 affects SKU-8921-X. Lodging is $250 in Tier-1."
    tokens = tokenize_for_bm25(text)
    assert "cve-2024-1234" in tokens
    assert "sku-8921-x" in tokens
    assert "$250" in tokens
    assert "tier-1" in tokens


@pytest.mark.asyncio
async def test_bm25_store_index_and_search_exact_keyword() -> None:
    """Verify BM25 retrieves and ranks exact keyword/code matches above general terms."""
    store = InMemoryBM25Store()

    chunks = [
        DocumentChunk(
            id="c1",
            document_id="doc_sec",
            content="Protocol CVE-2024-9988 requires immediate patch within 24 hours.",
            chunk_index=0,
            metadata={"department": "Security", "source": "sec_bulletin.md"},
        ),
        DocumentChunk(
            id="c2",
            document_id="doc_gen",
            content="General software patch and maintenance policy for enterprise systems.",
            chunk_index=0,
            metadata={"department": "IT", "source": "it_handbook.md"},
        ),
    ]
    await store.index_chunks(chunks)

    # Search exact CVE code
    results = await store.search(query="CVE-2024-9988", top_k=5)
    assert len(results) >= 1
    assert results[0].chunk_id == "c1"
    assert results[0].score > 0.5
    assert "CVE-2024-9988" in results[0].content


@pytest.mark.asyncio
async def test_bm25_store_metadata_filtering() -> None:
    """Verify BM25 search respects metadata filters."""
    store = InMemoryBM25Store()

    chunks = [
        DocumentChunk(
            id="c1",
            document_id="d1",
            content="Firewall configuration rule 42.",
            chunk_index=0,
            metadata={"department": "Security"},
        ),
        DocumentChunk(
            id="c2",
            document_id="d2",
            content="Firewall expenses rule 42.",
            chunk_index=0,
            metadata={"department": "Finance"},
        ),
    ]
    await store.index_chunks(chunks)

    # Filter for Security
    res = await store.search(query="Firewall", filters={"department": "Security"})
    assert len(res) == 1
    assert res[0].chunk_id == "c1"


@pytest.mark.asyncio
async def test_bm25_store_delete_by_document_id() -> None:
    """Verify purging chunks by parent document ID."""
    store = InMemoryBM25Store()

    chunks = [
        DocumentChunk(
            id="c1",
            document_id="doc_to_delete",
            content="Confidential acquisition memo.",
            chunk_index=0,
        ),
        DocumentChunk(
            id="c2",
            document_id="doc_to_keep",
            content="Public company holiday schedule.",
            chunk_index=0,
        ),
    ]
    await store.index_chunks(chunks)

    deleted_count = await store.delete_by_document_id("doc_to_delete")
    assert deleted_count == 1

    res = await store.search(query="acquisition")
    assert len(res) == 0

    res_keep = await store.search(query="holiday")
    assert len(res_keep) == 1


@pytest.mark.asyncio
async def test_bm25_store_empty_behavior() -> None:
    """Verify querying empty store returns empty list gracefully."""
    store = InMemoryBM25Store()
    results = await store.search(query="anything")
    assert results == []
