"""Unit tests for recursive character chunking and IngestionService."""

import pytest

from enterprise_agent.ingestion.chunking.recursive import RecursiveCharacterChunker
from enterprise_agent.ingestion.models import Document
from enterprise_agent.ingestion.service import IngestionService


def test_chunker_overlap_validation() -> None:
    """Ensure chunk_overlap >= chunk_size raises ValueError."""
    with pytest.raises(ValueError):
        RecursiveCharacterChunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError):
        RecursiveCharacterChunker(chunk_size=100, chunk_overlap=150)


def test_recursive_chunker_basic() -> None:
    """Verify recursive chunker splits long text and attaches metadata."""
    chunker = RecursiveCharacterChunker(chunk_size=120, chunk_overlap=30)
    text = (
        "Section 1: Introduction to company values.\n\n"
        "Section 2: Security protocols. Passwords must have 12 characters.\n\n"
        "Section 3: Remote work policies. Employees can work from home on Fridays.\n\n"
        "Section 4: Expense reports must be submitted before month end."
    )
    doc = Document(
        title="Employee Handbook",
        content=text,
        source="handbook.md",
    )

    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.document_id == doc.id
        assert c.metadata["title"] == "Employee Handbook"
        assert c.metadata["source"] == "handbook.md"
        assert c.start_char >= 0
        assert c.end_char > c.start_char
        assert c.token_count > 0
        assert len(c.chunk_hash) == 64  # SHA-256 length


def test_recursive_chunker_page_mapping() -> None:
    """Verify chunker maps page boundaries into chunk metadata."""
    chunker = RecursiveCharacterChunker(chunk_size=80, chunk_overlap=10)
    page_1 = "This is the content on page one. It contains preliminary definitions."
    page_2 = "This is the content on page two. It contains strict compliance rules."

    full_text = f"{page_1}\n\n{page_2}"
    doc = Document(title="Compliance Guide", content=full_text, source="compliance.pdf")

    pages = [(1, page_1), (2, page_2)]
    chunks = chunker.chunk(doc, pages=pages)

    assert len(chunks) >= 2
    assert chunks[0].metadata["page_number"] == 1
    assert chunks[-1].metadata["page_number"] == 2


@pytest.mark.asyncio
async def test_ingestion_service_deduplication() -> None:
    """Verify IngestionService skips duplicate ingestions by checksum."""
    service = IngestionService(chunk_size=200, chunk_overlap=50)
    content = "Company travel reimbursement is capped at $150 per diem for meals."

    # 1. First ingestion
    res1 = await service.ingest_text(text=content, title="Travel Policy", source="travel.txt")
    assert res1.status == "success"
    assert len(res1.chunks) >= 1

    # 2. Ingest identical content
    res2 = await service.ingest_text(
        text=content, title="Duplicate Travel Policy", source="travel.txt"
    )
    assert res2.status == "skipped_duplicate"
    assert res2.document.id == res1.document.id
    assert "already ingested" in res2.message

    # 3. Verify registry counts
    docs = service.list_documents()
    assert len(docs) == 1
    assert docs[0].title == "Travel Policy"
