"""Unit tests for synthetic enterprise knowledge base and seeding pipeline."""

from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from enterprise_agent.config.settings import Settings
from enterprise_agent.embeddings.mock import MockEmbeddingProvider
from enterprise_agent.embeddings.service import EmbeddingsService
from enterprise_agent.ingestion.service import IngestionService
from enterprise_agent.sparse.bm25 import InMemoryBM25Store
from enterprise_agent.vectorstore.qdrant import QdrantVectorStore
from enterprise_agent.vectorstore.service import VectorSearchService
from scripts.seed_knowledge_base import extract_title_from_markdown, seed_knowledge_base

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "knowledge_base"


def test_knowledge_base_documents_exist_and_valid() -> None:
    """Verify all synthetic documents exist and contain substantive enterprise content."""
    assert DATA_DIR.is_dir(), "data/knowledge_base directory must exist."

    expected_departments = [
        "hr",
        "engineering",
        "legal_compliance",
        "finance",
        "customer_support",
    ]
    for dept in expected_departments:
        dept_dir = DATA_DIR / dept
        assert dept_dir.is_dir(), f"Department directory '{dept}' must exist."
        dept_files = list(dept_dir.glob("*.md"))
        assert len(dept_files) >= 2, (
            f"Department '{dept}' must contain at least 2 policy documents."
        )

    all_docs = list(DATA_DIR.rglob("*.md"))
    assert len(all_docs) >= 10, "Knowledge base must contain at least 10 documents."

    for doc_path in all_docs:
        content = doc_path.read_text(encoding="utf-8")
        assert content.strip(), f"Document '{doc_path.name}' must not be empty."
        words = content.split()
        assert len(words) >= 150, (
            f"Document '{doc_path.name}' must have >= 150 words (found {len(words)})."
        )
        assert "# " in content, f"Document '{doc_path.name}' must have an H1 header."


def test_extract_title_from_markdown() -> None:
    """Verify title extraction from markdown headers and fallback behavior."""
    md_with_h1 = "# Global Travel Policy\n\nSome introductory content."
    assert extract_title_from_markdown(md_with_h1, "Default") == "Global Travel Policy"

    md_with_spaces = "\n  #   Engineering Incident Runbook  \n\nText."
    assert extract_title_from_markdown(md_with_spaces, "Default") == "Engineering Incident Runbook"

    md_without_h1 = "## Section Header\n\nNo top level header."
    assert extract_title_from_markdown(md_without_h1, "Fallback Title") == "Fallback Title"


@pytest.mark.asyncio
async def test_seed_knowledge_base_dry_run() -> None:
    """Verify dry-run mode chunks all documents without vector store writes."""
    ingestion_service = IngestionService(chunk_size=600, chunk_overlap=100)

    summary = await seed_knowledge_base(
        data_dir=DATA_DIR,
        dry_run=True,
        ingestion_service=ingestion_service,
    )

    assert summary["documents_ingested"] >= 10
    assert summary["chunks_produced"] >= 10
    assert summary["vectors_indexed"] == 0
    assert summary["dry_run"] is True
    assert len(summary["departments"]) >= 5
    assert "hr" in summary["departments"]
    assert "engineering" in summary["departments"]
    assert "finance" in summary["departments"]


@pytest.mark.asyncio
async def test_seed_knowledge_base_indexing_and_search() -> None:
    """Verify automated seeding indexes chunks into vector and sparse stores."""
    settings = Settings(
        app_env="testing",
        embedding_provider="mock",
        embedding_dimensions=16,
        qdrant_url=":memory:",
    )

    client = AsyncQdrantClient(location=":memory:")
    vector_store = QdrantVectorStore(client=client, collection_name="test_seeded_kb")
    embedding_provider = MockEmbeddingProvider(dimensions=16)
    embeddings_service = EmbeddingsService(provider=embedding_provider)
    vector_service = VectorSearchService(
        vector_store=vector_store,
        embeddings_service=embeddings_service,
    )
    sparse_store = InMemoryBM25Store()
    ingestion_service = IngestionService(chunk_size=700, chunk_overlap=120)

    summary = await seed_knowledge_base(
        data_dir=DATA_DIR,
        settings=settings,
        dry_run=False,
        include_sparse=True,
        vector_service=vector_service,
        sparse_store=sparse_store,
        ingestion_service=ingestion_service,
    )

    assert summary["documents_ingested"] >= 10
    assert summary["chunks_produced"] > 0
    assert summary["vectors_indexed"] == summary["chunks_produced"]

    # Verify search against seeded sparse store
    bm25_results = await sparse_store.search("parental leave 16 weeks", top_k=3)
    assert len(bm25_results) > 0
    top_bm25 = bm25_results[0]
    assert "parental leave" in top_bm25.content.lower()
    assert top_bm25.metadata.get("department") == "hr"

    # Verify search for SOC2 audit retention
    soc2_results = await sparse_store.search("WORM audit log retention 7 years", top_k=3)
    assert len(soc2_results) > 0
    top_soc2 = soc2_results[0]
    assert "retention" in top_soc2.content.lower() or "audit" in top_soc2.content.lower()
    assert top_soc2.metadata.get("department") == "legal_compliance"
