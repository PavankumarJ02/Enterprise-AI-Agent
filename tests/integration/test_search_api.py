from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from enterprise_agent.api.deps import (
    get_embeddings,
    get_ingestion_service,
    get_vector_store,
    reset_vector_store,
)
from enterprise_agent.config.settings import Settings
from enterprise_agent.embeddings.factory import get_embeddings_service
from enterprise_agent.ingestion.service import IngestionService
from enterprise_agent.main import create_application
from enterprise_agent.vectorstore.factory import create_vector_store


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """Create test client with isolated in-memory stores."""
    settings = Settings(
        app_env="testing",
        llm_provider="mock",
        embedding_provider="mock",
        embedding_dimensions=16,
        qdrant_url=":memory:",
        qdrant_collection_name="test_api_knowledge",
    )
    app = create_application(settings)

    # Isolated services for each test
    ingestion_service = IngestionService()
    embeddings_service = get_embeddings_service(settings)
    vector_store = create_vector_store(settings)

    app.dependency_overrides[get_ingestion_service] = lambda: ingestion_service
    app.dependency_overrides[get_embeddings] = lambda: embeddings_service
    app.dependency_overrides[get_vector_store] = lambda: vector_store

    reset_vector_store()

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


def test_semantic_search_empty_store(test_client: TestClient) -> None:
    """Semantic search against an empty store returns 0 results gracefully."""
    response = test_client.post(
        "/api/v1/search/semantic",
        json={"query": "What is our vacation policy?", "top_k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is our vacation policy?"
    assert data["total_results"] == 0
    assert data["results"] == []


def test_ingest_auto_index_and_semantic_search(test_client: TestClient) -> None:
    """Ingest document with auto-indexing and verify semantic retrieval."""
    content = (
        "Enterprise Annual Leave Policy: All full-time staff members are entitled to "
        "25 paid vacation days per calendar year. Leave requests must be submitted "
        "via the HR portal at least two weeks in advance."
    )
    ingest_res = test_client.post(
        "/api/v1/documents/ingest/text?auto_index=true",
        json={
            "title": "Vacation Policy 2026",
            "content": content,
            "source": "hr_portal_leave.txt",
            "metadata": {"department": "HR", "category": "policy"},
        },
    )
    assert ingest_res.status_code == 201
    doc_id = ingest_res.json()["document_id"]

    # Now execute semantic search
    search_res = test_client.post(
        "/api/v1/search/semantic",
        json={
            "query": "How many days of paid vacation do employees receive?",
            "top_k": 3,
        },
    )
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert search_data["total_results"] >= 1

    top_match = search_data["results"][0]
    assert top_match["document_id"] == doc_id
    assert "Enterprise Annual Leave Policy" in top_match["content"]
    assert "score" in top_match
    assert top_match["metadata"]["department"] == "HR"


def test_search_with_metadata_filter(test_client: TestClient) -> None:
    """Filter search results using payload metadata."""
    # Ingest Doc A (Engineering)
    test_client.post(
        "/api/v1/documents/ingest/text?auto_index=true",
        json={
            "title": "Engineering Standards",
            "content": "Python 3.11 and FastAPI are mandatory for all microservices.",
            "source": "eng_guidelines.txt",
            "metadata": {"team": "Engineering"},
        },
    )

    # Ingest Doc B (Finance)
    res_b = test_client.post(
        "/api/v1/documents/ingest/text?auto_index=true",
        json={
            "title": "Expense Policy",
            "content": "Meal expenses up to 50 dollars per day are reimbursed with receipt.",
            "source": "finance_rules.txt",
            "metadata": {"team": "Finance"},
        },
    )
    doc_b_id = res_b.json()["document_id"]

    # Search filtering by document_id == doc_b_id
    filtered_res = test_client.post(
        "/api/v1/search/semantic",
        json={
            "query": "What are the rules and guidelines?",
            "top_k": 5,
            "filters": {"document_id": doc_b_id},
        },
    )
    assert filtered_res.status_code == 200
    results = filtered_res.json()["results"]
    assert len(results) == 1
    assert results[0]["document_id"] == doc_b_id
    assert "Expense Policy" in results[0]["content"] or "Meal expenses" in results[0]["content"]


def test_manual_index_document_endpoint(test_client: TestClient) -> None:
    """Ingest without auto-indexing, then manually trigger index endpoint."""
    ingest_res = test_client.post(
        "/api/v1/documents/ingest/text?auto_index=false",
        json={
            "title": "Manual Indexing Test",
            "content": "Secret knowledge only to be indexed on demand.",
            "source": "manual.txt",
        },
    )
    assert ingest_res.status_code == 201
    doc_id = ingest_res.json()["document_id"]

    # Search should initially return empty since auto_index was false
    search_before = test_client.post(
        "/api/v1/search/semantic",
        json={"query": "Secret knowledge", "filters": {"document_id": doc_id}},
    )
    assert search_before.json()["total_results"] == 0

    # Explicitly trigger indexing
    idx_res = test_client.post(f"/api/v1/documents/{doc_id}/index")
    assert idx_res.status_code == 200
    idx_data = idx_res.json()
    assert idx_data["document_id"] == doc_id
    assert idx_data["chunks_indexed"] >= 1
    assert idx_data["status"] == "success"

    # Search should now find the indexed chunk
    search_after = test_client.post(
        "/api/v1/search/semantic",
        json={"query": "Secret knowledge", "filters": {"document_id": doc_id}},
    )
    assert search_after.json()["total_results"] >= 1


def test_delete_document_purges_vectors(test_client: TestClient) -> None:
    """Deleting a document purges both document record and its vector points."""
    ingest_res = test_client.post(
        "/api/v1/documents/ingest/text?auto_index=true",
        json={
            "title": "Ephemeral Document",
            "content": "This document will be deleted immediately after indexing.",
            "source": "temp.txt",
        },
    )
    doc_id = ingest_res.json()["document_id"]

    # Verify search finds it
    search_before = test_client.post(
        "/api/v1/search/semantic",
        json={"query": "Ephemeral Document", "filters": {"document_id": doc_id}},
    )
    assert search_before.json()["total_results"] >= 1

    # Delete document
    del_res = test_client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # Search again with filter -> 0 results
    search_after = test_client.post(
        "/api/v1/search/semantic",
        json={"query": "Ephemeral Document", "filters": {"document_id": doc_id}},
    )
    assert search_after.json()["total_results"] == 0

    # Deleting non-existent document returns 404
    del_again = test_client.delete(f"/api/v1/documents/{doc_id}")
    assert del_again.status_code == 404


def test_semantic_search_validation_errors(test_client: TestClient) -> None:
    """Empty query or invalid top_k triggers validation error."""
    # Empty query
    res = test_client.post("/api/v1/search/semantic", json={"query": ""})
    assert res.status_code == 422

    # Negative top_k
    res2 = test_client.post("/api/v1/search/semantic", json={"query": "hello", "top_k": -1})
    assert res2.status_code == 422
