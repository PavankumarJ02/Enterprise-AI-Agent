"""Integration tests for Retrieval-Augmented Generation (RAG) REST and streaming endpoints."""

import json
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from enterprise_agent.api.deps import (
    get_embeddings,
    get_ingestion_service,
    get_llm,
    get_vector_store,
    reset_vector_store,
)
from enterprise_agent.config.settings import Settings
from enterprise_agent.embeddings.factory import get_embeddings_service
from enterprise_agent.ingestion.service import IngestionService
from enterprise_agent.llm.factory import get_llm_provider
from enterprise_agent.main import create_application
from enterprise_agent.rag.service import INSUFFICIENT_CONTEXT_MESSAGE
from enterprise_agent.vectorstore.factory import create_vector_store


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """Create test client with isolated in-memory stores and mock services."""
    settings = Settings(
        app_env="testing",
        llm_provider="mock",
        embedding_provider="mock",
        embedding_dimensions=16,
        qdrant_url=":memory:",
        qdrant_collection_name="test_rag_api_knowledge",
    )
    app = create_application(settings)

    # Isolated services for each test
    ingestion_service = IngestionService()
    embeddings_service = get_embeddings_service(settings)
    vector_store = create_vector_store(settings)
    llm_provider = get_llm_provider(settings)

    app.dependency_overrides[get_ingestion_service] = lambda: ingestion_service
    app.dependency_overrides[get_embeddings] = lambda: embeddings_service
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_llm] = lambda: llm_provider

    reset_vector_store()

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


def test_rag_query_with_ingested_knowledge(test_client: TestClient) -> None:
    """Ingest enterprise policy, query RAG endpoint, and verify grounded answer and citations."""
    policy_content = (
        "Enterprise Security Protocol: Multi-Factor Authentication (MFA) is strictly "
        "mandatory for all remote logins. Hardware security keys or authenticator apps "
        "are approved; SMS verification is strictly prohibited."
    )
    ingest_res = test_client.post(
        "/api/v1/documents/ingest/text?auto_index=true",
        json={
            "title": "Security Protocol 2026",
            "content": policy_content,
            "source": "security_handbook.pdf",
            "metadata": {"department": "InfoSec", "version": "4.2"},
        },
    )
    assert ingest_res.status_code == 201
    doc_id = ingest_res.json()["document_id"]

    # Execute RAG query
    response = test_client.post(
        "/api/v1/rag/query",
        json={
            "query": "Is SMS verification allowed for multi-factor authentication?",
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "Is SMS verification allowed for multi-factor authentication?"
    assert data["status"] == "success"
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["document_id"] == doc_id
    assert "SMS verification is strictly prohibited" in data["sources"][0]["content"]
    assert "model" in data and data["model"]
    assert data["latency_ms"] > 0
    assert "usage" in data
    assert "grounding" in data and data["grounding"] is not None


def test_rag_query_empty_knowledge_short_circuits(test_client: TestClient) -> None:
    """RAG query against an empty database returns insufficient context message without error."""
    response = test_client.post(
        "/api/v1/rag/query",
        json={
            "query": "What is our company policy on interstellar travel?",
            "top_k": 5,
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "insufficient_context"
    assert data["answer"] == INSUFFICIENT_CONTEXT_MESSAGE
    assert len(data["sources"]) == 0
    assert data["usage"]["total_tokens"] == 0
    assert data["grounding"]["grounding_status"] == "insufficient_context"


def test_rag_query_with_metadata_filter(test_client: TestClient) -> None:
    """Filter RAG retrieval so that only documents matching the metadata filter are cited."""
    # Doc 1: Legal
    test_client.post(
        "/api/v1/documents/ingest/text?auto_index=true",
        json={
            "title": "Legal Non-Disclosure",
            "content": "All NDAs remain in effect for 3 years post-termination.",
            "source": "legal_nda.txt",
            "metadata": {"department": "Legal"},
        },
    )

    # Doc 2: Engineering
    res_eng = test_client.post(
        "/api/v1/documents/ingest/text?auto_index=true",
        json={
            "title": "Git Branching Standards",
            "content": "All pull requests require two peer approvals before merging to main.",
            "source": "eng_git.txt",
            "metadata": {"department": "Engineering"},
        },
    )
    eng_id = res_eng.json()["document_id"]

    # Query with filter for Engineering
    response = test_client.post(
        "/api/v1/rag/query",
        json={
            "query": "What are the rules and requirements?",
            "top_k": 5,
            "filters": {"department": "Engineering"},
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert len(data["sources"]) == 1
    assert data["sources"][0]["document_id"] == eng_id
    assert "pull requests require two peer approvals" in data["sources"][0]["content"]


def test_rag_stream_endpoint(test_client: TestClient) -> None:
    """Verify streaming RAG response emits upfront sources, grounding evaluation, and [DONE]."""
    test_client.post(
        "/api/v1/documents/ingest/text?auto_index=true",
        json={
            "title": "Onboarding Guide",
            "content": "New hires receive a welcome kit on their first working day.",
            "source": "onboarding.md",
        },
    )

    response = test_client.post(
        "/api/v1/rag/stream",
        json={"query": "When do new hires receive their welcome kit?"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    body = response.text
    assert "event: sources" in body
    assert "event: grounding" in body
    assert "data: [DONE]" in body

    # Parse initial sources event
    lines = body.split("\n")
    sources_data = None
    for i, line in enumerate(lines):
        if line.startswith("event: sources") and i + 1 < len(lines):
            next_line = lines[i + 1]
            if next_line.startswith("data: "):
                sources_data = json.loads(next_line[len("data: ") :])
                break

    assert sources_data is not None
    assert len(sources_data) >= 1
    assert "welcome kit" in sources_data[0]["content"]


def test_rag_query_with_grounding_audit(test_client: TestClient) -> None:
    """Verify that grounding evaluation returns structured citation analysis."""
    test_client.post(
        "/api/v1/documents/ingest/text?auto_index=true",
        json={
            "title": "Travel Guidelines",
            "content": "Flight expenses must be submitted within 30 days of travel.",
            "source": "travel_guidelines.txt",
        },
    )

    response = test_client.post(
        "/api/v1/rag/query",
        json={
            "query": "What is the deadline for submitting flight expenses?",
            "verify_grounding": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "grounding" in data
    grounding = data["grounding"]
    assert "faithfulness_score" in grounding
    assert "grounding_status" in grounding
    assert "total_claims" in grounding
    assert "citations" in grounding
    assert isinstance(grounding["citations"], list)


def test_rag_query_validation_error(test_client: TestClient) -> None:
    """Invalid RAG requests return 422 Unprocessable Entity."""
    # Empty query string
    res = test_client.post("/api/v1/rag/query", json={"query": ""})
    assert res.status_code == 422

    # Negative top_k
    res2 = test_client.post("/api/v1/rag/query", json={"query": "test", "top_k": -5})
    assert res2.status_code == 422
