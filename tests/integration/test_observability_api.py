"""Integration tests for Observability and Distributed Tracing REST API endpoints."""

import pytest
from starlette.testclient import TestClient

from enterprise_agent.api.deps import get_tracer, reset_observability_service
from enterprise_agent.config.settings import Settings
from enterprise_agent.main import create_application
from enterprise_agent.schemas.observability import SpanKind


@pytest.fixture
def client() -> TestClient:
    """Provide TestClient with freshly initialized test application."""
    reset_observability_service()
    settings = Settings(
        app_env="testing",
        llm_provider="mock",
        embedding_provider="mock",
        embedding_dimensions=16,
        qdrant_url=":memory:",
        experiments_db_path=":memory:",
        observability_enabled=True,
        observability_exporter="memory",
    )
    app = create_application(settings)
    return TestClient(app)


def test_api_observability_crud_lifecycle(client: TestClient) -> None:
    """Verify listing traces, fetching trace by ID, retrieving stats, and clearing traces."""
    # 1. Initially empty
    list_resp = client.get("/api/v1/observability/traces")
    assert list_resp.status_code == 200
    assert list_resp.json() == []

    # 2. Stats initial
    stats_resp = client.get("/api/v1/observability/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["total_traces"] == 0
    assert stats["active_exporter"] == "memory"

    # 3. Generate a trace using the active tracer
    tracer = client.app.dependency_overrides.get(get_tracer, None)  # type: ignore[attr-defined]
    if tracer:
        active_tracer = tracer()
    else:
        from enterprise_agent.api.deps import get_observability_service

        active_tracer = get_observability_service().tracer

    with active_tracer.span("api.integration.test", kind=SpanKind.SERVER) as span:
        span.set_attribute("integration.client", "testclient")
        span.set_genai_metrics(model="mock-model", prompt_tokens=15, completion_tokens=10)

    # 4. List traces now
    list_resp = client.get("/api/v1/observability/traces")
    assert list_resp.status_code == 200
    traces = list_resp.json()
    assert len(traces) == 1
    trace_id = traces[0]["trace_id"]
    assert traces[0]["root_span_name"] == "api.integration.test"
    assert traces[0]["total_tokens"] == 25

    # 5. Get trace by ID
    get_resp = client.get(f"/api/v1/observability/traces/{trace_id}")
    assert get_resp.status_code == 200
    detail = get_resp.json()
    assert detail["trace_id"] == trace_id
    assert len(detail["spans"]) == 1
    assert detail["spans"][0]["attributes"]["integration.client"] == "testclient"

    # 6. Non-existent trace ID
    unknown_resp = client.get("/api/v1/observability/traces/nonexistent-id")
    assert unknown_resp.status_code == 404

    # 7. Clear traces
    del_resp = client.delete("/api/v1/observability/traces")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "cleared"

    # 8. Verify cleared
    list_after = client.get("/api/v1/observability/traces")
    assert list_after.json() == []


def test_api_rag_trace_generation(client: TestClient) -> None:
    """Verify that executing a RAG query automatically records distributed spans."""
    # 1. Ingest a sample document first
    ingest_resp = client.post(
        "/api/v1/documents/ingest/text",
        json={
            "title": "Observability Policy",
            "content": "All microservices must emit OpenTelemetry spans with W3C propagation.",
            "auto_index": True,
        },
    )
    assert ingest_resp.status_code == 201

    # 2. Execute RAG query
    rag_resp = client.post(
        "/api/v1/rag/query",
        json={
            "query": "What standard must microservices emit?",
            "top_k": 3,
        },
    )
    assert rag_resp.status_code == 200

    # 3. Query observability traces
    traces_resp = client.get("/api/v1/observability/traces?root_name=rag.query")
    assert traces_resp.status_code == 200
    traces = traces_resp.json()
    assert len(traces) >= 1

    rag_trace = traces[0]
    assert rag_trace["root_span_name"] == "rag.query"
    span_names = {s["name"] for s in rag_trace["spans"]}
    assert "rag.query" in span_names
    assert "rag.retrieval" in span_names
    assert "rag.synthesis" in span_names
