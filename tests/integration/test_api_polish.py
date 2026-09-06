"""Integration tests for API Polish: Rate Limiting, Correlation ID, Security, and OpenAPI."""

from starlette.testclient import TestClient

from enterprise_agent.config.settings import Settings
from enterprise_agent.main import create_application


def test_api_rate_limiting_enforcement() -> None:
    """Verify that requests exceeding configured rate limit receive 429 Too Many Requests."""
    settings = Settings(
        app_env="testing",
        llm_provider="mock",
        embedding_provider="mock",
        embedding_dimensions=16,
        qdrant_url=":memory:",
        experiments_db_path=":memory:",
        rate_limit_enabled=True,
        rate_limit_requests_per_minute=3,
        rate_limit_burst_limit=5,
    )
    app = create_application(settings)
    client = TestClient(app)

    # 1. Health endpoint is exempt
    for _ in range(5):
        resp = client.get("/health")
        assert resp.status_code == 200

    # 2. First 3 requests to an API route should succeed
    for i in range(1, 4):
        resp = client.post(
            "/api/v1/chat",
            json={"query": "hello"},
            headers={"X-API-Key": "test-key-1"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-RateLimit-Limit") == "3"
        assert resp.headers.get("X-RateLimit-Remaining") == str(3 - i)
        assert resp.headers.get("X-RateLimit-Reset") is not None

    # 3. 4th request should be blocked with 429
    blocked_resp = client.post(
        "/api/v1/chat",
        json={"query": "hello"},
        headers={"X-API-Key": "test-key-1"},
    )
    assert blocked_resp.status_code == 429
    data = blocked_resp.json()
    assert data["error"] == "RATE_LIMIT_EXCEEDED"
    assert "Retry-After" in blocked_resp.headers
    assert blocked_resp.headers.get("X-RateLimit-Remaining") == "0"

    # 4. A different client key is not blocked
    other_resp = client.post(
        "/api/v1/chat",
        json={"query": "hello"},
        headers={"X-API-Key": "different-key"},
    )
    assert other_resp.status_code == 200


def test_api_correlation_and_security_headers() -> None:
    """Verify responses include Correlation ID and OWASP security headers."""
    settings = Settings(
        app_env="testing",
        llm_provider="mock",
        embedding_provider="mock",
        security_headers_enabled=True,
    )
    app = create_application(settings)
    client = TestClient(app)

    # Send with custom correlation ID
    resp = client.get("/health", headers={"X-Correlation-ID": "trace-corr-999"})
    assert resp.status_code == 200

    headers = resp.headers
    assert headers.get("X-Correlation-ID") == "trace-corr-999"
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-Frame-Options") == "DENY"
    assert headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"


def test_api_openapi_metadata() -> None:
    """Verify OpenAPI specification contains comprehensive metadata, tags, and contact."""
    settings = Settings(app_env="testing", llm_provider="mock")
    app = create_application(settings)
    client = TestClient(app)

    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()

    assert schema["info"]["title"] == settings.app_name
    assert "Enterprise AI Platform Team" in schema["info"]["contact"]["name"]
    assert "License" in schema["info"]["license"]["name"]

    tag_names = {t["name"] for t in schema.get("tags", [])}
    assert "Health" in tag_names
    assert "Retrieval-Augmented Generation" in tag_names
    assert "Autonomous Agent" in tag_names
    assert "Observability & Distributed Tracing" in tag_names
    assert "Experiment Tracking" in tag_names
