"""Unit tests for CorrelationIdMiddleware and context propagation."""

from fastapi import FastAPI
from starlette.testclient import TestClient

from enterprise_agent.middleware.correlation import (
    CORRELATION_HEADER,
    REQUEST_ID_HEADER,
    CorrelationIdMiddleware,
    get_correlation_id,
    set_correlation_id,
)


def test_correlation_id_context_vars() -> None:
    """Verify context variable get and set behavior."""
    assert get_correlation_id() is None
    token = set_correlation_id("test-corr-id")
    assert get_correlation_id() == "test-corr-id"
    from enterprise_agent.middleware.correlation import reset_correlation_id

    reset_correlation_id(token)
    assert get_correlation_id() is None


def test_correlation_middleware_generates_new_id() -> None:
    """Verify middleware generates a new 32-character hex ID when none is sent."""
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/test")
    async def endpoint() -> dict[str, str | None]:
        return {"cid": get_correlation_id()}

    client = TestClient(app)
    response = client.get("/test")
    assert response.status_code == 200

    cid_header = response.headers.get(CORRELATION_HEADER)
    assert cid_header is not None
    assert len(cid_header) == 32
    assert response.json()["cid"] == cid_header


def test_correlation_middleware_propagates_existing_id() -> None:
    """Verify middleware preserves and echoes caller-provided correlation ID."""
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/test")
    async def endpoint() -> dict[str, str | None]:
        return {"cid": get_correlation_id()}

    client = TestClient(app)
    # Using X-Correlation-ID
    resp1 = client.get("/test", headers={CORRELATION_HEADER: "custom-corr-123"})
    assert resp1.status_code == 200
    assert resp1.headers.get(CORRELATION_HEADER) == "custom-corr-123"
    assert resp1.json()["cid"] == "custom-corr-123"

    # Using X-Request-ID fallback
    resp2 = client.get("/test", headers={REQUEST_ID_HEADER: "req-id-456"})
    assert resp2.status_code == 200
    assert resp2.headers.get(CORRELATION_HEADER) == "req-id-456"
    assert resp2.json()["cid"] == "req-id-456"
