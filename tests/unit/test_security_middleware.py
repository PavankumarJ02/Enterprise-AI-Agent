"""Unit tests for SecurityHeadersMiddleware."""

from fastapi import FastAPI
from starlette.testclient import TestClient

from enterprise_agent.middleware.security import SecurityHeadersMiddleware


def test_security_headers_injected_when_enabled() -> None:
    """Verify standard security headers are attached to responses."""
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware, enabled=True)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"message": "pong"}

    client = TestClient(app)
    resp = client.get("/ping")
    assert resp.status_code == 200

    headers = resp.headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "geolocation=()" in headers["Permissions-Policy"]
    assert headers["Content-Security-Policy"] == "default-src 'self'"


def test_security_headers_disabled() -> None:
    """Verify security headers are not injected when disabled."""
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware, enabled=False)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"message": "pong"}

    client = TestClient(app)
    resp = client.get("/ping")
    assert resp.status_code == 200

    assert "X-Content-Type-Options" not in resp.headers
    assert "X-Frame-Options" not in resp.headers


def test_csp_relaxed_for_docs() -> None:
    """Verify Content-Security-Policy is relaxed for documentation paths."""
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware, enabled=True)

    @app.get("/docs")
    async def docs() -> dict[str, str]:
        return {"docs": "swagger"}

    client = TestClient(app)
    resp = client.get("/docs")
    assert resp.status_code == 200
    assert "Content-Security-Policy" not in resp.headers
