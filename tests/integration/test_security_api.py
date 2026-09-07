"""Integration tests for Security API endpoints (/api/v1/security)."""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from enterprise_agent.api.deps import (
    get_api_key_manager,
    get_api_key_store,
    reset_security_services,
)
from enterprise_agent.config.settings import Settings
from enterprise_agent.main import create_application
from enterprise_agent.security.manager import APIKeyManager
from enterprise_agent.security.store import SQLiteAPIKeyStore


@pytest.fixture
def secured_client() -> Generator[TestClient, None, None]:
    """TestClient with api_security_enabled=True and in-memory key storage."""
    reset_security_services()
    settings = Settings(
        app_env="testing",
        llm_provider="mock",
        embedding_provider="mock",
        qdrant_url=":memory:",
        api_security_enabled=True,
        security_master_key=SecretStr("super-secret-enterprise-master-key"),
        security_api_keys_db_path=":memory:",
    )
    app = create_application(settings)

    # In-memory store for isolation
    store = SQLiteAPIKeyStore(db_path=":memory:")
    manager = APIKeyManager(store=store, default_expiry_days=90)

    app.dependency_overrides[get_api_key_store] = lambda: store
    app.dependency_overrides[get_api_key_manager] = lambda: manager

    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()
        reset_security_services()
        store.close()


@pytest.fixture
def open_client() -> Generator[TestClient, None, None]:
    """TestClient with api_security_enabled=False (dev/test default)."""
    reset_security_services()
    settings = Settings(
        app_env="testing",
        llm_provider="mock",
        embedding_provider="mock",
        qdrant_url=":memory:",
        api_security_enabled=False,
        security_api_keys_db_path=":memory:",
    )
    app = create_application(settings)

    store = SQLiteAPIKeyStore(db_path=":memory:")
    manager = APIKeyManager(store=store)

    app.dependency_overrides[get_api_key_store] = lambda: store
    app.dependency_overrides[get_api_key_manager] = lambda: manager

    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()
        reset_security_services()
        store.close()


def test_security_api_open_mode(open_client: TestClient) -> None:
    """Verify endpoints function without headers when api_security_enabled=False."""
    # Create key without auth headers
    create_resp = open_client.post(
        "/api/v1/security/keys",
        json={
            "name": "developer-local-key",
            "scopes": ["rag:read"],
            "description": "Local dev testing key",
        },
    )
    assert create_resp.status_code == 201
    data = create_resp.json()
    assert "raw_key" in data
    assert data["key"]["name"] == "developer-local-key"
    assert data["key"]["status"] == "active"

    # List keys without auth headers
    list_resp = open_client.get("/api/v1/security/keys")
    assert list_resp.status_code == 200
    assert list_resp.json()["total"] == 1


def test_security_api_auth_required(secured_client: TestClient) -> None:
    """Verify 401 Unauthorized when security is enabled and credentials are omitted."""
    resp = secured_client.get("/api/v1/security/keys")
    assert resp.status_code == 401
    assert "Authentication required" in resp.json()["detail"]
    assert resp.headers.get("www-authenticate") == "ApiKey"


def test_security_api_invalid_key_401(secured_client: TestClient) -> None:
    """Verify 401 Unauthorized when invalid API key is provided."""
    headers = {"X-API-Key": "ea_invalid_fake_key_12345"}
    resp = secured_client.get("/api/v1/security/keys", headers=headers)
    assert resp.status_code == 401
    assert "invalid" in resp.json()["detail"].lower()


def test_security_api_master_key_provisioning(secured_client: TestClient) -> None:
    """Verify system master key can provision and list API keys."""
    master_headers = {"X-API-Key": "super-secret-enterprise-master-key"}

    # Provision key via master key
    create_resp = secured_client.post(
        "/api/v1/security/keys",
        json={
            "name": "production-rag-client",
            "scopes": ["rag:read"],
            "expires_in_days": 60,
        },
        headers=master_headers,
    )
    assert create_resp.status_code == 201
    data = create_resp.json()
    raw_rag_key = data["raw_key"]
    key_id = data["key"]["key_id"]

    assert raw_rag_key.startswith("ea_")
    assert data["key"]["name"] == "production-rag-client"

    # List keys using Authorization: Bearer <master_key>
    bearer_headers = {"Authorization": "Bearer super-secret-enterprise-master-key"}
    list_resp = secured_client.get("/api/v1/security/keys", headers=bearer_headers)
    assert list_resp.status_code == 200
    assert list_resp.json()["total"] == 1

    # Verify RAG key lacks ADMIN scope and cannot manage keys (403 Forbidden)
    rag_headers = {"X-API-Key": raw_rag_key}
    denied_resp = secured_client.get("/api/v1/security/keys", headers=rag_headers)
    assert denied_resp.status_code == 403
    assert "lacks required operational scope" in denied_resp.json()["detail"]

    # Provision an admin key using master key
    admin_create = secured_client.post(
        "/api/v1/security/keys",
        json={
            "name": "secops-admin",
            "scopes": ["admin"],
        },
        headers=master_headers,
    )
    assert admin_create.status_code == 201
    raw_admin_key = admin_create.json()["raw_key"]
    admin_headers = {"X-API-Key": raw_admin_key}

    # Rotate RAG key using newly provisioned admin key
    rotate_resp = secured_client.post(
        f"/api/v1/security/keys/{key_id}/rotate",
        json={"grace_period_hours": 48},
        headers=admin_headers,
    )
    assert rotate_resp.status_code == 200
    new_rag_key = rotate_resp.json()["new_raw_key"]
    assert new_rag_key != raw_rag_key

    # Revoke RAG key
    revoke_resp = secured_client.delete(
        f"/api/v1/security/keys/{key_id}",
        headers=admin_headers,
    )
    assert revoke_resp.status_code == 200
    assert revoke_resp.json()["revoked"] is True
