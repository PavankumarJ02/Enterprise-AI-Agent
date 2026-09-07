"""Unit tests for security primitives, APIKeyManager lifecycle, storage, and telemetry redaction."""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from enterprise_agent.security.base import APIKeyMetadata, APIKeyScope, APIKeyStatus
from enterprise_agent.security.crypto import (
    constant_time_compare,
    extract_key_prefix,
    generate_api_key,
    hash_api_key,
)
from enterprise_agent.security.manager import APIKeyManager
from enterprise_agent.security.secrets import EnvSecretsManager, TelemetryRedactor
from enterprise_agent.security.store import InMemoryAPIKeyStore, SQLiteAPIKeyStore


def test_crypto_generate_api_key() -> None:
    """Verify generated API keys have custom prefix and sufficient CSPRNG entropy."""
    key1 = generate_api_key(prefix="ea_")
    key2 = generate_api_key(prefix="ea_")

    assert key1.startswith("ea_")
    assert key2.startswith("ea_")
    assert key1 != key2
    assert len(key1) >= 40


def test_crypto_hash_and_comparison() -> None:
    """Verify deterministic SHA-256 hashing and constant-time equality checks."""
    token = "ea_test_super_secret_token_12345"
    hash1 = hash_api_key(token)
    hash2 = hash_api_key(token)

    assert hash1 == hash2
    assert len(hash1) == 64
    assert constant_time_compare(token, token) is True
    assert constant_time_compare(token, "ea_wrong_token") is False


def test_crypto_extract_prefix() -> None:
    """Verify safe prefix extraction for operational logging and display."""
    token = "ea_abcdef1234567890"
    prefix = extract_key_prefix(token, visible_chars=8)
    assert prefix == "ea_abcde..."

    short = "ea_abc"
    assert extract_key_prefix(short, visible_chars=10) == "ea_abc"


@pytest.mark.asyncio
async def test_api_key_lifecycle_create_and_validate() -> None:
    """Verify provisioning and validating an active API key."""
    store = InMemoryAPIKeyStore()
    manager = APIKeyManager(store=store, default_expiry_days=30)

    raw_key, meta = await manager.create_key(
        name="finance-reporting-svc",
        scopes=[APIKeyScope.SQL_QUERY, APIKeyScope.RAG_READ],
        description="Automated SQL reporting pipeline",
    )

    assert raw_key.startswith("ea_")
    assert meta.name == "finance-reporting-svc"
    assert meta.status == APIKeyStatus.ACTIVE
    assert APIKeyScope.SQL_QUERY in meta.scopes

    # Validate with matching scope
    res = await manager.validate_key(raw_key, required_scope=APIKeyScope.SQL_QUERY)
    assert res.is_valid is True
    assert res.key_metadata is not None
    assert res.key_metadata.key_id == meta.key_id

    # Validate with None scope (any valid key)
    res_none = await manager.validate_key(raw_key, required_scope=None)
    assert res_none.is_valid is True


@pytest.mark.asyncio
async def test_api_key_validation_invalid_and_missing() -> None:
    """Verify proper error codes for invalid or missing tokens."""
    store = InMemoryAPIKeyStore()
    manager = APIKeyManager(store=store)

    # Missing / empty
    empty_res = await manager.validate_key("")
    assert empty_res.is_valid is False
    assert empty_res.error_code == "MISSING_KEY"

    # Non-existent
    unknown_res = await manager.validate_key("ea_non_existent_key_token_xyz")
    assert unknown_res.is_valid is False
    assert unknown_res.error_code == "INVALID_KEY"


@pytest.mark.asyncio
async def test_api_key_rbac_scope_enforcement() -> None:
    """Verify granular scope authorization and admin superuser bypass."""
    store = InMemoryAPIKeyStore()
    manager = APIKeyManager(store=store)

    # Key with only RAG_READ
    rag_raw, _ = await manager.create_key(
        name="rag-only-client",
        scopes=[APIKeyScope.RAG_READ],
    )

    # Allowed
    rag_check = await manager.validate_key(rag_raw, required_scope=APIKeyScope.RAG_READ)
    assert rag_check.is_valid is True

    # Denied for SQL_QUERY
    sql_check = await manager.validate_key(rag_raw, required_scope=APIKeyScope.SQL_QUERY)
    assert sql_check.is_valid is False
    assert sql_check.error_code == "INSUFFICIENT_SCOPE"

    # Key with ADMIN scope should satisfy any required scope
    admin_raw, _ = await manager.create_key(
        name="system-admin",
        scopes=[APIKeyScope.ADMIN],
    )
    admin_sql = await manager.validate_key(admin_raw, required_scope=APIKeyScope.SQL_QUERY)
    admin_rag = await manager.validate_key(admin_raw, required_scope=APIKeyScope.RAG_READ)
    admin_agent = await manager.validate_key(admin_raw, required_scope=APIKeyScope.AGENT_EXECUTE)
    assert admin_sql.is_valid is True
    assert admin_rag.is_valid is True
    assert admin_agent.is_valid is True


@pytest.mark.asyncio
async def test_api_key_expiration() -> None:
    """Verify naturally expired keys are rejected."""
    store = InMemoryAPIKeyStore()
    manager = APIKeyManager(store=store)

    raw_key, meta = await manager.create_key(name="temp-key", scopes=[APIKeyScope.RAG_READ])

    # Artificially expire the key in the past
    meta.expires_at = datetime.now(UTC) - timedelta(seconds=10)
    await store.update_key(meta)

    res = await manager.validate_key(raw_key)
    assert res.is_valid is False
    assert res.error_code == "EXPIRED_KEY"


@pytest.mark.asyncio
async def test_api_key_rotation_with_grace_period() -> None:
    """Verify zero-downtime rotation where both keys function during the grace window."""
    store = InMemoryAPIKeyStore()
    manager = APIKeyManager(store=store, default_expiry_days=90)

    old_raw, old_meta = await manager.create_key(
        name="payment-gateway",
        scopes=[APIKeyScope.AGENT_EXECUTE],
    )

    # Initiate rotation with a 2-hour grace period
    new_raw, new_meta = await manager.rotate_key(old_meta.key_id, grace_period_hours=2)

    assert new_raw != old_raw
    assert new_meta.key_id != old_meta.key_id
    assert new_meta.status == APIKeyStatus.ACTIVE

    # Both old and new keys must validate during grace period
    old_valid = await manager.validate_key(old_raw, required_scope=APIKeyScope.AGENT_EXECUTE)
    new_valid = await manager.validate_key(new_raw, required_scope=APIKeyScope.AGENT_EXECUTE)
    assert old_valid.is_valid is True
    assert new_valid.is_valid is True

    # Simulate expiration of grace window on old key
    updated_old = await store.get_key_by_id(old_meta.key_id)
    assert updated_old is not None
    assert updated_old.status == APIKeyStatus.ROTATING
    updated_old.grace_expires_at = datetime.now(UTC) - timedelta(seconds=5)
    await store.update_key(updated_old)

    # Old key is now rejected
    old_res = await manager.validate_key(old_raw)
    assert old_res.is_valid is False
    assert old_res.error_code == "GRACE_PERIOD_EXPIRED"

    # Replacement key continues to validate
    new_res = await manager.validate_key(new_raw)
    assert new_res.is_valid is True


@pytest.mark.asyncio
async def test_api_key_revocation() -> None:
    """Verify immediate revocation of credentials."""
    store = InMemoryAPIKeyStore()
    manager = APIKeyManager(store=store)

    raw_key, meta = await manager.create_key(name="compromised-worker")
    assert (await manager.validate_key(raw_key)).is_valid is True

    # Revoke
    revoked = await manager.revoke_key(meta.key_id)
    assert revoked is True

    # Validation must fail immediately
    res = await manager.validate_key(raw_key)
    assert res.is_valid is False
    assert res.error_code == "REVOKED_KEY"


@pytest.mark.asyncio
async def test_sqlite_api_key_store_persistence() -> None:
    """Verify SQLite store persists records and survives across connection instances."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "test_keys.db")

        # Session 1: Create and save
        store1 = SQLiteAPIKeyStore(db_path=db_path)
        meta = APIKeyMetadata(
            key_id="k-1001",
            name="analytics-bot",
            key_hash=hash_api_key("ea_sample_token_123"),
            prefix="ea_sampl...",
            scopes=[APIKeyScope.SQL_QUERY],
            status=APIKeyStatus.ACTIVE,
        )
        await store1.save_key(meta)
        store1.close()

        # Session 2: Read back from new store instance
        store2 = SQLiteAPIKeyStore(db_path=db_path)
        loaded = await store2.get_key_by_id("k-1001")
        assert loaded is not None
        assert loaded.name == "analytics-bot"
        assert loaded.scopes == [APIKeyScope.SQL_QUERY]

        by_hash = await store2.get_key_by_hash(meta.key_hash)
        assert by_hash is not None
        assert by_hash.key_id == "k-1001"

        all_keys = await store2.list_keys()
        assert len(all_keys) == 1

        # Delete
        deleted = await store2.delete_key("k-1001")
        assert deleted is True
        assert await store2.get_key_by_id("k-1001") is None
        store2.close()


def test_telemetry_redactor_text_sanitization() -> None:
    """Verify TelemetryRedactor masks all standard enterprise credential formats."""
    raw_log = (
        "Connecting to upstream LLM using sk-proj-1234567890abcdefghijklmn and "
        "internal token ea_prod_secret_token_1234567890. "
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz. "
        "GitHub token ghp_1234567890abcdefghijklmnopqrstuvwxyz and AWS AKIAIOSFODNN7EXAMPLE. "
        "Database url: postgresql://admin_user:super_secret_db_pass_123@db.prod.internal:5432/corp"
    )

    redacted = TelemetryRedactor.redact_text(raw_log)

    assert "sk-proj-1234567890abcdefghijklmn" not in redacted
    assert "[REDACTED_API_KEY]" in redacted

    assert "ea_prod_secret_token_1234567890" not in redacted
    assert "[REDACTED_AGENT_KEY]" in redacted

    assert "Bearer [REDACTED_BEARER_TOKEN]" in redacted
    assert "[REDACTED_GITHUB_TOKEN]" in redacted
    assert "[REDACTED_AWS_KEY]" in redacted
    assert "super_secret_db_pass_123" not in redacted
    assert "[REDACTED_PASSWORD]" in redacted


def test_telemetry_redactor_structured_data() -> None:
    """Verify recursive redaction over nested dictionaries and lists."""
    data = {
        "user": "alice",
        "api_key": "sk-1234567890123456789012345",
        "nested": {
            "token": "ea_internal_token_09876543210987654321",
            "safe_list": [
                "normal item",
                "sk-9999999999999999999999999",
            ],
        },
    }

    sanitized = TelemetryRedactor.redact_data(data)
    assert sanitized["api_key"] == "[REDACTED_SENSITIVE_VALUE]"
    assert sanitized["nested"]["token"] == "[REDACTED_SENSITIVE_VALUE]"
    assert sanitized["nested"]["safe_list"][0] == "normal item"
    assert sanitized["nested"]["safe_list"][1] == "[REDACTED_API_KEY]"


def test_env_secrets_manager() -> None:
    """Verify EnvSecretsManager resolution and existence checks."""
    manager = EnvSecretsManager(
        overrides={
            "OPENAI_API_KEY": "sk-test-mock-secret",
            "EMPTY_SECRET": "",
        }
    )

    assert manager.has_secret("OPENAI_API_KEY") is True
    assert manager.has_secret("EMPTY_SECRET") is False
    assert manager.has_secret("NON_EXISTENT") is False

    sec = manager.get_secret("OPENAI_API_KEY")
    assert sec is not None
    assert sec.get_secret_value() == "sk-test-mock-secret"
