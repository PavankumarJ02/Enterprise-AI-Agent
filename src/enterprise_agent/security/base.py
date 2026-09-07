"""Base models, enumerations, and abstract interfaces for security subsystem."""

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class APIKeyScope(StrEnum):
    """Granular role-based capability scopes for API keys."""

    RAG_READ = "rag:read"
    SQL_QUERY = "sql:query"
    AGENT_EXECUTE = "agent:execute"
    EXPERIMENTS_WRITE = "experiments:write"
    OBSERVABILITY_READ = "observability:read"
    ADMIN = "admin"


class APIKeyStatus(StrEnum):
    """Operational status flags for API keys."""

    ACTIVE = "active"
    ROTATING = "rotating"
    REVOKED = "revoked"
    EXPIRED = "expired"


class APIKeyMetadata(BaseModel):
    """Public metadata record representing an API key without exposing secret material."""

    model_config = ConfigDict(from_attributes=True)

    key_id: str = Field(description="Unique identifier for the API key (UUID4).")
    name: str = Field(description="Human-readable identifier or service name.")
    key_hash: str = Field(description="Cryptographic SHA-256 hash of the secret key token.")
    prefix: str = Field(description="Truncated key prefix (e.g. 'ea_a1b2...') for safe display.")
    scopes: list[APIKeyScope] = Field(
        default_factory=lambda: [APIKeyScope.RAG_READ],
        description="List of authorized operational scopes.",
    )
    status: APIKeyStatus = Field(
        default=APIKeyStatus.ACTIVE,
        description="Current key lifecycle status.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the key was provisioned.",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Timestamp when the key naturally expires.",
    )
    rotated_at: datetime | None = Field(
        default=None,
        description="Timestamp when rotation was initiated.",
    )
    grace_expires_at: datetime | None = Field(
        default=None,
        description="Timestamp when transitional grace period expires for a rotated key.",
    )
    description: str = Field(
        default="",
        description="Optional description or operational notes.",
    )


class APIKeyValidationResult(BaseModel):
    """Structured result of an API key authentication and authorization check."""

    is_valid: bool = Field(description="Whether authentication and scope checks passed.")
    key_metadata: APIKeyMetadata | None = Field(
        default=None,
        description="Metadata of the authenticated key if valid.",
    )
    error_code: str | None = Field(
        default=None,
        description="Specific machine-readable failure reason code if invalid.",
    )
    error_message: str | None = Field(
        default=None,
        description="Human-readable explanation of rejection.",
    )


class SecretItem(BaseModel):
    """Structured representation of an enterprise secret parameter."""

    key: str = Field(description="Name or key of the secret.")
    value: SecretStr = Field(description="Protected secret value.")
    version: str | None = Field(
        default=None,
        description="Secret version identifier if applicable.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific metadata.",
    )


class APIKeyStore(ABC):
    """Abstract storage backend for API key metadata records."""

    @abstractmethod
    async def save_key(self, key_meta: APIKeyMetadata) -> None:
        """Persist a new API key record."""
        pass

    @abstractmethod
    async def get_key_by_hash(self, key_hash: str) -> APIKeyMetadata | None:
        """Retrieve key metadata matching the provided cryptographic hash."""
        pass

    @abstractmethod
    async def get_key_by_id(self, key_id: str) -> APIKeyMetadata | None:
        """Retrieve key metadata matching the unique key identifier."""
        pass

    @abstractmethod
    async def list_keys(self) -> list[APIKeyMetadata]:
        """List all registered API key records."""
        pass

    @abstractmethod
    async def update_key(self, key_meta: APIKeyMetadata) -> None:
        """Update an existing API key record."""
        pass

    @abstractmethod
    async def delete_key(self, key_id: str) -> bool:
        """Delete an API key record by ID."""
        pass


class SecretsManager(ABC):
    """Abstract interface for managing and resolving enterprise application secrets."""

    @abstractmethod
    def get_secret(self, key: str, default: str | None = None) -> SecretStr | None:
        """Retrieve a sensitive secret by key name."""
        pass

    @abstractmethod
    def has_secret(self, key: str) -> bool:
        """Check if a secret is configured and non-empty."""
        pass
