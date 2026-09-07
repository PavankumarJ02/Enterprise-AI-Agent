"""Pydantic request and response schemas for the security and API key management endpoints."""

from pydantic import BaseModel, Field

from enterprise_agent.security.base import APIKeyMetadata, APIKeyScope


class APIKeyCreateRequest(BaseModel):
    """Payload for provisioning a new enterprise API key."""

    name: str = Field(min_length=1, max_length=100, description="Service name or key owner.")
    scopes: list[APIKeyScope] = Field(
        default_factory=lambda: [APIKeyScope.RAG_READ],
        description="Authorized capability scopes for the key.",
    )
    expires_in_days: int | None = Field(
        default=None,
        ge=1,
        le=3650,
        description="Optional lifetime in days. If omitted, uses system default.",
    )
    description: str = Field(
        default="",
        max_length=500,
        description="Optional operational notes or justification.",
    )


class APIKeyCreateResponse(BaseModel):
    """Response returned upon API key provisioning containing the plaintext secret token."""

    raw_key: str = Field(
        description="Plaintext API key. Displayed ONLY ONCE; store in a secure secret manager."
    )
    key: APIKeyMetadata = Field(description="Public key metadata record.")


class APIKeyRotateRequest(BaseModel):
    """Payload for initiating an API key rotation with transitional grace period."""

    grace_period_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="Hours during which the prior key remains valid alongside the new key.",
    )


class APIKeyRotateResponse(BaseModel):
    """Response returned upon successful key rotation."""

    new_raw_key: str = Field(
        description="Plaintext replacement API key token. Displayed only once."
    )
    key: APIKeyMetadata = Field(description="Public metadata for the newly issued key.")


class APIKeyListResponse(BaseModel):
    """Paginated collection of registered API key metadata records."""

    keys: list[APIKeyMetadata] = Field(description="List of active and historical key records.")
    total: int = Field(description="Total count of records returned.")


class APIKeyRevokeResponse(BaseModel):
    """Confirmation payload for key revocation."""

    key_id: str = Field(description="Identifier of the revoked key.")
    revoked: bool = Field(description="Whether revocation succeeded.")
    message: str = Field(description="Human-readable status summary.")
