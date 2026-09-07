"""API key lifecycle manager coordinating creation, validation, rotation, and revocation."""

import uuid
from datetime import UTC, datetime, timedelta

from enterprise_agent.core.logging import get_logger
from enterprise_agent.security.base import (
    APIKeyMetadata,
    APIKeyScope,
    APIKeyStatus,
    APIKeyStore,
    APIKeyValidationResult,
)
from enterprise_agent.security.crypto import extract_key_prefix, generate_api_key, hash_api_key

logger = get_logger(__name__)


class APIKeyManager:
    """Manages API key lifecycle operations, RBAC scope validation, and rotation grace periods."""

    def __init__(
        self,
        store: APIKeyStore,
        default_expiry_days: int = 90,
    ) -> None:
        self.store = store
        self.default_expiry_days = default_expiry_days

    async def create_key(
        self,
        name: str,
        scopes: list[APIKeyScope] | None = None,
        expires_in_days: int | None = None,
        description: str = "",
    ) -> tuple[str, APIKeyMetadata]:
        """Issue a new API key with the specified operational scopes and expiration.

        Returns a tuple of (raw_secret_key, metadata). The raw secret key is only
        returned once upon creation and cannot be retrieved again.
        """
        raw_key = generate_api_key()
        key_hash = hash_api_key(raw_key)
        prefix = extract_key_prefix(raw_key)

        now = datetime.now(UTC)
        expiry_days = expires_in_days if expires_in_days is not None else self.default_expiry_days
        expires_at = now + timedelta(days=expiry_days) if expiry_days > 0 else None

        effective_scopes = scopes if scopes is not None else [APIKeyScope.RAG_READ]

        metadata = APIKeyMetadata(
            key_id=str(uuid.uuid4()),
            name=name,
            key_hash=key_hash,
            prefix=prefix,
            scopes=effective_scopes,
            status=APIKeyStatus.ACTIVE,
            created_at=now,
            expires_at=expires_at,
            description=description,
        )

        await self.store.save_key(metadata)
        logger.info(
            "Provisioned new API key name=%s id=%s prefix=%s scopes=%s expires_at=%s",
            name,
            metadata.key_id,
            prefix,
            [s.value for s in effective_scopes],
            expires_at.isoformat() if expires_at else "never",
        )
        return raw_key, metadata

    async def validate_key(
        self,
        raw_key: str,
        required_scope: APIKeyScope | None = None,
    ) -> APIKeyValidationResult:
        """Validate API key token, verify status, and enforce RBAC scope authorization."""
        if not raw_key or not raw_key.strip():
            return APIKeyValidationResult(
                is_valid=False,
                error_code="MISSING_KEY",
                error_message="API key token is missing or empty.",
            )

        key_hash = hash_api_key(raw_key)
        key_meta = await self.store.get_key_by_hash(key_hash)

        if not key_meta:
            return APIKeyValidationResult(
                is_valid=False,
                error_code="INVALID_KEY",
                error_message="API key not found or invalid.",
            )

        now = datetime.now(UTC)

        # 1. Lifecycle status check
        if key_meta.status == APIKeyStatus.REVOKED:
            return APIKeyValidationResult(
                is_valid=False,
                key_metadata=key_meta,
                error_code="REVOKED_KEY",
                error_message="API key has been revoked.",
            )

        if key_meta.status == APIKeyStatus.EXPIRED:
            return APIKeyValidationResult(
                is_valid=False,
                key_metadata=key_meta,
                error_code="EXPIRED_KEY",
                error_message="API key has expired.",
            )

        # 2. Expiration and rotation grace period check
        if key_meta.status == APIKeyStatus.ROTATING:
            if key_meta.grace_expires_at and now > key_meta.grace_expires_at:
                key_meta.status = APIKeyStatus.EXPIRED
                await self.store.update_key(key_meta)
                return APIKeyValidationResult(
                    is_valid=False,
                    key_metadata=key_meta,
                    error_code="GRACE_PERIOD_EXPIRED",
                    error_message="API key rotation grace period has expired.",
                )
        else:
            if key_meta.expires_at and now > key_meta.expires_at:
                key_meta.status = APIKeyStatus.EXPIRED
                await self.store.update_key(key_meta)
                return APIKeyValidationResult(
                    is_valid=False,
                    key_metadata=key_meta,
                    error_code="EXPIRED_KEY",
                    error_message="API key has naturally expired.",
                )

        # 3. Scope / RBAC Authorization
        if required_scope is not None:
            has_admin = APIKeyScope.ADMIN in key_meta.scopes
            has_required = required_scope in key_meta.scopes
            if not (has_admin or has_required):
                return APIKeyValidationResult(
                    is_valid=False,
                    key_metadata=key_meta,
                    error_code="INSUFFICIENT_SCOPE",
                    error_message=(
                        f"API key lacks required operational scope '{required_scope.value}'."
                    ),
                )

        return APIKeyValidationResult(
            is_valid=True,
            key_metadata=key_meta,
        )

    async def rotate_key(
        self,
        key_id: str,
        grace_period_hours: int = 24,
    ) -> tuple[str, APIKeyMetadata]:
        """Rotate an active key, creating a new token while granting a grace window to the old."""
        old_meta = await self.store.get_key_by_id(key_id)
        if not old_meta:
            raise ValueError(f"API key with ID '{key_id}' not found.")
        if old_meta.status == APIKeyStatus.REVOKED:
            raise ValueError("Cannot rotate a revoked API key.")

        now = datetime.now(UTC)

        # Update old key to ROTATING with grace window
        old_meta.status = APIKeyStatus.ROTATING
        old_meta.rotated_at = now
        old_meta.grace_expires_at = now + timedelta(hours=grace_period_hours)
        await self.store.update_key(old_meta)

        # Provision replacement key with identical scopes and name
        new_raw_key = generate_api_key()
        new_key_hash = hash_api_key(new_raw_key)
        new_prefix = extract_key_prefix(new_raw_key)
        new_expires_at = (
            now + timedelta(days=self.default_expiry_days) if self.default_expiry_days > 0 else None
        )

        new_meta = APIKeyMetadata(
            key_id=str(uuid.uuid4()),
            name=old_meta.name,
            key_hash=new_key_hash,
            prefix=new_prefix,
            scopes=old_meta.scopes,
            status=APIKeyStatus.ACTIVE,
            created_at=now,
            expires_at=new_expires_at,
            description=f"Rotated replacement for {old_meta.key_id}",
        )
        await self.store.save_key(new_meta)

        logger.info(
            "Rotated key old_id=%s new_id=%s grace_period_hours=%d",
            old_meta.key_id,
            new_meta.key_id,
            grace_period_hours,
        )
        return new_raw_key, new_meta

    async def revoke_key(self, key_id: str) -> bool:
        """Immediately revoke an API key, preventing any further authentication."""
        meta = await self.store.get_key_by_id(key_id)
        if not meta:
            return False

        meta.status = APIKeyStatus.REVOKED
        await self.store.update_key(meta)
        logger.info("Revoked API key id=%s prefix=%s", meta.key_id, meta.prefix)
        return True

    async def list_keys(self) -> list[APIKeyMetadata]:
        """Retrieve all registered API key metadata records."""
        return await self.store.list_keys()

    async def get_key_by_id(self, key_id: str) -> APIKeyMetadata | None:
        """Retrieve metadata for a specific key identifier."""
        return await self.store.get_key_by_id(key_id)
