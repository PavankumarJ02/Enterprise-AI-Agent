"""API endpoints for enterprise API key lifecycle management, RBAC, and rotation."""

from fastapi import APIRouter, Depends, HTTPException, status

from enterprise_agent.api.deps import get_api_key_manager, require_api_key
from enterprise_agent.core.logging import get_logger
from enterprise_agent.schemas.security import (
    APIKeyCreateRequest,
    APIKeyCreateResponse,
    APIKeyListResponse,
    APIKeyRevokeResponse,
    APIKeyRotateRequest,
    APIKeyRotateResponse,
)
from enterprise_agent.security.base import APIKeyMetadata, APIKeyScope
from enterprise_agent.security.manager import APIKeyManager

logger = get_logger(__name__)

router = APIRouter(prefix="/security", tags=["Security & Access Control"])


@router.post(
    "/keys",
    response_model=APIKeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Provision New API Key",
    description="Provision a new enterprise API key with specific operational capability scopes.",
)
async def create_api_key(
    payload: APIKeyCreateRequest,
    current_admin: APIKeyMetadata | None = Depends(require_api_key(APIKeyScope.ADMIN)),
    manager: APIKeyManager = Depends(get_api_key_manager),
) -> APIKeyCreateResponse:
    """Create a new API key record and return the plaintext token once."""
    raw_key, metadata = await manager.create_key(
        name=payload.name,
        scopes=payload.scopes,
        expires_in_days=payload.expires_in_days,
        description=payload.description,
    )
    return APIKeyCreateResponse(raw_key=raw_key, key=metadata)


@router.get(
    "/keys",
    response_model=APIKeyListResponse,
    status_code=status.HTTP_200_OK,
    summary="List API Keys",
    description="Retrieve all registered API key metadata (secret keys are never returned).",
)
async def list_api_keys(
    current_admin: APIKeyMetadata | None = Depends(require_api_key(APIKeyScope.ADMIN)),
    manager: APIKeyManager = Depends(get_api_key_manager),
) -> APIKeyListResponse:
    """List all stored API key records."""
    keys = await manager.list_keys()
    return APIKeyListResponse(keys=keys, total=len(keys))


@router.post(
    "/keys/{key_id}/rotate",
    response_model=APIKeyRotateResponse,
    status_code=status.HTTP_200_OK,
    summary="Rotate API Key",
    description="Rotate an active API key, provisioning a replacement with a grace window.",
)
async def rotate_api_key(
    key_id: str,
    payload: APIKeyRotateRequest,
    current_admin: APIKeyMetadata | None = Depends(require_api_key(APIKeyScope.ADMIN)),
    manager: APIKeyManager = Depends(get_api_key_manager),
) -> APIKeyRotateResponse:
    """Initiate key rotation with dual-validity grace period."""
    try:
        new_raw_key, new_metadata = await manager.rotate_key(
            key_id=key_id,
            grace_period_hours=payload.grace_period_hours,
        )
        return APIKeyRotateResponse(new_raw_key=new_raw_key, key=new_metadata)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.delete(
    "/keys/{key_id}",
    response_model=APIKeyRevokeResponse,
    status_code=status.HTTP_200_OK,
    summary="Revoke API Key",
    description="Immediately revoke an API key, preventing any subsequent authentication.",
)
async def revoke_api_key(
    key_id: str,
    current_admin: APIKeyMetadata | None = Depends(require_api_key(APIKeyScope.ADMIN)),
    manager: APIKeyManager = Depends(get_api_key_manager),
) -> APIKeyRevokeResponse:
    """Revoke key immediately."""
    revoked = await manager.revoke_key(key_id)
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key '{key_id}' not found.",
        )
    return APIKeyRevokeResponse(
        key_id=key_id,
        revoked=True,
        message=f"API key '{key_id}' successfully revoked.",
    )
