"""Health check and readiness endpoints."""

from fastapi import APIRouter, Depends

from enterprise_agent.api.deps import get_app_settings, get_llm
from enterprise_agent.config.settings import Settings
from enterprise_agent.llm.base import LLMProvider
from enterprise_agent.schemas.chat import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns service metadata and upstream connectivity status.",
)
async def health_check(
    settings: Settings = Depends(get_app_settings),
    llm: LLMProvider = Depends(get_llm),
) -> HealthResponse:
    """Verify system uptime and LLM client availability."""
    llm_healthy = await llm.health_check()
    status = "healthy" if llm_healthy else "degraded"

    return HealthResponse(
        status=status,
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        llm_healthy=llm_healthy,
    )
