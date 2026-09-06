"""Main application entrypoint configuring FastAPI, lifespan, middlewares, and routes."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from enterprise_agent.api.v1.api import api_v1_router
from enterprise_agent.config.settings import Settings, get_settings
from enterprise_agent.core.exceptions import (
    AppException,
    ConfigurationError,
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from enterprise_agent.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager handling application startup and shutdown."""
    settings = get_settings()
    setup_logging(level=settings.log_level)
    logger.info(
        "Starting %s [version=%s, env=%s, provider=%s]",
        settings.app_name,
        settings.app_version,
        settings.app_env,
        settings.llm_provider,
    )
    yield
    logger.info("Shutting down %s cleanly", settings.app_name)


def create_application(settings: Settings | None = None) -> FastAPI:
    """Factory creating and configuring FastAPI instance."""
    app_settings = settings or get_settings()

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description=(
            "Production-oriented Enterprise AI Knowledge & Decision Agent providing "
            "adaptive retrieval, tool-augmented reasoning, and safe analytical execution."
        ),
        lifespan=lifespan,
    )

    if settings is not None:
        from enterprise_agent.api.deps import get_app_settings

        app.dependency_overrides[get_app_settings] = lambda: app_settings

    # -------------------------------------------------------------------------
    # Middlewares
    # -------------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -------------------------------------------------------------------------
    # Exception Handlers
    # -------------------------------------------------------------------------
    @app.exception_handler(LLMAuthenticationError)
    async def handle_llm_auth_error(request: Request, exc: LLMAuthenticationError) -> JSONResponse:
        logger.error("LLM authentication error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "LLM_AUTH_ERROR", "message": exc.message},
        )

    @app.exception_handler(LLMRateLimitError)
    async def handle_llm_rate_limit(request: Request, exc: LLMRateLimitError) -> JSONResponse:
        logger.warning("LLM rate limit encountered: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"error": "LLM_RATE_LIMIT", "message": exc.message},
        )

    @app.exception_handler(LLMTimeoutError)
    async def handle_llm_timeout(request: Request, exc: LLMTimeoutError) -> JSONResponse:
        logger.error("LLM timeout: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={"error": "LLM_TIMEOUT", "message": exc.message},
        )

    @app.exception_handler(LLMProviderError)
    async def handle_llm_provider_error(request: Request, exc: LLMProviderError) -> JSONResponse:
        logger.error("LLM upstream provider failure: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"error": "LLM_UPSTREAM_ERROR", "message": exc.message},
        )

    @app.exception_handler(ConfigurationError)
    async def handle_config_error(request: Request, exc: ConfigurationError) -> JSONResponse:
        logger.critical("Application configuration error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "CONFIGURATION_ERROR", "message": exc.message},
        )

    @app.exception_handler(AppException)
    async def handle_app_exception(request: Request, exc: AppException) -> JSONResponse:
        logger.error("Domain exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "APP_ERROR", "message": exc.message, "details": exc.details},
        )

    # -------------------------------------------------------------------------
    # Route Registration
    # -------------------------------------------------------------------------
    app.include_router(api_v1_router, prefix="/api/v1")

    # Root health alias
    @app.get("/health", tags=["Health"], include_in_schema=False)
    async def root_health() -> dict[str, str]:
        return {"status": "healthy", "service": app_settings.app_name}

    return app


app = create_application()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "enterprise_agent.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
