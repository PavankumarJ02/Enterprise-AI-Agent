"""FastAPI dependency injection providers."""

from collections.abc import Generator

from fastapi import Depends

from enterprise_agent.config.settings import Settings, get_settings
from enterprise_agent.llm.base import LLMProvider
from enterprise_agent.llm.factory import get_llm_provider


def get_app_settings() -> Settings:
    """Dependency provider for application settings."""
    return get_settings()


def get_llm(
    settings: Settings = Depends(get_app_settings),
) -> Generator[LLMProvider, None, None]:
    """Dependency provider yielding the configured LLM provider instance."""
    provider = get_llm_provider(settings)
    yield provider
