"""Factory to construct LLM providers based on application configuration."""

from enterprise_agent.config.settings import Settings
from enterprise_agent.core.exceptions import ConfigurationError
from enterprise_agent.core.logging import get_logger
from enterprise_agent.llm.base import LLMProvider
from enterprise_agent.llm.mock_client import MockLLMProvider
from enterprise_agent.llm.openai_client import OpenAILLMProvider

logger = get_logger(__name__)


def get_llm_provider(settings: Settings) -> LLMProvider:
    """Instantiate and return the configured LLMProvider implementation."""
    provider_type = settings.llm_provider.lower()

    if provider_type == "mock":
        logger.info("Initializing MockLLMProvider (offline development/testing mode)")
        return MockLLMProvider(model_name=settings.llm_model)

    if provider_type in {"openai", "groq", "ollama", "azure"}:
        api_key = settings.llm_api_key.get_secret_value()
        if not api_key and provider_type not in {"ollama"}:
            raise ConfigurationError(
                f"LLM provider '{provider_type}' requires LLM_API_KEY to be configured in .env."
            )

        logger.info(
            "Initializing OpenAILLMProvider [provider=%s, model=%s, base_url=%s]",
            provider_type,
            settings.llm_model,
            settings.llm_base_url,
        )
        return OpenAILLMProvider(
            api_key=api_key or "no-key-required",
            base_url=settings.llm_base_url,
            default_model=settings.llm_model,
            default_temperature=settings.llm_temperature,
            default_max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.llm_request_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    raise ConfigurationError(f"Unsupported LLM provider requested: '{provider_type}'")
