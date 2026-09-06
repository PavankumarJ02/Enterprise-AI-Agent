"""Factory resolving EmbeddingProvider and EmbeddingsService instances."""

from enterprise_agent.config.settings import Settings
from enterprise_agent.core.exceptions import ConfigurationError
from enterprise_agent.core.logging import get_logger
from enterprise_agent.embeddings.base import EmbeddingProvider
from enterprise_agent.embeddings.gemini import GeminiEmbeddingProvider
from enterprise_agent.embeddings.mock import MockEmbeddingProvider
from enterprise_agent.embeddings.openai import OpenAIEmbeddingProvider
from enterprise_agent.embeddings.service import EmbeddingsService

logger = get_logger(__name__)


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Instantiate and return the configured EmbeddingProvider implementation."""
    provider_type = settings.embedding_provider.lower()

    if provider_type == "mock":
        logger.info(
            "Initializing MockEmbeddingProvider [model=%s, dimensions=%d]",
            settings.embedding_model,
            settings.embedding_dimensions,
        )
        return MockEmbeddingProvider(
            dimensions=settings.embedding_dimensions,
            model_name=settings.embedding_model,
        )

    if provider_type == "gemini":
        api_key = settings.effective_gemini_api_key
        if not api_key:
            raise ConfigurationError(
                "Embedding provider 'gemini' requires GEMINI_API_KEY (or LLM_API_KEY) "
                "to be configured in .env."
            )
        logger.info(
            "Initializing GeminiEmbeddingProvider [model=%s, dimensions=%d]",
            settings.embedding_model,
            settings.embedding_dimensions,
        )
        return GeminiEmbeddingProvider(
            api_key=api_key,
            model_name=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )

    if provider_type == "openai":
        api_key = settings.llm_api_key.get_secret_value()
        if not api_key:
            raise ConfigurationError(
                "Embedding provider 'openai' requires LLM_API_KEY to be configured in .env."
            )
        logger.info(
            "Initializing OpenAIEmbeddingProvider [model=%s, dimensions=%d, base_url=%s]",
            settings.embedding_model,
            settings.embedding_dimensions,
            settings.llm_base_url,
        )
        return OpenAIEmbeddingProvider(
            api_key=api_key,
            base_url=settings.llm_base_url,
            model_name=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            timeout_seconds=settings.llm_request_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    raise ConfigurationError(f"Unsupported embedding provider requested: '{provider_type}'")


def get_embeddings_service(settings: Settings) -> EmbeddingsService:
    """Instantiate and return configured EmbeddingsService."""
    provider = get_embedding_provider(settings)
    return EmbeddingsService(
        provider=provider,
        batch_size=settings.embedding_batch_size,
    )
