"""Factory for creating cross-encoder reranker instances."""

from functools import lru_cache

from enterprise_agent.config.settings import Settings, get_settings
from enterprise_agent.core.exceptions import ConfigurationError
from enterprise_agent.reranking.base import Reranker
from enterprise_agent.reranking.flashrank import FlashRankReranker
from enterprise_agent.reranking.mock import MockReranker


@lru_cache
def _get_cached_flashrank_reranker(model_name: str) -> FlashRankReranker:
    """Cache singleton FlashRank model instance to avoid redundant ONNX model loads."""
    return FlashRankReranker(model_name=model_name)


def create_reranker(settings: Settings | None = None) -> Reranker:
    """Instantiate and return the configured Reranker provider.

    Args:
        settings: Optional application settings. If None, resolves from get_settings().

    Returns:
        Configured Reranker implementation instance.

    Raises:
        ConfigurationError: If unknown reranker_provider is specified.
    """
    cfg = settings or get_settings()
    provider = cfg.reranker_provider.lower()

    if provider == "flashrank":
        return _get_cached_flashrank_reranker(cfg.reranker_model)
    elif provider == "mock":
        return MockReranker(model_name=cfg.reranker_model)
    else:
        raise ConfigurationError(
            f"Unsupported reranker provider '{cfg.reranker_provider}'. "
            f"Supported providers: ['flashrank', 'mock']."
        )
