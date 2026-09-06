"""Embeddings package."""

from enterprise_agent.embeddings.base import EmbeddingProvider, EmbeddingResult
from enterprise_agent.embeddings.factory import get_embedding_provider, get_embeddings_service
from enterprise_agent.embeddings.gemini import GeminiEmbeddingProvider
from enterprise_agent.embeddings.mock import MockEmbeddingProvider
from enterprise_agent.embeddings.openai import OpenAIEmbeddingProvider
from enterprise_agent.embeddings.service import EmbeddingsService

__all__ = [
    "EmbeddingProvider",
    "EmbeddingResult",
    "GeminiEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "MockEmbeddingProvider",
    "EmbeddingsService",
    "get_embedding_provider",
    "get_embeddings_service",
]
