"""OpenAI-compatible Embeddings Provider."""

import openai
from openai import AsyncOpenAI

from enterprise_agent.core.exceptions import (
    LLMAuthenticationError,
    LLMException,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from enterprise_agent.core.logging import get_logger
from enterprise_agent.embeddings.base import EmbeddingProvider, EmbeddingResult

logger = get_logger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible dense vector embedding provider."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "text-embedding-3-small",
        dimensions: int = 1536,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.model_name = model_name
        self._dimensions = dimensions
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    @property
    def dimensions(self) -> int:
        """Configured embedding dimensionality."""
        return self._dimensions

    def _map_exception(self, err: Exception) -> LLMException:
        """Map OpenAI SDK exceptions to domain exceptions."""
        if isinstance(err, openai.AuthenticationError):
            return LLMAuthenticationError(f"OpenAI embedding auth failure: {err}")
        if isinstance(err, openai.RateLimitError):
            return LLMRateLimitError(f"OpenAI embedding quota exceeded: {err}")
        if isinstance(err, openai.APITimeoutError):
            return LLMTimeoutError(f"OpenAI embedding request timed out: {err}")
        if isinstance(err, (openai.APIConnectionError, openai.APIStatusError, openai.APIError)):
            return LLMProviderError(f"OpenAI embedding server error: {err}")
        return LLMException(f"Unexpected error during OpenAI embedding: {err}")

    async def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        """Embed list of texts via OpenAI Embeddings API."""
        if not texts:
            return EmbeddingResult(
                vectors=[],
                model=self.model_name,
                dimensions=self._dimensions,
                token_count=0,
            )

        try:
            resp = await self._client.embeddings.create(
                model=self.model_name,
                input=texts,
            )

            vectors = [item.embedding for item in resp.data]
            tokens = resp.usage.total_tokens if resp.usage else sum(len(t.split()) for t in texts)

            return EmbeddingResult(
                vectors=vectors,
                model=self.model_name,
                dimensions=len(vectors[0]) if vectors else self._dimensions,
                token_count=tokens,
            )
        except Exception as e:
            logger.error("Error generating OpenAI embeddings: %s", e)
            raise self._map_exception(e) from e

    async def embed_query(self, query: str) -> list[float]:
        """Embed single query string."""
        res = await self.embed_texts([query])
        if res.vectors:
            return res.vectors[0]
        return [0.0] * self._dimensions

    async def health_check(self) -> bool:
        """Verify OpenAI embedding connectivity."""
        try:
            res = await self.embed_query("probe")
            return len(res) > 0
        except Exception as e:
            logger.warning("OpenAI embedding health check failed: %s", e)
            return False
