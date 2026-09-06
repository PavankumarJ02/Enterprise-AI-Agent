"""Google Gemini Embeddings Provider using official google-genai SDK."""

from google import genai
from google.genai import errors, types

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


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini dense vector embedding provider (text-embedding-004)."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-004",
        dimensions: int = 768,
    ) -> None:
        self.model_name = model_name
        self._dimensions = dimensions
        self._client = genai.Client(api_key=api_key)

    @property
    def dimensions(self) -> int:
        """Configured embedding dimensionality."""
        return self._dimensions

    def _map_exception(self, err: Exception) -> LLMException:
        """Map google-genai errors to domain exceptions."""
        if isinstance(err, errors.APIError):
            code = getattr(err, "code", None)
            msg = str(err.message or err)
            if code in {401, 403}:
                return LLMAuthenticationError(f"Gemini embedding auth failure: {msg}")
            if code == 429:
                return LLMRateLimitError(f"Gemini embedding quota exceeded: {msg}")
            if code in {408, 504}:
                return LLMTimeoutError(f"Gemini embedding request timed out: {msg}")
            if code is not None and code >= 500:
                return LLMProviderError(f"Gemini embedding server error: {msg}")
            return LLMException(f"Gemini embedding error: {msg}")
        return LLMException(f"Unexpected error during embedding generation: {err}")

    async def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        """Generate embeddings optimized for document retrieval."""
        if not texts:
            return EmbeddingResult(
                vectors=[],
                model=self.model_name,
                dimensions=self._dimensions,
                token_count=0,
            )

        config = types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=self._dimensions,
        )

        try:
            resp = await self._client.aio.models.embed_content(
                model=self.model_name,
                contents=texts,
                config=config,
            )

            vectors: list[list[float]] = []
            if resp.embeddings:
                for emb in resp.embeddings:
                    if emb.values:
                        vectors.append(emb.values)

            # Fallback if fewer vectors returned
            while len(vectors) < len(texts):
                vectors.append([0.0] * self._dimensions)

            token_count = sum(max(1, len(t.split())) for t in texts)

            return EmbeddingResult(
                vectors=vectors,
                model=self.model_name,
                dimensions=self._dimensions,
                token_count=token_count,
            )
        except Exception as e:
            logger.error("Error generating Gemini embeddings: %s", e)
            raise self._map_exception(e) from e

    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding optimized for search query matching."""
        config = types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=self._dimensions,
        )

        try:
            resp = await self._client.aio.models.embed_content(
                model=self.model_name,
                contents=query,
                config=config,
            )

            if resp.embeddings and len(resp.embeddings) > 0 and resp.embeddings[0].values:
                return resp.embeddings[0].values

            return [0.0] * self._dimensions
        except Exception as e:
            logger.error("Error generating Gemini query embedding: %s", e)
            raise self._map_exception(e) from e

    async def health_check(self) -> bool:
        """Verify upstream Gemini embedding connectivity."""
        try:
            res = await self.embed_query("health check probe")
            return len(res) == self._dimensions
        except Exception as e:
            logger.warning("Gemini embedding health check failed: %s", e)
            return False
