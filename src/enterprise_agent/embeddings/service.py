"""Embeddings service coordinating batching, concurrency, and chunk vectorization."""

from enterprise_agent.core.logging import get_logger
from enterprise_agent.embeddings.base import EmbeddingProvider, EmbeddingResult
from enterprise_agent.ingestion.models import DocumentChunk

logger = get_logger(__name__)


class EmbeddingsService:
    """Production service for batch embedding generation and document chunk vectorization."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        batch_size: int = 64,
    ) -> None:
        self.provider = provider
        self.batch_size = max(1, batch_size)

    @property
    def dimensions(self) -> int:
        """Vector dimensionality of the underlying provider."""
        return self.provider.dimensions

    async def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        """Batch-embed a list of strings respecting provider batch size limits."""
        if not texts:
            return EmbeddingResult(
                vectors=[],
                model="empty",
                dimensions=self.dimensions,
                token_count=0,
            )

        all_vectors: list[list[float]] = []
        total_tokens = 0
        model_name = ""

        # Slicing into safe batch sizes
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.debug("Embedding batch %d-%d of %d items", i, i + len(batch), len(texts))
            res = await self.provider.embed_texts(batch)
            all_vectors.extend(res.vectors)
            total_tokens += res.token_count
            model_name = res.model

        return EmbeddingResult(
            vectors=all_vectors,
            model=model_name,
            dimensions=self.dimensions,
            token_count=total_tokens,
        )

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for semantic similarity search."""
        return await self.provider.embed_query(query)

    async def embed_document_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> tuple[list[DocumentChunk], list[list[float]]]:
        """Embed a list of DocumentChunks in-place and return parallel vector array."""
        if not chunks:
            return [], []

        chunk_texts = [c.content for c in chunks]
        res = await self.embed_texts(chunk_texts)

        return chunks, res.vectors
