"""Embeddings service coordinating batching, concurrency, and chunk vectorization."""

from enterprise_agent.core.logging import get_logger
from enterprise_agent.embeddings.base import EmbeddingProvider, EmbeddingResult
from enterprise_agent.ingestion.models import DocumentChunk
from enterprise_agent.performance.cache import QueryEmbeddingCache

logger = get_logger(__name__)


class EmbeddingsService:
    """Production service for batch embedding generation and document chunk vectorization."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        batch_size: int = 64,
        cache: QueryEmbeddingCache | None = None,
    ) -> None:
        self.provider = provider
        self.batch_size = max(1, batch_size)
        self.cache = cache

    @property
    def dimensions(self) -> int:
        """Vector dimensionality of the underlying provider."""
        return self.provider.dimensions

    @property
    def model_name(self) -> str:
        """Name of the embedding model configured on the provider."""
        return getattr(self.provider, "model_name", self.provider.__class__.__name__)

    async def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        """Batch-embed a list of strings respecting provider batch size limits and cache."""
        if not texts:
            return EmbeddingResult(
                vectors=[],
                model=self.model_name,
                dimensions=self.dimensions,
                token_count=0,
            )

        # 1. If cache is enabled, check for cached vectors
        vectors: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        if self.cache is not None:
            for idx, text in enumerate(texts):
                cached_vec = self.cache.get_embedding(text, self.model_name, self.dimensions)
                if cached_vec is not None:
                    vectors[idx] = cached_vec
                else:
                    uncached_indices.append(idx)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        total_tokens = 0
        model_name = self.model_name

        # 2. Process uncached batches
        if uncached_texts:
            for i in range(0, len(uncached_texts), self.batch_size):
                batch = uncached_texts[i : i + self.batch_size]
                indices_batch = uncached_indices[i : i + self.batch_size]
                logger.debug(
                    "Embedding uncached batch %d-%d of %d items",
                    i,
                    i + len(batch),
                    len(uncached_texts),
                )
                res = await self.provider.embed_texts(batch)
                total_tokens += res.token_count
                model_name = res.model

                for local_idx, vec in enumerate(res.vectors):
                    global_idx = indices_batch[local_idx]
                    vectors[global_idx] = vec
                    if self.cache is not None:
                        self.cache.set_embedding(
                            text=uncached_texts[i + local_idx],
                            model=self.model_name,
                            dimensions=self.dimensions,
                            vector=vec,
                        )

        # Reconstructed dense vectors in original order
        final_vectors: list[list[float]] = [v for v in vectors if v is not None]

        return EmbeddingResult(
            vectors=final_vectors,
            model=model_name,
            dimensions=self.dimensions,
            token_count=total_tokens,
        )

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for semantic similarity search with optional caching."""
        if self.cache is not None:
            cached_vector = self.cache.get_embedding(
                text=query,
                model=self.model_name,
                dimensions=self.dimensions,
            )
            if cached_vector is not None:
                logger.debug("Query embedding cache hit for query: %.30s...", query)
                return cached_vector

        vector = await self.provider.embed_query(query)

        if self.cache is not None:
            self.cache.set_embedding(
                text=query,
                model=self.model_name,
                dimensions=self.dimensions,
                vector=vector,
            )

        return vector

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
