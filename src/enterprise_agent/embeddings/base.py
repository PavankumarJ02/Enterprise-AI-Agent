"""Base domain interfaces and models for dense vector embeddings."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class EmbeddingResult(BaseModel):
    """Container for batch embedding output."""

    vectors: list[list[float]] = Field(..., description="Dense float vectors.")
    model: str = Field(..., description="Embedding model used.")
    dimensions: int = Field(..., description="Vector dimensionality.")
    token_count: int = Field(default=0, ge=0, description="Tokens consumed in embedding process.")


class EmbeddingProvider(ABC):
    """Abstract interface for dense embedding generation."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the vector dimensionality produced by this provider."""
        pass

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        """Generate embeddings for a list of document passages or chunks."""
        pass

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Generate a single embedding vector optimized for search queries."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify connectivity to upstream embedding service."""
        pass
