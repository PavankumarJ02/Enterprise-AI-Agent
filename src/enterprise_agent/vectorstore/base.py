"""Abstract base class and domain models for Vector Store implementations."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from enterprise_agent.ingestion.models import DocumentChunk


class SearchResult(BaseModel):
    """Domain representation of a retrieved chunk from vector store search."""

    chunk_id: str = Field(..., description="Unique chunk identifier.")
    document_id: str = Field(..., description="Parent document identifier.")
    content: str = Field(..., description="Extracted text chunk content.")
    score: float = Field(..., description="Similarity score (typically cosine similarity).")
    chunk_index: int = Field(..., description="Position index of the chunk.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Associated chunk metadata.",
    )


class VectorStore(ABC):
    """Abstract vector database interface supporting ANN search, filtering, and indexing."""

    @abstractmethod
    async def create_collection_if_not_exists(self, dimensions: int) -> None:
        """Create the target vector collection if it does not already exist."""
        pass

    @abstractmethod
    async def collection_exists(self) -> bool:
        """Check if the vector collection currently exists."""
        pass

    @abstractmethod
    async def upsert(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> int:
        """Upsert a list of document chunks and their associated embedding vectors.

        Returns the count of successfully upserted points.
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Perform approximate nearest neighbor (ANN) vector search.

        Returns top-k search results sorted by descending similarity score.
        """
        pass

    @abstractmethod
    async def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunk vectors associated with a specific document ID.

        Returns the number of points deleted (or affected).
        """
        pass

    @abstractmethod
    async def count(self) -> int:
        """Return the total number of vectors in the collection."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify vector database connectivity and readiness."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Cleanly close underlying connection pools."""
        pass
