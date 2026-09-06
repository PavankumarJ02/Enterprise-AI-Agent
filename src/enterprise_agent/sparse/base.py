"""Abstract Base Class for sparse keyword (BM25 / Elasticsearch) index providers."""

from abc import ABC, abstractmethod
from typing import Any

from enterprise_agent.ingestion.models import DocumentChunk
from enterprise_agent.schemas.search import SearchResultItem


class SparseStore(ABC):
    """Abstract interface defining contract for sparse/lexical keyword retrieval engines."""

    @abstractmethod
    async def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Index a batch of document chunks into the sparse keyword index.

        Args:
            chunks: List of document chunks containing textual content and metadata.

        Returns:
            Count of chunks successfully indexed.
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResultItem]:
        """Perform sparse keyword/lexical search matching query terms against indexed chunks.

        Args:
            query: User's keyword query.
            top_k: Maximum number of ranked chunks to return.
            filters: Optional key-value metadata constraints.
            min_score: Minimum threshold for normalized relevance score.

        Returns:
            List of SearchResultItem ranked by BM25 relevance score descending.
        """
        ...

    @abstractmethod
    async def delete_by_document_id(self, document_id: str) -> int:
        """Purge all indexed chunks belonging to a specific parent document ID.

        Args:
            document_id: UUID or string ID of the document to purge.

        Returns:
            Count of chunks removed.
        """
        ...

    @abstractmethod
    async def delete_collection(self) -> None:
        """Purge the entire sparse index/collection."""
        ...
