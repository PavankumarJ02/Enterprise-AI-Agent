"""Base class for document chunking strategies."""

from abc import ABC, abstractmethod

from enterprise_agent.ingestion.models import Document, DocumentChunk


class BaseChunker(ABC):
    """Abstract interface for splitting documents into retrievable chunks."""

    @abstractmethod
    def chunk(
        self,
        document: Document,
        pages: list[tuple[int, str]] | None = None,
    ) -> list[DocumentChunk]:
        """Split document into an ordered sequence of DocumentChunks."""
        pass
