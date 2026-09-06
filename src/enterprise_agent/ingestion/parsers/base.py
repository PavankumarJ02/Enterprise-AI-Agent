"""Abstract Base Class for document parsers."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ParsedDocument(BaseModel):
    """Normalized output produced by any document parser."""

    title: str = Field(default="Untitled Document")
    content: str = Field(..., description="Full extracted and cleaned plain text.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Parser-extracted metadata.")
    pages: list[tuple[int, str]] = Field(
        default_factory=list,
        description="Optional list of (page_number, text) tuples for paginated formats.",
    )


class BaseDocumentParser(ABC):
    """Abstract interface defining the parsing protocol for enterprise file types."""

    @abstractmethod
    async def parse(self, content_bytes: bytes, filename: str) -> ParsedDocument:
        """Parse raw file bytes into a clean, normalized ParsedDocument."""
        pass
