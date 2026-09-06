"""Domain data models for document ingestion, parsing, and chunking."""

import hashlib
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def compute_sha256(content: str) -> str:
    """Compute deterministic SHA-256 hash for content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class Document(BaseModel):
    """Raw enterprise document representation."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(..., min_length=1, description="Document human-readable title.")
    content: str = Field(..., description="Full text content of document.")
    source: str = Field(..., description="Document source location (filename, path, or URL).")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Document metadata.")
    checksum: str = Field(default="", description="SHA-256 hash of content.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def model_post_init(self, __context: Any) -> None:
        """Compute checksum automatically if omitted."""
        if not self.checksum and self.content:
            self.checksum = compute_sha256(self.content)


class DocumentChunk(BaseModel):
    """Atomic text segment extracted from a Document for vector search and BM25."""

    id: str = Field(..., description="Unique chunk identifier (e.g. {doc_id}:chunk_{index}).")
    document_id: str = Field(..., description="Foreign key back to parent Document.")
    content: str = Field(..., min_length=1, description="Text slice for embedding/retrieval.")
    chunk_index: int = Field(..., ge=0, description="Sequential index within the document.")
    start_char: int = Field(default=0, ge=0, description="Start character offset in document.")
    end_char: int = Field(default=0, ge=0, description="End character offset in document.")
    token_count: int = Field(default=0, ge=0, description="Estimated token count.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Propagated metadata (e.g., page number, section header, title).",
    )
    chunk_hash: str = Field(default="", description="SHA-256 hash of chunk content.")

    def model_post_init(self, __context: Any) -> None:
        """Compute chunk hash automatically if omitted."""
        if not self.chunk_hash and self.content:
            self.chunk_hash = compute_sha256(self.content)


class IngestionResult(BaseModel):
    """Result of processing and chunking a document."""

    document: Document
    chunks: list[DocumentChunk]
    status: str = "success"  # "success" or "skipped_duplicate"
    message: str = "Document successfully ingested and chunked."


class DocumentSummary(BaseModel):
    """Lightweight summary of an ingested document."""

    id: str
    title: str
    source: str
    num_chunks: int
    total_characters: int
    checksum: str
    created_at: datetime
