"""Request and response schemas for document ingestion and inspection."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    """Payload for ingesting raw text or markdown."""

    title: str = Field(..., min_length=1, max_length=255, description="Document title.")
    content: str = Field(..., min_length=1, description="Raw text or markdown content.")
    source: str = Field(default="api_direct", description="Origin tag or filename.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata attributes."
    )


class IngestResponse(BaseModel):
    """Output confirmation after document ingestion and chunking."""

    document_id: str
    title: str
    source: str
    num_chunks: int
    total_characters: int
    checksum: str
    status: str
    message: str


class DocumentChunkResponse(BaseModel):
    """Single chunk schema for API inspection."""

    id: str
    document_id: str
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    token_count: int
    metadata: dict[str, Any]


class DocumentSummaryResponse(BaseModel):
    """Summary of an ingested document."""

    id: str
    title: str
    source: str
    num_chunks: int
    total_characters: int
    checksum: str
    created_at: datetime


class DocumentListResponse(BaseModel):
    """Collection response listing ingested documents."""

    documents: list[DocumentSummaryResponse]
    total_documents: int
