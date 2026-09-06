"""Ingestion service orchestrating parsing, sanitization, chunking, and deduplication."""

from typing import Any

from enterprise_agent.core.logging import get_logger
from enterprise_agent.ingestion.chunking.recursive import RecursiveCharacterChunker
from enterprise_agent.ingestion.models import (
    Document,
    DocumentChunk,
    DocumentSummary,
    IngestionResult,
    compute_sha256,
)
from enterprise_agent.ingestion.parsers.factory import get_parser_for_filename
from enterprise_agent.ingestion.sanitization import sanitize_document_text

logger = get_logger(__name__)


class IngestionService:
    """Enterprise document ingestion engine managing parsing, chunking, and storage."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ) -> None:
        self.chunker = RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        # In-memory document and chunk registry
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, list[DocumentChunk]] = {}
        self._checksum_index: dict[str, str] = {}  # checksum -> document_id

    async def ingest_text(
        self,
        text: str,
        title: str,
        source: str = "direct_text",
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Ingest raw plain text or Markdown string."""
        cleaned_text = sanitize_document_text(text)
        checksum = compute_sha256(cleaned_text)

        # 1. Deduplication check
        if checksum in self._checksum_index:
            existing_doc_id = self._checksum_index[checksum]
            existing_doc = self._documents[existing_doc_id]
            existing_chunks = self._chunks.get(existing_doc_id, [])
            logger.info("Document '%s' already ingested (checksum match). Skipping.", title)
            return IngestionResult(
                document=existing_doc,
                chunks=existing_chunks,
                status="skipped_duplicate",
                message=f"Document with identical content already ingested ({existing_doc_id}).",
            )

        # 2. Construct Document entity
        doc_meta = metadata or {}
        document = Document(
            title=title,
            content=cleaned_text,
            source=source,
            metadata=doc_meta,
            checksum=checksum,
        )

        # 3. Chunk the document
        chunks = self.chunker.chunk(document)

        # 4. Register
        self._documents[document.id] = document
        self._chunks[document.id] = chunks
        self._checksum_index[checksum] = document.id

        logger.info(
            "Ingested text document '%s' [id=%s, length=%d chars, chunks=%d]",
            title,
            document.id,
            len(cleaned_text),
            len(chunks),
        )

        return IngestionResult(
            document=document,
            chunks=chunks,
            status="success",
            message=f"Successfully ingested and produced {len(chunks)} chunks.",
        )

    async def ingest_file(
        self,
        content_bytes: bytes,
        filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Parse raw file bytes (PDF, Markdown, Text) and ingest into system."""
        parser = get_parser_for_filename(filename)
        parsed = await parser.parse(content_bytes, filename)

        checksum = compute_sha256(parsed.content)

        # 1. Deduplication check
        if checksum in self._checksum_index:
            existing_doc_id = self._checksum_index[checksum]
            existing_doc = self._documents[existing_doc_id]
            existing_chunks = self._chunks.get(existing_doc_id, [])
            logger.info("File '%s' already ingested (checksum match). Skipping.", filename)
            return IngestionResult(
                document=existing_doc,
                chunks=existing_chunks,
                status="skipped_duplicate",
                message=f"File with identical content already ingested ({existing_doc_id}).",
            )

        # 2. Build Document entity
        merged_meta = {**(metadata or {}), **parsed.metadata}
        document = Document(
            title=parsed.title,
            content=parsed.content,
            source=filename,
            metadata=merged_meta,
            checksum=checksum,
        )

        # 3. Chunk with page-awareness
        chunks = self.chunker.chunk(document, pages=parsed.pages)

        # 4. Store
        self._documents[document.id] = document
        self._chunks[document.id] = chunks
        self._checksum_index[checksum] = document.id

        logger.info(
            "Ingested file '%s' [id=%s, title='%s', pages=%d, chunks=%d]",
            filename,
            document.id,
            parsed.title,
            len(parsed.pages),
            len(chunks),
        )

        return IngestionResult(
            document=document,
            chunks=chunks,
            status="success",
            message=f"Successfully ingested '{filename}' and created {len(chunks)} chunks.",
        )

    def list_documents(self) -> list[DocumentSummary]:
        """List summary view of all ingested documents."""
        summaries: list[DocumentSummary] = []
        for doc in self._documents.values():
            chunks = self._chunks.get(doc.id, [])
            summaries.append(
                DocumentSummary(
                    id=doc.id,
                    title=doc.title,
                    source=doc.source,
                    num_chunks=len(chunks),
                    total_characters=len(doc.content),
                    checksum=doc.checksum,
                    created_at=doc.created_at,
                )
            )
        return summaries

    def get_document(self, doc_id: str) -> Document | None:
        """Retrieve full document by id."""
        return self._documents.get(doc_id)

    def get_document_chunks(self, doc_id: str) -> list[DocumentChunk]:
        """Retrieve chunks generated for a specific document."""
        return self._chunks.get(doc_id, [])

    def delete_document(self, doc_id: str) -> bool:
        """Delete document and associated chunks from registry."""
        doc = self._documents.pop(doc_id, None)
        if not doc:
            return False
        self._chunks.pop(doc_id, None)
        if doc.checksum in self._checksum_index:
            del self._checksum_index[doc.checksum]
        return True

    def clear(self) -> None:
        """Clear all ingested documents and chunks (for testing)."""
        self._documents.clear()
        self._chunks.clear()
        self._checksum_index.clear()
