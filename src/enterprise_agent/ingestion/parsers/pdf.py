"""PDF document parser extracting paginated content and metadata."""

import io
import os

import pypdf

from enterprise_agent.core.logging import get_logger
from enterprise_agent.ingestion.parsers.base import BaseDocumentParser, ParsedDocument
from enterprise_agent.ingestion.sanitization import sanitize_document_text

logger = get_logger(__name__)


class PDFParser(BaseDocumentParser):
    """Parser for Adobe Portable Document Format (.pdf) files."""

    async def parse(self, content_bytes: bytes, filename: str) -> ParsedDocument:
        """Parse PDF byte stream, preserving page-number attributions."""
        stream = io.BytesIO(content_bytes)
        reader = pypdf.PdfReader(stream)

        total_pages = len(reader.pages)
        pages_data: list[tuple[int, str]] = []
        combined_text_parts: list[str] = []

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                raw_page_text = page.extract_text() or ""
                cleaned_page_text = sanitize_document_text(raw_page_text)
                if cleaned_page_text:
                    pages_data.append((page_idx, cleaned_page_text))
                    combined_text_parts.append(cleaned_page_text)
            except Exception as e:
                logger.warning(
                    "Failed to extract text from page %d in %s: %s", page_idx, filename, e
                )

        full_content = "\n\n".join(combined_text_parts)

        # Attempt to read PDF embedded metadata
        title = ""
        metadata: dict[str, object] = {
            "source_type": "pdf",
            "filename": filename,
            "total_pages": total_pages,
        }

        if reader.metadata:
            pdf_meta = reader.metadata
            if pdf_meta.title:
                title = str(pdf_meta.title).strip()
            if pdf_meta.author:
                metadata["author"] = str(pdf_meta.author)

        if not title:
            base_name = os.path.splitext(os.path.basename(filename))[0]
            title = base_name.replace("-", " ").replace("_", " ").title()

        return ParsedDocument(
            title=title,
            content=full_content,
            metadata=metadata,
            pages=pages_data,
        )
