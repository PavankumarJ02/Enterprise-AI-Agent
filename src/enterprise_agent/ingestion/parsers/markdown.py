"""Markdown document parser extracting hierarchical section metadata."""

import os
import re

from enterprise_agent.ingestion.parsers.base import BaseDocumentParser, ParsedDocument
from enterprise_agent.ingestion.sanitization import sanitize_document_text


class MarkdownParser(BaseDocumentParser):
    """Parser for Markdown (.md, .markdown) documentation files."""

    async def parse(self, content_bytes: bytes, filename: str) -> ParsedDocument:
        """Parse markdown text, extract title from H1, and extract section structure."""
        try:
            raw_text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raw_text = content_bytes.decode("latin-1", errors="replace")

        cleaned = sanitize_document_text(raw_text)

        # 1. Extract title from first H1 header (e.g., # Company Refund Policy)
        title_match = re.search(r"^#\s+(.+)$", cleaned, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
        else:
            base_name = os.path.splitext(os.path.basename(filename))[0]
            title = base_name.replace("-", " ").replace("_", " ").title()

        # 2. Extract top-level section headings
        headings = [h.strip() for h in re.findall(r"^#{1,3}\s+(.+)$", cleaned, re.MULTILINE)]

        return ParsedDocument(
            title=title,
            content=cleaned,
            metadata={
                "source_type": "markdown",
                "filename": filename,
                "sections": headings,
            },
            pages=[(1, cleaned)],
        )
