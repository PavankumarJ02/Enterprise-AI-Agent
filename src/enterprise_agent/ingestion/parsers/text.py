"""Plain text document parser with multi-encoding fallback."""

import os

from enterprise_agent.ingestion.parsers.base import BaseDocumentParser, ParsedDocument
from enterprise_agent.ingestion.sanitization import sanitize_document_text


class PlainTextParser(BaseDocumentParser):
    """Parser for raw text, log, and CSV files."""

    async def parse(self, content_bytes: bytes, filename: str) -> ParsedDocument:
        """Decode raw bytes into sanitized plain text."""
        decoded_text = ""
        # Try UTF-8 first, fallback to Latin-1
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                decoded_text = content_bytes.decode(encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if not decoded_text:
            decoded_text = content_bytes.decode("utf-8", errors="replace")

        cleaned = sanitize_document_text(decoded_text)
        title = os.path.splitext(os.path.basename(filename))[0].replace("_", " ").title()

        return ParsedDocument(
            title=title,
            content=cleaned,
            metadata={"source_type": "text", "filename": filename},
            pages=[(1, cleaned)],
        )
