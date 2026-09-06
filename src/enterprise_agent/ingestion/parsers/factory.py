"""Factory to resolve appropriate document parser based on file extension."""

import os

from enterprise_agent.ingestion.parsers.base import BaseDocumentParser
from enterprise_agent.ingestion.parsers.markdown import MarkdownParser
from enterprise_agent.ingestion.parsers.pdf import PDFParser
from enterprise_agent.ingestion.parsers.text import PlainTextParser


def get_parser_for_filename(filename: str) -> BaseDocumentParser:
    """Select the appropriate parser based on file extension."""
    ext = os.path.splitext(filename)[1].lower()

    if ext in {".pdf"}:
        return PDFParser()
    if ext in {".md", ".markdown"}:
        return MarkdownParser()
    # Default fallback to plain text for .txt, .log, .csv, etc.
    return PlainTextParser()
