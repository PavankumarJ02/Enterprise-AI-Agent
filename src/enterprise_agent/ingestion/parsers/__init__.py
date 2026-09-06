"""Document parsers package."""

from enterprise_agent.ingestion.parsers.base import BaseDocumentParser, ParsedDocument
from enterprise_agent.ingestion.parsers.factory import get_parser_for_filename
from enterprise_agent.ingestion.parsers.markdown import MarkdownParser
from enterprise_agent.ingestion.parsers.pdf import PDFParser
from enterprise_agent.ingestion.parsers.text import PlainTextParser

__all__ = [
    "BaseDocumentParser",
    "ParsedDocument",
    "PlainTextParser",
    "MarkdownParser",
    "PDFParser",
    "get_parser_for_filename",
]
