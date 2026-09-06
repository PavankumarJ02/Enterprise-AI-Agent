"""Unit tests for document parsers (Text, Markdown, PDF)."""

import io

import pypdf
import pytest

from enterprise_agent.ingestion.parsers.factory import get_parser_for_filename
from enterprise_agent.ingestion.parsers.markdown import MarkdownParser
from enterprise_agent.ingestion.parsers.pdf import PDFParser
from enterprise_agent.ingestion.parsers.text import PlainTextParser


@pytest.mark.asyncio
async def test_plain_text_parser() -> None:
    """Verify PlainTextParser handles encoding and sets title from filename."""
    parser = PlainTextParser()
    raw_bytes = b"Standard Operating Procedure: Password renewal every 90 days."

    parsed = await parser.parse(raw_bytes, "password_policy.txt")

    assert parsed.title == "Password Policy"
    assert "Password renewal every 90 days" in parsed.content
    assert parsed.metadata["source_type"] == "text"
    assert len(parsed.pages) == 1


@pytest.mark.asyncio
async def test_markdown_parser_extracts_h1_and_sections() -> None:
    """Verify MarkdownParser extracts document H1 title and section headers."""
    parser = MarkdownParser()
    md_content = """# Customer Refund Policy

## Eligibility Window
Customers may request a full refund within 30 calendar days of delivery.

## Excluded Items
Digital gift cards and personalized goods are strictly non-refundable.
"""
    raw_bytes = md_content.encode("utf-8")
    parsed = await parser.parse(raw_bytes, "refund_sop.md")

    assert parsed.title == "Customer Refund Policy"
    assert "Customer Refund Policy" in parsed.content
    assert "Eligibility Window" in parsed.metadata["sections"]
    assert "Excluded Items" in parsed.metadata["sections"]


@pytest.mark.asyncio
async def test_pdf_parser_with_in_memory_pdf() -> None:
    """Verify PDFParser parses PDF binary stream and extracts pages."""
    parser = PDFParser()

    # Create a minimal valid PDF in-memory using pypdf
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    pdf_buffer = io.BytesIO()
    writer.write(pdf_buffer)
    pdf_bytes = pdf_buffer.getvalue()

    parsed = await parser.parse(pdf_bytes, "company_handbook.pdf")

    assert parsed.title == "Company Handbook"
    assert parsed.metadata["source_type"] == "pdf"
    assert parsed.metadata["total_pages"] == 1


def test_parser_factory_resolution() -> None:
    """Verify get_parser_for_filename resolves correct parser types."""
    assert isinstance(get_parser_for_filename("doc.pdf"), PDFParser)
    assert isinstance(get_parser_for_filename("notes.md"), MarkdownParser)
    assert isinstance(get_parser_for_filename("guide.markdown"), MarkdownParser)
    assert isinstance(get_parser_for_filename("data.txt"), PlainTextParser)
    assert isinstance(get_parser_for_filename("unknown.xyz"), PlainTextParser)
