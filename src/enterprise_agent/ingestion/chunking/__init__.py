"""Document chunking strategies package."""

from enterprise_agent.ingestion.chunking.base import BaseChunker
from enterprise_agent.ingestion.chunking.recursive import RecursiveCharacterChunker

__all__ = ["BaseChunker", "RecursiveCharacterChunker"]
