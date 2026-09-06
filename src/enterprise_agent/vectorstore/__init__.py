"""Vector store package for high-dimensional semantic search and chunk indexing."""

from enterprise_agent.vectorstore.base import SearchResult, VectorStore
from enterprise_agent.vectorstore.factory import create_qdrant_client, create_vector_store
from enterprise_agent.vectorstore.qdrant import QdrantVectorStore
from enterprise_agent.vectorstore.service import VectorSearchService

__all__ = [
    "SearchResult",
    "VectorStore",
    "QdrantVectorStore",
    "create_qdrant_client",
    "create_vector_store",
    "VectorSearchService",
]
