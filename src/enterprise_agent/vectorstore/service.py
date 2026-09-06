"""Vector retrieval service orchestrating embedding generation, indexing, and vector search."""

from typing import Any

from enterprise_agent.core.logging import get_logger
from enterprise_agent.embeddings.service import EmbeddingsService
from enterprise_agent.ingestion.models import DocumentChunk
from enterprise_agent.schemas.search import (
    IndexDocumentResponse,
    SearchResultItem,
    SemanticSearchResponse,
)
from enterprise_agent.vectorstore.base import VectorStore

logger = get_logger(__name__)


class VectorSearchService:
    """Orchestrates vector indexing and dense semantic retrieval operations."""

    def __init__(
        self,
        vector_store: VectorStore,
        embeddings_service: EmbeddingsService,
    ) -> None:
        self.vector_store = vector_store
        self.embeddings_service = embeddings_service

    async def initialize_collection(self) -> None:
        """Ensure the vector collection is provisioned with proper dimensions."""
        dimensions = self.embeddings_service.dimensions
        await self.vector_store.create_collection_if_not_exists(dimensions=dimensions)

    async def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Embed document chunks and index them into the vector database."""
        if not chunks:
            return 0

        # Provision collection if not yet created
        await self.initialize_collection()

        logger.info("Generating embeddings for %d document chunks", len(chunks))
        _, vectors = await self.embeddings_service.embed_document_chunks(chunks)

        logger.info("Upserting %d vectors into vector store", len(vectors))
        count = await self.vector_store.upsert(chunks=chunks, vectors=vectors)
        return count

    async def index_document(
        self,
        document_id: str,
        chunks: list[DocumentChunk],
    ) -> IndexDocumentResponse:
        """Index all chunks for a specific document and return summary response."""
        count = await self.index_chunks(chunks)
        return IndexDocumentResponse(
            document_id=document_id,
            chunks_indexed=count,
            status="success",
            message=f"Successfully indexed {count} chunks for document {document_id}.",
        )

    async def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> SemanticSearchResponse:
        """Embed search query and retrieve top-k semantically relevant chunks."""
        # Provision collection if needed
        await self.initialize_collection()

        logger.debug("Generating query embedding for: '%s'", query)
        query_vector = await self.embeddings_service.embed_query(query)

        logger.debug("Querying vector store (top_k=%d, min_score=%.2f)", top_k, min_score)
        search_results = await self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            min_score=min_score,
        )

        items = [
            SearchResultItem(
                chunk_id=res.chunk_id,
                document_id=res.document_id,
                content=res.content,
                score=res.score,
                chunk_index=res.chunk_index,
                metadata=res.metadata,
            )
            for res in search_results
        ]

        return SemanticSearchResponse(
            query=query,
            total_results=len(items),
            results=items,
        )
