"""Hybrid search service coordinating concurrent dense and sparse retrieval with rank fusion."""

import asyncio
from typing import Any

from enterprise_agent.core.logging import get_logger
from enterprise_agent.hybrid.fusion import linear_score_fusion, reciprocal_rank_fusion
from enterprise_agent.ingestion.models import DocumentChunk
from enterprise_agent.schemas.search import (
    HybridSearchRequest,
    HybridSearchResponse,
    HybridSearchResultItem,
    SearchResultItem,
)
from enterprise_agent.sparse.base import SparseStore
from enterprise_agent.vectorstore.service import VectorSearchService

logger = get_logger(__name__)


class HybridSearchService:
    """Orchestrates multi-modal retrieval across dense vector space and sparse lexical index."""

    def __init__(
        self,
        vector_service: VectorSearchService,
        sparse_store: SparseStore,
        default_fusion_method: str = "rrf",
        default_dense_weight: float = 0.5,
        default_sparse_weight: float = 0.5,
        default_rrf_k: int = 60,
    ) -> None:
        self.vector_service = vector_service
        self.sparse_store = sparse_store
        self.default_fusion_method = default_fusion_method
        self.default_dense_weight = default_dense_weight
        self.default_sparse_weight = default_sparse_weight
        self.default_rrf_k = default_rrf_k

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
        fusion_method: str | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        rrf_k: int | None = None,
    ) -> HybridSearchResponse:
        """Execute concurrent dense and sparse queries and combine results using rank fusion."""
        method = (fusion_method or self.default_fusion_method).lower()
        w_dense = dense_weight if dense_weight is not None else self.default_dense_weight
        w_sparse = sparse_weight if sparse_weight is not None else self.default_sparse_weight
        k_val = rrf_k if rrf_k is not None else self.default_rrf_k

        # Retrieve more candidates (e.g. 2x top_k) for optimal fusion coverage
        retrieval_limit = max(top_k * 2, 20)

        logger.debug(
            "Executing hybrid search: '%s' (fusion=%s, w_dense=%.2f, w_sparse=%.2f)",
            query,
            method,
            w_dense,
            w_sparse,
        )

        # 1. Execute dense and sparse retrieval concurrently
        dense_task = self.vector_service.semantic_search(
            query=query,
            top_k=retrieval_limit,
            filters=filters,
            min_score=0.0,
        )
        sparse_task = self.sparse_store.search(
            query=query,
            top_k=retrieval_limit,
            filters=filters,
            min_score=0.0,
        )

        dense_res, sparse_res = await asyncio.gather(dense_task, sparse_task)

        # 2. Fuse ranked candidate lists
        if method == "linear":
            fused: list[HybridSearchResultItem] = linear_score_fusion(
                dense_results=dense_res.results,
                sparse_results=sparse_res,
                dense_weight=w_dense,
                sparse_weight=w_sparse,
            )
        else:
            fused = reciprocal_rank_fusion(
                dense_results=dense_res.results,
                sparse_results=sparse_res,
                k=k_val,
                dense_weight=w_dense,
                sparse_weight=w_sparse,
            )

        # 3. Apply min_score filter and slice to top_k
        filtered_results = [r for r in fused if r.combined_score >= min_score][:top_k]

        logger.info(
            "Hybrid search complete: %d fused results (dense=%d, sparse=%d)",
            len(filtered_results),
            len(dense_res.results),
            len(sparse_res),
        )

        return HybridSearchResponse(
            query=query,
            total_results=len(filtered_results),
            fusion_method=method,
            results=filtered_results,
        )

    async def search_by_request(self, request: HybridSearchRequest) -> HybridSearchResponse:
        """Helper to invoke search using a Pydantic HybridSearchRequest payload."""
        return await self.search(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            min_score=request.min_score,
            fusion_method=request.fusion_method,
            dense_weight=request.dense_weight,
            sparse_weight=request.sparse_weight,
            rrf_k=request.rrf_k,
        )

    async def search_sparse_only(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResultItem]:
        """Direct access to sparse keyword search."""
        return await self.sparse_store.search(
            query=query,
            top_k=top_k,
            filters=filters,
            min_score=min_score,
        )

    async def index_chunks(self, chunks: list[DocumentChunk]) -> tuple[int, int]:
        """Dual index chunks across both dense vector store and sparse keyword store."""
        dense_task = self.vector_service.index_chunks(chunks)
        sparse_task = self.sparse_store.index_chunks(chunks)

        dense_count, sparse_count = await asyncio.gather(dense_task, sparse_task)
        logger.info(
            "Dual indexing complete: %d dense vectors, %d sparse keyword entries",
            dense_count,
            sparse_count,
        )
        return dense_count, sparse_count

    async def delete_by_document_id(self, document_id: str) -> tuple[int, int]:
        """Dual purge chunks from both dense and sparse stores."""
        dense_task = self.vector_service.vector_store.delete_by_document_id(document_id)
        sparse_task = self.sparse_store.delete_by_document_id(document_id)

        dense_count, sparse_count = await asyncio.gather(dense_task, sparse_task)
        logger.info(
            "Dual deletion complete for doc %s: %d dense, %d sparse",
            document_id,
            dense_count,
            sparse_count,
        )
        return dense_count, sparse_count
