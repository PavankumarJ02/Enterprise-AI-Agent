"""Two-Stage Retrieval Service coordinating coarse recall and cross-encoder reranking."""

from enterprise_agent.core.logging import get_logger
from enterprise_agent.hybrid.service import HybridSearchService
from enterprise_agent.reranking.base import Reranker
from enterprise_agent.schemas.search import (
    DirectRerankRequest,
    RerankRequest,
    RerankResponse,
    SearchResultItem,
)
from enterprise_agent.vectorstore.service import VectorSearchService

logger = get_logger(__name__)


class TwoStageRetrievalService:
    """Orchestrates Stage 1 coarse candidate recall and Stage 2 Cross-Encoder reranking."""

    def __init__(
        self,
        hybrid_service: HybridSearchService,
        vector_service: VectorSearchService,
        reranker: Reranker,
    ) -> None:
        self.hybrid_service = hybrid_service
        self.vector_service = vector_service
        self.reranker = reranker

    async def retrieve_and_rerank(self, request: RerankRequest) -> RerankResponse:
        """Execute two-stage retrieval: candidate pool -> Cross-Encoder reranking."""
        candidate_k = max(request.candidate_k, request.top_k)
        strategy = request.retrieval_strategy.lower()

        logger.info(
            "Executing two-stage retrieval: '%s' [strategy=%s, candidate_k=%d, top_k=%d]",
            request.query,
            strategy,
            candidate_k,
            request.top_k,
        )

        # Stage 1: Coarse candidate retrieval
        candidates: list[SearchResultItem] = []
        if strategy == "semantic":
            semantic_res = await self.vector_service.semantic_search(
                query=request.query,
                top_k=candidate_k,
                filters=request.filters,
            )
            candidates = list(semantic_res.results)
        elif strategy == "sparse":
            sparse_res = await self.hybrid_service.search_sparse_only(
                query=request.query,
                top_k=candidate_k,
                filters=request.filters,
            )
            candidates = list(sparse_res)
        else:
            # Default to hybrid retrieval
            hybrid_res = await self.hybrid_service.search(
                query=request.query,
                top_k=candidate_k,
                filters=request.filters,
                fusion_method=request.fusion_method,
                dense_weight=request.dense_weight,
                sparse_weight=request.sparse_weight,
                rrf_k=request.rrf_k,
            )
            candidates = list(hybrid_res.results)

        # Short-circuit if no candidates matched Stage 1
        if not candidates:
            logger.info("Zero candidates retrieved in Stage 1 for query: '%s'", request.query)
            return RerankResponse(
                query=request.query,
                total_candidates=0,
                total_results=0,
                retrieval_strategy=strategy,
                model=self.reranker.model_name,
                results=[],
            )

        # Stage 2: Cross-Encoder fine reranking
        reranked = await self.reranker.rerank(
            query=request.query,
            items=candidates,
            top_k=request.top_k,
        )

        logger.info(
            "Two-stage retrieval complete: %d candidates -> %d reranked results [model=%s]",
            len(candidates),
            len(reranked),
            self.reranker.model_name,
        )

        return RerankResponse(
            query=request.query,
            total_candidates=len(candidates),
            total_results=len(reranked),
            retrieval_strategy=strategy,
            model=self.reranker.model_name,
            results=reranked,
        )

    async def direct_rerank(self, request: DirectRerankRequest) -> RerankResponse:
        """Rerank an explicitly provided candidate list without performing store lookups."""
        logger.info(
            "Direct reranking %d candidates for query: '%s'",
            len(request.items),
            request.query,
        )

        if not request.items:
            return RerankResponse(
                query=request.query,
                total_candidates=0,
                total_results=0,
                retrieval_strategy="direct",
                model=self.reranker.model_name,
                results=[],
            )

        reranked = await self.reranker.rerank(
            query=request.query,
            items=request.items,
            top_k=request.top_k,
        )

        return RerankResponse(
            query=request.query,
            total_candidates=len(request.items),
            total_results=len(reranked),
            retrieval_strategy="direct",
            model=self.reranker.model_name,
            results=reranked,
        )
