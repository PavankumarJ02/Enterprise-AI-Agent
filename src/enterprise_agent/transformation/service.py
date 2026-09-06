"""Query Transformation Service unifying prompt strategies, retrieval, and reranking."""

import asyncio
import time

from enterprise_agent.core.logging import get_logger
from enterprise_agent.hybrid.service import HybridSearchService
from enterprise_agent.reranking.service import TwoStageRetrievalService
from enterprise_agent.schemas.search import SearchResultItem
from enterprise_agent.schemas.transformation import (
    HyDETransformRequest,
    HyDETransformResponse,
    MultiQueryTransformRequest,
    MultiQueryTransformResponse,
    StepBackTransformRequest,
    StepBackTransformResponse,
    TransformedSearchRequest,
    TransformedSearchResponse,
)
from enterprise_agent.transformation.hyde import HyDETransformer
from enterprise_agent.transformation.multi_query import MultiQueryTransformer
from enterprise_agent.transformation.step_back import StepBackTransformer
from enterprise_agent.vectorstore.service import VectorSearchService

logger = get_logger(__name__)


class QueryTransformationService:
    """Coordinates query transformation strategies with hybrid retrieval and reranking."""

    def __init__(
        self,
        hyde_transformer: HyDETransformer,
        multi_query_transformer: MultiQueryTransformer,
        step_back_transformer: StepBackTransformer,
        two_stage_service: TwoStageRetrievalService,
        hybrid_service: HybridSearchService,
        vector_service: VectorSearchService,
    ) -> None:
        self.hyde_transformer = hyde_transformer
        self.multi_query_transformer = multi_query_transformer
        self.step_back_transformer = step_back_transformer
        self.two_stage_service = two_stage_service
        self.hybrid_service = hybrid_service
        self.vector_service = vector_service

    async def transform_hyde(self, request: HyDETransformRequest) -> HyDETransformResponse:
        """Generate hypothetical document answer passages."""
        start = time.perf_counter()
        hypotheses = await self.hyde_transformer.transform(
            query=request.query,
            num_hypotheses=request.num_hypotheses,
        )
        latency = round((time.perf_counter() - start) * 1000, 2)
        return HyDETransformResponse(
            original_query=request.query,
            hypothetical_documents=hypotheses,
            latency_ms=latency,
        )

    async def transform_multi_query(
        self,
        request: MultiQueryTransformRequest,
    ) -> MultiQueryTransformResponse:
        """Decompose query into multiple semantic perspectives."""
        start = time.perf_counter()
        variations = await self.multi_query_transformer.transform(
            query=request.query,
            num_variations=request.num_variations,
        )
        latency = round((time.perf_counter() - start) * 1000, 2)
        return MultiQueryTransformResponse(
            original_query=request.query,
            variations=variations,
            latency_ms=latency,
        )

    async def transform_step_back(
        self,
        request: StepBackTransformRequest,
    ) -> StepBackTransformResponse:
        """Abstract query into a high-level conceptual question."""
        start = time.perf_counter()
        step_back_q, rationale = await self.step_back_transformer.generate_step_back(
            query=request.query
        )
        latency = round((time.perf_counter() - start) * 1000, 2)
        return StepBackTransformResponse(
            original_query=request.query,
            step_back_query=step_back_q,
            rationale=rationale,
            latency_ms=latency,
        )

    async def search_with_transformation(
        self,
        request: TransformedSearchRequest,
    ) -> TransformedSearchResponse:
        """Execute full transformation pipeline followed by retrieval and optional reranking."""
        start = time.perf_counter()
        strategy = request.strategy.lower()
        transformed_queries: list[str] = []
        candidates: list[SearchResultItem] = []

        if strategy == "hyde":
            # 1. Generate hypothetical document
            hypotheses = await self.hyde_transformer.transform(
                query=request.query,
                num_hypotheses=request.num_variations,
            )
            transformed_queries = hypotheses
            hyde_passage = hypotheses[0] if hypotheses else request.query

            # 2. Dense retrieval using hypothetical passage representation
            dense_res = await self.vector_service.semantic_search(
                query=hyde_passage,
                top_k=request.candidate_k,
                filters=request.filters,
            )
            candidates = list(dense_res.results)

        elif strategy == "step_back":
            # 1. Generate step-back query
            step_back_q, _ = await self.step_back_transformer.generate_step_back(request.query)
            transformed_queries = [request.query, step_back_q]

            # 2. Concurrently retrieve for both specific and step-back queries
            task_orig = self.hybrid_service.search(
                query=request.query,
                top_k=request.candidate_k,
                filters=request.filters,
            )
            task_sb = self.hybrid_service.search(
                query=step_back_q,
                top_k=request.candidate_k,
                filters=request.filters,
            )
            res_orig, res_sb = await asyncio.gather(task_orig, task_sb)

            # Deduplicate by chunk_id
            seen_chunks: dict[str, SearchResultItem] = {}
            for item in res_orig.results + res_sb.results:
                if item.chunk_id not in seen_chunks:
                    seen_chunks[item.chunk_id] = item
                elif item.score > seen_chunks[item.chunk_id].score:
                    seen_chunks[item.chunk_id] = item

            candidates = list(seen_chunks.values())

        else:
            # Default: multi_query
            variations = await self.multi_query_transformer.transform(
                query=request.query,
                num_variations=request.num_variations,
            )
            transformed_queries = variations

            # Concurrently search for each query variation
            tasks = [
                self.hybrid_service.search(
                    query=q,
                    top_k=request.candidate_k,
                    filters=request.filters,
                )
                for q in variations
            ]
            responses = await asyncio.gather(*tasks)

            # Deduplicate candidate chunks across all query results
            seen_chunks = {}
            for resp in responses:
                for item in resp.results:
                    if item.chunk_id not in seen_chunks:
                        seen_chunks[item.chunk_id] = item
                    elif item.score > seen_chunks[item.chunk_id].score:
                        seen_chunks[item.chunk_id] = item

            candidates = list(seen_chunks.values())

        # Stage 2: Cross-Encoder Reranking against the ORIGINAL user query
        final_results: list[SearchResultItem] = []
        if request.enable_reranking and candidates:
            reranked = await self.two_stage_service.reranker.rerank(
                query=request.query,
                items=candidates,
                top_k=request.top_k,
            )
            final_results = list(reranked)
        else:
            # Sort descending by initial score and slice to top_k
            candidates.sort(key=lambda x: x.score, reverse=True)
            final_results = candidates[: request.top_k]

        latency = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "Transformed search complete [%s]: %d queries -> %d chunks (latency=%.2fms)",
            strategy,
            len(transformed_queries),
            len(final_results),
            latency,
        )

        return TransformedSearchResponse(
            original_query=request.query,
            strategy=strategy,
            transformed_queries=transformed_queries,
            total_results=len(final_results),
            results=final_results,
            latency_ms=latency,
        )
