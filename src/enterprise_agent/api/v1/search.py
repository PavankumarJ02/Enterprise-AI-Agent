"""API routes for semantic vector search and index operations."""

from fastapi import APIRouter, Depends, HTTPException, status

from enterprise_agent.api.deps import (
    get_hybrid_search_service,
    get_two_stage_retrieval_service,
    get_vector_search_service,
)
from enterprise_agent.core.exceptions import RerankingError, VectorStoreError
from enterprise_agent.core.logging import get_logger
from enterprise_agent.hybrid.service import HybridSearchService
from enterprise_agent.reranking.service import TwoStageRetrievalService
from enterprise_agent.schemas.search import (
    DirectRerankRequest,
    HybridSearchRequest,
    HybridSearchResponse,
    RerankRequest,
    RerankResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SparseSearchRequest,
    SparseSearchResponse,
)
from enterprise_agent.vectorstore.service import VectorSearchService

logger = get_logger(__name__)

router = APIRouter(prefix="/search", tags=["Semantic Search"])


@router.post(
    "/semantic",
    response_model=SemanticSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic vector retrieval",
    description=(
        "Embeds natural language query and performs ANN similarity search "
        "against indexed document chunks."
    ),
)
async def semantic_search(
    request: SemanticSearchRequest,
    service: VectorSearchService = Depends(get_vector_search_service),
) -> SemanticSearchResponse:
    """Execute dense semantic search using high-dimensional cosine similarity."""
    try:
        response = await service.semantic_search(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            min_score=request.min_score,
        )
        return response
    except VectorStoreError as vse:
        logger.error("Vector store error during semantic search: %s", vse)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector store search failed: {vse}",
        ) from vse
    except Exception as e:
        logger.error("Unexpected error in semantic search: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search operation failed: {e}",
        ) from e


@router.post(
    "/sparse",
    response_model=SparseSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Sparse BM25 keyword retrieval",
    description="Performs exact term and BM25 lexical search against indexed document chunks.",
)
async def sparse_search(
    request: SparseSearchRequest,
    hybrid_service: HybridSearchService = Depends(get_hybrid_search_service),
) -> SparseSearchResponse:
    """Execute sparse BM25 keyword search."""
    try:
        results = await hybrid_service.search_sparse_only(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            min_score=request.min_score,
        )
        return SparseSearchResponse(
            query=request.query,
            total_results=len(results),
            results=results,
        )
    except Exception as e:
        logger.error("Error during sparse search: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sparse search operation failed: {e}",
        ) from e


@router.post(
    "/hybrid",
    response_model=HybridSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Hybrid dense + sparse retrieval with rank fusion",
    description=(
        "Concurrently executes dense semantic search and sparse BM25 keyword search, "
        "merging candidate lists via Reciprocal Rank Fusion (RRF) or linear score combination."
    ),
)
async def hybrid_search(
    request: HybridSearchRequest,
    hybrid_service: HybridSearchService = Depends(get_hybrid_search_service),
) -> HybridSearchResponse:
    """Execute hybrid retrieval unifying dense vector and sparse lexical rankings."""
    try:
        response = await hybrid_service.search_by_request(request)
        return response
    except Exception as e:
        logger.error("Error during hybrid search: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid search operation failed: {e}",
        ) from e


@router.post(
    "/rerank",
    response_model=RerankResponse,
    status_code=status.HTTP_200_OK,
    summary="Two-stage retrieval: coarse candidate recall + cross-encoder reranking",
    description=(
        "Retrieves a candidate pool (K1) via hybrid, semantic, or sparse search, "
        "then applies a fine-grained cross-encoder transformer to re-score and re-sort "
        "the candidates, returning the top K2 most relevant chunks."
    ),
)
async def rerank_search(
    request: RerankRequest,
    two_stage_service: TwoStageRetrievalService = Depends(get_two_stage_retrieval_service),
) -> RerankResponse:
    """Execute two-stage coarse retrieval and cross-encoder reranking."""
    try:
        response = await two_stage_service.retrieve_and_rerank(request)
        return response
    except RerankingError as re_err:
        logger.error("Reranking error: %s", re_err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reranking failure: {re_err}",
        ) from re_err
    except Exception as e:
        logger.error("Error during two-stage rerank retrieval: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Two-stage retrieval failed: {e}",
        ) from e


@router.post(
    "/rerank/direct",
    response_model=RerankResponse,
    status_code=status.HTTP_200_OK,
    summary="Direct cross-encoder reranking of provided candidate items",
    description=(
        "Re-scores and re-ranks an explicitly provided list of candidates "
        "using the cross-encoder model."
    ),
)
async def direct_rerank(
    request: DirectRerankRequest,
    two_stage_service: TwoStageRetrievalService = Depends(get_two_stage_retrieval_service),
) -> RerankResponse:
    """Directly rerank provided search result items."""
    try:
        response = await two_stage_service.direct_rerank(request)
        return response
    except RerankingError as re_err:
        logger.error("Reranking error in direct rerank: %s", re_err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Direct reranking failure: {re_err}",
        ) from re_err
    except Exception as e:
        logger.error("Error during direct rerank: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Direct reranking failed: {e}",
        ) from e
