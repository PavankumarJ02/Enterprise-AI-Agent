"""API routes for query transformation strategies (HyDE, Multi-Query, Step-Back)."""

from fastapi import APIRouter, Depends, HTTPException, status

from enterprise_agent.api.deps import get_query_transformation_service
from enterprise_agent.core.logging import get_logger
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
from enterprise_agent.transformation.service import QueryTransformationService

logger = get_logger(__name__)

router = APIRouter(prefix="/transform", tags=["Query Transformation"])


@router.post(
    "/hyde",
    response_model=HyDETransformResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate HyDE hypothetical document",
    description="Generates synthetic enterprise documentation passage answering the query.",
)
async def transform_hyde(
    request: HyDETransformRequest,
    service: QueryTransformationService = Depends(get_query_transformation_service),
) -> HyDETransformResponse:
    """Synthesize hypothetical answer passage for vector query alignment."""
    try:
        return await service.transform_hyde(request)
    except Exception as exc:
        logger.error("HyDE transformation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"HyDE generation failed: {exc}",
        ) from exc


@router.post(
    "/multi-query",
    response_model=MultiQueryTransformResponse,
    status_code=status.HTTP_200_OK,
    summary="Decompose query into Multi-Query perspectives",
    description="Expands query into multiple alternative phrasing angles and technical synonyms.",
)
async def transform_multi_query(
    request: MultiQueryTransformRequest,
    service: QueryTransformationService = Depends(get_query_transformation_service),
) -> MultiQueryTransformResponse:
    """Generate multi-query reformulations."""
    try:
        return await service.transform_multi_query(request)
    except Exception as exc:
        logger.error("Multi-query transformation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multi-query expansion failed: {exc}",
        ) from exc


@router.post(
    "/step-back",
    response_model=StepBackTransformResponse,
    status_code=status.HTTP_200_OK,
    summary="Abstract specific query into Step-Back concept question",
    description="Derives a higher-level, foundational question about core principles.",
)
async def transform_step_back(
    request: StepBackTransformRequest,
    service: QueryTransformationService = Depends(get_query_transformation_service),
) -> StepBackTransformResponse:
    """Abstract specific problem into high-level conceptual question."""
    try:
        return await service.transform_step_back(request)
    except Exception as exc:
        logger.error("Step-Back transformation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Step-Back abstraction failed: {exc}",
        ) from exc


@router.post(
    "/search",
    response_model=TransformedSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="End-to-end transformed query retrieval",
    description=(
        "Applies selected transformation strategy (HyDE, Multi-Query, or Step-Back), "
        "retrieves candidates across dense/hybrid stores, and optionally reranks "
        "with Cross-Encoder."
    ),
)
async def search_with_transformation(
    request: TransformedSearchRequest,
    service: QueryTransformationService = Depends(get_query_transformation_service),
) -> TransformedSearchResponse:
    """Execute end-to-end retrieval with query transformation and optional reranking."""
    try:
        return await service.search_with_transformation(request)
    except Exception as exc:
        logger.error("Transformed search failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transformed search failed: {exc}",
        ) from exc
