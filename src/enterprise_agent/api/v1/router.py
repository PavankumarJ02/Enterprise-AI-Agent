"""API routes for Semantic Query Routing and multi-engine dispatch."""

from fastapi import APIRouter, Depends, HTTPException, status

from enterprise_agent.api.deps import get_query_router_service
from enterprise_agent.core.logging import get_logger
from enterprise_agent.router.service import QueryRouterService
from enterprise_agent.schemas.router import (
    RouterClassifyRequest,
    RouterClassifyResponse,
    RouterDispatchRequest,
    RouterDispatchResponse,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/router", tags=["Semantic Query Router"])


@router.post(
    "/classify",
    response_model=RouterClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify query intent across cascade",
    description=(
        "Analyzes user input using a 3-tier cascade (Heuristics -> Semantic Embedding "
        "-> Zero-shot LLM) to detect target execution intent."
    ),
)
async def classify_query(
    request: RouterClassifyRequest,
    router_service: QueryRouterService = Depends(get_query_router_service),
) -> RouterClassifyResponse:
    """Classify the intent of a user query."""
    try:
        return await router_service.classify(request.query)
    except Exception as exc:
        logger.error("Intent classification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query classification error: {exc}",
        ) from exc


@router.post(
    "/dispatch",
    response_model=RouterDispatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify and dispatch query to optimal backend",
    description=(
        "Routes query to the best-matched subsystem (Direct Chat, RAG, SQL, or Agent), "
        "executes the query, and returns the response alongside routing telemetry."
    ),
)
async def dispatch_query(
    request: RouterDispatchRequest,
    router_service: QueryRouterService = Depends(get_query_router_service),
) -> RouterDispatchResponse:
    """Classify and execute query against designated backend."""
    try:
        return await router_service.dispatch(request)
    except Exception as exc:
        logger.error("Query routing dispatch failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query dispatch error: {exc}",
        ) from exc
