"""API routes for Retrieval-Augmented Generation (RAG) queries and streaming."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from enterprise_agent.api.deps import get_rag_service
from enterprise_agent.core.exceptions import LLMException, VectorStoreError
from enterprise_agent.core.logging import get_logger
from enterprise_agent.rag.service import RAGService
from enterprise_agent.schemas.rag import (
    RAGQueryRequest,
    RAGQueryResponse,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/rag", tags=["Retrieval-Augmented Generation"])


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Grounded RAG Query",
    description=(
        "Retrieves relevant document passages from the vector database "
        "and synthesizes an authoritative, grounded answer with citations."
    ),
)
async def rag_query(
    request: RAGQueryRequest,
    service: RAGService = Depends(get_rag_service),
) -> RAGQueryResponse:
    """Execute end-to-end grounded RAG retrieval and synthesis."""
    try:
        return await service.query(request)
    except (VectorStoreError, LLMException) as domain_err:
        logger.error("RAG pipeline domain failure: %s", domain_err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"RAG execution failed: {domain_err}",
        ) from domain_err
    except Exception as e:
        logger.error("Unexpected error in RAG pipeline: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error processing RAG query: {e}",
        ) from e


@router.post(
    "/stream",
    status_code=status.HTTP_200_OK,
    summary="Streaming Grounded RAG Query",
    description=(
        "Executes vector retrieval, emits retrieved source chunks via SSE metadata event, "
        "and streams synthesized tokens in real-time."
    ),
)
async def rag_stream(
    request: RAGQueryRequest,
    service: RAGService = Depends(get_rag_service),
) -> StreamingResponse:
    """Stream grounded answer tokens via Server-Sent Events (SSE)."""
    logger.info("Initiating streaming RAG query: '%s'", request.query)

    return StreamingResponse(
        service.query_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
