"""Dense vector embeddings API endpoints."""

import time

from fastapi import APIRouter, Depends, status

from enterprise_agent.api.deps import get_embeddings
from enterprise_agent.core.logging import get_logger
from enterprise_agent.embeddings.service import EmbeddingsService
from enterprise_agent.schemas.embeddings import (
    EmbedQueryRequest,
    EmbedQueryResponse,
    EmbedTextsRequest,
    EmbedTextsResponse,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])


@router.post(
    "",
    response_model=EmbedTextsResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch Embed Texts",
    description=(
        "Vectorize a batch of text passages or document chunks with automatic batch slicing."
    ),
)
async def embed_texts_endpoint(
    request: EmbedTextsRequest,
    service: EmbeddingsService = Depends(get_embeddings),
) -> EmbedTextsResponse:
    """Generate dense embeddings for a batch of texts."""
    start_time = time.perf_counter()
    res = await service.embed_texts(request.texts)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    logger.info(
        "Batch embedded %d texts [model=%s, dimensions=%d, latency=%.2fms]",
        len(request.texts),
        res.model,
        res.dimensions,
        latency_ms,
    )

    return EmbedTextsResponse(
        vectors=res.vectors,
        dimensions=res.dimensions,
        count=len(res.vectors),
        model=res.model,
        token_count=res.token_count,
        latency_ms=latency_ms,
    )


@router.post(
    "/query",
    response_model=EmbedQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Embed Search Query",
    description=(
        "Generate an asymmetric dense vector embedding optimized for query-passage retrieval."
    ),
)
async def embed_query_endpoint(
    request: EmbedQueryRequest,
    service: EmbeddingsService = Depends(get_embeddings),
) -> EmbedQueryResponse:
    """Generate an asymmetric retrieval vector for a search query."""
    start_time = time.perf_counter()
    vector = await service.embed_query(request.query)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return EmbedQueryResponse(
        vector=vector,
        dimensions=len(vector),
        model=service.provider.__class__.__name__,
        latency_ms=latency_ms,
    )
