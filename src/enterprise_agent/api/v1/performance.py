"""Performance and cache telemetry API endpoints."""

from typing import Any

from fastapi import APIRouter, Depends, status

from enterprise_agent.api.deps import (
    get_embedding_cache,
    get_retrieval_cache,
    reset_performance_caches,
)
from enterprise_agent.core.logging import get_logger
from enterprise_agent.performance.cache import QueryEmbeddingCache, RetrievalCache

logger = get_logger(__name__)

router = APIRouter(prefix="/performance", tags=["Performance & Telemetry"])


@router.get(
    "/stats",
    status_code=status.HTTP_200_OK,
    summary="Get Performance Cache Statistics",
    description="Retrieve live telemetry for query embedding and hybrid retrieval caches.",
)
async def get_performance_stats(
    embedding_cache: QueryEmbeddingCache | None = Depends(get_embedding_cache),
    retrieval_cache: RetrievalCache | None = Depends(get_retrieval_cache),
) -> dict[str, Any]:
    """Return aggregated cache statistics and health indicators."""
    emb_stats = (
        embedding_cache.get_stats()
        if embedding_cache is not None
        else {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0,
            "hit_ratio": 0.0,
            "enabled": False,
        }
    )

    ret_stats = (
        retrieval_cache.get_stats()
        if retrieval_cache is not None
        else {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0,
            "hit_ratio": 0.0,
            "enabled": False,
        }
    )

    total_hits = emb_stats.get("hits", 0) + ret_stats.get("hits", 0)
    total_misses = emb_stats.get("misses", 0) + ret_stats.get("misses", 0)
    total_reqs = total_hits + total_misses
    overall_hit_ratio = round((total_hits / total_reqs) * 100, 1) if total_reqs > 0 else 0.0

    return {
        "status": "active",
        "overall_hit_ratio_pct": overall_hit_ratio,
        "total_requests": total_reqs,
        "total_hits": total_hits,
        "total_misses": total_misses,
        "embedding_cache": emb_stats,
        "retrieval_cache": ret_stats,
    }


@router.post(
    "/reset",
    status_code=status.HTTP_200_OK,
    summary="Reset Performance Caches",
    description="Purge all in-memory query embedding and hybrid retrieval cached entries.",
)
async def reset_caches_endpoint(
    embedding_cache: QueryEmbeddingCache | None = Depends(get_embedding_cache),
    retrieval_cache: RetrievalCache | None = Depends(get_retrieval_cache),
) -> dict[str, str]:
    """Purge in-memory performance caches."""
    if embedding_cache is not None:
        embedding_cache.clear()
    if retrieval_cache is not None:
        retrieval_cache.clear()
    reset_performance_caches()
    logger.info("Performance caches cleared via API request.")
    return {"status": "success", "message": "Performance caches successfully cleared."}
