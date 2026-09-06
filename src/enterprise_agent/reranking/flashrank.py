"""FlashRank ONNX-based Cross-Encoder Reranker implementation."""

import asyncio
from typing import Any

from flashrank import Ranker, RerankRequest

from enterprise_agent.core.exceptions import RerankingError
from enterprise_agent.core.logging import get_logger
from enterprise_agent.reranking.base import Reranker
from enterprise_agent.schemas.search import RerankResultItem, SearchResultItem

logger = get_logger(__name__)


class FlashRankReranker(Reranker):
    """Ultra-lightweight Cross-Encoder using FlashRank and ONNX Runtime CPU inference."""

    def __init__(
        self,
        model_name: str = "ms-marco-TinyBERT-L-2-v2",
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model_name
        self.cache_dir = cache_dir
        logger.info("Initializing FlashRank Cross-Encoder [model=%s]", self._model_name)
        try:
            if self.cache_dir:
                self._ranker = Ranker(model_name=self._model_name, cache_dir=self.cache_dir)
            else:
                self._ranker = Ranker(model_name=self._model_name)
        except Exception as exc:
            logger.error("Failed to initialize FlashRank ranker: %s", exc)
            raise RerankingError(
                f"FlashRank initialization failed for model '{self._model_name}': {exc}"
            ) from exc

    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return self._model_name

    def _sync_rerank(self, request: RerankRequest) -> list[dict[str, Any]]:
        """Synchronous CPU inference call to FlashRank."""
        results: list[dict[str, Any]] = self._ranker.rerank(request)
        return results

    async def rerank(
        self,
        query: str,
        items: list[SearchResultItem],
        top_k: int = 5,
    ) -> list[RerankResultItem]:
        """Execute cross-encoder token-level reranking asynchronously in thread pool.

        Args:
            query: User's natural language query.
            items: Stage 1 candidate passages.
            top_k: Maximum number of top passages to return.

        Returns:
            Sorted list of RerankResultItem with fine-grained cross-attention scores.
        """
        if not items:
            return []

        # Map chunks by unique chunk_id to retain original attributes and initial rank
        chunk_lookup: dict[str, tuple[int, SearchResultItem]] = {
            item.chunk_id: (idx, item) for idx, item in enumerate(items, start=1)
        }

        # Format input passages for FlashRank
        passages = [{"id": item.chunk_id, "text": item.content} for item in items]
        req = RerankRequest(query=query, passages=passages)

        try:
            # Execute CPU-bound inference in threadpool to avoid blocking FastAPI event loop
            raw_results = await asyncio.to_thread(self._sync_rerank, req)
        except Exception as exc:
            logger.error("FlashRank inference error during reranking: %s", exc)
            raise RerankingError(f"FlashRank inference failed: {exc}") from exc

        # Construct enriched rerank results with initial and final rank telemetry
        reranked_results: list[RerankResultItem] = []
        for final_idx, res in enumerate(raw_results[:top_k], start=1):
            chunk_id = str(res.get("id", ""))
            if chunk_id not in chunk_lookup:
                continue

            initial_rank, item = chunk_lookup[chunk_id]
            raw_score = res.get("score", 0.0)
            score = round(float(raw_score), 6)

            reranked_results.append(
                RerankResultItem(
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    content=item.content,
                    score=item.score,
                    chunk_index=item.chunk_index,
                    metadata=item.metadata,
                    rerank_score=score,
                    initial_rank=initial_rank,
                    final_rank=final_idx,
                    initial_score=item.score,
                )
            )

        return reranked_results
