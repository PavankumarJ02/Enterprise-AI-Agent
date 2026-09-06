"""Mock Reranker for deterministic unit and integration testing."""

import re

from enterprise_agent.reranking.base import Reranker
from enterprise_agent.schemas.search import RerankResultItem, SearchResultItem


class MockReranker(Reranker):
    """Deterministic reranker using lexical token overlap and configurable score overrides."""

    def __init__(
        self,
        model_name: str = "mock-cross-encoder-v1",
        forced_scores: dict[str, float] | None = None,
    ) -> None:
        self._model_name = model_name
        self.forced_scores = forced_scores or {}

    @property
    def model_name(self) -> str:
        """Return mock model name."""
        return self._model_name

    def set_forced_scores(self, scores: dict[str, float]) -> None:
        """Configure explicit scores keyed by chunk_id for testing."""
        self.forced_scores = scores

    async def rerank(
        self,
        query: str,
        items: list[SearchResultItem],
        top_k: int = 5,
    ) -> list[RerankResultItem]:
        """Score items using token overlap or forced scores, and return top_k reranked results."""
        if not items:
            return []

        query_tokens = set(re.findall(r"\w+", query.lower()))

        scored_items: list[tuple[float, int, SearchResultItem]] = []
        for idx, item in enumerate(items, start=1):
            if item.chunk_id in self.forced_scores:
                score = self.forced_scores[item.chunk_id]
            else:
                # Deterministic overlap heuristic: overlap ratio + fraction of initial score
                doc_tokens = set(re.findall(r"\w+", item.content.lower()))
                overlap = len(query_tokens & doc_tokens)
                overlap_ratio = overlap / max(len(query_tokens), 1)
                score = round(0.7 * overlap_ratio + 0.3 * item.score, 4)

            scored_items.append((score, idx, item))

        # Sort descending by score; break ties by maintaining earlier initial rank
        scored_items.sort(key=lambda x: (x[0], -x[1]), reverse=True)

        reranked: list[RerankResultItem] = []
        for final_idx, (score, initial_rank, item) in enumerate(scored_items[:top_k], start=1):
            reranked.append(
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

        return reranked
