"""Abstract Base Class for Cross-Encoder Rerankers."""

from abc import ABC, abstractmethod

from enterprise_agent.schemas.search import RerankResultItem, SearchResultItem


class Reranker(ABC):
    """Interface contract for cross-encoder reranking models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the underlying model identifier."""
        ...

    @abstractmethod
    async def rerank(
        self,
        query: str,
        items: list[SearchResultItem],
        top_k: int = 5,
    ) -> list[RerankResultItem]:
        """Compute fine-grained cross-attention relevance scores and reorder candidates.

        Args:
            query: The natural language search query.
            items: The candidate search results retrieved from Stage 1.
            top_k: Maximum number of reordered results to return.

        Returns:
            List of RerankResultItem sorted in descending order of rerank_score.
        """
        ...
