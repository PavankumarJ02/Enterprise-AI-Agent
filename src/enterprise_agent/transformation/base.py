"""Base interface contract for Query Transformation strategies."""

from abc import ABC, abstractmethod
from typing import Any


class QueryTransformer(ABC):
    """Abstract contract for transforming or expanding user search queries."""

    @abstractmethod
    async def transform(self, query: str, **kwargs: Any) -> list[str]:
        """Transform an input query into one or more reformulated queries or hypothetical passages.

        Args:
            query: Original user natural language query string.
            **kwargs: Strategy-specific parameters (e.g., num_variations, temperature).

        Returns:
            List of transformed query strings or synthetic document passages.
        """
        ...
