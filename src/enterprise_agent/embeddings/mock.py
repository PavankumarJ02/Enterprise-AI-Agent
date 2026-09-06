"""Deterministic mock embedding provider for offline testing and fast CI runs."""

import hashlib
import math

from enterprise_agent.embeddings.base import EmbeddingProvider, EmbeddingResult


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic, unit-normalized mock embedding generator."""

    def __init__(
        self,
        dimensions: int = 768,
        model_name: str = "mock-embedding-004",
    ) -> None:
        self._dimensions = dimensions
        self.model_name = model_name
        self.invocations: list[list[str]] = []

    @property
    def dimensions(self) -> int:
        """Return configured dimensionality."""
        return self._dimensions

    def _generate_vector(self, text: str) -> list[float]:
        """Generate deterministic pseudo-random unit-normalized vector from text hash."""
        # Use SHA-256 hash to seed deterministic values
        h = hashlib.sha256(text.encode("utf-8")).digest()
        raw_vals: list[float] = []

        # Generate floats from rolling hash bytes
        for i in range(self._dimensions):
            byte_val = h[i % len(h)]
            # Map byte [0, 255] to float in [-1.0, 1.0] with variation based on index
            val = ((byte_val ^ (i % 256)) / 127.5) - 1.0
            raw_vals.append(val)

        # L2-normalize vector: v / ||v||
        norm = math.sqrt(sum(x * x for x in raw_vals))
        if norm == 0.0:
            return [1.0 / math.sqrt(self._dimensions)] * self._dimensions
        return [round(x / norm, 6) for x in raw_vals]

    async def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        """Embed a list of text strings deterministically."""
        self.invocations.append(texts)
        vectors = [self._generate_vector(t) for t in texts]
        total_tokens = sum(max(1, len(t.split())) for t in texts)

        return EmbeddingResult(
            vectors=vectors,
            model=self.model_name,
            dimensions=self._dimensions,
            token_count=total_tokens,
        )

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query deterministically."""
        self.invocations.append([query])
        return self._generate_vector(query)

    async def health_check(self) -> bool:
        """Mock provider is always healthy."""
        return True
