"""Semantic embedding-based query intent classifier using cosine exemplar matching."""

import math

from enterprise_agent.embeddings.base import EmbeddingProvider
from enterprise_agent.router.exemplars import ROUTER_EXEMPLARS
from enterprise_agent.schemas.router import QueryIntent


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


class SemanticEmbeddingRouter:
    """Classifies queries by measuring vector cosine similarity against intent exemplars."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        threshold: float = 0.78,
        exemplars: dict[QueryIntent, list[str]] | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.threshold = threshold
        self.exemplars = exemplars or ROUTER_EXEMPLARS
        self._exemplar_embeddings: dict[QueryIntent, list[list[float]]] = {}
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Lazily embed and cache all exemplar query vectors."""
        if self._initialized:
            return

        for intent, texts in self.exemplars.items():
            res = await self.embedding_provider.embed_texts(texts)
            self._exemplar_embeddings[intent] = res.vectors

        self._initialized = True

    async def classify(self, query: str) -> tuple[QueryIntent, float, str] | None:
        """Classify query using cosine similarity against exemplar embeddings."""
        text = query.strip()
        if not text:
            return None

        await self._ensure_initialized()

        query_vec = await self.embedding_provider.embed_query(text)

        best_intent: QueryIntent = QueryIntent.DIRECT_CHAT
        highest_score = -1.0

        # Calculate max and top-2 average similarity per intent
        for intent, exemplar_vecs in self._exemplar_embeddings.items():
            scores = [_cosine_similarity(query_vec, e_vec) for e_vec in exemplar_vecs]
            if not scores:
                continue
            scores.sort(reverse=True)
            top_score = scores[0]

            if top_score > highest_score:
                highest_score = top_score
                best_intent = intent

        confidence = max(0.0, min(1.0, round(highest_score, 3)))
        if confidence >= self.threshold:
            return (
                best_intent,
                confidence,
                f"Semantic embedding similarity ({confidence:.2f}) matched {best_intent.value}.",
            )

        return None
