"""Thread-safe LRU and TTL memory caches for query embeddings and retrieval results."""

import threading
import time
from collections import OrderedDict
from typing import Any, Generic, TypeVar

from enterprise_agent.core.logging import get_logger

logger = get_logger(__name__)

K = TypeVar("K")
V = TypeVar("V")


class TTLCache(Generic[K, V]):
    """Thread-safe, generic Least-Recently-Used (LRU) cache with Time-To-Live (TTL) expiration."""

    def __init__(self, max_size: int = 1024, default_ttl_seconds: float = 3600.0) -> None:
        self.max_size = max(1, max_size)
        self.default_ttl = max(0.1, default_ttl_seconds)
        self._cache: OrderedDict[K, tuple[V, float]] = OrderedDict()
        self._lock = threading.RLock()

        # Telemetry metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: K) -> V | None:
        """Retrieve cached value if present and unexpired, updating LRU recency."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, expiry = self._cache[key]
            now = time.monotonic()

            if now > expiry:
                # Expired item: evict and record as miss
                del self._cache[key]
                self._misses += 1
                return None

            # Mark as most recently used
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: K, value: V, ttl_seconds: float | None = None) -> None:
        """Store or update key-value pair, enforcing capacity constraints via LRU eviction."""
        with self._lock:
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
            expiry = time.monotonic() + max(0.1, ttl)

            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self.max_size:
                # Evict least recently used (first item)
                self._cache.popitem(last=False)
                self._evictions += 1

            self._cache[key] = (value, expiry)

    def invalidate(self, key: K) -> bool:
        """Explicitly remove a key from the cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Flush all cached entries."""
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        """Current count of active entries."""
        with self._lock:
            return len(self._cache)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def evictions(self) -> int:
        return self._evictions

    @property
    def hit_ratio(self) -> float:
        """Calculate hit percentage between 0.0 and 1.0."""
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> dict[str, Any]:
        """Return operational telemetry summary."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_ratio": round(self.hit_ratio, 4),
            }


class QueryEmbeddingCache:
    """Specialized cache for dense query vector embeddings."""

    def __init__(self, max_size: int = 1024, ttl_seconds: float = 3600.0) -> None:
        self._cache: TTLCache[str, list[float]] = TTLCache(
            max_size=max_size,
            default_ttl_seconds=ttl_seconds,
        )

    def _make_key(self, text: str, model: str, dimensions: int) -> str:
        return f"{model.strip()}:{dimensions}:{text.strip().lower()}"

    def get_embedding(self, text: str, model: str, dimensions: int) -> list[float] | None:
        """Fetch cached vector embedding if present."""
        key = self._make_key(text, model, dimensions)
        return self._cache.get(key)

    def set_embedding(
        self,
        text: str,
        model: str,
        dimensions: int,
        vector: list[float],
        ttl_seconds: float | None = None,
    ) -> None:
        """Store vector embedding in cache."""
        key = self._make_key(text, model, dimensions)
        self._cache.set(key, vector, ttl_seconds=ttl_seconds)

    def clear(self) -> None:
        self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        return self._cache.get_stats()


class RetrievalCache:
    """Specialized cache for two-stage hybrid search and reranked passage results."""

    def __init__(self, max_size: int = 1024, ttl_seconds: float = 1800.0) -> None:
        self._cache: TTLCache[str, Any] = TTLCache(
            max_size=max_size,
            default_ttl_seconds=ttl_seconds,
        )

    def _make_key(self, query: str, top_k: int, alpha: float | None = None) -> str:
        alpha_str = f"{alpha:.2f}" if alpha is not None else "default"
        return f"{query.strip().lower()}:k={top_k}:alpha={alpha_str}"

    def get_retrieval(
        self,
        query: str,
        top_k: int,
        alpha: float | None = None,
    ) -> Any | None:
        """Fetch cached retrieval passages if present."""
        key = self._make_key(query, top_k, alpha)
        return self._cache.get(key)

    def set_retrieval(
        self,
        query: str,
        top_k: int,
        results: Any,
        alpha: float | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        """Store retrieval passages in cache."""
        key = self._make_key(query, top_k, alpha)
        self._cache.set(key, results, ttl_seconds=ttl_seconds)

    def clear(self) -> None:
        self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        return self._cache.get_stats()
