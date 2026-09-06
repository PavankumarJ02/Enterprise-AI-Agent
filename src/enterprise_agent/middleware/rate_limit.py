"""Thread-safe sliding-window rate limiting middleware and rate limiter engine."""

import threading
import time
from collections.abc import Callable
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from enterprise_agent.core.logging import get_logger

logger = get_logger(__name__)

EXEMPT_PATHS = ("/health", "/docs", "/redoc", "/openapi.json")


class SlidingWindowRateLimiter:
    """Thread-safe sliding-window rate limiter evaluating requests per rolling window."""

    def __init__(
        self,
        requests_per_minute: int = 120,
        burst_limit: int = 30,
        window_seconds: float = 60.0,
    ) -> None:
        self.requests_per_minute = max(1, requests_per_minute)
        self.burst_limit = max(1, burst_limit)
        self.window_seconds = max(0.01, window_seconds)
        self._lock = threading.Lock()
        # client_key -> list of timestamp floats
        self._records: dict[str, list[float]] = {}

    def check(self, client_key: str) -> tuple[bool, int, int, int]:
        """Evaluate whether a request from client_key is allowed.

        Returns:
            (is_allowed, remaining_requests, reset_after_seconds, retry_after_seconds)
        """
        now = time.time()
        window_start = now - self.window_seconds
        burst_start = now - 1.0

        with self._lock:
            timestamps = self._records.get(client_key, [])

            # Evict timestamps outside the rolling window
            valid_timestamps = [t for t in timestamps if t > window_start]
            self._records[client_key] = valid_timestamps

            # 1. Evaluate 1-second burst limit
            burst_count = sum(1 for t in valid_timestamps if t > burst_start)
            if burst_count >= self.burst_limit:
                retry_after = 1
                reset_after = 1
                return False, 0, reset_after, retry_after

            # 2. Evaluate rolling window limit
            if len(valid_timestamps) >= self.requests_per_minute:
                earliest = valid_timestamps[0]
                retry_after = max(1, int(self.window_seconds - (now - earliest)))
                reset_after = retry_after
                return False, 0, reset_after, retry_after

            # Request is allowed: record timestamp
            valid_timestamps.append(now)
            self._records[client_key] = valid_timestamps

            remaining = max(0, self.requests_per_minute - len(valid_timestamps))
            earliest = valid_timestamps[0]
            reset_after = max(1, int(self.window_seconds - (now - earliest)))
            return True, remaining, reset_after, 0

    def clear(self) -> None:
        """Reset and flush all tracked client rate limit records."""
        with self._lock:
            self._records.clear()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing sliding-window rate limits and injecting rate limit headers."""

    def __init__(
        self,
        app: Any,
        limiter: SlidingWindowRateLimiter | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(app)
        self.limiter = limiter or SlidingWindowRateLimiter()
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        if not self.enabled:
            return await call_next(request)  # type: ignore[no-any-return]

        # Bypass exempt endpoints
        if any(request.url.path.startswith(p) for p in EXEMPT_PATHS):
            return await call_next(request)  # type: ignore[no-any-return]

        # Resolve client identifier: prefer X-API-Key, then client host, then anonymous
        client_key = request.headers.get("X-API-Key") or (
            request.client.host if request.client else "anonymous"
        )

        allowed, remaining, reset_after, retry_after = self.limiter.check(client_key)

        if not allowed:
            logger.warning(
                "Rate limit exceeded for client '%s' on %s [retry_after=%ds]",
                client_key,
                request.url.path,
                retry_after,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded. Please retry in {retry_after} seconds.",
                    "details": {
                        "limit": self.limiter.requests_per_minute,
                        "retry_after_seconds": retry_after,
                    },
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.limiter.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_after),
                },
            )

        response: Response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.limiter.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_after)
        return response
