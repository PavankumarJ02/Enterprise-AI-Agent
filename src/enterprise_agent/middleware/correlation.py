"""HTTP Correlation ID middleware and execution context management."""

import uuid
from collections.abc import Callable
from contextvars import ContextVar, Token
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Context variable binding correlation ID to the current async execution task
_current_correlation_id: ContextVar[str | None] = ContextVar("current_correlation_id", default=None)

CORRELATION_HEADER = "X-Correlation-ID"
REQUEST_ID_HEADER = "X-Request-ID"


def get_correlation_id() -> str | None:
    """Retrieve active correlation ID from context."""
    return _current_correlation_id.get()


def set_correlation_id(cid: str | None) -> Token[str | None]:
    """Bind correlation ID to the current context."""
    return _current_correlation_id.set(cid)


def reset_correlation_id(token: Token[str | None]) -> None:
    """Reset context token."""
    _current_correlation_id.reset(token)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware extracting, generating, and propagating request correlation IDs."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        # Extract existing correlation ID from headers or generate new one
        correlation_id = (
            request.headers.get(CORRELATION_HEADER)
            or request.headers.get(REQUEST_ID_HEADER)
            or uuid.uuid4().hex
        )

        token = set_correlation_id(correlation_id)
        request.state.correlation_id = correlation_id

        try:
            response: Response = await call_next(request)
            response.headers[CORRELATION_HEADER] = correlation_id
            return response
        finally:
            reset_correlation_id(token)
