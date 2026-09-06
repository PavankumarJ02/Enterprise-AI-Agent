"""Security headers middleware enforcing OWASP defensive HTTP headers."""

from collections.abc import Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware attaching OWASP-recommended HTTP security headers to all responses."""

    def __init__(self, app: Any, enabled: bool = True) -> None:
        super().__init__(app)
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        response: Response = await call_next(request)

        if not self.enabled:
            return response

        headers = response.headers
        headers["X-Content-Type-Options"] = "nosniff"
        headers["X-Frame-Options"] = "DENY"
        headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"

        # Permit Swagger UI / ReDoc CDN assets for interactive documentation paths
        if not any(request.url.path.startswith(p) for p in ("/docs", "/redoc", "/openapi.json")):
            headers["Content-Security-Policy"] = "default-src 'self'"

        return response
