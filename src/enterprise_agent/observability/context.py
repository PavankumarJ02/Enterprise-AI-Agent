"""Thread-safe and async-safe context management for distributed tracing."""

import re
import uuid
from contextvars import ContextVar, Token

# Context variables isolating trace and span identities per async task / thread
_current_trace_id: ContextVar[str | None] = ContextVar("current_trace_id", default=None)
_current_span_id: ContextVar[str | None] = ContextVar("current_span_id", default=None)

# W3C traceparent regex: version-trace_id-parent_id-trace_flags
# e.g., 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
W3C_TRACEPARENT_PATTERN = re.compile(r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")


def generate_trace_id() -> str:
    """Generate a 32-hex-character W3C-compliant trace identifier."""
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generate a 16-hex-character W3C-compliant span identifier."""
    return uuid.uuid4().hex[:16]


def get_current_trace_id() -> str | None:
    """Retrieve active trace ID from execution context."""
    return _current_trace_id.get()


def set_current_trace_id(trace_id: str | None) -> Token[str | None]:
    """Bind trace ID to execution context."""
    return _current_trace_id.set(trace_id)


def reset_current_trace_id(token: Token[str | None]) -> None:
    """Restore previous trace ID context token."""
    _current_trace_id.reset(token)


def get_current_span_id() -> str | None:
    """Retrieve active span ID from execution context."""
    return _current_span_id.get()


def set_current_span_id(span_id: str | None) -> Token[str | None]:
    """Bind active span ID to execution context."""
    return _current_span_id.set(span_id)


def reset_current_span_id(token: Token[str | None]) -> None:
    """Restore previous span ID context token."""
    _current_span_id.reset(token)


def extract_w3c_traceparent(header_value: str | None) -> tuple[str | None, str | None]:
    """Extract (trace_id, parent_span_id) from standard W3C traceparent header.

    Format: {version}-{trace_id}-{parent_id}-{trace_flags}
    Example: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
    """
    if not header_value:
        return None, None

    match = W3C_TRACEPARENT_PATTERN.match(header_value.strip())
    if not match:
        return None, None

    _, trace_id, parent_id, _ = match.groups()
    return trace_id, parent_id


def format_w3c_traceparent(trace_id: str, span_id: str, sampled: bool = True) -> str:
    """Format trace_id and span_id into W3C traceparent header string."""
    flags = "01" if sampled else "00"
    return f"00-{trace_id}-{span_id}-{flags}"
