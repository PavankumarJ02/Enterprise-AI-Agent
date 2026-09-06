"""High-performance Tracer engine supporting async/sync context managers and OTel conventions."""

import functools
import inspect
import time
import traceback
from collections.abc import AsyncIterator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, TypeVar

from enterprise_agent.core.logging import get_logger
from enterprise_agent.observability.context import (
    generate_span_id,
    generate_trace_id,
    get_current_span_id,
    get_current_trace_id,
    reset_current_span_id,
    reset_current_trace_id,
    set_current_span_id,
    set_current_trace_id,
)
from enterprise_agent.observability.exporters import InMemorySpanExporter, SpanExporter
from enterprise_agent.schemas.observability import (
    SpanEvent,
    SpanKind,
    SpanRecord,
    SpanStatusCode,
)

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class SpanContext:
    """Active mutable handle for enriching an ongoing span with attributes and events."""

    def __init__(
        self,
        span_id: str,
        trace_id: str,
        name: str,
        kind: SpanKind,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.span_id = span_id
        self.trace_id = trace_id
        self.name = name
        self.kind = kind
        self.parent_span_id = parent_span_id
        self.attributes: dict[str, Any] = attributes or {}
        self.events: list[SpanEvent] = []
        self.status_code = SpanStatusCode.OK
        self.error_message: str | None = None
        self.start_time_ns = time.time_ns()
        self.end_time_ns = 0
        self.duration_ms = 0.0

    def set_attribute(self, key: str, value: Any) -> "SpanContext":
        """Attach a metadata attribute to this span."""
        self.attributes[key] = value
        return self

    def set_attributes(self, attrs: dict[str, Any]) -> "SpanContext":
        """Attach multiple metadata attributes to this span."""
        self.attributes.update(attrs)
        return self

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> "SpanContext":
        """Record a timestamped event on this span."""
        self.events.append(
            SpanEvent(
                name=name,
                timestamp_ns=time.time_ns(),
                attributes=attributes or {},
            )
        )
        return self

    def set_genai_metrics(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        system: str = "gemini",
        temperature: float | None = None,
    ) -> "SpanContext":
        """Record standard OpenTelemetry GenAI semantic conventions."""
        self.attributes["gen_ai.system"] = system
        self.attributes["gen_ai.request.model"] = model
        self.attributes["gen_ai.usage.prompt_tokens"] = prompt_tokens
        self.attributes["gen_ai.usage.completion_tokens"] = completion_tokens
        self.attributes["gen_ai.usage.total_tokens"] = prompt_tokens + completion_tokens
        if temperature is not None:
            self.attributes["gen_ai.request.temperature"] = temperature
        return self

    def set_status(
        self,
        status_code: SpanStatusCode,
        error_message: str | None = None,
    ) -> "SpanContext":
        """Update span status code and error description."""
        self.status_code = status_code
        self.error_message = error_message
        return self

    def close(self) -> SpanRecord:
        """Mark span completion, compute duration, and generate immutable SpanRecord."""
        if self.end_time_ns == 0:
            self.end_time_ns = time.time_ns()
        self.duration_ms = round((self.end_time_ns - self.start_time_ns) / 1_000_000.0, 3)

        return SpanRecord(
            span_id=self.span_id,
            trace_id=self.trace_id,
            parent_span_id=self.parent_span_id,
            name=self.name,
            kind=self.kind,
            attributes=self.attributes,
            events=self.events,
            start_time_ns=self.start_time_ns,
            end_time_ns=self.end_time_ns,
            duration_ms=self.duration_ms,
            status_code=self.status_code,
            error_message=self.error_message,
        )


class Tracer:
    """Core distributed tracing engine generating, contextualizing, and shipping spans."""

    def __init__(
        self,
        exporter: SpanExporter | None = None,
        enabled: bool = True,
        service_name: str = "enterprise-ai-agent",
    ) -> None:
        self.exporter = exporter or InMemorySpanExporter()
        self.enabled = enabled
        self.service_name = service_name

    @asynccontextmanager
    async def async_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> AsyncIterator[SpanContext]:
        """Asynchronous context manager creating and recording a trace span."""
        if not self.enabled:
            # Yield dummy context if disabled
            yield SpanContext("disabled", "disabled", name, kind)
            return

        resolved_trace_id = trace_id or get_current_trace_id() or generate_trace_id()
        resolved_parent_span_id = parent_span_id or get_current_span_id()
        current_span_id = generate_span_id()

        trace_token = set_current_trace_id(resolved_trace_id)
        span_token = set_current_span_id(current_span_id)

        ctx = SpanContext(
            span_id=current_span_id,
            trace_id=resolved_trace_id,
            name=name,
            kind=kind,
            parent_span_id=resolved_parent_span_id,
            attributes=attributes,
        )
        ctx.set_attribute("service.name", self.service_name)

        try:
            yield ctx
        except Exception as exc:
            ctx.set_status(SpanStatusCode.ERROR, str(exc))
            ctx.set_attribute("exception.type", type(exc).__name__)
            ctx.set_attribute("exception.message", str(exc))
            ctx.set_attribute("exception.stacktrace", traceback.format_exc())
            raise
        finally:
            record = ctx.close()
            reset_current_span_id(span_token)
            reset_current_trace_id(trace_token)
            try:
                self.exporter.export([record])
            except Exception as exp_err:
                logger.warning("Tracer failed to export span '%s': %s", name, exp_err)

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> Generator[SpanContext, None, None]:
        """Synchronous context manager creating and recording a trace span."""
        if not self.enabled:
            yield SpanContext("disabled", "disabled", name, kind)
            return

        resolved_trace_id = trace_id or get_current_trace_id() or generate_trace_id()
        resolved_parent_span_id = parent_span_id or get_current_span_id()
        current_span_id = generate_span_id()

        trace_token = set_current_trace_id(resolved_trace_id)
        span_token = set_current_span_id(current_span_id)

        ctx = SpanContext(
            span_id=current_span_id,
            trace_id=resolved_trace_id,
            name=name,
            kind=kind,
            parent_span_id=resolved_parent_span_id,
            attributes=attributes,
        )
        ctx.set_attribute("service.name", self.service_name)

        try:
            yield ctx
        except Exception as exc:
            ctx.set_status(SpanStatusCode.ERROR, str(exc))
            ctx.set_attribute("exception.type", type(exc).__name__)
            ctx.set_attribute("exception.message", str(exc))
            ctx.set_attribute("exception.stacktrace", traceback.format_exc())
            raise
        finally:
            record = ctx.close()
            reset_current_span_id(span_token)
            reset_current_trace_id(trace_token)
            try:
                self.exporter.export([record])
            except Exception as exp_err:
                logger.warning("Tracer failed to export span '%s': %s", name, exp_err)

    def trace(
        self,
        name: str | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ) -> Callable[[F], F]:
        """Function/method decorator wrapping execution in an OpenTelemetry span."""

        def decorator(func: F) -> F:
            op_name = name or f"{func.__module__}.{func.__qualname__}"

            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    async with self.async_span(op_name, kind=kind, attributes=attributes):
                        return await func(*args, **kwargs)

                return async_wrapper  # type: ignore[return-value]
            else:

                @functools.wraps(func)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    with self.span(op_name, kind=kind, attributes=attributes):
                        return func(*args, **kwargs)

                return sync_wrapper  # type: ignore[return-value]

        return decorator
