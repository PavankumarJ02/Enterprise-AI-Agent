"""Span exporters dispatching telemetry records to in-memory buffers or OTLP collectors."""

import abc
import threading
from collections import OrderedDict
from typing import Any

import httpx

from enterprise_agent.config.settings import Settings
from enterprise_agent.core.logging import get_logger
from enterprise_agent.schemas.observability import (
    ObservabilityStats,
    SpanRecord,
    SpanStatusCode,
    TraceQueryFilter,
    TraceRecord,
)

logger = get_logger(__name__)


class SpanExporter(abc.ABC):
    """Abstract interface for exporting completed telemetry spans."""

    @abc.abstractmethod
    def export(self, spans: list[SpanRecord]) -> None:
        """Export a batch of completed span records."""

    def flush(self) -> None:  # noqa: B027
        """Flush pending spans if buffered (optional hook)."""

    def shutdown(self) -> None:  # noqa: B027
        """Release exporter resources upon application shutdown (optional hook)."""


class InMemorySpanExporter(SpanExporter):
    """Thread-safe in-memory span and trace buffer for testing and API telemetry inspection."""

    def __init__(self, max_traces: int = 100) -> None:
        self.max_traces = max(10, max_traces)
        self._lock = threading.Lock()
        # trace_id -> list of SpanRecord
        self._traces: OrderedDict[str, list[SpanRecord]] = OrderedDict()

    def export(self, spans: list[SpanRecord]) -> None:
        """Record newly completed spans, organizing by trace_id."""
        with self._lock:
            for span in spans:
                if span.trace_id not in self._traces:
                    # Enforce FIFO bound when capacity is exceeded
                    if len(self._traces) >= self.max_traces:
                        self._traces.popitem(last=False)
                    self._traces[span.trace_id] = []
                self._traces[span.trace_id].append(span)

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        """Retrieve aggregated trace record by trace ID."""
        with self._lock:
            spans = self._traces.get(trace_id)
            if not spans:
                return None
            return self._aggregate_trace(trace_id, list(spans))

    def list_traces(self, filter_params: TraceQueryFilter | None = None) -> list[TraceRecord]:
        """List and optionally filter recorded traces, newest first."""
        filters = filter_params or TraceQueryFilter()
        with self._lock:
            trace_items = list(self._traces.items())

        # Process in reverse order (newest first)
        results: list[TraceRecord] = []
        for trace_id, spans in reversed(trace_items):
            trace = self._aggregate_trace(trace_id, spans)

            # Apply filters
            if filters.status_code and trace.status_code != filters.status_code:
                continue
            if filters.root_name and filters.root_name.lower() not in trace.root_span_name.lower():
                continue
            if (
                filters.min_duration_ms is not None
                and trace.total_duration_ms < filters.min_duration_ms
            ):
                continue

            results.append(trace)
            if len(results) >= filters.limit:
                break

        return results

    def get_stats(self) -> ObservabilityStats:
        """Calculate aggregated observability statistics across all buffered traces."""
        with self._lock:
            traces_copy = {tid: list(spans) for tid, spans in self._traces.items()}

        total_traces = len(traces_copy)
        total_spans = sum(len(spans) for spans in traces_copy.values())
        total_tokens = 0
        total_duration = 0.0
        error_count = 0

        for tid, spans in traces_copy.items():
            trace = self._aggregate_trace(tid, spans)
            total_tokens += trace.total_tokens
            total_duration += trace.total_duration_ms
            if trace.status_code == SpanStatusCode.ERROR:
                error_count += 1

        avg_duration = round(total_duration / total_traces, 2) if total_traces > 0 else 0.0
        error_rate = round(error_count / total_traces, 4) if total_traces > 0 else 0.0

        return ObservabilityStats(
            total_traces=total_traces,
            total_spans=total_spans,
            total_tokens=total_tokens,
            average_duration_ms=avg_duration,
            error_count=error_count,
            error_rate=error_rate,
            active_exporter="memory",
        )

    def clear(self) -> None:
        """Flush and reset all buffered traces."""
        with self._lock:
            self._traces.clear()

    @staticmethod
    def _aggregate_trace(trace_id: str, spans: list[SpanRecord]) -> TraceRecord:
        """Aggregate individual spans into a comprehensive TraceRecord."""
        if not spans:
            return TraceRecord(
                trace_id=trace_id,
                root_span_name="empty",
                spans=[],
                span_count=0,
                total_duration_ms=0.0,
                start_time_ns=0,
                end_time_ns=0,
            )

        # Identify root span (span without parent_span_id, or earliest span)
        root_candidates = [s for s in spans if s.parent_span_id is None]
        root_span = (
            root_candidates[0] if root_candidates else min(spans, key=lambda s: s.start_time_ns)
        )

        start_time_ns = min(s.start_time_ns for s in spans)
        end_time_ns = max(s.end_time_ns for s in spans)
        total_duration_ms = round((end_time_ns - start_time_ns) / 1_000_000.0, 3)

        # Determine overall trace status
        has_error = any(s.status_code == SpanStatusCode.ERROR for s in spans)
        status_code = SpanStatusCode.ERROR if has_error else SpanStatusCode.OK

        prompt_tokens = 0
        completion_tokens = 0

        for s in spans:
            prompt_tokens += int(s.attributes.get("gen_ai.usage.prompt_tokens", 0))
            completion_tokens += int(s.attributes.get("gen_ai.usage.completion_tokens", 0))

        total_tokens = prompt_tokens + completion_tokens

        return TraceRecord(
            trace_id=trace_id,
            root_span_name=root_span.name,
            spans=spans,
            span_count=len(spans),
            total_duration_ms=total_duration_ms,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            status_code=status_code,
            total_prompt_tokens=prompt_tokens,
            total_completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            metadata=root_span.attributes.copy(),
        )


class ConsoleSpanExporter(SpanExporter):
    """Prints completed spans and GenAI metrics cleanly to logger/console."""

    def export(self, spans: list[SpanRecord]) -> None:
        for span in spans:
            prefix = "└──" if span.parent_span_id else "┌──"
            status_tag = (
                "[OK]" if span.status_code == SpanStatusCode.OK else f"[FAIL: {span.error_message}]"
            )
            logger.info(
                "%s [SPAN] %s (%s, %.2fms) %s | trace=%s parent=%s",
                prefix,
                span.name,
                span.kind.value,
                span.duration_ms,
                status_tag,
                span.trace_id[:8],
                span.parent_span_id[:8] if span.parent_span_id else "root",
            )
            if span.attributes:
                logger.debug("   attrs: %s", span.attributes)


class OTLPSpanExporter(SpanExporter):
    """Exports spans formatted as OpenTelemetry JSON payload over HTTP."""

    def __init__(self, endpoint: str | None = None, headers: dict[str, str] | None = None) -> None:
        self.endpoint = endpoint or "http://localhost:4318/v1/traces"
        self.headers = headers or {"Content-Type": "application/json"}
        self._client = httpx.Client(timeout=3.0)

    def export(self, spans: list[SpanRecord]) -> None:
        payload = self._format_otlp_payload(spans)
        try:
            resp = self._client.post(self.endpoint, json=payload, headers=self.headers)
            if resp.status_code >= 400:
                logger.warning(
                    "OTLP collector responded with status %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
        except Exception as exc:
            logger.warning("Failed to export %d spans to OTLP collector: %s", len(spans), exc)

    def shutdown(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    @staticmethod
    def _format_otlp_payload(spans: list[SpanRecord]) -> dict[str, Any]:
        """Convert SpanRecords into OTLP HTTP JSON payload format."""
        resource_spans = []
        scope_spans = []

        for span in spans:
            attributes = [
                {"key": k, "value": {"stringValue": str(v)}} for k, v in span.attributes.items()
            ]
            span_dict: dict[str, Any] = {
                "traceId": span.trace_id,
                "spanId": span.span_id,
                "name": span.name,
                "kind": 1,
                "startTimeUnixNano": str(span.start_time_ns),
                "endTimeUnixNano": str(span.end_time_ns),
                "attributes": attributes,
                "status": {
                    "code": 1 if span.status_code == SpanStatusCode.OK else 2,
                    "message": span.error_message or "",
                },
            }
            if span.parent_span_id:
                span_dict["parentSpanId"] = span.parent_span_id
            scope_spans.append(span_dict)

        resource_spans.append(
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "enterprise-ai-agent"}}
                    ]
                },
                "scopeSpans": [
                    {"scope": {"name": "enterprise_agent.tracer"}, "spans": scope_spans}
                ],
            }
        )
        return {"resourceSpans": resource_spans}


def create_span_exporter(settings: Settings) -> SpanExporter:
    """Factory creating the appropriate SpanExporter implementation."""
    exporter_type = settings.observability_exporter.lower()

    if exporter_type == "console":
        return ConsoleSpanExporter()

    if exporter_type in {"otlp", "langfuse"}:
        endpoint = settings.observability_otlp_endpoint
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if exporter_type == "langfuse":
            pk = settings.observability_langfuse_public_key.get_secret_value()
            sk = settings.observability_langfuse_secret_key.get_secret_value()
            if pk and sk:
                import base64

                auth_val = base64.b64encode(f"{pk}:{sk}".encode()).decode()
                headers["Authorization"] = f"Basic {auth_val}"
            endpoint = (
                endpoint or f"{settings.observability_langfuse_host.rstrip('/')}/api/public/traces"
            )

        return OTLPSpanExporter(endpoint=endpoint, headers=headers)

    # Default in-memory exporter
    return InMemorySpanExporter(max_traces=settings.observability_max_in_memory_traces)
