"""Unit tests for InMemorySpanExporter and OTLPSpanExporter."""

import time

from enterprise_agent.observability.exporters import (
    ConsoleSpanExporter,
    InMemorySpanExporter,
    OTLPSpanExporter,
)
from enterprise_agent.schemas.observability import (
    SpanKind,
    SpanRecord,
    SpanStatusCode,
    TraceQueryFilter,
)


def _create_mock_span(
    trace_id: str,
    span_id: str,
    name: str,
    duration_ms: float = 10.0,
    parent_span_id: str | None = None,
    status_code: SpanStatusCode = SpanStatusCode.OK,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> SpanRecord:
    """Helper to generate a mock SpanRecord for testing."""
    now_ns = time.time_ns()
    attrs = {}
    if prompt_tokens > 0:
        attrs["gen_ai.usage.prompt_tokens"] = prompt_tokens
    if completion_tokens > 0:
        attrs["gen_ai.usage.completion_tokens"] = completion_tokens

    return SpanRecord(
        span_id=span_id,
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        name=name,
        kind=SpanKind.INTERNAL,
        attributes=attrs,
        events=[],
        start_time_ns=now_ns,
        end_time_ns=now_ns + int(duration_ms * 1_000_000),
        duration_ms=duration_ms,
        status_code=status_code,
        error_message="mock error" if status_code == SpanStatusCode.ERROR else None,
    )


def test_in_memory_exporter_crud() -> None:
    """Verify storing, retrieving, and clearing traces in-memory."""
    exporter = InMemorySpanExporter(max_traces=50)

    span1 = _create_mock_span("trace-1", "span-1", "root.op", duration_ms=25.0)
    span2 = _create_mock_span(
        "trace-1", "span-2", "child.op", duration_ms=10.0, parent_span_id="span-1"
    )
    exporter.export([span1, span2])

    trace = exporter.get_trace("trace-1")
    assert trace is not None
    assert trace.trace_id == "trace-1"
    assert trace.root_span_name == "root.op"
    assert trace.span_count == 2
    assert trace.status_code == SpanStatusCode.OK

    # Non-existent trace
    assert exporter.get_trace("trace-unknown") is None

    # Clear
    exporter.clear()
    assert exporter.get_trace("trace-1") is None
    assert len(exporter.list_traces()) == 0


def test_in_memory_exporter_fifo_ejection() -> None:
    """Verify oldest traces are ejected when capacity bound is reached."""
    exporter = InMemorySpanExporter(max_traces=10)

    for i in range(15):
        span = _create_mock_span(f"trace-{i}", f"span-{i}", f"op_{i}")
        exporter.export([span])

    traces = exporter.list_traces(TraceQueryFilter(limit=50))
    assert len(traces) == 10

    # Traces 0 to 4 should be evicted
    trace_ids = {t.trace_id for t in traces}
    assert "trace-0" not in trace_ids
    assert "trace-4" not in trace_ids
    assert "trace-14" in trace_ids
    assert "trace-5" in trace_ids


def test_in_memory_exporter_filters() -> None:
    """Verify trace querying with filters for status, root name, and duration."""
    exporter = InMemorySpanExporter(max_traces=50)

    exporter.export(
        [
            _create_mock_span(
                "t1", "s1", "rag.query", duration_ms=100.0, status_code=SpanStatusCode.OK
            )
        ]
    )
    exporter.export(
        [
            _create_mock_span(
                "t2", "s2", "agent.chat", duration_ms=250.0, status_code=SpanStatusCode.ERROR
            )
        ]
    )
    exporter.export(
        [
            _create_mock_span(
                "t3", "s3", "sql.query", duration_ms=40.0, status_code=SpanStatusCode.OK
            )
        ]
    )

    # Filter by status ERROR
    err_traces = exporter.list_traces(TraceQueryFilter(status_code=SpanStatusCode.ERROR))
    assert len(err_traces) == 1
    assert err_traces[0].trace_id == "t2"

    # Filter by root name
    rag_traces = exporter.list_traces(TraceQueryFilter(root_name="rag"))
    assert len(rag_traces) == 1
    assert rag_traces[0].trace_id == "t1"

    # Filter by minimum duration
    slow_traces = exporter.list_traces(TraceQueryFilter(min_duration_ms=80.0))
    assert len(slow_traces) == 2
    assert {t.trace_id for t in slow_traces} == {"t1", "t2"}


def test_in_memory_exporter_stats() -> None:
    """Verify aggregate statistics calculation."""
    exporter = InMemorySpanExporter(max_traces=50)

    exporter.export(
        [
            _create_mock_span(
                "t1", "s1", "op1", duration_ms=100.0, prompt_tokens=10, completion_tokens=5
            )
        ]
    )
    exporter.export(
        [
            _create_mock_span(
                "t2",
                "s2",
                "op2",
                duration_ms=200.0,
                status_code=SpanStatusCode.ERROR,
                prompt_tokens=20,
                completion_tokens=15,
            )
        ]
    )

    stats = exporter.get_stats()
    assert stats.total_traces == 2
    assert stats.total_spans == 2
    assert stats.total_tokens == 50
    assert stats.error_count == 1
    assert stats.error_rate == 0.5
    assert stats.average_duration_ms == 150.0
    assert stats.active_exporter == "memory"


def test_console_exporter() -> None:
    """Verify console exporter processes spans without error."""
    exporter = ConsoleSpanExporter()
    span = _create_mock_span("t1", "s1", "console.test")
    exporter.export([span])
    exporter.flush()
    exporter.shutdown()


def test_otlp_payload_formatting() -> None:
    """Verify OTLP JSON serialization structure."""
    span = _create_mock_span("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7", "test.span")
    span.attributes["test.key"] = "test.val"

    payload = OTLPSpanExporter._format_otlp_payload([span])
    assert "resourceSpans" in payload
    resource = payload["resourceSpans"][0]
    scope_spans = resource["scopeSpans"][0]["spans"]
    assert len(scope_spans) == 1
    assert scope_spans[0]["name"] == "test.span"
    assert scope_spans[0]["traceId"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert scope_spans[0]["spanId"] == "00f067aa0ba902b7"
