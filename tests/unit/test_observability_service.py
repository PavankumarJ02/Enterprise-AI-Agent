"""Unit tests for ObservabilityService and W3C context utilities."""

from enterprise_agent.config.settings import Settings
from enterprise_agent.observability.context import (
    extract_w3c_traceparent,
    format_w3c_traceparent,
    generate_span_id,
    generate_trace_id,
)
from enterprise_agent.observability.exporters import InMemorySpanExporter
from enterprise_agent.observability.service import ObservabilityService
from enterprise_agent.observability.tracer import Tracer
from enterprise_agent.schemas.observability import SpanKind, TraceQueryFilter


def test_w3c_context_propagation() -> None:
    """Verify standard W3C traceparent formatting and parsing."""
    trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    span_id = "00f067aa0ba902b7"

    header = format_w3c_traceparent(trace_id, span_id, sampled=True)
    assert header == f"00-{trace_id}-{span_id}-01"

    parsed_trace, parsed_parent = extract_w3c_traceparent(header)
    assert parsed_trace == trace_id
    assert parsed_parent == span_id

    # Invalid header handling
    assert extract_w3c_traceparent(None) == (None, None)
    assert extract_w3c_traceparent("malformed-header") == (None, None)


def test_id_generation_formats() -> None:
    """Verify trace and span identifier lengths and hex validity."""
    tid = generate_trace_id()
    assert len(tid) == 32
    int(tid, 16)  # Validates hex

    sid = generate_span_id()
    assert len(sid) == 16
    int(sid, 16)  # Validates hex


def test_observability_service_coordinator() -> None:
    """Verify ObservabilityService coordinator methods."""
    settings = Settings(observability_exporter="memory", observability_max_in_memory_traces=50)
    exporter = InMemorySpanExporter(max_traces=50)
    tracer = Tracer(exporter=exporter, service_name="test-app")
    service = ObservabilityService(settings=settings, exporter=exporter, tracer=tracer)

    # Initially empty
    assert len(service.list_traces()) == 0
    stats = service.get_stats()
    assert stats.total_traces == 0

    # Record trace
    with tracer.span("service.test", kind=SpanKind.SERVER) as span:
        span.set_attribute("env", "testing")

    traces = service.list_traces(TraceQueryFilter(limit=10))
    assert len(traces) == 1
    trace_id = traces[0].trace_id

    fetched = service.get_trace(trace_id)
    assert fetched is not None
    assert fetched.trace_id == trace_id
    assert fetched.root_span_name == "service.test"

    # Verify stats
    stats = service.get_stats()
    assert stats.total_traces == 1
    assert stats.total_spans == 1

    # Clear
    service.clear_traces()
    assert len(service.list_traces()) == 0
