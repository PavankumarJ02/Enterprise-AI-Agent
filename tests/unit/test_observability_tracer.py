"""Unit tests for Tracer engine and context manager span life cycle."""

import pytest

from enterprise_agent.observability.context import get_current_span_id, get_current_trace_id
from enterprise_agent.observability.exporters import InMemorySpanExporter
from enterprise_agent.observability.tracer import Tracer
from enterprise_agent.schemas.observability import SpanKind, SpanStatusCode


def test_tracer_span_lifecycle() -> None:
    """Verify synchronous span creation, timing, attributes, and export."""
    exporter = InMemorySpanExporter()
    tracer = Tracer(exporter=exporter, service_name="test-service")

    with tracer.span(
        name="test.operation",
        kind=SpanKind.INTERNAL,
        attributes={"user.id": "usr-123"},
    ) as span:
        span.set_attribute("custom.tag", "alpha")
        span.add_event("checkpoint_1", {"items": 5})

    traces = exporter.list_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.root_span_name == "test.operation"
    assert trace.span_count == 1
    assert trace.status_code == SpanStatusCode.OK
    assert trace.total_duration_ms >= 0.0

    recorded_span = trace.spans[0]
    assert recorded_span.name == "test.operation"
    assert recorded_span.kind == SpanKind.INTERNAL
    assert recorded_span.attributes["user.id"] == "usr-123"
    assert recorded_span.attributes["custom.tag"] == "alpha"
    assert len(recorded_span.events) == 1
    assert recorded_span.events[0].name == "checkpoint_1"


@pytest.mark.asyncio
async def test_tracer_async_span_nesting() -> None:
    """Verify hierarchical parent-child span relationships in async execution."""
    exporter = InMemorySpanExporter()
    tracer = Tracer(exporter=exporter)

    async with tracer.async_span("parent.operation", kind=SpanKind.SERVER) as parent:
        parent.set_attribute("level", "root")

        # Child 1
        async with tracer.async_span("child.retrieval", kind=SpanKind.RETRIEVER) as child1:
            child1.set_attribute("results", 10)

        # Child 2
        async with tracer.async_span("child.llm", kind=SpanKind.LLM) as child2:
            child2.set_genai_metrics(
                model="gemini-2.5-flash",
                prompt_tokens=45,
                completion_tokens=20,
                temperature=0.2,
            )

    traces = exporter.list_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.span_count == 3
    assert trace.root_span_name == "parent.operation"
    assert trace.total_prompt_tokens == 45
    assert trace.total_completion_tokens == 20
    assert trace.total_tokens == 65

    # Check parent-child links
    root_span = [s for s in trace.spans if s.parent_span_id is None][0]
    child_spans = [s for s in trace.spans if s.parent_span_id == root_span.span_id]
    assert len(child_spans) == 2
    child_names = {s.name for s in child_spans}
    assert child_names == {"child.retrieval", "child.llm"}


def test_tracer_error_capture() -> None:
    """Verify exceptions inside span mark status as ERROR and record stacktrace."""
    exporter = InMemorySpanExporter()
    tracer = Tracer(exporter=exporter)

    with pytest.raises(ValueError, match="simulated failure"):
        with tracer.span("faulty.operation", kind=SpanKind.INTERNAL):
            raise ValueError("simulated failure")

    traces = exporter.list_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.status_code == SpanStatusCode.ERROR
    span = trace.spans[0]
    assert span.status_code == SpanStatusCode.ERROR
    assert span.error_message == "simulated failure"
    assert span.attributes["exception.type"] == "ValueError"
    assert "simulated failure" in span.attributes["exception.stacktrace"]


def test_tracer_context_cleanup() -> None:
    """Verify context variables are cleanly restored after span exit."""
    tracer = Tracer()
    assert get_current_trace_id() is None
    assert get_current_span_id() is None

    with tracer.span("outer"):
        assert get_current_trace_id() is not None
        assert get_current_span_id() is not None
        outer_trace_id = get_current_trace_id()

        with tracer.span("inner"):
            assert get_current_trace_id() == outer_trace_id
            assert get_current_span_id() is not None

        # After inner, outer is restored
        assert get_current_trace_id() == outer_trace_id

    # After outer, context is reset
    assert get_current_trace_id() is None
    assert get_current_span_id() is None


@pytest.mark.asyncio
async def test_tracer_decorators() -> None:
    """Verify @tracer.trace decorator on sync and async functions."""
    exporter = InMemorySpanExporter()
    tracer = Tracer(exporter=exporter)

    @tracer.trace(name="decorated.sync", kind=SpanKind.TOOL)
    def sync_tool(x: int) -> int:
        return x * 2

    @tracer.trace(name="decorated.async", kind=SpanKind.LLM)
    async def async_llm(prompt: str) -> str:
        return f"Echo: {prompt}"

    assert sync_tool(5) == 10
    assert await async_llm("hello") == "Echo: hello"

    traces = exporter.list_traces()
    assert len(traces) == 2
    names = {t.root_span_name for t in traces}
    assert names == {"decorated.sync", "decorated.async"}


def test_tracer_disabled() -> None:
    """Verify disabled tracer returns dummy context and records no spans."""
    exporter = InMemorySpanExporter()
    tracer = Tracer(exporter=exporter, enabled=False)

    with tracer.span("noop.span") as span:
        span.set_attribute("key", "val")

    assert len(exporter.list_traces()) == 0
