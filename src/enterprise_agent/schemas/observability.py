"""Pydantic models and schemas for OpenTelemetry / Langfuse distributed tracing."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SpanKind(StrEnum):
    """Classification of the span's operational role."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    LLM = "llm"
    RETRIEVER = "retriever"
    TOOL = "tool"
    GUARDRAIL = "guardrail"


class SpanStatusCode(StrEnum):
    """OpenTelemetry-standard span status code."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class SpanEvent(BaseModel):
    """Timestamped event emitted during a span's lifecycle."""

    name: str = Field(..., description="Name or key of the event.")
    timestamp_ns: int = Field(..., description="Nanosecond Unix timestamp.")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured key-value pairs associated with the event.",
    )


class SpanRecord(BaseModel):
    """Representation of an individual completed tracing span."""

    span_id: str = Field(..., description="Unique 16-hex-character span identifier.")
    trace_id: str = Field(..., description="Unique 32-hex-character trace identifier.")
    parent_span_id: str | None = Field(
        default=None,
        description="Parent span ID if this span is nested.",
    )
    name: str = Field(..., description="Logical name of the operation (e.g. 'gemini.generate').")
    kind: SpanKind = Field(
        default=SpanKind.INTERNAL,
        description="Operational category of the span.",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="OpenTelemetry / GenAI semantic conventions attributes.",
    )
    events: list[SpanEvent] = Field(
        default_factory=list,
        description="List of events recorded within this span.",
    )
    start_time_ns: int = Field(..., description="Span start timestamp in nanoseconds.")
    end_time_ns: int = Field(..., description="Span end timestamp in nanoseconds.")
    duration_ms: float = Field(..., description="Total wall-clock duration in milliseconds.")
    status_code: SpanStatusCode = Field(
        default=SpanStatusCode.OK,
        description="Status outcome of the span operation.",
    )
    error_message: str | None = Field(
        default=None,
        description="Error details if the span failed.",
    )


class TraceRecord(BaseModel):
    """Hierarchical aggregation of all spans belonging to a single trace."""

    trace_id: str = Field(..., description="Unique 32-hex-character trace identifier.")
    root_span_name: str = Field(..., description="Name of the entry root span.")
    spans: list[SpanRecord] = Field(
        default_factory=list,
        description="List of all spans recorded for this trace.",
    )
    span_count: int = Field(..., description="Total number of spans in the trace.")
    total_duration_ms: float = Field(..., description="Total wall-clock duration in milliseconds.")
    start_time_ns: int = Field(..., description="Earliest span start timestamp in nanoseconds.")
    end_time_ns: int = Field(..., description="Latest span end timestamp in nanoseconds.")
    status_code: SpanStatusCode = Field(
        default=SpanStatusCode.OK,
        description="Aggregated status outcome across all child spans.",
    )
    total_prompt_tokens: int = Field(
        default=0,
        description="Sum of prompt tokens consumed across all LLM spans.",
    )
    total_completion_tokens: int = Field(
        default=0,
        description="Sum of completion tokens consumed across all LLM spans.",
    )
    total_tokens: int = Field(
        default=0,
        description="Sum of total tokens consumed across all LLM spans.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="High-level user or session metadata associated with the trace.",
    )


class TraceQueryFilter(BaseModel):
    """Filter parameters for listing and querying traces."""

    limit: int = Field(default=50, ge=1, le=500, description="Maximum traces to return.")
    status_code: SpanStatusCode | None = Field(
        default=None,
        description="Filter by trace status code (ok, error).",
    )
    root_name: str | None = Field(
        default=None,
        description="Filter by root span operation name.",
    )
    min_duration_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Filter traces exceeding a minimum duration.",
    )


class ObservabilityStats(BaseModel):
    """Aggregated observability and telemetry statistics."""

    total_traces: int = Field(..., description="Total traces recorded.")
    total_spans: int = Field(..., description="Total individual spans across all traces.")
    total_tokens: int = Field(..., description="Total LLM tokens consumed.")
    average_duration_ms: float = Field(..., description="Average trace latency in milliseconds.")
    error_count: int = Field(..., description="Total traces with error status.")
    error_rate: float = Field(..., description="Error rate percentage (0.0 to 1.0).")
    active_exporter: str = Field(..., description="Name of the active span exporter.")
