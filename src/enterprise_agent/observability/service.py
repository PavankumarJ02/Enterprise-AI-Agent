"""Observability service coordinating Tracer, Exporters, and API telemetry queries."""

from enterprise_agent.config.settings import Settings
from enterprise_agent.core.logging import get_logger
from enterprise_agent.observability.exporters import (
    InMemorySpanExporter,
    SpanExporter,
    create_span_exporter,
)
from enterprise_agent.observability.tracer import Tracer
from enterprise_agent.schemas.observability import (
    ObservabilityStats,
    TraceQueryFilter,
    TraceRecord,
)

logger = get_logger(__name__)


class ObservabilityService:
    """Enterprise service coordinator for distributed tracing and telemetry inspection."""

    def __init__(
        self,
        settings: Settings | None = None,
        exporter: SpanExporter | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.exporter = exporter or create_span_exporter(self.settings)
        self.tracer = tracer or Tracer(
            exporter=self.exporter,
            enabled=self.settings.observability_enabled,
            service_name=self.settings.app_name,
        )

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        """Fetch a complete trace by ID if supported by the active exporter."""
        if isinstance(self.exporter, InMemorySpanExporter):
            return self.exporter.get_trace(trace_id)
        return None

    def list_traces(self, filter_params: TraceQueryFilter | None = None) -> list[TraceRecord]:
        """List recorded traces matching filter criteria from in-memory buffer."""
        if isinstance(self.exporter, InMemorySpanExporter):
            return self.exporter.list_traces(filter_params)
        return []

    def get_stats(self) -> ObservabilityStats:
        """Calculate aggregate telemetry metrics across the platform."""
        if isinstance(self.exporter, InMemorySpanExporter):
            return self.exporter.get_stats()

        return ObservabilityStats(
            total_traces=0,
            total_spans=0,
            total_tokens=0,
            average_duration_ms=0.0,
            error_count=0,
            error_rate=0.0,
            active_exporter=self.settings.observability_exporter,
        )

    def clear_traces(self) -> None:
        """Flush and clear buffered in-memory traces."""
        if isinstance(self.exporter, InMemorySpanExporter):
            self.exporter.clear()
