"""REST API endpoints for OpenTelemetry / Langfuse distributed tracing and observability."""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from enterprise_agent.api.deps import get_observability_service
from enterprise_agent.core.logging import get_logger
from enterprise_agent.observability.service import ObservabilityService
from enterprise_agent.schemas.observability import (
    ObservabilityStats,
    SpanStatusCode,
    TraceQueryFilter,
    TraceRecord,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/observability", tags=["Observability & Distributed Tracing"])


@router.get(
    "/traces",
    response_model=list[TraceRecord],
    status_code=status.HTTP_200_OK,
    summary="List recorded execution traces",
    description="Retrieve a paginated list of completed distributed traces with filtering.",
)
async def list_traces(
    limit: int = Query(default=50, ge=1, le=500, description="Max traces to return"),
    status_code: SpanStatusCode | None = Query(
        default=None, description="Filter by status (ok, error)"
    ),
    root_name: str | None = Query(default=None, description="Filter by root span operation name"),
    min_duration_ms: float | None = Query(
        default=None, ge=0.0, description="Minimum duration in milliseconds"
    ),
    service: ObservabilityService = Depends(get_observability_service),
) -> list[TraceRecord]:
    """Fetch recorded execution traces."""
    filters = TraceQueryFilter(
        limit=limit,
        status_code=status_code,
        root_name=root_name,
        min_duration_ms=min_duration_ms,
    )
    return service.list_traces(filters)


@router.get(
    "/traces/{trace_id}",
    response_model=TraceRecord,
    status_code=status.HTTP_200_OK,
    summary="Get trace by ID with full span tree",
    description="Retrieve hierarchical span execution tree and attributes for a specific trace ID.",
)
async def get_trace(
    trace_id: str,
    service: ObservabilityService = Depends(get_observability_service),
) -> TraceRecord:
    """Fetch detailed trace record by unique trace ID."""
    trace = service.get_trace(trace_id)
    if not trace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace with ID '{trace_id}' not found in active telemetry buffer.",
        )
    return trace


@router.delete(
    "/traces",
    status_code=status.HTTP_200_OK,
    summary="Clear recorded telemetry traces",
    description="Flush and clear in-memory trace records from the active telemetry buffer.",
)
async def clear_traces(
    service: ObservabilityService = Depends(get_observability_service),
) -> dict[str, str]:
    """Clear all in-memory traces."""
    service.clear_traces()
    return {"status": "cleared", "message": "In-memory telemetry buffer reset successfully."}


@router.get(
    "/stats",
    response_model=ObservabilityStats,
    status_code=status.HTTP_200_OK,
    summary="Retrieve platform telemetry statistics",
    description="Get metrics on total traces, spans, token consumption, latency, and error rates.",
)
async def get_telemetry_stats(
    service: ObservabilityService = Depends(get_observability_service),
) -> ObservabilityStats:
    """Get aggregated observability statistics."""
    return service.get_stats()
