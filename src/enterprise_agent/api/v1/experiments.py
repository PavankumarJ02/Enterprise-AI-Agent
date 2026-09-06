"""REST API endpoints for experiment tracking, run comparisons, and grid sweeps."""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from enterprise_agent.api.deps import get_experiment_service
from enterprise_agent.experiments.service import ExperimentTrackingService
from enterprise_agent.schemas.experiments import (
    ExperimentCompareRequest,
    ExperimentRun,
    ExperimentRunCreate,
    ExperimentRunUpdate,
    GridSweepRequest,
    GridSweepResponse,
    RunComparison,
)

router = APIRouter(prefix="/experiments", tags=["Experiment Tracking & Benchmarking"])


@router.post(
    "/runs",
    response_model=ExperimentRun,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new experiment run",
    description="Initializes and logs a new experiment run with hyperparameters and metadata.",
)
async def create_experiment_run(
    request: ExperimentRunCreate,
    service: ExperimentTrackingService = Depends(get_experiment_service),
) -> ExperimentRun:
    """Create a new experiment run record."""
    return service.create_run(request)


@router.get(
    "/runs",
    response_model=list[ExperimentRun],
    status_code=status.HTTP_200_OK,
    summary="List experiment runs",
    description="Retrieve past experiment runs, optionally filtered by experiment name.",
)
async def list_experiment_runs(
    experiment_name: str | None = Query(
        default=None, description="Filter runs by parent experiment name."
    ),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum number of runs to return."),
    service: ExperimentTrackingService = Depends(get_experiment_service),
) -> list[ExperimentRun]:
    """List historical experiment runs."""
    return service.list_runs(experiment_name=experiment_name, limit=limit)


@router.get(
    "/runs/{run_id}",
    response_model=ExperimentRun,
    status_code=status.HTTP_200_OK,
    summary="Get experiment run details",
    description="Retrieve details and metrics for a specific experiment run ID.",
)
async def get_experiment_run(
    run_id: str,
    service: ExperimentTrackingService = Depends(get_experiment_service),
) -> ExperimentRun:
    """Fetch run details by ID."""
    run = service.get_run(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment run '{run_id}' not found.",
        )
    return run


@router.put(
    "/runs/{run_id}",
    response_model=ExperimentRun,
    status_code=status.HTTP_200_OK,
    summary="Update metrics for an experiment run",
    description="Log computed metrics, latency, and status on an existing run.",
)
async def update_experiment_run(
    run_id: str,
    update: ExperimentRunUpdate,
    service: ExperimentTrackingService = Depends(get_experiment_service),
) -> ExperimentRun:
    """Update metrics and completion status for a run."""
    try:
        return service.log_run_metrics(
            run_id=run_id,
            metrics=update.metrics,
            total_latency_ms=update.total_latency_ms,
            status=update.status,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.post(
    "/compare",
    response_model=RunComparison,
    status_code=status.HTTP_200_OK,
    summary="Compare experiment runs side-by-side",
    description=(
        "Compute metric deltas, percentage improvements, and leaders across candidate runs."
    ),
)
async def compare_experiment_runs(
    request: ExperimentCompareRequest,
    service: ExperimentTrackingService = Depends(get_experiment_service),
) -> RunComparison:
    """Perform comparative analysis between baseline and candidate runs."""
    try:
        return service.compare_runs(
            run_ids=request.run_ids,
            baseline_run_id=request.baseline_run_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/grid",
    response_model=GridSweepResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute hyperparameter grid sweep",
    description=(
        "Run automated benchmark evaluations over retrieval strategies and top_k combinations."
    ),
)
async def run_grid_sweep(
    request: GridSweepRequest,
    service: ExperimentTrackingService = Depends(get_experiment_service),
) -> GridSweepResponse:
    """Trigger grid search sweep across parameter space."""
    return await service.run_grid_sweep(request)
