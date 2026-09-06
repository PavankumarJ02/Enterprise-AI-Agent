"""Experiment Tracking Service coordinating run persistence, comparisons, and grid sweeps."""

import time
import uuid
from datetime import UTC, datetime
from typing import Any

from enterprise_agent.config.settings import Settings, get_settings
from enterprise_agent.core.logging import get_logger
from enterprise_agent.evaluation.service import RAGEvaluationService
from enterprise_agent.experiments.comparator import RunComparisonEngine
from enterprise_agent.experiments.storage import ExperimentRunRepository
from enterprise_agent.schemas.experiments import (
    ExperimentRun,
    ExperimentRunCreate,
    GridSweepRequest,
    GridSweepResponse,
    RunComparison,
)

logger = get_logger(__name__)


class ExperimentTrackingService:
    """Coordinates lifecycle of experiments, hyperparameter trials, and benchmark leaderboards."""

    def __init__(
        self,
        repository: ExperimentRunRepository | None = None,
        evaluation_service: RAGEvaluationService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repository = repository or ExperimentRunRepository(
            db_path=self.settings.experiments_db_path
        )
        self.evaluation_service = evaluation_service

    def create_run(self, request: ExperimentRunCreate) -> ExperimentRun:
        """Initialize and persist a new experiment trial run."""
        run_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        run = ExperimentRun(
            run_id=run_id,
            experiment_name=request.experiment_name,
            run_name=request.run_name,
            parameters=request.parameters,
            metrics={},
            tags=request.tags,
            status="running",
            total_latency_ms=0.0,
            created_at=created_at,
        )
        return self.repository.create_run(run)

    def log_run_metrics(
        self,
        run_id: str,
        metrics: dict[str, float],
        total_latency_ms: float = 0.0,
        status: str = "completed",
    ) -> ExperimentRun:
        """Update metrics and completion status for a run."""
        updated = self.repository.update_run(
            run_id=run_id,
            metrics=metrics,
            status=status,
            total_latency_ms=total_latency_ms,
        )
        if not updated:
            raise ValueError(f"Run ID '{run_id}' not found.")
        return updated

    def get_run(self, run_id: str) -> ExperimentRun | None:
        """Retrieve run details by ID."""
        return self.repository.get_run(run_id)

    def list_runs(self, experiment_name: str | None = None, limit: int = 50) -> list[ExperimentRun]:
        """List past experiment runs."""
        return self.repository.list_runs(experiment_name=experiment_name, limit=limit)

    def compare_runs(
        self,
        run_ids: list[str],
        baseline_run_id: str | None = None,
    ) -> RunComparison:
        """Compare a set of runs side-by-side."""
        runs: list[ExperimentRun] = []
        for rid in run_ids:
            run = self.repository.get_run(rid)
            if not run:
                raise ValueError(f"Run ID '{rid}' not found for comparison.")
            runs.append(run)

        return RunComparisonEngine.compare_runs(runs, baseline_run_id=baseline_run_id)

    async def run_grid_sweep(self, request: GridSweepRequest) -> GridSweepResponse:
        """Execute automated benchmark evaluations across parameter combinations."""
        executed_runs: list[ExperimentRun] = []
        logger.info(
            "Launching grid sweep for experiment '%s' (%d strategies x %d top_k)",
            request.experiment_name,
            len(request.retrieval_strategies),
            len(request.top_k_values),
        )

        for strategy in request.retrieval_strategies:
            for top_k in request.top_k_values:
                run_name = f"{strategy}_k{top_k}"
                params: dict[str, Any] = {
                    "strategy": strategy,
                    "top_k": top_k,
                }
                run = self.create_run(
                    ExperimentRunCreate(
                        experiment_name=request.experiment_name,
                        run_name=run_name,
                        parameters=params,
                        tags=[strategy, f"k={top_k}"],
                    )
                )

                t0 = time.perf_counter()
                metrics: dict[str, float] = {}

                if self.evaluation_service:
                    eval_report = await self.evaluation_service.evaluate_dataset(
                        use_live_rag=False, top_k=top_k
                    )
                    metrics = {
                        "faithfulness": eval_report.metrics["faithfulness"].mean,
                        "answer_relevance": eval_report.metrics["answer_relevance"].mean,
                        "context_precision": eval_report.metrics["context_precision"].mean,
                        "context_recall": eval_report.metrics["context_recall"].mean,
                    }

                latency = round((time.perf_counter() - t0) * 1000.0, 2)
                completed_run = self.log_run_metrics(
                    run_id=run.run_id,
                    metrics=metrics,
                    total_latency_ms=latency,
                    status="completed",
                )
                executed_runs.append(completed_run)

        # Select best run by highest composite score
        def _score(r: ExperimentRun) -> float:
            return (
                0.40 * r.metrics.get("faithfulness", 0.0)
                + 0.30 * r.metrics.get("answer_relevance", 0.0)
                + 0.15 * r.metrics.get("context_precision", 0.0)
                + 0.15 * r.metrics.get("context_recall", 0.0)
            )

        best_run = max(executed_runs, key=_score) if executed_runs else executed_runs[0]

        return GridSweepResponse(
            experiment_name=request.experiment_name,
            total_runs=len(executed_runs),
            runs=executed_runs,
            best_run_id=best_run.run_id if best_run else "",
            best_metric="composite_rag_score",
        )
