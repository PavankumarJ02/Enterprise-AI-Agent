"""Unit tests for ExperimentTrackingService."""

import pytest

from enterprise_agent.evaluation.service import RAGEvaluationService
from enterprise_agent.experiments.service import ExperimentTrackingService
from enterprise_agent.experiments.storage import ExperimentRunRepository
from enterprise_agent.schemas.experiments import ExperimentRunCreate, GridSweepRequest


@pytest.fixture
def experiment_service() -> ExperimentTrackingService:
    """Provide ExperimentTrackingService connected to an in-memory repository."""
    repo = ExperimentRunRepository(db_path=":memory:")
    eval_svc = RAGEvaluationService()
    return ExperimentTrackingService(repository=repo, evaluation_service=eval_svc)


def test_service_create_and_log_run(experiment_service: ExperimentTrackingService) -> None:
    """Verify end-to-end run creation and metric logging."""
    run = experiment_service.create_run(
        ExperimentRunCreate(
            experiment_name="test_exp",
            run_name="trial_1",
            parameters={"model": "gemini-2.5-flash", "top_k": 3},
            tags=["gemini", "k3"],
        )
    )
    assert run.status == "running"
    assert run.parameters["top_k"] == 3

    updated = experiment_service.log_run_metrics(
        run_id=run.run_id,
        metrics={"faithfulness": 0.88, "answer_relevance": 0.82},
        total_latency_ms=115.0,
        status="completed",
    )
    assert updated.status == "completed"
    assert updated.metrics["faithfulness"] == 0.88
    assert updated.total_latency_ms == 115.0


def test_service_compare_runs(experiment_service: ExperimentTrackingService) -> None:
    """Verify comparing runs through service coordinator."""
    run1 = experiment_service.create_run(
        ExperimentRunCreate(experiment_name="cmp_exp", run_name="run_1")
    )
    experiment_service.log_run_metrics(
        run_id=run1.run_id,
        metrics={"faithfulness": 0.80},
        total_latency_ms=100.0,
    )

    run2 = experiment_service.create_run(
        ExperimentRunCreate(experiment_name="cmp_exp", run_name="run_2")
    )
    experiment_service.log_run_metrics(
        run_id=run2.run_id,
        metrics={"faithfulness": 0.90},
        total_latency_ms=80.0,
    )

    comparison = experiment_service.compare_runs(
        run_ids=[run1.run_id, run2.run_id],
        baseline_run_id=run1.run_id,
    )
    assert comparison.baseline_run.run_id == run1.run_id
    assert len(comparison.candidate_runs) == 1
    assert comparison.best_runs_per_metric["faithfulness"] == run2.run_id
    assert comparison.best_runs_per_metric["total_latency_ms"] == run2.run_id


@pytest.mark.asyncio
async def test_service_grid_sweep(experiment_service: ExperimentTrackingService) -> None:
    """Verify grid sweep executes all combinations and records leaderboard."""
    request = GridSweepRequest(
        experiment_name="grid_test",
        retrieval_strategies=["dense", "hybrid"],
        top_k_values=[3, 5],
    )
    response = await experiment_service.run_grid_sweep(request)

    assert response.experiment_name == "grid_test"
    assert response.total_runs == 4
    assert len(response.runs) == 4
    assert response.best_run_id != ""

    # Verify all runs were saved in the repository
    stored_runs = experiment_service.list_runs(experiment_name="grid_test")
    assert len(stored_runs) == 4
