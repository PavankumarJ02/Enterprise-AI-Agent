"""Unit tests for RunComparisonEngine."""

import pytest

from enterprise_agent.experiments.comparator import RunComparisonEngine
from enterprise_agent.schemas.experiments import ExperimentRun


@pytest.fixture
def sample_runs() -> list[ExperimentRun]:
    """Provide sample baseline and candidate runs."""
    baseline = ExperimentRun(
        run_id="run-base",
        experiment_name="rag_tuning",
        run_name="baseline_dense",
        parameters={"retriever": "dense"},
        metrics={"faithfulness": 0.80, "context_recall": 0.75},
        tags=["baseline"],
        status="completed",
        total_latency_ms=200.0,
        created_at="2026-09-06T10:00:00Z",
    )
    candidate1 = ExperimentRun(
        run_id="run-cand1",
        experiment_name="rag_tuning",
        run_name="hybrid_rrf",
        parameters={"retriever": "hybrid"},
        metrics={"faithfulness": 0.90, "context_recall": 0.90},
        tags=["hybrid"],
        status="completed",
        total_latency_ms=150.0,
        created_at="2026-09-06T10:05:00Z",
    )
    candidate2 = ExperimentRun(
        run_id="run-cand2",
        experiment_name="rag_tuning",
        run_name="two_stage_rerank",
        parameters={"retriever": "rerank"},
        metrics={"faithfulness": 0.95, "context_recall": 0.85},
        tags=["rerank"],
        status="completed",
        total_latency_ms=250.0,
        created_at="2026-09-06T10:10:00Z",
    )
    return [baseline, candidate1, candidate2]


def test_compare_runs_metrics_and_deltas(sample_runs: list[ExperimentRun]) -> None:
    """Verify delta and percentage calculations against baseline."""
    comp = RunComparisonEngine.compare_runs(sample_runs)

    assert comp.baseline_run.run_id == "run-base"
    assert len(comp.candidate_runs) == 2

    # Verify cand1 improvements
    cand1_deltas = comp.metric_deltas["run-cand1"]
    assert cand1_deltas["faithfulness"].absolute_delta == 0.10
    assert cand1_deltas["faithfulness"].percentage_delta == 12.50
    assert cand1_deltas["faithfulness"].improved is True

    # Latency: 150ms vs 200ms -> lower latency is an improvement
    assert cand1_deltas["total_latency_ms"].absolute_delta == -50.0
    assert cand1_deltas["total_latency_ms"].improved is True

    # Cand2 latency: 250ms vs 200ms -> higher latency is NOT an improvement
    cand2_deltas = comp.metric_deltas["run-cand2"]
    assert cand2_deltas["total_latency_ms"].absolute_delta == 50.0
    assert cand2_deltas["total_latency_ms"].improved is False


def test_compare_runs_best_leader(sample_runs: list[ExperimentRun]) -> None:
    """Verify best_runs_per_metric accurately identifies winners."""
    comp = RunComparisonEngine.compare_runs(sample_runs)
    assert comp.best_runs_per_metric["faithfulness"] == "run-cand2"  # 0.95
    assert comp.best_runs_per_metric["context_recall"] == "run-cand1"  # 0.90
    assert comp.best_runs_per_metric["total_latency_ms"] == "run-cand1"  # 150ms (fastest)


def test_compare_runs_custom_baseline(sample_runs: list[ExperimentRun]) -> None:
    """Verify explicit baseline_run_id sets the designated run as reference."""
    comp = RunComparisonEngine.compare_runs(sample_runs, baseline_run_id="run-cand1")
    assert comp.baseline_run.run_id == "run-cand1"
    assert len(comp.candidate_runs) == 2
    assert {r.run_id for r in comp.candidate_runs} == {"run-base", "run-cand2"}


def test_compare_runs_validation_errors(sample_runs: list[ExperimentRun]) -> None:
    """Verify validation errors for single run or non-existent baseline ID."""
    with pytest.raises(ValueError, match="at least two"):
        RunComparisonEngine.compare_runs([sample_runs[0]])

    with pytest.raises(ValueError, match="not found"):
        RunComparisonEngine.compare_runs(sample_runs, baseline_run_id="invalid-id")
