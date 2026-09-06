"""Unit tests for SQLite ExperimentRunRepository."""

from enterprise_agent.experiments.storage import ExperimentRunRepository
from enterprise_agent.schemas.experiments import ExperimentRun


def test_repository_create_and_get() -> None:
    """Verify persisting and reading back an experiment run in-memory."""
    repo = ExperimentRunRepository(db_path=":memory:")
    run = ExperimentRun(
        run_id="run-101",
        experiment_name="rag_optimization",
        run_name="dense_baseline",
        parameters={"top_k": 3, "retriever": "dense"},
        metrics={"faithfulness": 0.85, "context_recall": 0.90},
        tags=["baseline", "dense"],
        status="completed",
        total_latency_ms=120.5,
        created_at="2026-09-06T10:00:00Z",
    )
    repo.create_run(run)

    fetched = repo.get_run("run-101")
    assert fetched is not None
    assert fetched.run_id == "run-101"
    assert fetched.experiment_name == "rag_optimization"
    assert fetched.parameters["retriever"] == "dense"
    assert fetched.metrics["faithfulness"] == 0.85
    assert fetched.tags == ["baseline", "dense"]


def test_repository_update_run() -> None:
    """Verify updating metrics and status of an existing run."""
    repo = ExperimentRunRepository(db_path=":memory:")
    run = ExperimentRun(
        run_id="run-102",
        experiment_name="rag_optimization",
        run_name="hybrid_test",
        parameters={"retriever": "hybrid"},
        metrics={},
        tags=[],
        status="running",
        total_latency_ms=0.0,
        created_at="2026-09-06T10:05:00Z",
    )
    repo.create_run(run)

    updated = repo.update_run(
        run_id="run-102",
        metrics={"faithfulness": 0.95, "context_recall": 0.95},
        status="completed",
        total_latency_ms=145.2,
    )
    assert updated is not None
    assert updated.status == "completed"
    assert updated.metrics["faithfulness"] == 0.95
    assert updated.total_latency_ms == 145.2


def test_repository_list_and_filter() -> None:
    """Verify listing runs with experiment_name filtering and limits."""
    repo = ExperimentRunRepository(db_path=":memory:")
    for i in range(5):
        run = ExperimentRun(
            run_id=f"run-{i}",
            experiment_name="exp_a" if i < 3 else "exp_b",
            run_name=f"trial_{i}",
            parameters={},
            metrics={},
            tags=[],
            status="completed",
            total_latency_ms=10.0,
            created_at=f"2026-09-06T10:0{i}:00Z",
        )
        repo.create_run(run)

    all_runs = repo.list_runs()
    assert len(all_runs) == 5

    exp_a_runs = repo.list_runs(experiment_name="exp_a")
    assert len(exp_a_runs) == 3

    exp_b_runs = repo.list_runs(experiment_name="exp_b")
    assert len(exp_b_runs) == 2


def test_repository_delete() -> None:
    """Verify deleting a run record."""
    repo = ExperimentRunRepository(db_path=":memory:")
    run = ExperimentRun(
        run_id="run-delete-me",
        experiment_name="temp",
        run_name="to_delete",
        parameters={},
        metrics={},
        tags=[],
        status="completed",
        total_latency_ms=5.0,
        created_at="2026-09-06T10:00:00Z",
    )
    repo.create_run(run)
    assert repo.get_run("run-delete-me") is not None

    deleted = repo.delete_run("run-delete-me")
    assert deleted is True
    assert repo.get_run("run-delete-me") is None

    # Deleting non-existent run returns False
    assert repo.delete_run("non-existent") is False
