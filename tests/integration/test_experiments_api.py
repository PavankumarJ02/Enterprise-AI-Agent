"""Integration tests for Experiment Tracking REST API endpoints."""

import pytest
from starlette.testclient import TestClient

from enterprise_agent.config.settings import Settings
from enterprise_agent.main import create_application


@pytest.fixture
def client() -> TestClient:
    """Provide TestClient with freshly created FastAPI application."""
    settings = Settings(
        app_env="testing",
        llm_provider="mock",
        embedding_provider="mock",
        embedding_dimensions=16,
        qdrant_url=":memory:",
        experiments_db_path=":memory:",
    )
    app = create_application(settings)
    return TestClient(app)


def test_api_experiments_crud_lifecycle(client: TestClient) -> None:
    """Verify create, get, list, and update REST flow for experiment runs."""
    # 1. Create run
    create_resp = client.post(
        "/api/v1/experiments/runs",
        json={
            "experiment_name": "api_test_exp",
            "run_name": "trial_alpha",
            "parameters": {"strategy": "hybrid", "k": 3},
            "tags": ["alpha", "v1"],
        },
    )
    assert create_resp.status_code == 201
    run_data = create_resp.json()
    run_id = run_data["run_id"]
    assert run_data["experiment_name"] == "api_test_exp"
    assert run_data["status"] == "running"

    # 2. Get run by ID
    get_resp = client.get(f"/api/v1/experiments/runs/{run_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["run_id"] == run_id

    # 3. Update metrics
    update_resp = client.put(
        f"/api/v1/experiments/runs/{run_id}",
        json={
            "metrics": {"faithfulness": 0.89, "answer_relevance": 0.85},
            "status": "completed",
            "total_latency_ms": 130.0,
        },
    )
    assert update_resp.status_code == 200
    updated_data = update_resp.json()
    assert updated_data["status"] == "completed"
    assert updated_data["metrics"]["faithfulness"] == 0.89

    # 4. List runs
    list_resp = client.get("/api/v1/experiments/runs?experiment_name=api_test_exp")
    assert list_resp.status_code == 200
    runs = list_resp.json()
    assert len(runs) >= 1
    assert any(r["run_id"] == run_id for r in runs)


def test_api_experiments_compare(client: TestClient) -> None:
    """Verify comparing two runs side-by-side."""
    # Create run 1
    r1 = client.post(
        "/api/v1/experiments/runs",
        json={"experiment_name": "compare_api", "run_name": "r1"},
    ).json()
    client.put(
        f"/api/v1/experiments/runs/{r1['run_id']}",
        json={"metrics": {"faithfulness": 0.70}, "total_latency_ms": 200.0},
    )

    # Create run 2
    r2 = client.post(
        "/api/v1/experiments/runs",
        json={"experiment_name": "compare_api", "run_name": "r2"},
    ).json()
    client.put(
        f"/api/v1/experiments/runs/{r2['run_id']}",
        json={"metrics": {"faithfulness": 0.90}, "total_latency_ms": 150.0},
    )

    # Compare
    compare_resp = client.post(
        "/api/v1/experiments/compare",
        json={"run_ids": [r1["run_id"], r2["run_id"]], "baseline_run_id": r1["run_id"]},
    )
    assert compare_resp.status_code == 200
    comp_data = compare_resp.json()
    assert comp_data["baseline_run"]["run_id"] == r1["run_id"]
    assert len(comp_data["candidate_runs"]) == 1

    deltas = comp_data["metric_deltas"][r2["run_id"]]
    assert deltas["faithfulness"]["improved"] is True
    assert deltas["total_latency_ms"]["improved"] is True


def test_api_experiments_grid_sweep(client: TestClient) -> None:
    """Verify triggering an automated grid search sweep."""
    resp = client.post(
        "/api/v1/experiments/grid",
        json={
            "experiment_name": "api_grid_sweep",
            "retrieval_strategies": ["dense", "hybrid"],
            "top_k_values": [3],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["experiment_name"] == "api_grid_sweep"
    assert data["total_runs"] == 2
    assert len(data["runs"]) == 2
    assert data["best_run_id"] != ""
