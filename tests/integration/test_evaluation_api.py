"""Integration tests for RAG Evaluation REST API endpoints."""

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
        eval_faithfulness_threshold=0.75,
        eval_answer_relevance_threshold=0.60,
        eval_context_precision_threshold=0.70,
        eval_context_recall_threshold=0.70,
    )
    app = create_application(settings)
    return TestClient(app)


def test_api_evaluation_dataset(client: TestClient) -> None:
    """Verify endpoint returning the benchmark dataset."""
    resp = client.get("/api/v1/evaluation/dataset")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "enterprise_benchmark"
    assert data["total_samples"] == 10
    assert len(data["samples"]) == 10
    assert "query" in data["samples"][0]
    assert "ground_truth" in data["samples"][0]
    assert len(data["samples"][0]["golden_contexts"]) >= 1


def test_api_evaluation_run_default(client: TestClient) -> None:
    """Verify evaluation run over default benchmark dataset."""
    resp = client.post(
        "/api/v1/evaluation/run",
        json={"use_live_rag": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_samples"] == 10
    assert "faithfulness" in data["metrics"]
    assert "answer_relevance" in data["metrics"]
    assert "context_precision" in data["metrics"]
    assert "context_recall" in data["metrics"]
    assert data["overall_passed"] is True
    assert len(data["samples"]) == 10


def test_api_evaluation_run_custom_samples(client: TestClient) -> None:
    """Verify evaluation run with custom caller-provided samples."""
    custom_samples = [
        {
            "query": "What is the internet stipend?",
            "ground_truth": "The internet stipend is $100 per month.",
            "golden_contexts": [
                "Section 3.1: Remote employees receive a $100 monthly internet stipend."
            ],
        }
    ]
    resp = client.post(
        "/api/v1/evaluation/run",
        json={"samples": custom_samples, "use_live_rag": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_samples"] == 1
    assert len(data["samples"]) == 1
    sample_res = data["samples"][0]
    assert sample_res["faithfulness"] >= 0.90
    assert sample_res["context_precision"] == 1.0
    assert sample_res["context_recall"] == 1.0
