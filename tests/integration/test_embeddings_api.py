"""Integration tests for dense embedding API endpoints."""

from fastapi.testclient import TestClient


def test_embed_texts_endpoint(test_client: TestClient) -> None:
    """Verify POST /api/v1/embeddings generates dense vectors for text batches."""
    payload = {
        "texts": [
            "Corporate password policy requires minimum 12 characters.",
            "Vacation allowance is 20 days per fiscal year.",
            "Travel per diem for domestic flights is $120/day.",
        ]
    }

    response = test_client.post("/api/v1/embeddings", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["count"] == 3
    assert len(data["vectors"]) == 3
    assert data["dimensions"] > 0
    assert len(data["vectors"][0]) == data["dimensions"]
    assert data["latency_ms"] >= 0.0
    assert "model" in data


def test_embed_texts_empty_validation(test_client: TestClient) -> None:
    """Verify empty text batch returns 422 Unprocessable Entity."""
    response = test_client.post("/api/v1/embeddings", json={"texts": []})
    assert response.status_code == 422


def test_embed_query_endpoint(test_client: TestClient) -> None:
    """Verify POST /api/v1/embeddings/query returns single vector for search query."""
    payload = {"query": "What is the password expiration interval?"}

    response = test_client.post("/api/v1/embeddings/query", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "vector" in data
    assert len(data["vector"]) == data["dimensions"]
    assert data["latency_ms"] >= 0.0


def test_embed_query_empty_validation(test_client: TestClient) -> None:
    """Verify empty query string is rejected with 422 Unprocessable Entity."""
    response = test_client.post("/api/v1/embeddings/query", json={"query": ""})
    assert response.status_code == 422
