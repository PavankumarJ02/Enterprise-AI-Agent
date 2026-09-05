"""Integration tests for FastAPI endpoints."""

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from enterprise_agent.api.deps import get_llm
from enterprise_agent.core.exceptions import LLMAuthenticationError, LLMRateLimitError
from enterprise_agent.llm.base import LLMProvider


def test_health_endpoints(test_client: TestClient) -> None:
    """Verify both root /health and /api/v1/health return expected schemas."""
    # Root alias
    res_root = test_client.get("/health")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "healthy"

    # API v1 health
    res_v1 = test_client.get("/api/v1/health")
    assert res_v1.status_code == 200
    data = res_v1.json()
    assert data["status"] == "healthy"
    assert data["llm_healthy"] is True
    assert "version" in data
    assert "environment" in data


def test_chat_endpoint_success(test_client: TestClient) -> None:
    """Verify POST /api/v1/chat processes query and returns structured answer."""
    payload = {
        "query": "What is the standard password policy?",
        "system_prompt": "You are an enterprise assistant.",
        "temperature": 0.1,
    }

    response = test_client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "success"
    assert "answer" in body
    assert "What is the standard password policy?" in body["answer"]
    assert body["model"] == "mock-test-model"
    assert body["latency_ms"] >= 0.0
    assert "usage" in body
    assert body["usage"]["total_tokens"] > 0


def test_chat_endpoint_with_conversation_history(test_client: TestClient) -> None:
    """Verify chat endpoint accepts multi-turn conversation history."""
    payload = {
        "query": "Can you summarize it?",
        "history": [
            {"role": "user", "content": "Our vacation allowance is 20 days."},
            {"role": "assistant", "content": "Acknowledged. You have 20 vacation days."},
        ],
    }

    response = test_client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "Can you summarize it?" in body["answer"]


def test_chat_endpoint_validation_empty_query(test_client: TestClient) -> None:
    """Ensure empty query is rejected with 422 Unprocessable Entity."""
    response = test_client.post("/api/v1/chat", json={"query": ""})
    assert response.status_code == 422


def test_chat_endpoint_validation_invalid_temperature(test_client: TestClient) -> None:
    """Ensure out-of-range temperature is rejected with 422 Unprocessable Entity."""
    response = test_client.post(
        "/api/v1/chat",
        json={"query": "Valid query", "temperature": 3.5},
    )
    assert response.status_code == 422


def test_chat_endpoint_auth_error_handling(test_client: TestClient) -> None:
    """Ensure LLMAuthenticationError maps to HTTP 401 with structured error."""
    mock_failing_llm = AsyncMock(spec=LLMProvider)
    mock_failing_llm.generate.side_effect = LLMAuthenticationError("Invalid API Key supplied.")

    test_client.app.dependency_overrides[get_llm] = lambda: mock_failing_llm  # type: ignore[attr-defined]

    try:
        response = test_client.post("/api/v1/chat", json={"query": "Hello"})
        assert response.status_code == 401
        data = response.json()
        assert data["error"] == "LLM_AUTH_ERROR"
        assert "Invalid API Key" in data["message"]
    finally:
        test_client.app.dependency_overrides.pop(get_llm, None)  # type: ignore[attr-defined]


def test_chat_endpoint_rate_limit_error_handling(test_client: TestClient) -> None:
    """Ensure LLMRateLimitError maps to HTTP 429."""
    mock_failing_llm = AsyncMock(spec=LLMProvider)
    mock_failing_llm.generate.side_effect = LLMRateLimitError("Rate limit reached.")

    test_client.app.dependency_overrides[get_llm] = lambda: mock_failing_llm  # type: ignore[attr-defined]

    try:
        response = test_client.post("/api/v1/chat", json={"query": "Hello"})
        assert response.status_code == 429
        data = response.json()
        assert data["error"] == "LLM_RATE_LIMIT"
    finally:
        test_client.app.dependency_overrides.pop(get_llm, None)  # type: ignore[attr-defined]
