"""Integration tests for Semantic Query Router API endpoints (/api/v1/router)."""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from enterprise_agent.config.settings import Settings
from enterprise_agent.main import create_application


@pytest.fixture
def router_client() -> Generator[TestClient, None, None]:
    """Test client configured with mock providers for router testing."""
    settings = Settings(
        app_env="testing",
        llm_provider="mock",
        embedding_provider="mock",
        embedding_dimensions=16,
        qdrant_url=":memory:",
        sparse_search_provider="bm25",
        reranker_provider="mock",
        router_strategy="cascade",
    )
    app = create_application(settings)
    client = TestClient(app)
    yield client


def test_api_classify_greeting(router_client: TestClient) -> None:
    """Verify POST /api/v1/router/classify routes greetings to DIRECT_CHAT."""
    response = router_client.post(
        "/api/v1/router/classify",
        json={"query": "Hello there! How are you?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "direct_chat"
    assert data["confidence"] >= 0.90
    assert data["strategy_used"] == "heuristic"
    assert data["latency_ms"] >= 0.0


def test_api_classify_policy_question(router_client: TestClient) -> None:
    """Verify POST /api/v1/router/classify routes policy question to RAG_SEARCH."""
    response = router_client.post(
        "/api/v1/router/classify",
        json={"query": "What is the company parental leave policy?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "rag_search"
    assert data["confidence"] >= 0.80


def test_api_classify_sql_metrics(router_client: TestClient) -> None:
    """Verify POST /api/v1/router/classify routes employee metrics to SQL_DATABASE."""
    response = router_client.post(
        "/api/v1/router/classify",
        json={"query": "How many employees work in Engineering?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "sql_database"
    assert data["confidence"] >= 0.85


def test_api_classify_compound_task(router_client: TestClient) -> None:
    """Verify POST /api/v1/router/classify routes compound task to AUTONOMOUS_AGENT."""
    response = router_client.post(
        "/api/v1/router/classify",
        json={"query": "Look up the meal rate in handbook and calculate total for 5 people"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "autonomous_agent"


def test_api_classify_empty_query_validation(router_client: TestClient) -> None:
    """Verify POST /api/v1/router/classify returns 422 for empty query."""
    response = router_client.post("/api/v1/router/classify", json={"query": ""})
    assert response.status_code == 422


def test_api_dispatch_direct_chat(router_client: TestClient) -> None:
    """Verify POST /api/v1/router/dispatch routes and executes conversational query."""
    response = router_client.post(
        "/api/v1/router/dispatch",
        json={"query": "Good morning assistant!"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "direct_chat"
    assert "response" in data
    assert "answer" in data["response"]


def test_api_dispatch_forced_intent(router_client: TestClient) -> None:
    """Verify POST /api/v1/router/dispatch honors force_intent override."""
    response = router_client.post(
        "/api/v1/router/dispatch",
        json={
            "query": "Tell me a joke",
            "force_intent": "direct_chat",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "direct_chat"
    assert data["strategy_used"] == "forced_override"
