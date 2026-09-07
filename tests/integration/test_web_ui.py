"""Integration tests verifying web UI serving, static assets, and
performance telemetry endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from enterprise_agent.main import create_application


@pytest.fixture
def web_client() -> TestClient:
    """Test client fixture for Web UI testing."""
    app = create_application()
    return TestClient(app)


def test_root_index_serves_html(web_client: TestClient) -> None:
    """Verify GET / returns HTTP 200 with HTML content."""
    response = web_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Enterprise AI" in response.text
    assert "data-theme" in response.text


def test_performance_stats_endpoint(web_client: TestClient) -> None:
    """Verify GET /api/v1/performance/stats returns aggregated metrics."""
    response = web_client.get("/api/v1/performance/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert "overall_hit_ratio_pct" in data
    assert "embedding_cache" in data
    assert "retrieval_cache" in data


def test_performance_reset_endpoint(web_client: TestClient) -> None:
    """Verify POST /api/v1/performance/reset clears caches."""
    response = web_client.post("/api/v1/performance/reset")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
