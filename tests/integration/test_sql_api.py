"""Integration tests for SQL Database API endpoints (/api/v1/sql)."""

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from enterprise_agent.api.deps import (
    get_sql_service,
    reset_sql_service,
)
from enterprise_agent.config.settings import Settings
from enterprise_agent.main import create_application
from enterprise_agent.sql.connection import SQLiteManager
from enterprise_agent.sql.service import SQLDatabaseService


@pytest.fixture
def sql_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    """Test client configured with temporary seeded SQLite database."""
    reset_sql_service()
    db_file = tmp_path / "api_enterprise.db"
    settings = Settings(
        app_env="testing",
        llm_provider="mock",
        embedding_provider="mock",
        embedding_dimensions=16,
        qdrant_url=":memory:",
        sparse_search_provider="bm25",
        reranker_provider="mock",
        sql_db_path=str(db_file),
        sql_max_rows=25,
    )
    app = create_application(settings)

    manager = SQLiteManager(db_path=db_file)
    sql_service = SQLDatabaseService(manager=manager, default_max_rows=25)
    app.dependency_overrides[get_sql_service] = lambda: sql_service

    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()
        reset_sql_service()


def test_api_get_schema_all_tables(sql_client: TestClient) -> None:
    """Verify GET /api/v1/sql/schema returns all 4 enterprise tables."""
    response = sql_client.get("/api/v1/sql/schema")
    assert response.status_code == 200
    data = response.json()
    assert data["total_tables"] == 4
    table_names = [t["table_name"] for t in data["tables"]]
    assert "products" in table_names
    assert "employees" in table_names


def test_api_get_schema_filtered(sql_client: TestClient) -> None:
    """Verify GET /api/v1/sql/schema filters by tables query parameter."""
    response = sql_client.get("/api/v1/sql/schema?tables=products")
    assert response.status_code == 200
    data = response.json()
    assert data["total_tables"] == 1
    assert data["tables"][0]["table_name"] == "products"


def test_api_execute_valid_sql_query(sql_client: TestClient) -> None:
    """Verify POST /api/v1/sql/query executes read-only query and returns rows."""
    payload = {
        "query": "SELECT category, count(*) AS cnt FROM products GROUP BY category;",
        "max_rows": 10,
    }
    response = sql_client.post("/api/v1/sql/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["columns"] == ["category", "cnt"]
    assert len(data["rows"]) == 2
    assert data["is_truncated"] is False


def test_api_rejects_mutation_query(sql_client: TestClient) -> None:
    """Verify POST /api/v1/sql/query rejects forbidden mutations with 400 Bad Request."""
    payload = {"query": "DROP TABLE products;"}
    response = sql_client.post("/api/v1/sql/query", json=payload)
    assert response.status_code == 400
    assert "Forbidden" in response.json()["detail"]


def test_api_rejects_invalid_syntax(sql_client: TestClient) -> None:
    """Verify POST /api/v1/sql/query returns 400 Bad Request on SQL syntax errors."""
    payload = {"query": "SELECT * FROM imaginary_table_xyz;"}
    response = sql_client.post("/api/v1/sql/query", json=payload)
    assert response.status_code == 400
    assert "Database execution error" in response.json()["detail"]


def test_api_validates_empty_query(sql_client: TestClient) -> None:
    """Verify POST /api/v1/sql/query returns 422 for empty query string."""
    response = sql_client.post("/api/v1/sql/query", json={"query": ""})
    assert response.status_code == 422
