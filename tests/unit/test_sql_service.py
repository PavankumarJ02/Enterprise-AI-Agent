"""Unit tests for SQLiteManager and SQLDatabaseService."""

from pathlib import Path

import pytest

from enterprise_agent.core.exceptions import SQLExecutionError, SQLSecurityError
from enterprise_agent.sql.connection import SQLiteManager
from enterprise_agent.sql.service import SQLDatabaseService


@pytest.fixture
def temp_sql_service(tmp_path: Path) -> SQLDatabaseService:
    """Fixture providing a fresh seeded database service in a temp directory."""
    db_file = tmp_path / "test_enterprise.db"
    manager = SQLiteManager(db_path=db_file)
    return SQLDatabaseService(
        manager=manager,
        default_max_rows=10,
        query_timeout_seconds=5.0,
    )


def test_sql_service_seeds_tables_and_schema(temp_sql_service: SQLDatabaseService) -> None:
    """Verify seeder generates all 4 expected tables with non-zero row counts."""
    schema = temp_sql_service.get_schema()
    table_names = {t.table_name for t in schema.tables}
    assert table_names == {"departments", "employees", "products", "sales_orders"}

    dept_table = next(t for t in schema.tables if t.table_name == "departments")
    assert dept_table.row_count == 5
    col_names = [c.name for c in dept_table.columns]
    assert "name" in col_names
    assert "budget" in col_names
    assert len(dept_table.sample_rows) == 2


def test_sql_service_filters_schema_by_table_name(temp_sql_service: SQLDatabaseService) -> None:
    """Verify get_schema accepts specific tables list."""
    schema = temp_sql_service.get_schema(tables=["products"])
    assert len(schema.tables) == 1
    assert schema.tables[0].table_name == "products"
    assert schema.tables[0].row_count == 10


def test_sql_service_executes_valid_query(temp_sql_service: SQLDatabaseService) -> None:
    """Verify successful execution of read-only aggregation query."""
    query = (
        "SELECT category, COUNT(*) AS count, AVG(price) AS avg_price "
        "FROM products GROUP BY category ORDER BY count DESC"
    )
    result = temp_sql_service.execute_query(query=query)
    assert result.columns == ["category", "count", "avg_price"]
    assert len(result.rows) == 2  # Software, Hardware
    assert not result.is_truncated
    assert result.execution_time_ms >= 0.0


def test_sql_service_truncation_detection(temp_sql_service: SQLDatabaseService) -> None:
    """Verify is_truncated is True when query returns more rows than max_rows limit."""
    query = "SELECT id, name FROM employees"
    # employees has 20 rows, max_rows is 5
    result = temp_sql_service.execute_query(query=query, max_rows=5)
    assert result.row_count == 5
    assert result.is_truncated is True


def test_sql_service_rejects_unsafe_query(temp_sql_service: SQLDatabaseService) -> None:
    """Verify SQLSecurityError is propagated on disallowed mutations."""
    with pytest.raises(SQLSecurityError):
        temp_sql_service.execute_query("DELETE FROM products WHERE id = 1")


def test_sql_service_handles_syntax_error(temp_sql_service: SQLDatabaseService) -> None:
    """Verify syntax error in SELECT statement raises SQLExecutionError."""
    with pytest.raises(SQLExecutionError):
        temp_sql_service.execute_query("SELECT non_existent_col FROM nonexistent_table")


def test_format_as_markdown_table() -> None:
    """Verify markdown table formatting utility."""
    cols = ["id", "name", "price"]
    rows = [[1, "Widget", 19.99], [2, "Gadget", None]]
    table_md = SQLDatabaseService.format_as_markdown_table(cols, rows)

    assert "| id | name | price |" in table_md
    assert "| --- | --- | --- |" in table_md
    assert "| 1 | Widget | 19.99 |" in table_md
    assert "| 2 | Gadget | NULL |" in table_md
