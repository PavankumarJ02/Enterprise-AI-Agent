"""Unit tests for SQL Agent Tools (SQLSchemaTool, SQLQueryTool)."""

from pathlib import Path

import pytest

from enterprise_agent.sql.connection import SQLiteManager
from enterprise_agent.sql.service import SQLDatabaseService
from enterprise_agent.tools.registry import ToolRegistry
from enterprise_agent.tools.sql import SQLQueryTool, SQLSchemaTool


@pytest.fixture
def sql_service(tmp_path: Path) -> SQLDatabaseService:
    """Fixture providing initialized SQLDatabaseService."""
    db_file = tmp_path / "tools_enterprise.db"
    manager = SQLiteManager(db_path=db_file)
    return SQLDatabaseService(manager=manager, default_max_rows=10)


@pytest.mark.asyncio
async def test_sql_schema_tool_execution(sql_service: SQLDatabaseService) -> None:
    """Verify SQLSchemaTool returns formatted markdown schema."""
    tool = SQLSchemaTool(sql_service=sql_service)
    assert tool.name == "sql_schema"
    assert "departments" in tool.description

    res = await tool.execute(tables=["departments"])
    assert not res.is_error
    assert "### Table: `departments`" in res.output
    assert "budget (REAL)" in res.output
    assert res.metadata.get("total_tables") == 1


@pytest.mark.asyncio
async def test_sql_query_tool_execution(sql_service: SQLDatabaseService) -> None:
    """Verify SQLQueryTool returns markdown table result and metadata."""
    tool = SQLQueryTool(sql_service=sql_service)
    assert tool.name == "sql_query"

    query = "SELECT name, price FROM products WHERE price > 4000 ORDER BY price DESC"
    res = await tool.execute(query=query)
    assert not res.is_error
    assert "| name | price |" in res.output
    row_cnt = res.metadata.get("row_count")
    assert isinstance(row_cnt, int) and row_cnt >= 1


@pytest.mark.asyncio
async def test_sql_query_tool_error_containment(sql_service: SQLDatabaseService) -> None:
    """Verify SQL errors are gracefully contained in ToolResult without throwing."""
    tool = SQLQueryTool(sql_service=sql_service)
    res = await tool.execute(query="DROP TABLE products")
    assert res.is_error is True
    assert "SQL Execution Error" in res.output


@pytest.mark.asyncio
async def test_sql_tools_registered_in_registry(sql_service: SQLDatabaseService) -> None:
    """Verify ToolRegistry can register and dispatch both SQL tools."""
    registry = ToolRegistry()
    registry.register(SQLSchemaTool(sql_service=sql_service))
    registry.register(SQLQueryTool(sql_service=sql_service))

    assert "sql_schema" in registry.get_tool_names()
    assert "sql_query" in registry.get_tool_names()

    # Execute via registry
    res = await registry.execute("sql_query", {"query": "SELECT COUNT(*) AS total FROM employees"})
    assert not res.is_error
    assert "total" in res.output
