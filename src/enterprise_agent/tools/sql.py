"""SQL Database tools for autonomous agent schema inspection and query execution."""

from typing import Any

from enterprise_agent.core.logging import get_logger
from enterprise_agent.sql.service import SQLDatabaseService
from enterprise_agent.tools.base import BaseTool, ToolResult

logger = get_logger(__name__)


class SQLSchemaTool(BaseTool):
    """Tool that introspects database tables, column types, and sample records."""

    def __init__(self, sql_service: SQLDatabaseService) -> None:
        self.sql_service = sql_service

    @property
    def name(self) -> str:
        """Tool name."""
        return "sql_schema"

    @property
    def description(self) -> str:
        """Tool purpose and instructions."""
        return (
            "Inspect the schema and sample rows of enterprise relational database tables. "
            "Available tables: departments, employees, products, sales_orders. "
            "Always call this first before writing a SQL query to verify exact column names."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema for parameters."""
        return {
            "type": "object",
            "properties": {
                "tables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of table names to inspect (e.g. ['products']).",
                }
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Retrieve and format database table schemas."""
        tables_arg = kwargs.get("tables")
        tables_list: list[str] | None = None
        if isinstance(tables_arg, list):
            tables_list = [str(t) for t in tables_arg]

        try:
            schema_resp = self.sql_service.get_schema(tables=tables_list)
            if not schema_resp.tables:
                return ToolResult(
                    output="No matching tables found in enterprise database.",
                    is_error=False,
                )

            sections: list[str] = []
            for t in schema_resp.tables:
                col_defs = ", ".join(f"{c.name} ({c.type})" for c in t.columns)
                sec = (
                    f"### Table: `{t.table_name}` (Total rows: {t.row_count})\n"
                    f"- **Columns**: {col_defs}\n"
                    f"- **Sample Data**:\n"
                )
                if t.sample_rows:
                    cols = [c.name for c in t.columns]
                    rows_data = [[row.get(c) for c in cols] for row in t.sample_rows]
                    table_md = SQLDatabaseService.format_as_markdown_table(cols, rows_data)
                    sec += f"{table_md}\n"
                else:
                    sec += "  (Table is empty)\n"
                sections.append(sec)

            output_text = "\n".join(sections)
            return ToolResult(
                output=output_text,
                metadata={"total_tables": schema_resp.total_tables},
            )
        except Exception as exc:
            logger.error("SQLSchemaTool failed: %s", exc)
            return ToolResult(output=f"Error inspecting database schema: {exc}", is_error=True)


class SQLQueryTool(BaseTool):
    """Tool that validates and executes read-only SQL queries against enterprise database."""

    def __init__(self, sql_service: SQLDatabaseService) -> None:
        self.sql_service = sql_service

    @property
    def name(self) -> str:
        """Tool name."""
        return "sql_query"

    @property
    def description(self) -> str:
        """Tool purpose and instructions."""
        return (
            "Execute a read-only SQL SELECT query against the enterprise database to retrieve "
            "tabular records, calculations, or aggregations. "
            "Only SELECT and WITH statements are permitted. Mutations and DDL are rejected."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema for parameters."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Read-only SQL query to execute.",
                }
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute validated query and format result into markdown table."""
        query_str = str(kwargs.get("query", "")).strip()
        if not query_str:
            return ToolResult(output="Error: No SQL query string provided.", is_error=True)

        try:
            res = self.sql_service.execute_query(query=query_str)
            table_md = SQLDatabaseService.format_as_markdown_table(res.columns, res.rows)
            summary = (
                f"{table_md}\n\n"
                f"*(Returned {res.row_count} rows in {res.execution_time_ms}ms"
                f"{', truncated to limit' if res.is_truncated else ''})*"
            )
            return ToolResult(
                output=summary,
                metadata={
                    "row_count": res.row_count,
                    "columns": res.columns,
                    "execution_time_ms": res.execution_time_ms,
                    "is_truncated": res.is_truncated,
                },
            )
        except Exception as exc:
            logger.error("SQLQueryTool execution error: %s", exc)
            return ToolResult(output=f"SQL Execution Error: {exc}", is_error=True)
