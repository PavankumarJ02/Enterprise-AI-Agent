"""SQL Database service coordinating validation, execution, and schema reflection."""

import re
import sqlite3
import time
from typing import Any

from enterprise_agent.core.exceptions import SQLExecutionError, SQLSecurityError
from enterprise_agent.core.logging import get_logger
from enterprise_agent.schemas.sql import (
    ColumnInfo,
    DatabaseSchemaResponse,
    SQLQueryResponse,
    TableSchema,
)
from enterprise_agent.sql.connection import SQLiteManager
from enterprise_agent.sql.validator import SQLValidator

logger = get_logger(__name__)


class SQLDatabaseService:
    """Provides read-only enterprise database queries and schema introspection."""

    def __init__(
        self,
        manager: SQLiteManager,
        validator: SQLValidator | None = None,
        default_max_rows: int = 50,
        query_timeout_seconds: float = 5.0,
    ) -> None:
        self.manager = manager
        self.default_max_rows = default_max_rows
        self.query_timeout_seconds = query_timeout_seconds
        self.validator = validator or SQLValidator(default_max_rows=default_max_rows)

    def get_schema(self, tables: list[str] | None = None) -> DatabaseSchemaResponse:
        """Inspect and return schema and sample rows for enterprise database tables."""
        conn = self.manager.get_read_only_connection()
        cursor = conn.cursor()
        try:
            # Query existing user tables
            cursor.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name;"
            )
            all_tables = [row["name"] for row in cursor.fetchall()]

            target_tables = all_tables
            if tables:
                table_set = {t.lower() for t in tables}
                target_tables = [t for t in all_tables if t.lower() in table_set]

            schemas: list[TableSchema] = []
            for t_name in target_tables:
                # Introspect column info
                cursor.execute(f"PRAGMA table_info('{t_name}');")  # noqa: S608
                col_rows = cursor.fetchall()
                columns = [
                    ColumnInfo(
                        name=col["name"],
                        type=col["type"],
                        nullable=not bool(col["notnull"]),
                        primary_key=bool(col["pk"]),
                    )
                    for col in col_rows
                ]

                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM '{t_name}';")  # noqa: S608
                row_count = int(cursor.fetchone()[0])

                # Get 2 sample rows
                cursor.execute(f"SELECT * FROM '{t_name}' LIMIT 2;")  # noqa: S608
                sample_rows = [dict(r) for r in cursor.fetchall()]

                schemas.append(
                    TableSchema(
                        table_name=t_name,
                        columns=columns,
                        sample_rows=sample_rows,
                        row_count=row_count,
                    )
                )

            return DatabaseSchemaResponse(tables=schemas, total_tables=len(schemas))
        finally:
            conn.close()

    def execute_query(
        self,
        query: str,
        max_rows: int | None = None,
    ) -> SQLQueryResponse:
        """Validate, sanitize, and execute read-only SQL query against the database."""
        start_time = time.perf_counter()
        effective_limit = max_rows or self.default_max_rows

        # 1. AST Validation and Limit Injection
        try:
            sanitized_query = self.validator.validate_and_sanitize(
                query=query,
                max_rows=effective_limit,
            )
        except SQLSecurityError:
            raise
        except Exception as exc:
            raise SQLSecurityError(f"SQL validation error: {exc}") from exc

        # 2. Execute on Read-Only Connection
        conn = self.manager.get_read_only_connection()
        cursor = conn.cursor()
        try:
            exec_query = re.sub(
                r"\bLIMIT\s+\d+",
                f"LIMIT {effective_limit + 1}",
                sanitized_query,
                flags=re.IGNORECASE,
            )
            cursor.execute(exec_query)
            description = cursor.description or []
            col_names = [col[0] for col in description]

            # Fetch limit + 1 to detect truncation
            raw_rows = cursor.fetchall()
            is_truncated = len(raw_rows) > effective_limit
            result_rows = [list(r) for r in raw_rows[:effective_limit]]

            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info(
                "SQL query executed in %.2fms, returned %d rows (truncated=%s)",
                duration_ms,
                len(result_rows),
                is_truncated,
            )

            return SQLQueryResponse(
                query=sanitized_query,
                columns=col_names,
                rows=result_rows,
                row_count=len(result_rows),
                is_truncated=is_truncated,
                execution_time_ms=duration_ms,
            )
        except sqlite3.OperationalError as op_err:
            logger.error("SQL operational error during query execution: %s", op_err)
            raise SQLExecutionError(f"Database execution error: {op_err}") from op_err
        except Exception as exc:
            logger.error("Unexpected error during SQL query execution: %s", exc)
            raise SQLExecutionError(f"Query failure: {exc}") from exc
        finally:
            conn.close()

    @staticmethod
    def format_as_markdown_table(columns: list[str], rows: list[list[Any]]) -> str:
        """Format tabular SQL query result into a clean markdown table."""
        if not columns:
            return "No columns returned."
        if not rows:
            return "Query executed successfully. 0 rows returned."

        header = "| " + " | ".join(str(c) for c in columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        data_rows = [
            "| " + " | ".join(str(val) if val is not None else "NULL" for val in row) + " |"
            for row in rows
        ]
        return "\n".join([header, separator] + data_rows)
