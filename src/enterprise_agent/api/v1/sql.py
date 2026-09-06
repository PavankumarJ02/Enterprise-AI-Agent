"""SQL database API endpoints for schema introspection and safe read-only queries."""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from enterprise_agent.api.deps import get_sql_service
from enterprise_agent.core.exceptions import SQLExecutionError, SQLSecurityError
from enterprise_agent.core.logging import get_logger
from enterprise_agent.schemas.sql import (
    DatabaseSchemaResponse,
    SQLQueryRequest,
    SQLQueryResponse,
)
from enterprise_agent.sql.service import SQLDatabaseService

logger = get_logger(__name__)

router = APIRouter(prefix="/sql", tags=["SQL Database Tool"])


@router.get(
    "/schema",
    response_model=DatabaseSchemaResponse,
    status_code=status.HTTP_200_OK,
    summary="Inspect enterprise database schema",
    description=(
        "Returns table names, column data types, nullability, primary key markers, "
        "and sample rows for enterprise database tables."
    ),
)
async def get_database_schema(
    tables: list[str] | None = Query(
        default=None,
        description="Optional list of table names to filter introspection results.",
    ),
    sql_service: SQLDatabaseService = Depends(get_sql_service),
) -> DatabaseSchemaResponse:
    """Retrieve table schemas and previews from enterprise database."""
    try:
        return sql_service.get_schema(tables=tables)
    except Exception as exc:
        logger.error("Failed to introspect database schema: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to introspect database schema: {exc}",
        ) from exc


@router.post(
    "/query",
    response_model=SQLQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute validated read-only SQL query",
    description=(
        "Executes a sanitized read-only SQL query against the enterprise database. "
        "Mutations (INSERT, UPDATE, DELETE, DROP), DDL, and admin operations are rejected."
    ),
)
async def execute_sql_query(
    request: SQLQueryRequest,
    sql_service: SQLDatabaseService = Depends(get_sql_service),
) -> SQLQueryResponse:
    """Execute sanitized read-only SQL query."""
    try:
        return sql_service.execute_query(
            query=request.query,
            max_rows=request.max_rows,
        )
    except SQLSecurityError as sec_err:
        logger.warning("Rejected unsafe SQL query '%s': %s", request.query, sec_err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(sec_err),
        ) from sec_err
    except SQLExecutionError as exec_err:
        logger.warning("SQL execution failed for query '%s': %s", request.query, exec_err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exec_err),
        ) from exec_err
    except Exception as exc:
        logger.error("Unexpected error executing SQL query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal database query error: {exc}",
        ) from exc
