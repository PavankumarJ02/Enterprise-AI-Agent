"""Request and response schemas for SQL database tooling and schema reflection."""

from typing import Any

from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    """Metadata describing a single database table column."""

    name: str = Field(..., description="Column identifier name.")
    type: str = Field(..., description="SQL data type (e.g. INTEGER, TEXT, REAL).")
    nullable: bool = Field(default=True, description="Whether NULL values are permitted.")
    primary_key: bool = Field(default=False, description="Whether column is the primary key.")


class TableSchema(BaseModel):
    """Schema and preview metadata for a single database table."""

    table_name: str = Field(..., description="Name of the table.")
    columns: list[ColumnInfo] = Field(
        default_factory=list,
        description="List of column definitions.",
    )
    sample_rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Sample rows demonstrating data values.",
    )
    row_count: int = Field(default=0, description="Total rows in the table.")


class DatabaseSchemaResponse(BaseModel):
    """Complete schema inspection response covering all or selected database tables."""

    tables: list[TableSchema] = Field(
        default_factory=list,
        description="List of table schemas.",
    )
    total_tables: int = Field(default=0, description="Number of tables included.")


class SQLQueryRequest(BaseModel):
    """Request payload to execute a read-only SQL query."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Read-only SQL query to execute (must start with SELECT or WITH).",
        examples=[
            "SELECT category, SUM(price * stock_quantity) AS total_val "
            "FROM products GROUP BY category;"
        ],
    )
    max_rows: int | None = Field(
        default=None,
        ge=1,
        le=500,
        description="Optional maximum number of rows to retrieve (capped by system limit).",
    )


class SQLQueryResponse(BaseModel):
    """Execution result and metadata for a read-only SQL query."""

    query: str = Field(..., description="Executed SQL query string.")
    columns: list[str] = Field(
        default_factory=list,
        description="Column header names for the result set.",
    )
    rows: list[list[Any]] = Field(
        default_factory=list,
        description="Result set rows as lists of positional column values.",
    )
    row_count: int = Field(default=0, description="Total number of returned rows.")
    is_truncated: bool = Field(
        default=False,
        description="Whether result rows were truncated by row limits.",
    )
    execution_time_ms: float = Field(
        default=0.0,
        description="Execution duration in milliseconds.",
    )
    error: str | None = Field(
        default=None,
        description="Error message if query execution failed.",
    )
