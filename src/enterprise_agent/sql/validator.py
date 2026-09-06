"""AST-based SQL security validator and sanitizer."""

import re

import sqlparse
from sqlparse.sql import Statement, TokenList
from sqlparse.tokens import Keyword, Name

from enterprise_agent.core.exceptions import SQLSecurityError
from enterprise_agent.core.logging import get_logger

logger = get_logger(__name__)

# Disallowed SQL mutation, DDL, and administrative keywords
_FORBIDDEN_KEYWORDS: set[str] = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "REPLACE",
    "EXEC",
    "EXECUTE",
    "PRAGMA",
    "ATTACH",
    "DETACH",
    "VACUUM",
    "REINDEX",
    "GRANT",
    "REVOKE",
    "COMMIT",
    "ROLLBACK",
    "SAVEPOINT",
    "TRANSACTION",
    "LOCK",
    "MERGE",
    "UPSERT",
}

# Dangerous SQLite functions that interact with filesystem or runtime
_FORBIDDEN_FUNCTIONS: set[str] = {
    "LOAD_EXTENSION",
    "READFILE",
    "WRITEFILE",
    "EDIT",
}


class SQLValidator:
    """Validates that SQL queries are strictly single-statement, read-only SELECTs."""

    def __init__(self, default_max_rows: int = 50) -> None:
        self.default_max_rows = default_max_rows

    def _extract_all_token_values(self, token_list: TokenList) -> list[str]:
        """Recursively extract string values of all tokens in a parsed statement."""
        values: list[str] = []
        for token in token_list.tokens:
            if isinstance(token, TokenList):
                values.extend(self._extract_all_token_values(token))
            else:
                val = token.value.strip()
                if val:
                    values.append(val)
        return values

    def _check_forbidden_tokens(self, stmt: Statement) -> None:
        """Inspect token tree for forbidden DDL, DML, or administrative keywords."""
        for token in stmt.flatten():  # type: ignore[no-untyped-call]
            # Check keywords
            if token.ttype in (Keyword, Keyword.DML, Keyword.DDL):
                word = token.value.upper()
                if word in _FORBIDDEN_KEYWORDS:
                    raise SQLSecurityError(
                        f"Forbidden SQL operation '{word}'. "
                        "Only read-only SELECT queries are allowed."
                    )
            # Check function calls and names
            if token.ttype in (Name, None) or hasattr(token, "value"):
                word = token.value.upper()
                if word in _FORBIDDEN_FUNCTIONS:
                    raise SQLSecurityError(f"Forbidden SQL function '{word}' detected.")
                if word in _FORBIDDEN_KEYWORDS:
                    raise SQLSecurityError(f"Forbidden keyword '{word}' detected in query.")

    def validate_and_sanitize(self, query: str, max_rows: int | None = None) -> str:
        """Validate query safety and enforce row limit constraints."""
        clean_query = query.strip()
        if not clean_query:
            raise SQLSecurityError("SQL query string cannot be empty.")

        # Strip trailing semicolon
        if clean_query.endswith(";"):
            clean_query = clean_query[:-1].strip()

        # Parse into AST statements
        parsed = sqlparse.parse(clean_query)
        non_empty = [s for s in parsed if str(s).strip() and str(s).strip() != ";"]

        if not non_empty:
            raise SQLSecurityError("No valid SQL statement detected.")

        if len(non_empty) > 1:
            raise SQLSecurityError(
                "Multiple SQL statements detected (stacked queries are strictly prohibited)."
            )

        stmt = non_empty[0]
        stmt_type = stmt.get_type()  # type: ignore[no-untyped-call]

        # Must be SELECT or CTE starting with WITH
        if stmt_type != "SELECT":
            # Check if query starts with WITH (CTE)
            first_token = next(
                (
                    t
                    for t in stmt.tokens
                    if not t.is_whitespace and t.ttype != sqlparse.tokens.Comment
                ),
                None,
            )
            if not first_token or first_token.value.upper() != "WITH":
                raise SQLSecurityError(
                    f"Forbidden statement type '{stmt_type}'. Only SELECT queries are permitted."
                )

        # Inspect AST tokens for blacklisted operations
        self._check_forbidden_tokens(stmt)

        # Limit enforcement
        limit = max_rows or self.default_max_rows
        limit_match = re.search(r"\bLIMIT\s+(\d+)", clean_query, re.IGNORECASE)
        if limit_match:
            existing_limit = int(limit_match.group(1))
            if existing_limit > limit:
                # Clamp excessive limit
                clean_query = re.sub(
                    r"\bLIMIT\s+\d+",
                    f"LIMIT {limit}",
                    clean_query,
                    flags=re.IGNORECASE,
                )
        else:
            # Append limit clause
            clean_query = f"{clean_query} LIMIT {limit}"

        return clean_query
