"""Unit tests for AST-based SQLValidator."""

import pytest

from enterprise_agent.core.exceptions import SQLSecurityError
from enterprise_agent.sql.validator import SQLValidator


@pytest.fixture
def validator() -> SQLValidator:
    """Fixture providing an instance of SQLValidator."""
    return SQLValidator(default_max_rows=50)


def test_validator_simple_select_appends_limit(validator: SQLValidator) -> None:
    """Verify clean SELECT statement appends default limit."""
    query = "SELECT * FROM products"
    sanitized = validator.validate_and_sanitize(query)
    assert sanitized == "SELECT * FROM products LIMIT 50"


def test_validator_preserves_smaller_limit(validator: SQLValidator) -> None:
    """Verify query with smaller limit retains specified limit."""
    query = "SELECT id, name FROM employees LIMIT 10"
    sanitized = validator.validate_and_sanitize(query)
    assert sanitized == "SELECT id, name FROM employees LIMIT 10"


def test_validator_clamps_excessive_limit(validator: SQLValidator) -> None:
    """Verify query requesting excessive limit is clamped to default_max_rows."""
    query = "SELECT id, name FROM employees LIMIT 500"
    sanitized = validator.validate_and_sanitize(query)
    assert sanitized == "SELECT id, name FROM employees LIMIT 50"


def test_validator_handles_trailing_semicolon(validator: SQLValidator) -> None:
    """Verify trailing semicolon is stripped before appending limit."""
    query = "SELECT name, price FROM products WHERE price > 2000;"
    sanitized = validator.validate_and_sanitize(query)
    assert sanitized == "SELECT name, price FROM products WHERE price > 2000 LIMIT 50"


def test_validator_permits_complex_join_and_aggregation(validator: SQLValidator) -> None:
    """Verify complex JOIN, GROUP BY, and aggregates pass validation."""
    query = (
        "SELECT d.name, COUNT(e.id) AS emp_count, AVG(e.salary) AS avg_sal "
        "FROM departments d "
        "JOIN employees e ON d.id = e.department_id "
        "GROUP BY d.name HAVING count(e.id) > 2 "
        "ORDER BY avg_sal DESC"
    )
    sanitized = validator.validate_and_sanitize(query)
    assert "LIMIT 50" in sanitized
    assert "JOIN employees" in sanitized


def test_validator_permits_common_table_expression(validator: SQLValidator) -> None:
    """Verify CTEs starting with WITH are permitted."""
    query = (
        "WITH high_earners AS ("
        "  SELECT * FROM employees WHERE salary > 150000"
        ") SELECT name, role FROM high_earners"
    )
    sanitized = validator.validate_and_sanitize(query)
    assert sanitized.startswith("WITH high_earners AS")
    assert "LIMIT 50" in sanitized


@pytest.mark.parametrize(
    "forbidden_query",
    [
        "INSERT INTO products (name, category, price, stock_quantity) VALUES ('Bad', 'X', 1, 1)",
        "UPDATE products SET price = 0 WHERE id = 1",
        "DELETE FROM products WHERE id = 1",
        "DROP TABLE products",
        "ALTER TABLE products ADD COLUMN discount REAL",
        "CREATE TABLE hackers (id INT)",
        "TRUNCATE TABLE products",
        "REPLACE INTO products (id, name) VALUES (1, 'Replaced')",
        "PRAGMA foreign_keys = OFF",
        "ATTACH DATABASE 'dummy.db' AS dummy",
        "VACUUM",
    ],
)
def test_validator_rejects_mutations_and_ddl(validator: SQLValidator, forbidden_query: str) -> None:
    """Verify all mutation, DDL, and administrative queries are rejected."""
    with pytest.raises(SQLSecurityError):
        validator.validate_and_sanitize(forbidden_query)


def test_validator_rejects_stacked_queries(validator: SQLValidator) -> None:
    """Verify multiple statements separated by semicolons are strictly rejected."""
    query = "SELECT * FROM products; DROP TABLE products;"
    with pytest.raises(SQLSecurityError, match="Multiple SQL statements"):
        validator.validate_and_sanitize(query)


def test_validator_rejects_empty_query(validator: SQLValidator) -> None:
    """Verify empty query raises error."""
    with pytest.raises(SQLSecurityError, match="empty"):
        validator.validate_and_sanitize("   ")


def test_validator_rejects_hazardous_sqlite_functions(validator: SQLValidator) -> None:
    """Verify hazardous SQLite filesystem access functions are rejected."""
    query = "SELECT readfile('/etc/passwd') FROM products"
    with pytest.raises(SQLSecurityError, match="Forbidden SQL function"):
        validator.validate_and_sanitize(query)
