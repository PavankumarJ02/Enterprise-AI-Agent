"""SQLite connection manager enforcing read-only access modes."""

import sqlite3
from pathlib import Path

from enterprise_agent.core.logging import get_logger
from enterprise_agent.sql.seed import seed_enterprise_db

logger = get_logger(__name__)


class SQLiteManager:
    """Manages SQLite enterprise database lifecycle and read-only connections."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).resolve()
        # Seed or verify database schema on startup
        seed_enterprise_db(self.db_path)

    def get_read_only_connection(self) -> sqlite3.Connection:
        """Create a dedicated read-only SQLite connection using URI mode."""
        # file:path?mode=ro guarantees SQLite rejects all DDL and DML operations
        db_uri = f"file:{self.db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn
