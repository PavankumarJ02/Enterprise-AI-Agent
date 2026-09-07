"""Persistent and in-memory storage implementations for API key metadata."""

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from enterprise_agent.core.logging import get_logger
from enterprise_agent.security.base import APIKeyMetadata, APIKeyScope, APIKeyStatus, APIKeyStore

logger = get_logger(__name__)


class InMemoryAPIKeyStore(APIKeyStore):
    """Thread-safe ephemeral in-memory storage for API key metadata."""

    def __init__(self) -> None:
        self._keys_by_id: dict[str, APIKeyMetadata] = {}
        self._keys_by_hash: dict[str, str] = {}  # key_hash -> key_id

    async def save_key(self, key_meta: APIKeyMetadata) -> None:
        self._keys_by_id[key_meta.key_id] = key_meta
        self._keys_by_hash[key_meta.key_hash] = key_meta.key_id

    async def get_key_by_hash(self, key_hash: str) -> APIKeyMetadata | None:
        key_id = self._keys_by_hash.get(key_hash)
        if not key_id:
            return None
        return self._keys_by_id.get(key_id)

    async def get_key_by_id(self, key_id: str) -> APIKeyMetadata | None:
        return self._keys_by_id.get(key_id)

    async def list_keys(self) -> list[APIKeyMetadata]:
        return list(self._keys_by_id.values())

    async def update_key(self, key_meta: APIKeyMetadata) -> None:
        old_meta = self._keys_by_id.get(key_meta.key_id)
        if old_meta and old_meta.key_hash != key_meta.key_hash:
            self._keys_by_hash.pop(old_meta.key_hash, None)
        self._keys_by_id[key_meta.key_id] = key_meta
        self._keys_by_hash[key_meta.key_hash] = key_meta.key_id

    async def delete_key(self, key_id: str) -> bool:
        meta = self._keys_by_id.pop(key_id, None)
        if meta:
            self._keys_by_hash.pop(meta.key_hash, None)
            return True
        return False


class SQLiteAPIKeyStore(APIKeyStore):
    """SQLite-backed persistent storage for API key metadata records."""

    def __init__(self, db_path: str = "data/api_keys.db") -> None:
        self.db_path = db_path
        self._shared_conn: sqlite3.Connection | None = None
        if db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        if self._shared_conn is not None:
            with self._shared_conn:
                yield self._shared_conn
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                with conn:
                    yield conn
            finally:
                conn.close()

    def close(self) -> None:
        """Close shared connection if open."""
        if self._shared_conn is not None:
            self._shared_conn.close()
            self._shared_conn = None

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    key_hash TEXT NOT NULL UNIQUE,
                    prefix TEXT NOT NULL,
                    scopes TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    rotated_at TEXT,
                    grace_expires_at TEXT,
                    description TEXT NOT NULL DEFAULT ''
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);")

    def _row_to_metadata(self, row: sqlite3.Row) -> APIKeyMetadata:
        scopes_raw = json.loads(row["scopes"])
        return APIKeyMetadata(
            key_id=row["key_id"],
            name=row["name"],
            key_hash=row["key_hash"],
            prefix=row["prefix"],
            scopes=[APIKeyScope(s) for s in scopes_raw],
            status=APIKeyStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            rotated_at=datetime.fromisoformat(row["rotated_at"]) if row["rotated_at"] else None,
            grace_expires_at=(
                datetime.fromisoformat(row["grace_expires_at"]) if row["grace_expires_at"] else None
            ),
            description=row["description"],
        )

    async def save_key(self, key_meta: APIKeyMetadata) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO api_keys (
                    key_id, name, key_hash, prefix, scopes, status,
                    created_at, expires_at, rotated_at, grace_expires_at, description
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    key_meta.key_id,
                    key_meta.name,
                    key_meta.key_hash,
                    key_meta.prefix,
                    json.dumps([s.value for s in key_meta.scopes]),
                    key_meta.status.value,
                    key_meta.created_at.isoformat(),
                    key_meta.expires_at.isoformat() if key_meta.expires_at else None,
                    key_meta.rotated_at.isoformat() if key_meta.rotated_at else None,
                    (key_meta.grace_expires_at.isoformat() if key_meta.grace_expires_at else None),
                    key_meta.description,
                ),
            )

    async def get_key_by_hash(self, key_hash: str) -> APIKeyMetadata | None:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM api_keys WHERE key_hash = ?;", (key_hash,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_metadata(row)

    async def get_key_by_id(self, key_id: str) -> APIKeyMetadata | None:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM api_keys WHERE key_id = ?;", (key_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_metadata(row)

    async def list_keys(self) -> list[APIKeyMetadata]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM api_keys ORDER BY created_at DESC;")
            return [self._row_to_metadata(r) for r in cursor.fetchall()]

    async def update_key(self, key_meta: APIKeyMetadata) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE api_keys
                SET name = ?, key_hash = ?, prefix = ?, scopes = ?, status = ?,
                    created_at = ?, expires_at = ?, rotated_at = ?, grace_expires_at = ?,
                    description = ?
                WHERE key_id = ?;
                """,
                (
                    key_meta.name,
                    key_meta.key_hash,
                    key_meta.prefix,
                    json.dumps([s.value for s in key_meta.scopes]),
                    key_meta.status.value,
                    key_meta.created_at.isoformat(),
                    key_meta.expires_at.isoformat() if key_meta.expires_at else None,
                    key_meta.rotated_at.isoformat() if key_meta.rotated_at else None,
                    (key_meta.grace_expires_at.isoformat() if key_meta.grace_expires_at else None),
                    key_meta.description,
                    key_meta.key_id,
                ),
            )

    async def delete_key(self, key_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM api_keys WHERE key_id = ?;", (key_id,))
            return cursor.rowcount > 0
