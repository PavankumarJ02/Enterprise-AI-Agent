"""Persistent SQLite repository for experiment runs and benchmark telemetry."""

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from enterprise_agent.core.logging import get_logger
from enterprise_agent.schemas.experiments import ExperimentRun

logger = get_logger(__name__)


class ExperimentRunRepository:
    """SQLite-backed storage repository for experiment runs and evaluation metrics."""

    def __init__(self, db_path: str = "data/experiments.db") -> None:
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
        """Provide a managed SQLite connection with dict-like row access."""
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
        """Initialize database tables and indexes."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    run_name TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_latency_ms REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_experiment_name
                ON experiment_runs(experiment_name)
                """
            )
            conn.commit()

    def create_run(self, run: ExperimentRun) -> ExperimentRun:
        """Persist a new experiment run record."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO experiment_runs (
                    id, experiment_name, run_name, parameters,
                    metrics, tags, status, total_latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.experiment_name,
                    run.run_name,
                    json.dumps(run.parameters),
                    json.dumps(run.metrics),
                    json.dumps(run.tags),
                    run.status,
                    run.total_latency_ms,
                    run.created_at,
                ),
            )
            conn.commit()
        return run

    def update_run(
        self,
        run_id: str,
        metrics: dict[str, float] | None = None,
        status: str | None = None,
        total_latency_ms: float | None = None,
    ) -> ExperimentRun | None:
        """Update metrics, status, or latency for an existing run."""
        existing = self.get_run(run_id)
        if not existing:
            return None

        updated_metrics = metrics if metrics is not None else existing.metrics
        updated_status = status if status is not None else existing.status
        updated_latency = (
            total_latency_ms if total_latency_ms is not None else existing.total_latency_ms
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE experiment_runs
                SET metrics = ?, status = ?, total_latency_ms = ?
                WHERE id = ?
                """,
                (
                    json.dumps(updated_metrics),
                    updated_status,
                    updated_latency,
                    run_id,
                ),
            )
            conn.commit()

        return self.get_run(run_id)

    def get_run(self, run_id: str) -> ExperimentRun | None:
        """Fetch an experiment run by its unique ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM experiment_runs WHERE id = ?", (run_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_run(row)

    def list_runs(self, experiment_name: str | None = None, limit: int = 50) -> list[ExperimentRun]:
        """List latest runs, optionally filtered by parent experiment name."""
        with self._get_connection() as conn:
            if experiment_name:
                cursor = conn.execute(
                    """
                    SELECT * FROM experiment_runs
                    WHERE experiment_name = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (experiment_name, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM experiment_runs
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            rows = cursor.fetchall()
            return [self._row_to_run(r) for r in rows]

    def delete_run(self, run_id: str) -> bool:
        """Delete an experiment run by ID. Returns True if record existed."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM experiment_runs WHERE id = ?", (run_id,))
            conn.commit()
            return cursor.rowcount > 0

    @staticmethod
    def _row_to_run(row: Any) -> ExperimentRun:
        """Convert an SQLite Row to an ExperimentRun schema model."""
        return ExperimentRun(
            run_id=row["id"],
            experiment_name=row["experiment_name"],
            run_name=row["run_name"],
            parameters=json.loads(row["parameters"]),
            metrics=json.loads(row["metrics"]),
            tags=json.loads(row["tags"]),
            status=row["status"],
            total_latency_ms=row["total_latency_ms"],
            created_at=row["created_at"],
        )
