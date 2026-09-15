"""SQLite durability adapter for the P2.0 RuntimeManager.

SQLite is storage only.  RuntimeManager remains the single live authority and
owns lifecycle validation.  This adapter performs short, thread-local
transactions and stores only serializable runtime metadata.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
import time
from pathlib import Path
from typing import Any

SENSITIVE_KEY = re.compile(r"(?:password|passwd|token|secret|cookie|credential|authorization|api[_-]?key|access[_-]?key|private[_-]?key)", re.I)


def safe_json(value: Any) -> Any:
    """Return JSON-safe metadata with obvious credential material redacted."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            key_s = str(key)
            if SENSITIVE_KEY.search(key_s):
                out[key_s] = "[REDACTED]"
            else:
                out[key_s] = safe_json(item)
        return out
    if isinstance(value, (list, tuple)):
        return [safe_json(x) for x in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def safe_goal(goal: str) -> str:
    """Keep the task summary useful without duplicating secrets."""
    text = str(goal or "").strip()
    # Do not retain common inline credential assignments in the durable summary.
    text = re.sub(r"(?i)(password|passwd|token|secret|api[_-]?key)\s*[:=]\s*\S+", r"\1=[REDACTED]", text)
    return text[:1000]


class RuntimePersistence:
    """Small SQLite adapter; no lifecycle decisions are made here."""

    def __init__(self, path: str | Path = ".agent_state/runtime.sqlite3") -> None:
        self.path = Path(path)
        self._memory = str(path) == ":memory:"
        self._dsn = (f"file:deimos_runtime_{uuid.uuid4().hex}?mode=memory&cache=shared" if self._memory else str(self.path))
        if not self._memory:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._schema_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "connection", None)
        if conn is None:
            conn = sqlite3.connect(self._dsn, timeout=30.0, uri=self._memory)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.connection = conn
        return conn

    def _initialize(self) -> None:
        with self._schema_lock:
            conn = self._connect()
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL DEFAULT '',
                    goal TEXT NOT NULL DEFAULT '',
                    source_input_id TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    parent_task_id TEXT,
                    current_phase TEXT NOT NULL DEFAULT '',
                    approval_state TEXT NOT NULL DEFAULT 'NONE',
                    failure_category TEXT NOT NULL DEFAULT '',
                    failure_message TEXT NOT NULL DEFAULT '',
                    verification_state TEXT NOT NULL DEFAULT 'NOT_STARTED',
                    verification_summary TEXT NOT NULL DEFAULT '',
                    browser_resource_key TEXT,
                    browser_site TEXT,
                    browser_resource_state TEXT,
                    result_summary TEXT NOT NULL DEFAULT '',
                    recovery_context TEXT NOT NULL DEFAULT '{}',
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS runtime_events (
                    event_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                );
                CREATE INDEX IF NOT EXISTS idx_runtime_events_task_time
                    ON runtime_events(task_id, timestamp, event_id);
                CREATE TABLE IF NOT EXISTS runtime_approvals (
                    task_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                );
                CREATE TABLE IF NOT EXISTS runtime_resources (
                    resource_key TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    site TEXT,
                    state TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                );
                """
            )
            conn.commit()

    @staticmethod
    def _task_values(task: Any) -> tuple[Any, ...]:
        return (
            task.task_id, task.task_type, safe_goal(task.goal), task.source_input_id,
            task.state, task.created_at, task.updated_at,
            getattr(task, "started_at", None), getattr(task, "completed_at", None),
            task.parent_task_id, task.current_phase, task.approval_state,
            getattr(task, "failure_category", "") or (task.failure_state if task.failure_state else ""),
            task.failure_state, task.verification_state,
            getattr(task, "verification_summary", "") or task.result_summary,
            task.browser_resource_key, task.browser_site, task.browser_resource_state,
            task.result_summary,
            json.dumps(safe_json(task.recovery_context), sort_keys=True),
            json.dumps(safe_json(task.metadata), sort_keys=True),
        )

    def _upsert_sql(self) -> str:
        return """
        INSERT INTO tasks (
            task_id, task_type, goal, source_input_id, state, created_at, updated_at,
            started_at, completed_at, parent_task_id, current_phase, approval_state,
            failure_category, failure_message, verification_state, verification_summary,
            browser_resource_key, browser_site, browser_resource_state, result_summary,
            recovery_context, metadata
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(task_id) DO UPDATE SET
            task_type=excluded.task_type, goal=excluded.goal,
            source_input_id=excluded.source_input_id, state=excluded.state,
            created_at=excluded.created_at, updated_at=excluded.updated_at,
            started_at=excluded.started_at, completed_at=excluded.completed_at,
            parent_task_id=excluded.parent_task_id, current_phase=excluded.current_phase,
            approval_state=excluded.approval_state,
            failure_category=excluded.failure_category,
            failure_message=excluded.failure_message,
            verification_state=excluded.verification_state,
            verification_summary=excluded.verification_summary,
            browser_resource_key=excluded.browser_resource_key,
            browser_site=excluded.browser_site,
            browser_resource_state=excluded.browser_resource_state,
            result_summary=excluded.result_summary,
            recovery_context=excluded.recovery_context, metadata=excluded.metadata
        """

    def commit(self, task: Any, event: Any | None = None, *, approval: dict[str, Any] | None = None,
               remove_approval: bool = False, resource: dict[str, Any] | None = None) -> None:
        """Atomically persist task + optional event/approval/resource."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(self._upsert_sql(), self._task_values(task))
            if event is not None:
                conn.execute(
                    "INSERT INTO runtime_events(event_id, task_id, timestamp, event_type, metadata) VALUES (?,?,?,?,?)",
                    (event.event_id, event.task_id, event.timestamp, event.event_type,
                     json.dumps(safe_json(event.metadata), sort_keys=True)),
                )
            if approval is not None:
                conn.execute(
                    "INSERT INTO runtime_approvals(task_id,payload,updated_at) VALUES(?,?,?) "
                    "ON CONFLICT(task_id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
                    (task.task_id, json.dumps(safe_json(approval), sort_keys=True), time.time()),
                )
            if remove_approval:
                conn.execute("DELETE FROM runtime_approvals WHERE task_id=?", (task.task_id,))
            if resource is not None:
                conn.execute(
                    "INSERT INTO runtime_resources(resource_key,task_id,site,state,updated_at) VALUES(?,?,?,?,?) "
                    "ON CONFLICT(resource_key) DO UPDATE SET task_id=excluded.task_id,site=excluded.site,state=excluded.state,updated_at=excluded.updated_at",
                    (resource["resource_key"], resource["task_id"], resource.get("site"), resource.get("state", "ATTACHED"), time.time()),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def delete_resource(self, task_id: str, resource_key: str, task: Any, event: Any) -> None:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(self._upsert_sql(), self._task_values(task))
            conn.execute("UPDATE runtime_resources SET state='DETACHED', updated_at=? WHERE resource_key=? AND task_id=?", (time.time(), resource_key, task_id))
            conn.execute(
                "INSERT INTO runtime_events(event_id, task_id, timestamp, event_type, metadata) VALUES (?,?,?,?,?)",
                (event.event_id, event.task_id, event.timestamp, event.event_type, json.dumps(safe_json(event.metadata), sort_keys=True)),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def load(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        conn = self._connect()
        tasks = [dict(r) for r in conn.execute("SELECT * FROM tasks ORDER BY created_at, task_id")]
        events = [dict(r) for r in conn.execute("SELECT * FROM runtime_events ORDER BY timestamp, rowid")]
        approvals = [dict(r) for r in conn.execute("SELECT * FROM runtime_approvals ORDER BY updated_at, task_id")]
        resources = [dict(r) for r in conn.execute("SELECT * FROM runtime_resources ORDER BY resource_key")]
        return tasks, events, approvals, resources

    def close(self) -> None:
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            conn.commit()
            conn.close()
            self._local.connection = None
