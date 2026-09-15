"""Durable local runtime state for DEIMOS Phase 2.

This module is deliberately boring: SQLite is the source of truth for durable
*task lifecycle* state; live Python objects (BrowserSkill sessions, futures,
planners, speakers) are never serialized.  The store is independent of the
execution control loop and can therefore be replaced without changing Policy,
Skills, Executors, Observation, or Verification.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

DEFAULT_RUNTIME_DB = Path(__file__).resolve().parent.parent / ".agent_state" / "runtime.db"

TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED"})
WAITING_STATES = frozenset({"WAITING_FOR_APPROVAL", "WAITING_FOR_USER", "WAITING_FOR_HUMAN"})
RECOVERY_UNSAFE_STATES = frozenset({"RUNNING", "VERIFYING", "PLANNING", "RECOVERING", "RECOVERY_REQUIRED"})

_SECRET_KEY = re.compile(r"(?:password|passwd|token|secret|cookie|authorization|api[_-]?key|access[_-]?key|refresh[_-]?token|credential)", re.I)


def _safe_json(value: Any) -> Any:
    """Redact obvious credential-bearing keys without storing live/private objects."""
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if _SECRET_KEY.search(str(k)) else _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return "[UNSERIALIZABLE]"


def _json(value: Any) -> str:
    return json.dumps(_safe_json(value), ensure_ascii=False, sort_keys=True, default=str)


@dataclass(frozen=True)
class TaskState:
    task_id: str
    goal: str
    workflow_task_id: str = ""
    parent_task_id: str | None = None
    status: str = "CREATED"
    current_step: str = ""
    approval_state: str = "NONE"
    execution_state: str = "IDLE"
    verification_state: str = "NOT_STARTED"
    failure_context: dict[str, Any] = field(default_factory=dict)
    recovery_context: dict[str, Any] = field(default_factory=dict)
    result_metadata: dict[str, Any] = field(default_factory=dict)
    cancellation_state: str = "NOT_CANCELLED"
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "workflow_task_id": self.workflow_task_id,
            "parent_task_id": self.parent_task_id,
            "status": self.status,
            "current_step": self.current_step,
            "approval_state": self.approval_state,
            "execution_state": self.execution_state,
            "verification_state": self.verification_state,
            "failure_context": dict(self.failure_context),
            "recovery_context": dict(self.recovery_context),
            "result_metadata": dict(self.result_metadata),
            "cancellation_state": self.cancellation_state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class RuntimeStore:
    """Small SQLite repository for authoritative durable runtime state."""

    def __init__(self, path: str | Path = DEFAULT_RUNTIME_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, timeout=10.0, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                workflow_task_id TEXT NOT NULL DEFAULT '',
                parent_task_id TEXT,
                status TEXT NOT NULL,
                current_step TEXT NOT NULL DEFAULT '',
                approval_state TEXT NOT NULL DEFAULT 'NONE',
                execution_state TEXT NOT NULL DEFAULT 'IDLE',
                verification_state TEXT NOT NULL DEFAULT 'NOT_STARTED',
                failure_context TEXT NOT NULL DEFAULT '{}',
                recovery_context TEXT NOT NULL DEFAULT '{}',
                result_metadata TEXT NOT NULL DEFAULT '{}',
                cancellation_state TEXT NOT NULL DEFAULT 'NOT_CANCELLED',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS task_events (
                event_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_task_events_task_time ON task_events(task_id, timestamp, event_id);
            CREATE TABLE IF NOT EXISTS pending_interactions (
                task_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
            );
            """
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "RuntimeStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def create_task(
        self,
        goal: str,
        *,
        workflow_task_id: str = "",
        parent_task_id: str | None = None,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskState:
        now = time.time()
        task_id = task_id or f"task-{uuid.uuid4().hex[:12]}"
        row = TaskState(
            task_id=task_id,
            goal=goal,
            workflow_task_id=workflow_task_id,
            parent_task_id=parent_task_id,
            status="CREATED",
            created_at=now,
            updated_at=now,
            result_metadata=metadata or {},
        )
        with self._db:
            self._db.execute(
                """INSERT INTO tasks(task_id,goal,workflow_task_id,parent_task_id,status,current_step,
                   approval_state,execution_state,verification_state,failure_context,recovery_context,
                   result_metadata,cancellation_state,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (row.task_id, row.goal, row.workflow_task_id, row.parent_task_id, row.status,
                 row.current_step, row.approval_state, row.execution_state, row.verification_state,
                 _json(row.failure_context), _json(row.recovery_context), _json(row.result_metadata),
                 row.cancellation_state, row.created_at, row.updated_at),
            )
        self.event(row.task_id, "TASK_CREATED", {"workflow_task_id": row.workflow_task_id})
        return row

    def get(self, task_id: str) -> TaskState | None:
        row = self._db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return self._row(row) if row else None

    def list_tasks(self, *, statuses: Iterable[str] | None = None) -> list[TaskState]:
        if statuses:
            values = tuple(statuses)
            marks = ",".join("?" for _ in values)
            rows = self._db.execute(f"SELECT * FROM tasks WHERE status IN ({marks}) ORDER BY created_at", values).fetchall()
        else:
            rows = self._db.execute("SELECT * FROM tasks ORDER BY created_at").fetchall()
        return [self._row(r) for r in rows]

    def transition(
        self,
        task_id: str,
        status: str,
        *,
        event_type: str | None = None,
        current_step: str | None = None,
        approval_state: str | None = None,
        execution_state: str | None = None,
        verification_state: str | None = None,
        failure_context: dict[str, Any] | None = None,
        recovery_context: dict[str, Any] | None = None,
        result_metadata: dict[str, Any] | None = None,
        cancellation_state: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskState:
        current = self.get(task_id)
        if current is None:
            raise KeyError(task_id)
        now = time.time()
        fields = {
            "status": status,
            "current_step": current.current_step if current_step is None else current_step,
            "approval_state": current.approval_state if approval_state is None else approval_state,
            "execution_state": current.execution_state if execution_state is None else execution_state,
            "verification_state": current.verification_state if verification_state is None else verification_state,
            "failure_context": current.failure_context if failure_context is None else failure_context,
            "recovery_context": current.recovery_context if recovery_context is None else recovery_context,
            "result_metadata": current.result_metadata if result_metadata is None else result_metadata,
            "cancellation_state": current.cancellation_state if cancellation_state is None else cancellation_state,
            "updated_at": now,
        }
        with self._db:
            self._db.execute(
                """UPDATE tasks SET status=?,current_step=?,approval_state=?,execution_state=?,
                   verification_state=?,failure_context=?,recovery_context=?,result_metadata=?,
                   cancellation_state=?,updated_at=? WHERE task_id=?""",
                (fields["status"], fields["current_step"], fields["approval_state"], fields["execution_state"],
                 fields["verification_state"], _json(fields["failure_context"]), _json(fields["recovery_context"]),
                 _json(fields["result_metadata"]), fields["cancellation_state"], now, task_id),
            )
        if event_type:
            self.event(task_id, event_type, metadata or {"status": status})
        return self.get(task_id)  # type: ignore[return-value]

    def save_interaction(self, task_id: str, kind: str, payload: dict[str, Any]) -> None:
        now = time.time()
        with self._db:
            self._db.execute(
                "INSERT INTO pending_interactions(task_id,kind,payload,updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(task_id) DO UPDATE SET kind=excluded.kind,payload=excluded.payload,updated_at=excluded.updated_at",
                (task_id, kind, _json(payload), now),
            )

    def load_interaction(self, task_id: str) -> tuple[str, dict[str, Any]] | None:
        row = self._db.execute("SELECT kind,payload FROM pending_interactions WHERE task_id=?", (task_id,)).fetchone()
        if not row:
            return None
        return str(row["kind"]), json.loads(row["payload"])

    def clear_interaction(self, task_id: str) -> None:
        with self._db:
            self._db.execute("DELETE FROM pending_interactions WHERE task_id=?", (task_id,))

    def event(self, task_id: str, event_type: str, metadata: dict[str, Any] | None = None) -> str:
        event_id = f"event-{uuid.uuid4().hex[:12]}"
        with self._db:
            self._db.execute(
                "INSERT INTO task_events(event_id,task_id,event_type,timestamp,metadata) VALUES(?,?,?,?,?)",
                (event_id, task_id, event_type, time.time(), _json(metadata or {})),
            )
        return event_id

    def events(self, task_id: str) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT event_id,task_id,event_type,timestamp,metadata FROM task_events WHERE task_id=? ORDER BY timestamp,event_id",
            (task_id,),
        ).fetchall()
        return [
            {"event_id": r["event_id"], "task_id": r["task_id"], "event_type": r["event_type"],
             "timestamp": r["timestamp"], "metadata": json.loads(r["metadata"])}
            for r in rows
        ]

    def recover_unfinished(self) -> list[TaskState]:
        recovered: list[TaskState] = []
        for task in self.list_tasks():
            if task.status in TERMINAL_STATES or task.status in WAITING_STATES:
                recovered.append(task)
                continue
            if task.status in RECOVERY_UNSAFE_STATES:
                recovered.append(self.transition(
                    task.task_id,
                    "RECOVERY_REQUIRED",
                    event_type="TASK_RECOVERY_REQUIRED",
                    recovery_context={**task.recovery_context, "reason": "process_restart", "previous_status": task.status, "automatic_replay": False},
                    metadata={"previous_status": task.status, "automatic_replay": False},
                ))
            else:
                recovered.append(task)
        return recovered

    def _row(self, row: sqlite3.Row) -> TaskState:
        def obj(name: str) -> dict[str, Any]:
            try:
                value = json.loads(row[name])
            except Exception:
                return {}
            return value if isinstance(value, dict) else {}
        return TaskState(
            task_id=row["task_id"], goal=row["goal"], workflow_task_id=row["workflow_task_id"],
            parent_task_id=row["parent_task_id"], status=row["status"], current_step=row["current_step"],
            approval_state=row["approval_state"], execution_state=row["execution_state"],
            verification_state=row["verification_state"], failure_context=obj("failure_context"),
            recovery_context=obj("recovery_context"), result_metadata=obj("result_metadata"),
            cancellation_state=row["cancellation_state"], created_at=row["created_at"], updated_at=row["updated_at"],
        )
