"""Authoritative runtime truth for DEIMOS P2.0/P2.1.

RuntimeManager owns live task state and lifecycle validation.  P2.1 adds a
SQLite adapter underneath it; SQLite is durable storage, never a second task
registry or state machine.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .runtime_persistence import RuntimePersistence, safe_json

STATES = frozenset({
    "CREATED", "PLANNING", "WAITING_FOR_APPROVAL", "WAITING_FOR_USER",
    "WAITING_FOR_HUMAN", "RUNNING", "VERIFYING", "COMPLETED", "FAILED",
    "CANCELLED", "BLOCKED", "RECOVERY_REQUIRED",
})
TERMINAL = frozenset({"COMPLETED", "FAILED", "CANCELLED"})
_RECOVERABLE_ON_RESTART = frozenset({"PLANNING", "RUNNING", "VERIFYING"})

_ALLOWED: dict[str, frozenset[str]] = {
    "CREATED": frozenset({"PLANNING", "RUNNING", "WAITING_FOR_APPROVAL", "WAITING_FOR_USER", "WAITING_FOR_HUMAN", "BLOCKED", "CANCELLED", "FAILED"}),
    "PLANNING": frozenset({"RUNNING", "WAITING_FOR_APPROVAL", "WAITING_FOR_USER", "WAITING_FOR_HUMAN", "BLOCKED", "FAILED", "CANCELLED", "RECOVERY_REQUIRED"}),
    "WAITING_FOR_APPROVAL": frozenset({"RUNNING", "CANCELLED", "FAILED"}),
    "WAITING_FOR_USER": frozenset({"RUNNING", "WAITING_FOR_APPROVAL", "CANCELLED", "FAILED"}),
    "WAITING_FOR_HUMAN": frozenset({"RUNNING", "CANCELLED", "FAILED"}),
    "RUNNING": frozenset({"PLANNING", "VERIFYING", "WAITING_FOR_APPROVAL", "WAITING_FOR_USER", "WAITING_FOR_HUMAN", "COMPLETED", "FAILED", "CANCELLED", "RECOVERY_REQUIRED", "BLOCKED"}),
    "VERIFYING": frozenset({"COMPLETED", "FAILED", "WAITING_FOR_USER", "WAITING_FOR_HUMAN", "RECOVERY_REQUIRED", "CANCELLED", "RUNNING"}),
    "BLOCKED": frozenset({"RUNNING", "WAITING_FOR_USER", "WAITING_FOR_HUMAN", "CANCELLED", "FAILED", "RECOVERY_REQUIRED"}),
    "RECOVERY_REQUIRED": frozenset({"RUNNING", "WAITING_FOR_USER", "WAITING_FOR_HUMAN", "CANCELLED", "FAILED"}),
    "COMPLETED": frozenset(), "FAILED": frozenset(), "CANCELLED": frozenset(),
}


@dataclass(frozen=True)
class RuntimeTask:
    task_id: str
    goal: str
    task_type: str = ""
    source_input_id: str = ""
    state: str = "CREATED"
    created_at: float = 0.0
    updated_at: float = 0.0
    started_at: float | None = None
    completed_at: float | None = None
    current_phase: str = "created"
    approval_state: str = "NONE"
    verification_state: str = "NOT_STARTED"
    verification_summary: str = ""
    failure_state: str = ""
    failure_category: str = ""
    parent_task_id: str | None = None
    browser_resource_key: str | None = None
    browser_site: str | None = None
    browser_resource_state: str | None = None
    result_summary: str = ""
    recovery_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "goal": self.goal, "task_type": self.task_type,
            "source_input_id": self.source_input_id, "state": self.state,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "started_at": self.started_at, "completed_at": self.completed_at,
            "current_phase": self.current_phase, "approval_state": self.approval_state,
            "verification_state": self.verification_state,
            "verification_summary": self.verification_summary,
            "failure_state": self.failure_state, "failure_category": self.failure_category,
            "parent_task_id": self.parent_task_id,
            "browser_resource_key": self.browser_resource_key,
            "browser_site": self.browser_site,
            "browser_resource_state": self.browser_resource_state,
            "result_summary": self.result_summary,
            "recovery_context": dict(self.recovery_context),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RuntimeEvent:
    event_id: str
    task_id: str
    event_type: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return {"event_id": self.event_id, "task_id": self.task_id,
                "event_type": self.event_type, "timestamp": self.timestamp,
                "metadata": dict(self.metadata)}


class RuntimeManager:
    """The one authoritative live registry, backed by optional SQLite durability."""

    def __init__(self, persistence: RuntimePersistence | None = None,
                 *, persistence_path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        self._tasks: dict[str, RuntimeTask] = {}
        self._events: list[RuntimeEvent] = []
        self._approvals: dict[str, dict[str, Any]] = {}
        self._resources: dict[str, dict[str, Any]] = {}
        self._focused_task_id: str | None = None
        self._persistence = persistence if persistence is not None else RuntimePersistence(persistence_path or ":memory:")
        self._load()

    @property
    def persistence(self) -> RuntimePersistence:
        return self._persistence

    def _load(self) -> None:
        tasks, events, approvals, resources = self._persistence.load()
        with self._lock:
            for row in tasks:
                try:
                    task = RuntimeTask(
                        task_id=row["task_id"], goal=row["goal"], task_type=row["task_type"],
                        source_input_id=row.get("source_input_id", ""), state=row["state"],
                        created_at=row["created_at"], updated_at=row["updated_at"],
                        started_at=row.get("started_at"), completed_at=row.get("completed_at"),
                        current_phase=row.get("current_phase", "created"),
                        approval_state=row.get("approval_state", "NONE"),
                        verification_state=row.get("verification_state", "NOT_STARTED"),
                        verification_summary=row.get("verification_summary", ""),
                        failure_state=row.get("failure_message", ""),
                        failure_category=row.get("failure_category", ""),
                        parent_task_id=row.get("parent_task_id"),
                        browser_resource_key=row.get("browser_resource_key"),
                        browser_site=row.get("browser_site"),
                        browser_resource_state=row.get("browser_resource_state"),
                        result_summary=row.get("result_summary", ""),
                        recovery_context=json.loads(row.get("recovery_context") or "{}"),
                        metadata=json.loads(row.get("metadata") or "{}"),
                    )
                except Exception:
                    continue
                self._tasks[task.task_id] = task
            for row in events:
                try:
                    self._events.append(RuntimeEvent(
                        event_id=row["event_id"], task_id=row["task_id"],
                        event_type=row["event_type"], timestamp=row["timestamp"],
                        metadata=json.loads(row.get("metadata") or "{}"),
                    ))
                except Exception:
                    continue
            for row in approvals:
                try:
                    self._approvals[row["task_id"]] = json.loads(row["payload"] or "{}")
                except Exception:
                    pass
            for row in resources:
                self._resources[row["resource_key"]] = {
                    "task_id": row["task_id"], "resource_key": row["resource_key"],
                    "site": row.get("site"), "state": row.get("state", "ATTACHED"),
                }
            if self._tasks:
                self._focused_task_id = next(iter(self._tasks))

        self._rehydrate_approval_states()
        self._recover_after_restart()

    def _rehydrate_approval_states(self) -> None:
        """Reconcile persisted approval ownership with the task lifecycle.

        Older P2.0/P2.1 snapshots could contain the approval row while the task
        row was still ``WAITING_FOR_USER``.  The approval row is the durable
        ownership evidence, so on startup the live RuntimeManager repairs that
        task to ``WAITING_FOR_APPROVAL`` in one transaction.  This is a repair of
        one authoritative state, not a second approval registry.
        """
        for task_id, approval in list(self._approvals.items()):
            task = self._tasks.get(task_id)
            if task is None:
                continue
            if task.state == "WAITING_FOR_APPROVAL" and task.approval_state == "WAITING":
                continue
            if task.state in TERMINAL or task.state == "RECOVERY_REQUIRED":
                # A terminal/recovery task must never acquire a new executable
                # owner merely because stale approval metadata exists.
                continue
            now = time.time()
            updated = replace(
                task, state="WAITING_FOR_APPROVAL", approval_state="WAITING",
                current_phase="approval", updated_at=now, completed_at=None,
            )
            event = self._make_event(task_id, "APPROVAL_REHYDRATED", {
                "previous_state": task.state,
                "reason": "persisted_approval_owner",
            })
            try:
                self._persistence.commit(updated, event, approval=approval)
            except Exception:
                # Never expose a repaired in-memory state that failed to become
                # durable. The persisted row remains the source for the next
                # startup attempt.
                continue
            self._tasks[task_id] = updated
            self._events.append(event)

    def _recover_after_restart(self) -> None:
        # Only states that imply in-flight execution are converted. Waiting
        # approval/user/human tasks remain waiting and therefore retain ownership.
        for task in list(self._tasks.values()):
            if task.state not in _RECOVERABLE_ON_RESTART:
                continue
            recovery = dict(task.recovery_context)
            recovery.update({"previous_state": task.state, "reason": "process_restart", "recovery_required": True})
            updated = replace(task, state="RECOVERY_REQUIRED", updated_at=time.time(),
                              current_phase="recovery", recovery_context=recovery,
                              verification_state="UNKNOWN",
                              browser_resource_state=("STALE" if task.browser_resource_key else task.browser_resource_state))
            event = self._make_event(task.task_id, "RECOVERY_REQUIRED", {"previous_state": task.state, "reason": "process_restart"})
            try:
                self._persistence.commit(updated, event)
            except Exception:
                # Never claim a durable recovery that failed to persist.
                continue
            self._tasks[task.task_id] = updated
            self._events.append(event)
        for key, resource in list(self._resources.items()):
            if resource.get("state") in {"ATTACHED", "RUNNING"}:
                resource["state"] = "STALE"
                task_id = resource.get("task_id")
                if task_id in self._tasks:
                    task = self._tasks[task_id]
                    updated = replace(task, browser_resource_state="STALE", updated_at=time.time())
                    try:
                        self._persistence.commit(updated)
                        self._tasks[task_id] = updated
                    except Exception:
                        pass

    def create_task(self, goal: str, *, task_id: str | None = None, task_type: str = "",
                    source_input_id: str = "", parent_task_id: str | None = None,
                    metadata: dict[str, Any] | None = None) -> RuntimeTask:
        with self._lock:
            task_id = task_id or f"task-{uuid.uuid4().hex[:12]}"
            if task_id in self._tasks:
                return self._tasks[task_id]
            now = time.time()
            task = RuntimeTask(task_id=task_id, goal=goal, task_type=task_type,
                               source_input_id=source_input_id, parent_task_id=parent_task_id,
                               created_at=now, updated_at=now, metadata=safe_json(metadata or {}))
            event = self._make_event(task_id, "TASK_CREATED")
            self._persistence.commit(task, event)
            self._tasks[task_id] = task
            self._events.append(event)
            if self._focused_task_id is None:
                self._focused_task_id = task_id
            return task

    def set_workflow(self, task_id: str, workflow: dict[str, Any]) -> RuntimeTask:
        """Persist the workflow projection on the authoritative runtime task."""
        return self.update_task(task_id, metadata={**self._require(task_id).metadata, "workflow": safe_json(workflow)})

    def update_workflow(self, task_id: str, workflow: dict[str, Any], *, event_type: str = "WORKFLOW_UPDATED") -> RuntimeTask:
        """Persist workflow state and journal the change atomically through RuntimeManager."""
        with self._lock:
            current = self._require(task_id)
            metadata = {**current.metadata, "workflow": safe_json(workflow)}
            updated = replace(current, metadata=metadata, updated_at=time.time())
            event = self._make_event(task_id, event_type, {"workflow": safe_json(workflow)})
            self._persistence.commit(updated, event)
            self._tasks[task_id] = updated
            self._events.append(event)
            return updated

    def get_task(self, task_id: str) -> RuntimeTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def set_pending_input(self, task_id: str, pending_input: dict[str, Any]) -> RuntimeTask:
        """Durably register the single input owner under the authoritative task."""
        with self._lock:
            current = self._require(task_id)
            metadata = {**current.metadata, "pending_input": safe_json(pending_input)}
            updated = replace(current, metadata=metadata, updated_at=time.time())
            event = self._make_event(task_id, "INPUT_REQUESTED", {"pending_input": safe_json(pending_input)})
            self._persistence.commit(updated, event)
            self._tasks[task_id] = updated
            self._events.append(event)
            return updated

    def clear_pending_input(self, task_id: str, *, event_type: str = "INPUT_RESOLVED") -> RuntimeTask:
        """Atomically consume the durable input owner without changing task identity."""
        with self._lock:
            current = self._require(task_id)
            metadata = dict(current.metadata)
            metadata.pop("pending_input", None)
            updated = replace(current, metadata=metadata, updated_at=time.time())
            event = self._make_event(task_id, event_type)
            self._persistence.commit(updated, event)
            self._tasks[task_id] = updated
            self._events.append(event)
            return updated

    def update_task(self, task_id: str, **changes: Any) -> RuntimeTask:
        with self._lock:
            current = self._require(task_id)
            allowed = {f.name for f in RuntimeTask.__dataclass_fields__.values()}
            changes = {k: v for k, v in changes.items() if k in allowed and k not in {"task_id", "created_at"}}
            updated = replace(current, **changes, updated_at=time.time())
            self._persistence.commit(updated)
            self._tasks[task_id] = updated
            return updated

    def transition_task(self, task_id: str, state: str, *, event_type: str | None = None, **changes: Any) -> RuntimeTask:
        if state not in STATES:
            raise ValueError(f"unknown runtime state: {state}")
        with self._lock:
            current = self._require(task_id)
            if current.state != state and state not in _ALLOWED.get(current.state, frozenset()):
                raise ValueError(f"invalid task transition {current.state} -> {state} for {task_id}")
            now = time.time()
            if state == "RUNNING" and current.started_at is None:
                changes.setdefault("started_at", now)
            if state in TERMINAL:
                changes.setdefault("completed_at", now)
            if state == "VERIFYING":
                changes.setdefault("current_phase", "verification")
            changes["state"] = state
            updated = replace(current, **{k: v for k, v in changes.items() if k in RuntimeTask.__dataclass_fields__ and k not in {"task_id", "created_at"}}, updated_at=now)
            event = self._make_event(task_id, event_type or ("TASK_STATE_CHANGED" if current.state != state else "TASK_STATE_UPDATED"),
                                     {"from": current.state, "to": state} if current.state != state else None)
            self._persistence.commit(updated, event)
            self._tasks[task_id] = updated
            self._events.append(event)
            return updated

    def cancel_task(self, task_id: str, *, reason: str = "") -> RuntimeTask:
        return self.transition_task(task_id, "CANCELLED", event_type="TASK_CANCELLED", failure_state=reason, current_phase="cancelled")

    def set_focus(self, task_id: str | None) -> None:
        with self._lock:
            if task_id is not None:
                self._require(task_id)
            self._focused_task_id = task_id

    def attach_resource(self, task_id: str, resource_key: str, *, site: str | None = None, state: str = "ATTACHED") -> RuntimeTask:
        with self._lock:
            self._require(task_id)
            resource = {"task_id": task_id, "resource_key": resource_key, "site": site, "state": state}
            task = replace(self._require(task_id), browser_resource_key=resource_key,
                           browser_site=site, browser_resource_state=state, updated_at=time.time())
            event = self._make_event(task_id, "RESOURCE_ATTACHED", {"resource_key": resource_key, "site": site})
            self._persistence.commit(task, event, resource=resource)
            self._tasks[task_id] = task
            self._resources[resource_key] = resource
            self._events.append(event)
            return task

    def detach_resource(self, task_id: str, resource_key: str) -> None:
        with self._lock:
            resource = self._resources.get(resource_key)
            if resource and resource.get("task_id") == task_id:
                resource = dict(resource); resource["state"] = "DETACHED"
                task = replace(self._require(task_id), browser_resource_state="DETACHED", updated_at=time.time())
                event = self._make_event(task_id, "RESOURCE_DETACHED", {"resource_key": resource_key})
                self._persistence.delete_resource(task_id, resource_key, task, event)
                self._resources[resource_key] = resource
                self._tasks[task_id] = task
                self._events.append(event)

    def resource_for_task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._require(task_id)
            if not task.browser_resource_key:
                return None
            return dict(self._resources.get(task.browser_resource_key, {
                "task_id": task_id, "resource_key": task.browser_resource_key,
                "site": task.browser_site, "state": task.browser_resource_state,
            }))

    def request_approval(self, task_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            current = self._require(task_id)
            # The executor publishes WAITING_FOR_USER before returning a
            # NeedUserInput outcome.  For an approval, that intermediate
            # executor state is immediately replaced by the durable approval
            # owner below.  Keep this promotion scoped to request_approval so
            # generic lifecycle transitions can never manufacture an approval.
            approval_transition_allowed = (
                current.state == "WAITING_FOR_APPROVAL"
                or "WAITING_FOR_APPROVAL" in _ALLOWED.get(current.state, frozenset())
                or current.state == "WAITING_FOR_USER"
            )
            if not approval_transition_allowed:
                raise ValueError(f"invalid approval transition {current.state} -> WAITING_FOR_APPROVAL for {task_id}")
            approval = safe_json(payload)
            task = replace(current, state="WAITING_FOR_APPROVAL",
                           approval_state="WAITING", current_phase="approval", updated_at=time.time())
            event = self._make_event(task_id, "APPROVAL_REQUESTED")
            self._persistence.commit(task, event, approval=approval)
            self._tasks[task_id] = task; self._approvals[task_id] = approval; self._events.append(event)

    def get_approval(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._approvals.get(task_id)
            return dict(value) if value is not None else None

    def resolve_approval(self, task_id: str, approved: bool, *, approval_request_id: str | None = None) -> None:
        with self._lock:
            current = self._require(task_id)
            existing = self._approvals.get(task_id)
            if approval_request_id is not None:
                if existing is None:
                    raise ValueError(f"approval request {approval_request_id} is no longer pending for {task_id}")
                stored_id = existing.get("approval_request_id")
                if stored_id != approval_request_id:
                    raise ValueError(
                        f"approval request mismatch for {task_id}: expected {stored_id!r}, got {approval_request_id!r}"
                    )
            self._approvals.pop(task_id, None)
            state = "RUNNING" if approved else "CANCELLED"
            if state not in _ALLOWED.get(current.state, frozenset()) and current.state != state:
                raise ValueError(f"invalid approval resolution {current.state} -> {state} for {task_id}")
            now = time.time()
            changes = {"state": state, "approval_state": "GRANTED" if approved else "DENIED",
                       "current_phase": "execution" if approved else "cancelled", "updated_at": now}
            if approved:
                changes["started_at"] = current.started_at or now
            else:
                changes["completed_at"] = now
                changes["failure_state"] = "approval denied"
            task = replace(current, **changes)
            event = self._make_event(task_id, "APPROVAL_GRANTED" if approved else "APPROVAL_DENIED")
            self._persistence.commit(task, event, remove_approval=True)
            self._tasks[task_id] = task; self._events.append(event)

    def event(self, task_id: str, event_type: str, metadata: dict[str, Any] | None = None) -> RuntimeEvent:
        with self._lock:
            self._require(task_id)
            event = self._make_event(task_id, event_type, metadata)
            self._persistence.commit(self._require(task_id), event)
            self._events.append(event)
            return event

    def events(self, task_id: str | None = None) -> list[RuntimeEvent]:
        with self._lock:
            return list(self._events if task_id is None else [e for e in self._events if e.task_id == task_id])

    def list_active_tasks(self) -> list[RuntimeTask]:
        with self._lock:
            return [t for t in self._tasks.values() if t.state not in TERMINAL]

    def list_waiting_tasks(self) -> list[RuntimeTask]:
        with self._lock:
            return [t for t in self._tasks.values() if t.state.startswith("WAITING_")]

    def list_running_tasks(self) -> list[RuntimeTask]:
        with self._lock:
            return [t for t in self._tasks.values() if t.state in {"RUNNING", "PLANNING", "VERIFYING"}]

    def list_completed_tasks(self) -> list[RuntimeTask]:
        with self._lock:
            return [t for t in self._tasks.values() if t.state == "COMPLETED"]

    def list_failed_tasks(self) -> list[RuntimeTask]:
        with self._lock:
            return [t for t in self._tasks.values() if t.state == "FAILED"]

    def list_tasks(self) -> list[RuntimeTask]:
        with self._lock:
            return list(self._tasks.values())

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            tasks = [t.public() for t in self._tasks.values()]
            return {
                "focused_task_id": self._focused_task_id,
                "active_tasks": [t for t in tasks if t["state"] not in TERMINAL],
                "waiting_tasks": [t for t in tasks if t["state"].startswith("WAITING_")],
                "running_tasks": [t for t in tasks if t["state"] in {"RUNNING", "PLANNING", "VERIFYING"}],
                "recovery_tasks": [t for t in tasks if t["state"] == "RECOVERY_REQUIRED"],
                "completed_tasks": [t for t in tasks if t["state"] == "COMPLETED"],
                "failed_tasks": [t for t in tasks if t["state"] == "FAILED"],
                "cancelled_tasks": [t for t in tasks if t["state"] == "CANCELLED"],
                "pending_approvals": [{"task_id": k, "payload": dict(v)} for k, v in self._approvals.items()],
                "resources": [dict(v) for v in self._resources.values()],
            }

    def close(self) -> None:
        self._persistence.close()

    def _require(self, task_id: str) -> RuntimeTask:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(task_id)
        return task

    @staticmethod
    def _make_event(task_id: str, event_type: str, metadata: dict[str, Any] | None = None) -> RuntimeEvent:
        return RuntimeEvent(event_id=f"event-{uuid.uuid4().hex[:12]}", task_id=task_id,
                            event_type=event_type, timestamp=time.time(), metadata=safe_json(metadata or {}))
