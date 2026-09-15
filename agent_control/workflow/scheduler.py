"""Dependency-aware workflow scheduling for P2.5.

The scheduler owns *eligibility*, not capability execution.  A node is sent to
an injected executor only after dependency, policy, data, concurrency and
resource checks pass.  Browser/skill implementations remain unchanged.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ..planner.base import ALLOWED_ACTION_KINDS
from .models import DependencyType, StepStatus, Workflow, WorkflowStatus, WorkflowStep


class WorkflowGraphError(ValueError):
    """A workflow graph cannot be safely scheduled."""


@dataclass(frozen=True)
class ScheduleResult:
    results: dict[str, dict[str, Any]]
    status: WorkflowStatus


def _max_concurrency() -> int:
    raw = os.getenv("DEIMOS_WORKFLOW_MAX_CONCURRENCY", "2")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 2
    return max(1, min(value, 32))


class ResourceLockManager:
    """Bounded exclusive locks; failed/exceptional work always releases them."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._owners: dict[str, str] = {}

    def _lock(self, resource: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(resource, threading.Lock())

    def try_acquire(self, resource: str | None, node_id: str) -> bool:
        if not resource:
            return True
        lock = self._lock(resource)
        acquired = lock.acquire(blocking=False)
        if acquired:
            with self._guard:
                self._owners[resource] = node_id
        return acquired

    def release(self, resource: str | None, node_id: str) -> None:
        if not resource:
            return
        with self._guard:
            owner = self._owners.get(resource)
            if owner != node_id:
                return
            self._owners.pop(resource, None)
            lock = self._locks.get(resource)
        if lock is not None:
            lock.release()

    def owner(self, resource: str) -> str | None:
        with self._guard:
            return self._owners.get(resource)


class DependencyScheduler:
    """Validate and execute a workflow DAG with safe bounded concurrency."""

    def __init__(self, *, max_concurrency: int | None = None, resources: ResourceLockManager | None = None) -> None:
        self.max_concurrency = max(1, max_concurrency or _max_concurrency())
        self.resources = resources or ResourceLockManager()

    @staticmethod
    def validate(workflow: Workflow) -> None:
        nodes = workflow.steps
        ids = [node.step_id for node in nodes]
        if len(ids) != len(set(ids)):
            raise WorkflowGraphError("duplicate workflow node IDs")
        known = set(ids)
        for node in nodes:
            if not node.step_id:
                raise WorkflowGraphError("node ID is empty")
            if not node.action.kind or node.action.kind not in ALLOWED_ACTION_KINDS:
                raise WorkflowGraphError(f"invalid capability for {node.step_id}: {node.action.kind!r}")
            if not isinstance(node.action.params, dict):
                raise WorkflowGraphError(f"malformed arguments for {node.step_id}")
            if node.step_id in node.dependencies:
                raise WorkflowGraphError(f"self dependency: {node.step_id}")
            for dep in node.dependencies:
                if dep not in known:
                    raise WorkflowGraphError(f"unknown dependency {dep!r} for {node.step_id}")

        # Kahn validation catches every cycle before anything can execute.
        indegree = {node.step_id: 0 for node in nodes}
        children: dict[str, list[str]] = {node.step_id: [] for node in nodes}
        for node in nodes:
            for dep in node.dependencies:
                indegree[node.step_id] += 1
                children[dep].append(node.step_id)
        queue = [node_id for node_id, degree in indegree.items() if degree == 0]
        visited = 0
        while queue:
            current = queue.pop()
            visited += 1
            for child in children[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if visited != len(nodes):
            raise WorkflowGraphError("dependency cycle detected")

        # A persisted dependency may not point at an already-terminal impossible
        # prerequisite when the workflow is being newly scheduled.
        for node in nodes:
            for dep_id in node.dependencies:
                dep = next(dep for dep in nodes if dep.step_id == dep_id)
                if dep.status in {StepStatus.CANCELLED, StepStatus.BLOCKED}:
                    raise WorkflowGraphError(
                        f"impossible dependency: {node.step_id} depends on {dep_id} ({dep.status.value})"
                    )

    @staticmethod
    def ready_nodes(workflow: Workflow) -> list[WorkflowStep]:
        completed = {n.step_id for n in workflow.steps if n.status is StepStatus.COMPLETED}
        ready: list[WorkflowStep] = []
        for node in workflow.steps:
            if node.status not in {StepStatus.PENDING, StepStatus.WAITING_RESOURCE, StepStatus.READY}:
                continue
            if all(dep in completed for dep in node.dependencies) and node.policy_state in {"", "ALLOWED", "APPROVED"}:
                if node.requires_data and not all(key in workflow.context.values for key in node.requires_data):
                    continue
                node.status = StepStatus.READY
                ready.append(node)
            elif any(
                next(dep for dep in workflow.steps if dep.step_id == dep_id).status
                in {StepStatus.FAILED, StepStatus.UNKNOWN, StepStatus.BLOCKED, StepStatus.CANCELLED, StepStatus.RECOVERY_REQUIRED}
                for dep_id in node.dependencies
            ):
                node.status = StepStatus.BLOCKED
        return ready

    @staticmethod
    def aggregate(workflow: Workflow) -> WorkflowStatus:
        states = [node.status for node in workflow.steps]
        if not states:
            return WorkflowStatus.COMPLETED
        if all(state is StepStatus.COMPLETED for state in states):
            return WorkflowStatus.COMPLETED
        if all(state is StepStatus.CANCELLED for state in states):
            return WorkflowStatus.CANCELLED
        if any(state is StepStatus.UNKNOWN for state in states):
            return WorkflowStatus.UNKNOWN
        if any(state is StepStatus.WAITING_FOR_APPROVAL for state in states):
            return WorkflowStatus.WAITING_FOR_APPROVAL
        if any(state is StepStatus.WAITING_FOR_USER for state in states):
            return WorkflowStatus.WAITING_FOR_USER
        if any(state in {StepStatus.FAILED, StepStatus.BLOCKED, StepStatus.CANCELLED} for state in states):
            if any(state is StepStatus.COMPLETED for state in states):
                return WorkflowStatus.PARTIAL_FAILURE
            return WorkflowStatus.FAILED
        return WorkflowStatus.RUNNING

    def run(
        self,
        workflow: Workflow,
        executor: Callable[[WorkflowStep], dict[str, Any]],
        *,
        max_rounds: int = 1000,
    ) -> ScheduleResult:
        self.validate(workflow)
        results: dict[str, dict[str, Any]] = {}
        rounds = 0
        with ThreadPoolExecutor(max_workers=self.max_concurrency, thread_name_prefix="deimos-workflow") as pool:
            while rounds < max_rounds:
                rounds += 1
                ready = self.ready_nodes(workflow)
                if not ready:
                    workflow.status = self.aggregate(workflow)
                    if workflow.status in {WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.PARTIAL_FAILURE, WorkflowStatus.UNKNOWN, WorkflowStatus.CANCELLED, WorkflowStatus.WAITING_FOR_APPROVAL, WorkflowStatus.WAITING_FOR_USER}:
                        break
                    # Nothing can progress: remaining dependencies are unresolved.
                    if any(node.status in {StepStatus.PENDING, StepStatus.READY, StepStatus.WAITING_RESOURCE} for node in workflow.steps):
                        workflow.status = WorkflowStatus.BLOCKED
                    break

                futures = {}
                for node in ready:
                    if len(futures) >= self.max_concurrency:
                        node.status = StepStatus.WAITING_RESOURCE
                        continue
                    if not self.resources.try_acquire(node.resource_key, node.step_id):
                        node.status = StepStatus.WAITING_RESOURCE
                        continue
                    node.status = StepStatus.RUNNING
                    futures[pool.submit(executor, node)] = node

                if not futures:
                    # All candidates are waiting on exclusive resources.  The
                    # current batch cannot own any resource, so yield briefly and
                    # rescan rather than busy-spin.
                    continue

                for future in as_completed(futures):
                    node = futures[future]
                    try:
                        result = dict(future.result() or {})
                        results[node.step_id] = result
                        status = str(result.get("status", "COMPLETED" if result.get("ok") else "FAILED")).upper()
                        if status == "COMPLETED":
                            node.status = StepStatus.COMPLETED
                        elif status == "UNKNOWN":
                            node.status = StepStatus.UNKNOWN
                        elif status == "RECOVERING":
                            node.status = StepStatus.RECOVERING
                        elif status == "WAITING_FOR_APPROVAL":
                            node.status = StepStatus.WAITING_FOR_APPROVAL
                        elif status == "WAITING_FOR_USER":
                            node.status = StepStatus.WAITING_FOR_USER
                        elif status == "CANCELLED":
                            node.status = StepStatus.CANCELLED
                        else:
                            node.status = StepStatus.FAILED
                        node.result = dict(result.get("result") or {})
                        node.verification = dict(result.get("verification") or {})
                        node.error = result.get("error")
                        # Only verified output becomes dependency data. A failed
                        # or UNKNOWN producer can never satisfy a data edge.
                        if node.status is StepStatus.COMPLETED and str(node.verification.get("verdict", "")).upper() == "PASS":
                            data = result.get("data")
                            if isinstance(data, dict):
                                workflow.context.values.update(data)
                            output_key = result.get("output_key")
                            if output_key:
                                workflow.context.values[str(output_key)] = result.get("data", result.get("result"))
                            for dependent in workflow.steps:
                                if node.step_id in dependent.dependencies and dependent.requires_data:
                                    for key in dependent.requires_data:
                                        if key in workflow.context.values:
                                            token = "${" + key + "}"
                                            for param, value in list(dependent.action.params.items()):
                                                if isinstance(value, str) and value == token:
                                                    dependent.action.params[param] = workflow.context.values[key]
                    except Exception as exc:
                        node.status = StepStatus.FAILED
                        node.error = f"{type(exc).__name__}: {exc}"
                        results[node.step_id] = {"status": "FAILED", "error": node.error}
                    finally:
                        self.resources.release(node.resource_key, node.step_id)

                workflow.status = self.aggregate(workflow)
                if workflow.status in {WorkflowStatus.COMPLETED, WorkflowStatus.PARTIAL_FAILURE, WorkflowStatus.UNKNOWN, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}:
                    # Continue only if there are recoverable READY/PENDING branches;
                    # independent branches already in this round have been joined.
                    if not any(n.status in {StepStatus.PENDING, StepStatus.READY, StepStatus.WAITING_RESOURCE} for n in workflow.steps):
                        break

        return ScheduleResult(results=results, status=workflow.status)


def recover_failed_node(workflow: Workflow, node_id: str) -> WorkflowStep:
    """Explicit recovery transition; normal scheduling cannot revive FAILED."""
    node = next((n for n in workflow.steps if n.step_id == node_id), None)
    if node is None:
        raise WorkflowGraphError(f"unknown recovery node {node_id!r}")
    if node.status not in {StepStatus.FAILED, StepStatus.UNKNOWN, StepStatus.RECOVERY_REQUIRED}:
        raise WorkflowGraphError(f"node {node_id} is not recoverable from {node.status.value}")
    node.status = StepStatus.RECOVERING
    node.status = StepStatus.READY
    node.error = None
    node.recovery["recovered"] = True
    return node
