"""Typed workflow-domain models used by the LangGraph orchestration layer.

The workflow domain is deliberately separate from RuntimeManager's task state
machine. RuntimeManager remains authoritative for DEIMOS task lifecycle; these
models describe durable multi-step orchestration state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from ..types import Action


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, field: str, *, allow_verdict_string: bool = False) -> dict[str, Any]:
    """Normalize a workflow payload field without blindly calling ``dict(value)``."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if allow_verdict_string and isinstance(value, str):
        return {"verdict": value}
    raise ValueError(
        f"workflow step {field} must be a mapping"
        + (" or verdict string" if allow_verdict_string else "")
        + f", got {type(value).__name__}"
    )


# Boundary mappings from the existing DEIMOS execution lifecycle.  BLOCKED is
# intentionally *not* a WorkflowStep state: in the runner it means a dependent
# step was never attempted, so the durable workflow representation keeps that
# step PENDING while the workflow itself remains FAILED/RECOVERY_REQUIRED.
_RUNNER_STEP_STATUS_MAP = {
    "PENDING": "PENDING",
    "WAITING_FOR_APPROVAL": "WAITING_FOR_APPROVAL",
    "WAITING_FOR_USER": "WAITING_FOR_USER",
    "RUNNING": "RUNNING",
    "VERIFYING": "VERIFYING",
    "COMPLETED": "COMPLETED",
    "FAILED": "FAILED",
    "RECOVERY_REQUIRED": "RECOVERY_REQUIRED",
    "CANCELLED": "CANCELLED",
    "UNKNOWN": "RECOVERY_REQUIRED",
    "BLOCKED": "PENDING",
}

_RUNTIME_STEP_STATUS_MAP = {
    "RUNNING": "RUNNING",
    "WAITING_FOR_APPROVAL": "WAITING_FOR_APPROVAL",
    "WAITING_FOR_USER": "WAITING_FOR_USER",
    "VERIFYING": "VERIFYING",
    "COMPLETED": "COMPLETED",
    "FAILED": "FAILED",
    "CANCELLED": "CANCELLED",
    "BLOCKED": "RECOVERY_REQUIRED",
    "RECOVERY_REQUIRED": "RECOVERY_REQUIRED",
}


def workflow_step_status_from_runner_state(value: str) -> "StepStatus":
    raw = str(value).upper()
    mapped = _RUNNER_STEP_STATUS_MAP.get(raw)
    if mapped is None:
        raise ValueError(f"unmapped DEIMOS runner step state {value!r}")
    return StepStatus(mapped)


def workflow_step_status_from_runtime_state(value: str) -> "StepStatus":
    raw = str(value).upper()
    mapped = _RUNTIME_STEP_STATUS_MAP.get(raw)
    if mapped is None:
        # Fail closed: an unknown lifecycle state can never become completion.
        return StepStatus.RECOVERY_REQUIRED
    return StepStatus(mapped)


class StepStatus(str, Enum):
    PENDING = "PENDING"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    RUNNING = "RUNNING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    CANCELLED = "CANCELLED"


class WorkflowStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    RECOVERING = "RECOVERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class ResourceBinding:
    resource_key: str
    site: str | None = None
    reuse: bool = True

    def to_json(self) -> dict[str, Any]:
        return {"resource_key": self.resource_key, "site": self.site, "reuse": self.reuse}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ResourceBinding":
        return cls(str(data["resource_key"]), data.get("site"), bool(data.get("reuse", True)))


@dataclass
class WorkflowStep:
    step_id: str
    action: Action
    status: StepStatus = StepStatus.PENDING
    attempt_count: int = 0
    resource_key: str | None = None
    dependencies: list[str] = field(default_factory=list)
    side_effect: bool = True
    requires_verification: bool = True
    observation: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    failure_category: str | None = None
    recovery: dict[str, Any] = field(default_factory=dict)
    # Kept for compatibility with the P2.3 workflow projection.
    index: int = 0

    @property
    def state(self) -> str:
        return self.status.value

    @state.setter
    def state(self, value: str) -> None:
        # Compatibility boundary for the existing runner.  Never expose its
        # BLOCKED/UNKNOWN vocabulary as fake workflow enum members.
        self.status = workflow_step_status_from_runner_state(value)

    def to_json(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "index": self.index,
            "capability": self.action.kind,
            "arguments": dict(self.action.params),
            "action": self.action.to_json(),
            "state": self.status.value,
            "status": self.status.value,
            "attempt_count": self.attempt_count,
            "resource_key": self.resource_key,
            "dependencies": list(self.dependencies),
            "side_effect": self.side_effect,
            "requires_verification": self.requires_verification,
            "observation": dict(self.observation),
            "result": _mapping(self.result, "result"),
            "verification": _mapping(self.verification, "verification", allow_verdict_string=True),
            "error": self.error,
            "failure_category": self.failure_category,
            "recovery": dict(self.recovery),
            "result_summary": str(self.result.get("summary", "")),
            "failure_reason": self.error or "",
        }

    @classmethod
    def from_json(cls, data: dict[str, Any], index: int = 0) -> "WorkflowStep":
        action_data = data.get("action") or {}
        kind = action_data.get("kind", data.get("capability", ""))
        params = action_data.get("params", data.get("arguments", {}))
        action = Action(kind=str(kind), params=dict(params or {}),
                        consequential=bool(action_data.get("consequential", data.get("side_effect", True))),
                        rationale=str(action_data.get("rationale", "")))
        raw_status = data.get("status", data.get("state", "PENDING"))
        status = workflow_step_status_from_runner_state(str(raw_status))
        result = _mapping(data.get("result"), "result")
        if data.get("result_summary") and "summary" not in result:
            result["summary"] = data["result_summary"]
        return cls(
            step_id=str(data.get("step_id", f"step-{index + 1}")),
            action=action,
            status=status,
            attempt_count=int(data.get("attempt_count", 0)),
            resource_key=data.get("resource_key"),
            dependencies=[str(x) for x in data.get("dependencies", [])],
            side_effect=bool(data.get("side_effect", action.consequential)),
            requires_verification=bool(data.get("requires_verification", True)),
            observation=dict(data.get("observation") or {}),
            result=result,
            verification=_mapping(data.get("verification"), "verification", allow_verdict_string=True),
            error=data.get("error") or data.get("failure_reason") or None,
            failure_category=data.get("failure_category"),
            recovery=dict(data.get("recovery") or {}),
            index=int(data.get("index", index)),
        )


@dataclass
class WorkflowContext:
    values: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return dict(self.values)


@dataclass
class Workflow:
    workflow_id: str
    goal: str
    steps: list[WorkflowStep] = field(default_factory=list)
    current_step_id: str | None = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    context: WorkflowContext = field(default_factory=WorkflowContext)
    resources: dict[str, ResourceBinding] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    @property
    def current_step(self) -> int:
        if self.current_step_id is None:
            return 0
        for step in self.steps:
            if step.step_id == self.current_step_id:
                return step.index
        return 0

    @current_step.setter
    def current_step(self, value: int) -> None:
        if self.steps and 0 <= value < len(self.steps):
            self.current_step_id = self.steps[value].step_id
        elif not self.steps:
            self.current_step_id = None

    @property
    def state(self) -> str:
        return self.status.value

    @state.setter
    def state(self, value: str) -> None:
        self.status = WorkflowStatus(value)

    def to_json(self) -> dict[str, Any]:
        self.updated_at = _now()
        return {
            "workflow_id": self.workflow_id,
            "goal": self.goal,
            "steps": [step.to_json() for step in self.steps],
            "current_step_id": self.current_step_id,
            "current_step": self.current_step,
            "status": self.status.value,
            "state": self.status.value,
            "context": self.context.to_json(),
            "resources": {k: v.to_json() for k, v in self.resources.items()},
            "history": list(self.history),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "Workflow":
        if not isinstance(payload, dict):
            raise ValueError("workflow payload must be an object")
        raw_status = payload.get("status", payload.get("state", "PENDING"))
        try:
            status = WorkflowStatus(str(raw_status))
        except ValueError:
            status = WorkflowStatus.PENDING
        steps = [WorkflowStep.from_json(item, i) for i, item in enumerate(payload.get("steps", []))]
        current_step_id = payload.get("current_step_id")
        if current_step_id is None and steps:
            index = int(payload.get("current_step", 0))
            if 0 <= index < len(steps):
                current_step_id = steps[index].step_id
        resources = {
            str(k): ResourceBinding.from_json(v)
            for k, v in (payload.get("resources") or {}).items()
            if isinstance(v, dict) and v.get("resource_key")
        }
        # P2.3 workflow payloads did not have explicit resources/context.
        if not resources:
            for step in steps:
                if step.resource_key and step.resource_key not in resources:
                    resources[step.resource_key] = ResourceBinding(step.resource_key, site=step.resource_key)
        return cls(
            workflow_id=str(payload.get("workflow_id", "")),
            goal=str(payload.get("goal", "")),
            steps=steps,
            current_step_id=current_step_id,
            status=status,
            context=WorkflowContext(dict(payload.get("context") or {})),
            resources=resources,
            history=list(payload.get("history") or []),
            created_at=str(payload.get("created_at") or _now()),
            updated_at=str(payload.get("updated_at") or _now()),
        )

    @classmethod
    def from_actions(cls, workflow_id: str, goal: str, actions: list[Action]) -> "Workflow":
        steps: list[WorkflowStep] = []
        resources: dict[str, ResourceBinding] = {}
        for i, action in enumerate(actions):
            resource_key = resource_key_for_action(action)
            step = WorkflowStep(
                step_id=f"{workflow_id}:step-{i + 1}",
                index=i,
                action=action,
                resource_key=resource_key,
                side_effect=action.consequential,
                requires_verification=True,
                dependencies=[steps[-1].step_id] if steps else [],
            )
            steps.append(step)
            if resource_key:
                resources.setdefault(resource_key, ResourceBinding(resource_key, site=resource_key, reuse=True))
        wf = cls(workflow_id=workflow_id, goal=goal, steps=steps, resources=resources)
        if steps:
            wf.current_step_id = steps[0].step_id
        return wf


def resource_key_for_action(action: Action) -> str | None:
    if action.kind.startswith("whatsapp_"):
        return "whatsapp"
    if action.kind.startswith("gmail_"):
        return "gmail"
    if action.kind == "browser_play_song":
        return "youtube"
    if action.kind.startswith("browser_") or action.kind == "open_url":
        return "browser"
    return None
