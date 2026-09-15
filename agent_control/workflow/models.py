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
    "READY": "READY",
    "WAITING_RESOURCE": "WAITING_RESOURCE",
    "WAITING_FOR_APPROVAL": "WAITING_FOR_APPROVAL",
    "WAITING_FOR_USER": "WAITING_FOR_USER",
    "RUNNING": "RUNNING",
    "VERIFYING": "VERIFYING",
    "COMPLETED": "COMPLETED",
    "FAILED": "FAILED",
    "RECOVERY_REQUIRED": "RECOVERY_REQUIRED",
    "UNKNOWN": "UNKNOWN",
    "RECOVERING": "RECOVERING",
    "BLOCKED": "BLOCKED",
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
    "UNKNOWN": "RECOVERY_REQUIRED",
    "RECOVERING": "RECOVERY_REQUIRED",
    "READY": "RUNNING",
    "WAITING_RESOURCE": "RUNNING",
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


class DependencyType(str, Enum):
    STATE_DEPENDENCY = "STATE_DEPENDENCY"
    DATA_DEPENDENCY = "DATA_DEPENDENCY"
    AUTH_SESSION_DEPENDENCY = "AUTH_SESSION_DEPENDENCY"
    RESOURCE_DEPENDENCY = "RESOURCE_DEPENDENCY"
    EXPLICIT_USER_ORDER = "EXPLICIT_USER_ORDER"


class StepStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    WAITING_RESOURCE = "WAITING_RESOURCE"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    RUNNING = "RUNNING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"
    BLOCKED = "BLOCKED"
    RECOVERING = "RECOVERING"


class WorkflowStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    RECOVERING = "RECOVERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    UNKNOWN = "UNKNOWN"
    BLOCKED = "BLOCKED"
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
    dependency_reasons: dict[str, str] = field(default_factory=dict)
    dependency_types: dict[str, str] = field(default_factory=dict)
    policy_state: str = ""
    requires_data: list[str] = field(default_factory=list)
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
    def node_id(self) -> str:
        return self.step_id

    @property
    def intent(self) -> str:
        return self.original_intent or self.action.kind.replace("_", " ")

    @property
    def original_intent(self) -> str:
        return self.action.rationale or self.action.kind.replace("_", " ")

    @property
    def normalized_intent(self) -> str:
        return self.action.kind

    @property
    def normalized_arguments(self) -> dict[str, Any]:
        return self.action.params

    @property
    def capability(self) -> str:
        return self.action.kind

    @property
    def args(self) -> dict[str, Any]:
        return self.action.params

    def transition_to(self, target: StepStatus) -> None:
        allowed = {
            StepStatus.PENDING: {StepStatus.READY, StepStatus.WAITING_RESOURCE, StepStatus.WAITING_FOR_APPROVAL, StepStatus.WAITING_FOR_USER, StepStatus.CANCELLED, StepStatus.BLOCKED},
            StepStatus.READY: {StepStatus.RUNNING, StepStatus.WAITING_RESOURCE, StepStatus.CANCELLED},
            StepStatus.WAITING_RESOURCE: {StepStatus.READY, StepStatus.RUNNING, StepStatus.CANCELLED},
            StepStatus.WAITING_FOR_APPROVAL: {StepStatus.READY, StepStatus.CANCELLED},
            StepStatus.WAITING_FOR_USER: {StepStatus.PENDING, StepStatus.CANCELLED},
            StepStatus.RUNNING: {StepStatus.VERIFYING, StepStatus.FAILED, StepStatus.UNKNOWN, StepStatus.CANCELLED},
            StepStatus.VERIFYING: {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.UNKNOWN, StepStatus.RECOVERY_REQUIRED},
            StepStatus.RECOVERING: {StepStatus.READY, StepStatus.CANCELLED, StepStatus.FAILED},
            StepStatus.FAILED: {StepStatus.RECOVERING, StepStatus.CANCELLED},
            StepStatus.UNKNOWN: {StepStatus.RECOVERING, StepStatus.CANCELLED},
            StepStatus.BLOCKED: {StepStatus.RECOVERING, StepStatus.CANCELLED},
            StepStatus.COMPLETED: set(),
            StepStatus.CANCELLED: set(),
            StepStatus.RECOVERY_REQUIRED: {StepStatus.RECOVERING, StepStatus.CANCELLED},
        }
        if target not in allowed.get(self.status, set()):
            raise ValueError(f"invalid workflow node transition {self.status.value} -> {target.value}")
        self.status = target

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
            "dependency_reasons": dict(self.dependency_reasons),
            "dependency_types": dict(self.dependency_types),
            "policy_state": self.policy_state,
            "requires_data": list(self.requires_data),
            "node_id": self.node_id,
            "intent": self.intent,
            "normalized_intent": self.normalized_intent,
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
            dependency_reasons={str(k): str(v) for k, v in (data.get("dependency_reasons") or {}).items()},
            dependency_types={str(k): str(v) for k, v in (data.get("dependency_types") or {}).items()},
            policy_state=str(data.get("policy_state", "")),
            requires_data=[str(x) for x in data.get("requires_data", [])],
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
            dependencies, reasons, types = infer_dependencies(goal, action, steps)
            step = WorkflowStep(
                step_id=f"{workflow_id}:step-{i + 1}",
                index=i,
                action=action,
                resource_key=resource_key,
                side_effect=action.consequential,
                requires_verification=True,
                dependencies=dependencies,
                dependency_reasons=reasons,
                dependency_types=types,
            )
            steps.append(step)
            if resource_key:
                resources.setdefault(resource_key, ResourceBinding(resource_key, site=resource_key, reuse=True))
        wf = cls(workflow_id=workflow_id, goal=goal, steps=steps, resources=resources)
        if steps:
            wf.current_step_id = steps[0].step_id
        return wf


def infer_dependencies(goal: str, action: Action, prior_steps: list[WorkflowStep]) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Infer only genuine ordering dependencies; linguistic adjacency is not enough."""
    deps: list[str] = []
    reasons: dict[str, str] = {}
    types: dict[str, str] = {}
    text = " ".join(str(goal or "").casefold().split())
    explicit_order = bool(__import__("re").search(r"\bthen\b|\bafter that\b|\bonce\b", text))
    if explicit_order and prior_steps:
        previous = prior_steps[-1]
        deps.append(previous.step_id)
        reasons[previous.step_id] = "explicit_user_order"
        types[previous.step_id] = DependencyType.EXPLICIT_USER_ORDER.value
        return deps, reasons, types

    # State dependency: an action that operates on the foreground application
    # needs a preceding launch/open action for that same application.
    if action.kind in {"browser_type", "browser_click", "browser_press_key", "open_file"} and prior_steps:
        previous = prior_steps[-1]
        if previous.action.kind == "launch_app":
            app = str(previous.action.params.get("app", "")).casefold()
            if app and (app in text or app in str(action.params).casefold()):
                deps.append(previous.step_id)
                reasons[previous.step_id] = "requires_foreground_application"
                types[previous.step_id] = DependencyType.STATE_DEPENDENCY.value

    # Data dependency: explicit planner references are represented as node IDs
    # rather than hidden context injection.
    source = action.params.get("depends_on") or action.params.get("input_from")
    if isinstance(source, str):
        for previous in prior_steps:
            if source in {previous.step_id, previous.node_id}:
                if previous.step_id not in deps:
                    deps.append(previous.step_id)
                reasons[previous.step_id] = "verified_output_required"
                types[previous.step_id] = DependencyType.DATA_DEPENDENCY.value

    # Authentication/session dependencies are explicit in capability naming;
    # never infer them merely because two actions share a service.
    if prior_steps and any(token in action.kind.casefold() for token in ("authenticated", "account", "send", "delete", "apply")):
        previous = prior_steps[-1]
        if "login" in previous.action.kind.casefold() or "authenticate" in previous.action.kind.casefold():
            if previous.step_id not in deps:
                deps.append(previous.step_id)
            reasons[previous.step_id] = "authenticated_session_required"
            types[previous.step_id] = DependencyType.AUTH_SESSION_DEPENDENCY.value
    return deps, reasons, types

# P2.5 public name: WorkflowStep remains the compatibility surface while
# exposing the dependency-aware TaskNode contract.
TaskNode = WorkflowStep

def resource_key_for_action(action: Action) -> str | None:
    if action.kind == "launch_app":
        app = str(action.params.get("app", "")).strip().casefold()
        if app:
            return app
    if action.kind.startswith("whatsapp_"):
        return "whatsapp"
    if action.kind.startswith("gmail_"):
        return "gmail"
    if action.kind == "browser_play_song":
        return "youtube"
    if action.kind.startswith("browser_") or action.kind == "open_url":
        return "browser"
    return None
