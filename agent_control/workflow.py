"""Small durable sequential-workflow projection for the existing runtime.

This is deliberately not a second runtime or workflow engine.  Workflow state is
serialized into the authoritative RuntimeManager task metadata.  The objects in
this module are only typed projections used at the planner/executor boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import Action, Check, Observation, Source, VerificationResult, Verdict
from .general_task import GeneralTask


WORKFLOW_STATES = frozenset({"PENDING", "RUNNING", "WAITING_FOR_APPROVAL", "WAITING_FOR_USER", "COMPLETED", "FAILED", "UNKNOWN", "BLOCKED", "RECOVERY_REQUIRED"})


class WorkflowError(RuntimeError):
    """Structured workflow lifecycle/data error; never an opaque IndexError."""

    def __init__(self, category: str, message: str, *, step_index: int | None = None, step_id: str | None = None):
        super().__init__(message)
        self.category = category
        self.step_index = step_index
        self.step_id = step_id


def require_step(workflow: "Workflow", index: int) -> "WorkflowStep":
    if not workflow.steps:
        raise WorkflowError("workflow_has_no_steps", f"workflow {workflow.workflow_id!r} has no steps")
    if index < 0 or index >= len(workflow.steps):
        raise WorkflowError(
            "invalid_current_step",
            f"workflow {workflow.workflow_id!r} has invalid current_step={index} (steps={len(workflow.steps)})",
            step_index=index,
        )
    return workflow.steps[index]


@dataclass
class WorkflowStep:
    step_id: str
    index: int
    action: Action
    state: str = "PENDING"
    result_summary: str = ""
    verification: str = "NOT_STARTED"
    failure_reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "index": self.index,
            "capability": self.action.kind,
            "arguments": dict(self.action.params),
            "state": self.state,
            "result_summary": self.result_summary,
            "verification": self.verification,
            "failure_reason": self.failure_reason,
        }


@dataclass
class Workflow:
    workflow_id: str
    goal: str
    steps: list[WorkflowStep] = field(default_factory=list)
    current_step: int = 0
    state: str = "PENDING"

    def to_json(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "goal": self.goal,
            "state": self.state,
            "current_step": self.current_step,
            "steps": [s.to_json() for s in self.steps],
        }

    @classmethod
    def from_actions(cls, workflow_id: str, goal: str, actions: list[Action]) -> "Workflow":
        return cls(
            workflow_id=workflow_id,
            goal=goal,
            steps=[WorkflowStep(f"{workflow_id}:step-{i + 1}", i, action) for i, action in enumerate(actions)],
        )

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "Workflow":
        if not isinstance(payload, dict):
            raise WorkflowError("workflow_state_invalid", "workflow payload is not an object")
        steps = []
        for pos, item in enumerate(payload.get("steps", [])):
            if not isinstance(item, dict):
                raise WorkflowError("workflow_step_invalid", f"workflow step {pos} is not an object", step_index=pos)
            try:
                index = int(item.get("index", pos))
            except (TypeError, ValueError) as exc:
                raise WorkflowError("workflow_step_invalid", f"workflow step {pos} has invalid index", step_index=pos) from exc
            if index != pos:
                raise WorkflowError("workflow_step_index_mismatch", f"workflow step {pos} has index={index}", step_index=index)
            steps.append(WorkflowStep(
                step_id=str(item.get("step_id", "")),
                index=index,
                action=Action(kind=str(item.get("capability", "")), params=dict(item.get("arguments", {}))),
                state=str(item.get("state", "PENDING")),
                result_summary=str(item.get("result_summary", "")),
                verification=str(item.get("verification", "NOT_STARTED")),
                failure_reason=str(item.get("failure_reason", "")),
            ))
        try:
            current_step = int(payload.get("current_step", 0))
        except (TypeError, ValueError) as exc:
            raise WorkflowError("invalid_current_step", "workflow current_step is not an integer") from exc
        if steps and not (0 <= current_step < len(steps)):
            raise WorkflowError("invalid_current_step", f"workflow current_step={current_step} outside 0..{len(steps)-1}", step_index=current_step)
        if not steps and current_step != 0:
            raise WorkflowError("invalid_current_step", f"empty workflow has current_step={current_step}", step_index=current_step)
        workflow_id = str(payload.get("workflow_id", ""))
        goal = str(payload.get("goal", ""))
        if not workflow_id:
            raise WorkflowError("workflow_state_invalid", "workflow id is missing")
        if not goal:
            raise WorkflowError("workflow_state_invalid", "workflow goal is missing")
        return cls(workflow_id, goal, steps, current_step, str(payload.get("state", "PENDING")))


class WorkflowStepTask(GeneralTask):
    """Existing GeneralTask contract narrowed to exactly one structured step."""

    def __init__(self, *, workflow: Workflow, step: WorkflowStep, readable_roots: tuple = ()) -> None:
        super().__init__(request=workflow.goal, readable_roots=readable_roots, task_id=workflow.workflow_id)
        self.workflow = workflow
        self.workflow_step = step
        self.goal = workflow.goal

    def reference_plan(self, policy: Any) -> list[Action]:
        return [self.workflow_step.action]

    def observe(self, policy: Any, trace: Any = None) -> dict[str, Observation]:
        return {"workflow_step": Observation(Source.BROWSER, "workflow step", self.workflow_step.action.kind)}
