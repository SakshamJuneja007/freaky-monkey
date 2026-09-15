"""Durable multi-step workflow orchestration for DEIMOS."""
from .models import (
    ResourceBinding,
    StepStatus,
    Workflow,
    WorkflowContext,
    WorkflowStatus,
    WorkflowStep,
    resource_key_for_action,
    workflow_step_status_from_runner_state,
    workflow_step_status_from_runtime_state,
)

__all__ = [
    "ResourceBinding", "StepStatus", "Workflow", "WorkflowContext",
    "WorkflowStatus", "WorkflowStep", "resource_key_for_action",
    "workflow_step_status_from_runner_state", "workflow_step_status_from_runtime_state",
]

from ..general_task import GeneralTask
from ..types import Observation, Source


class WorkflowStepTask(GeneralTask):
    """Existing GeneralTask contract narrowed to exactly one workflow step."""
    def __init__(self, *, workflow: Workflow, step: WorkflowStep, readable_roots: tuple = ()) -> None:
        super().__init__(request=workflow.goal, readable_roots=readable_roots, task_id=workflow.workflow_id)
        self.workflow = workflow
        self.workflow_step = step
        self.goal = workflow.goal

    def reference_plan(self, policy):
        return [self.workflow_step.action]

    def observe(self, policy, trace=None):
        return {"workflow_step": Observation(Source.BROWSER, "workflow step", self.workflow_step.action.kind)}

__all__.append("WorkflowStepTask")
