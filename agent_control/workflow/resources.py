"""Workflow resource bindings; actual browser ownership stays in Session."""
from __future__ import annotations

from .models import ResourceBinding, Workflow, WorkflowStep


def bind_resources(workflow: Workflow) -> Workflow:
    """Populate resource bindings without creating or manipulating browsers."""
    for step in workflow.steps:
        if not step.resource_key:
            continue
        workflow.resources.setdefault(
            step.resource_key,
            ResourceBinding(step.resource_key, site=step.resource_key, reuse=True),
        )
    return workflow


def resource_for_step(workflow: Workflow, step: WorkflowStep) -> ResourceBinding | None:
    if not step.resource_key:
        return None
    return workflow.resources.get(step.resource_key)
