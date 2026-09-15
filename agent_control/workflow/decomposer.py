"""Workflow decomposition adapter over the existing DEIMOS planner."""
from __future__ import annotations

from typing import Any

from ..planner.base import Planner
from .models import Workflow


class WorkflowDecomposer:
    """Ask the existing planner for an ordered action plan; never execute it."""

    def __init__(self, planner: Planner) -> None:
        self.planner = planner

    def decompose(self, workflow_id: str, goal: str, *, state: dict[str, Any] | None = None) -> Workflow:
        planned = self.planner.plan(goal, state or {}, [])
        if planned.error:
            raise RuntimeError(planned.error)
        if not planned.actions:
            raise RuntimeError("planner produced no executable actions and did not claim completion")
        return Workflow.from_actions(workflow_id, goal, list(planned.actions))
