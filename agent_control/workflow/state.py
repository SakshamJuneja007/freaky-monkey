"""LangGraph-serializable workflow state."""
from __future__ import annotations

from typing import Any, TypedDict


class WorkflowGraphState(TypedDict, total=False):
    workflow: dict[str, Any]
    goal: str
    workflow_id: str
    current_step_id: str | None
    decision: str
    decision_reason: str
    approved_action: dict[str, Any] | None
    pending_input: dict[str, Any] | None
    step_result: dict[str, Any] | None
    step_verification: dict[str, Any] | None
    last_observation: dict[str, Any] | None
    failure_class: str | None
    recovery_attempts: int
    resume_value: Any
    result: dict[str, Any] | None
