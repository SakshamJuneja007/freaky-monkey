"""Adapter from LangGraph nodes to existing DEIMOS Session capabilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..api import AgentResult, TaskStatus
from ..planner.openai_compat import LLMClient, OpenAICompatPlanner
from ..policy import Decision, Policy
from ..types import Action
from .models import StepStatus, Workflow, workflow_step_status_from_runtime_state
from .decomposer import WorkflowDecomposer


class WorkflowResultNormalizationError(ValueError):
    """Raised when an existing DEIMOS result crosses the workflow boundary malformed."""


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    raise WorkflowResultNormalizationError(
        f"workflow result field {field!r} must be a mapping, got {type(value).__name__}"
    )


def normalize_execution_result(result: AgentResult) -> dict[str, Any]:
    """Normalize one authoritative DEIMOS AgentResult for a workflow step.

    The runner/result types are the source of truth. This adapter never infers
    success from a string or from the absence of an exception. A step is
    executable-complete only when the public result is SUCCESS *and* its final
    independent verification is PASS.
    """
    if not isinstance(result, AgentResult):
        raise WorkflowResultNormalizationError(
            f"workflow execution result must be AgentResult, got {type(result).__name__}"
        )

    result_json = result.to_json()
    if not isinstance(result_json, dict):
        raise WorkflowResultNormalizationError("AgentResult.to_json() did not return a mapping")

    outcome = result.outcome
    verification = outcome.final.to_json() if outcome is not None else {"verdict": result.verified}
    verification = _mapping(verification, "verification")
    verdict = str(verification.get("verdict") or result.verified or "UNKNOWN").upper()

    workflow_payload = outcome.workflow if outcome is not None else None
    if workflow_payload is not None:
        workflow_payload = _mapping(workflow_payload, "workflow")

    execution_ok = bool(result.ok)
    verified_pass = verdict == "PASS"
    ok = execution_ok and verified_pass

    failure_category = None
    if result.failure_categories:
        failure_category = str(result.failure_categories[-1])

    # ``AgentResult.detail`` is historically allowed to contain the short
    # execution detail (and can be the literal ``"ok"`` even when the final
    # verification is UNKNOWN).  Never surface that as a workflow failure
    # reason.  The authoritative failure category / verification verdict is
    # the useful terminal diagnostic.
    detail_text = str(result.detail or "").strip()
    if detail_text.lower() == "ok":
        detail_text = ""
    terminal_error = result.aborted_reason or detail_text
    if not terminal_error and verdict == "UNKNOWN" and failure_category:
        terminal_error = failure_category

    return {
        "ok": ok,
        "execution_ok": execution_ok,
        "execution_status": "COMPLETED" if execution_ok else result.status.value,
        "verification_status": verdict,
        "status": "COMPLETED" if ok else result.status.value,
        "result": result_json,
        "verification": verification,
        "workflow": workflow_payload,
        "error": None if ok else terminal_error,
        "error_category": failure_category,
        "error_detail": None if ok else terminal_error,
        "needs_input": bool(result.needs_input),
    }


class SessionWorkflowRuntime:
    """Thin orchestration adapter; it never owns browser or policy internals."""

    def __init__(self, session: Any) -> None:
        self.session = session
        self._decomposer: WorkflowDecomposer | None = None
        self._policy_cache: Policy | None = None


    @staticmethod
    def map_runtime_step_status(runtime_state: str) -> StepStatus:
        """Map authoritative RuntimeManager state without sharing its enum.

        BLOCKED is a runtime lifecycle condition, not a WorkflowStep member;
        it therefore becomes RECOVERY_REQUIRED. Unknown states fail closed to
        the same non-success workflow state.
        """
        return workflow_step_status_from_runtime_state(runtime_state)

    def _planner(self) -> OpenAICompatPlanner:
        if self._decomposer is None:
            if self.session.planner == "mock":
                class _Client:
                    model = "mock"
                planner = OpenAICompatPlanner(client=_Client())
            else:
                planner = OpenAICompatPlanner(client=LLMClient.from_env())
            self._decomposer = WorkflowDecomposer(planner)
        return self._decomposer

    def decompose_workflow(self, workflow_id: str, goal: str) -> Workflow:
        workflow = self._planner().decompose(workflow_id, goal)
        self.emit_workflow_event(workflow, "WORKFLOW_CREATED", step_count=len(workflow.steps))
        return workflow

    def _policy(self) -> Policy:
        # Policy construction is deterministic for the lifetime of this workflow
        # adapter. Reuse the immutable policy object instead of rebuilding it for
        # every atomic child; the policy check itself still runs for every action.
        if self._policy_cache is None:
            root = self.session._general_workspace
            if root is None:
                root = (self.session.workspace or Path.cwd()).resolve()
            self._policy_cache = Policy(workspace=root, confirm_mode="ask")
        return self._policy_cache

    @staticmethod
    def _missing(action: Action) -> str | None:
        if action.kind == "whatsapp_send_message":
            for field in ("recipient", "message"):
                value = action.params.get(field)
                if not isinstance(value, str) or not value.strip():
                    return field
        if action.kind in {"whatsapp_search_contact", "browser_play_song"}:
            value = action.params.get("query")
            if not isinstance(value, str) or not value.strip():
                return "query"
        return None

    def workflow_policy(self, workflow: Workflow, step_index: int) -> tuple[str, str, dict[str, Any] | None]:
        action = workflow.steps[step_index].action
        missing = self._missing(action)
        if missing:
            prompt = (
                f"What should I send to {action.params.get('recipient', 'the contact')}?"
                if missing == "message" else
                "Who should I send this message to?"
                if missing == "recipient" else
                "What should I search for?"
            )
            return "WAITING_FOR_USER", prompt, {
                "workflow_id": workflow.workflow_id,
                "step_id": workflow.steps[step_index].step_id,
                "field": missing,
                "prompt": prompt,
                "action": action.to_json(),
            }
        # Approval is evaluated for this atomic child only. A grant for one
        # high-impact action never authorizes a different decomposed child.
        decision, reason = self._policy().check(action)
        return decision.value, reason, None

    def request_workflow_approval(self, workflow: Workflow, step_index: int) -> dict[str, Any]:
        step = workflow.steps[step_index]
        task = self.session._runtime.get_task(workflow.workflow_id)
        import time
        approval_request_id = f"approval-{workflow.workflow_id}-{step.step_id}"
        payload = {
            "request": workflow.goal,
            "action": step.action.to_json(),
            "workflow_id": workflow.workflow_id,
            "step_id": step.step_id,
            "step_index": step.index,
            "approval_request_id": approval_request_id,
            "asked_at": time.time(),
        }
        # If the graph node is replayed after a resume, RuntimeManager may already
        # have been returned to RUNNING. In that case the durable approval has been
        # consumed and the node should not mint a second approval record.
        if task is not None and task.state == "WAITING_FOR_APPROVAL":
            existing = self.session._runtime.get_approval(workflow.workflow_id)
            if existing is None:
                raise RuntimeError(
                    f"workflow {workflow.workflow_id} is waiting for approval but has no durable approval owner"
                )
            if existing.get("workflow_id") != workflow.workflow_id or existing.get("step_id") != step.step_id:
                raise RuntimeError(
                    f"workflow approval owner mismatch for {workflow.workflow_id}: "
                    f"pending step={existing.get('step_id')!r}, current step={step.step_id!r}"
                )
            payload.update(existing)
        else:
            self.session._runtime.request_approval(workflow.workflow_id, payload)
        return {
            "type": "DEIMOS_APPROVAL",
            "workflow_id": workflow.workflow_id,
            "step_id": step.step_id,
            "approval_request_id": payload["approval_request_id"],
            "action": step.action.to_json(),
            "question": f"Approve {step.action.kind.replace('_', ' ')}?",
        }


    def resolve_workflow_approval(self, workflow: Workflow, approved: bool) -> None:
        approval = self.session._runtime.get_approval(workflow.workflow_id)
        if approval is None:
            raise RuntimeError(f"no pending approval exists for workflow {workflow.workflow_id}")
        approval_request_id = approval.get("approval_request_id")
        expected_step_id = approval.get("step_id")
        current_step_id = workflow.steps[workflow.current_step].step_id
        if expected_step_id != current_step_id:
            raise RuntimeError(
                f"approval owner mismatch for workflow {workflow.workflow_id}: "
                f"pending step {expected_step_id!r}, current step {current_step_id!r}"
            )
        self.session._runtime.resolve_approval(
            workflow.workflow_id, approved, approval_request_id=approval_request_id
        )
        if approved:
            task = self.session._runtime.get_task(workflow.workflow_id)
            if task is None:
                raise RuntimeError(f"workflow approval owner disappeared: {workflow.workflow_id}")
            metadata = dict(task.metadata)
            metadata["workflow_approval_granted"] = True
            metadata["workflow_approval_granted_step"] = current_step_id
            self.session._runtime.update_task(workflow.workflow_id, metadata=metadata)

    def request_workflow_input(self, workflow: Workflow, step_index: int, pending: dict[str, Any]) -> None:
        owner = {
            "input_id": f"input-{workflow.workflow_id}-{step_index}",
            "workflow_id": workflow.workflow_id,
            "step_id": workflow.steps[step_index].step_id,
            "task_id": workflow.workflow_id,
            "field": pending.get("field", ""),
            "prompt": pending.get("prompt", "Please provide the missing value."),
            "state": "PENDING",
        }
        try:
            self.session._runtime.set_pending_input(workflow.workflow_id, owner)
            self.session._runtime_transition(workflow.workflow_id, "WAITING_FOR_USER", event_type="WORKFLOW_INPUT_REQUESTED", execution_state="WAITING_FOR_USER")
        except Exception:
            pass

    def resolve_workflow_input(self, workflow: Workflow, step_index: int) -> None:
        try:
            self.session._runtime.clear_pending_input(workflow.workflow_id, event_type="WORKFLOW_INPUT_RESOLVED")
            self.session._runtime_transition(workflow.workflow_id, "RUNNING", event_type="WORKFLOW_INPUT_RESUMED", execution_state="RUNNING")
        except Exception:
            pass

    def observe_workflow_step(self, workflow: Workflow, step_index: int) -> dict[str, Any]:
        # Actual freshness gating remains in runner.run_task. This observation is
        # orchestration telemetry only; it never becomes verification evidence.
        step = workflow.steps[step_index]
        self.emit_workflow_event(workflow, "STEP_SELECTED", step_id=step.step_id, resource=step.resource_key)
        return {"resource_key": step.resource_key or "none", "capability": step.action.kind}

    def execute_workflow_step(self, workflow: Workflow, step_index: int, approved_action: dict[str, Any] | None) -> dict[str, Any]:
        from ..session import Prepared, UserTask
        from ..workflow import WorkflowStep as PublicWorkflowStep
        from ..workflow import Workflow as PublicWorkflow
        # The compatibility workflow model and the new domain model share the
        # same JSON representation. Convert through JSON so no second execution
        # path or parallel action model is introduced.
        legacy = PublicWorkflow.from_json(workflow.to_json())
        step = legacy.steps[step_index]

        # ``approved_action`` is supplied only by the current approval node and
        # is fingerprinted by the existing runner policy gate. Never synthesize
        # approval for a later child from workflow metadata.
        task = UserTask(raw=workflow.goal, text=workflow.goal, source="text", task_id=workflow.workflow_id, status="accepted")
        turn = self.session._run(Prepared(
            task=task,
            goal=workflow.goal,
            kind="workflow step",
            task_obj=__import__("agent_control.workflow", fromlist=["WorkflowStepTask"]).WorkflowStepTask(workflow=legacy, step=step),
            workspace=self.session.workspace,
            approved_action=approved_action,
            browser_resource_key=step.resource_key,
            runtime_task_id=workflow.workflow_id,
            suppress_presentation=True,
        ))
        result = turn.result
        if result is None:
            raise WorkflowResultNormalizationError("workflow step returned no AgentResult")
        normalized = normalize_execution_result(result)
        if normalized["workflow"] is None:
            normalized["workflow"] = legacy.to_json()
        return normalized

    def verify_workflow_step(self, workflow: Workflow, step_index: int) -> dict[str, Any]:
        """Run the existing independent checkpoint verifier without executing."""
        from ..workflow import Workflow as PublicWorkflow
        legacy = PublicWorkflow.from_json(workflow.to_json())
        step = legacy.steps[step_index]
        verification = step_task_verification = __import__("agent_control.workflow", fromlist=["WorkflowStepTask"]).WorkflowStepTask(workflow=legacy, step=step).verify_checkpoint(self._policy(), step.action)
        return verification.to_json() if verification is not None else {"verdict": "UNKNOWN", "reason": "no independent checkpoint verifier"}

    def recover_workflow_resource(self, workflow: Workflow, step_index: int) -> bool:
        """Delegate resource recovery to the existing session/browser owner."""
        hook = getattr(self.session, "recover_workflow_resource", None)
        if callable(hook):
            return bool(hook(workflow, step_index))
        return False

    def replan_workflow_step(self, workflow: Workflow, step_index: int) -> bool:
        """Delegate semantic re-planning to the existing decomposer/planner."""
        hook = getattr(self.session, "replan_workflow_step", None)
        if callable(hook):
            return bool(hook(workflow, step_index))
        return False

    def emit_workflow_event(self, workflow: Workflow, event: str, **metadata: Any) -> None:
        try:
            if event == "WORKFLOW_COMPLETED":
                self.session._runtime_transition(workflow.workflow_id, "COMPLETED", event_type="WORKFLOW_COMPLETED", execution_state="COMPLETED", verification_state="PASS", verification_summary="all workflow steps independently verified")
            elif event == "WORKFLOW_FAILED":
                self.session._runtime_transition(workflow.workflow_id, "FAILED", event_type="WORKFLOW_FAILED", execution_state="FAILED", verification_state="UNKNOWN", failure_category="WORKFLOW")
            self.session._runtime.update_workflow(
                workflow.workflow_id,
                workflow.to_json(),
                event_type=event,
            )
        except Exception:
            pass
        if self.session.debug:
            self.session.narrator.note(f"[{workflow.workflow_id}] {event}" + (f" {json.dumps(metadata, default=str)}" if metadata else ""))
