"""LangGraph nodes for the DEIMOS workflow orchestration layer.

Nodes only orchestrate. Execution, policy, browser control, observation and
verification are delegated to an injected DEIMOS runtime adapter.
"""
from __future__ import annotations

from typing import Any, Protocol

from ..policy import Decision
from .models import StepStatus, Workflow, WorkflowStatus
from ..recovery import action_can_retry, retry_safety
from ..types import RetrySafety
from .resources import bind_resources
from .scheduler import DependencyScheduler
from concurrent.futures import ThreadPoolExecutor, as_completed
from .state import WorkflowGraphState


class WorkflowRuntimeAdapter(Protocol):
    def decompose_workflow(self, workflow_id: str, goal: str) -> Workflow: ...
    def workflow_policy(self, workflow: Workflow, step_index: int) -> tuple[str, str, dict[str, Any] | None]: ...
    def request_workflow_approval(self, workflow: Workflow, step_index: int) -> dict[str, Any]: ...
    def resolve_workflow_approval(self, workflow: Workflow, approved: bool) -> None: ...
    def request_workflow_input(self, workflow: Workflow, step_index: int, pending: dict[str, Any]) -> None: ...
    def resolve_workflow_input(self, workflow: Workflow, step_index: int) -> None: ...
    def execute_workflow_step(self, workflow: Workflow, step_index: int, approved_action: dict[str, Any] | None) -> dict[str, Any]: ...
    def observe_workflow_step(self, workflow: Workflow, step_index: int) -> dict[str, Any]: ...
    def emit_workflow_event(self, workflow: Workflow, event: str, **metadata: Any) -> None: ...


def _workflow(state: WorkflowGraphState) -> Workflow:
    return Workflow.from_json(state["workflow"])


def _completed_ids(workflow: Workflow) -> set[str]:
    return {step.step_id for step in workflow.steps if step.status is StepStatus.COMPLETED}


def _next_runnable_step(workflow: Workflow) -> Any | None:
    completed = _completed_ids(workflow)
    for step in workflow.steps:
        if step.status is StepStatus.COMPLETED:
            continue
        if step.status is StepStatus.PENDING and all(dep in completed for dep in step.dependencies):
            return step
    return None


def load_workflow(state: WorkflowGraphState, runtime: WorkflowRuntimeAdapter) -> dict[str, Any]:
    workflow = _workflow(state) if state.get("workflow") else runtime.decompose_workflow(state["workflow_id"], state["goal"])
    bind_resources(workflow)
    if workflow.cancel_requested or workflow.status is WorkflowStatus.CANCELLED:
        workflow.status = WorkflowStatus.CANCELLED
        runtime.emit_workflow_event(workflow, "WORKFLOW_CANCELLED")
        return {"workflow": workflow.to_json(), "decision": "CANCELLED"}
    workflow.status = WorkflowStatus.RUNNING
    runtime.emit_workflow_event(workflow, "WORKFLOW_LOADED")
    if any(step.status in {StepStatus.RECOVERY_REQUIRED, StepStatus.UNKNOWN} for step in workflow.steps):
        runtime.emit_workflow_event(workflow, "WORKFLOW_RESUMED", step_id=workflow.current_step_id)
    return {"workflow": workflow.to_json(), "goal": workflow.goal, "current_step_id": workflow.current_step_id}


def decompose(state: WorkflowGraphState, runtime: WorkflowRuntimeAdapter) -> dict[str, Any]:
    if state.get("workflow"):
        workflow = _workflow(state)
    else:
        workflow = runtime.decompose_workflow(state["workflow_id"], state["goal"])
    bind_resources(workflow)
    if workflow.cancel_requested or workflow.status is WorkflowStatus.CANCELLED:
        workflow.status = WorkflowStatus.CANCELLED
        runtime.emit_workflow_event(workflow, "WORKFLOW_CANCELLED")
        return {"workflow": workflow.to_json(), "current_step_id": workflow.current_step_id, "decision": "CANCELLED"}
    runtime.emit_workflow_event(workflow, "WORKFLOW_DECOMPOSED", step_count=len(workflow.steps))
    return {"workflow": workflow.to_json(), "current_step_id": workflow.current_step_id}


def select_next_step(state: WorkflowGraphState, runtime: WorkflowRuntimeAdapter) -> dict[str, Any]:
    workflow = _workflow(state)
    if workflow.cancel_requested or workflow.status is WorkflowStatus.CANCELLED:
        workflow.status = WorkflowStatus.CANCELLED
        for step in workflow.steps:
            if step.status not in {StepStatus.COMPLETED, StepStatus.CANCELLED}:
                step.status = StepStatus.CANCELLED
        runtime.emit_workflow_event(workflow, "WORKFLOW_CANCELLED")
        return {"workflow": workflow.to_json(), "decision": "CANCELLED"}
    scheduler = DependencyScheduler()
    scheduler.validate(workflow)
    ready = scheduler.ready_nodes(workflow)
    if ready:
        workflow.status = WorkflowStatus.RUNNING
        ids = [node.step_id for node in ready]
        workflow.current_step_id = ready[0].step_id
        runtime.emit_workflow_event(workflow, "READY_NODES_SELECTED", step_ids=ids)
        return {"workflow": workflow.to_json(), "current_step_id": ready[0].step_id, "ready_step_ids": ids}
    if all(step.status is StepStatus.COMPLETED for step in workflow.steps):
        workflow.status = WorkflowStatus.COMPLETED
        runtime.emit_workflow_event(workflow, "WORKFLOW_COMPLETED")
        return {"workflow": workflow.to_json(), "result": {"status": "COMPLETED"}}
    workflow.status = DependencyScheduler.aggregate(workflow)
    runtime.emit_workflow_event(workflow, "WORKFLOW_NOT_READY", status=workflow.status.value)
    return {"workflow": workflow.to_json(), "decision": "FAILED" if workflow.status in {WorkflowStatus.FAILED, WorkflowStatus.BLOCKED, WorkflowStatus.PARTIAL_FAILURE, WorkflowStatus.UNKNOWN} else "WAIT"}


def route_after_select(state: WorkflowGraphState) -> str:
    workflow = _workflow(state)
    if workflow.status is WorkflowStatus.COMPLETED:
        return "done"
    if workflow.status in {WorkflowStatus.FAILED, WorkflowStatus.PARTIAL_FAILURE, WorkflowStatus.UNKNOWN, WorkflowStatus.CANCELLED, WorkflowStatus.BLOCKED}:
        return "failed"
    ready = state.get("ready_step_ids") or []
    return "batch" if len(ready) > 1 else "observe"


def route_after_batch(state: WorkflowGraphState) -> str:
    decision = str(state.get("decision", "BATCH_DONE"))
    if decision == "APPROVAL_REQUIRED":
        return "approval"
    if decision == "WAITING_FOR_USER":
        return "wait"
    workflow = _workflow(state)
    if workflow.status is WorkflowStatus.COMPLETED:
        return "done"
    if workflow.status in {WorkflowStatus.FAILED, WorkflowStatus.PARTIAL_FAILURE, WorkflowStatus.UNKNOWN, WorkflowStatus.CANCELLED, WorkflowStatus.BLOCKED}:
        return "failed"
    return "next"


def observe(state: WorkflowGraphState, runtime: WorkflowRuntimeAdapter) -> dict[str, Any]:
    workflow = _workflow(state)
    index = workflow.current_step
    import time
    started = time.perf_counter()
    observation = runtime.observe_workflow_step(workflow, index)
    step = workflow.steps[index]
    step.observation = dict(observation or {})
    step.observation["observed_at"] = time.time()
    step.timing["observation_ms"] = round((time.perf_counter() - started) * 1000, 3)
    if step.observation.get("stale") is True:
        step.failure_category = "stale_observation"
        runtime.emit_workflow_event(workflow, "STALE_STATE_DETECTED", step_id=step.step_id, attempt=step.attempt_count, failure_category="stale_observation")
    runtime.emit_workflow_event(workflow, "STEP_OBSERVED", step_id=step.step_id)
    return {"workflow": workflow.to_json(), "last_observation": step.observation}


def policy(state: WorkflowGraphState, runtime: WorkflowRuntimeAdapter) -> dict[str, Any]:
    workflow = _workflow(state)
    index = workflow.current_step
    step = workflow.steps[index]
    # A technical recovery of an already-approved action does not constitute a
    # new approval request. Only a genuinely new action/fingerprint should pass
    # through policy again.
    if step.policy_state == "APPROVED":
        decision, reason, pending_input = Decision.ALLOW.value, "previously approved action is being recovered", None
    else:
        decision, reason, pending_input = runtime.workflow_policy(workflow, index)
    if pending_input is not None:
        runtime.request_workflow_input(workflow, index, pending_input)
        step.status = StepStatus.WAITING_FOR_USER
        workflow.status = WorkflowStatus.WAITING_FOR_USER
        runtime.emit_workflow_event(workflow, "STEP_WAITING_FOR_USER", step_id=step.step_id)
        return {"workflow": workflow.to_json(), "decision": "WAITING_FOR_USER", "decision_reason": reason, "pending_input": pending_input}
    if decision == Decision.CONFIRM.value:
        step.status = StepStatus.WAITING_FOR_APPROVAL
        workflow.status = WorkflowStatus.WAITING_FOR_APPROVAL
        runtime.emit_workflow_event(workflow, "STEP_WAITING_FOR_APPROVAL", step_id=step.step_id)
        return {"workflow": workflow.to_json(), "decision": "APPROVAL_REQUIRED", "decision_reason": reason}
    if decision == Decision.DENY.value:
        step.status = StepStatus.FAILED
        step.error = reason
        workflow.status = WorkflowStatus.FAILED
        runtime.emit_workflow_event(workflow, "STEP_FAILED", step_id=step.step_id, reason=reason)
        return {"workflow": workflow.to_json(), "decision": "FAILURE", "decision_reason": reason}
    runtime.emit_workflow_event(workflow, "POLICY_ALLOWED", step_id=step.step_id)
    return {"workflow": workflow.to_json(), "decision": "ALLOW", "decision_reason": reason}


def route_after_policy(state: WorkflowGraphState) -> str:
    return {
        "ALLOW": "execute",
        "APPROVAL_REQUIRED": "approval",
        "WAITING_FOR_USER": "workflow_input",
        "FAILURE": "failed",
    }.get(str(state.get("decision")), "failed")


def approval(state: WorkflowGraphState, runtime: WorkflowRuntimeAdapter) -> dict[str, Any]:
    workflow = _workflow(state)
    index = workflow.current_step
    # The request itself is persisted by RuntimeManager before the interrupt.
    # On resume the node is replayed from the beginning; the adapter must treat
    # an already-owned approval as idempotent and return the same payload.
    payload = runtime.request_workflow_approval(workflow, index)
    from langgraph.types import interrupt
    decision = interrupt(payload)
    approved = bool(decision) if not isinstance(decision, dict) else bool(decision.get("approved"))
    if not approved:
        runtime.resolve_workflow_approval(workflow, False)
        workflow.steps[index].status = StepStatus.CANCELLED
        workflow.status = WorkflowStatus.CANCELLED
        runtime.emit_workflow_event(workflow, "STEP_CANCELLED", step_id=workflow.steps[index].step_id)
        return {"workflow": workflow.to_json(), "decision": "FAILURE", "decision_reason": "workflow step was not approved"}
    runtime.resolve_workflow_approval(workflow, True)
    workflow.steps[index].policy_state = "APPROVED"
    action = workflow.steps[index].action.to_json()
    runtime.emit_workflow_event(workflow, "APPROVAL_GRANTED", step_id=workflow.steps[index].step_id)
    return {"workflow": workflow.to_json(), "approved_action": action, "decision": "ALLOW"}


def workflow_input(state: WorkflowGraphState, runtime: WorkflowRuntimeAdapter) -> dict[str, Any]:
    workflow = _workflow(state)
    index = workflow.current_step
    pending = state.get("pending_input") or {}
    from langgraph.types import interrupt
    value = interrupt(pending)
    if value is None or not str(value).strip():
        return {"decision": "WAITING_FOR_USER"}
    field = str(pending.get("field", ""))
    if not field:
        raise RuntimeError("workflow input owner is missing field")
    workflow.steps[index].action.params[field] = str(value).strip()
    runtime.resolve_workflow_input(workflow, index)
    runtime.emit_workflow_event(workflow, "WORKFLOW_INPUT_CONSUMED", step_id=workflow.steps[index].step_id, field=field)
    workflow.steps[index].status = StepStatus.PENDING
    workflow.steps[index].error = None
    workflow.status = WorkflowStatus.RUNNING
    runtime.emit_workflow_event(workflow, "WORKFLOW_INPUT_RECEIVED", step_id=workflow.steps[index].step_id, field=field)
    return {"workflow": workflow.to_json(), "pending_input": None, "decision": "ALLOW"}


def execute_ready_batch(state: WorkflowGraphState, runtime: WorkflowRuntimeAdapter) -> dict[str, Any]:
    """Run policy-approved ready branches concurrently through the existing adapter."""
    workflow = _workflow(state)
    ready_ids = list(state.get("ready_step_ids") or [n.step_id for n in DependencyScheduler.ready_nodes(workflow)])
    pending_approval = None
    executable = []
    for node_id in ready_ids:
        node = next((n for n in workflow.steps if n.step_id == node_id), None)
        if node is None or node.status is StepStatus.COMPLETED:
            continue
        index = node.index
        runtime.emit_workflow_event(workflow, "STEP_SELECTED", step_id=node.step_id, step_index=index)
        observation = runtime.observe_workflow_step(workflow, index)
        node.observation = dict(observation or {})
        decision, reason, pending_input = runtime.workflow_policy(workflow, index)
        if pending_input is not None:
            node.status = StepStatus.WAITING_FOR_USER
            workflow.status = WorkflowStatus.WAITING_FOR_USER
            runtime.request_workflow_input(workflow, index, pending_input)
            return {"workflow": workflow.to_json(), "decision": "WAITING_FOR_USER", "pending_input": pending_input}
        if decision == Decision.CONFIRM.value:
            node.status = StepStatus.WAITING_FOR_APPROVAL
            pending_approval = node.step_id
            continue
        if decision == Decision.DENY.value:
            node.status = StepStatus.FAILED
            node.error = reason
            continue
        node.policy_state = "ALLOWED"
        executable.append(node)

    def run_branch(node):
        import time
        started = time.perf_counter()
        node.status = StepStatus.RUNNING
        node.attempt_count += 1
        runtime.emit_workflow_event(workflow, "STEP_EXECUTION_STARTED", step_id=node.step_id, resource=node.resource_key)
        result = runtime.execute_workflow_step(workflow, node.index, None)
        node.timing["step_total_ms"] = round((time.perf_counter() - started) * 1000, 3)
        runtime.emit_workflow_event(workflow, "STEP_TIMING", step_id=node.step_id, timing=node.timing)
        return node, result

    branch_results = dict(state.get("branch_results") or {})
    scheduler = DependencyScheduler()
    # Resource ownership is acquired per branch; no capability implementation is
    # duplicated here. Different resources overlap, shared resources serialize.
    futures = {}
    with ThreadPoolExecutor(max_workers=scheduler.max_concurrency, thread_name_prefix="deimos-workflow") as pool:
        for node in executable:
            if not scheduler.resources.try_acquire(node.resource_key, node.step_id):
                node.status = StepStatus.WAITING_RESOURCE
                continue
            futures[pool.submit(run_branch, node)] = node
        for future in as_completed(futures):
            node = futures[future]
            try:
                _, result = future.result()
                normalized = dict(result or {})
                verification = dict(normalized.get("verification") or {})
                step_result = dict(normalized.get("result") or {})
                branch_results[node.step_id] = normalized
                if normalized.get("needs_input"):
                    node.status = StepStatus.WAITING_FOR_USER
                elif normalized.get("ok") and str(verification.get("verdict", "UNKNOWN")).upper() == "PASS":
                    node.status = StepStatus.COMPLETED
                    node.result = step_result
                    node.verification = verification
                elif str(verification.get("verdict", "")).upper() == "UNKNOWN":
                    node.status = StepStatus.UNKNOWN
                    node.verification = verification
                else:
                    node.status = StepStatus.FAILED
                    node.error = str(normalized.get("error") or "workflow step failed")
                    node.verification = verification
            except Exception as exc:
                node.status = StepStatus.FAILED
                node.error = f"{type(exc).__name__}: {exc}"
                branch_results[node.step_id] = {"status": "FAILED", "error": node.error}
            finally:
                scheduler.resources.release(node.resource_key, node.step_id)

    workflow.status = DependencyScheduler.aggregate(workflow)
    result = {"status": workflow.status.value, "branches": branch_results}
    if pending_approval:
        workflow.current_step_id = pending_approval
        workflow.status = WorkflowStatus.WAITING_FOR_APPROVAL
        return {"workflow": workflow.to_json(), "decision": "APPROVAL_REQUIRED", "pending_approval_step_id": pending_approval, "branch_results": branch_results, "result": result}
    return {"workflow": workflow.to_json(), "decision": "BATCH_DONE", "branch_results": branch_results, "result": result}

def execute(state: WorkflowGraphState, runtime: WorkflowRuntimeAdapter) -> dict[str, Any]:
    workflow = _workflow(state)
    index = workflow.current_step
    step = workflow.steps[index]
    import time
    execution_started = time.perf_counter()
    step.status = StepStatus.RUNNING
    step.attempt_count += 1
    workflow.status = WorkflowStatus.RUNNING
    runtime.emit_workflow_event(workflow, "STEP_EXECUTION_STARTED", step_id=step.step_id, resource=step.resource_key)
    result = runtime.execute_workflow_step(workflow, index, state.get("approved_action"))
    step.timing["execution_ms"] = round((time.perf_counter() - execution_started) * 1000, 3)
    step.timing["total_step_ms"] = step.timing["execution_ms"]
    approved = state.get("approved_action")
    step_result = dict(result.get("result") or {})
    verification = dict(result.get("verification") or {})
    workflow_payload = result.get("workflow")
    if isinstance(workflow_payload, dict):
        workflow = Workflow.from_json(workflow_payload)
        step = workflow.steps[index]
    if result.get("needs_input"):
        workflow.status = WorkflowStatus.WAITING_FOR_USER
    elif result.get("ok"):
        step.status = StepStatus.VERIFYING
    else:
        step.error = str(result.get("error") or "workflow step failed")
        raw_result = (
            result.get("error_category")
            or step_result.get("failure_class")
            or step_result.get("skill_failure_class")
        )
        if raw_result:
            step.failure_category = str(raw_result)
        outcome = result.get("outcome") or {}
        if isinstance(outcome, dict):
            categories = outcome.get("failure_categories") or []
            if not step.failure_category and categories:
                step.failure_category = str(categories[-1])
            if outcome.get("aborted_reason"):
                step.recovery = {
                    "decision": "ABORTED",
                    "reason": str(outcome["aborted_reason"]),
                    "attempts": int(outcome.get("recovery_attempts", 0) or 0),
                }
        step.verification = verification or {"verdict": result.get("verification", {}).get("verdict", "UNKNOWN")}
        verdict = str(step.verification.get("verdict") or result.get("verification_status") or "UNKNOWN").upper()
        step.failure_category = step.failure_category or ("verification_unknown" if verdict == "UNKNOWN" else "execution_exception")
        # UNKNOWN is an unresolved postcondition, not an execution failure.
        # Preserve it so the recovery graph can re-observe instead of converting
        # it into FAILED merely because AgentResult.ok is false.
        if workflow.cancel_requested:
            step.status = StepStatus.CANCELLED
            workflow.status = WorkflowStatus.CANCELLED
        elif step.failure_category in {"application_not_found", "environment", "permission_denied", "policy_denied"}:
            # A known execution/environment failure is a real failure, not an
            # unresolved verification state. Dependents are blocked by the DAG
            # and the original category remains attached to this failed step.
            step.status = StepStatus.FAILED
            workflow.status = WorkflowStatus.FAILED
        elif verdict == "UNKNOWN":
            step.status = StepStatus.RECOVERY_REQUIRED
            workflow.status = WorkflowStatus.RECOVERING
        elif action_can_retry(step.action) or step.failure_category in {"browser_connection_failed", "stale_observation", "target_not_found", "application_not_found"}:
            step.status = StepStatus.RECOVERY_REQUIRED
            workflow.status = WorkflowStatus.RECOVERING
        else:
            step.status = StepStatus.UNKNOWN
            workflow.status = WorkflowStatus.UNKNOWN
    runtime.emit_workflow_event(workflow, "STEP_EXECUTION_RESULT", step_id=step.step_id, ok=bool(result.get("ok")), failure_category=step.failure_category)
    return {
        "workflow": workflow.to_json(),
        "step_result": step_result,
        "step_verification": verification,
        "approved_action": approved,
        "decision": "WAITING_FOR_USER" if result.get("needs_input") else ("VERIFY" if result.get("ok") else "FAILURE"),
    }


def verify(state: WorkflowGraphState, runtime: WorkflowRuntimeAdapter) -> dict[str, Any]:
    workflow = _workflow(state)
    index = workflow.current_step
    step = workflow.steps[index]
    verification_started = __import__("time").perf_counter()
    verification = dict(state.get("step_verification") or {})
    verdict = str(verification.get("verdict") or "UNKNOWN").upper()
    step.timing["verification_ms"] = round((__import__("time").perf_counter() - verification_started) * 1000, 3)
    if verdict == "PASS":
        step.status = StepStatus.COMPLETED
        step.verification = verification
        step.result = dict(state.get("step_result") or {})
        workflow.status = WorkflowStatus.RUNNING
        runtime.emit_workflow_event(workflow, "STEP_COMPLETED", step_id=step.step_id)
        return {"workflow": workflow.to_json(), "decision": "PASS", "approved_action": None}
    if verdict == "UNKNOWN":
        step.status = StepStatus.RECOVERY_REQUIRED
        step.verification = verification
        step.failure_category = "verification_unknown"
        workflow.status = WorkflowStatus.RECOVERING
        runtime.emit_workflow_event(workflow, "STEP_FAILURE_CLASSIFIED", step_id=step.step_id, attempt=step.attempt_count, failure_category="verification_unknown", recoverability="REOBSERVE")
        runtime.emit_workflow_event(workflow, "STEP_RECOVERY_REQUIRED", step_id=step.step_id)
        return {"workflow": workflow.to_json(), "decision": "UNKNOWN", "failure_class": "INCONCLUSIVE", "approved_action": None}
    step.status = StepStatus.FAILED
    step.verification = verification
    step.error = str(verification.get("reason") or "independent verification failed")
    workflow.status = WorkflowStatus.FAILED
    runtime.emit_workflow_event(workflow, "STEP_FAILED", step_id=step.step_id, reason=step.error)
    return {"workflow": workflow.to_json(), "decision": "FAILURE", "approved_action": None}


def route_after_verify(state: WorkflowGraphState) -> str:
    return {"PASS": "next_step", "UNKNOWN": "recover", "FAILURE": "failed"}.get(str(state.get("decision")), "failed")


def next_step(state: WorkflowGraphState, runtime: WorkflowRuntimeAdapter) -> dict[str, Any]:
    workflow = _workflow(state)
    previous_id = workflow.current_step_id
    runnable = _next_runnable_step(workflow)
    if runnable is not None:
        workflow.current_step_id = runnable.step_id
        workflow.status = WorkflowStatus.RUNNING
        runtime.emit_workflow_event(workflow, "STEP_ADVANCED", from_step=previous_id, to_step=runnable.step_id)
        runtime.emit_workflow_event(workflow, "STEP_SELECTED", step_id=runnable.step_id, step_index=runnable.index)
        return {"workflow": workflow.to_json(), "current_step_id": runnable.step_id}
    from .scheduler import DependencyScheduler
    workflow.status = DependencyScheduler.aggregate(workflow)
    if workflow.status is WorkflowStatus.COMPLETED:
        runtime.emit_workflow_event(workflow, "WORKFLOW_COMPLETED")
        return {"workflow": workflow.to_json(), "decision": "DONE", "result": {"status": "COMPLETED"}}
    if workflow.status is WorkflowStatus.PARTIAL_FAILURE:
        runtime.emit_workflow_event(workflow, "WORKFLOW_PARTIAL_COMPLETION", status=workflow.status.value)
    elif workflow.status is WorkflowStatus.UNKNOWN:
        runtime.emit_workflow_event(workflow, "WORKFLOW_TERMINAL", status="UNKNOWN")
    return {
        "workflow": workflow.to_json(),
        "decision": "TERMINAL",
        "result": {
            "status": workflow.status.value,
            "completed_steps": [s.step_id for s in workflow.steps if s.status is StepStatus.COMPLETED],
            "failed_steps": [s.step_id for s in workflow.steps if s.status in {StepStatus.FAILED, StepStatus.BLOCKED}],
            "unknown_steps": [s.step_id for s in workflow.steps if s.status is StepStatus.UNKNOWN],
        },
    }


def recover(state: WorkflowGraphState, runtime: WorkflowRuntimeAdapter) -> dict[str, Any]:
    """Bounded P2.6 recovery coordinator for the existing graph.

    It never marks a step complete from execution evidence.  Recovery either
    obtains new verification evidence, prepares a safe retry, requests a replan,
    or leaves the workflow honestly unresolved.
    """
    workflow = _workflow(state)
    index = workflow.current_step
    step = workflow.steps[index]
    import time
    recovery_started = time.perf_counter()

    if workflow.cancel_requested or workflow.status is WorkflowStatus.CANCELLED:
        step.status = StepStatus.CANCELLED
        workflow.status = WorkflowStatus.CANCELLED
        runtime.emit_workflow_event(workflow, "WORKFLOW_CANCELLED", step_id=step.step_id)
        step.timing["recovery_ms"] = round((time.perf_counter() - recovery_started) * 1000, 3)
        return {"workflow": workflow.to_json(), "decision": "CANCELLED"}

    if step.attempt_count > step.max_step_retries:
        step.failure_category = "recovery_exhausted"
        step.recovery["decision"] = "STEP_RETRY_EXHAUSTED"
        workflow.status = WorkflowStatus.UNKNOWN if step.status in {StepStatus.UNKNOWN, StepStatus.RECOVERY_REQUIRED} else WorkflowStatus.FAILED
        runtime.emit_workflow_event(workflow, "STEP_RETRY_EXHAUSTED", step_id=step.step_id, attempt=step.attempt_count, failure_category=step.failure_category)
        step.timing["recovery_ms"] = round((time.perf_counter() - recovery_started) * 1000, 3)
        return {"workflow": workflow.to_json(), "decision": "UNKNOWN" if workflow.status is WorkflowStatus.UNKNOWN else "FAILURE"}

    if workflow.recovery_attempts >= workflow.max_recovery_attempts:
        step.failure_category = "recovery_exhausted"
        step.recovery["decision"] = "RECOVERY_EXHAUSTED"
        workflow.status = WorkflowStatus.UNKNOWN if step.status in {StepStatus.UNKNOWN, StepStatus.RECOVERY_REQUIRED} else WorkflowStatus.FAILED
        runtime.emit_workflow_event(workflow, "RECOVERY_EXHAUSTED", step_id=step.step_id, attempt=workflow.recovery_attempts, failure_category=step.failure_category)
        step.timing["recovery_ms"] = round((time.perf_counter() - recovery_started) * 1000, 3)
        return {"workflow": workflow.to_json(), "decision": "UNKNOWN" if workflow.status is WorkflowStatus.UNKNOWN else "FAILURE"}

    workflow.recovery_attempts += 1
    attempt = workflow.recovery_attempts
    failure = str(state.get("failure_class") or step.failure_category or "execution_exception")
    safety = retry_safety(step.action)
    step.recovery.update({"attempt": attempt, "max_attempts": workflow.max_recovery_attempts, "failure_category": failure, "retry_safety": safety.value})
    runtime.emit_workflow_event(workflow, "WORKFLOW_RECOVERY_STARTED", step_id=step.step_id, attempt=attempt, failure_category=failure, recovery_strategy=step.recovery.get("decision", ""))
    runtime.emit_workflow_event(workflow, "STEP_FAILURE_CLASSIFIED", step_id=step.step_id, attempt=attempt, failure_category=failure, recoverability=safety.value)

    def finish_recovery() -> None:
        step.timing["recovery_ms"] = round((time.perf_counter() - recovery_started) * 1000, 3)
        step.timing["retry_ms"] = step.timing["recovery_ms"]

    # UNKNOWN after a possibly completed side effect: observe and verify, never
    # execute the side effect again unless the fresh evidence proves it is absent.
    if failure in {"verification_unknown", "INCONCLUSIVE", "action_may_have_succeeded", "ACTION_MAY_HAVE_SUCCEEDED"} or safety is RetrySafety.REOBSERVE_FIRST:
        runtime.emit_workflow_event(workflow, "FRESH_OBSERVATION_REQUESTED", step_id=step.step_id, attempt=attempt, failure_category=failure, recovery_strategy="REOBSERVE")
        observed = runtime.observe_workflow_step(workflow, index)
        step.observation = dict(observed or {})
        verifier = getattr(runtime, "verify_workflow_step", None)
        if callable(verifier):
            import time
            started = time.perf_counter()
            verification = dict(verifier(workflow, index) or {})
            step.timing["verification_ms"] = round((time.perf_counter() - started) * 1000, 3)
            step.verification = verification
            verdict = str(verification.get("verdict") or "UNKNOWN").upper()
            if verdict == "PASS":
                step.status = StepStatus.COMPLETED
                workflow.status = DependencyScheduler.aggregate(workflow)
                runtime.emit_workflow_event(workflow, "STEP_COMPLETED", step_id=step.step_id, recovered=True)
                finish_recovery()
                return {"workflow": workflow.to_json(), "decision": "PASS"}
            if verdict == "FAIL" and safety is RetrySafety.SAFE_TO_RETRY:
                step.status = StepStatus.READY
                runtime.emit_workflow_event(workflow, "STEP_RETRY_STARTED", step_id=step.step_id, attempt=step.attempt_count + 1, recovery_strategy="RETRY_STEP")
                finish_recovery()
                return {"workflow": workflow.to_json(), "decision": "RETRY"}
        step.status = StepStatus.UNKNOWN
        workflow.status = WorkflowStatus.UNKNOWN
        step.failure_category = "verification_unknown"
        finish_recovery()
        return {"workflow": workflow.to_json(), "decision": "UNKNOWN"}

    if failure in {"browser_connection_failed", "resource_unavailable"}:
        recover_resource = getattr(runtime, "recover_workflow_resource", None)
        if callable(recover_resource):
            runtime.emit_workflow_event(workflow, "RESOURCE_RECOVERY_STARTED", step_id=step.step_id, attempt=attempt, failure_category=failure, recovery_strategy="RECOVER_RESOURCE")
            ok = bool(recover_resource(workflow, index))
            if ok:
                step.status = StepStatus.READY
                workflow.status = WorkflowStatus.RUNNING
                finish_recovery()
                return {"workflow": workflow.to_json(), "decision": "RETRY"}

    if failure in {"target_not_found", "application_not_found", "precondition_failed", "STALE_STATE"}:
        replan = getattr(runtime, "replan_workflow_step", None)
        if callable(replan):
            runtime.emit_workflow_event(workflow, "REPLAN_STARTED", step_id=step.step_id, attempt=attempt, failure_category=failure, recovery_strategy="REPLAN")
            ok = bool(replan(workflow, index))
            if ok:
                step.status = StepStatus.READY
                workflow.status = WorkflowStatus.RUNNING
                finish_recovery()
                return {"workflow": workflow.to_json(), "decision": "RETRY"}

    if failure in {"browser_connection_failed", "target_not_found", "application_not_found", "stale_observation"}:
        runtime.emit_workflow_event(workflow, "FRESH_OBSERVATION_REQUESTED", step_id=step.step_id, attempt=attempt, failure_category=failure, recovery_strategy="REOBSERVE")
        observed = runtime.observe_workflow_step(workflow, index)
        step.observation = dict(observed or {})

    if action_can_retry(step.action):
        step.status = StepStatus.READY
        workflow.status = WorkflowStatus.RUNNING
        runtime.emit_workflow_event(workflow, "STEP_RETRY_STARTED", step_id=step.step_id, attempt=step.attempt_count + 1, recovery_strategy="RETRY_STEP")
        finish_recovery()
        return {"workflow": workflow.to_json(), "decision": "RETRY"}

    step.status = StepStatus.UNKNOWN
    workflow.status = WorkflowStatus.UNKNOWN
    step.failure_category = failure
    step.recovery["decision"] = "ASK_USER" if safety is RetrySafety.ASK_USER else "UNKNOWN"
    finish_recovery()
    return {"workflow": workflow.to_json(), "decision": "ASK_USER" if safety is RetrySafety.ASK_USER else "UNKNOWN"}
