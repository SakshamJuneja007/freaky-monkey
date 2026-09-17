"""P3.2-B deterministic extraction of verified workflows into procedure candidates.

This module deliberately sits *after* the existing workflow verification boundary.
It observes authoritative workflow/result state, converts semantic actions into the
P3.2-A procedure model, and optionally persists the resulting candidate.  It never
executes actions, calls the planner/LLM, or decides whether an action was successful.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .api import AgentResult, TaskStatus
from .procedure_store import (
    LearnedProcedure,
    ProcedureCondition,
    ProcedureParameter,
    ProcedureProvenance,
    ProcedureSourceType,
    ProcedureStatus,
    ProcedureStep,
    ProcedureStore,
)
from .types import Verdict
from .workflow.models import StepStatus, Workflow, WorkflowStatus


# These are runtime/input-shaped fields. They are parameterized only when an
# explicit parameter schema or a {parameter} placeholder exists; otherwise their
# concrete value is deliberately omitted from the learned procedure.
_INPUT_KEYS = {
    "query", "text", "message", "recipient", "subject", "body", "email",
    "to", "cc", "bcc", "url", "song", "title", "value",
}
_FRAGILE_KEYS = {"x", "y", "x1", "y1", "x2", "y2", "coordinate", "coordinates", "pixel", "pixels", "css", "css_selector", "xpath", "dom_path", "selector", "element_id", "hwnd", "window_handle"}
_SENSITIVE_KEYS = {
    "password", "passwd", "secret", "token", "api_key", "access_token",
    "auth_token", "cookie", "session_token", "bearer", "private_key",
    "security_answer", "one_time_code", "otp", "payment", "card_number",
}
_PLACEHOLDER = re.compile(r"^\{([A-Za-z_][A-Za-z0-9_-]*)\}$")


@dataclass(frozen=True)
class LearningEligibility:
    eligible: bool
    reason: str = ""


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _verification_verdict(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("verdict", "UNKNOWN")).upper()
    if isinstance(value, str):
        return value.upper()
    return "UNKNOWN"


def _has_policy_denial(workflow: Workflow) -> bool:
    for step in workflow.steps:
        state = str(step.policy_state or "").upper()
        if state in {"DENY", "DENIED", "BLOCKED", "POLICY_DENIED"}:
            return True
        category = str(step.failure_category or "").upper()
        if "POLICY_DENIED" in category or category == "PERMISSION_DENIED":
            return True
    return False


def is_learning_eligible(
    workflow_result: Any,
    *,
    workflow: Workflow | Mapping[str, Any] | None = None,
) -> LearningEligibility:
    """Check the existing authoritative completion/verification boundary.

    ``WorkflowStatus.COMPLETED`` plus every step being ``COMPLETED`` and every
    required step having verification ``PASS`` is the workflow-side authority.
    When an AgentResult is supplied, its existing SUCCESS/PASS contract is also
    required.  No second success heuristic is introduced.
    """
    if isinstance(workflow, Mapping):
        try:
            workflow = Workflow.from_json(dict(workflow))
        except Exception:
            return LearningEligibility(False, "missing_workflow_metadata")
    if workflow is None:
        if isinstance(workflow_result, Workflow):
            workflow = workflow_result
        elif isinstance(workflow_result, AgentResult) and workflow_result.outcome and workflow_result.outcome.workflow:
            try:
                workflow = Workflow.from_json(workflow_result.outcome.workflow)
            except Exception:
                workflow = None
    if workflow is None:
        return LearningEligibility(False, "missing_workflow_metadata")

    if isinstance(workflow_result, AgentResult):
        if workflow_result.status is TaskStatus.POLICY_BLOCKED:
            return LearningEligibility(False, "policy_denied")
        if workflow_result.status is TaskStatus.CANCELLED:
            return LearningEligibility(False, "cancelled")
        if workflow_result.status is not TaskStatus.SUCCESS:
            return LearningEligibility(False, "execution_failed")
        if str(workflow_result.verified).upper() != Verdict.PASS.value:
            return LearningEligibility(False, "not_verified")

    if workflow.cancel_requested or workflow.status is WorkflowStatus.CANCELLED:
        return LearningEligibility(False, "cancelled")
    if workflow.status is not WorkflowStatus.COMPLETED:
        reasons = {
            WorkflowStatus.UNKNOWN: "verification_unknown",
            WorkflowStatus.PARTIAL_FAILURE: "partial_workflow",
            WorkflowStatus.FAILED: "execution_failed",
            WorkflowStatus.BLOCKED: "partial_workflow",
            WorkflowStatus.RECOVERING: "recovery_exhausted",
            WorkflowStatus.WAITING_FOR_APPROVAL: "not_verified",
            WorkflowStatus.WAITING_FOR_USER: "not_verified",
        }
        return LearningEligibility(False, reasons.get(workflow.status, "not_verified"))
    if _has_policy_denial(workflow):
        return LearningEligibility(False, "policy_denied")
    if not workflow.steps:
        return LearningEligibility(False, "missing_workflow_metadata")

    for step in workflow.steps:
        if step.status is not StepStatus.COMPLETED:
            if step.status is StepStatus.UNKNOWN:
                return LearningEligibility(False, "verification_unknown")
            if step.status in {StepStatus.FAILED, StepStatus.RECOVERY_REQUIRED, StepStatus.RECOVERING}:
                return LearningEligibility(False, "unrecovered_failure")
            if step.status is StepStatus.CANCELLED:
                return LearningEligibility(False, "cancelled")
            return LearningEligibility(False, "partial_workflow")
        if step.requires_verification and _verification_verdict(step.verification) != Verdict.PASS.value:
            return LearningEligibility(False, "not_verified")
        if str(step.failure_category or "").upper() in {"RECOVERY_EXHAUSTED", "POLICY_DENIED"}:
            return LearningEligibility(False, "recovery_exhausted" if "RECOVERY" in str(step.failure_category).upper() else "policy_denied")
    return LearningEligibility(True, "verified_workflow")


def _explicit_parameters(task_context: Any) -> dict[str, ProcedureParameter]:
    """Read only explicit parameter schemas supplied by an existing caller."""
    raw = _as_mapping(task_context).get("parameters")
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, ProcedureParameter] = {}
    for name, spec in raw.items():
        if not isinstance(spec, Mapping):
            continue
        try:
            p = ProcedureParameter(
                name=str(name), description=str(spec.get("description", "")),
                type=str(spec.get("type", "string")), required=bool(spec.get("required", True)),
                default=None, constraints=dict(spec.get("constraints") or {}),
                sensitive=bool(spec.get("sensitive", False)),
            )
            result[p.name] = p
        except (ValueError, TypeError):
            continue
    return result


def _placeholder(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = _PLACEHOLDER.fullmatch(value.strip())
    return match.group(1) if match else None


def _safe_static(value: Any) -> Any:
    """Return safe semantic constants; never retain runtime-looking input."""
    if isinstance(value, Mapping):
        return {str(k): _safe_static(v) for k, v in value.items() if str(k).casefold() not in _SENSITIVE_KEYS and str(k).casefold() not in _FRAGILE_KEYS}
    if isinstance(value, (list, tuple)):
        return [_safe_static(v) for v in value]
    return value


def _parameterize_arguments(params: Mapping[str, Any], explicit: dict[str, ProcedureParameter]) -> tuple[dict[str, Any], dict[str, ProcedureParameter]]:
    arguments: dict[str, Any] = {}
    discovered = dict(explicit)
    for raw_key, value in params.items():
        key = str(raw_key)
        key_lower = key.casefold()
        if key_lower in _FRAGILE_KEYS:
            continue
        placeholder = _placeholder(value)
        if placeholder:
            p = discovered.get(placeholder)
            if p is None:
                p = ProcedureParameter(name=placeholder, type="string", required=True, sensitive=placeholder.casefold() in _SENSITIVE_KEYS)
                discovered[placeholder] = p
            arguments[key] = "{" + placeholder + "}"
            continue
        if key_lower in _SENSITIVE_KEYS:
            # Keep the parameter schema if explicitly supplied, but never its value.
            p = discovered.get(key)
            if p is None:
                p = ProcedureParameter(name=key, type="string", required=True, sensitive=True)
                discovered[key] = p
            arguments[key] = "{" + p.name + "}"
            continue
        if key_lower in _INPUT_KEYS:
            p = discovered.get(key)
            if p is not None:
                arguments[key] = "{" + p.name + "}"
            else:
                # Concrete task input is not procedure definition data. Omit it
                # rather than guessing that every string is a reusable parameter.
                continue
            continue
        arguments[key] = _safe_static(value)
    return arguments, discovered


def _procedure_name(workflow: Workflow) -> str:
    kinds = [step.action.kind.strip().lower().replace("_", ".") for step in workflow.steps]
    if len(kinds) == 1:
        return kinds[0]
    return "workflow." + ".".join(kinds)


def _description(workflow: Workflow) -> str:
    return "Verified workflow procedure: " + ", ".join(step.action.kind for step in workflow.steps)


def _conditions(workflow: Workflow) -> tuple[ProcedureCondition, ...]:
    conditions: list[ProcedureCondition] = []
    for step in workflow.steps:
        verification = _as_mapping(step.verification)
        checks = verification.get("checks")
        if not isinstance(checks, list):
            continue
        for index, check in enumerate(checks):
            if not isinstance(check, Mapping) or str(check.get("verdict", "")).upper() != "PASS":
                continue
            name = str(check.get("name", "")).strip()
            if not name:
                continue
            condition_id = f"{step.step_id}:check:{index}"
            conditions.append(ProcedureCondition(condition_id, name, {"step_id": step.step_id, "source": "independent_verification"}))
    return tuple(conditions)


def extract_procedure_candidate(
    workflow_result: Any,
    task_context: Any = None,
    *,
    workflow: Workflow | Mapping[str, Any] | None = None,
) -> LearnedProcedure | None:
    """Pure deterministic conversion. Returns ``None`` unless fully verified."""
    eligibility = is_learning_eligible(workflow_result, workflow=workflow)
    if not eligibility.eligible:
        return None

    if isinstance(workflow, Mapping):
        workflow = Workflow.from_json(dict(workflow))
    if workflow is None:
        if isinstance(workflow_result, Workflow):
            workflow = workflow_result
        else:
            payload = workflow_result.outcome.workflow if isinstance(workflow_result, AgentResult) and workflow_result.outcome else None
            workflow = Workflow.from_json(payload) if isinstance(payload, Mapping) else None
    if workflow is None:
        return None

    explicit = _explicit_parameters(task_context)
    parameters: dict[str, ProcedureParameter] = dict(explicit)
    steps: list[ProcedureStep] = []
    required_capabilities: set[str] = set()
    for step in workflow.steps:
        arguments, discovered = _parameterize_arguments(step.action.params, parameters)
        parameters.update(discovered)
        # Action kinds are DEIMOS' existing semantic capabilities. Resource
        # keys are runtime bindings, not new capability identifiers.
        capabilities = (step.action.kind,)
        required_capabilities.add(step.action.kind)
        steps.append(ProcedureStep(
            step_id=step.step_id,
            order=len(steps),
            action_name=step.action.kind,
            arguments=arguments,
            description=step.intent,
            dependencies=tuple(step.dependencies),
            required_capabilities=capabilities,
            expected_observation={},
            expected_verification={"verdict": "PASS"} if step.requires_verification else {},
            metadata={"source_workflow_step_id": step.step_id},
        ))

    source_task_id = getattr(workflow_result, "task_id", None) or workflow.workflow_id
    source_session_id = _as_mapping(task_context).get("session_id")
    provenance = ProcedureProvenance(
        ProcedureSourceType.VERIFIED_WORKFLOW,
        source_task_id=str(source_task_id) if source_task_id else None,
        source_session_id=str(source_session_id) if source_session_id else None,
    )
    name = _procedure_name(workflow)
    signature = {
        "name": name,
        "steps": [{"action": s.action_name, "arguments": s.arguments, "dependencies": list(s.dependencies)} for s in steps],
    }
    procedure_id = "proc-" + hashlib.sha256(json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    resource_requirements = {
        str(key): {"site": binding.site, "reuse": binding.reuse}
        for key, binding in workflow.resources.items()
    }
    return LearnedProcedure(
        procedure_id=procedure_id,
        name=name,
        description=_description(workflow),
        trigger_pattern=name,
        status=ProcedureStatus.CANDIDATE,
        version=1,
        parameters=tuple(sorted(parameters.values(), key=lambda p: p.name.casefold())),
        steps=tuple(steps),
        preconditions=(),
        postconditions=_conditions(workflow),
        required_capabilities=tuple(sorted(required_capabilities)),
        resource_requirements=resource_requirements,
        confidence=0.5,
        importance=0.5,
        provenance=provenance,
        source_task_id=provenance.source_task_id,
        source_session_id=provenance.source_session_id,
        metadata={"learning": "deterministic_verified_workflow", "source_workflow_id": workflow.workflow_id},
    )


def store_procedure_candidate(
    candidate: LearnedProcedure,
    store: ProcedureStore,
) -> LearnedProcedure:
    """Persist a candidate without silently overwriting an existing procedure."""
    existing = store.get_procedure(candidate.procedure_id)
    if existing is not None:
        return existing
    try:
        return store.create_procedure(candidate)
    except Exception:
        # Same semantic name/version may already exist under a different ID.
        # Re-check deterministically; do not replace the existing row.
        existing = store.get_procedure_version(candidate.name, candidate.version)
        if existing is not None:
            return existing
        raise


def learn_verified_workflow(
    workflow_result: Any,
    task_context: Any = None,
    *,
    store: ProcedureStore,
) -> LearnedProcedure | None:
    """Explicit persistence boundary used by callers after workflow completion."""
    candidate = extract_procedure_candidate(workflow_result, task_context)
    if candidate is None:
        return None
    return store_procedure_candidate(candidate, store)


__all__ = [
    "LearningEligibility",
    "is_learning_eligible",
    "extract_procedure_candidate",
    "store_procedure_candidate",
    "learn_verified_workflow",
]
