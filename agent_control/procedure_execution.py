"""P3.2-D safe execution of an already validated procedure match.

This module is intentionally a thin execution boundary.  It does not retrieve
or bind procedures, promote their lifecycle status, or invent a verifier.  A
ProcedureMatch must already be BOUND and APPLICABLE; execution then goes through
the existing Policy -> Skill adapter -> executor -> verifier path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from .policy import Decision, Policy
from .procedure_retrieval import BindingStatus, ProcedureMatch, ValidationStatus
from .types import Action, ActionResult, Check, FailureClass, Verdict, VerificationResult


class ProcedureExecutionStatus(str, Enum):
    NOT_EXECUTED = "NOT_EXECUTED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ProcedureExecutionResult:
    procedure_id: str
    execution_status: ProcedureExecutionStatus
    verification_status: Verdict
    failure_reason: str | None = None
    verification_evidence: VerificationResult | None = None
    action_results: tuple[ActionResult, ...] = ()
    fresh_observation: Any | None = None

    @property
    def verified_success(self) -> bool:
        return (
            self.execution_status is ProcedureExecutionStatus.EXECUTED
            and self.verification_status is Verdict.PASS
        )

    @property
    def success(self) -> bool:
        """Compatibility convenience: only independently verified PASS counts."""
        return self.verified_success


_PLACEHOLDER = re.compile(r"^\{([A-Za-z_][A-Za-z0-9_-]*)\}$")


def _resolve_value(value: Any, bindings: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        match = _PLACEHOLDER.fullmatch(value.strip())
        if match:
            name = match.group(1)
            if name not in bindings:
                raise ValueError(f"unresolved_parameter:{name}")
            return bindings[name]
        # Procedure definitions may contain embedded placeholders only when the
        # whole semantic argument is the parameter. Do not perform text
        # interpolation that could silently alter URLs, messages, or commands.
        return value
    if isinstance(value, Mapping):
        return {str(k): _resolve_value(v, bindings) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(v, bindings) for v in value]
    if isinstance(value, tuple):
        return tuple(_resolve_value(v, bindings) for v in value)
    return value


def _redact_action_params(action: Action, match: ProcedureMatch) -> dict[str, Any]:
    sensitive = {
        p.parameter.name.casefold()
        for p in match.binding.bindings
        if p.parameter.sensitive
    }
    result: dict[str, Any] = {}
    for key, value in action.params.items():
        result[key] = "<redacted>" if str(key).casefold() in sensitive else value
    return result


def _failure(
    match: ProcedureMatch,
    reason: str,
    *,
    status: ProcedureExecutionStatus = ProcedureExecutionStatus.NOT_EXECUTED,
    verification: Verdict = Verdict.UNKNOWN,
) -> ProcedureExecutionResult:
    return ProcedureExecutionResult(
        procedure_id=match.procedure.procedure_id,
        execution_status=status,
        verification_status=verification,
        failure_reason=reason,
    )


def _build_actions(match: ProcedureMatch) -> tuple[Action, ...]:
    """Turn semantic procedure steps into core Actions using bound values only."""
    if not match.procedure.steps:
        raise ValueError("procedure_has_no_steps")

    bindings = match.binding.values
    actions: list[Action] = []
    seen: set[str] = set()
    for step in sorted(match.procedure.steps, key=lambda s: s.order):
        if step.step_id in seen:
            raise ValueError(f"duplicate_step:{step.step_id}")
        seen.add(step.step_id)
        if not step.action_name.strip():
            raise ValueError(f"malformed_step:{step.step_id}:missing_action")
        try:
            params = _resolve_value(step.arguments, bindings)
        except ValueError:
            raise
        if not isinstance(params, dict):
            raise ValueError(f"malformed_step:{step.step_id}:arguments_not_mapping")
        actions.append(
            Action(
                kind=step.action_name,
                params=params,
                consequential=True,
                rationale=f"learned procedure {match.procedure.name} step {step.step_id}",
                requires_fresh_observation=True,
                verification_required=True,
            )
        )
    return tuple(actions)


def _fresh_observe(skill: Any) -> Any:
    """Read current skill state without invoking an action executor."""
    backend = getattr(skill, "_backend", None)
    target = backend if backend is not None else skill
    observe = getattr(target, "observe", None)
    if callable(observe):
        return observe()

    # BrowserSkill always exposes its backend.  The fallback below is only for
    # lightweight skills whose verifier already owns a fresh read interface.
    current_url = getattr(target, "current_url", None)
    page_title = getattr(target, "page_title", None)
    page_text = getattr(target, "page_text", None)
    if callable(current_url) or callable(page_title) or callable(page_text):
        return {
            "url": current_url() if callable(current_url) else None,
            "title": page_title() if callable(page_title) else None,
            "text": page_text() if callable(page_text) else None,
        }
    return None


def _normalize_verification(value: Any) -> tuple[Verdict, VerificationResult]:
    if isinstance(value, VerificationResult):
        return value.verdict, value
    status = str(getattr(value, "status", "")).upper()
    ok = bool(getattr(value, "ok", False))
    detail = str(getattr(value, "detail", "") or "")
    if status == Verdict.PASS.value or (ok and not status):
        verdict = Verdict.PASS
    elif status == Verdict.UNKNOWN.value:
        verdict = Verdict.UNKNOWN
    else:
        verdict = Verdict.FAIL
    return verdict, VerificationResult(
        label="procedure-independent-verification",
        checks=[Check(
            name="independent_verification",
            verdict=verdict,
            evidence={"status": status},
            reason=detail,
        )],
    )


def execute_procedure(
    match: ProcedureMatch,
    *,
    policy: Policy,
    skills: Any,
    trace: Any | None = None,
    max_steps: int = 8,
) -> ProcedureExecutionResult:
    """Execute an ACTIVE, bound, currently applicable procedure safely.

    The function deliberately does not mutate procedure lifecycle state.  It
    uses the existing policy and skill executor/verifier interfaces and performs
    a fresh observation after the final action before asking the existing skill
    verifier for independent evidence.
    """
    if match is None:
        raise ValueError("procedure_match_required")

    procedure = match.procedure
    if procedure.status.value != "ACTIVE":
        return _failure(match, "procedure_not_executable:status_not_active")
    if match.binding.status is not BindingStatus.BOUND:
        return _failure(match, f"parameter_binding_{match.binding.status.value.lower()}")
    if match.validation.status is not ValidationStatus.APPLICABLE:
        reason = match.validation.reasons[0] if match.validation.reasons else "current_state_not_applicable"
        return _failure(match, reason if match.validation.status is ValidationStatus.NOT_APPLICABLE else "current_state_unknown")
    if not match.execution_allowed:
        return _failure(match, "execution_not_allowed")
    if skills is None:
        return _failure(match, "skill_registry_missing")

    try:
        actions = _build_actions(match)
    except ValueError as exc:
        return _failure(match, str(exc))
    if len(actions) > max(1, int(max_steps)):
        return _failure(match, "procedure_step_budget_exceeded")

    action_results: list[ActionResult] = []
    last_skill: Any | None = None
    last_skill_action: Any | None = None
    last_execution: Any | None = None

    for action in actions:
        # The same policy gate used by normal DEIMOS actions remains authoritative.
        try:
            decision, reason = policy.check(action)
        except Exception as exc:
            if trace is not None and hasattr(trace, "emit"):
                trace.emit("procedure_policy", decision="ERROR")
            return _failure(match, f"policy_check_failed:{type(exc).__name__}")
        if trace is not None and hasattr(trace, "emit"):
            trace.emit("procedure_policy", decision=decision.value)
        if decision is Decision.CONFIRM:
            return _failure(match, "approval_required")
        if decision is not Decision.ALLOW:
            return _failure(match, "policy_denied")

        find_for_action = getattr(skills, "find_for_action", None)
        if not callable(find_for_action):
            return _failure(match, "skill_registry_invalid")
        try:
            skill = find_for_action(action.kind)
        except Exception as exc:
            return _failure(match, f"skill_resolution_failed:{type(exc).__name__}")
        if skill is None:
            return _failure(match, f"unsupported_action:{action.kind}")

        try:
            adapt = getattr(skill, "adapt_action")
            executor_factory = getattr(skill, "executor")
            verifier_factory = getattr(skill, "verifier")
            skill_action = adapt(action)
            executor = executor_factory()
            execution = executor.execute(skill_action)
            if trace is not None and hasattr(trace, "emit"):
                trace.emit("procedure_execution_action", result="SUCCESS")
        except Exception as exc:
            return ProcedureExecutionResult(
                procedure_id=procedure.procedure_id,
                execution_status=ProcedureExecutionStatus.FAILED,
                verification_status=Verdict.UNKNOWN,
                failure_reason=f"execution_failed:{type(exc).__name__}",
                action_results=tuple(action_results),
            )

        ok = bool(getattr(execution, "ok", False))
        action_result = ActionResult(
            action=action,
            ok=ok,
            detail={"procedure_id": procedure.procedure_id},
            error=getattr(execution, "error", None),
            failure_class=getattr(execution, "failure_class", None),
        )
        action_results.append(action_result)
        if not ok:
            return ProcedureExecutionResult(
                procedure_id=procedure.procedure_id,
                execution_status=ProcedureExecutionStatus.FAILED,
                verification_status=Verdict.UNKNOWN,
                failure_reason="execution_failed",
                action_results=tuple(action_results),
            )

        last_skill = skill
        last_skill_action = skill_action
        last_execution = execution

    # No executor return value is treated as proof. Acquire a fresh read before
    # the independent verifier is asked to establish the postcondition.
    if trace is not None and hasattr(trace, "emit"):
        trace.emit("procedure_fresh_observation_start")
    try:
        fresh = _fresh_observe(last_skill)
    except Exception as exc:
        fresh = None
        if trace is not None and hasattr(trace, "note"):
            trace.note("procedure_fresh_observation_failed", error=type(exc).__name__)

    if trace is not None and hasattr(trace, "emit"):
        trace.emit(
            "procedure_execution_observation",
            procedure_id=procedure.procedure_id,
            freshness="FRESH" if fresh is not None else "UNKNOWN",
        )

    try:
        verifier = last_skill.verifier()
        verification_raw = verifier.verify(last_skill_action, last_execution)
        verdict, verification = _normalize_verification(verification_raw)
    except Exception as exc:
        verification = VerificationResult(
            label="procedure-independent-verification",
            checks=[Check(
                name="independent_verification",
                verdict=Verdict.UNKNOWN,
                reason=f"verification_exception:{type(exc).__name__}",
            )],
        )
        verdict = Verdict.UNKNOWN

    if trace is not None and hasattr(trace, "emit"):
        trace.emit(
            "procedure_execution_verification",
            procedure_id=procedure.procedure_id,
            verification=verdict.value,
        )

    if verdict is Verdict.PASS:
        status = ProcedureExecutionStatus.EXECUTED
        reason = None
    elif verdict is Verdict.FAIL:
        status = ProcedureExecutionStatus.FAILED
        reason = "verification_failed"
    else:
        status = ProcedureExecutionStatus.EXECUTED
        reason = "verification_unknown"

    return ProcedureExecutionResult(
        procedure_id=procedure.procedure_id,
        execution_status=status,
        verification_status=verdict,
        failure_reason=reason,
        verification_evidence=verification,
        action_results=tuple(action_results),
        fresh_observation=fresh,
    )


__all__ = ["ProcedureExecutionStatus", "ProcedureExecutionResult", "execute_procedure"]
