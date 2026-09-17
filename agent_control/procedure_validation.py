"""P3.2-E learned-procedure validation and lifecycle authority.

This module is deliberately separate from retrieval and execution.  It validates
persisted procedure definitions and explicit evidence, then performs only the
existing ProcedureStore lifecycle transitions.  It never executes a procedure,
invokes a BrowserSkill, or treats Policy action approval as procedure activation.

Activation is an explicit policy decision.  The safe default requires one
independently verified workflow evidence item *and* explicit human activation
approval.  The evidence threshold is configurable; the lifecycle mechanics do
not contain a success-count heuristic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from .procedure_store import (
    LearnedProcedure,
    ProcedureSourceType,
    ProcedureStatus,
    ProcedureStore,
    ProcedureValidationEventType,
)

_PLACEHOLDER = re.compile(r"^\{([A-Za-z_][A-Za-z0-9_-]*)\}$")


class ProcedureValidationDecision(str, Enum):
    PROMOTE = "PROMOTE"
    KEEP_CANDIDATE = "KEEP_CANDIDATE"
    INVALIDATE = "INVALIDATE"


class ProcedureHealthDecision(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADE = "DEGRADE"
    INVALIDATE = "INVALIDATE"
    REACTIVATE = "REACTIVATE"
    NO_CHANGE = "NO_CHANGE"


@dataclass(frozen=True)
class ProcedureActivationPolicy:
    """Explicit, deterministic trust policy for lifecycle promotion.

    ``min_verified_evidence`` counts explicit independently verified workflow
    evidence supplied to the validator.  It is intentionally not read from the
    execution success counter: a historical counter alone does not establish
    independent verification provenance.
    """

    min_verified_evidence: int = 1
    require_human_approval: bool = True
    require_verified_workflow_provenance: bool = True
    require_reusable: bool = True
    require_semantic_actions: bool = True
    require_capabilities: bool = True
    reject_failure_evidence: bool = True
    reject_unknown_evidence: bool = True

    def __post_init__(self) -> None:
        if self.min_verified_evidence < 1:
            raise ValueError("min_verified_evidence must be >= 1")


@dataclass(frozen=True)
class ProcedureValidationEvidence:
    """Evidence supplied by an existing workflow/runtime boundary.

    The validator does not manufacture execution or verification results.  The
    caller must explicitly provide the evidence it wants considered.
    """

    verified_evidence: int = 0
    independent_verification_pass: bool = False
    policy_compatible: bool = False
    recovery_clean: bool = False
    reusable: bool = False
    semantic_actions_valid: bool = False
    capabilities_available: bool = False
    human_approved: bool = False
    failure_evidence: bool = False
    unknown_evidence: bool = False
    source_task_id: str | None = None
    source_session_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verified_evidence": self.verified_evidence,
            "independent_verification_pass": self.independent_verification_pass,
            "policy_compatible": self.policy_compatible,
            "recovery_clean": self.recovery_clean,
            "reusable": self.reusable,
            "semantic_actions_valid": self.semantic_actions_valid,
            "capabilities_available": self.capabilities_available,
            "human_approved": self.human_approved,
            "failure_evidence": self.failure_evidence,
            "unknown_evidence": self.unknown_evidence,
            "source_task_id": self.source_task_id,
            "source_session_id": self.source_session_id,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ProcedureValidationResult:
    procedure_id: str
    decision: ProcedureValidationDecision
    reasons: tuple[str, ...] = ()
    transitioned_from: ProcedureStatus | None = None
    transitioned_to: ProcedureStatus | None = None
    evidence: ProcedureValidationEvidence | None = None

    @property
    def promoted(self) -> bool:
        return self.decision is ProcedureValidationDecision.PROMOTE


@dataclass(frozen=True)
class ProcedureHealthPolicy:
    """Configurable health thresholds for explicit lifecycle assessment."""

    degrade_after_failures: int | None = 1
    degrade_after_unknowns: int | None = 1
    invalidate_after_failures: int | None = None
    invalidate_after_unknowns: int | None = None
    require_explicit_invalid_evidence: bool = True

    def __post_init__(self) -> None:
        for value, name in (
            (self.degrade_after_failures, "degrade_after_failures"),
            (self.degrade_after_unknowns, "degrade_after_unknowns"),
            (self.invalidate_after_failures, "invalidate_after_failures"),
            (self.invalidate_after_unknowns, "invalidate_after_unknowns"),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 or None")


def _has_placeholder(value: Any, parameter_names: set[str]) -> tuple[bool, set[str], list[str]]:
    used: set[str] = set()
    invalid: list[str] = []
    if isinstance(value, str):
        matches = re.findall(r"\{([A-Za-z_][A-Za-z0-9_-]*)\}", value)
        for name in matches:
            if name not in parameter_names:
                invalid.append(f"unresolved_parameter:{name}")
            else:
                used.add(name)
        return bool(matches), used, invalid
    if isinstance(value, Mapping):
        found = False
        for key, item in value.items():
            f, u, bad = _has_placeholder(key, parameter_names)
            found = found or f
            used.update(u)
            invalid.extend(bad)
            f, u, bad = _has_placeholder(item, parameter_names)
            found = found or f
            used.update(u)
            invalid.extend(bad)
        return found, used, invalid
    if isinstance(value, (list, tuple)):
        found = False
        for item in value:
            f, u, bad = _has_placeholder(item, parameter_names)
            found = found or f
            used.update(u)
            invalid.extend(bad)
        return found, used, invalid
    return False, used, invalid


def validate_procedure_definition(
    procedure: LearnedProcedure,
    *,
    available_capabilities: Iterable[str] = (),
    semantic_actions_valid: bool | None = None,
) -> tuple[str, ...]:
    """Validate the persisted definition without executing it.

    This is intentionally independent of P3.2-C binding/current-state checks.
    It validates reusable representation, not whether a particular invocation is
    applicable right now.
    """
    reasons: list[str] = []
    try:
        procedure.validate()
    except Exception as exc:
        return (f"malformed_procedure:{type(exc).__name__}",)

    if not procedure.provenance:
        reasons.append("provenance_missing")
    if procedure.provenance.source_type is not ProcedureSourceType.VERIFIED_WORKFLOW:
        reasons.append("provenance_not_verified_workflow")
    if not procedure.provenance.source_task_id:
        reasons.append("provenance_source_task_missing")
    if not procedure.steps:
        reasons.append("procedure_has_no_steps")

    parameter_names = {p.name for p in procedure.parameters}
    used_parameters: set[str] = set()
    used_in_arguments: set[str] = set()
    for step in procedure.steps:
        if not step.action_name.strip():
            reasons.append(f"semantic_action_missing:{step.step_id}")
        _, used, bad = _has_placeholder(step.arguments, parameter_names)
        used_parameters.update(used)
        used_in_arguments.update(used)
        reasons.extend(bad)
        if not step.expected_verification or str(step.expected_verification.get("verdict", "")).upper() != "PASS":
            reasons.append(f"independent_verification_missing:{step.step_id}")

    _, trigger_parameters, bad = _has_placeholder(procedure.trigger_pattern, parameter_names)
    used_parameters.update(trigger_parameters)
    reasons.extend(bad)

    required = {p.name for p in procedure.parameters if p.required}
    missing_parameter_usage = required - used_parameters
    reasons.extend(f"parameter_not_used:{name}" for name in sorted(missing_parameter_usage))
    missing_parameter_arguments = required - used_in_arguments
    reasons.extend(f"parameter_not_bound_into_action:{name}" for name in sorted(missing_parameter_arguments))

    # A parameterized learned procedure must expose its reusable inputs in its
    # trigger.  This rejects concrete one-off procedures such as a fixed song
    # definition while preserving static workflows that are explicitly marked
    # reusable by the evidence supplied to validate_and_promote().
    if procedure.parameters and not trigger_parameters:
        reasons.append("not_generalizable:no_trigger_parameters")
    elif not procedure.parameters:
        # A learned procedure with no declared parameters cannot demonstrate
        # reusable/generalizable behavior from its persisted definition alone.
        # Keep this definition-level validator consistent with the activation
        # gate in _structural_reusable(): concrete one-off workflows must not
        # look structurally valid merely because their capabilities exist.
        reasons.append("not_generalizable:no_parameters")

    available = {str(x).casefold() for x in available_capabilities}
    required_caps = {str(x).casefold() for x in procedure.required_capabilities}
    for step in procedure.steps:
        required_caps.update(str(x).casefold() for x in step.required_capabilities)
    if required_caps and not required_caps.issubset(available):
        reasons.append("required_capabilities_unavailable")

    if semantic_actions_valid is False:
        reasons.append("semantic_actions_invalid")

    return tuple(dict.fromkeys(reasons))


def _structural_reusable(procedure: LearnedProcedure) -> bool:
    """Conservative representation-level reusability check."""
    if not procedure.steps:
        return False
    params = {p.name for p in procedure.parameters}
    for step in procedure.steps:
        _, _, bad = _has_placeholder(step.arguments, params)
        if bad:
            return False
    if procedure.parameters:
        _, trigger_params, bad = _has_placeholder(procedure.trigger_pattern, params)
        required = {p.name for p in procedure.parameters if p.required}
        return not bad and bool(trigger_params) and required.issubset(trigger_params)
    # A parameterless procedure is reusable only when the caller explicitly
    # supplies reusable evidence.  The definition alone cannot prove that a
    # concrete one-off workflow is generalizable.
    return False


def _inspect_skill_registry(procedure: LearnedProcedure, skill_registry: Any | None) -> tuple[bool, bool, tuple[str, ...]]:
    """Inspect existing approved skill dispatchability without executing it."""
    if skill_registry is None:
        return False, False, ()
    finder = getattr(skill_registry, "find_for_action", None)
    if not callable(finder):
        return False, False, ()
    try:
        for step in procedure.steps:
            if finder(step.action_name) is None:
                return False, False, ()
    except Exception:
        return False, False, ()
    caps = getattr(skill_registry, "capabilities", None)
    try:
        advertised = tuple(caps()) if callable(caps) else ()
    except Exception:
        advertised = ()
    return True, True, advertised


def validate_and_promote(
    procedure_id: str,
    *,
    store: ProcedureStore,
    evidence: ProcedureValidationEvidence,
    policy: ProcedureActivationPolicy | None = None,
    available_capabilities: Iterable[str] = (),
    semantic_actions_valid: bool | None = None,
    skill_registry: Any | None = None,
) -> ProcedureValidationResult:
    """Validate a candidate and, only when policy is satisfied, activate it."""
    policy = policy or ProcedureActivationPolicy()
    procedure = store.get_procedure(procedure_id)
    if procedure is None:
        raise KeyError(procedure_id)

    if procedure.status not in {ProcedureStatus.CANDIDATE, ProcedureStatus.VALIDATING}:
        return ProcedureValidationResult(
            procedure_id, ProcedureValidationDecision.KEEP_CANDIDATE,
            (f"invalid_validation_start_status:{procedure.status.value}",),
            procedure.status, procedure.status, evidence,
        )

    started_from = procedure.status
    registry_semantic_valid, registry_capabilities_valid, registry_capabilities = _inspect_skill_registry(procedure, skill_registry)
    if semantic_actions_valid is None:
        semantic_actions_valid = registry_semantic_valid
    if not tuple(available_capabilities) and registry_capabilities_valid:
        available_capabilities = registry_capabilities
    if procedure.status is ProcedureStatus.CANDIDATE:
        store.set_status(procedure_id, ProcedureStatus.VALIDATING)
        store.record_validation_event(
            procedure_id, ProcedureValidationEventType.VALIDATION_STARTED,
            result="STARTED", reason="candidate_validation_started",
            source_task_id=evidence.source_task_id, source_session_id=evidence.source_session_id,
            evidence=evidence.as_dict(),
        )

    reasons = list(validate_procedure_definition(
        procedure,
        available_capabilities=available_capabilities,
        semantic_actions_valid=semantic_actions_valid,
    ))

    if policy.require_verified_workflow_provenance and (
        procedure.provenance.source_type is not ProcedureSourceType.VERIFIED_WORKFLOW
        or not procedure.provenance.source_task_id
    ):
        reasons.append("verified_workflow_provenance_required")
    if policy.require_reusable and not (evidence.reusable and _structural_reusable(procedure)):
        reasons.append("procedure_not_reusable")
    if policy.require_semantic_actions and not evidence.semantic_actions_valid:
        reasons.append("semantic_actions_not_verified")
    if policy.require_capabilities and not evidence.capabilities_available:
        reasons.append("capabilities_not_verified")
    if not evidence.independent_verification_pass:
        reasons.append("independent_verification_required")
    if not evidence.policy_compatible:
        reasons.append("policy_compatibility_not_verified")
    if not evidence.recovery_clean:
        reasons.append("recovery_clean_evidence_required")
    if evidence.verified_evidence < policy.min_verified_evidence:
        reasons.append("insufficient_verified_evidence")
    if policy.reject_failure_evidence and evidence.failure_evidence:
        reasons.append("failure_evidence_present")
    if policy.reject_unknown_evidence and evidence.unknown_evidence:
        reasons.append("unknown_evidence_present")
    if policy.require_human_approval and not evidence.human_approved:
        reasons.append("human_activation_approval_required")

    reasons = list(dict.fromkeys(reasons))
    invalid_definition_reasons = {
        "malformed_procedure", "provenance_missing", "provenance_not_verified_workflow",
        "provenance_source_task_missing", "procedure_has_no_steps", "semantic_action_missing",
        "unresolved_parameter", "parameter_not_used", "parameter_not_bound_into_action", "not_generalizable",
        "required_capabilities_unavailable", "semantic_actions_invalid", "procedure_not_reusable",
        "independent_verification_missing",
    }
    permanently_invalid = any(
        any(reason.startswith(prefix) for prefix in invalid_definition_reasons)
        for reason in reasons
    )

    if reasons:
        target = ProcedureStatus.INVALIDATED if permanently_invalid else ProcedureStatus.CANDIDATE
        current = store.get_procedure(procedure_id)
        if current is not None and current.status is ProcedureStatus.VALIDATING:
            store.set_status(procedure_id, target)
        event_type = ProcedureValidationEventType.VALIDATION_FAILED
        store.record_validation_event(
            procedure_id, event_type, result="FAIL", reason=";".join(reasons),
            source_task_id=evidence.source_task_id, source_session_id=evidence.source_session_id,
            evidence=evidence.as_dict(),
        )
        return ProcedureValidationResult(
            procedure_id,
            ProcedureValidationDecision.INVALIDATE if permanently_invalid else ProcedureValidationDecision.KEEP_CANDIDATE,
            tuple(reasons), started_from, target, evidence,
        )

    store.set_status(procedure_id, ProcedureStatus.ACTIVE)
    store.record_validation_event(
        procedure_id, ProcedureValidationEventType.VALIDATION_PASSED,
        result="PASS", reason="activation_criteria_satisfied",
        source_task_id=evidence.source_task_id, source_session_id=evidence.source_session_id,
        evidence=evidence.as_dict(),
    )
    store.record_validation_event(
        procedure_id, ProcedureValidationEventType.PROMOTION_GRANTED,
        result="ACTIVE", reason="explicit_activation_policy_satisfied",
        source_task_id=evidence.source_task_id, source_session_id=evidence.source_session_id,
        evidence={**evidence.as_dict(), "policy": policy.__dict__},
    )
    return ProcedureValidationResult(
        procedure_id, ProcedureValidationDecision.PROMOTE,
        ("activation_criteria_satisfied",), started_from, ProcedureStatus.ACTIVE, evidence,
    )


def assess_health(
    procedure_id: str,
    *,
    store: ProcedureStore,
    policy: ProcedureHealthPolicy | None = None,
    invalid_evidence: bool = False,
    revalidation_passed: bool = False,
) -> ProcedureHealthDecision:
    """Apply explicit health evidence to an existing ACTIVE/DEGRADED procedure.

    This function only uses persisted counters and explicit invalid/revalidation
    evidence.  It never executes or retrieves a procedure.
    """
    policy = policy or ProcedureHealthPolicy()
    procedure = store.get_procedure(procedure_id)
    if procedure is None:
        raise KeyError(procedure_id)

    if procedure.status is ProcedureStatus.ACTIVE:
        if invalid_evidence and policy.require_explicit_invalid_evidence:
            store.set_status(procedure_id, ProcedureStatus.INVALIDATED)
            store.record_validation_event(
                procedure_id, ProcedureValidationEventType.INVALIDATION_DETECTED,
                result="INVALIDATED", reason="explicit_invalid_evidence",
                evidence={"invalid_evidence": True},
            )
            return ProcedureHealthDecision.INVALIDATE
        if (
            policy.invalidate_after_failures is not None and procedure.failure_count >= policy.invalidate_after_failures
        ) or (
            policy.invalidate_after_unknowns is not None and procedure.unknown_count >= policy.invalidate_after_unknowns
        ):
            store.set_status(procedure_id, ProcedureStatus.INVALIDATED)
            store.record_validation_event(
                procedure_id, ProcedureValidationEventType.INVALIDATION_DETECTED,
                result="INVALIDATED", reason="health_threshold_exceeded",
                evidence={"failure_count": procedure.failure_count, "unknown_count": procedure.unknown_count},
            )
            return ProcedureHealthDecision.INVALIDATE
        if (
            policy.degrade_after_failures is not None and procedure.failure_count >= policy.degrade_after_failures
        ) or (
            policy.degrade_after_unknowns is not None and procedure.unknown_count >= policy.degrade_after_unknowns
        ):
            store.set_status(procedure_id, ProcedureStatus.DEGRADED)
            store.record_validation_event(
                procedure_id, ProcedureValidationEventType.DEGRADATION_DETECTED,
                result="DEGRADED", reason="health_threshold_exceeded",
                evidence={"failure_count": procedure.failure_count, "unknown_count": procedure.unknown_count},
            )
            return ProcedureHealthDecision.DEGRADE
        return ProcedureHealthDecision.HEALTHY

    if procedure.status is ProcedureStatus.DEGRADED:
        if invalid_evidence:
            store.set_status(procedure_id, ProcedureStatus.INVALIDATED)
            store.record_validation_event(
                procedure_id, ProcedureValidationEventType.INVALIDATION_DETECTED,
                result="INVALIDATED", reason="explicit_invalid_evidence",
                evidence={"invalid_evidence": True},
            )
            return ProcedureHealthDecision.INVALIDATE
        if revalidation_passed:
            store.set_status(procedure_id, ProcedureStatus.ACTIVE)
            store.record_validation_event(
                procedure_id, ProcedureValidationEventType.REVALIDATION_PASSED,
                result="ACTIVE", reason="explicit_revalidation_passed",
                evidence={"revalidation_passed": True},
            )
            return ProcedureHealthDecision.REACTIVATE
        return ProcedureHealthDecision.NO_CHANGE

    return ProcedureHealthDecision.NO_CHANGE


__all__ = [
    "ProcedureActivationPolicy", "ProcedureHealthPolicy", "ProcedureValidationDecision",
    "ProcedureHealthDecision", "ProcedureValidationEvidence", "ProcedureValidationResult",
    "validate_procedure_definition", "validate_and_promote", "assess_health",
]
