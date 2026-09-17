"""P3.2-C deterministic procedure retrieval, binding, and state validation.

This module is deliberately read-only.  It loads persisted procedure candidates,
semantically matches them to a current task, binds only invocation parameters
that are actually present, and validates semantic preconditions against a
current observation.  It never executes a procedure or changes its lifecycle.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .fast_interaction import classify_fast
from .procedure_store import (
    LearnedProcedure,
    ProcedureCondition,
    ProcedureParameter,
    ProcedureStatus,
    ProcedureStore,
)
from .types import Observation


class BindingStatus(str, Enum):
    BOUND = "BOUND"
    MISSING = "MISSING"
    INVALID = "INVALID"
    AMBIGUOUS = "AMBIGUOUS"


class ValidationStatus(str, Enum):
    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ParameterBinding:
    parameter: ProcedureParameter
    value: Any = None
    status: BindingStatus = BindingStatus.MISSING
    reason: str = ""

    @property
    def safe_value(self) -> Any:
        return "<redacted>" if self.parameter.sensitive and self.status is BindingStatus.BOUND else self.value


@dataclass(frozen=True)
class BindingResult:
    status: BindingStatus
    bindings: tuple[ParameterBinding, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def values(self) -> dict[str, Any]:
        return {
            b.parameter.name: b.value
            for b in self.bindings
            if b.status is BindingStatus.BOUND
        }

    def debug_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "bindings": {
                b.parameter.name: b.safe_value for b in self.bindings
            },
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ConditionResult:
    condition: ProcedureCondition
    status: ValidationStatus
    reason: str


@dataclass(frozen=True)
class ProcedureValidationResult:
    status: ValidationStatus
    reasons: tuple[str, ...] = ()
    matched_conditions: tuple[str, ...] = ()
    failed_conditions: tuple[str, ...] = ()
    condition_results: tuple[ConditionResult, ...] = ()


@dataclass(frozen=True)
class ProcedureMatch:
    procedure: LearnedProcedure
    binding: BindingResult
    validation: ProcedureValidationResult
    execution_allowed: bool = False


@dataclass(frozen=True)
class _TaskSemantics:
    text: str
    action_kind: str | None
    parameters: Mapping[str, Any]
    capabilities: frozenset[str]


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", ".", str(value or "").casefold()).strip(".")



def _semantic_kind(value: Any) -> str:
    raw = str(value or "").strip().casefold()
    aliases = {
        "browser_play_song": "browser.play.song",
        "browser.play.song": "browser.play.song",
        "play_song": "browser.play.song",
        "browser_search": "browser.search",
        "browser_open": "browser.open",
        "browser_open_url": "browser.open.url",
        "browser.open.url": "browser.open.url",
        "browser_click": "browser.click",
        "browser_type": "browser.type",
        "whatsapp_send_message": "whatsapp.send.message",
    }
    return aliases.get(raw, _norm(raw))


def _task_semantics(task_context: Any) -> _TaskSemantics:
    if isinstance(task_context, str):
        text = task_context.strip()
        mapping: Mapping[str, Any] = {}
    elif isinstance(task_context, Mapping):
        mapping = task_context
        text = str(mapping.get("text", mapping.get("goal", mapping.get("task", ""))) or "").strip()
    else:
        text = str(getattr(task_context, "text", getattr(task_context, "goal", "")) or "").strip()
        mapping = getattr(task_context, "__dict__", {}) if task_context is not None else {}

    action_kind = mapping.get("action_kind") or mapping.get("capability")
    parameters = mapping.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {}
    capabilities = mapping.get("capabilities")
    if isinstance(capabilities, str):
        capabilities = {capabilities}
    elif not isinstance(capabilities, Sequence):
        capabilities = set()

    if not action_kind and text:
        try:
            route = classify_fast(text).route
            if route is not None:
                action_kind = route.action_kind
                route_params = route.params if isinstance(route.params, Mapping) else {}
                merged = dict(route_params)
                merged.update(parameters)
                parameters = merged
        except Exception:
            pass

    return _TaskSemantics(
        text=text,
        action_kind=_semantic_kind(action_kind) if action_kind else None,
        parameters=parameters,
        capabilities=frozenset(_semantic_kind(x) for x in capabilities),
    )


def _procedure_kinds(procedure: LearnedProcedure) -> set[str]:
    kinds = {_semantic_kind(procedure.name)}
    kinds.update(_semantic_kind(step.action_name) for step in procedure.steps)
    kinds.update(_semantic_kind(cap) for cap in procedure.required_capabilities)
    kinds.update(_semantic_kind(cap) for step in procedure.steps for cap in step.required_capabilities)
    return kinds


def _meaningful_trigger_tokens(trigger: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", trigger.casefold())
        if token not in {"the", "a", "an", "on", "in", "to", "and", "with"}
        and not (token.startswith("{") and token.endswith("}"))
    }


def _matches_task(procedure: LearnedProcedure, semantics: _TaskSemantics) -> bool:
    kinds = _procedure_kinds(procedure)
    if semantics.action_kind and semantics.action_kind in kinds:
        return True
    if semantics.capabilities and semantics.capabilities.intersection(kinds):
        return True
    # Trigger text is only a secondary deterministic signal. Require all
    # meaningful non-parameter words so unrelated procedures are not selected.
    if semantics.text:
        trigger = _meaningful_trigger_tokens(procedure.trigger_pattern)
        text = set(re.findall(r"[a-z0-9]+", semantics.text.casefold()))
        if trigger and trigger.issubset(text):
            return True
    return False


def retrieve_procedures(
    task_context: Any,
    store: ProcedureStore,
    *,
    limit: int = 100,
) -> list[LearnedProcedure]:
    """Return only retrievable candidates matching the task semantically."""
    semantics = _task_semantics(task_context)
    candidates = store.list_procedures(status=ProcedureStatus.CANDIDATE, limit=limit)
    return [p for p in candidates if _matches_task(p, semantics)]


def _task_values(task_context: Any) -> Mapping[str, Any]:
    return _task_semantics(task_context).parameters


def _value_aliases(name: str) -> tuple[str, ...]:
    key = name.casefold()
    if key in {"song", "title", "query"}:
        return (key, "song", "title", "query")
    if key == "text":
        return ("text", "message", "value")
    return (key,)


def _placeholder_names(procedure: LearnedProcedure) -> set[str]:
    found: set[str] = set()
    for step in procedure.steps:
        for value in step.arguments.values():
            if isinstance(value, str):
                match = re.fullmatch(r"\{([A-Za-z_][A-Za-z0-9_-]*)\}", value.strip())
                if match:
                    found.add(match.group(1))
    return found


def _validate_value(parameter: ProcedureParameter, value: Any) -> str | None:
    typ = parameter.type.casefold()
    if typ in {"string", "str"} and not isinstance(value, str):
        return "expected_string"
    if typ in {"integer", "int"} and (isinstance(value, bool) or not isinstance(value, int)):
        return "expected_integer"
    if typ in {"number", "float"} and (isinstance(value, bool) or not isinstance(value, (int, float))):
        return "expected_number"
    if typ in {"boolean", "bool"} and not isinstance(value, bool):
        return "expected_boolean"

    constraints = parameter.constraints
    if isinstance(value, str):
        if "min_length" in constraints and len(value) < int(constraints["min_length"]):
            return "below_min_length"
        if "max_length" in constraints and len(value) > int(constraints["max_length"]):
            return "above_max_length"
        if "pattern" in constraints and not re.fullmatch(str(constraints["pattern"]), value):
            return "pattern_mismatch"
    if "choices" in constraints and value not in constraints["choices"]:
        return "not_an_allowed_choice"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "min" in constraints and value < constraints["min"]:
            return "below_min"
        if "max" in constraints and value > constraints["max"]:
            return "above_max"
    return None


def bind_procedure_parameters(procedure: LearnedProcedure, task_context: Any) -> BindingResult:
    """Bind invocation values without mutating the stored procedure."""
    task_values = _task_values(task_context)
    placeholders = _placeholder_names(procedure)
    definitions = {p.name.casefold(): p for p in procedure.parameters}
    # A placeholder is part of the persisted semantic procedure definition even
    # if an older candidate was persisted without a duplicated parameter row.
    for name in placeholders:
        if name.casefold() not in definitions:
            return BindingResult(BindingStatus.MISSING, reasons=(f"parameter_schema_missing:{name}",))

    bindings: list[ParameterBinding] = []
    reasons: list[str] = []
    for parameter in procedure.parameters:
        matches = [key for key in _value_aliases(parameter.name) if key in {str(k).casefold() for k in task_values}]
        # Preserve exact original key/value lookup while matching case-insensitively.
        lookup = {str(k).casefold(): v for k, v in task_values.items()}
        matches = [key for key in matches if key in lookup]
        if len(matches) > 1 and len({repr(lookup[key]) for key in matches}) > 1:
            bindings.append(ParameterBinding(parameter, status=BindingStatus.AMBIGUOUS, reason="multiple_values"))
            reasons.append(f"parameter_ambiguous:{parameter.name}")
            continue
        if matches:
            value = lookup[matches[0]]
            error = _validate_value(parameter, value)
            if error:
                bindings.append(ParameterBinding(parameter, value, BindingStatus.INVALID, error))
                reasons.append(f"parameter_invalid:{parameter.name}:{error}")
            else:
                bindings.append(ParameterBinding(parameter, value, BindingStatus.BOUND))
            continue
        if not parameter.required:
            if parameter.default is not None:
                error = _validate_value(parameter, parameter.default)
                if error:
                    bindings.append(ParameterBinding(parameter, status=BindingStatus.INVALID, reason="invalid_default"))
                    reasons.append(f"parameter_invalid:{parameter.name}:invalid_default")
                else:
                    bindings.append(ParameterBinding(parameter, parameter.default, BindingStatus.BOUND, "default"))
            else:
                bindings.append(ParameterBinding(parameter, status=BindingStatus.BOUND, reason="optional_unset"))
            continue
        bindings.append(ParameterBinding(parameter, status=BindingStatus.MISSING, reason="required"))
        reasons.append(f"parameter_missing:{parameter.name}")

    # A browser.play.song candidate without a parameterized query cannot safely
    # accept a current song value. Treat it as missing rather than replaying the
    # value that happened to be present when the procedure was learned.
    if not procedure.parameters and any(_semantic_kind(s.action_name) == "browser.play.song" for s in procedure.steps):
        has_static_query = any(
            isinstance(s.arguments.get("query"), str) and s.arguments.get("query", "").strip()
            for s in procedure.steps
        )
        if not has_static_query:
            reasons.append("parameter_missing:query")
            return BindingResult(BindingStatus.MISSING, tuple(bindings), tuple(reasons))

    if any(b.status is BindingStatus.AMBIGUOUS for b in bindings):
        status = BindingStatus.AMBIGUOUS
    elif any(b.status is BindingStatus.INVALID for b in bindings):
        status = BindingStatus.INVALID
    elif any(b.status is BindingStatus.MISSING for b in bindings):
        status = BindingStatus.MISSING
    else:
        status = BindingStatus.BOUND
    return BindingResult(status, tuple(bindings), tuple(reasons))


def _observation_items(current_state: Any) -> tuple[list[Any], set[str]]:
    if current_state is None:
        return [], set()
    if isinstance(current_state, Mapping):
        observations = current_state.get("observations", current_state.get("observation"))
        if observations is None and ("url" in current_state or "text" in current_state):
            observations = [current_state]
        if isinstance(observations, Mapping):
            observations = list(observations.values())
        elif observations is None:
            observations = []
        capabilities = current_state.get("capabilities", ())
        if isinstance(capabilities, str):
            capabilities = (capabilities,)
        return list(observations) if isinstance(observations, Sequence) else [], {_semantic_kind(x) for x in capabilities}
    if isinstance(current_state, Sequence) and not isinstance(current_state, (str, bytes)):
        return list(current_state), set()
    return [current_state], set()


def _observation_fresh(obs: Any, max_age_s: float) -> bool | None:
    if isinstance(obs, Observation):
        return obs.is_fresh(max_age_s)
    if isinstance(obs, Mapping):
        if "ok" in obs and not bool(obs.get("ok")):
            return False
        if "observed_at" in obs:
            try:
                import time
                return (time.time() - float(obs["observed_at"])) <= max_age_s
            except (TypeError, ValueError):
                return None
        return True
    # BrowserObservation is explicitly produced by BrowserSkill.observe(); it
    # carries a fresh semantic snapshot but no wall-clock timestamp in the
    # existing model. Treat the supplied object as current evidence.
    if hasattr(obs, "generation") and hasattr(obs, "elements") and hasattr(obs, "url"):
        return True
    return None


def _browser_text(obs: Any) -> str:
    parts: list[str] = []
    for attr in ("url", "text"):
        value = getattr(obs, attr, None)
        if isinstance(value, str):
            parts.append(value)
    if isinstance(obs, Mapping):
        for key in ("url", "text", "title", "application", "app"):
            value = obs.get(key)
            if isinstance(value, str):
                parts.append(value)
    return " ".join(parts).casefold()


def _has_browser_observation(observations: Sequence[Any]) -> bool:
    return any(hasattr(o, "url") and hasattr(o, "elements") for o in observations) or any(
        isinstance(o, Mapping) and ("url" in o or str(o.get("source", "")).casefold() == "browser")
        for o in observations
    )


def _evaluate_condition(condition: ProcedureCondition, observations: Sequence[Any], capabilities: set[str], max_age_s: float) -> ConditionResult:
    fresh = [_observation_fresh(o, max_age_s) for o in observations]
    if any(value is False for value in fresh):
        return ConditionResult(condition, ValidationStatus.UNKNOWN, f"stale_or_failed_observation:{condition.condition_id}")
    if observations and any(value is None for value in fresh):
        return ConditionResult(condition, ValidationStatus.UNKNOWN, f"observation_freshness_unknown:{condition.condition_id}")
    text = " ".join(_browser_text(o) for o in observations)
    meta = condition.metadata if isinstance(condition.metadata, Mapping) else {}
    kind = str(meta.get("type", meta.get("kind", ""))).casefold()
    expected = str(meta.get("value", meta.get("expected", "")))
    phrase = condition.condition.casefold().strip()

    if kind in {"capability", "required_capability"}:
        return ConditionResult(condition, ValidationStatus.APPLICABLE if _semantic_kind(expected) in capabilities else ValidationStatus.NOT_APPLICABLE, "capability_present" if _semantic_kind(expected) in capabilities else "capability_missing")
    if kind in {"url_contains", "browser_url_contains"}:
        ok = bool(expected) and expected.casefold() in text
        return ConditionResult(condition, ValidationStatus.APPLICABLE if ok else ValidationStatus.NOT_APPLICABLE, "url_matches" if ok else "url_mismatch")
    if kind in {"text_contains", "browser_text_contains"}:
        ok = bool(expected) and expected.casefold() in text
        return ConditionResult(condition, ValidationStatus.APPLICABLE if ok else ValidationStatus.NOT_APPLICABLE, "text_matches" if ok else "text_mismatch")
    if kind in {"application", "app"}:
        ok = bool(expected) and expected.casefold() in text
        return ConditionResult(condition, ValidationStatus.APPLICABLE if ok else ValidationStatus.NOT_APPLICABLE, "application_matches" if ok else "application_mismatch")

    if phrase in {"browser is available", "browser available", "a browser is available"}:
        ok = _has_browser_observation(observations) or "browser" in capabilities
        return ConditionResult(condition, ValidationStatus.APPLICABLE if ok else ValidationStatus.NOT_APPLICABLE, "browser_available" if ok else "browser_unavailable")
    if "youtube" in phrase and ("open" in phrase or "page" in phrase):
        ok = "youtube.com" in text
        return ConditionResult(condition, ValidationStatus.APPLICABLE if ok else ValidationStatus.NOT_APPLICABLE, "youtube_page_observed" if ok else "youtube_page_not_observed")

    # Unknown condition language fails closed. A condition is never inferred to
    # hold merely because the observation exists.
    return ConditionResult(condition, ValidationStatus.UNKNOWN, "unsupported_condition")


def validate_procedure_current_state(
    procedure: LearnedProcedure,
    current_state: Any,
    *,
    max_age_s: float = 1.0,
) -> ProcedureValidationResult:
    """Validate semantic preconditions against supplied current observations."""
    if procedure.status not in {ProcedureStatus.CANDIDATE, ProcedureStatus.ACTIVE}:
        return ProcedureValidationResult(ValidationStatus.NOT_APPLICABLE, ("status_not_retrievable",))

    observations, capabilities = _observation_items(current_state)
    if not observations and not capabilities:
        return ProcedureValidationResult(ValidationStatus.UNKNOWN, ("current_observation_missing",))

    # Required capabilities are checked without executing anything. A browser
    # semantic observation establishes the browser capability, not arbitrary
    # action execution authority.
    for capability in procedure.required_capabilities:
        cap = _semantic_kind(capability)
        if cap.startswith("browser."):
            if not (_has_browser_observation(observations) or "browser" in capabilities or cap in capabilities):
                return ProcedureValidationResult(ValidationStatus.NOT_APPLICABLE, (f"capability_missing:{capability}",))
        elif cap == "browser" and _has_browser_observation(observations):
            continue
        elif cap not in capabilities:
            return ProcedureValidationResult(ValidationStatus.UNKNOWN, (f"capability_evidence_missing:{capability}",))

    results = tuple(_evaluate_condition(c, observations, capabilities, max_age_s) for c in procedure.preconditions)
    failed = tuple(r.condition.condition_id for r in results if r.status is ValidationStatus.NOT_APPLICABLE)
    unknown = tuple(r.condition.condition_id for r in results if r.status is ValidationStatus.UNKNOWN)
    matched = tuple(r.condition.condition_id for r in results if r.status is ValidationStatus.APPLICABLE)
    if failed:
        return ProcedureValidationResult(ValidationStatus.NOT_APPLICABLE, tuple(r.reason for r in results if r.status is ValidationStatus.NOT_APPLICABLE), matched, failed, results)
    if unknown:
        return ProcedureValidationResult(ValidationStatus.UNKNOWN, tuple(r.reason for r in results if r.status is ValidationStatus.UNKNOWN), matched, failed, results)
    return ProcedureValidationResult(ValidationStatus.APPLICABLE, ("preconditions_satisfied",), matched, failed, results)


def find_executable_procedures(
    task_context: Any,
    store: ProcedureStore,
    current_state: Any,
    *,
    limit: int = 100,
    max_age_s: float = 1.0,
) -> list[ProcedureMatch]:
    """Retrieve only ACTIVE procedures and prepare explicit execution gates.

    P3.2-C keeps ``find_applicable_procedures`` candidate-only and
    execution-disabled. This separate lane is the P3.2-D boundary: only an
    ACTIVE procedure can reach it, and the returned match is still gated by
    binding and current-state validation. No lifecycle mutation occurs here.
    """
    matches: list[ProcedureMatch] = []
    for procedure in retrieve_executable_procedures(task_context, store, limit=limit):
        binding = bind_procedure_parameters(procedure, task_context)
        if binding.status is not BindingStatus.BOUND:
            validation = ProcedureValidationResult(ValidationStatus.UNKNOWN, ("binding_not_ready",))
        else:
            validation = validate_procedure_current_state(procedure, current_state, max_age_s=max_age_s)
        matches.append(ProcedureMatch(procedure, binding, validation, execution_allowed=True))
    return matches


def retrieve_executable_procedures(
    task_context: Any,
    store: ProcedureStore,
    *,
    limit: int = 100,
) -> list[LearnedProcedure]:
    """Return only ACTIVE procedures matching the task semantically."""
    semantics = _task_semantics(task_context)
    active = store.list_procedures(status=ProcedureStatus.ACTIVE, limit=limit)
    return [p for p in active if _matches_task(p, semantics)]


def find_applicable_procedures(
    task_context: Any,
    store: ProcedureStore,
    current_state: Any,
    *,
    limit: int = 100,
    max_age_s: float = 1.0,
) -> list[ProcedureMatch]:
    """Retrieve, bind, and validate candidates. Never executes or mutates them."""
    matches: list[ProcedureMatch] = []
    for procedure in retrieve_procedures(task_context, store, limit=limit):
        binding = bind_procedure_parameters(procedure, task_context)
        if binding.status is not BindingStatus.BOUND:
            validation = ProcedureValidationResult(ValidationStatus.UNKNOWN, ("binding_not_ready",))
        else:
            validation = validate_procedure_current_state(procedure, current_state, max_age_s=max_age_s)
        matches.append(ProcedureMatch(procedure, binding, validation, execution_allowed=False))
    return matches


__all__ = [
    "BindingStatus", "ValidationStatus", "ParameterBinding", "BindingResult",
    "ConditionResult", "ProcedureValidationResult", "ProcedureMatch",
    "retrieve_procedures", "retrieve_executable_procedures", "bind_procedure_parameters",
    "validate_procedure_current_state", "find_applicable_procedures", "find_executable_procedures",
]
