"""Core contracts for the control plane.

Everything the runtime passes between layers is defined here. Kept in one
module deliberately: per the project plan (S23) no abstraction is introduced
until repeated use justifies it, so there is one vocabulary, not five.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """Verification outcome. UNKNOWN is not success (plan S3)."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class FailureClass(str, Enum):
    """Failure taxonomy driving recovery (plan S9)."""

    TRANSIENT = "TRANSIENT"
    STALE_STATE = "STALE_STATE"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    ACTION_FAILED = "ACTION_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    ENVIRONMENT = "ENVIRONMENT"
    UNKNOWN = "UNKNOWN"


class Source(str, Enum):
    """Where a piece of state came from. Ordered strongest-first (plan S6)."""

    FILESYSTEM = "filesystem"
    PROCESS = "process"
    SHELL = "shell"
    NETWORK = "network"
    WINDOW = "window"
    ACCESSIBILITY = "accessibility"
    BROWSER = "browser"
    VISION = "vision"


#: Any observation older than this must be re-taken before a consequential
#: action. This single number is what hypothesis H2 is testing, so it is a
#: named constant rather than a scattered literal.
DEFAULT_MAX_STALENESS_S = 1.0


@dataclass(frozen=True)
class Observation:
    """One reading of machine state, stamped with time and provenance.

    ``ok=False`` means the *reading* failed (we could not determine state), which
    is different from reading a state that says "absent". Verifiers must map the
    former to UNKNOWN and the latter to FAIL.
    """

    source: Source
    query: str
    value: Any
    observed_at: float = field(default_factory=time.time)
    ok: bool = True
    error: str | None = None

    def age(self, now: float | None = None) -> float:
        return (time.time() if now is None else now) - self.observed_at

    def is_fresh(self, max_age_s: float = DEFAULT_MAX_STALENESS_S) -> bool:
        return self.ok and self.age() <= max_age_s

    def to_json(self) -> dict:
        return {
            "source": self.source.value,
            "query": self.query,
            "value": _jsonable(self.value),
            "observed_at": self.observed_at,
            "age_s": round(self.age(), 4),
            "ok": self.ok,
            "error": self.error,
        }


@dataclass(frozen=True)
class Action:
    """A semantic action (plan S3), never raw coordinates unless kind=="vision_*"."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    #: Consequential actions get a fresh precondition check and a verification
    #: path. Read-only actions do not.
    consequential: bool = True
    #: Free-text rationale from the planner. Recorded, never trusted.
    rationale: str = ""

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "params": _jsonable(self.params),
            "consequential": self.consequential,
            "rationale": self.rationale,
        }


@dataclass
class ActionResult:
    """What the execution layer observed while performing the action.

    This is *evidence*, not proof. Verifiers deliberately do not read it
    (plan S8: never treat belief that an action succeeded as proof).
    """

    action: Action
    ok: bool
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    failure_class: FailureClass | None = None
    duration_s: float = 0.0

    def to_json(self) -> dict:
        return {
            "action": self.action.to_json(),
            "ok": self.ok,
            "detail": _jsonable(self.detail),
            "error": self.error,
            "failure_class": self.failure_class.value if self.failure_class else None,
            "duration_s": round(self.duration_s, 4),
        }


@dataclass
class Check:
    """One independent assertion about end state."""

    name: str
    verdict: Verdict
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "verdict": self.verdict.value,
            "evidence": _jsonable(self.evidence),
            "reason": self.reason,
        }


@dataclass
class VerificationResult:
    """Aggregate of independent checks.

    Aggregation rule (plan S3/S8): any FAIL -> FAIL; else any UNKNOWN -> UNKNOWN;
    else PASS. UNKNOWN never silently becomes PASS.
    """

    checks: list[Check] = field(default_factory=list)
    label: str = ""

    @property
    def verdict(self) -> Verdict:
        if not self.checks:
            return Verdict.UNKNOWN
        if any(c.verdict is Verdict.FAIL for c in self.checks):
            return Verdict.FAIL
        if any(c.verdict is Verdict.UNKNOWN for c in self.checks):
            return Verdict.UNKNOWN
        return Verdict.PASS

    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.verdict is not Verdict.PASS]

    def to_json(self) -> dict:
        return {
            "label": self.label,
            "verdict": self.verdict.value,
            "checks": [c.to_json() for c in self.checks],
        }


class ControlPlaneError(Exception):
    """Base for errors the runtime raises deliberately."""

    failure_class = FailureClass.UNKNOWN


class StaleStateError(ControlPlaneError):
    """Raised when a consequential action was about to run on stale state."""

    failure_class = FailureClass.STALE_STATE


class PolicyDenied(ControlPlaneError):
    """Raised when the policy layer refuses an action. Enforced outside the LLM."""

    failure_class = FailureClass.PERMISSION_DENIED


def _jsonable(value: Any) -> Any:
    """Best-effort conversion for trace logging. Never raises."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_json"):
        try:
            return value.to_json()
        except Exception:  # pragma: no cover - trace must never break a run
            pass
    return repr(value)
