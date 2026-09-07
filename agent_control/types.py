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
    #: The action ran and reported success but verification could not tell.
    #: Unlike UNKNOWN there is a concrete thing to re-read, so it is recoverable.
    INCONCLUSIVE = "INCONCLUSIVE"
    #: The machine offers more than one valid continuation and nothing the agent
    #: can observe decides between them, so a person has to. Distinct from
    #: INCONCLUSIVE (evidence missing, re-reading may help) and from the failure
    #: classes (something went wrong): here nothing is wrong and re-reading
    #: cannot help, because what is missing is a preference.
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"


class AgentState(str, Enum):
    """Where the control loop is, as one word (plan S9/S30).

    The runner sets this at the junctions it already passes through; it is not a
    second control flow. Every value is entered immediately before the work it
    names, so a display driven by these transitions can lag reality but never
    invent it.
    """

    IDLE = "IDLE"
    UNDERSTANDING = "UNDERSTANDING"
    OBSERVING = "OBSERVING"
    PLANNING = "PLANNING"
    ACTING = "ACTING"
    VERIFYING = "VERIFYING"
    RECOVERING = "RECOVERING"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class DecisionKind(str, Enum):
    """What the loop chose to do next."""

    ACT = "ACT"
    OBSERVE = "OBSERVE"
    VERIFY = "VERIFY"
    REPLAN = "REPLAN"
    RECOVER = "RECOVER"
    ASK_USER = "ASK_USER"
    STOP = "STOP"


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


@dataclass(frozen=True)
class AgentDecision:
    """Why the loop is about to do what it does next.

    Structured so it can be traced and queried rather than read as prose. The
    ``reason`` is authored by the control plane, never copied from the planner, and
    the terminal renders fixed sentences from ``kind`` -- displaying this text
    verbatim would be dumping reasoning at the user, which plan S30 forbids.
    """

    kind: DecisionKind
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "kind": self.kind.value,
            "reason": self.reason,
            "detail": _jsonable(self.detail),
        }


@dataclass(frozen=True)
class Clarification:
    """A decision the run cannot make for itself, addressed to the user.

    ``options`` may only contain continuations the agent actually observed and can
    actually take. When the ambiguity is visible but its alternatives are not --
    a Windows app chooser, whose contents no current backend can read -- options
    stays empty and ``unobservable`` names the limit instead. Inventing plausible
    options here would be the system claiming to see something it cannot.
    """

    question: str
    options: tuple[str, ...] = ()
    context: str = ""
    #: Which reading raised this, when one did.
    source: Source | None = None
    #: What could not be read, when ``options`` is empty for that reason.
    unobservable: str = ""

    def to_json(self) -> dict:
        return {
            "question": self.question,
            "options": list(self.options),
            "context": self.context,
            "source": self.source.value if self.source else None,
            "unobservable": self.unobservable,
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


class NeedUserInput(ControlPlaneError):
    """Raised when only a person can choose what happens next.

    Not a failure and not a retry. Raising it unwinds the remaining actions of the
    current batch -- which is how a run stops without executing steps that were
    planned on an assumption that no longer holds -- and the runner turns it into a
    suspended run carrying the question. ``verify_final`` is deliberately not
    reached on that path, for the same reason a cancelled run skips it: several
    checks are preconditions rather than outcomes, so a run stopped early could
    otherwise report PASS for work it never did.
    """

    failure_class = FailureClass.AMBIGUOUS

    def __init__(self, clarification: Clarification) -> None:
        super().__init__(clarification.question)
        self.clarification = clarification


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
