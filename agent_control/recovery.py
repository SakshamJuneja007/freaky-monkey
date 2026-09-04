"""Bounded recovery (plan S9).

The budget is the whole point. An agent that retries until it works has no
measurable reliability, so every attempt is counted against an explicit ceiling
and logged (plan S30: "every retry must be bounded and logged").

V1 budget: retry once, re-observe once, re-plan once, then abort and report. No
rollback -- the plan defers checkpoints and transactional actions to V4.

``enabled=False`` is the recovery ablation from plan S18: identical code path,
every failure aborts immediately, so the delta attributable to recovery is
measurable rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .types import (
    ActionResult,
    FailureClass,
    VerificationResult,
    Verdict,
)


class RecoveryDecision(str, Enum):
    RETRY = "RETRY"
    REOBSERVE_THEN_RETRY = "REOBSERVE_THEN_RETRY"
    REPLAN = "REPLAN"
    ASK_USER = "ASK_USER"
    ABORT = "ABORT"


@dataclass
class RecoveryBudget:
    """Hard ceilings, consumed in place."""

    max_retries: int = 1
    max_reobserves: int = 1
    max_replans: int = 1

    retries_used: int = 0
    reobserves_used: int = 0
    replans_used: int = 0

    def remaining(self) -> dict[str, int]:
        return {
            "retries": max(0, self.max_retries - self.retries_used),
            "reobserves": max(0, self.max_reobserves - self.reobserves_used),
            "replans": max(0, self.max_replans - self.replans_used),
        }

    def spend(self, decision: RecoveryDecision) -> bool:
        """Consume budget for a decision. False if the ceiling is already hit."""
        left = self.remaining()
        if decision is RecoveryDecision.RETRY:
            if left["retries"] <= 0:
                return False
            self.retries_used += 1
        elif decision is RecoveryDecision.REOBSERVE_THEN_RETRY:
            if left["reobserves"] <= 0 or left["retries"] <= 0:
                return False
            self.reobserves_used += 1
            self.retries_used += 1
        elif decision is RecoveryDecision.REPLAN:
            if left["replans"] <= 0:
                return False
            self.replans_used += 1
        return True

    def exhausted(self) -> bool:
        return all(v <= 0 for v in self.remaining().values())

    def to_json(self) -> dict:
        return {
            "limits": {"retries": self.max_retries, "reobserves": self.max_reobserves,
                       "replans": self.max_replans},
            "used": {"retries": self.retries_used, "reobserves": self.reobserves_used,
                     "replans": self.replans_used},
            "remaining": self.remaining(),
        }


#: Failure class -> first-choice response (plan S9). Anything not listed aborts.
_STRATEGY: dict[FailureClass, RecoveryDecision] = {
    FailureClass.TRANSIENT: RecoveryDecision.RETRY,
    FailureClass.STALE_STATE: RecoveryDecision.REOBSERVE_THEN_RETRY,
    FailureClass.PRECONDITION_FAILED: RecoveryDecision.REPLAN,
    # No alternate-action registry exists in V1, so ACTION_FAILED retries once
    # rather than pretending to know a second route (plan S23).
    FailureClass.ACTION_FAILED: RecoveryDecision.RETRY,
    FailureClass.VERIFICATION_FAILED: RecoveryDecision.REOBSERVE_THEN_RETRY,
    FailureClass.PERMISSION_DENIED: RecoveryDecision.ASK_USER,
    FailureClass.ENVIRONMENT: RecoveryDecision.ABORT,
    FailureClass.UNKNOWN: RecoveryDecision.ABORT,
}


def classify(
    result: ActionResult | None,
    verification: VerificationResult | None,
    *,
    precondition_age_s: float | None = None,
    max_staleness_s: float | None = None,
) -> FailureClass:
    """Assign a failure class from evidence, most specific cause first."""
    if result is not None and result.failure_class is FailureClass.PERMISSION_DENIED:
        return FailureClass.PERMISSION_DENIED

    stale = (
        precondition_age_s is not None
        and max_staleness_s is not None
        and precondition_age_s > max_staleness_s
    )
    if stale:
        return FailureClass.STALE_STATE

    if result is not None and not result.ok:
        return result.failure_class or FailureClass.ACTION_FAILED

    if verification is not None:
        if verification.verdict is Verdict.FAIL:
            return FailureClass.VERIFICATION_FAILED
        if verification.verdict is Verdict.UNKNOWN:
            return FailureClass.UNKNOWN
    return FailureClass.UNKNOWN


@dataclass
class Recovery:
    """Decides what to do next, and refuses to decide more than the budget allows."""

    budget: RecoveryBudget = field(default_factory=RecoveryBudget)
    #: Ablation switch (plan S18). False -> every failure aborts immediately.
    enabled: bool = True
    #: Unattended runs must not escalate on their own, so ASK_USER aborts.
    interactive: bool = False

    def decide(self, failure_class: FailureClass) -> tuple[RecoveryDecision, str]:
        if not self.enabled:
            return RecoveryDecision.ABORT, "recovery disabled (ablation condition)"

        preferred = _STRATEGY.get(failure_class, RecoveryDecision.ABORT)

        if preferred is RecoveryDecision.ASK_USER and not self.interactive:
            return RecoveryDecision.ABORT, "permission denied; no interactive approver"
        if preferred is RecoveryDecision.ABORT:
            return preferred, f"{failure_class.value} is not recoverable in V1"
        if not self.budget.spend(preferred):
            return RecoveryDecision.ABORT, f"budget exhausted for {preferred.value}"
        return preferred, f"{failure_class.value} -> {preferred.value}"

    def to_json(self) -> dict:
        return {"enabled": self.enabled, "interactive": self.interactive,
                "budget": self.budget.to_json()}
