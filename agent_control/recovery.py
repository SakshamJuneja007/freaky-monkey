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
    Action,
    ActionResult,
    FailureClass,
    RetrySafety,
    VerificationResult,
    Verdict,
)


# Conservative defaults for actions whose outcome can be ambiguous.  Skills may
# override these through Action metadata; the workflow engine never guesses that
# a side effect is idempotent merely because the executor returned.
_NON_IDEMPOTENT_KINDS = frozenset({
    "gmail_send_email",
    "whatsapp_send_message",
    "delete_file",
    "form_submit",
    "payment",
    # Text entry is inherently non-idempotent: replaying after UNKNOWN
    # appends/duplicates user-visible content (e.g. ``hellohello``).
    "type_text",
})


def retry_safety(action: Action) -> RetrySafety:
    """Return the effective retry policy for an action.

    Explicit metadata wins.  AUTO uses a conservative capability default:
    consequential actions that can create an external side effect require a
    fresh observation before any possible repeat; known non-idempotent actions
    are never blindly replayed.
    """
    if action.retry_safety is not RetrySafety.AUTO:
        return action.retry_safety
    if action.kind in _NON_IDEMPOTENT_KINDS or action.idempotent is False:
        return RetrySafety.REOBSERVE_FIRST
    if action.idempotent is True or not action.consequential:
        return RetrySafety.SAFE_TO_RETRY
    if action.side_effect_level.lower() in {"high", "critical"}:
        return RetrySafety.ASK_USER
    return RetrySafety.SAFE_TO_RETRY


def action_can_retry(action: Action, *, post_action_unknown: bool = False) -> bool:
    """Whether a side-effect may be executed again at this point."""
    safety = retry_safety(action)
    if post_action_unknown:
        return safety is RetrySafety.SAFE_TO_RETRY
    return safety in {RetrySafety.SAFE_TO_RETRY, RetrySafety.REOBSERVE_FIRST}


def classify_failure_name(value: object | None) -> FailureClass | None:
    """Map executor/runtime strings into the P2.6 failure vocabulary."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for item in FailureClass:
        if text == item.value or text.upper() == item.name:
            return item
    aliases = {
        "timeout": FailureClass.EXECUTION_TIMEOUT,
        "browser_disconnected": FailureClass.BROWSER_CONNECTION_FAILED,
        "browser_connection_failed": FailureClass.BROWSER_CONNECTION_FAILED,
        "not_found": FailureClass.TARGET_NOT_FOUND,
        "application_not_found": FailureClass.APPLICATION_NOT_FOUND,
        "stale": FailureClass.STALE_OBSERVATION,
        "unknown": FailureClass.VERIFICATION_UNKNOWN,
        "verification_unknown": FailureClass.VERIFICATION_UNKNOWN,
        "policy_denied": FailureClass.POLICY_DENIED,
        "approval_required": FailureClass.APPROVAL_REQUIRED,
        "media_navigation_failed": FailureClass.EXPECTED_STATE_NOT_REACHED,
        "expected_state_not_reached": FailureClass.EXPECTED_STATE_NOT_REACHED,
        "navigation_timeout": FailureClass.NAVIGATION_TIMEOUT,
        "browser_session_dead": FailureClass.BROWSER_CONNECTION_FAILED,
    }
    return aliases.get(text.lower())


class RecoveryDecision(str, Enum):
    RETRY = "RETRY"
    REOBSERVE = "REOBSERVE"
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
    max_recovery_attempts: int = 4

    retries_used: int = 0
    reobserves_used: int = 0
    replans_used: int = 0
    recovery_attempts_used: int = 0

    def remaining(self) -> dict[str, int]:
        return {
            "retries": max(0, self.max_retries - self.retries_used),
            "reobserves": max(0, self.max_reobserves - self.reobserves_used),
            "replans": max(0, self.max_replans - self.replans_used),
        }

    def spend(self, decision: RecoveryDecision) -> bool:
        """Consume budget for a decision. False if the ceiling is already hit.

        ``ASK_USER`` and ``ABORT`` have no branch here and therefore cost nothing.
        That omission is the mechanism, not an oversight: a run suspended on a
        question has not attempted anything, so charging it a retry would let a
        person's hesitation exhaust the ceiling and turn a decision they are still
        making into a failure they did not cause.
        """
        left = self.remaining()
        if decision in {RecoveryDecision.RETRY, RecoveryDecision.REOBSERVE, RecoveryDecision.REOBSERVE_THEN_RETRY, RecoveryDecision.REPLAN}:
            if self.max_recovery_attempts - self.recovery_attempts_used <= 0:
                return False
            self.recovery_attempts_used += 1
        if decision is RecoveryDecision.RETRY:
            if left["retries"] <= 0:
                return False
            self.retries_used += 1
        elif decision is RecoveryDecision.REOBSERVE:
            if left["reobserves"] <= 0:
                self.recovery_attempts_used -= 1
                return False
            self.reobserves_used += 1
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
        return self.max_recovery_attempts - self.recovery_attempts_used <= 0

    def to_json(self) -> dict:
        return {
            "limits": {"retries": self.max_retries, "reobserves": self.max_reobserves,
                       "replans": self.max_replans},
            "used": {"retries": self.retries_used, "reobserves": self.reobserves_used,
                     "replans": self.replans_used},
            "remaining": {**self.remaining(), "recovery_attempts": max(0, self.max_recovery_attempts - self.recovery_attempts_used)},
        }


#: Failure class -> first-choice response (plan S9). Anything not listed aborts.
_STRATEGY: dict[FailureClass, RecoveryDecision] = {
    FailureClass.TRANSIENT: RecoveryDecision.RETRY,
    FailureClass.STALE_STATE: RecoveryDecision.REOBSERVE_THEN_RETRY,
    FailureClass.PRECONDITION_FAILED: RecoveryDecision.REPLAN,
    # No alternate-action registry exists in V1, so ACTION_FAILED retries once
    # rather than pretending to know a second route (plan S23).
    FailureClass.ACTION_FAILED: RecoveryDecision.RETRY,
    # BrowserSkill performs a bounded, observation-scoped navigation recovery
    # before returning this class. Do not spend the generic retry budget again
    # on the same unresolved destination-state failure.
    FailureClass.EXPECTED_STATE_NOT_REACHED: RecoveryDecision.REOBSERVE_THEN_RETRY,
    FailureClass.NAVIGATION_TIMEOUT: RecoveryDecision.ABORT,
    FailureClass.VERIFICATION_FAILED: RecoveryDecision.REOBSERVE_THEN_RETRY,
    FailureClass.PERMISSION_DENIED: RecoveryDecision.ASK_USER,
    FailureClass.ENVIRONMENT: RecoveryDecision.ABORT,
    # Same decision as VERIFICATION_FAILED, different reason: the evidence is
    # missing rather than negative, so re-reading is what can change the answer.
    FailureClass.INCONCLUSIVE: RecoveryDecision.REOBSERVE_THEN_RETRY,
    # Nothing is broken and re-reading cannot help -- the missing input is a
    # preference -- so the only move that can change the answer is asking.
    FailureClass.AMBIGUOUS: RecoveryDecision.ASK_USER,
    FailureClass.UNKNOWN: RecoveryDecision.ABORT,
    FailureClass.EXECUTION_EXCEPTION: RecoveryDecision.RETRY,
    FailureClass.EXECUTION_TIMEOUT: RecoveryDecision.RETRY,
    FailureClass.BROWSER_CONNECTION_FAILED: RecoveryDecision.REOBSERVE_THEN_RETRY,
    FailureClass.APPLICATION_NOT_FOUND: RecoveryDecision.REPLAN,
    FailureClass.TARGET_NOT_FOUND: RecoveryDecision.REPLAN,
    FailureClass.STALE_OBSERVATION: RecoveryDecision.REOBSERVE_THEN_RETRY,
    FailureClass.VERIFICATION_UNKNOWN: RecoveryDecision.REOBSERVE_THEN_RETRY,
    FailureClass.RESOURCE_UNAVAILABLE: RecoveryDecision.RETRY,
    FailureClass.DEPENDENCY_FAILED: RecoveryDecision.ABORT,
    FailureClass.ACTION_MAY_HAVE_SUCCEEDED: RecoveryDecision.REOBSERVE_THEN_RETRY,
    FailureClass.CANCELLATION_REQUESTED: RecoveryDecision.ABORT,
    FailureClass.RECOVERY_EXHAUSTED: RecoveryDecision.ABORT,
    FailureClass.UNRECOVERABLE_FAILURE: RecoveryDecision.ABORT,
}


def classify(
    result: ActionResult | None,
    verification: VerificationResult | None,
    *,
    precondition_age_s: float | None = None,
    max_staleness_s: float | None = None,
) -> FailureClass:
    """Assign a failure class from evidence, most specific cause first."""
    if result is not None and result.failure_class is not None:
        explicit = classify_failure_name(result.failure_class)
        if explicit is not None and explicit is not FailureClass.UNKNOWN:
            return explicit
        if result.failure_class is FailureClass.PERMISSION_DENIED:
            return FailureClass.PERMISSION_DENIED

    stale = (
        precondition_age_s is not None
        and max_staleness_s is not None
        and precondition_age_s > max_staleness_s
    )
    if stale:
        return FailureClass.STALE_STATE

    if result is not None and not result.ok:
        if isinstance(result.detail, dict):
            underlying = classify_failure_name(result.detail.get("skill_failure_class") or result.detail.get("failure_class"))
            if underlying is not None and underlying is not FailureClass.UNKNOWN:
                return underlying
        return result.failure_class or FailureClass.ACTION_FAILED

    if verification is not None:
        if verification.verdict is Verdict.FAIL:
            if result is not None and result.ok and result.action.kind == "launch_app":
                target = result.action.params.get("url") or result.action.params.get("open_path")
                if isinstance(target, str) and target.startswith(("http://", "https://")):
                    # Never open the same URL again just because browser-title
                    # evidence was inconclusive. Report honestly instead.
                    return FailureClass.ENVIRONMENT
            return FailureClass.VERIFICATION_FAILED
        # Verification could not tell, but the action ran and reported success,
        # so there is something concrete to re-read rather than nothing to try.
        if verification.verdict is Verdict.UNKNOWN and result is not None and result.ok:
            return FailureClass.INCONCLUSIVE
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
            # Two classes route here now -- a denial to approve and an ambiguity to
            # resolve -- and the reason names which, because "permission denied"
            # over a suspended app-chooser would misreport why the run stopped.
            return (RecoveryDecision.ABORT,
                    f"{failure_class.value} needs a user decision; "
                    "no interactive approver")
        if preferred is RecoveryDecision.ABORT:
            return preferred, f"{failure_class.value} is not recoverable in V1"
        # ASK_USER never reaches this line, so no budget is charged for waiting.
        if not self.budget.spend(preferred):
            return RecoveryDecision.ABORT, f"budget exhausted for {preferred.value}"
        return preferred, f"{failure_class.value} -> {preferred.value}"

    def to_json(self) -> dict:
        return {"enabled": self.enabled, "interactive": self.interactive,
                "budget": self.budget.to_json()}
