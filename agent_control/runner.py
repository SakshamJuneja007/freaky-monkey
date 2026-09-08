"""The control loop: observe -> select -> permit -> execute -> verify -> recover.

This is the structured-first condition of the experiment (plan S18). The
vision-only baseline lives in ``benchmark/baselines/vision_only.py`` and shares
this module's task contract, verifiers, and metrics so the two conditions are
scored identically.

Three switches exist purely so the ablations in plan S18 are the same code
path with a flag flipped, never a second implementation:

* ``fresh_precondition`` -- re-read state immediately before consequential
  actions (hypothesis H2).

* ``recovery_enabled`` -- bounded retry/re-observe/abort (hypothesis H3).

* ``max_staleness_s`` -- how old a reading may be before it is refused.

The control loop deliberately keeps planning, policy, execution, verification,
and recovery separate. This makes the experimental conditions comparable while
also allowing individual components to evolve independently.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from . import os_tools
from .observe import summarize
from .planner.base import Planner, PlannerStep
from .policy import Decision, Policy
from .recovery import (
    Recovery,
    RecoveryBudget,
    RecoveryDecision,
    classify,
)
from .skills.registry import SkillRegistry
from .task import Task, oldest_age_s, state_fingerprint
from .trace import Trace
from .types import (
    Action,
    ActionResult,
    AgentDecision,
    AgentState,
    Check,
    Clarification,
    DecisionKind,
    DEFAULT_MAX_STALENESS_S,
    FailureClass,
    NeedUserInput,
    VerificationResult,
    Verdict,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Configuration for a single experimental run."""

    condition: str = "structured_hybrid"
    max_steps: int = 8
    max_staleness_s: float = DEFAULT_MAX_STALENESS_S
    fresh_precondition: bool = True
    recovery_enabled: bool = True
    budget: RecoveryBudget = field(default_factory=RecoveryBudget)
    interactive: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "max_steps": self.max_steps,
            "max_staleness_s": self.max_staleness_s,
            "fresh_precondition": self.fresh_precondition,
            "recovery_enabled": self.recovery_enabled,
            "interactive": self.interactive,
            "budget": self.budget.to_json()["limits"],
        }


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------


@dataclass
class RunOutcome:
    """Result of one trial.

    ``false_success`` is the headline metric of plan S19:

    the agent reported success, but independent verification disagreed.

    Inspection-only tasks are explicitly excluded from false-success scoring
    because they may legitimately complete without a PASS-style task effect.
    """

    task_id: str
    condition: str
    trial: int
    planner_name: str
    reported_success: bool
    verified: Verdict
    final: VerificationResult
    checkpoints: list[VerificationResult] = field(default_factory=list)
    failure_categories: list[str] = field(default_factory=list)
    steps_used: int = 0
    wall_clock_s: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
    aborted_reason: str | None = None
    synthetic: bool = False
    cancelled: bool = False
    state: AgentState = AgentState.IDLE
    awaiting: bool = False
    question: Clarification | None = None

    #: Planner's final findings for an inspection-only run.
    #: This is informational and does not affect ``verified``.
    informational_output: str = ""

    #: Explicitly marks a legitimate informational completion.
    informational_completion: bool = False

    @property
    def verified_success(self) -> bool:
        return self.verified is Verdict.PASS

    @property
    def false_success(self) -> bool:
        """Agent said done; independent verification disagreed.

        Cancelled, suspended, and legitimate inspection-only completions
        are excluded from this metric.
        """
        return (
            self.reported_success
            and not self.verified_success
            and not self.cancelled
            and not self.awaiting
            and not self.informational_completion
        )

    @property
    def silent_failure(self) -> bool:
        """Verification passed but the agent never claimed completion."""
        return self.verified_success and not self.reported_success

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "condition": self.condition,
            "trial": self.trial,
            "planner": self.planner_name,
            "reported_success": self.reported_success,
            "verified": self.verified.value,
            "verified_success": self.verified_success,
            "informational_completion": self.informational_completion,
            "informational_output": self.informational_output,
            "false_success": self.false_success,
            "silent_failure": self.silent_failure,
            "cancelled": self.cancelled,
            "state": self.state.value,
            "awaiting": self.awaiting,
            "question": (
                self.question.to_json()
                if self.question
                else None
            ),
            "steps_used": self.steps_used,
            "wall_clock_s": round(self.wall_clock_s, 4),
            "aborted_reason": self.aborted_reason,
            "failure_categories": self.failure_categories,
            "usage": self.usage,
            "final_verification": self.final.to_json(),
            "checkpoints": [
                checkpoint.to_json()
                for checkpoint in self.checkpoints
            ],
            "trace": self.trace,
            "synthetic": self.synthetic,
        }


# ---------------------------------------------------------------------------
# Mutable loop context
# ---------------------------------------------------------------------------


@dataclass
class _Loop:
    """Mutable execution context shared by the control-loop helpers."""

    task: Task
    policy: Policy
    config: RunConfig
    trace: Trace
    recovery: Recovery
    skills: SkillRegistry | None = None

    observations: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""
    checkpoints: list[VerificationResult] = field(
        default_factory=list
    )
    abort_reason: str | None = None
    replan: bool = False
    steps_used: int = 0
    extra_state: dict[str, Any] = field(default_factory=dict)
    state: AgentState = AgentState.IDLE
    question: Clarification | None = None

    def refresh(self) -> float:
        """Observe the task and establish a fresh state fingerprint."""
        self.observations = self.task.observe(
            self.policy,
            self.trace,
        )

        self.fingerprint = state_fingerprint(
            self.observations
        )

        return oldest_age_s(
            self.observations
        )

    def enter(
        self,
        state: AgentState,
        **fields: Any,
    ) -> AgentState:
        """Transition to a new agent state and record it in the trace."""

        if state is self.state:
            return state

        previous = self.state
        self.state = state

        self.trace.emit(
            "agent_state",
            state=state.value,
            previous=previous.value,
            **fields,
        )

        return state

    def decide(
        self,
        kind: DecisionKind,
        reason: str,
        **detail: Any,
    ) -> AgentDecision:
        """Create and trace a control decision."""

        decision = AgentDecision(
            kind=kind,
            reason=reason,
            detail=dict(detail),
        )

        self.trace.emit(
            "decision",
            **decision.to_json(),
        )

        return decision


# ---------------------------------------------------------------------------
# Recovery helpers
# ---------------------------------------------------------------------------


_MAX_ATTEMPTS_PER_ACTION = 4


_DECISION_KIND = {
    RecoveryDecision.RETRY: DecisionKind.RECOVER,
    RecoveryDecision.REOBSERVE_THEN_RETRY: DecisionKind.OBSERVE,
    RecoveryDecision.REPLAN: DecisionKind.REPLAN,
    RecoveryDecision.ASK_USER: DecisionKind.ASK_USER,
    RecoveryDecision.ABORT: DecisionKind.STOP,
}


def _skill_name(skill: Any) -> str:
    """Return the canonical skill name.

    Supports both the intended ``skill.name`` interface and older
    implementations that expose only ``skill.info.name``.
    """
    name = getattr(skill, "name", None)

    if isinstance(name, str) and name.strip():
        return name

    info = getattr(skill, "info", None)
    info_name = getattr(info, "name", None)

    if isinstance(info_name, str) and info_name.strip():
        return info_name

    return type(skill).__name__


def _clarification_in(
    result: ActionResult | None,
) -> Clarification | None:
    """Extract a clarification request from an action result."""

    if result is None:
        return None

    found = result.detail.get("clarification")

    if isinstance(found, Clarification):
        return found

    return None


def _dispatch(
    loop: _Loop,
    failure_class: FailureClass,
    attempt: int,
) -> bool:
    """Apply the configured recovery policy.

    Returns ``True`` when the caller should retry the current action.
    Returns ``False`` when the caller should stop the current action.
    """

    decision, reason = loop.recovery.decide(
        failure_class
    )

    # Asking the user only makes sense when we actually have a question.
    if (
        decision is RecoveryDecision.ASK_USER
        and loop.question is not None
    ):
        loop.trace.emit(
            "clarification_needed",
            failure_class=failure_class.value,
            attempt=attempt,
            question=loop.question.to_json(),
        )

        loop.decide(
            DecisionKind.ASK_USER,
            "only a person can choose the continuation",
            failure_class=failure_class.value,
        )

        loop.enter(
            AgentState.WAITING_FOR_USER
        )

        return False

    # A recovery policy may request ASK_USER even when no clarification was
    # attached. Treat that as an abort rather than silently pretending we can
    # continue.
    if decision is RecoveryDecision.ASK_USER:
        decision = RecoveryDecision.ABORT
        reason = (
            f"{reason}; recovery requested user input "
            "but no clarification was provided"
        )

    loop.enter(
        AgentState.RECOVERING,
        failure_class=failure_class.value,
    )

    loop.trace.recovery(
        failure_class=failure_class,
        decision=decision.value,
        attempt=attempt,
        budget_left=loop.recovery.budget.remaining(),
    )

    loop.decide(
        _DECISION_KIND[decision],
        reason,
        failure_class=failure_class.value,
        attempt=attempt,
    )

    if decision is RecoveryDecision.REOBSERVE_THEN_RETRY:
        loop.enter(
            AgentState.OBSERVING,
            purpose="recovery",
        )

        loop.refresh()

        return True

    if decision is RecoveryDecision.RETRY:
        return True

    if decision is RecoveryDecision.REPLAN:
        loop.enter(
            AgentState.OBSERVING,
            purpose="replan",
        )

        loop.refresh()
        loop.replan = True

        return False

    # No continuation is possible.
    loop.question = None
    loop.abort_reason = (
        f"{failure_class.value}: {reason}"
    )

    return False


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def _precondition_failure(
    loop: _Loop,
    action: Action,
) -> FailureClass | None:
    """Check freshness and detect state changes before consequential actions."""

    if not (
        action.consequential
        and loop.config.fresh_precondition
    ):
        return None

    before = loop.fingerprint

    loop.enter(
        AgentState.OBSERVING,
        purpose="precondition",
    )

    age = loop.refresh()

    if age > loop.config.max_staleness_s:
        loop.trace.note(
            "stale_reading_refused",
            age_s=round(age, 4),
            max_staleness_s=loop.config.max_staleness_s,
            action=action.kind,
        )

        return FailureClass.STALE_STATE

    if (
        before
        and loop.fingerprint != before
    ):
        loop.trace.note(
            "state_changed_during_reasoning",
            action=action.kind,
            before=before[:16],
            after=loop.fingerprint[:16],
        )

        return FailureClass.STALE_STATE

    return None


# ---------------------------------------------------------------------------
# Skill execution
# ---------------------------------------------------------------------------


def _skill_action_from_core(
    action: Action,
    skill: Any,
) -> Any:
    """Convert a core Action into a skill-specific action.

    Action conversion belongs to the selected skill. The control loop
    therefore remains independent of browser-specific or other skill-specific
    action schemas.
    """

    adapter = getattr(
        skill,
        "adapt_action",
        None,
    )

    if not callable(adapter):
        raise TypeError(
            f"skill {_skill_name(skill)!r} does not expose "
            "a callable adapt_action()"
        )

    return adapter(action)


def _skill_verification_is_deferred(
    skill_action: Any,
) -> bool:
    """Return whether skill-level verification should be deferred.

    The current BrowserVerifier uses ``ok=False`` when no skill-specific
    verification target is present. In that case verification belongs to
    the task-level checkpoint verifier.

    This compatibility helper can later be replaced by an explicit
    PASS/FAIL/DEFERRED verification result.
    """

    explicit = getattr(
        skill_action,
        "verification_deferred",
        None,
    )

    if isinstance(explicit, bool):
        return explicit

    params = getattr(
        skill_action,
        "params",
        None,
    )

    if not isinstance(params, dict):
        return False

    verification_targets = (
        "expected_url",
        "expected_url_contains",
        "expected_title",
        "expected_text",
    )

    return not any(
        target in params
        for target in verification_targets
    )


def _normalize_skill_verification(
    verification: Any,
) -> tuple[bool, str]:
    """Normalize the current skill verification result."""

    ok = bool(
        getattr(
            verification,
            "ok",
            False,
        )
    )

    detail = getattr(
        verification,
        "detail",
        "",
    )

    return ok, str(detail or "")


def _execute_skill_action(
    loop: _Loop,
    action: Action,
    skill: Any,
) -> ActionResult:
    """Execute and independently verify one action through a skill.

    Pipeline:

        core Action
            ↓
        skill.adapt_action()
            ↓
        skill.executor().execute()
            ↓
        skill.verifier().verify()
            ↓
        normalized ActionResult

    Policy approval has already happened before this function is called.
    """

    skill_name = _skill_name(skill)

    try:
        # ---------------------------------------------------------------
        # Adapt core action -> skill-specific action
        # ---------------------------------------------------------------
        skill_action = _skill_action_from_core(
            action,
            skill,
        )

        loop.trace.emit(
            "skill_dispatch",
            skill=skill_name,
            action=action.kind,
        )

        # ---------------------------------------------------------------
        # Execute
        # ---------------------------------------------------------------
        executor_factory = getattr(
            skill,
            "executor",
            None,
        )

        if not callable(executor_factory):
            raise TypeError(
                f"skill {skill_name!r} does not expose "
                "a callable executor()"
            )

        executor = executor_factory()

        execute = getattr(
            executor,
            "execute",
            None,
        )

        if not callable(execute):
            raise TypeError(
                f"executor for skill {skill_name!r} "
                "does not expose execute()"
            )

        execution = execute(
            skill_action
        )

        execution_ok = bool(
            getattr(
                execution,
                "ok",
                False,
            )
        )

        execution_detail = getattr(
            execution,
            "detail",
            "",
        )

        # ---------------------------------------------------------------
        # Executor failure
        # ---------------------------------------------------------------
        if not execution_ok:
            return ActionResult(
                action=action,
                ok=False,
                error=str(
                    execution_detail
                    or "skill execution failed"
                ),
                detail={
                    "skill": skill_name,
                    "skill_detail": execution_detail,
                    "skill_value": getattr(
                        execution,
                        "value",
                        None,
                    ),
                    "skill_executed": False,
                    "skill_verified": False,
                },
                failure_class=FailureClass.UNKNOWN,
            )

        # ---------------------------------------------------------------
        # Independent skill verification
        # ---------------------------------------------------------------
        verifier_factory = getattr(
            skill,
            "verifier",
            None,
        )

        if not callable(verifier_factory):
            raise TypeError(
                f"skill {skill_name!r} does not expose "
                "a callable verifier()"
            )

        verifier = verifier_factory()

        verify = getattr(
            verifier,
            "verify",
            None,
        )

        if not callable(verify):
            raise TypeError(
                f"verifier for skill {skill_name!r} "
                "does not expose verify()"
            )

        verification = verify(
            skill_action,
            execution,
        )

        verification_ok, verification_detail = (
            _normalize_skill_verification(
                verification
            )
        )

        # Current BrowserVerifier semantics:
        #
        #   ok=True
        #       -> skill-level verification passed
        #
        #   ok=False + no verification target
        #       -> task-level verification required
        #
        #   ok=False + verification target
        #       -> actual skill-level verification failure
        deferred = (
            not verification_ok
            and _skill_verification_is_deferred(
                skill_action
            )
        )

        loop.trace.emit(
            "skill_verification",
            skill=skill_name,
            action=action.kind,
            ok=verification_ok,
            deferred=deferred,
            detail=verification_detail,
        )

        # ---------------------------------------------------------------
        # Skill verification deferred to task checkpoint
        # ---------------------------------------------------------------
        if deferred:
            return ActionResult(
                action=action,
                ok=True,
                error=None,
                detail={
                    "skill": skill_name,
                    "skill_detail": execution_detail,
                    "skill_value": getattr(
                        execution,
                        "value",
                        None,
                    ),
                    "skill_executed": True,
                    "skill_verified": False,
                    "skill_verification_deferred": True,
                    "skill_verification": {
                        "status": "deferred",
                        "detail": verification_detail,
                    },
                },
                failure_class=None,
            )

        # ---------------------------------------------------------------
        # Actual skill verification failure
        # ---------------------------------------------------------------
        if not verification_ok:
            return ActionResult(
                action=action,
                ok=False,
                error=(
                    verification_detail
                    or "skill verification failed"
                ),
                detail={
                    "skill": skill_name,
                    "skill_detail": execution_detail,
                    "skill_value": getattr(
                        execution,
                        "value",
                        None,
                    ),
                    "skill_executed": True,
                    "skill_verified": False,
                    "skill_verification_deferred": False,
                    "skill_verification": {
                        "status": "failed",
                        "detail": verification_detail,
                    },
                },
                failure_class=FailureClass.UNKNOWN,
            )

        # ---------------------------------------------------------------
        # Skill verification passed
        # ---------------------------------------------------------------
        return ActionResult(
            action=action,
            ok=True,
            error=None,
            detail={
                "skill": skill_name,
                "skill_detail": execution_detail,
                "skill_value": getattr(
                    execution,
                    "value",
                    None,
                ),
                "skill_executed": True,
                "skill_verified": True,
                "skill_verification_deferred": False,
                "skill_verification": {
                    "status": "passed",
                    "detail": verification_detail,
                },
            },
            failure_class=None,
        )

    except Exception as exc:  # noqa: BLE001
        loop.trace.note(
            "skill_execution_exception",
            skill=skill_name,
            action=action.kind,
            error=f"{type(exc).__name__}: {exc}",
        )

        return ActionResult(
            action=action,
            ok=False,
            error=(
                f"skill {skill_name!r} execution failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            detail={
                "skill": skill_name,
                "skill_executed": False,
                "skill_verified": False,
                "exception_type": type(exc).__name__,
            },
            failure_class=FailureClass.UNKNOWN,
        )


# ---------------------------------------------------------------------------
# Policy + execution
# ---------------------------------------------------------------------------


def _permit_and_execute(
    loop: _Loop,
    action: Action,
) -> ActionResult:
    """Check policy and execute the approved action."""

    decision, reason = loop.policy.check(
        action
    )

    loop.trace.policy(
        action,
        decision.value,
        reason,
    )

    if decision is not Decision.ALLOW:
        return ActionResult(
            action=action,
            ok=False,
            error=(
                f"policy {decision.value}: {reason}"
            ),
            failure_class=FailureClass.PERMISSION_DENIED,
        )

    if loop.skills is not None:
        skill = loop.skills.find_for_action(
            action.kind
        )

        if skill is not None:
            return _execute_skill_action(
                loop,
                action,
                skill,
            )

    return os_tools.execute(
        loop.policy,
        action,
    )


# ---------------------------------------------------------------------------
# Action execution + verification
# ---------------------------------------------------------------------------


def _run_action(
    loop: _Loop,
    action: Action,
) -> tuple[
    ActionResult | None,
    VerificationResult | None,
]:
    """Run one action with bounded recovery and checkpoint verification."""

    recovering: FailureClass | None = None
    checkpoint: VerificationResult | None = None

    for attempt in range(
        _MAX_ATTEMPTS_PER_ACTION
    ):
        stale = _precondition_failure(
            loop,
            action,
        )

        if stale is not None:
            recovering = stale

            if _dispatch(
                loop,
                stale,
                attempt,
            ):
                continue

            return None, checkpoint

        age = oldest_age_s(
            loop.observations
        )

        loop.enter(
            AgentState.ACTING,
            action=action.kind,
            params=action.params,
        )

        result = _permit_and_execute(
            loop,
            action,
        )

        loop.trace.action(
            result,
            precondition_age_s=age,
        )

        loop.enter(
            AgentState.VERIFYING,
            action=action.kind,
            params=action.params,
        )

        checkpoint = loop.task.verify_checkpoint(
            loop.policy,
            action,
            loop.trace,
        )

        if checkpoint is not None:
            loop.trace.verification(
                checkpoint,
                checkpoint=True,
            )

            loop.checkpoints.append(
                checkpoint
            )

        # Execution succeeded and either:
        #
        # 1. there is no task-level checkpoint verifier, or
        #
        # 2. the task-level checkpoint independently confirmed the effect.
        #
        # A skill-level verification failure keeps result.ok=False and
        # therefore cannot be masked by a passing task checkpoint.
        if (
            result.ok
            and (
                checkpoint is None
                or checkpoint.verdict is Verdict.PASS
            )
        ):
            if recovering is not None:
                loop.trace.recovery_resolved(
                    failure_class=recovering,
                    attempt=attempt,
                    budget_left=(
                        loop.recovery.budget.remaining()
                    ),
                )

            # A previous failed attempt may have left a clarification behind.
            # A successful retry resolves it.
            loop.question = None

            loop.enter(
                AgentState.OBSERVING,
                purpose="baseline",
            )

            loop.refresh()

            return result, checkpoint

        # Preserve a clarification supplied by the executor.
        loop.question = _clarification_in(
            result
        )

        recovering = classify(
            result,
            checkpoint,
            precondition_age_s=age,
            max_staleness_s=loop.config.max_staleness_s,
        )

        if not _dispatch(
            loop,
            recovering,
            attempt,
        ):
            return None, checkpoint

    # The fixed per-action ceiling is reached regardless of the recovery
    # policy's remaining budget.
    loop.abort_reason = (
        f"attempt ceiling reached for action "
        f"{action.kind!r} "
        f"after {_MAX_ATTEMPTS_PER_ACTION} attempts"
    )

    loop.trace.note(
        "action_attempt_ceiling",
        action=action.kind,
        attempts=_MAX_ATTEMPTS_PER_ACTION,
    )

    return None, checkpoint


# ---------------------------------------------------------------------------
# Planner helpers
# ---------------------------------------------------------------------------


def _planner_name(
    planner: Planner,
) -> str:
    """Return a stable planner name for metrics and traces."""

    name = getattr(
        planner,
        "name",
        None,
    )

    if isinstance(name, str) and name.strip():
        return name

    return type(planner).__name__


def _planner_usage_snapshot(
    planner: Planner,
) -> tuple[int, int]:
    """Safely read planner token counters."""

    usage = getattr(
        planner,
        "usage",
        None,
    )

    if usage is None:
        return 0, 0

    return (
        int(
            getattr(
                usage,
                "prompt_tokens",
                0,
            )
            or 0
        ),
        int(
            getattr(
                usage,
                "completion_tokens",
                0,
            )
            or 0
        ),
    )


def _plan(
    loop: _Loop,
    planner: Planner,
    history: list[dict],
) -> PlannerStep:
    """Ask the planner for the next step and record planner telemetry."""

    before_in, before_out = _planner_usage_snapshot(
        planner
    )

    began = time.time()

    state = summarize(
        loop.observations
    )

    if loop.extra_state:
        state = {
            **loop.extra_state,
            **state,
        }

    try:
        step = planner.plan(
            loop.task.goal,
            state,
            history,
        )

    except Exception as exc:  # noqa: BLE001
        latency = time.time() - began

        loop.trace.planner_call(
            planner=_planner_name(planner),
            kind="plan",
            prompt_tokens=0,
            completion_tokens=0,
            latency_s=latency,
            error=(
                f"{type(exc).__name__}: {exc}"
            ),
        )

        loop.trace.note(
            "planner_exception",
            error=(
                f"{type(exc).__name__}: {exc}"
            ),
        )

        raise

    after_in, after_out = _planner_usage_snapshot(
        planner
    )

    loop.trace.planner_call(
        planner=_planner_name(planner),
        kind="plan",
        prompt_tokens=max(
            0,
            after_in - before_in,
        ),
        completion_tokens=max(
            0,
            after_out - before_out,
        ),
        latency_s=time.time() - began,
        error=step.error,
    )

    loop.trace.emit(
        "planner_step",
        done=step.done,
        reasoning=(step.reasoning or "")[:1000],
        error=step.error,
        state_fingerprint=(
            loop.fingerprint[:16]
        ),
        actions=[
            {
                "kind": action.kind,
                "params": action.params,
            }
            for action in step.actions
        ],
        rejected=list(
            step.rejected
        ),
    )

    return step


def _history_entry(
    action: Action,
    result: ActionResult | None,
    checkpoint: VerificationResult | None,
    *,
    skipped_verified: bool = False,
) -> dict[str, Any]:
    """Create a stable planner-history record."""

    return {
        "action": {
            "kind": action.kind,
            "params": action.params,
        },
        "ok": bool(
            (result and result.ok)
            or skipped_verified
        ),
        "error": (
            None
            if skipped_verified
            else result.error
            if result
            else "given up on"
        ),
        "detail": (
            result.detail
            if result
            else {}
        ),
        "checkpoint": (
            checkpoint.verdict.value
            if checkpoint
            else None
        ),
        "skipped_verified": skipped_verified,
    }


# ---------------------------------------------------------------------------
# Main planner loop
# ---------------------------------------------------------------------------


def _drive(
    loop: _Loop,
    planner: Planner,
    history: list[dict],
) -> tuple[
    bool,
    int,
    str,
]:
    """Drive the planner until completion, suspension, or failure.

    Returns:

        ``(reported_success, steps_used, final_reasoning)``

    ``final_reasoning`` is populated only when the planner explicitly reports
    completion.
    """

    from .general_task import GeneralTask

    steps_used = 0

    for index in range(
        loop.config.max_steps
    ):
        # Assignment to the loop attribute must be separate from the local
        # assignment; ``(loop.steps_used := ...)`` is invalid Python syntax.
        steps_used = index + 1
        loop.steps_used = steps_used

        loop.enter(
            AgentState.PLANNING,
            step=index,
        )

        try:
            step = _plan(
                loop,
                planner,
                history,
            )

        except Exception as exc:  # noqa: BLE001
            if (
                isinstance(loop.task, GeneralTask)
                and loop.task.requested_effects_complete()
            ):
                loop.trace.note(
                    "planner_failed_after_verified_completion",
                    error=(
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

                loop.decide(
                    DecisionKind.VERIFY,
                    "all explicitly requested tracked "
                    "effects already passed",
                )

                return (
                    False,
                    steps_used,
                    "",
                )

            if isinstance(
                loop.task,
                GeneralTask,
            ):
                loop.task.mark_completion_uncertain(
                    "the planner became unavailable before all "
                    "explicitly requested effects were established"
                )

            loop.decide(
                DecisionKind.STOP,
                "the planner raised an exception",
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )

            # Planner failures are not action failures. Route them through
            # the recovery policy as UNKNOWN so the terminal reason is
            # consistent with the recovery taxonomy.
            _dispatch(
                loop,
                FailureClass.UNKNOWN,
                index,
            )

            return (
                False,
                steps_used,
                "",
            )

        if step.error:
            if (
                isinstance(
                    loop.task,
                    GeneralTask,
                )
                and loop.task.requested_effects_complete()
            ):
                loop.trace.note(
                    "planner_failed_after_verified_completion",
                    error=step.error,
                )

                loop.decide(
                    DecisionKind.VERIFY,
                    "all explicitly requested tracked "
                    "effects already passed",
                )

                return (
                    False,
                    steps_used,
                    "",
                )

            if isinstance(
                loop.task,
                GeneralTask,
            ):
                loop.task.mark_completion_uncertain(
                    "the planner became unavailable before all "
                    "explicitly requested effects were established"
                )

            loop.decide(
                DecisionKind.STOP,
                "the planner returned no usable step",
                error=step.error,
            )

            # A planner-reported error is also an UNKNOWN failure rather
            # than a planner-specific abort reason.
            _dispatch(
                loop,
                FailureClass.UNKNOWN,
                index,
            )

            return (
                False,
                steps_used,
                "",
            )

        # ``done`` is a planner claim, not independent proof of success.
        #
        # A planner may legitimately return the final action batch together
        # with ``done=True``. The actions still have to pass through policy,
        # execution, and checkpoint verification. The important part is that
        # we remember the completion claim so that we STOP REPLANNING after
        # the final action batch has been processed.
        planner_done = bool(step.done)

        if (
            planner_done
            and step.actions
        ):
            loop.trace.emit(
                "planner_claimed_done_with_pending_actions",
                step=index,
                actions=[
                    action.kind
                    for action in step.actions
                ],
            )

        elif planner_done:
            loop.trace.note(
                "planner_reported_done",
                step=index,
            )

        if not step.actions:
            if planner_done:
                loop.decide(
                    DecisionKind.VERIFY,
                    "the planner claims the goal is met; "
                    "verification decides",
                )

                return (
                    True,
                    steps_used,
                    step.reasoning or "",
                )

            if step.rejected:
                reason = (
                    "the planner proposed no executable actions; "
                    "rejected unsupported kinds: "
                    f"{step.rejected}"
                )

                loop.trace.emit(
                    "planner_rejected_actions",
                    step=index,
                    rejected=list(
                        step.rejected
                    ),
                )

            else:
                reason = (
                    "the planner neither acted nor "
                    "claimed completion"
                )

            loop.decide(
                DecisionKind.STOP,
                reason,
            )

            loop.abort_reason = (
                "planner produced no executable actions"
                + (
                    f"; rejected unsupported kinds: "
                    f"{step.rejected}"
                    if step.rejected
                    else " and did not claim completion"
                )
            )

            return (
                False,
                steps_used,
                "",
            )

        loop.decide(
            DecisionKind.ACT,
            "the next planned step is executable",
            actions=[
                action.kind
                for action in step.actions
            ],
        )

        for action in step.actions:
            if isinstance(
                loop.task,
                GeneralTask,
            ):
                already = (
                    loop.task.verified_equivalent(
                        loop.policy,
                        action,
                        loop.trace,
                    )
                )

                if already is not None:
                    loop.trace.verification(
                        already,
                        checkpoint=True,
                    )

                    loop.checkpoints.append(
                        already
                    )

                    loop.trace.emit(
                        "action_skipped",
                        action=action.to_json(),
                        reason=(
                            "equivalent effect already verified "
                            "PASS and still holds"
                        ),
                    )

                    history.append(
                        _history_entry(
                            action,
                            None,
                            already,
                            skipped_verified=True,
                        )
                    )

                    loop.enter(
                        AgentState.OBSERVING,
                        purpose="baseline",
                    )

                    loop.refresh()

                    continue

            result, checkpoint = _run_action(
                loop,
                action,
            )

            history.append(
                _history_entry(
                    action,
                    result,
                    checkpoint,
                )
            )

            if result is None:
                break

        # The action loop can suspend waiting for a person.
        if loop.question is not None:
            return (
                False,
                steps_used,
                "",
            )

        # A hard failure stops the run.
        if loop.abort_reason:
            return (
                False,
                steps_used,
                "",
            )

        # IMPORTANT:
        #
        # If the planner marked this action batch as its final batch, do not
        # call the planner again. We have already executed and independently
        # checkpoint-verified the actions above. The public ``run_task()``
        # function will perform the final task verification from the world
        # state, so the planner's ``done`` claim is never trusted as proof.
        if planner_done:
            loop.trace.note(
                "planner_reported_done",
                step=index,
            )

            loop.decide(
                DecisionKind.VERIFY,
                "the planner claims this was the final action batch; "
                "verification decides",
            )

            return (
                True,
                steps_used,
                step.reasoning or "",
            )

        # A recovery decision may request replanning. The next iteration
        # naturally calls the planner again using the refreshed state/history.
        if loop.replan:
            loop.trace.note(
                "replanning",
                step=index,
                reason="recovery requested replanning",
            )

            loop.replan = False

    loop.abort_reason = (
        loop.abort_reason
        or (
            f"step ceiling "
            f"({loop.config.max_steps}) reached"
        )
    )

    loop.trace.note(
        "step_ceiling_reached",
        max_steps=loop.config.max_steps,
    )

    return (
        False,
        steps_used,
        "",
    )


# ---------------------------------------------------------------------------
# Final verification helpers
# ---------------------------------------------------------------------------


def _harness_error(
    task_id: str,
    exc: BaseException,
) -> VerificationResult:
    return VerificationResult(
        label=task_id,
        checks=[
            Check(
                name="run_completed",
                verdict=Verdict.UNKNOWN,
                evidence={},
                reason=(
                    f"run raised "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        ],
    )


def _cancelled(
    task_id: str,
) -> VerificationResult:
    return VerificationResult(
        checks=[],
        label=f"cancelled:{task_id}",
    )


def _awaiting(
    task_id: str,
    label: str = "awaiting",
) -> VerificationResult:
    return VerificationResult(
        checks=[],
        label=f"{label}:{task_id}",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_task(
    task: Task,
    planner: Planner,
    policy: Policy,
    config: RunConfig | None = None,
    *,
    trial: int = 0,
    trace: Trace | None = None,
    synthetic: bool = False,
    teardown: bool = True,
    extra_state: dict | None = None,
    skills: SkillRegistry | None = None,
) -> RunOutcome:
    """Run one complete task trial.

    The function owns lifecycle management but delegates actual behavior to
    the planner, policy, skills, task verifier, and recovery policy.
    """

    config = config or RunConfig()

    close_trace = trace is None

    trace = trace or Trace(
        task_id=task.task_id,
        condition=config.condition,
        trial=trial,
    )

    loop = _Loop(
        task=task,
        policy=policy,
        config=config,
        trace=trace,
        recovery=Recovery(
            budget=config.budget,
            enabled=config.recovery_enabled,
            interactive=config.interactive,
        ),
        skills=skills,
        extra_state=dict(
            extra_state or {}
        ),
    )

    started = time.time()

    reported_success = False
    steps_used = 0
    cancelled = False
    awaiting = False

    final_reasoning = ""
    informational_output = ""
    informational_completion = False

    history: list[dict] = []

    trace.emit(
        "run_start",
        goal=task.goal,
        bucket=getattr(
            task,
            "bucket",
            "",
        ),
        planner=_planner_name(planner),
        config=config.to_json(),
        workspace=str(
            policy.workspace
        ),
        refuse_if_elevated=(
            policy.refuse_if_elevated
        ),
        skills=(
            list(skills.names())
            if skills is not None
            else []
        ),
    )

    final: VerificationResult

    try:
        loop.enter(
            AgentState.OBSERVING,
            purpose="initial",
        )

        task.setup(policy)

        loop.refresh()

        (
            reported_success,
            steps_used,
            final_reasoning,
        ) = _drive(
            loop,
            planner,
            history,
        )

        if loop.question is not None:
            awaiting = True

            final = _awaiting(
                task.task_id
            )

        elif loop.abort_reason is not None:
            # Diagnostic verification is useful after an abort because an
            # earlier action may nevertheless have achieved the goal.
            # Do not send this through the normal completion-verification
            # trace channel because the run has already aborted.
            final = task.verify_final(
                policy,
                trace,
            )

            trace.note(
                "diagnostic_verification_after_abort",
                verdict=final.verdict.value,
                abort_reason=loop.abort_reason,
            )

        else:
            loop.enter(
                AgentState.VERIFYING,
                purpose="final",
            )

            final = task.verify_final(
                policy,
                trace,
            )

            trace.verification(
                final,
                checkpoint=False,
            )

    except KeyboardInterrupt:
        cancelled = True
        steps_used = loop.steps_used
        loop.question = None

        trace.note(
            "run_cancelled",
            reason="KeyboardInterrupt",
            steps_used=steps_used,
            reported_success=reported_success,
        )

        loop.abort_reason = (
            "cancelled by user before verification "
            "finished"
        )

        final = _cancelled(
            task.task_id
        )

    except NeedUserInput as need:
        steps_used = loop.steps_used

        decision, reason = loop.recovery.decide(
            FailureClass.AMBIGUOUS
        )

        if decision is RecoveryDecision.ASK_USER:
            awaiting = True
            loop.question = need.clarification

            trace.emit(
                "clarification_needed",
                failure_class=FailureClass.AMBIGUOUS.value,
                attempt=steps_used,
                question=need.clarification.to_json(),
            )

            loop.decide(
                DecisionKind.ASK_USER,
                "only a person can choose the continuation",
            )

            loop.enter(
                AgentState.WAITING_FOR_USER
            )

            final = _awaiting(
                task.task_id
            )

        else:
            loop.question = None

            trace.failure(
                FailureClass.AMBIGUOUS,
                question=need.clarification.to_json(),
                steps_used=steps_used,
            )

            loop.abort_reason = (
                f"{FailureClass.AMBIGUOUS.value}: "
                f"{reason}"
            )

            final = _awaiting(
                task.task_id,
                "ambiguous",
            )

    except Exception as exc:  # noqa: BLE001
        steps_used = loop.steps_used
        loop.question = None

        trace.note(
            "run_exception",
            error=(
                f"{type(exc).__name__}: {exc}"
            ),
        )

        loop.abort_reason = (
            loop.abort_reason
            or (
                f"harness error: "
                f"{type(exc).__name__}: {exc}"
            )
        )

        final = _harness_error(
            task.task_id,
            exc,
        )

    finally:
        if (
            teardown
            and not cancelled
            and not awaiting
        ):
            try:
                task.teardown(
                    policy
                )

            except Exception as exc:  # noqa: BLE001
                trace.note(
                    "teardown_failed",
                    error=(
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

    from .general_task import GeneralTask

    inspection_completed = (
        isinstance(
            task,
            GeneralTask,
        )
        and task.inspection_only()
        and reported_success
        and loop.abort_reason is None
        and not cancelled
        and not awaiting
    )

    if inspection_completed:
        informational_output = (
            final_reasoning
        )

        informational_completion = True

    # Determine the terminal state only after all lifecycle handling is done.
    if cancelled:
        terminal_state = AgentState.CANCELLED

    elif awaiting:
        terminal_state = AgentState.WAITING_FOR_USER

    elif loop.abort_reason is not None:
        terminal_state = AgentState.FAILED

    elif (
        final.verdict is Verdict.PASS
        or inspection_completed
    ):
        terminal_state = AgentState.COMPLETED

    else:
        terminal_state = AgentState.FAILED

    loop.enter(
        terminal_state
    )

    outcome = RunOutcome(
        task_id=task.task_id,
        condition=config.condition,
        trial=trial,
        planner_name=_planner_name(
            planner
        ),
        reported_success=reported_success,
        verified=final.verdict,
        final=final,
        checkpoints=list(
            loop.checkpoints
        ),
        failure_categories=sorted(
            trace.failure_categories.elements()
        ),
        steps_used=steps_used,
        wall_clock_s=(
            time.time() - started
        ),
        usage=(
            planner.usage.to_json()
            if getattr(
                planner,
                "usage",
                None,
            )
            else {}
        ),
        trace=trace.summary(),
        aborted_reason=loop.abort_reason,
        synthetic=(
            synthetic
            or _planner_name(
                planner
            ).startswith("mock")
        ),
        cancelled=cancelled,
        state=loop.state,
        awaiting=awaiting,
        question=(
            loop.question
            if awaiting
            else None
        ),
        informational_output=(
            informational_output
        ),
        informational_completion=(
            informational_completion
        ),
    )

    trace.emit(
        "run_end",
        outcome={
            key: value
            for key, value in outcome.to_json().items()
            if key not in (
                "trace",
                "checkpoints",
            )
        },
    )

    if close_trace:
        trace.close()

    return outcome