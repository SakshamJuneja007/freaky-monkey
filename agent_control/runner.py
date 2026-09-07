"""The control loop: observe -> select -> permit -> execute -> verify -> recover.

This is the structured-first condition of the experiment (plan S18). The
vision-only baseline lives in ``benchmark/baselines/vision_only.py`` and shares
this module's task contract, verifiers, and metrics so the two conditions are
scored identically.

Three switches exist purely so the ablations in plan S18 are the *same* code path
with a flag flipped, never a second implementation:

* ``fresh_precondition`` -- re-read state immediately before consequential
  actions (hypothesis H2).
* ``recovery_enabled``   -- bounded retry/re-observe/abort (hypothesis H3).
* ``max_staleness_s``    -- how old a reading may be before it is refused.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from . import os_tools
from .observe import summarize
from .planner.base import Planner, PlannerStep
from .policy import Decision, Policy
from .recovery import Recovery, RecoveryBudget, RecoveryDecision, classify
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


@dataclass
class RunConfig:
    condition: str = "structured_hybrid"
    max_steps: int = 8
    max_staleness_s: float = DEFAULT_MAX_STALENESS_S
    fresh_precondition: bool = True
    recovery_enabled: bool = True
    budget: RecoveryBudget = field(default_factory=RecoveryBudget)
    interactive: bool = False

    def to_json(self) -> dict:
        return {
            "condition": self.condition,
            "max_steps": self.max_steps,
            "max_staleness_s": self.max_staleness_s,
            "fresh_precondition": self.fresh_precondition,
            "recovery_enabled": self.recovery_enabled,
            "interactive": self.interactive,
            "budget": self.budget.to_json()["limits"],
        }


@dataclass
class RunOutcome:
    """One trial. ``false_success`` is the headline metric of plan S19."""

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

    #: Planner's final findings for an inspection-only run. This is not
    #: verification evidence and does not affect ``verified``.
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
        are excluded.
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

    def to_json(self) -> dict:
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
            "question": self.question.to_json() if self.question else None,
            "steps_used": self.steps_used,
            "wall_clock_s": round(self.wall_clock_s, 4),
            "aborted_reason": self.aborted_reason,
            "failure_categories": self.failure_categories,
            "usage": self.usage,
            "final_verification": self.final.to_json(),
            "checkpoints": [c.to_json() for c in self.checkpoints],
            "trace": self.trace,
            "synthetic": self.synthetic,
        }


@dataclass
class _Loop:
    """Mutable loop context threaded through per-action attempts."""

    task: Task
    policy: Policy
    config: RunConfig
    trace: Trace
    recovery: Recovery
    observations: dict = field(default_factory=dict)
    fingerprint: str = ""
    checkpoints: list[VerificationResult] = field(default_factory=list)
    abort_reason: str | None = None
    replan: bool = False
    steps_used: int = 0
    extra_state: dict = field(default_factory=dict)
    state: AgentState = AgentState.IDLE
    question: Clarification | None = None

    def refresh(self) -> float:
        """Re-read state, re-baseline the fingerprint."""
        self.observations = self.task.observe(self.policy, self.trace)
        self.fingerprint = state_fingerprint(self.observations)
        return oldest_age_s(self.observations)

    def enter(self, state: AgentState, **fields: Any) -> AgentState:
        if state is self.state:
            return state

        previous, self.state = self.state, state
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
        decision = AgentDecision(
            kind=kind,
            reason=reason,
            detail=dict(detail),
        )
        self.trace.emit("decision", **decision.to_json())
        return decision


_MAX_ATTEMPTS_PER_ACTION = 4


_DECISION_KIND = {
    RecoveryDecision.RETRY: DecisionKind.RECOVER,
    RecoveryDecision.REOBSERVE_THEN_RETRY: DecisionKind.OBSERVE,
    RecoveryDecision.REPLAN: DecisionKind.REPLAN,
    RecoveryDecision.ASK_USER: DecisionKind.ASK_USER,
    RecoveryDecision.ABORT: DecisionKind.STOP,
}


def _clarification_in(
    result: ActionResult | None,
) -> Clarification | None:
    if result is None:
        return None

    found = result.detail.get("clarification")
    return found if isinstance(found, Clarification) else None


def _dispatch(
    loop: _Loop,
    failure_class: FailureClass,
    attempt: int,
) -> bool:
    decision, reason = loop.recovery.decide(failure_class)

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
        loop.enter(AgentState.WAITING_FOR_USER)
        return False

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
        loop.enter(AgentState.OBSERVING)
        loop.refresh()
        return True

    if decision is RecoveryDecision.RETRY:
        return True

    if decision is RecoveryDecision.REPLAN:
        loop.enter(AgentState.OBSERVING)
        loop.refresh()
        loop.replan = True
        return False

    loop.question = None
    loop.abort_reason = f"{failure_class.value}: {reason}"
    return False


def _precondition_failure(
    loop: _Loop,
    action: Action,
) -> FailureClass | None:
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

    if before and loop.fingerprint != before:
        loop.trace.note(
            "state_changed_during_reasoning",
            action=action.kind,
            before=before[:16],
            after=loop.fingerprint[:16],
        )
        return FailureClass.STALE_STATE

    return None


def _permit_and_execute(
    loop: _Loop,
    action: Action,
) -> ActionResult:
    decision, reason = loop.policy.check(action)
    loop.trace.policy(action, decision.value, reason)

    if decision is not Decision.ALLOW:
        return ActionResult(
            action=action,
            ok=False,
            error=f"policy {decision.value}: {reason}",
            failure_class=FailureClass.PERMISSION_DENIED,
        )

    return os_tools.execute(loop.policy, action)


def _run_action(
    loop: _Loop,
    action: Action,
) -> tuple[
    ActionResult | None,
    VerificationResult | None,
]:
    recovering: FailureClass | None = None
    checkpoint: VerificationResult | None = None

    for attempt in range(_MAX_ATTEMPTS_PER_ACTION):
        stale = _precondition_failure(loop, action)

        if stale is not None:
            recovering = stale

            if _dispatch(loop, stale, attempt):
                continue

            return None, checkpoint

        age = oldest_age_s(loop.observations)

        loop.enter(
            AgentState.ACTING,
            action=action.kind,
            params=action.params,
        )

        result = _permit_and_execute(loop, action)

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
            loop.checkpoints.append(checkpoint)

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
                    budget_left=loop.recovery.budget.remaining(),
                )

            loop.enter(
                AgentState.OBSERVING,
                purpose="baseline",
            )

            loop.refresh()
            return result, checkpoint

        loop.question = _clarification_in(result)

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

    loop.abort_reason = (
        f"attempt ceiling reached for action {action.kind!r}"
    )

    return None, checkpoint


def _planner_name(planner: Planner) -> str:
    return getattr(
        planner,
        "name",
        type(planner).__name__,
    )


def _plan(
    loop: _Loop,
    planner: Planner,
    history: list[dict],
) -> PlannerStep:
    usage = getattr(planner, "usage", None)

    before_in = usage.prompt_tokens if usage else 0
    before_out = usage.completion_tokens if usage else 0

    began = time.time()

    state = summarize(loop.observations)

    if loop.extra_state:
        state = {
            **loop.extra_state,
            **state,
        }

    step = planner.plan(
        loop.task.goal,
        state,
        history,
    )

    loop.trace.planner_call(
        planner=_planner_name(planner),
        kind="plan",
        prompt_tokens=(
            usage.prompt_tokens - before_in
            if usage
            else 0
        ),
        completion_tokens=(
            usage.completion_tokens - before_out
            if usage
            else 0
        ),
        latency_s=time.time() - began,
        error=step.error,
    )

    loop.trace.emit(
        "planner_step",
        done=step.done,
        reasoning=step.reasoning[:1000],
        error=step.error,
        state_fingerprint=loop.fingerprint[:16],
        actions=[
            {
                "kind": action.kind,
                "params": action.params,
            }
            for action in step.actions
        ],
    )

    return step


def _history_entry(
    action: Action,
    result: ActionResult | None,
    checkpoint: VerificationResult | None,
    *,
    skipped_verified: bool = False,
) -> dict:
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


def _drive(
    loop: _Loop,
    planner: Planner,
    history: list[dict],
) -> tuple[bool, int, str]:
    """The planner loop.

    Returns ``(reported_success, steps_used, final_reasoning)``.
    ``final_reasoning`` is populated only when the planner explicitly reports
    completion.
    """
    steps_used = 0

    for index in range(loop.config.max_steps):
        steps_used = loop.steps_used = index + 1

        loop.enter(
            AgentState.PLANNING,
            step=index,
        )

        step = _plan(
            loop,
            planner,
            history,
        )

        if step.error:
            from .general_task import GeneralTask

            if (
                isinstance(loop.task, GeneralTask)
                and loop.task.requested_effects_complete()
            ):
                loop.trace.note(
                    "planner_failed_after_verified_completion",
                    error=step.error,
                )

                loop.decide(
                    DecisionKind.VERIFY,
                    "all explicitly requested tracked effects already passed",
                )

                return False, steps_used, ""

            if isinstance(loop.task, GeneralTask):
                loop.task.mark_completion_uncertain(
                    "the planner became unavailable before all explicitly "
                    "requested effects were established"
                )

            loop.decide(
                DecisionKind.STOP,
                "the planner returned no usable step",
            )

            _dispatch(
                loop,
                FailureClass.UNKNOWN,
                index,
            )

            loop.abort_reason = (
                loop.abort_reason
                or f"planner error: {step.error}"
            )

            return False, steps_used, ""

        if step.done and step.actions:
            # Actions in this step have not executed yet, so the planner's
            # same-turn completion claim cannot suppress them. Execute first;
            # independent verification remains the authority on success.
            loop.trace.emit(
                "planner_claimed_done_with_pending_actions",
                step=index,
                actions=[action.kind for action in step.actions],
            )

        elif step.done:
            loop.trace.note(
                "planner_reported_done",
                step=index,
            )

            loop.decide(
                DecisionKind.VERIFY,
                "the planner claims the goal is met; verification decides",
            )

            return True, steps_used, step.reasoning

        if not step.actions:
            if step.rejected:
                reason = (
                    "the planner proposed no executable actions; rejected "
                    f"unsupported kinds: {step.rejected}"
                )
                loop.trace.emit(
                    "planner_rejected_actions",
                    step=index,
                    rejected=list(step.rejected),
                )
            else:
                reason = "the planner neither acted nor claimed completion"

            loop.decide(DecisionKind.STOP, reason)

            loop.abort_reason = (
                "planner produced no executable actions"
                + (f"; rejected unsupported kinds: {step.rejected}"
                   if step.rejected else " and did not claim completion")
            )

            return False, steps_used, ""

        loop.decide(
            DecisionKind.ACT,
            "the next planned step is executable",
            actions=[
                action.kind
                for action in step.actions
            ],
        )

        for action in step.actions:
            from .general_task import GeneralTask

            if isinstance(loop.task, GeneralTask):
                already = loop.task.verified_equivalent(
                    loop.policy,
                    action,
                    loop.trace,
                )

                if already is not None:
                    loop.trace.verification(
                        already,
                        checkpoint=True,
                    )

                    loop.checkpoints.append(already)

                    loop.trace.emit(
                        "action_skipped",
                        action=action.to_json(),
                        reason=(
                            "equivalent effect already verified PASS "
                            "and still holds"
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

        if loop.question is not None:
            return False, steps_used, ""

        if loop.abort_reason:
            return False, steps_used, ""

        if loop.replan:
            loop.replan = False

    loop.abort_reason = (
        loop.abort_reason
        or f"step ceiling ({loop.config.max_steps}) reached"
    )

    return False, steps_used, ""


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
) -> RunOutcome:
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
        extra_state=dict(extra_state or {}),
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
        bucket=getattr(task, "bucket", ""),
        planner=_planner_name(planner),
        config=config.to_json(),
        workspace=str(policy.workspace),
        refuse_if_elevated=policy.refuse_if_elevated,
    )

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
            final = _awaiting(task.task_id)

        elif loop.abort_reason is not None:
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
            "cancelled by user before verification finished"
        )

        final = _cancelled(task.task_id)

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

            final = _awaiting(task.task_id)

        else:
            trace.failure(
                FailureClass.AMBIGUOUS,
                question=need.clarification.to_json(),
                steps_used=steps_used,
            )

            loop.abort_reason = (
                f"{FailureClass.AMBIGUOUS.value}: {reason}"
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
            error=f"{type(exc).__name__}: {exc}",
        )

        loop.abort_reason = (
            loop.abort_reason
            or f"harness error: {type(exc).__name__}: {exc}"
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
                task.teardown(policy)

            except Exception as exc:  # noqa: BLE001
                trace.note(
                    "teardown_failed",
                    error=f"{type(exc).__name__}: {exc}",
                )

    from .general_task import GeneralTask

    inspection_completed = (
        isinstance(task, GeneralTask)
        and task.inspection_only()
        and reported_success
        and loop.abort_reason is None
        and not cancelled
        and not awaiting
    )

    if inspection_completed:
        informational_output = final_reasoning
        informational_completion = True

    loop.enter(
        AgentState.CANCELLED
        if cancelled
        else AgentState.WAITING_FOR_USER
        if awaiting
        else AgentState.FAILED
        if loop.abort_reason is not None
        else AgentState.COMPLETED
        if (
            final.verdict is Verdict.PASS
            or inspection_completed
        )
        else AgentState.FAILED
    )

    outcome = RunOutcome(
        task_id=task.task_id,
        condition=config.condition,
        trial=trial,
        planner_name=_planner_name(planner),
        reported_success=reported_success,
        verified=final.verdict,
        final=final,
        checkpoints=list(loop.checkpoints),
        failure_categories=sorted(
            trace.failure_categories.elements()
        ),
        steps_used=steps_used,
        wall_clock_s=time.time() - started,
        usage=(
            planner.usage.to_json()
            if getattr(planner, "usage", None)
            else {}
        ),
        trace=trace.summary(),
        aborted_reason=loop.abort_reason,
        synthetic=(
            synthetic
            or _planner_name(planner).startswith("mock")
        ),
        cancelled=cancelled,
        state=loop.state,
        awaiting=awaiting,
        question=(
            loop.question
            if awaiting
            else None
        ),
        informational_output=informational_output,
        informational_completion=informational_completion,
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