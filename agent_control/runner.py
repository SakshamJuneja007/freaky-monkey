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
    #: Someone is present to answer a question. Default False, so an unattended
    #: run -- every benchmark trial -- still ends rather than blocking on a person
    #: who is not there, and no measured number moves because this flag exists.
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
    #: What the agent claimed.
    reported_success: bool
    #: What independent verification found.
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
    #: The user interrupted before verification ran. Not a failure and not a
    #: result: the absence of one. Kept separate from ``verified`` and from
    #: ``failure_categories`` because a cancelled run is not evidence about the
    #: agent, and counting it as either would corrupt both the metrics table and
    #: the report a person reads (plan E phase 2).
    cancelled: bool = False
    #: Where the loop stopped. ``IDLE`` only for an outcome built without a run.
    state: AgentState = AgentState.IDLE
    #: The run stopped to ask a person something and can be resumed from here.
    #: Like ``cancelled`` this is the absence of a result rather than a bad one:
    #: nothing was verified, nothing is claimed, and no budget was spent.
    awaiting: bool = False
    #: What is being asked, when ``awaiting``. Carries only continuations that
    #: were actually observed (see :class:`~agent_control.types.Clarification`).
    question: Clarification | None = None

    @property
    def verified_success(self) -> bool:
        return self.verified is Verdict.PASS

    @property
    def false_success(self) -> bool:
        """Agent said done; independent verification disagreed.

        Cancelled and suspended runs are excluded. Ctrl+C can land after the
        planner claimed completion but before ``verify_final`` returns, and a run
        suspended on a question never reaches it at all; counting either as a false
        claim would blame the planner for a check *we* stopped -- inventing the one
        metric this project exists to measure.
        """
        return (self.reported_success and not self.verified_success
                and not self.cancelled and not self.awaiting)

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
    """Mutable loop context threaded through per-action attempts.

    Bundled into one object so the helpers below read as loop steps rather than
    as functions taking five positional dependencies.
    """

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
    #: Planner turns begun. Kept here rather than only as ``_drive``'s local so an
    #: interrupt, which unwinds ``_drive`` without letting it return, still leaves
    #: behind how far the run had got. Reporting 0 steps for a run that had already
    #: planned and dispatched an action would understate it in the one direction
    #: that matters: it would read as "nothing happened".
    steps_used: int = 0
    #: Read-only context supplied by the caller and merged into the planner's
    #: view of state -- currently remembered file locations (``memory.Recall``).
    #: Deliberately *not* part of ``observations``: it is not a reading of the
    #: world taken by this run, it takes no part in the state fingerprint, and it
    #: must never satisfy a freshness precondition.
    extra_state: dict = field(default_factory=dict)
    #: Where the loop is, as one word. Set at the junctions the loop already
    #: passes through -- this is a label on the existing control flow, not a
    #: second one, and nothing branches on it.
    state: AgentState = AgentState.IDLE
    #: A question the run cannot answer for itself. Once set, the run stops after
    #: the current action rather than executing steps that were planned on an
    #: assumption the user is about to settle.
    question: Clarification | None = None

    def refresh(self) -> float:
        """Re-read state, re-baseline the fingerprint, return the stalest age."""
        self.observations = self.task.observe(self.policy, self.trace)
        self.fingerprint = state_fingerprint(self.observations)
        return oldest_age_s(self.observations)

    def enter(self, state: AgentState, **fields: Any) -> AgentState:
        """Record that the loop has reached *state*. Returns it, for call sites.

        Called immediately *before* the work the state names, never after and never
        speculatively, so a display driven by these events can lag the machine but
        cannot describe something that did not happen. Re-entering the same state
        emits nothing: the repeat is already visible in the action and observation
        events, and a status line that reprints itself reads as noise.
        """
        if state is self.state:
            return state
        previous, self.state = self.state, state
        self.trace.emit("agent_state", state=state.value,
                        previous=previous.value, **fields)
        return state

    def decide(self, kind: DecisionKind, reason: str, **detail: Any) -> AgentDecision:
        """Record why the loop is about to do what it does next.

        ``reason`` is authored here, in the control plane, and never copied from a
        planner: the trace is a record of what the loop decided, and quoting the
        model's own prose into it would turn an audit log into hidden reasoning
        wearing a structured hat.
        """
        decision = AgentDecision(kind=kind, reason=reason, detail=dict(detail))
        self.trace.emit("decision", **decision.to_json())
        return decision


#: Belt-and-braces ceiling. Budget exhaustion already terminates the retry loop;
#: this guarantees termination even if a future strategy table forgets to.
_MAX_ATTEMPTS_PER_ACTION = 4

#: Recovery decision -> the decision kind recorded in the trace. Two vocabularies
#: exist because they answer different questions: ``RecoveryDecision`` is what the
#: budget was asked for, ``DecisionKind`` is what the loop does next, and
#: ``REOBSERVE_THEN_RETRY`` is an OBSERVE from the outside.
_DECISION_KIND = {
    RecoveryDecision.RETRY: DecisionKind.RECOVER,
    RecoveryDecision.REOBSERVE_THEN_RETRY: DecisionKind.OBSERVE,
    RecoveryDecision.REPLAN: DecisionKind.REPLAN,
    RecoveryDecision.ASK_USER: DecisionKind.ASK_USER,
    RecoveryDecision.ABORT: DecisionKind.STOP,
}


def _clarification_in(result: ActionResult | None) -> Clarification | None:
    """A question an executor reported as data rather than by raising.

    Two routes into a suspended run exist because two kinds of code find
    ambiguity. A task's observer has nowhere to put a return value, so it raises
    :class:`~agent_control.types.NeedUserInput`; an executor is already returning
    an ``ActionResult`` and can carry the question inside it. Only a real
    ``Clarification`` is accepted -- a dict shaped like one is not, because the
    whole point of the type is that its options came from something observed.
    """
    if result is None:
        return None
    found = result.detail.get("clarification")
    return found if isinstance(found, Clarification) else None


def _dispatch(loop: _Loop, failure_class: FailureClass, attempt: int) -> bool:
    """Apply one recovery decision. Returns True if the action should be retried.

    Every branch is logged with the budget remaining at the time, so plan S19's
    "recovery attempts and recovery success rate" comes out of the trace rather
    than out of a counter someone remembered to increment.
    """
    decision, reason = loop.recovery.decide(failure_class)

    if decision is RecoveryDecision.ASK_USER and loop.question is not None:
        # Reachable only with an interactive approver *and* an observed question.
        # Logged as a clarification rather than through ``trace.recovery``, which
        # feeds the failure-category column and the recovery success rate: a
        # question nobody has answered yet is neither a failure nor an attempt at
        # anything. Nothing is spent here either -- ``RecoveryBudget.spend`` has no
        # branch for ASK_USER -- so hesitating cannot exhaust a ceiling.
        loop.trace.emit("clarification_needed", failure_class=failure_class.value,
                        attempt=attempt, question=loop.question.to_json())
        loop.decide(DecisionKind.ASK_USER, "only a person can choose the continuation",
                    failure_class=failure_class.value)
        loop.enter(AgentState.WAITING_FOR_USER)
        return False

    loop.enter(AgentState.RECOVERING, failure_class=failure_class.value)
    loop.trace.recovery(
        failure_class=failure_class, decision=decision.value,
        attempt=attempt, budget_left=loop.recovery.budget.remaining(),
    )
    loop.decide(_DECISION_KIND[decision], reason,
                failure_class=failure_class.value, attempt=attempt)

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
    # Either an outright ABORT, or an ASK_USER with nothing to ask: a policy
    # refusal arrives here, and a refusal is not a choice. Turning one into a
    # question would invite a person to approve what Policy already denied, which
    # would move the authority for access out of the policy layer (plan S11).
    # Any question parked earlier is dropped, because this reason supersedes it.
    loop.question = None
    loop.abort_reason = f"{failure_class.value}: {reason}"
    return False


def _precondition_failure(loop: _Loop, action: Action) -> FailureClass | None:
    """Freshness gate for consequential actions (plan S3 bullet 2, hypothesis H2).

    Two distinct refusals, both classified STALE_STATE:

    * the reading the planner acted on is older than the staleness budget;
    * the world's fingerprint moved between the planner seeing it and now --
      someone else changed the machine, or the last action is still settling.

    With ``fresh_precondition=False`` (the H2 ablation) neither check runs and the
    action goes ahead on whatever the planner last saw.
    """
    if not (action.consequential and loop.config.fresh_precondition):
        return None

    before = loop.fingerprint
    loop.enter(AgentState.OBSERVING, purpose="precondition")
    age = loop.refresh()
    if age > loop.config.max_staleness_s:
        loop.trace.note("stale_reading_refused", age_s=round(age, 4),
                        max_staleness_s=loop.config.max_staleness_s, action=action.kind)
        return FailureClass.STALE_STATE
    if before and loop.fingerprint != before:
        loop.trace.note("state_changed_during_reasoning", action=action.kind,
                        before=before[:16], after=loop.fingerprint[:16])
        return FailureClass.STALE_STATE
    return None


def _permit_and_execute(loop: _Loop, action: Action) -> ActionResult:
    """The permit step. Policy is consulted here even though os_tools asks again.

    Defence in depth is the point: the runner is where an injected action is
    stopped before it reaches an executor, and where the decision is audited
    (plan S10 "Audit", S11 "enforce permissions outside the model").
    """
    decision, reason = loop.policy.check(action)
    loop.trace.policy(action, decision.value, reason)
    if decision is not Decision.ALLOW:
        return ActionResult(
            action=action, ok=False,
            error=f"policy {decision.value}: {reason}",
            failure_class=FailureClass.PERMISSION_DENIED,
        )
    return os_tools.execute(loop.policy, action)


def _run_action(loop: _Loop, action: Action) -> tuple[ActionResult | None,
                                                      VerificationResult | None]:
    """observe -> permit -> execute -> verify -> recover, for one action.

    Returns ``(result, checkpoint)`` where ``result`` is None when the action was
    given up on -- in which case ``loop.replan`` or ``loop.abort_reason`` says
    what the caller should do next.
    """
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
        loop.enter(AgentState.ACTING, action=action.kind, params=action.params)
        result = _permit_and_execute(loop, action)
        loop.trace.action(result, precondition_age_s=age)

        loop.enter(AgentState.VERIFYING, action=action.kind, params=action.params)
        checkpoint = loop.task.verify_checkpoint(loop.policy, action, loop.trace)
        if checkpoint is not None:
            loop.trace.verification(checkpoint, checkpoint=True)
            loop.checkpoints.append(checkpoint)

        if result.ok and (checkpoint is None or checkpoint.verdict is Verdict.PASS):
            if recovering is not None:
                # Closes the loop on plan S19's "recovery success rate": the same
                # failure class is named again, this time as recovered.
                loop.trace.recovery_resolved(
                    failure_class=recovering, attempt=attempt,
                    budget_left=loop.recovery.budget.remaining(),
                )
            loop.enter(AgentState.OBSERVING, purpose="baseline")
            loop.refresh()  # our own change becomes the new baseline
            return result, checkpoint

        # An executor that found a genuine choice rather than a fault says so here,
        # and ``_dispatch`` is what decides whether there is anyone to ask.
        loop.question = _clarification_in(result)
        recovering = classify(
            result, checkpoint,
            precondition_age_s=age, max_staleness_s=loop.config.max_staleness_s,
        )
        if not _dispatch(loop, recovering, attempt):
            return None, checkpoint

    loop.abort_reason = f"attempt ceiling reached for action {action.kind!r}"
    return None, checkpoint


def _planner_name(planner: Planner) -> str:
    return getattr(planner, "name", type(planner).__name__)


def _plan(loop: _Loop, planner: Planner, history: list[dict]) -> PlannerStep:
    """One planner turn, over *summarised structured state* -- never a screenshot.

    Token deltas are diffed from the planner's cumulative Usage so the trace has
    per-call cost, which is what plan S19's cost column is built from.
    """
    usage = getattr(planner, "usage", None)
    before_in = usage.prompt_tokens if usage else 0
    before_out = usage.completion_tokens if usage else 0

    began = time.time()
    state = summarize(loop.observations)
    if loop.extra_state:
        # Caller-supplied context (remembered file locations) is added *beside*
        # the observations, never merged into one of them, and observations win a
        # name collision -- a cached path must not be able to overwrite something
        # this run actually measured.
        state = {**loop.extra_state, **state}
    step = planner.plan(loop.task.goal, state, history)
    loop.trace.planner_call(
        planner=_planner_name(planner), kind="plan",
        prompt_tokens=(usage.prompt_tokens - before_in) if usage else 0,
        completion_tokens=(usage.completion_tokens - before_out) if usage else 0,
        latency_s=time.time() - began, error=step.error,
    )
    loop.trace.emit(
        "planner_step",
        done=step.done, reasoning=step.reasoning[:1000], error=step.error,
        state_fingerprint=loop.fingerprint[:16],
        actions=[{"kind": a.kind, "params": a.params} for a in step.actions],
    )
    return step


def _history_entry(action: Action, result: ActionResult | None,
                   checkpoint: VerificationResult | None,
                   *, skipped_verified: bool = False) -> dict:
    """What the planner is told about its own last action.

    Evidence, not encouragement: the checkpoint verdict comes from the verifier
    re-reading the world, so a planner that failed sees the failure.
    """
    return {
        "action": {"kind": action.kind, "params": action.params},
        "ok": bool((result and result.ok) or skipped_verified),
        "error": (None if skipped_verified
                  else result.error if result else "given up on"),
        "detail": (result.detail if result else {}),
        "checkpoint": checkpoint.verdict.value if checkpoint else None,
        "skipped_verified": skipped_verified,
    }


def _drive(loop: _Loop, planner: Planner, history: list[dict]) -> tuple[bool, int]:
    """The planner loop. Returns ``(reported_success, steps_used)``.

    ``reported_success`` is the planner's *claim* and nothing more. It is compared
    against independent verification by the caller; the gap between the two is
    plan S19's false-success rate.
    """
    steps_used = 0
    for index in range(loop.config.max_steps):
        steps_used = loop.steps_used = index + 1
        loop.enter(AgentState.PLANNING, step=index)
        step = _plan(loop, planner, history)

        if step.error:
            # A planner that cannot produce a step is not a world failure; UNKNOWN
            # aborts safely rather than guessing an action (plan S9 last row).
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
                return False, steps_used

            if isinstance(loop.task, GeneralTask):
                loop.task.mark_completion_uncertain(
                    "the planner became unavailable before all explicitly "
                    "requested effects were established"
                )
            loop.decide(DecisionKind.STOP, "the planner returned no usable step")
            _dispatch(loop, FailureClass.UNKNOWN, index)
            loop.abort_reason = loop.abort_reason or f"planner error: {step.error}"
            return False, steps_used
        if step.done:
            loop.trace.note("planner_reported_done", step=index)
            loop.decide(DecisionKind.VERIFY,
                        "the planner claims the goal is met; verification decides")
            return True, steps_used
        if not step.actions:
            loop.decide(DecisionKind.STOP,
                        "the planner neither acted nor claimed completion")
            loop.abort_reason = "planner produced no actions and did not claim completion"
            return False, steps_used

        loop.decide(DecisionKind.ACT, "the next planned step is executable",
                    actions=[a.kind for a in step.actions])
        for action in step.actions:
            # GeneralTask owns the per-run effect ledger.  Re-read an equivalent
            # prior PASS immediately before replaying it; only a fresh PASS may
            # suppress execution, and no registered task is affected.
            from .general_task import GeneralTask

            if isinstance(loop.task, GeneralTask):
                already = loop.task.verified_equivalent(
                    loop.policy, action, loop.trace,
                )
                if already is not None:
                    loop.trace.verification(already, checkpoint=True)
                    loop.checkpoints.append(already)
                    loop.trace.emit(
                        "action_skipped",
                        action=action.to_json(),
                        reason="equivalent effect already verified PASS and still holds",
                    )
                    history.append(_history_entry(
                        action, None, already, skipped_verified=True,
                    ))
                    loop.enter(AgentState.OBSERVING, purpose="baseline")
                    loop.refresh()
                    continue

            result, checkpoint = _run_action(loop, action)
            history.append(_history_entry(action, result, checkpoint))
            if result is None:
                break

        if loop.question is not None:
            # A pending question outranks the rest of the plan: every later step
            # was chosen on an assumption the user is about to settle, so running
            # them would act on a guess. Returning here abandons them.
            return False, steps_used
        if loop.abort_reason:
            return False, steps_used
        if loop.replan:
            loop.replan = False

    loop.abort_reason = loop.abort_reason or f"step ceiling ({loop.config.max_steps}) reached"
    return False, steps_used


def _harness_error(task_id: str, exc: BaseException) -> VerificationResult:
    """A crashed run verifies UNKNOWN, never FAIL and never PASS.

    FAIL would claim we observed the goal unmet; we did not observe anything.
    Plan S8's rule applies to the harness's own bugs too.
    """
    return VerificationResult(
        label=task_id,
        checks=[Check(
            name="run_completed", verdict=Verdict.UNKNOWN, evidence={},
            reason=f"run raised {type(exc).__name__}: {exc}",
        )],
    )


def _cancelled(task_id: str) -> VerificationResult:
    """An interrupted run verifies *nothing*: zero checks, verdict UNKNOWN.

    ``verify_final`` is deliberately **not** called on this path. Several of its
    checks are preconditions rather than outcomes -- ``file_exists`` passes for a
    file the agent never touched -- so a run stopped before it acted could emit a
    final verdict of PASS. Plan E phase 2 forbids exactly that: "Cancellation must
    never be reported as PASS or ordinary FAIL." Skipping verification keeps the
    prohibition structural rather than something a downstream ``if`` has to
    remember, and keeps ``verified_success`` out of the metrics table for a trial
    that produced no evidence.

    Zero checks is not a shortcut for UNKNOWN, it is the accurate statement:
    nobody looked. ``VerificationResult.verdict`` already maps that to UNKNOWN.
    """
    return VerificationResult(checks=[], label=f"cancelled:{task_id}")


def _awaiting(task_id: str, label: str = "awaiting") -> VerificationResult:
    """A run stopped on an unanswered question verifies *nothing*: zero checks.

    Identical reasoning to :func:`_cancelled`, and identical mechanism, because
    the two situations share the property that matters: the run stopped somewhere
    other than the end. ``verify_final`` is deliberately **not** called, since
    several of its checks are preconditions rather than outcomes -- ``file_exists``
    passes for a file the agent never touched -- so a run suspended at its first
    step could otherwise report PASS for work it never did. Waiting must not be
    able to look like completion (PART 3 requirement 8).

    Zero checks is not shorthand for UNKNOWN; it is the accurate statement that
    nobody looked. ``VerificationResult.verdict`` already maps that to UNKNOWN.

    ``label`` distinguishes the two ways a question ends a run: ``awaiting`` when
    it was put to someone, ``ambiguous`` when there was nobody to ask.
    """
    return VerificationResult(checks=[], label=f"{label}:{task_id}")


def run_task(task: Task, planner: Planner, policy: Policy,
             config: RunConfig | None = None, *, trial: int = 0,
             trace: Trace | None = None, synthetic: bool = False,
             teardown: bool = True,
             extra_state: dict | None = None) -> RunOutcome:
    """One trial of one task under one condition (plan S21: >= 3 of these each).

    The ordering is the whole point and is not negotiable per-task: state is read
    before the planner speaks, policy is consulted before an executor runs,
    verification re-reads the world after, and recovery only ever spends from a
    budget. ``verify_final`` runs before ``teardown`` so the evidence outlives the
    workspace.

    ``extra_state`` is optional read-only context for the planner (remembered file
    locations). It changes what the planner is *told*, never what is executed,
    permitted, or verified.

    A ``KeyboardInterrupt`` is caught and returned as ``cancelled``, not re-raised:
    the caller needs a record of the interrupted trial, and every interface derives
    its report from this object. Callers that loop over trials must check
    ``outcome.cancelled`` and stop -- swallowing the interrupt here would otherwise
    turn one Ctrl+C into "carry on with the next one".

    ``NeedUserInput`` is handled the same way and for the same reason, returning an
    ``awaiting`` outcome that carries the question. With ``config.interactive``
    False -- the default, and every benchmark trial -- there is nobody to ask, so
    the run ends as an ordinary abort instead.
    """
    config = config or RunConfig()
    close_trace = trace is None
    trace = trace or Trace(task_id=task.task_id, condition=config.condition, trial=trial)
    loop = _Loop(
        task=task, policy=policy, config=config, trace=trace,
        recovery=Recovery(budget=config.budget, enabled=config.recovery_enabled,
                          interactive=config.interactive),
        extra_state=dict(extra_state or {}),
    )

    started = time.time()
    reported_success = False
    steps_used = 0
    cancelled = False
    awaiting = False
    history: list[dict] = []

    trace.emit(
        "run_start", goal=task.goal, bucket=getattr(task, "bucket", ""),
        planner=_planner_name(planner), config=config.to_json(),
        workspace=str(policy.workspace), refuse_if_elevated=policy.refuse_if_elevated,
    )
    try:
        loop.enter(AgentState.OBSERVING, purpose="initial")
        task.setup(policy)
        loop.refresh()
        reported_success, steps_used = _drive(loop, planner, history)
        if loop.question is not None:
            # ``_drive`` stopped on a question rather than finishing. Same rule as
            # the interrupt below: no ``verify_final``, so nothing can be claimed.
            awaiting = True
            final = _awaiting(task.task_id)
        elif loop.abort_reason is not None:
            # ``_drive`` did not finish normally -- some action was denied or the
            # batch was otherwise aborted. ``verify_final`` still runs: it is
            # useful diagnostic information about which of the *attempted*
            # effects genuinely hold. But an abort means the run itself did not
            # complete, so this verdict must not be recorded through the normal
            # completion channel (``AgentState.VERIFYING`` + checkpoint
            # verification), which would make it look like ordinary terminal
            # evidence. It is logged distinctly instead, and the terminal state
            # selection below treats ``abort_reason`` as decisive regardless of
            # what this verdict says.
            final = task.verify_final(policy, trace)
            trace.note("diagnostic_verification_after_abort",
                       verdict=final.verdict.value, abort_reason=loop.abort_reason)
        else:
            loop.enter(AgentState.VERIFYING, purpose="final")
            final = task.verify_final(policy, trace)
            trace.verification(final, checkpoint=False)
    except KeyboardInterrupt:
        # KeyboardInterrupt is a BaseException, so the handler below never saw it:
        # before this branch existed, Ctrl+C mid-run skipped the outcome entirely
        # and no ``run_end`` was ever written. The interrupt is a datum about the
        # session, not about the agent, so it is recorded and returned rather than
        # re-raised or dressed up as a failure.
        cancelled = True
        # ``_drive`` was unwound rather than returned from, so the local
        # ``steps_used`` above is still 0. ``loop.steps_used`` is what the run
        # actually reached.
        steps_used = loop.steps_used
        loop.question = None
        trace.note("run_cancelled", reason="KeyboardInterrupt",
                   steps_used=steps_used, reported_success=reported_success)
        loop.abort_reason = "cancelled by user before verification finished"
        final = _cancelled(task.task_id)
    except NeedUserInput as need:
        # Raised from inside the task -- an observer that found two valid
        # continuations and no way to choose between them. Unwinding is how the
        # rest of the planned batch is abandoned: those actions were chosen on an
        # assumption that is now in question, so none of them run.
        steps_used = loop.steps_used
        decision, reason = loop.recovery.decide(FailureClass.AMBIGUOUS)
        if decision is RecoveryDecision.ASK_USER:
            awaiting = True
            loop.question = need.clarification
            trace.emit("clarification_needed", failure_class=FailureClass.AMBIGUOUS.value,
                       attempt=steps_used, question=need.clarification.to_json())
            loop.decide(DecisionKind.ASK_USER,
                        "only a person can choose the continuation")
            loop.enter(AgentState.WAITING_FOR_USER)
            final = _awaiting(task.task_id)
        else:
            # Nobody to ask, so this *is* a failure of the run -- logged through
            # ``trace.failure``, which is the channel for a class that ended a
            # trial without any recovery attempt.
            trace.failure(FailureClass.AMBIGUOUS, question=need.clarification.to_json(),
                          steps_used=steps_used)
            loop.abort_reason = f"{FailureClass.AMBIGUOUS.value}: {reason}"
            final = _awaiting(task.task_id, "ambiguous")
    except Exception as exc:  # noqa: BLE001 - a crashed trial is a datum, not a stop
        # Same reason as the branch above: a crash inside ``_drive`` also unwinds it
        # without a return value, and a trial that crashed on its fourth step is not
        # a trial that took zero steps.
        steps_used = loop.steps_used
        loop.question = None
        trace.note("run_exception", error=f"{type(exc).__name__}: {exc}")
        loop.abort_reason = loop.abort_reason or f"harness error: {type(exc).__name__}: {exc}"
        final = _harness_error(task.task_id, exc)
    finally:
        # A cancelled or suspended run keeps its workspace. Teardown would delete
        # the only evidence of how far it got -- and for a suspended run, the state
        # the answer is about to be applied to. This is not rollback or
        # resumability (plan E phase 2 rules both out) -- it is simply not
        # destroying state.
        if teardown and not cancelled and not awaiting:
            try:
                task.teardown(policy)
            except Exception as exc:  # noqa: BLE001
                trace.note("teardown_failed", error=f"{type(exc).__name__}: {exc}")

    loop.enter(
        AgentState.CANCELLED if cancelled
        else AgentState.WAITING_FOR_USER if awaiting
        # An abort (e.g. a denied action) is decisive on its own: no
        # ``final.verdict`` -- diagnostic or otherwise -- can promote an
        # aborted run to COMPLETED. This is checked ahead of the PASS check
        # for the same reason ``api.py:_status`` checks ``aborted_reason``
        # ahead of ``Verdict.PASS`` -- an incidental successful effect must
        # not be read as the run having succeeded.
        else AgentState.FAILED if loop.abort_reason is not None
        else AgentState.COMPLETED if final.verdict is Verdict.PASS
        else AgentState.FAILED
    )

    outcome = RunOutcome(
        task_id=task.task_id, condition=config.condition, trial=trial,
        planner_name=_planner_name(planner),
        reported_success=reported_success,
        verified=final.verdict, final=final,
        checkpoints=list(loop.checkpoints),
        failure_categories=sorted(trace.failure_categories.elements()),
        steps_used=steps_used,
        wall_clock_s=time.time() - started,
        usage=planner.usage.to_json() if getattr(planner, "usage", None) else {},
        trace=trace.summary(),
        aborted_reason=loop.abort_reason,
        synthetic=synthetic or _planner_name(planner).startswith("mock"),
        cancelled=cancelled,
        state=loop.state,
        awaiting=awaiting,
        question=loop.question if awaiting else None,
    )
    trace.emit("run_end", outcome={k: v for k, v in outcome.to_json().items()
                                   if k not in ("trace", "checkpoints")})
    if close_trace:
        trace.close()
    return outcome