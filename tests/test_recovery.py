"""Tests for what recovery does with a verification that could not decide.

Narrow on purpose: this file covers the seam between a checkpoint that ran and
could not tell, and a failure the runner cannot name at all. Four properties:

1. An action that executed and reported success, followed by a checkpoint verdict
   of UNKNOWN, classifies as INCONCLUSIVE and earns one re-observe.
2. Every other route to UNKNOWN still aborts, so this is not a licence for the
   whole system to retry on ignorance.
3. Recovery stays inside the existing budget: one re-observe, one retry, stop.
4. A recovered action is not a successful run. Final verification still decides.
"""

from __future__ import annotations

from agent_control import runner
from agent_control.planner.base import PlannerStep
from agent_control.recovery import (
    Recovery,
    RecoveryBudget,
    RecoveryDecision,
    classify,
)
from agent_control.runner import RunConfig, _Loop, run_task
from agent_control.types import (
    Action,
    ActionResult,
    Check,
    FailureClass,
    Observation,
    Source,
    VerificationResult,
    Verdict,
)

#: ``create_dir`` is idempotent and always reports ``ok=True``, so repeating it
#: exercises recovery without the action itself being the variable under test.
ACTION = Action(kind="create_dir", params={"path": "made"})

FULL_BUDGET = {"retries": 1, "reobserves": 1, "replans": 1}


# -- test doubles -------------------------------------------------------------


class Undecided:
    """A task whose checkpoint verdicts are scripted.

    Real verifiers return UNKNOWN when the *reading* failed -- no window backend,
    a handler that has not drawn yet. Scripting the verdict keeps these tests
    about what recovery does with that answer. Unscripted checkpoints stay UNKNOWN.
    """

    task_id = "inconclusive_demo"
    goal = "do the thing"
    bucket = "test"

    def __init__(self, verdicts: list[Verdict], *, final: Verdict | None = None) -> None:
        self.verdicts = list(verdicts)
        self.final = final
        #: "observe" and "checkpoint" in the order they happened.
        self.calls: list[str] = []

    def setup(self, policy) -> None:
        return None

    def observe(self, policy, trace=None) -> dict:
        self.calls.append("observe")
        return {"thing": Observation(source=Source.FILESYSTEM, query="q", value="here")}

    def reference_plan(self, policy) -> list:
        return []

    def verify_checkpoint(self, policy, action, trace=None) -> VerificationResult:
        self.calls.append("checkpoint")
        verdict = self.verdicts.pop(0) if self.verdicts else Verdict.UNKNOWN
        return VerificationResult(label=self.task_id, checks=[
            Check(name="made_progress", verdict=verdict, evidence={},
                  reason="re-read the world after the action"),
        ])

    def verify_final(self, policy, trace=None) -> VerificationResult:
        if self.final is None:
            return VerificationResult(label=self.task_id, checks=[])
        return VerificationResult(label=self.task_id, checks=[
            Check(name="goal_reached", verdict=self.final, evidence={},
                  reason="independent of what the action reported"),
        ])

    def teardown(self, policy) -> None:
        return None

    @property
    def executions(self) -> int:
        """How many times the action ran: one checkpoint is taken per execution."""
        return self.calls.count("checkpoint")


class Repeating:
    """Emits the same action every turn and never claims completion."""

    name = "repeating"

    def plan(self, goal, state, history) -> PlannerStep:
        return PlannerStep(actions=[ACTION], done=False, reasoning="working")


class OnceThenDone:
    """Emits the action once, then claims the goal is met."""

    name = "once_then_done"

    def __init__(self) -> None:
        self.turns = 0

    def plan(self, goal, state, history) -> PlannerStep:
        turn, self.turns = self.turns, self.turns + 1
        if turn == 0:
            return PlannerStep(actions=[ACTION], done=False, reasoning="working")
        return PlannerStep(done=True, reasoning="already there")


class Broken:
    """A planner that cannot produce a step at all."""

    name = "broken"

    def plan(self, goal, state, history) -> PlannerStep:
        return PlannerStep(error="bad JSON from the model")


def loop_for(task, policy, trace, *, budget=None, enabled: bool = True) -> _Loop:
    policy.workspace.mkdir(parents=True, exist_ok=True)
    return _Loop(
        task=task, policy=policy, config=RunConfig(recovery_enabled=enabled),
        trace=trace,
        recovery=Recovery(budget=budget or RecoveryBudget(), enabled=enabled),
    )


def undecided() -> VerificationResult:
    return VerificationResult(label="x", checks=[
        Check(name="cannot_tell", verdict=Verdict.UNKNOWN, evidence={},
              reason="could not read the state"),
    ])


def executed_ok() -> ActionResult:
    return ActionResult(action=ACTION, ok=True)


# -- A. an undecided checkpoint after a successful action ---------------------


def test_an_executed_action_with_an_undecided_checkpoint_is_inconclusive() -> None:
    assert classify(executed_ok(), undecided()) is FailureClass.INCONCLUSIVE


def test_inconclusive_chooses_reobserve_then_retry() -> None:
    recovery = Recovery(budget=RecoveryBudget(), enabled=True)
    decision, reason = recovery.decide(FailureClass.INCONCLUSIVE)
    assert decision is RecoveryDecision.REOBSERVE_THEN_RETRY
    assert reason == "INCONCLUSIVE -> REOBSERVE_THEN_RETRY"


def test_expected_state_not_reached_chooses_existing_reobserve_then_retry() -> None:
    recovery = Recovery(budget=RecoveryBudget(), enabled=True)
    decision, reason = recovery.decide(FailureClass.EXPECTED_STATE_NOT_REACHED)

    assert decision is RecoveryDecision.REOBSERVE_THEN_RETRY
    assert reason == "expected_state_not_reached -> REOBSERVE_THEN_RETRY"
    assert recovery.budget.remaining() == {
        "retries": 0,
        "reobserves": 0,
        "replans": 1,
    }


def test_the_runner_reobserves_and_runs_the_action_again(policy, trace) -> None:
    task = Undecided([Verdict.UNKNOWN, Verdict.PASS])
    loop = loop_for(task, policy, trace)

    result, checkpoint = runner._run_action(loop, ACTION)

    assert task.executions == 2
    assert result is not None and result.ok is True
    assert checkpoint is not None and checkpoint.verdict is Verdict.PASS
    assert loop.abort_reason is None
    # The re-observe is the point: a second blind check would have returned the
    # same UNKNOWN, so state must have been re-read between the two executions.
    first = task.calls.index("checkpoint")
    second = task.calls.index("checkpoint", first + 1)
    assert "observe" in task.calls[first + 1:second]


# -- B. every other route to UNKNOWN still aborts -----------------------------


def test_a_missing_verification_is_still_unknown() -> None:
    assert classify(executed_ok(), None) is FailureClass.UNKNOWN
    assert classify(None, None) is FailureClass.UNKNOWN


def test_an_undecided_checkpoint_without_an_executed_action_is_unknown() -> None:
    assert classify(None, undecided()) is FailureClass.UNKNOWN


def test_a_failed_action_is_not_promoted_to_inconclusive() -> None:
    failed = ActionResult(action=ACTION, ok=False, error="no")
    assert classify(failed, undecided()) is FailureClass.ACTION_FAILED


def test_expected_state_recovery_preserves_other_strategy_mappings() -> None:
    stale_decision, _ = Recovery(budget=RecoveryBudget(), enabled=True).decide(
        FailureClass.STALE_STATE
    )
    exception_decision, _ = Recovery(budget=RecoveryBudget(), enabled=True).decide(
        FailureClass.EXECUTION_EXCEPTION
    )
    dead_browser_decision, _ = Recovery(budget=RecoveryBudget(), enabled=True).decide(
        FailureClass.BROWSER_CONNECTION_FAILED
    )

    assert stale_decision is RecoveryDecision.REOBSERVE_THEN_RETRY
    assert exception_decision is RecoveryDecision.RETRY
    assert dead_browser_decision is RecoveryDecision.REOBSERVE_THEN_RETRY


def test_generic_unknown_still_aborts_without_spending_budget() -> None:
    recovery = Recovery(budget=RecoveryBudget(), enabled=True)
    decision, reason = recovery.decide(FailureClass.UNKNOWN)
    assert decision is RecoveryDecision.ABORT
    assert reason == "UNKNOWN is not recoverable in V1"
    assert recovery.budget.remaining() == FULL_BUDGET


def test_dispatching_unknown_ends_the_action(policy, trace) -> None:
    loop = loop_for(Undecided([]), policy, trace)
    assert runner._dispatch(loop, FailureClass.UNKNOWN, 0) is False
    assert loop.abort_reason == "UNKNOWN: UNKNOWN is not recoverable in V1"


def test_a_planner_error_still_aborts_the_run(policy, trace) -> None:
    budget = RecoveryBudget()
    outcome = run_task(Undecided([]), Broken(), policy,
                       RunConfig(max_steps=2, budget=budget),
                       trace=trace, teardown=False)
    assert outcome.aborted_reason == "UNKNOWN: UNKNOWN is not recoverable in V1"
    assert outcome.verified_success is False
    assert budget.remaining() == FULL_BUDGET


# -- C. recovery stays inside the existing budget ------------------------------


def test_a_persistent_inconclusive_checkpoint_stops_after_two_executions(
        policy, trace) -> None:
    budget = RecoveryBudget()
    task = Undecided([Verdict.UNKNOWN, Verdict.UNKNOWN])
    loop = loop_for(task, policy, trace, budget=budget)

    result, checkpoint = runner._run_action(loop, ACTION)

    assert task.executions == 2, "one execution, one bounded retry, then stop"
    assert result is None
    assert checkpoint is not None and checkpoint.verdict is Verdict.UNKNOWN
    assert loop.abort_reason == (
        "INCONCLUSIVE: budget exhausted for REOBSERVE_THEN_RETRY")
    assert budget.remaining() == {"retries": 0, "reobserves": 0, "replans": 1}
    assert budget.retries_used == 1 and budget.reobserves_used == 1


def test_the_recovery_ablation_still_aborts_an_inconclusive_result(
        policy, trace) -> None:
    budget = RecoveryBudget()
    task = Undecided([])
    loop = loop_for(task, policy, trace, budget=budget, enabled=False)

    result, _ = runner._run_action(loop, ACTION)

    assert task.executions == 1
    assert result is None
    assert loop.abort_reason == "INCONCLUSIVE: recovery disabled (ablation condition)"
    assert budget.remaining() == FULL_BUDGET


# -- D. a recovered action is not a successful run -----------------------------


def test_a_full_run_spends_the_budget_and_still_does_not_succeed(policy, trace) -> None:
    budget = RecoveryBudget()
    task = Undecided([])          # every checkpoint UNKNOWN
    outcome = run_task(task, Repeating(), policy,
                       RunConfig(max_steps=3, budget=budget),
                       trace=trace, teardown=False)

    assert task.executions == 2
    assert budget.remaining() == {"retries": 0, "reobserves": 0, "replans": 1}
    assert outcome.aborted_reason == (
        "INCONCLUSIVE: budget exhausted for REOBSERVE_THEN_RETRY")
    assert outcome.verified is Verdict.UNKNOWN
    assert outcome.verified_success is False
    assert outcome.reported_success is False
    assert outcome.false_success is False
    assert outcome.failure_categories == ["INCONCLUSIVE", "INCONCLUSIVE"]
    assert outcome.steps_used == 1


def test_a_recovered_action_does_not_by_itself_make_the_run_a_success(
        policy, trace) -> None:
    """The retry must not manufacture success: the checkpoint passed and the
    planner claimed done, and the run is still not a success because
    ``verify_final`` did not agree."""
    task = Undecided([Verdict.UNKNOWN, Verdict.PASS])
    outcome = run_task(task, OnceThenDone(), policy, RunConfig(max_steps=3),
                       trace=trace, teardown=False)

    assert task.executions == 2
    assert outcome.aborted_reason is None
    assert outcome.reported_success is True
    assert outcome.verified_success is False
    assert outcome.false_success is True


def test_recovery_lets_a_genuinely_verified_run_through(policy, trace) -> None:
    task = Undecided([Verdict.UNKNOWN, Verdict.PASS], final=Verdict.PASS)
    outcome = run_task(task, OnceThenDone(), policy, RunConfig(max_steps=3),
                       trace=trace, teardown=False)

    assert outcome.verified_success is True
    assert outcome.false_success is False


# -- every attempt is logged (plan S30) ----------------------------------------
# Last in the file on purpose: the ``events`` fixture closes the trace to read it.


def test_the_recovery_attempts_are_logged(policy, trace, events) -> None:
    loop = loop_for(Undecided([Verdict.UNKNOWN, Verdict.UNKNOWN]), policy, trace)

    runner._run_action(loop, ACTION)

    records = [r for r in events() if r.get("event") == "recovery"]
    assert [r["decision"] for r in records] == ["REOBSERVE_THEN_RETRY", "ABORT"]
    assert {r["failure_class"] for r in records} == {"INCONCLUSIVE"}
    assert records[0]["budget_left"] == {"retries": 0, "reobserves": 0, "replans": 1}
