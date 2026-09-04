"""Tests for Ctrl+C: the one outcome that is neither success nor failure.

Plan E phase 2 states the rule as a prohibition -- "cancellation must never be
reported as PASS or ordinary FAIL" -- so these tests are mostly about what must
*not* appear. Five properties:

1. ``run_task`` returns a cancelled outcome instead of letting the interrupt
   escape, because every interface builds its report from that object and an
   exception past this point means no report at all.
2. A cancelled run verifies nothing. ``verify_final`` is not called, so a task
   whose checks are preconditions cannot hand back a PASS for work never done.
3. ``CANCELLED`` is decided before the PASS check, and speech cannot phrase it as
   completion.
4. A cancelled run is not evidence about the planner: ``false_success`` stays
   False even when the planner had already claimed done.
5. One Ctrl+C stops the benchmark grid, and the cancelled cell stays out of the
   metrics rather than counting as a non-success nobody measured.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_control import api
from agent_control.api import AgentResult, TaskStatus, _status
from agent_control.response import phrase_result
from agent_control.runner import RunConfig, RunOutcome, run_task
from agent_control.types import (
    Action,
    Check,
    Observation,
    Source,
    VerificationResult,
    Verdict,
)


# -- test doubles -------------------------------------------------------------

class Interrupting:
    """A planner that is interrupted on its ``nth`` turn, as a person would."""

    name = "interrupting"

    def __init__(self, *, nth: int = 0) -> None:
        self.nth = nth
        self.turns = 0

    def plan(self, goal: str, state: dict, history: list[dict]):
        from agent_control.planner.base import PlannerStep

        turn, self.turns = self.turns, self.turns + 1
        if turn == self.nth:
            raise KeyboardInterrupt
        return PlannerStep(
            actions=[Action(kind="create_dir", params={"path": "made"})],
            done=False, reasoning="working",
        )


class Recorded:
    """A task that records whether it was torn down, and can pass a checkpoint.

    ``verify_final`` returns PASS unconditionally. That is not laziness: it is the
    shape of a real precondition check (``file_exists`` passes for a file the agent
    never touched), and it is what makes the "cancellation is never PASS" test mean
    something. If the runner ever calls it on the cancelled path, the test fails.
    """

    task_id = "cancel_demo"
    goal = "do the thing"
    bucket = "test"

    def __init__(self, *, checkpoint: Verdict | None = None) -> None:
        self.checkpoint = checkpoint
        self.torn_down = False
        self.verified_final = False

    def setup(self, policy) -> None:
        return None

    def observe(self, policy, trace=None) -> dict:
        return {"thing": Observation(source=Source.FILESYSTEM, query="q", value="here")}

    def reference_plan(self, policy) -> list:
        return []

    def verify_checkpoint(self, policy, action, trace=None):
        if self.checkpoint is None:
            return None
        return VerificationResult(label=self.task_id, checks=[
            Check(name="made_progress", verdict=self.checkpoint, evidence={},
                  reason="observed at a checkpoint"),
        ])

    def verify_final(self, policy, trace=None):
        self.verified_final = True
        return VerificationResult(label=self.task_id, checks=[
            Check(name="precondition", verdict=Verdict.PASS, evidence={},
                  reason="true before the run started, and still true"),
        ])

    def teardown(self, policy) -> None:
        self.torn_down = True


def outcome_of(*, cancelled: bool = True, reported: bool = False,
               final: VerificationResult | None = None,
               checkpoints: list[VerificationResult] | None = None,
               aborted: str | None = None) -> RunOutcome:
    result = final or VerificationResult(checks=[], label="cancelled:demo")
    return RunOutcome(
        task_id="demo", condition="structured_hybrid", trial=0,
        planner_name="mock", reported_success=reported,
        verified=result.verdict, final=result,
        checkpoints=list(checkpoints or []),
        aborted_reason=aborted, cancelled=cancelled,
    )


# -- 1. the interrupt is recorded, not propagated -----------------------------

def test_an_interrupt_returns_an_outcome_instead_of_escaping(policy, trace) -> None:
    """``except Exception`` never caught this: ``KeyboardInterrupt`` is a
    ``BaseException``, so before phase 2 a Ctrl+C skipped the outcome entirely and
    no interface had anything to report."""
    task = Recorded()

    outcome = run_task(task, Interrupting(), policy, RunConfig(), trace=trace)

    assert outcome.cancelled is True
    assert outcome.aborted_reason == "cancelled by user before verification finished"


def test_the_trace_records_the_cancellation_and_still_closes(policy, trace,
                                                             events) -> None:
    """Directive C requires Ctrl+C to preserve the trace. ``run_end`` is the record
    that the run stopped at all, so its absence would leave a trace that simply
    trails off."""
    run_task(Recorded(), Interrupting(), policy, RunConfig(), trace=trace)

    records = events()
    # ``Trace.note`` emits ``event="note"`` and puts the label in ``message``, so
    # read both fields rather than assuming the label is the event name.
    labels = {record.get("message") or record.get("event") for record in records}
    assert "run_cancelled" in labels
    assert "run_end" in labels

    cancelled = next(r for r in records if r.get("message") == "run_cancelled")
    assert cancelled["reason"] == "KeyboardInterrupt"
    # The outcome is in the trace too, so the JSONL alone shows it was not a PASS.
    ended = next(r for r in records if r.get("event") == "run_end")
    assert ended["outcome"]["cancelled"] is True
    assert ended["outcome"]["verified"] == "UNKNOWN"


def test_a_cancelled_run_keeps_its_workspace(policy, trace) -> None:
    """Teardown would delete the only evidence of how far the run got, and the
    person who pressed Ctrl+C is the one most likely to want to look."""
    task = Recorded()

    run_task(task, Interrupting(), policy, RunConfig(), trace=trace, teardown=True)

    assert task.torn_down is False


def test_an_ordinary_run_is_still_torn_down(policy, trace) -> None:
    """The guard is ``not cancelled``, so it must not have quietly disabled
    teardown for every other run."""
    task = Recorded()

    class Done:
        name = "done"

        def plan(self, goal, state, history):
            from agent_control.planner.base import PlannerStep

            return PlannerStep(done=True, reasoning="already there")

    outcome = run_task(task, Done(), policy, RunConfig(), trace=trace, teardown=True)

    assert task.torn_down is True
    assert outcome.cancelled is False


# -- 2. a cancelled run verifies nothing --------------------------------------

def test_final_verification_is_not_run_after_an_interrupt(policy, trace) -> None:
    """``Recorded.verify_final`` returns PASS for a precondition. Calling it here
    would produce ``verified: PASS`` on a run that did nothing -- the exact claim
    plan E phase 2 forbids -- so the cancelled path must not call it at all."""
    task = Recorded()

    outcome = run_task(task, Interrupting(), policy, RunConfig(), trace=trace)

    assert task.verified_final is False
    assert outcome.final.checks == []
    assert outcome.verified is Verdict.UNKNOWN
    assert outcome.verified_success is False


def test_zero_checks_is_unknown_not_pass() -> None:
    """The property the cancelled sentinel leans on, asserted directly rather than
    trusted: nobody looked, so nothing is claimed."""
    assert VerificationResult(checks=[], label="x").verdict is Verdict.UNKNOWN


def test_the_steps_that_had_begun_are_still_counted(policy, trace) -> None:
    """The interrupt unwinds ``_drive`` instead of letting it return, so its local
    step count never reaches the caller. Reporting 0 steps for a run that had
    already planned and dispatched twice reads as "nothing happened", which is the
    one direction the report must not err in. Observed live: an interrupt during
    ``open_file`` printed ``steps 0`` while the trace showed a planner turn."""
    task = Recorded()

    outcome = run_task(task, Interrupting(nth=2), policy, RunConfig(), trace=trace)

    assert outcome.cancelled is True
    assert outcome.steps_used == 3, "the third turn was the interrupted one"


def test_checkpoints_passed_before_the_interrupt_are_kept(policy, trace) -> None:
    """Progress that really was verified is still reported. The run gets no verdict;
    that is different from pretending nothing happened."""
    task = Recorded(checkpoint=Verdict.PASS)
    # Interrupted on the *second* turn, so the first action and its checkpoint ran.
    outcome = run_task(task, Interrupting(nth=1), policy, RunConfig(), trace=trace)

    assert outcome.cancelled is True
    assert [c.verdict for c in outcome.checkpoints] == [Verdict.PASS]

    result = api._result_from(request="demo", outcome=outcome, root=policy.workspace)
    assert result.completed == ["made_progress"]
    assert result.status is TaskStatus.CANCELLED
    assert result.ok is False


# -- 3. status and speech ------------------------------------------------------

def test_cancellation_outranks_a_passing_verdict() -> None:
    """Checked before the PASS branch, so even a final that somehow verified cannot
    turn an interrupted run into a success."""
    passing = VerificationResult(label="demo", checks=[
        Check(name="dir_exists", verdict=Verdict.PASS, evidence={}, reason="there"),
    ])
    status, _ = _status(outcome_of(final=passing), ["dir_exists"], [], [])

    assert status is TaskStatus.CANCELLED


def test_cancellation_is_not_reported_as_failure() -> None:
    """FAILED would claim we observed the goal unmet. We observed nothing."""
    status, detail = _status(outcome_of(aborted="cancelled by user before "
                                               "verification finished"), [], [], [])

    assert status is TaskStatus.CANCELLED
    assert "cancelled by user" in detail


def test_cancelled_is_a_status_of_its_own() -> None:
    """Distinct from UNKNOWN: "I looked and could not tell" and "you stopped me
    before I looked" are different facts about the machine."""
    assert TaskStatus.CANCELLED is not TaskStatus.UNKNOWN
    assert TaskStatus.CANCELLED.is_success is False
    assert TaskStatus("CANCELLED") is TaskStatus.CANCELLED


def test_the_json_payload_names_the_cancellation() -> None:
    payload = api._result_from(request="demo", outcome=outcome_of(),
                               root=Path("ws")).to_json()
    json.dumps(payload)  # must not raise
    assert payload["status"] == "CANCELLED"
    assert payload["ok"] is False
    assert payload["verified"] == "UNKNOWN"


def test_the_runner_json_carries_the_flag() -> None:
    assert outcome_of().to_json()["cancelled"] is True
    assert outcome_of(cancelled=False).to_json()["cancelled"] is False


def test_speech_never_phrases_a_cancellation_as_completion() -> None:
    spoken = phrase_result(AgentResult(request="demo", status=TaskStatus.CANCELLED))

    assert "stopped" in spoken.lower()
    assert "complete" not in spoken.lower()
    assert "failed" not in spoken.lower()


def test_speech_names_progress_without_claiming_the_task() -> None:
    """One sentence, and it has to do two things at once: say the run did not
    finish, and not throw away the checks that did pass."""
    spoken = phrase_result(AgentResult(request="demo", status=TaskStatus.CANCELLED,
                                       completed=["cloned"]))

    assert "1 check had passed" in spoken
    assert "not claiming it worked" in spoken


# -- 4. a cancelled run is not evidence about the planner ----------------------

def test_an_interrupted_check_is_not_a_false_success() -> None:
    """Ctrl+C can land after the planner claimed done and before ``verify_final``
    returns. Counting that as a false claim would blame the planner for a check we
    stopped -- and false-success rate is the one metric this project exists to
    measure."""
    outcome = outcome_of(reported=True)

    assert outcome.reported_success is True
    assert outcome.verified_success is False
    assert outcome.false_success is False
    assert outcome.silent_failure is False


def test_a_real_false_success_is_still_reported() -> None:
    """The exclusion is conditioned on ``cancelled`` alone, so it must not have
    weakened the ordinary case."""
    failing = VerificationResult(label="demo", checks=[
        Check(name="dir_exists", verdict=Verdict.FAIL, evidence={}, reason="absent"),
    ])
    outcome = outcome_of(cancelled=False, reported=True, final=failing)

    assert outcome.false_success is True


# -- 5. one Ctrl+C stops the grid ----------------------------------------------

def test_an_interrupt_before_the_run_starts_is_cancelled(monkeypatch,
                                                         tmp_path) -> None:
    """The window ``run_task`` cannot cover: recall reads the whole location index,
    which is the slowest thing that happens before execution."""
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(api, "_remembered", interrupt)

    result = api.run_agent_task("open_last_day_pdf", planner="mock",
                                workspace=tmp_path / "ws")

    assert result.status is TaskStatus.CANCELLED
    assert result.ok is False
    assert "before the run started" in result.detail


def test_the_cli_exits_130_rather_than_1(monkeypatch, capsys) -> None:
    """Exit 1 would tell a shell script the task failed. It did not fail; it was
    stopped."""
    import main

    monkeypatch.setattr(api, "run_agent_task",
                        lambda request, **kw: AgentResult(
                            request=request, task_id="open_last_day_pdf",
                            status=TaskStatus.CANCELLED,
                            detail="cancelled by user before verification finished"))
    args = __import__("argparse").Namespace(
        task="open_last_day_pdf", planner="mock", max_steps=4, no_fresh_state=False,
        no_recovery=False, keep_workspace=False, json=False, speak=False)

    assert main.cmd_run(args) == main.EXIT_INTERRUPTED
    assert "CANCELLED" in capsys.readouterr().out


def test_the_top_level_handler_reports_an_interrupt_plainly(monkeypatch,
                                                            capsys) -> None:
    """A bare traceback is not a report. Everything outside a run -- indexing,
    doctor, a prompt -- lands here."""
    import main

    def interrupt(args):
        raise KeyboardInterrupt

    # ``build_parser`` resolves ``cmd_tasks`` from the module globals when
    # ``main()`` runs, so patching the attribute is enough.
    monkeypatch.setattr(main, "cmd_tasks", interrupt)

    code = main.main(["tasks"])

    assert code == main.EXIT_INTERRUPTED
    assert "interrupted" in capsys.readouterr().err


def test_one_cancelled_trial_stops_the_grid(monkeypatch, tmp_path) -> None:
    """``run_task`` returns instead of raising, so the loop has to check. Without
    this, a single Ctrl+C would be swallowed and the grid would carry on -- the
    worst of both: the person cannot stop it, and the cancelled cell lands in the
    table as a non-success."""
    from benchmark import harness

    calls: list[tuple[str, int]] = []

    def fake_trial(condition, task_id, trial, **kwargs):
        calls.append((task_id, trial))
        cancelled = len(calls) == 2
        final = VerificationResult(label=task_id, checks=[] if cancelled else [
            Check(name="dir_exists", verdict=Verdict.PASS, evidence={}, reason="ok"),
        ])
        return RunOutcome(
            task_id=task_id, condition=condition.name, trial=trial,
            planner_name="mock", reported_success=not cancelled,
            verified=final.verdict, final=final, cancelled=cancelled,
            aborted_reason="cancelled by user" if cancelled else None,
            trace={"counters": {}},
        )

    monkeypatch.setattr(harness, "run_trial", fake_trial)
    monkeypatch.setattr(harness, "RESULTS_DIR", tmp_path / "results")

    code = harness.main([
        "--planner", "mock", "--trials", "5",
        "--conditions", "structured_hybrid",
        "--workspace-root", str(tmp_path / "ws"),
    ])

    assert code == harness.EXIT_INTERRUPTED
    assert len(calls) == 2, "the grid continued past the interrupt"

    report = json.loads((tmp_path / "results" / "run_latest.json").read_text())
    assert report["meta"]["interrupted_at"].endswith("trial 2")
    # One trial finished. The cancelled one is excluded rather than counted as a
    # trial that failed to verify.
    assert report["by_condition"]["structured_hybrid"]["trials"] == 1
    assert report["by_condition"]["structured_hybrid"]["verified_success_rate"] == 1.0


def test_an_interrupted_grid_says_so_in_the_report(monkeypatch, tmp_path) -> None:
    """Someone reading results_latest.md must not mistake two trials for five."""
    from benchmark import harness

    markdown = harness.render_markdown({
        "meta": {"started_at_iso": "now", "planner": "mock", "model": None,
                 "trials": 5, "tasks": ["open_last_day_pdf"], "price_in": 0.0,
                 "price_out": 0.0, "platform": "test", "python": "3.12",
                 "workspace_root": "ws", "trace_dir": "traces",
                 "interrupted_at": "structured_hybrid/open_last_day_pdf trial 2"},
        "by_condition": {}, "by_condition_task": {}, "by_bucket": {},
        "outcomes": [],
    })

    assert "Interrupted run" in markdown
    assert "incomplete" in markdown
    assert "excluded rather than counted as a non-success" in markdown


def test_nothing_is_written_when_no_trial_finished(monkeypatch, tmp_path) -> None:
    """A report of zero trials would overwrite run_latest.json -- replacing real
    evidence with the absence of any."""
    from benchmark import harness

    def cancelled_trial(condition, task_id, trial, **kwargs):
        return RunOutcome(
            task_id=task_id, condition=condition.name, trial=trial,
            planner_name="mock", reported_success=False, verified=Verdict.UNKNOWN,
            final=VerificationResult(label=task_id, checks=[]), cancelled=True,
        )

    results = tmp_path / "results"
    results.mkdir()
    (results / "run_latest.json").write_text('{"real": "evidence"}', encoding="utf-8")
    monkeypatch.setattr(harness, "run_trial", cancelled_trial)
    monkeypatch.setattr(harness, "RESULTS_DIR", results)

    code = harness.main(["--planner", "mock", "--trials", "3",
                         "--conditions", "structured_hybrid",
                         "--workspace-root", str(tmp_path / "ws")])

    assert code == harness.EXIT_INTERRUPTED
    assert json.loads((results / "run_latest.json").read_text()) == {"real": "evidence"}
    assert list(results.glob("run_2*.json")) == []
