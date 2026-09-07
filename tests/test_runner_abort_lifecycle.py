"""Tests for runner.run_task's terminal AgentState selection after an abort.

Covers the second half of the reported bug (Section E, problems #3 and #5 of
the architecture review): a run whose batch aborted on a PERMISSION_DENIED
action, but which had an earlier, unrelated, independently-verifiable effect,
must not be allowed to reach AgentState.COMPLETED. Before this fix,
run_task's terminal `loop.enter(...)` selection checked only
`final.verdict is Verdict.PASS`, with no regard for `loop.abort_reason`, so an
aborted run with any surviving PASS-ing effect was mislabeled COMPLETED. The
companion fix in api.py:_status() (tested separately in
test_policy_blocked_status.py) covers the same failure at the TaskStatus
layer; this file covers it at the AgentState layer, one level lower.

Also verifies the trace-level distinction introduced alongside it: verify_final
still runs (and is still diagnostically useful) after an abort, but is logged
through trace.note("diagnostic_verification_after_abort", ...) instead of the
normal trace.verification(..., checkpoint=False) completion channel.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agent_control.general_task import GeneralTask
from agent_control.policy import Policy
from agent_control.planner.mock import MockPlanner
from agent_control.runner import RunConfig, run_task
from agent_control.trace import Trace
from agent_control.types import Action, AgentState, Verdict


def _policy(workspace: Path) -> Policy:
    # This sandbox runs as root; refuse_if_elevated exists for a real user
    # environment and is unrelated to this fix, so it is disabled only here.
    return Policy(workspace=workspace, readable_roots=(), refuse_if_elevated=False)


def _capturing_trace(task_id: str) -> tuple[Trace, list[dict]]:
    events: list[dict] = []
    trace = Trace(task_id=task_id, condition="test", enabled=False,
                  on_event=events.append)
    return trace, events


def _states(events: list[dict]) -> list[str]:
    return [e["state"] for e in events if e["event"] == "agent_state"]


def _notes(events: list[dict], message: str) -> list[dict]:
    return [e for e in events if e["event"] == "note" and e["message"] == message]


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


def test_abort_after_unrelated_success_ends_failed_not_completed(workspace: Path):
    """The exact reported scenario: a scratch write succeeds, then a second
    action is denied for writing outside the workspace. The run must end
    AgentState.FAILED, never AgentState.COMPLETED."""
    policy = _policy(workspace)
    task = GeneralTask(request="write a scratch file, then write outside the workspace")
    trace, events = _capturing_trace(task.task_id)

    outside = Path(tempfile.mkdtemp()) / "escaped.txt"
    actions = [
        Action(kind="write_file", params={"path": "scratch.txt", "content": "hi"}),
        Action(kind="write_file", params={"path": str(outside), "content": "nope"}),
    ]
    planner = MockPlanner(reference_plan=actions, incremental=True)

    outcome = run_task(task, planner, policy, RunConfig(max_steps=8), trace=trace,
                        teardown=False)

    assert outcome.aborted_reason is not None
    assert outcome.aborted_reason.startswith("PERMISSION_DENIED")
    # The scratch write really did succeed and really does verify -- that is
    # what makes this a meaningful regression test rather than a vacuous one.
    assert outcome.verified is Verdict.PASS

    final_states = _states(events)
    # This is the actual regression check: even though verify_final PASSed,
    # the terminal state must be FAILED, and COMPLETED must never appear.
    assert final_states[-1] == AgentState.FAILED.value
    assert AgentState.COMPLETED.value not in final_states

    # verify_final's PASS was recorded diagnostically, not as a normal
    # completion. A checkpoint=True verification event is expected here (the
    # scratch write's own checkpoint, from mid-run) -- what must NOT appear
    # is a checkpoint=False event, which is verify_final's normal-completion
    # channel; the diagnostic note fired in its place instead.
    assert not [e for e in events if e["event"] == "verification"
                and e.get("checkpoint") is False]
    diag = _notes(events, "diagnostic_verification_after_abort")
    assert len(diag) == 1
    assert diag[0]["verdict"] == Verdict.PASS.value
    assert diag[0]["abort_reason"].startswith("PERMISSION_DENIED")


def test_abort_with_no_prior_effect_also_ends_failed(workspace: Path):
    """Negative-control-adjacent case: a denied action with nothing else in
    the ledger. verify_final's honest UNKNOWN fallback already routes this to
    FAILED even without the fix, so this pins down that the fix does not
    change that existing, correct behavior."""
    policy = _policy(workspace)
    task = GeneralTask(request="write outside the workspace")
    trace, events = _capturing_trace(task.task_id)

    outside = Path(tempfile.mkdtemp()) / "escaped.txt"
    actions = [Action(kind="write_file", params={"path": str(outside), "content": "nope"})]
    planner = MockPlanner(reference_plan=actions, incremental=True)

    outcome = run_task(task, planner, policy, RunConfig(max_steps=8), trace=trace,
                        teardown=False)

    assert outcome.aborted_reason is not None
    assert outcome.verified is Verdict.UNKNOWN
    assert _states(events)[-1] == AgentState.FAILED.value


def test_normal_success_still_reaches_completed(workspace: Path):
    """Control: an ordinary run with no denial still ends AgentState.COMPLETED.
    The fix must not make every run FAILED -- only aborted ones."""
    policy = _policy(workspace)
    task = GeneralTask(request="write a scratch file")
    trace, events = _capturing_trace(task.task_id)

    actions = [Action(kind="write_file", params={"path": "scratch.txt", "content": "hi"})]
    planner = MockPlanner(reference_plan=actions, incremental=True)

    outcome = run_task(task, planner, policy, RunConfig(max_steps=8), trace=trace,
                        teardown=False)

    assert outcome.aborted_reason is None
    assert outcome.verified is Verdict.PASS
    assert _states(events)[-1] == AgentState.COMPLETED.value

    # The normal completion channel was used -- a checkpoint=False
    # "verification" event, not the diagnostic note path (abort-only).
    assert [e for e in events if e["event"] == "verification"
            and e.get("checkpoint") is False]
    assert not _notes(events, "diagnostic_verification_after_abort")


def test_cancelled_run_unaffected_by_abort_reason(workspace: Path):
    """KeyboardInterrupt sets loop.abort_reason too (see run_task's except
    block), but `cancelled` must still take priority in the terminal state
    selection -- a cancelled run is not a failure. This pins the branch
    ordering (cancelled -> awaiting -> abort -> verdict) so a future edit
    can't silently swap FAILED in ahead of CANCELLED."""
    policy = _policy(workspace)
    task = GeneralTask(request="write a scratch file")
    trace, events = _capturing_trace(task.task_id)

    class _InterruptingPlanner:
        name = "interrupting"

        def plan(self, goal, state, history):
            raise KeyboardInterrupt

    outcome = run_task(task, _InterruptingPlanner(), policy, RunConfig(max_steps=8),
                        trace=trace, teardown=False)

    assert outcome.cancelled is True
    assert outcome.aborted_reason is not None  # set by the except block
    assert _states(events)[-1] == AgentState.CANCELLED.value

