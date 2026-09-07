"""Regression tests for the POLICY_BLOCKED / false-success integrity fix.

Bug this guards against: a run aborted because ``Policy`` denied one action
could still be reported as ``TaskStatus.SUCCESS`` if ``verify_final`` found an
*unrelated* effect (recorded before the denial) that verified PASS. The denied
action itself is invisible to ``verify_final`` by construction (a refused path
never becomes a recorded effect -- see ``GeneralTask._remember``), so nothing
in the verification result ever names the denial; only ``RunOutcome.aborted_reason``
does. ``api._status()`` must therefore treat a ``PERMISSION_DENIED`` abort as
terminal and check it *before* ``Verdict.PASS``.

Two layers are tested:

* Unit tests against ``api._status()`` directly, which pin the exact
  precedence rule with a hand-built ``RunOutcome`` -- fast, and exercises
  every combination of PASS/absence-of-PASS x aborted/not-aborted.
* One end-to-end test that runs a real ``GeneralTask`` through
  ``runner.run_task`` with a real ``Policy`` and a scripted planner that (a)
  writes a file inside the workspace (succeeds and verifies), then (b) writes
  a file outside the workspace (denied) -- reproducing the reported failure
  mode exactly, without mocking the policy/verification layers themselves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_control import api
from agent_control.api import AgentResult, TaskStatus, _result_from, _status
from agent_control.general_task import GeneralTask
from agent_control.planner.base import PlannerStep, Usage
from agent_control.policy import Policy
from agent_control.response import DeimosPresentation
from agent_control.runner import RunConfig, RunOutcome, run_task
from agent_control.types import Action, Check, FailureClass, Verdict, VerificationResult


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _outcome(
    *,
    verified: Verdict,
    aborted_reason: str | None,
    reported_success: bool = False,
    completed: tuple[str, ...] = ("some_effect/dir_exists",),
) -> RunOutcome:
    """A minimal, otherwise-ordinary RunOutcome for exercising ``_status``."""
    checks = [
        Check(name=name, verdict=Verdict.PASS, evidence={})
        for name in completed
    ]
    final = VerificationResult(checks=checks, label="general:test")
    return RunOutcome(
        task_id="general-test0000",
        condition="structured_hybrid",
        trial=0,
        planner_name="mock",
        reported_success=reported_success,
        verified=verified,
        final=final,
        aborted_reason=aborted_reason,
        cancelled=False,
        awaiting=False,
    )


def _derive_and_status(outcome: RunOutcome) -> tuple[TaskStatus, str]:
    completed, failed, unresolved, _ = api._derive(outcome)
    return _status(outcome, completed, failed, unresolved)


# ---------------------------------------------------------------------------
# 1. a permission-denied action, and 2. an abort_reason containing
#    permission_denied, and 3. verification returning PASS after that abort,
#    and 4. final TaskStatus still being POLICY_BLOCKED
#    -- pinned directly against _status()
# ---------------------------------------------------------------------------


def test_policy_blocked_beats_verified_pass():
    """The exact reported bug: verify_final found PASS on an unrelated
    effect, but the run was aborted by a policy denial. Status must be
    POLICY_BLOCKED, never SUCCESS."""
    outcome = _outcome(
        verified=Verdict.PASS,
        aborted_reason=(
            f"{FailureClass.PERMISSION_DENIED.value}: "
            "permission_denied needs a user decision; no interactive approver"
        ),
    )
    status, detail = _derive_and_status(outcome)
    assert status is TaskStatus.POLICY_BLOCKED
    assert "permission_denied" in detail.lower()


def test_policy_blocked_beats_verified_pass_even_with_reported_success():
    """Even if something upstream let ``reported_success`` come back True,
    an aborted-by-policy run must not become SUCCESS."""
    outcome = _outcome(
        verified=Verdict.PASS,
        aborted_reason=f"{FailureClass.PERMISSION_DENIED.value}: write outside workspace",
        reported_success=True,
    )
    status, _ = _derive_and_status(outcome)
    assert status is TaskStatus.POLICY_BLOCKED


def test_policy_blocked_when_verification_unknown():
    """Same precedence rule when verify_final could not determine a verdict
    at all (empty ledger) -- POLICY_BLOCKED must still win over falling
    through to plain UNKNOWN."""
    outcome = _outcome(
        verified=Verdict.UNKNOWN,
        aborted_reason=f"{FailureClass.PERMISSION_DENIED.value}: write outside workspace",
        completed=(),
    )
    status, _ = _derive_and_status(outcome)
    assert status is TaskStatus.POLICY_BLOCKED


def test_unrelated_abort_reason_does_not_trigger_policy_blocked():
    """Guard against over-broad matching: an abort for a different reason
    (not permission denial) with a PASS verdict should still be SUCCESS."""
    outcome = _outcome(
        verified=Verdict.PASS,
        aborted_reason="step ceiling (10) reached",
    )
    status, _ = _derive_and_status(outcome)
    assert status is TaskStatus.SUCCESS


def test_no_abort_and_pass_is_still_success():
    """Sanity check: the reorder must not affect the ordinary success path."""
    outcome = _outcome(verified=Verdict.PASS, aborted_reason=None)
    status, _ = _derive_and_status(outcome)
    assert status is TaskStatus.SUCCESS


# ---------------------------------------------------------------------------
# 5. AgentResult.ok being False, and diagnostic info being preserved
# ---------------------------------------------------------------------------


def test_agent_result_ok_is_false_when_policy_blocked():
    outcome = _outcome(
        verified=Verdict.PASS,
        aborted_reason=f"{FailureClass.PERMISSION_DENIED.value}: write outside workspace",
    )
    result = _result_from(
        request="update list_project.ps1",
        outcome=outcome,
        root=Path("/tmp/does-not-matter"),
    )
    assert result.status is TaskStatus.POLICY_BLOCKED
    assert result.ok is False
    # Diagnostic information from verify_final must still be visible on the
    # result even though it is not allowed to change the terminal status.
    assert result.completed == ["some_effect/dir_exists"]
    assert result.verified == Verdict.PASS.value
    assert result.aborted_reason is not None
    assert "permission_denied" in result.aborted_reason.lower()


# ---------------------------------------------------------------------------
# 6. DeimosPresentation.result() not saying "Done" or "verification passed"
# ---------------------------------------------------------------------------


def test_presentation_does_not_claim_done_when_policy_blocked():
    outcome = _outcome(
        verified=Verdict.PASS,
        aborted_reason=f"{FailureClass.PERMISSION_DENIED.value}: write outside workspace",
    )
    result = _result_from(
        request="update list_project.ps1",
        outcome=outcome,
        root=Path("/tmp/does-not-matter"),
    )
    sentence = DeimosPresentation().result(result)
    assert "Done" not in sentence
    assert "verification passed" not in sentence.lower()
    assert result.ok is False  # result() takes the ``not result.ok`` branch


# ---------------------------------------------------------------------------
# End-to-end: real Policy + real GeneralTask + scripted planner through the
# actual control loop, reproducing the reported scenario without mocking
# policy or verification.
# ---------------------------------------------------------------------------


class _ScriptedPlanner:
    """Replays a fixed sequence of PlannerStep objects, one per call."""

    name = "scripted"

    def __init__(self, steps: list[PlannerStep]):
        self._steps = list(steps)
        self.usage = Usage()

    def plan(self, goal, state, history) -> PlannerStep:
        self.usage.add(prompt=1, completion=1)
        if self._steps:
            return self._steps.pop(0)
        return PlannerStep(done=True, reasoning="nothing left scripted")


def test_end_to_end_permission_denied_action_yields_policy_blocked(tmp_path):
    """A write inside the workspace succeeds and verifies; a later write
    outside the workspace is denied by Policy. The overall run must report
    POLICY_BLOCKED, AgentResult.ok must be False, and the presentation layer
    must not claim completion -- even though the first effect really did
    verify PASS."""
    workspace = tmp_path / "sandbox"
    outside = tmp_path / "outside"
    outside.mkdir()

    # ``refuse_if_elevated=False`` only because this test may run under a
    # root/administrator CI or container user; the elevation refusal itself
    # (plan S10) is unrelated to this fix and is not being tested here.
    policy = Policy(workspace=workspace, readable_roots=(), refuse_if_elevated=False)

    ok_write = Action(
        kind="write_file",
        params={"path": "notes.md", "content": "investigation notes"},
    )
    denied_write = Action(
        kind="write_file",
        # Absolute path outside the workspace -> Policy.resolve_write_path
        # raises PolicyDenied inside os_tools' _guard, exactly reproducing
        # the reported "attempted to update list_project.ps1" scenario.
        params={"path": str(outside / "list_project.ps1"), "content": "# listing"},
    )

    planner = _ScriptedPlanner([
        PlannerStep(actions=[ok_write]),
        PlannerStep(actions=[denied_write]),
    ])

    task = GeneralTask(
        request="investigate the project and update list_project.ps1",
        readable_roots=(),
    )

    config = RunConfig(max_steps=5, interactive=False, recovery_enabled=True)
    outcome = run_task(task, planner, policy, config, teardown=False)

    # Sanity: the scenario actually reproduces what it claims to.
    assert outcome.aborted_reason is not None
    assert outcome.aborted_reason.startswith(FailureClass.PERMISSION_DENIED.value)
    assert outcome.verified is Verdict.PASS  # the earlier write really did verify
    assert outcome.reported_success is False  # planner never got to claim done

    result = _result_from(
        request="investigate the project and update list_project.ps1",
        outcome=outcome,
        root=workspace,
    )

    assert result.status is TaskStatus.POLICY_BLOCKED
    assert result.ok is False

    sentence = DeimosPresentation().result(result)
    assert "Done" not in sentence
    assert "verification passed" not in sentence.lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
