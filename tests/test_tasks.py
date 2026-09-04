"""Verification semantics of the registered task, checked against the world.

The task registry now holds exactly one task, ``open_last_day_pdf``, and these
tests are about the only question that matters for it: **what does the verifier
actually prove?** Existence is not opening, and a file landing at the right path
is the classic false success this project measures rather than assumes -- so the
weakness in the current check is pinned here as a test rather than left as a
comment nobody reads.

The earlier tests in this file covered a checkpoint defect in ``create_directory``
and ``full_project_setup`` (a correct first action verified against the task's
goal instead of its own postcondition). Those tasks are no longer registered, so
that coverage went with them; the defect itself is described in
``benchmark/tasks/group_a.py`` history, and the same shape is re-asserted below
for the task that exists now.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_control import api
from agent_control.policy import Policy
from agent_control.types import Action, Verdict
from benchmark.tasks import build_task


@pytest.fixture
def downloads(tmp_path: Path) -> Path:
    """A stand-in for the user's real Downloads directory."""
    root = tmp_path / "Downloads"
    (root / "Gyansetu Internship Assignment" / "Certificates").mkdir(parents=True)
    return root


@pytest.fixture
def task(monkeypatch, downloads: Path):
    """The real task, pointed at a temporary Downloads tree.

    Patched on the instance rather than on ``Path.home``: the point is to test the
    task's verification, not to relocate the whole process's idea of home.
    """
    made = build_task("open_last_day_pdf")
    monkeypatch.setattr(made, "downloads_dir", lambda: downloads)
    made.setup(_policy_for(downloads))
    return made


def _policy_for(downloads: Path) -> Policy:
    """Downloads granted read-only, exactly as ``api`` grants it for this task."""
    return Policy(workspace=downloads.parent / "ws",
                  readable_roots=(downloads,),
                  refuse_if_elevated=False)


@pytest.fixture
def confined(downloads: Path) -> Policy:
    return _policy_for(downloads)


def write_pdf(task, content: bytes) -> Path:
    target = task.pdf_path()
    target.write_bytes(content)
    return target


def open_file(path: str) -> Action:
    return Action(kind="open_file", params={"path": path})


# -- the goal check ----------------------------------------------------------

def test_a_missing_pdf_fails_rather_than_going_unresolved(task, confined) -> None:
    """FAIL and UNKNOWN are different claims. The parent directory is readable,
    so "not there" is a finding, not an unanswerable question."""
    result = task.verify_final(confined)
    assert result.verdict is Verdict.FAIL
    assert [c.name for c in result.checks] == ["file_exists"]


def test_an_empty_pdf_does_not_count_as_opened(task, confined) -> None:
    """A zero-byte file at the right path is the shape of a failed download or a
    truncated copy. ``min_bytes=1`` is what stops presence from meaning success."""
    write_pdf(task, b"")
    result = task.verify_final(confined)
    assert result.verdict is Verdict.FAIL
    assert any(c.name == "file_min_bytes" and c.verdict is Verdict.FAIL
               for c in result.checks)


def test_a_real_pdf_passes(task, confined) -> None:
    write_pdf(task, b"%PDF-1.7\n%%EOF\n")
    assert task.verify_final(confined).verdict is Verdict.PASS


# -- the checkpoint asserts the action's own postcondition -------------------

def test_only_the_open_action_is_checkpointed(task, confined) -> None:
    """An action of another kind returns None, so the runner does not judge it
    against this task's file and burn the recovery budget on an unrelated step."""
    assert task.verify_checkpoint(confined, Action(kind="create_dir",
                                                  params={"path": "x"})) is None


def test_the_checkpoint_fails_when_the_file_is_not_there(task, confined) -> None:
    result = task.verify_checkpoint(confined, open_file(str(task.pdf_path())))
    assert result is not None and result.verdict is Verdict.FAIL


def test_the_checkpoint_passes_on_the_real_file(task, confined) -> None:
    write_pdf(task, b"%PDF-1.7\n%%EOF\n")
    result = task.verify_checkpoint(confined, open_file(str(task.pdf_path())))
    assert result is not None and result.verdict is Verdict.PASS


# -- what the current check does *not* prove ---------------------------------

def test_the_final_check_cannot_tell_that_the_pdf_was_ever_opened(task,
                                                                  confined) -> None:
    """A recorded limitation, not an endorsement.

    ``verify_final`` observes the file, never a process or a window, so a run that
    opened nothing still verifies PASS as long as the PDF exists. That makes the
    task's success criterion "the file is present and non-empty" -- weaker than
    its goal sentence claims. Pinning it here means a future check on process or
    window state has a failing test to turn green, instead of this gap being
    discovered by trusting a spoken "task complete".
    """
    write_pdf(task, b"%PDF-1.7\n%%EOF\n")
    assert task.verify_final(confined).verdict is Verdict.PASS  # nothing was opened


# -- the one place workspace confinement is widened -------------------------

def test_downloads_is_granted_read_only_and_only_for_this_task() -> None:
    """The sandbox stays the only write root. This task needs to read an existing
    file outside it, so the grant is narrow, explicit, and per task id."""
    granted = api.readable_roots_for_task("open_last_day_pdf")
    assert granted == (Path.home() / "Downloads",)
    assert api.readable_roots_for_task("anything_else") == ()


def test_a_path_outside_the_granted_roots_is_still_refused(confined) -> None:
    from agent_control.policy import PolicyDenied

    with pytest.raises(PolicyDenied):
        confined.resolve_read_path(str(Path.home() / "Documents" / "secrets.txt"))


def test_the_benchmark_grants_a_trial_the_same_roots_as_the_api(monkeypatch,
                                                               tmp_path) -> None:
    """The two execution entry points must not disagree about what a task may read.

    They did. ``harness.run_trial`` built ``Policy(workspace=workspace)`` with no
    readable roots, so every ``open_last_day_pdf`` trial aborted on
    ``PolicyDenied: read outside permitted roots`` before its first action while
    the identical task verified PASS through ``run_agent_task``. The grid reported
    0% verified success for a task that works -- Directive A's "a broken baseline
    is not evidence", arrived at by a policy divergence rather than a broken agent.

    Asserted on the ``Policy`` the harness actually constructs, not on the source
    text, so the test still holds if the call is refactored.
    """
    from agent_control.trace import Trace
    from benchmark import harness

    seen: dict[str, object] = {}

    def capture(task, planner, policy, config, **kwargs):
        seen["policy"] = policy
        raise AssertionError("stop before the run: the policy is what is under test")

    monkeypatch.setattr(harness, "run_task", capture)
    monkeypatch.setattr(harness, "Trace",
                        lambda **kw: Trace(**kw, trace_dir=tmp_path / "traces"))

    with pytest.raises(AssertionError, match="stop before the run"):
        harness.run_trial(harness.CONDITIONS["structured_hybrid"],
                          "open_last_day_pdf", 0, planner_kind="mock",
                          root=tmp_path / "ws", max_steps=4, llm=None, vlm=None)

    policy = seen["policy"]
    assert policy.readable_roots == api.readable_roots_for_task("open_last_day_pdf")
    assert policy.resolve_read_path(str(Path.home() / "Downloads" / "last day.pdf"))
