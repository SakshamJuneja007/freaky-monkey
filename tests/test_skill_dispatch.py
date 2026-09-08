from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from agent_control import runner
from agent_control.policy import Decision, Policy
from agent_control.recovery import Recovery, RecoveryBudget
from agent_control.runner import RunConfig, _Loop
from agent_control.skills.base import (
    Skill,
    SkillAction,
    SkillInfo,
)
from agent_control.skills.registry import SkillRegistry
from agent_control.types import (
    Action,
    ActionResult,
)


@dataclass(frozen=True)
class DummyExecutionResult:
    """Minimal skill execution result used by dispatch tests."""

    ok: bool
    detail: str = ""
    value: object | None = None


@dataclass(frozen=True)
class DummyVerificationResult:
    """Minimal skill verification result used by dispatch tests."""

    ok: bool
    detail: str = ""


class RecordingPolicy:
    """Policy wrapper that records execution ordering."""

    def __init__(
        self,
        policy: Policy,
        events: list[str],
    ) -> None:
        self._policy = policy
        self._events = events

    def check(
        self,
        action: Action,
    ):
        self._events.append(
            f"policy:{action.kind}"
        )

        return self._policy.check(
            action
        )


class RecordingExecutor:
    """Executor that records when skill execution occurs."""

    def __init__(
        self,
        events: list[str],
    ) -> None:
        self._events = events
        self.actions: list[object] = []

    def execute(
        self,
        action,
    ) -> DummyExecutionResult:
        self._events.append("executor")
        self.actions.append(action)

        return DummyExecutionResult(
            ok=True,
            detail="executed",
            value={"executed": True},
        )


class RecordingVerifier:
    """Verifier that records when independent verification occurs."""

    def __init__(
        self,
        events: list[str],
    ) -> None:
        self._events = events
        self.calls: list[
            tuple[object, object]
        ] = []

    def verify(
        self,
        action,
        result,
    ) -> DummyVerificationResult:
        self._events.append("verifier")
        self.calls.append(
            (action, result)
        )

        return DummyVerificationResult(
            ok=True,
            detail="verified",
        )


class RecordingSkill(Skill):
    """Skill used to test the runner dispatch pipeline."""

    def __init__(
        self,
        events: list[str],
        *,
        name: str = "recording",
        action_kind: str = "skill_action",
    ) -> None:
        self._events = events

        self._info = SkillInfo(
            name=name,
            description="Skill used by dispatch tests",
            actions=(
                SkillAction(
                    kind=action_kind,
                    description="Test action",
                ),
            ),
        )

        self._executor = RecordingExecutor(
            events
        )
        self._verifier = RecordingVerifier(
            events
        )

    @property
    def info(self) -> SkillInfo:
        return self._info

    def adapt_action(
        self,
        action: Action,
    ):
        self._events.append("adapt")

        return {
            "kind": action.kind,
            "params": dict(
                action.params
            ),
        }

    def executor(self):
        return self._executor

    def verifier(self):
        return self._verifier


class MinimalTask:
    """Smallest task implementation needed to construct _Loop."""

    task_id = "skill_dispatch"
    goal = "test skill dispatch"
    bucket = "test"

    def setup(self, policy) -> None:
        return None

    def observe(
        self,
        policy,
        trace=None,
    ) -> dict:
        return {}

    def reference_plan(
        self,
        policy,
    ) -> list:
        return []

    def verify_final(
        self,
        policy,
        trace=None,
    ):
        return None

    def verify_checkpoint(
        self,
        policy,
        action,
        trace=None,
    ):
        return None

    def teardown(
        self,
        policy,
    ) -> None:
        return None


def make_loop(
    trace,
    tmp_path: Path,
    *,
    policy,
    skills: SkillRegistry | None,
) -> _Loop:
    """Create the smallest loop context needed by _permit_and_execute."""

    return _Loop(
        task=MinimalTask(),
        policy=policy,
        config=RunConfig(),
        trace=trace,
        recovery=Recovery(
            budget=RecoveryBudget(),
            enabled=True,
        ),
        skills=skills,
    )


def test_action_is_dispatched_to_correct_skill(
    trace,
    tmp_path,
):
    events: list[str] = []

    base_policy = Policy(
        workspace=tmp_path / "ws",
        refuse_if_elevated=False,
    )

    policy = RecordingPolicy(
        base_policy,
        events,
    )

    registry = SkillRegistry()

    first = RecordingSkill(
        events,
        name="first",
        action_kind="first_action",
    )

    second = RecordingSkill(
        events,
        name="second",
        action_kind="second_action",
    )

    registry.register(first)
    registry.register(second)

    loop = make_loop(
        trace,
        tmp_path,
        policy=policy,
        skills=registry,
    )

    result = runner._permit_and_execute(
        loop,
        Action(
            kind="second_action",
            consequential=False,
        ),
    )

    assert result.ok is True
    assert result.detail["skill"] == "second"

    assert first._executor.actions == []
    assert len(second._executor.actions) == 1


def test_policy_runs_before_skill_execution(
    trace,
    tmp_path,
):
    events: list[str] = []

    base_policy = Policy(
        workspace=tmp_path / "ws",
        refuse_if_elevated=False,
    )

    policy = RecordingPolicy(
        base_policy,
        events,
    )

    registry = SkillRegistry()

    skill = RecordingSkill(
        events,
        action_kind="skill_action",
    )

    registry.register(skill)

    loop = make_loop(
        trace,
        tmp_path,
        policy=policy,
        skills=registry,
    )

    result = runner._permit_and_execute(
        loop,
        Action(
            kind="skill_action",
            consequential=False,
        ),
    )

    assert result.ok is True

    assert events == [
        "policy:skill_action",
        "adapt",
        "executor",
        "verifier",
    ]


def test_skill_executor_runs(
    trace,
    tmp_path,
):
    events: list[str] = []

    base_policy = Policy(
        workspace=tmp_path / "ws",
        refuse_if_elevated=False,
    )

    policy = RecordingPolicy(
        base_policy,
        events,
    )

    registry = SkillRegistry()

    skill = RecordingSkill(
        events,
        action_kind="skill_action",
    )

    registry.register(skill)

    loop = make_loop(
        trace,
        tmp_path,
        policy=policy,
        skills=registry,
    )

    action = Action(
        kind="skill_action",
        params={"value": 123},
        consequential=False,
    )

    result = runner._permit_and_execute(
        loop,
        action,
    )

    assert result.ok is True
    assert len(
        skill._executor.actions
    ) == 1

    assert skill._executor.actions[0] == {
        "kind": "skill_action",
        "params": {"value": 123},
    }


def test_skill_verifier_runs(
    trace,
    tmp_path,
):
    events: list[str] = []

    base_policy = Policy(
        workspace=tmp_path / "ws",
        refuse_if_elevated=False,
    )

    policy = RecordingPolicy(
        base_policy,
        events,
    )

    registry = SkillRegistry()

    skill = RecordingSkill(
        events,
        action_kind="skill_action",
    )

    registry.register(skill)

    loop = make_loop(
        trace,
        tmp_path,
        policy=policy,
        skills=registry,
    )

    result = runner._permit_and_execute(
        loop,
        Action(
            kind="skill_action",
            consequential=False,
        ),
    )

    assert result.ok is True

    assert len(
        skill._verifier.calls
    ) == 1

    verified_action, execution_result = (
        skill._verifier.calls[0]
    )

    assert verified_action == {
        "kind": "skill_action",
        "params": {},
    }

    assert execution_result.ok is True


def test_unsupported_action_falls_back_to_os_tools(
    monkeypatch,
    trace,
    tmp_path,
):
    events: list[str] = []

    base_policy = Policy(
        workspace=tmp_path / "ws",
        refuse_if_elevated=False,
    )

    policy = RecordingPolicy(
        base_policy,
        events,
    )

    registry = SkillRegistry()

    skill = RecordingSkill(
        events,
        action_kind="supported_action",
    )

    registry.register(skill)

    loop = make_loop(
        trace,
        tmp_path,
        policy=policy,
        skills=registry,
    )

    fallback_calls: list[Action] = []

    def fake_execute(
        fallback_policy,
        action: Action,
    ) -> ActionResult:
        events.append("os_tools")
        fallback_calls.append(action)

        return ActionResult(
            action=action,
            ok=True,
            detail={
                "executor": "os_tools",
            },
        )

    monkeypatch.setattr(
        runner.os_tools,
        "execute",
        fake_execute,
    )

    action = Action(
        kind="unsupported_action",
        consequential=False,
    )

    result = runner._permit_and_execute(
        loop,
        action,
    )

    assert result.ok is True

    assert fallback_calls == [
        action
    ]

    assert skill._executor.actions == []
    assert skill._verifier.calls == []

    assert events == [
        "policy:unsupported_action",
        "os_tools",
    ]


def test_policy_denial_prevents_skill_execution(
    monkeypatch,
    trace,
    tmp_path,
):
    """Guard the central safety invariant explicitly.

    A matching skill must not execute when policy denies the action.
    """

    events: list[str] = []

    registry = SkillRegistry()

    skill = RecordingSkill(
        events,
        action_kind="blocked_action",
    )

    registry.register(skill)

    class DenyingPolicy:
        def check(
            self,
            action: Action,
        ):
            events.append(
                f"policy:{action.kind}"
            )

            return (
                Decision.DENY,
                "blocked for test",
            )

    loop = make_loop(
        trace,
        tmp_path,
        policy=DenyingPolicy(),
        skills=registry,
    )

    def fail_if_called(*args, **kwargs):
        pytest.fail(
            "os_tools fallback should not run "
            "when policy denies the action"
        )

    monkeypatch.setattr(
        runner.os_tools,
        "execute",
        fail_if_called,
    )

    result = runner._permit_and_execute(
        loop,
        Action(
            kind="blocked_action",
            consequential=False,
        ),
    )

    assert result.ok is False
    assert events == [
        "policy:blocked_action",
    ]

    assert skill._executor.actions == []
    assert skill._verifier.calls == []