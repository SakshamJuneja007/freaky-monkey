from __future__ import annotations

from pathlib import Path

import pytest

from agent_control import os_tools
from agent_control.policy import Decision, Policy
from agent_control.skills.filesystem import (
    FILESYSTEM_ACTION_KINDS,
    FilesystemSkill,
)
from agent_control.skills.registry import SkillRegistry
from agent_control.skills.security import Capability
from agent_control.types import Action


def test_filesystem_skill_exposes_existing_action_kinds():
    skill = FilesystemSkill(
        Policy(
            workspace=Path("."),
            refuse_if_elevated=False,
        )
    )

    assert skill.action_kinds == FILESYSTEM_ACTION_KINDS

    assert skill.supports("create_dir")
    assert skill.supports("write_file")
    assert skill.supports("fetch_file")
    assert skill.supports("open_file")
    assert skill.supports("list_directory")
    assert skill.supports("read_text_file")
    assert skill.supports("search_files")


def test_filesystem_skill_has_expected_metadata():
    skill = FilesystemSkill(
        Policy(
            workspace=Path("."),
            refuse_if_elevated=False,
        )
    )

    assert skill.name == "filesystem"
    assert skill.description

    assert Capability.FILESYSTEM_READ in skill.manifest.capabilities
    assert Capability.FILESYSTEM_WRITE in skill.manifest.capabilities
    assert skill.manifest.side_effecting is True


def test_filesystem_skill_rejects_unknown_action():
    skill = FilesystemSkill(
        Policy(
            workspace=Path("."),
            refuse_if_elevated=False,
        )
    )

    action = Action(
        kind="not_a_filesystem_action",
        consequential=False,
    )

    with pytest.raises(
        ValueError,
        match="does not support",
    ):
        skill.adapt_action(action)


def test_filesystem_skill_rejects_non_action():
    skill = FilesystemSkill(
        Policy(
            workspace=Path("."),
            refuse_if_elevated=False,
        )
    )

    with pytest.raises(
        TypeError,
        match="expects a core Action",
    ):
        skill.adapt_action(object())


def test_filesystem_skill_preserves_core_action():
    skill = FilesystemSkill(
        Policy(
            workspace=Path("."),
            refuse_if_elevated=False,
        )
    )

    action = Action(
        kind="read_text_file",
        params={"path": "example.txt"},
        consequential=False,
    )

    adapted = skill.adapt_action(action)

    assert adapted is action


def test_registry_resolves_filesystem_action():
    policy = Policy(
        workspace=Path("."),
        refuse_if_elevated=False,
    )

    skill = FilesystemSkill(policy)

    registry = SkillRegistry()
    report = registry.register(skill)

    assert report.status.value == "APPROVED"

    resolved = registry.find_for_action(
        "read_text_file"
    )

    assert resolved is skill


def test_registry_resolves_write_action_to_filesystem_skill():
    policy = Policy(
        workspace=Path("."),
        refuse_if_elevated=False,
    )

    skill = FilesystemSkill(policy)

    registry = SkillRegistry()
    registry.register(skill)

    assert registry.find_for_action(
        "write_file"
    ) is skill


def test_registry_returns_none_for_unknown_filesystem_action():
    policy = Policy(
        workspace=Path("."),
        refuse_if_elevated=False,
    )

    skill = FilesystemSkill(policy)

    registry = SkillRegistry()
    registry.register(skill)

    assert registry.find_for_action(
        "unknown_filesystem_action"
    ) is None


def test_filesystem_executor_delegates_to_existing_os_tools(
    monkeypatch,
):
    policy = Policy(
        workspace=Path("."),
        refuse_if_elevated=False,
    )

    skill = FilesystemSkill(policy)

    action = Action(
        kind="read_text_file",
        params={"path": "example.txt"},
        consequential=False,
    )

    expected = object()
    calls: list[tuple[object, Action]] = []

    def fake_execute(
        received_policy,
        received_action,
    ):
        calls.append(
            (
                received_policy,
                received_action,
            )
        )
        return expected

    monkeypatch.setattr(
        os_tools,
        "execute",
        fake_execute,
    )

    result = skill.executor().execute(action)

    assert result is expected

    assert calls == [
        (
            policy,
            action,
        )
    ]


def test_filesystem_skill_does_not_bypass_policy(
    monkeypatch,
):
    """Policy denial must occur before filesystem execution.

    This test intentionally exercises the policy boundary directly rather
    than allowing the skill to manufacture its own authorization decision.
    """

    events: list[str] = []

    class DenyingPolicy:
        def check(self, action):
            events.append(
                f"policy:{action.kind}"
            )
            return (
                Decision.DENY,
                "blocked for test",
            )

    policy = DenyingPolicy()

    skill = FilesystemSkill(policy)

    def fail_if_called(*args, **kwargs):
        pytest.fail(
            "filesystem executor was called despite policy denial"
        )

    monkeypatch.setattr(
        os_tools,
        "execute",
        fail_if_called,
    )

    action = Action(
        kind="write_file",
        params={
            "path": "blocked.txt",
            "content": "must not execute",
        },
        consequential=True,
    )

    decision, reason = policy.check(action)

    assert decision is Decision.DENY
    assert reason == "blocked for test"

    assert events == [
        "policy:write_file",
    ]


def test_filesystem_verifier_defers_to_core_verification():
    policy = Policy(
        workspace=Path("."),
        refuse_if_elevated=False,
    )

    skill = FilesystemSkill(policy)

    action = Action(
        kind="write_file",
        params={
            "path": "example.txt",
            "content": "hello",
        },
        consequential=True,
    )

    result = object()

    verification = skill.verifier().verify(
        action,
        result,
    )

    assert verification.ok is False

    assert (
        verification.detail
        == (
            "filesystem verification is delegated "
            "to the task-level verifier"
        )
    )


def test_filesystem_skill_does_not_claim_network_capability():
    skill = FilesystemSkill(
        Policy(
            workspace=Path("."),
            refuse_if_elevated=False,
        )
    )

    assert Capability.NETWORK not in (
        skill.manifest.capabilities
    )