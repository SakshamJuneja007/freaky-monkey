from __future__ import annotations

import pytest

from agent_control.skills.base import (
    Skill,
    SkillAction,
    SkillInfo,
)
from agent_control.skills.registry import SkillRegistry
from agent_control.skills.status import SkillStatus


class DummyExecutor:
    def execute(self, action):
        return {"executed": action}


class DummyVerifier:
    def verify(self, action, result):
        return True


class DummySkill(Skill):
    def __init__(
        self,
        name: str = "dummy",
        action_kinds: tuple[str, ...] = ("dummy_action",),
    ) -> None:
        self._info = SkillInfo(
            name=name,
            description=f"Dummy skill: {name}",
            actions=tuple(
                SkillAction(
                    kind=kind,
                    description=f"Does {kind}",
                )
                for kind in action_kinds
            ),
        )

        self._executor = DummyExecutor()
        self._verifier = DummyVerifier()

    @property
    def info(self) -> SkillInfo:
        return self._info

    def executor(self):
        return self._executor

    def verifier(self):
        return self._verifier

    def adapt_action(self, action):
        return action


def test_skill_name_is_derived_from_info():
    skill = DummySkill(name="example")

    assert skill.name == "example"


def test_skill_description_is_derived_from_info():
    skill = DummySkill(name="example")

    assert skill.description == "Dummy skill: example"


def test_skill_action_kinds_are_derived_from_info():
    skill = DummySkill(
        action_kinds=(
            "first_action",
            "second_action",
        )
    )

    assert skill.action_kinds == (
        "first_action",
        "second_action",
    )


def test_skill_supports_known_action():
    skill = DummySkill(
        action_kinds=("known_action",)
    )

    assert skill.supports("known_action") is True


def test_skill_does_not_support_unknown_action():
    skill = DummySkill(
        action_kinds=("known_action",)
    )

    assert skill.supports("unknown_action") is False


def test_registry_registers_and_gets_skill():
    registry = SkillRegistry()
    skill = DummySkill(name="dummy")

    registry.register(skill)

    assert registry.get("dummy") is skill


def test_registry_find_returns_none_for_unknown_skill():
    registry = SkillRegistry()

    assert registry.find("missing") is None


def test_registry_get_unknown_skill_raises_key_error():
    registry = SkillRegistry()

    with pytest.raises(KeyError):
        registry.get("missing")


def test_registry_rejects_duplicate_skill_names():
    registry = SkillRegistry()

    registry.register(DummySkill(name="duplicate"))

    with pytest.raises(
        ValueError,
        match="already registered",
    ):
        registry.register(DummySkill(name="duplicate"))


def test_registry_contains_registered_skill():
    registry = SkillRegistry()

    registry.register(DummySkill(name="dummy"))

    assert "dummy" in registry


def test_registry_does_not_contain_unknown_skill():
    registry = SkillRegistry()

    assert "missing" not in registry


def test_registry_length():
    registry = SkillRegistry()

    assert len(registry) == 0

    registry.register(DummySkill(name="first"))
    registry.register(DummySkill(name="second"))

    assert len(registry) == 2


def test_registry_names_are_deterministic():
    registry = SkillRegistry()

    registry.register(DummySkill(name="zebra"))
    registry.register(DummySkill(name="alpha"))

    assert registry.names() == (
        "alpha",
        "zebra",
    )


def test_registry_find_for_action_returns_matching_skill():
    registry = SkillRegistry()

    first = DummySkill(
        name="first",
        action_kinds=("first_action",),
    )
    second = DummySkill(
        name="second",
        action_kinds=("second_action",),
    )

    registry.register(first)
    registry.register(second)

    assert registry.find_for_action(
        "second_action"
    ) is second


def test_registry_find_for_unknown_action_returns_none():
    registry = SkillRegistry()

    registry.register(
        DummySkill(
            action_kinds=("known_action",),
        )
    )

    assert registry.find_for_action(
        "unknown_action"
    ) is None


def test_registry_rejects_duplicate_action_kinds():
    registry = SkillRegistry()

    registry.register(
        DummySkill(
            name="first",
            action_kinds=("shared_action",),
        )
    )

    registry.register(
        DummySkill(
            name="second",
            action_kinds=("shared_action",),
        )
    )

    with pytest.raises(
        ValueError,
        match="multiple skills support action",
    ):
        registry.find_for_action(
            "shared_action"
        )


def test_registry_action_kinds_returns_all_unique_actions():
    registry = SkillRegistry()

    registry.register(
        DummySkill(
            name="first",
            action_kinds=(
                "first_action",
                "another_action",
            ),
        )
    )

    registry.register(
        DummySkill(
            name="second",
            action_kinds=("second_action",),
        )
    )

    assert registry.action_kinds() == (
        "another_action",
        "first_action",
        "second_action",
    )
@pytest.mark.parametrize("status", [
    SkillStatus.RESTRICTED,
    SkillStatus.QUARANTINED,
    SkillStatus.BLOCKED,
])
def test_find_for_action_never_dispatches_non_approved_skill(status):
    from agent_control.skills.audit import SkillAuditReport
    from agent_control.skills.security import Capability

    registry = SkillRegistry()
    skill = DummySkill(name=f"{status.value.lower()}_skill", action_kinds=("protected_action",))
    registry.register(skill)

    registry._reports[skill.name] = SkillAuditReport(
        skill_name=skill.name,
        declared_capabilities=frozenset(),
        observed_capabilities=frozenset({Capability.DYNAMIC_CODE}),
        status=status,
    )

    assert registry.find_for_action("protected_action") is None
