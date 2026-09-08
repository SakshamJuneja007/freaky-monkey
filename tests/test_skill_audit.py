from agent_control.skills.audit import audit_skill
from agent_control.skills.base import Skill, SkillAction, SkillInfo
from agent_control.skills.manifest import SkillManifest
from agent_control.skills.security import Capability


class StringHelperSkill(Skill):
    _INFO = SkillInfo(
        name="string_helper",
        description="Uses string replacement.",
        actions=(SkillAction(kind="replace", description="replace text"),),
        manifest=SkillManifest(),
    )

    @property
    def info(self):
        return self._INFO

    def executor(self):
        return self

    def verifier(self):
        return self

    def adapt_action(self, action):
        return action

    def execute(self, action):
        kind = "a_b"
        return kind.replace("_", "-")

    def verify(self, action, result):
        return True


def test_audit_does_not_treat_string_replace_as_filesystem_write():
    report = audit_skill(StringHelperSkill())
    assert Capability.FILESYSTEM_WRITE not in report.observed_capabilities
    assert report.unexpected_capabilities == frozenset()
