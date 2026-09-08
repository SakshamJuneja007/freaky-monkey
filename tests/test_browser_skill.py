from __future__ import annotations

from agent_control.skills.browser.actions import (
    BROWSER_ACTION_KINDS,
)
from agent_control.skills.browser.skill import BrowserSkill
from agent_control.skills.registry import SkillRegistry


class FakeBrowserBackend:
    """Deterministic browser backend for skill tests.

    No real browser is started.
    """

    def __init__(self) -> None:
        self.current_url = "about:blank"
        self.current_title = ""
        self.page_text = ""

    def open_url(self, url: str):
        self.current_url = url
        return {
            "ok": True,
            "url": url,
        }

    def get_current_url(self) -> str:
        return self.current_url

    def get_title(self) -> str:
        return self.current_title

    def get_page_text(self) -> str:
        return self.page_text


def test_browser_skill_has_correct_name():
    backend = FakeBrowserBackend()
    skill = BrowserSkill(backend)

    assert skill.name == "browser"


def test_browser_skill_has_description():
    backend = FakeBrowserBackend()
    skill = BrowserSkill(backend)

    assert isinstance(skill.description, str)
    assert skill.description


def test_browser_skill_exposes_browser_actions():
    backend = FakeBrowserBackend()
    skill = BrowserSkill(backend)

    assert skill.action_kinds == tuple(
        sorted(BROWSER_ACTION_KINDS)
    )


def test_browser_skill_supports_every_declared_action():
    backend = FakeBrowserBackend()
    skill = BrowserSkill(backend)

    for kind in BROWSER_ACTION_KINDS:
        assert skill.supports(kind) is True


def test_browser_skill_rejects_unknown_action_kind():
    backend = FakeBrowserBackend()
    skill = BrowserSkill(backend)

    assert skill.supports("definitely_not_a_browser_action") is False


def test_browser_skill_provides_executor():
    backend = FakeBrowserBackend()
    skill = BrowserSkill(backend)

    assert skill.executor() is not None


def test_browser_skill_provides_verifier():
    backend = FakeBrowserBackend()
    skill = BrowserSkill(backend)

    assert skill.verifier() is not None


def test_browser_skill_can_be_registered():
    backend = FakeBrowserBackend()
    skill = BrowserSkill(backend)

    registry = SkillRegistry()
    registry.register(skill)

    assert registry.get("browser") is skill


def test_registry_finds_browser_for_browser_actions():
    backend = FakeBrowserBackend()
    skill = BrowserSkill(backend)

    registry = SkillRegistry()
    registry.register(skill)

    for kind in BROWSER_ACTION_KINDS:
        assert registry.find_for_action(kind) is skill


def test_browser_skill_describe_matches_metadata():
    backend = FakeBrowserBackend()
    skill = BrowserSkill(backend)

    description = skill.describe()

    assert description["name"] == "browser"
    assert description["description"] == skill.description
    assert description["action_kinds"] == list(
        skill.action_kinds
    )