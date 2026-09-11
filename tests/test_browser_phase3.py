from __future__ import annotations

from agent_control.types import Action
from agent_control.skills.browser.actions import BrowserActionKind
from agent_control.skills.browser.backend import canonical_url
from agent_control.skills.browser.skill import BrowserSkill
from agent_control.skills.browser.browser_verifiers import BrowserVerifier


class Backend:
    def __init__(self) -> None:
        self.url = "https://www.youtube.com/"
        self.title = "YouTube"
        self.text = "YouTube home"

    def open_url(self, url):
        self.url = url
        return {"url": url}

    def search(self, query):
        return {"query": query}

    def click(self, target):
        return {"target": target}

    def type_text(self, text):
        return {"text": text}

    def press_key(self, key):
        return {"key": key}

    def scroll(self, amount):
        return {"amount": amount}

    def select(self, target, value):
        return {"target": target, "value": value}

    def wait(self, seconds):
        return {"seconds": seconds}

    def close_tab(self):
        return {"closed": True}

    def current_url(self):
        return self.url

    def page_title(self):
        return self.title

    def page_contains_text(self, text):
        return text.casefold() in self.text.casefold()


def test_canonical_url_treats_www_and_root_slash_as_same():
    assert canonical_url("https://youtube.com") == canonical_url(
        "https://www.youtube.com/"
    )


def test_browser_skill_claims_core_open_url():
    skill = BrowserSkill(Backend())
    assert skill.supports("open_url") is True


def test_browser_skill_adapts_core_open_url_to_browser_action_and_verification():
    skill = BrowserSkill(Backend())
    action = skill.adapt_action(
        Action(kind="open_url", params={"url": "https://youtube.com"})
    )

    assert action.kind is BrowserActionKind.OPEN_URL
    assert action.params["url"] == "https://youtube.com"
    assert action.params["expected_url"] == "https://youtube.com"


def test_browser_verifier_accepts_normal_youtube_redirect():
    backend = Backend()
    verifier = BrowserVerifier(backend)
    action = BrowserSkill(backend).adapt_action(
        Action(kind="open_url", params={"url": "https://youtube.com"})
    )

    result = verifier.verify(
        action,
        type("Result", (), {"ok": True, "detail": ""})(),
    )

    assert result.ok is True
    assert "youtube.com" in result.detail
