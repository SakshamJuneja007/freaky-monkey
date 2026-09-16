from __future__ import annotations

from agent_control.skills.browser.actions import BrowserAction, BrowserActionKind
from agent_control.skills.browser.backend import BrowserObservation
from agent_control.skills.browser.browser_verifiers import BrowserVerifier
from agent_control.skills.browser.text_observer import BrowserTextObserver


class FakeBrowser:
    def __init__(self, observations):
        self.observations = list(observations)
        self._generation = 0
        self._last_observation = None
        self.observe_calls = 0

    def observe(self):
        self.observe_calls += 1
        value = self.observations.pop(0) if len(self.observations) > 1 else self.observations[0]
        self._generation += 1
        self._last_observation = BrowserObservation(
            generation=self._generation,
            session="s1",
            tab_id=1,
            url=value.get("url", ""),
            text=value.get("text", ""),
            elements=tuple(),
            raw=value,
        )
        return value


def test_browser_text_observer_extracts_readable_text_url_and_title():
    browser = FakeBrowser([{
        "url": "https://example.com/results?q=deimos",
        "title": "Deimos results",
        "text": "deimos agent",
        "elements": [{"role": "link", "name": "Deimos AI agent"}],
    }])
    observation = BrowserTextObserver(browser).observe()
    assert observation.ok
    assert observation.text == "deimos agent"
    assert observation.metadata["url"] == "https://example.com/results?q=deimos"
    assert observation.metadata["title"] == "Deimos results"
    assert observation.metadata["text_available"] is True


def test_browser_verifier_expected_text_uses_fresh_observation():
    browser = FakeBrowser([{"url": "https://example.com", "title": "Results", "text": "deimos agent"}])
    verifier = BrowserVerifier(browser)
    action = BrowserAction(BrowserActionKind.SEARCH, {"query": "deimos agent", "expected_text": "deimos agent"})
    result = verifier.verify(action, type("ExecutorResult", (), {"ok": False})())
    assert result.ok is True
    assert result.status == "PASS"
    assert browser.observe_calls == 1


def test_browser_verifier_expected_text_fails_when_absent():
    browser = FakeBrowser([{"url": "https://example.com", "title": "Results", "text": "other result"}])
    verifier = BrowserVerifier(browser)
    action = BrowserAction(BrowserActionKind.SEARCH, {"query": "deimos", "expected_text": "deimos"})
    result = verifier.verify(action, type("ExecutorResult", (), {"ok": True})())
    assert result.ok is False
    assert result.status == "FAIL"


def test_browser_verifier_returns_unknown_when_browser_observation_unavailable():
    class BrokenBrowser:
        def observe(self):
            raise RuntimeError("BrowserSkill unavailable")

    verifier = BrowserVerifier(BrokenBrowser())
    action = BrowserAction(BrowserActionKind.SEARCH, {"query": "deimos", "expected_text": "deimos"})
    result = verifier.verify(action, type("ExecutorResult", (), {"ok": True})())
    assert result.ok is False
    assert result.status == "UNKNOWN"


def test_executor_success_is_not_independent_browser_evidence():
    browser = FakeBrowser([{"url": "https://example.com", "title": "Results", "text": "different"}])
    verifier = BrowserVerifier(browser)
    action = BrowserAction(BrowserActionKind.SEARCH, {"query": "deimos", "expected_text": "deimos"})
    result = verifier.verify(action, type("ExecutorResult", (), {"ok": True})())
    assert result.status == "FAIL"


def test_browser_search_can_supply_expected_text_at_skill_boundary():
    from agent_control.skills.browser.skill import BrowserSkill

    class Backend(FakeBrowser):
        def search(self, query):
            return {"ok": True}
        def open_url(self, url): return {"ok": True}
        def current_url(self): return ""
        def page_title(self): return ""
        def page_text(self): return ""
        def snapshot(self): return {}
        def click(self, target): return {}
        def type_text(self, target, text): return {}
        def press_key(self, key, target=None): return {}
        def scroll(self, amount): return {}
        def scroll_to(self, target): return {}
        def select(self, target, value): return {}
        def upload_file(self, target, file_path, mode=None): return {}
        def download(self, target, output_path, overwrite=False): return {}
        def wait(self, seconds): return {}
        def close_tab(self, tab_id=None): return {}
        def list_tabs(self, scope="all"): return {}
        def create_tab(self, url=None): return {}
        def select_tab(self, tab_id): return {}
        def borrow_tab(self, tab_id): return {}
        def return_tab(self, tab_id): return {}
        def play_song(self, query): return {}
        def apply_job(self, job_url, resume_path, answers=None, *, submit=True): return {}

    backend = Backend([{"url": "https://example.com", "text": ""}])
    action = BrowserSkill(backend).adapt_action(
        type("Action", (), {"kind": "browser_search", "params": {"query": "deimos agent"}, "rationale": ""})()
    )
    assert action.params["expected_text"] == "deimos agent"


def test_navigation_changes_produce_fresh_browser_text():
    browser = FakeBrowser([
        {"url": "https://example.com/old", "title": "Old", "text": "old page"},
        {"url": "https://example.com/new", "title": "New", "text": "new page"},
    ])
    observer = BrowserTextObserver(browser)
    first = observer.observe()
    second = observer.observe()
    assert first.text == "old page"
    assert second.text == "new page"
    assert first.observed_at <= second.observed_at
    assert first.metadata["generation"] != second.metadata["generation"]
