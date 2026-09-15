from __future__ import annotations

from types import SimpleNamespace

from agent_control.fast_interaction import FastInteractionTask, classify_fast
from agent_control.skills.browser import BrowserElement, BrowserObservation, BrowserSkillAdapter, BrowserTarget
from agent_control.skills.browser.actions import BrowserAction, BrowserActionKind
from agent_control.skills.browser.skill import BrowserSkill
from agent_control.types import Action, Check, Verdict


class FakeBrowser:
    def __init__(self, elements=(), url="https://example.com"):
        self._generation = 1
        self._last_observation = BrowserObservation(1, "s", 1, url, "", tuple(elements), {})
        self.observations = 0
        self.calls = []

    def observe(self):
        self.observations += 1
        self._generation += 1
        self._last_observation = BrowserObservation(
            self._generation, "s", 1, self._last_observation.url,
            self._last_observation.text, self._last_observation.elements, {}
        )
        return {}

    def resolve_target(self, query, **kwargs):
        obs = kwargs["observation"]
        for e in obs.elements:
            if query.casefold() in e.name.casefold():
                return BrowserTarget(e.ref, role=e.role, name=e.name, generation=obs.generation)
        raise ValueError("not found")


def test_plain_scroll_is_fast():
    r = classify_fast("scroll")
    assert r.matched and r.route.action_kind == "browser_scroll"
    assert r.route.params["amount"] > 0


def test_scroll_directions_use_existing_amount_contract():
    assert classify_fast("scroll down").route.params == {"amount": 600}
    assert classify_fast("scroll up").route.params == {"amount": -600}


def test_open_youtube_is_valid_browser_action():
    r = classify_fast("open youtube").route.resolve(None, None)
    action = BrowserSkillAdapter  # production adapter exists; schema is checked below
    assert r.kind == "browser_open_url"
    assert r.params["url"] == "https://www.youtube.com"


def test_click_target_is_browser_target_until_skill_boundary_then_string():
    e = BrowserElement("@e7", role="link", name="Example")
    b = FakeBrowser((e,))
    route = classify_fast("click Example").route
    core = route.resolve(b, b._last_observation)
    assert isinstance(core.params["target"], BrowserTarget)
    adapted = BrowserSkill(b).adapt_action(core)
    assert isinstance(adapted.params["target"], str)
    assert adapted.params["target"] == "@e7"


def test_indexed_target_is_observation_backed():
    elements = (
        BrowserElement("@e1", role="link", name="First video"),
        BrowserElement("@e2", role="link", name="Second video"),
    )
    b = FakeBrowser(elements)
    route = classify_fast("play the second video").route
    action = route.resolve(b, b._last_observation)
    assert isinstance(action.params["target"], BrowserTarget)
    assert action.params["target"].ref == "@e2"
    assert action.params["target"].generation == b._generation


def test_stale_browser_target_is_rejected_at_skill_boundary():
    b = FakeBrowser((BrowserElement("@e1", role="link", name="Example"),))
    target = BrowserTarget("@e1", generation=b._generation)
    action = Action("browser_click", {"target": target})
    b.observe()
    try:
        BrowserSkill(b).adapt_action(action)
    except Exception as exc:
        assert "stale" in str(exc).lower()
    else:
        raise AssertionError("stale target was accepted")


def test_fast_verification_uses_existing_browser_verifier(monkeypatch):
    b = FakeBrowser(url="https://www.youtube.com")
    route = classify_fast("open youtube").route
    task = FastInteractionTask("open youtube", b, route)
    task.action = route.resolve(b, b._last_observation)
    seen = {}
    class V:
        def verify(self, action, result):
            seen["action"] = action
            return SimpleNamespace(ok=True, status="PASS", detail="verified")
    monkeypatch.setattr("agent_control.skills.browser.skill.BrowserSkill.verifier", lambda self: V())
    result = task.verify_final(None)
    assert result.verdict is Verdict.PASS
    assert isinstance(seen["action"], BrowserAction)


def test_complex_request_falls_through():
    assert not classify_fast("find a tutorial, compare three videos, choose the best one and summarize it").matched


class _ScrollCLI:
    def __init__(self):
        self.calls = []

    def run(self, arguments):
        args = list(arguments)
        self.calls.append(args)
        if args[:2] == ["session", "start"]:
            return {"session_id": "s1", "ok": True}
        return {"ok": True}


def test_scroll_down_uses_supported_press_primitive():
    cli = _ScrollCLI()
    browser = BrowserSkillAdapter(cli)
    browser.scroll(600)
    assert cli.calls[-1][:2] in (["wheel", "--session"], ["press", "--session"])
    assert ("600" in cli.calls[-1]) or ("PageDown" in cli.calls[-1])


def test_scroll_up_uses_supported_press_primitive():
    cli = _ScrollCLI()
    browser = BrowserSkillAdapter(cli)
    browser.scroll(-600)
    assert cli.calls[-1][:2] in (["wheel", "--session"], ["press", "--session"])
    assert ("-600" in cli.calls[-1]) or ("PageUp" in cli.calls[-1])
