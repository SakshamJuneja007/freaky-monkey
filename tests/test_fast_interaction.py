from __future__ import annotations

from agent_control.fast_interaction import FastInteractionTask, classify_fast
from agent_control.policy import Policy
from agent_control.runner import RunConfig, run_task
from agent_control.skills.browser.actions import BrowserActionKind
from agent_control.skills.browser.backend import BrowserSkillAdapter
from agent_control.skills.browser.skill import BrowserSkill
from agent_control.skills.registry import SkillRegistry
from agent_control.types import Verdict


class FakeCLI:
    def __init__(self, observation):
        self.observations = list(observation) if isinstance(observation, list) else [observation]
        self.calls = []
        self.timeout = 2.0

    def run(self, args, *, session=None, timeout_s=None):
        args = list(args)
        self.calls.append(args)
        if args[:2] == ["session", "start"]:
            return {"session_id": "s1"}
        if args[:1] == ["observe"]:
            return self.observations.pop(0) if len(self.observations) > 1 else self.observations[0]
        if args[:2] == ["tab", "list"]:
            return {"tabs": [{"tab_id": 1, "active": True, "url": "https://example.com", "title": "Example"}]}
        return {"ok": True}


def test_simple_scroll_is_fast():
    result = classify_fast("scroll down")
    assert result.matched
    assert result.route.action_kind == "browser_scroll"
    assert result.route.params["amount"] == 600


def test_back_forward_refresh_are_fast():
    assert classify_fast("back").route.action_kind == "browser_go_back"
    assert classify_fast("forward").route.action_kind == "browser_go_forward"
    assert classify_fast("refresh").route.action_kind == "browser_refresh"


def test_keyboard_and_typing_are_fast():
    assert classify_fast("press Enter").route.action_kind == "browser_press_key"
    assert classify_fast("type hello world").route.action_kind == "browser_type"


def test_media_controls_are_fast_but_require_semantic_media_buttons():
    for request in ("play", "pause", "mute", "unmute", "volume up", "volume down"):
        result = classify_fast(request)
        assert result.matched
        assert result.route.action_kind == "browser_click"
        assert result.route.target_query == request


def test_semantic_and_indexed_clicks_are_fast():
    assert classify_fast("click Settings").route.action_kind == "browser_click"
    indexed = classify_fast("play the third video")
    assert indexed.matched
    assert indexed.route.target_index == 3


def test_complex_and_ambiguous_requests_fall_through():
    assert not classify_fast("find a tutorial, compare three videos, choose the best one and summarize it").matched
    assert not classify_fast("do whatever is needed to fix this").matched
    assert not classify_fast("send papa a message saying hello").matched
    assert not classify_fast("open Gmail and email this person").matched
    assert not classify_fast("organize my Downloads folder").matched


def test_indexed_target_uses_current_observation_generation():
    cli = FakeCLI({"text": '@e1 link "one"\n@e2 link "two"\n@e3 link "three"'})
    browser = BrowserSkillAdapter(cli)
    browser.observe()
    obs = browser._last_observation
    route = classify_fast("play the third video").route
    action = route.resolve(browser, obs)
    target = action.params["target"]
    assert target.ref == "@e3"
    assert target.generation == obs.generation


def test_close_tab_uses_observed_active_tab():
    cli = FakeCLI({"text": ""})
    browser = BrowserSkillAdapter(cli)
    browser.observe()
    obs = browser._last_observation
    route = classify_fast("close tab").route
    action = route.resolve(browser, obs)
    assert action.params["tab_id"] == 1


def test_fast_path_runs_policy_and_skips_planner(tmp_path):
    cli = FakeCLI({"text": "@e1 button \"Search\""})
    browser = BrowserSkillAdapter(cli)
    task = FastInteractionTask("scroll down", browser, classify_fast("scroll down").route)
    browser.observe()
    task._first_observation = browser._last_observation
    task.action = task.route.resolve(browser, task._first_observation)

    policy = Policy(workspace=tmp_path / "ws", refuse_if_elevated=False)
    registry = SkillRegistry()
    registry.register(BrowserSkill(browser))

    outcome = run_task(
        task,
        None,
        policy,
        RunConfig(max_steps=1, fresh_precondition=True, recovery_enabled=False),
        skills=registry,
        teardown=False,
        direct_action_factory=lambda loop: task.action,
        fast_latency_seconds=0.0001,
    )

    counters = outcome.trace["counters"]
    assert counters.get("llm_calls", 0) == 0
    assert counters.get("actions.browser_scroll", 0) == 1
    assert counters.get("policy.allow", 0) == 1
    assert counters.get("fast_router_classifications", 0) == 1


def test_fast_target_is_not_reused_after_observation_generation_changes():
    cli = FakeCLI([
        {"text": '@e1 button "Go"'},
        {"text": '@e9 button "Go"'},
    ])
    browser = BrowserSkillAdapter(cli)
    first = browser.observe()
    old = browser.resolve_target("Go", observation=browser._last_observation)
    browser.observe()
    assert old.generation != browser._last_observation.generation
    try:
        browser.click(old)
    except Exception as exc:
        assert "stale" in str(exc).lower()
    else:
        raise AssertionError("stale semantic reference was reused")


def test_typing_resolves_one_visible_textbox_semantically():
    cli = FakeCLI({"text": '@e4 textbox "Search" =""'})
    browser = BrowserSkillAdapter(cli)
    browser.observe()
    route = classify_fast("type hello").route
    action = route.resolve(browser, browser._last_observation)
    assert action.kind == "browser_type"
    assert action.params["target"].ref == "@e4"
    assert action.params["text"] == "hello"


def test_ambiguous_semantic_click_is_rejected_without_execution():
    cli = FakeCLI({"text": '@e1 button "Settings"\n@e2 button "Settings"'})
    browser = BrowserSkillAdapter(cli)
    browser.observe()
    route = classify_fast("click Settings").route
    try:
        route.resolve(browser, browser._last_observation)
    except Exception as exc:
        assert "ambiguous" in str(exc).lower()
    else:
        raise AssertionError("ambiguous semantic target was accepted")


def test_high_impact_messaging_is_not_in_fast_action_set():
    for request in ("send papa a message saying hello", "open Gmail and email this person"):
        result = classify_fast(request)
        assert not result.matched


def test_fast_recovery_re_resolves_indexed_target_from_fresh_generation():
    cli = FakeCLI([
        {"text": '@e1 link "one"\n@e2 link "two"'},
        {"text": '@e9 link "one"\n@e8 link "two"'},
    ])
    browser = BrowserSkillAdapter(cli)
    task = FastInteractionTask("play the second video", browser, classify_fast("play the second video").route)
    browser.observe()
    first = browser._last_observation
    task._first_observation = first
    task.action = task.route.resolve(browser, first)
    task._first_observation = None
    task.observe(None)
    assert task.action.params["target"].ref == "@e8"
    assert task.action.params["target"].generation == browser._last_observation.generation


def test_youtube_indexed_playback_excludes_shorts():
    from agent_control.skills.browser.backend import BrowserElement
    browser = BrowserSkillAdapter(FakeCLI({
        "url": "https://www.youtube.com/results?search_query=test",
        "text": '@e1 link "Short result"\n@e2 link "Normal video"',
        "elements": [
            {"ref": "@e1", "role": "link", "name": "Short result", "raw": {"href": "https://www.youtube.com/shorts/x"}},
            {"ref": "@e2", "role": "link", "name": "Normal video", "raw": {"href": "https://www.youtube.com/watch?v=y"}},
        ],
    }))
    route = classify_fast("play the first video").route
    task = FastInteractionTask("play the first video", browser, route)
    browser.observe()
    action = task.resolve_current_action()
    assert action.params["target"].ref == "@e2"
    assert action.params["verify_youtube_playback"] is True
