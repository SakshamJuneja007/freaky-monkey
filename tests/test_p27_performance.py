from __future__ import annotations

import time

from agent_control.planner.openai_compat import LLMClient
from agent_control.trace import Trace
from agent_control.types import Observation, Source


class _Response:
    status_code = 200
    text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        }


class _HTTPClient:
    instances = []

    def __init__(self):
        self.calls = 0
        self.closed = False
        type(self).instances.append(self)

    def post(self, *args, **kwargs):
        self.calls += 1
        return _Response()

    def close(self):
        self.closed = True


def test_llm_client_reuses_connection_pool(monkeypatch):
    _HTTPClient.instances.clear()
    monkeypatch.setattr("agent_control.planner.openai_compat.httpx.Client", _HTTPClient)
    client = LLMClient(api_key="x", base_url="http://example.test", model="test")

    client.chat([], json_mode=False)
    client.chat([], json_mode=False)

    assert len(_HTTPClient.instances) == 1
    assert _HTTPClient.instances[0].calls == 2
    client.close()
    assert _HTTPClient.instances[0].closed is True


def test_trace_exposes_p27_phase_and_fresh_observation_metrics(tmp_path):
    trace = Trace(task_id="p27", condition="test", trace_dir=tmp_path)
    try:
        obs = Observation(Source.PROCESS, "test", {"ok": True}, ok=True)
        trace.observation(obs, purpose="profile")
        trace.phase("planner", 0.012)
        trace.phase("verification", 0.003)
        summary = trace.summary()
    finally:
        trace.close()

    assert summary["counters"]["fresh_observations"] == 1
    assert summary["counters"]["planner_ms"] >= 12.0
    assert summary["counters"]["verification_ms"] >= 3.0


def test_trace_phase_timing_is_additive(tmp_path):
    trace = Trace(task_id="p27-add", condition="test", trace_dir=tmp_path)
    try:
        trace.phase("execution", 0.001)
        trace.phase("execution", 0.002)
        assert 2.9 <= trace.counters["execution_ms"] <= 3.1
    finally:
        trace.close()


def test_trace_distinguishes_mock_planner_calls_from_llm_calls(tmp_path):
    trace = Trace(task_id="p27-planner", condition="test", trace_dir=tmp_path)
    try:
        trace.planner_call(planner="mock", kind="plan")
        trace.planner_call(planner="llm:test", kind="plan")
        assert trace.counters["planner_calls"] == 2
        assert trace.counters["llm_calls"] == 1
    finally:
        trace.close()


def test_text_target_cache_reuses_identity_but_not_text(monkeypatch):
    from agent_control import verifiers

    verifiers._TEXT_TARGET_CACHE.clear()
    checks = []
    monkeypatch.setattr("agent_control.verifiers.observe.validate_window_target", lambda handle, **kwargs: True)
    verifiers._cache_text_window("notepad", {"handle": 42, "pid": 7, "title": "Untitled - Notepad", "focused": True})
    target = verifiers._cached_text_window("notepad", "notepad", None)
    assert target == {"handle": 42, "pid": 7, "title": "Untitled - Notepad", "focused": True}
    assert "text" not in target


def test_chrome_launch_uses_existing_configuration(monkeypatch, tmp_path):
    from agent_control.os_tools import launch_app
    from agent_control.policy import Policy
    from agent_control.types import Action
    import agent_control.os_tools as os_tools

    monkeypatch.setenv("DEIMOS_CHROME_EXECUTABLE", str(tmp_path / "chrome.exe"))
    monkeypatch.setenv("DEIMOS_CHROME_USER_DATA", str(tmp_path / "User Data"))
    monkeypatch.setenv("DEIMOS_CHROME_PROFILE", "Profile 7")
    monkeypatch.setenv("DEBUG_PORT", "9222")
    exe = tmp_path / "chrome.exe"
    exe.write_text("stub")
    calls = []
    class P:
        pid = 123
    monkeypatch.setattr(os_tools.psutil, "process_iter", lambda *a, **k: [])
    monkeypatch.setattr(os_tools.subprocess, "Popen", lambda argv, **kwargs: calls.append(argv) or P())
    result = launch_app(Policy(workspace=tmp_path, refuse_if_elevated=False), Action("launch_app", {"app": "chrome", "settle_s": 0}))
    assert result.ok
    assert f"--user-data-dir={tmp_path / 'User Data'}" in calls[0]
    assert "--profile-directory=Profile 7" in calls[0]
    assert "--remote-debugging-port=9222" in calls[0]


def test_browser_semantic_target_is_resolved_at_skill_boundary():
    from agent_control.skills.browser.skill import BrowserSkill
    from agent_control.skills.browser.backend import BrowserSkillAdapter, BrowserTarget

    class Backend(BrowserSkillAdapter):
        def resolve_target(self, query, **kwargs):
            assert query == "search box"
            return BrowserTarget("@e7", role="textbox", name="search box", generation=0)

    skill = BrowserSkill(Backend())
    action = skill.adapt_action(__import__("agent_control.types", fromlist=["Action"]).Action(
        "browser_type",
        {"target_query": "search box", "target_semantic": {"role": "textbox"}, "text": "hello"},
    ))
    assert action.params["target"] == "@e7"


def test_chrome_application_ready_does_not_claim_browser_ready(monkeypatch, tmp_path):
    from agent_control.os_tools import launch_app
    from agent_control.policy import Policy
    from agent_control.types import Action
    import agent_control.os_tools as os_tools

    exe = tmp_path / "chrome.exe"
    exe.write_text("stub")
    monkeypatch.setenv("DEIMOS_CHROME_EXECUTABLE", str(exe))
    monkeypatch.setenv("DEIMOS_CHROME_USER_DATA", str(tmp_path / "User Data"))
    monkeypatch.setenv("DEIMOS_CHROME_PROFILE", "Default")
    monkeypatch.setattr(os_tools.psutil, "process_iter", lambda *a, **k: [])
    monkeypatch.setattr(os_tools.subprocess, "Popen", lambda *a, **k: type("P", (), {"pid": 123})())

    result = launch_app(Policy(workspace=tmp_path, refuse_if_elevated=False), Action("launch_app", {"app": "chrome", "settle_s": 0}))
    assert result.ok
    assert result.detail["browser_ready"] is False


def test_browser_search_gets_independent_url_postcondition():
    from agent_control.skills.browser.skill import BrowserSkill
    from agent_control.skills.browser.backend import BrowserSkillAdapter
    from agent_control.types import Action

    skill = BrowserSkill(BrowserSkillAdapter(type("CLI", (), {"run": lambda self, args: {"ok": True, "session_id": "s"}})()))
    action = skill.adapt_action(Action("browser_search", {"query": "deimos ai agent"}))
    assert action.params["expected_url_contains"].startswith("q=")
