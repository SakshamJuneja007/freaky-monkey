from __future__ import annotations

from agent_control.skills.browser.backend import BrowserSkillAdapter, BrowserSkillError


def _configured(monkeypatch):
    monkeypatch.setenv("DEIMOS_CHROME_USER_DATA", r"C:\Users\saksh\AppData\Local\Google\Chrome\User Data")
    monkeypatch.setenv("DEIMOS_CHROME_PROFILE", "Default")


def test_browser_session_is_discovered_started_and_observed(monkeypatch):
    _configured(monkeypatch)

    class CLI:
        def __init__(self):
            self.calls = []

        def run(self, args):
            args = list(args)
            self.calls.append(args)
            if args[:1] == ["browsers"]:
                return {"browsers": [{
                    "id": "chrome-default",
                    "browser": "Chrome",
                    "profile": "Default",
                    "user_data_dir": r"C:\Users\saksh\AppData\Local\Google\Chrome\User Data",
                }]}
            if args[:2] == ["session", "list"]:
                return {"sessions": []}
            if args[:2] == ["session", "start"]:
                return {"session_id": "session-1", "browser_id": "chrome-default"}
            if args[:2] == ["observe", "--session"]:
                return {"url": "https://www.google.com/", "text": 'textbox "Search" @e1'}
            raise AssertionError(args)

    cli = CLI()
    browser = BrowserSkillAdapter(cli)
    observation = browser.ensure_ready()

    assert browser.session_id == "session-1"
    assert observation.url == "https://www.google.com/"
    assert [c[:2] for c in cli.calls] == [
        ["browsers",], ["session", "list"], ["session", "start"], ["observe", "--session"],
    ]


def test_existing_browser_skill_session_is_reused(monkeypatch):
    _configured(monkeypatch)

    class CLI:
        def __init__(self):
            self.calls = []

        def run(self, args):
            args = list(args)
            self.calls.append(args)
            if args[:1] == ["browsers"]:
                return {"browsers": [{
                    "id": "chrome-default",
                    "browser": "Chrome",
                    "profile": "Default",
                    "user_data_dir": r"C:\Users\saksh\AppData\Local\Google\Chrome\User Data",
                }]}
            if args[:2] == ["session", "list"]:
                return {"sessions": [{
                    "session_id": "existing",
                    "browser_id": "chrome-default",
                    "browser": "Chrome",
                    "profile": "Default",
                    "user_data_dir": r"C:\Users\saksh\AppData\Local\Google\Chrome\User Data",
                }]}
            if args[:2] == ["observe", "--session"]:
                return {"url": "https://example.com/", "text": 'textbox "Search" @e1'}
            raise AssertionError(args)

    cli = CLI()
    browser = BrowserSkillAdapter(cli)
    browser.ensure_ready()
    browser.ensure_ready()

    assert browser.session_id == "existing"
    assert not any(c[:2] == ["session", "start"] for c in cli.calls)
    assert sum(c[:2] == ["session", "list"] for c in cli.calls) == 1
    assert sum(c[:2] == ["observe", "--session"] for c in cli.calls) == 2


def test_child_adapters_share_session_identity_but_not_observation_refs(monkeypatch):
    _configured(monkeypatch)

    class CLI:
        def __init__(self):
            self.calls = []

        def run(self, args):
            args = list(args)
            self.calls.append(args)
            if args[:1] == ["browsers"]:
                return {"browsers": [{"id": "chrome-default", "browser": "Chrome", "profile": "Default", "user_data_dir": r"C:\Users\saksh\AppData\Local\Google\Chrome\User Data"}]}
            if args[:2] == ["session", "list"]:
                return {"sessions": []}
            if args[:2] == ["session", "start"]:
                return {"session_id": "shared"}
            if args[:2] == ["observe", "--session"]:
                return {"url": "https://example.com/", "text": 'textbox "Search" @e1'}
            return {"ok": True}

    cli = CLI()
    base = BrowserSkillAdapter(cli)
    first = base.new_task_session()
    second = base.new_task_session()

    first.ensure_ready()
    second.ensure_ready()

    assert first.session_id == second.session_id == "shared"
    assert sum(c[:2] == ["session", "start"] for c in cli.calls) == 1
    assert first is not second
    assert first._last_observation is not second._last_observation


def test_profile_mismatch_fails_closed(monkeypatch):
    _configured(monkeypatch)

    class CLI:
        def run(self, args):
            args = list(args)
            if args[:1] == ["browsers"]:
                return {"browsers": [{"id": "wrong", "browser": "Chrome", "profile": "Profile 2", "user_data_dir": r"C:\Users\saksh\AppData\Local\Google\Chrome\User Data"}]}
            raise AssertionError(args)

    try:
        BrowserSkillAdapter(CLI()).ensure_session()
    except BrowserSkillError as exc:
        assert exc.code == "configured_profile_unavailable"
    else:
        raise AssertionError("wrong Chrome profile was silently accepted")


def test_default_windows_chrome_configuration_is_shared_with_browser_skill(monkeypatch):
    import agent_control.os_tools as os_tools
    monkeypatch.setattr(os_tools.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\saksh\AppData\Local")
    monkeypatch.delenv("DEIMOS_CHROME_USER_DATA", raising=False)
    monkeypatch.delenv("DEIMOS_CHROME_PROFILE", raising=False)
    monkeypatch.delenv("DEIMOS_CHROME_DEBUG_PORT", raising=False)
    monkeypatch.delenv("DEBUG_PORT", raising=False)

    from agent_control.skills.browser.backend import BrowserSkillAdapter
    config = BrowserSkillAdapter._configured_chrome_profile()
    assert config["user_data"].replace("/", "\\") == r"C:\Users\saksh\AppData\Local\Google\Chrome\User Data"
    assert config["profile"] == "Default"
    assert not config["user_data"].endswith(r"\Default")


def test_browser_target_resolution_reports_target_not_found():
    from agent_control.skills.browser.backend import BrowserSkillAdapter, BrowserSkillError

    class CLI:
        def run(self, args):
            args = list(args)
            if args[:2] == ["session", "start"]:
                return {"session_id": "s1"}
            if args[:2] == ["observe", "--session"]:
                return {"url": "chrome://newtab", "text": ""}
            if args[:1] == ["browsers"]:
                return {"browsers": [{"id": "c", "browser": "Chrome", "profile": "Default"}]}
            if args[:2] == ["session", "list"]:
                return {"sessions": []}
            raise AssertionError(args)

    browser = BrowserSkillAdapter(CLI())
    try:
        browser.resolve_target("address bar")
    except BrowserSkillError as exc:
        assert exc.code == "browser_target_not_found"
    else:
        raise AssertionError("target lookup unexpectedly succeeded")


def test_browser_skill_adaptation_requires_browser_readiness_before_target_resolution():
    from agent_control.skills.browser.backend import BrowserSkillAdapter, BrowserTarget, BrowserObservation
    from agent_control.skills.browser.skill import BrowserSkill
    from agent_control.types import Action

    class Backend(BrowserSkillAdapter):
        def __init__(self):
            super().__init__(cli=object(), session="s1")
            self.ready_calls = 0
            self.resolve_calls = 0
        def ensure_ready(self):
            self.ready_calls += 1
            self._generation = 1
            return BrowserObservation(1, "s1", None, "https://example.com", "", (), {})
        def resolve_target(self, query, **kwargs):
            self.resolve_calls += 1
            return BrowserTarget("@e1", role="textbox", name=query, generation=1)

    backend = Backend()
    skill = BrowserSkill(backend)
    adapted = skill.adapt_action(Action("browser_type", {"target_query": "search", "text": "hello"}))
    assert backend.ready_calls == 1
    assert backend.resolve_calls == 1
    assert adapted.params["target"] == "@e1"
