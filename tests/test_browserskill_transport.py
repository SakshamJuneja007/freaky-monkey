from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_control.skills.browser.backend import BrowserSkillAdapter, BrowserSkillCLI, BrowserSkillProtocolError, BrowserSkillUnavailable


class FakeCLI:
    def __init__(self):
        self.calls = []

    def run(self, args, *, session=None, timeout_s=None):
        self.calls.append((list(args), session))
        if list(args)[:2] == ["session", "start"]:
            return {"session_id": "ABCD"}
        if list(args)[:2] == ["session", "stop"]:
            return {"ok": True}
        if list(args)[:1] in (["snapshot"], ["observe"]):
            return {"data": {"title": "Example", "text": 'button "Compose" @e4', "url": "https://example.com/"}}
        return {"ok": True}


def test_adapter_uses_bsk_session_and_semantic_commands():
    cli = FakeCLI()
    browser = BrowserSkillAdapter(cli)
    browser.open_url("https://example.com")
    browser.click("@e4")
    browser.type_text("@e5", "hello")
    browser.scroll(500)
    assert any(call[0][:2] == ["session", "start"] for call in cli.calls)
    calls = [x[0] for x in cli.calls]
    assert any(c[:2] == ["navigate", "--session"] and "https://example.com" in c for c in calls)
    assert any(c[:2] == ["click", "--session"] and "@e4" in c for c in calls)
    assert any(c[:2] == ["fill", "--session"] and "--value" in c and "hello" in c for c in calls)
    assert any(c[:2] == ["wheel", "--session"] and "500" in c for c in calls)


def test_adapter_never_constructs_chrome_profile_paths():
    source = Path("agent_control/skills/browser/backend.py").read_text(encoding="utf-8")
    assert "--user-data-dir" not in source
    assert "connect_over_cdp" not in source
    assert "playwright" not in source.lower()


def test_find_ref_uses_observation_text():
    cli = FakeCLI()
    browser = BrowserSkillAdapter(cli)
    assert browser.find_ref(("Compose",)) == "@e4"


def test_cli_rejects_malformed_json(monkeypatch):
    class Completed:
        returncode = 0
        stdout = "not json"
        stderr = ""
    monkeypatch.setattr("subprocess.run", lambda *a, **k: Completed())
    monkeypatch.setattr("shutil.which", lambda _: "bsk")
    with pytest.raises(BrowserSkillProtocolError):
        BrowserSkillCLI().run(["status"])


def test_scroll_falls_back_to_supported_keyboard_primitive_when_wheel_is_unavailable():
    from agent_control.skills.browser.backend import BrowserSkillError

    class OldCLI(FakeCLI):
        def run(self, args, *, session=None, timeout_s=None):
            args = list(args)
            self.calls.append((args, session))
            if args[:2] == ["session", "start"]:
                return {"session_id": "s1"}
            if args[:1] == ["wheel"]:
                raise BrowserSkillError("BrowserSkill command failed (exit code 2): error: unrecognized subcommand 'wheel'")
            return {"ok": True}

    cli = OldCLI()
    browser = BrowserSkillAdapter(cli)
    result = browser.scroll(600)
    assert result["scroll_fallback"] == "keyboard"
    calls = [call[0] for call in cli.calls]
    assert any(c[:2] == ["wheel", "--session"] for c in calls)
    assert any(c[:2] == ["press", "--session"] and "PageDown" in c for c in calls)


def test_session_start_prefers_default_chrome_profile(monkeypatch):
    from agent_control.skills.browser.backend import BrowserSkillAdapter

    class CLI:
        def __init__(self):
            self.calls = []

        def run(self, args):
            self.calls.append(list(args))
            if args[:1] == ["browsers"]:
                return {
                    "browsers": [
                        {"id": "work", "browser": "Chrome", "profile": "Profile 3"},
                        {"id": "default", "browser": "Chrome", "profile": "Default"},
                        {"id": "edge", "browser": "Microsoft Edge", "profile": "Default"},
                    ]
                }
            if args[:2] == ["session", "start"]:
                return {"session_id": "abcd"}
            raise AssertionError(args)

    cli = CLI()
    browser = BrowserSkillAdapter(cli)
    browser.session_start()

    assert [call for call in cli.calls if call[:2] == ["session", "start"]] == [["session", "start", "--browser", "default", "--no-focus"]]


def test_phrase_accepted_never_exposes_internal_task_id():
    from agent_control.response import phrase_accepted

    text = phrase_accepted("fast-1234", "open chrome", debug=True)
    assert "fast-1234" not in text
    assert "Starting fast" not in text


def test_session_start_uses_configured_chrome_profile(monkeypatch):
    from agent_control.skills.browser.backend import BrowserSkillAdapter

    class CLI:
        def __init__(self):
            self.calls = []
        def run(self, args):
            self.calls.append(list(args))
            if args[:1] == ["browsers"]:
                return {"browsers": [
                    {"id": "work", "browser": "Chrome", "profile": "Profile 3", "user_data_dir": r"C:\Users\me\AppData\Local\Google\Chrome\User Data"},
                    {"id": "target", "browser": "Chrome", "profile": "Profile 5", "user_data_dir": r"C:\Users\me\AppData\Local\Google\Chrome\User Data"},
                ]}
            if args[:2] == ["session", "start"]:
                return {"session_id": "configured"}
            raise AssertionError(args)

    monkeypatch.setenv("DEIMOS_CHROME_USER_DATA", r"C:\Users\me\AppData\Local\Google\Chrome\User Data")
    monkeypatch.setenv("DEIMOS_CHROME_PROFILE", "Profile 5")
    cli = CLI()
    BrowserSkillAdapter(cli).session_start()
    assert [call for call in cli.calls if call[:2] == ["session", "start"]] == [["session", "start", "--browser", "target", "--no-focus"]]


def test_session_start_rejects_unavailable_configured_profile(monkeypatch):
    from agent_control.skills.browser.backend import BrowserSkillAdapter, BrowserSkillError

    class CLI:
        def run(self, args):
            if args[:1] == ["browsers"]:
                return {"browsers": [{"id": "other", "browser": "Chrome", "profile": "Profile 1"}]}
            raise AssertionError(args)

    monkeypatch.setenv("DEIMOS_CHROME_PROFILE", "Profile 9")
    try:
        BrowserSkillAdapter(CLI()).session_start()
    except BrowserSkillError as exc:
        assert exc.code == "configured_profile_unavailable"
    else:
        raise AssertionError("configured unavailable Chrome profile was silently accepted")


def test_existing_browser_session_is_reused_without_starting_another(monkeypatch):
    from agent_control.skills.browser.backend import BrowserSkillAdapter

    class CLI:
        def __init__(self):
            self.calls = []
        def run(self, args):
            self.calls.append(list(args))
            if args[:1] == ["observe"]:
                return {"data": {"text": "textbox \"Address\" @e1", "url": "https://example.com"}}
            return {"ok": True}

    cli = CLI()
    browser = BrowserSkillAdapter(cli, session="existing")
    browser.observe()
    assert not any(call[:2] == ["session", "start"] for call in cli.calls)
    assert cli.calls[0][:2] == ["observe", "--session"]
