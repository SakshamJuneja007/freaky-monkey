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
    assert cli.calls[0][0][:3] == ["session", "start", "--no-focus"]
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
