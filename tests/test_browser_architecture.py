from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from agent_control.skills.browser.backend import (
    BrowserElement,
    BrowserSkillAdapter,
    BrowserSkillCLI,
    BrowserSkillError,
    BrowserTarget,
)
from agent_control.skills.browser.actions import BrowserAction, BrowserActionKind
from agent_control.skills.browser.browser_verifiers import BrowserVerifier


class FakeCLI:
    def __init__(self, observations=None):
        self.calls = []
        self.observations = list(observations or [{"text": '@e1 button "Search"'}])
        self.timeout = 2.0

    def run(self, args, *, session=None, timeout_s=None):
        args = list(args)
        self.calls.append((args, session))
        if args[:2] == ["session", "start"]:
            return {"session_id": "s1"}
        if args[:2] == ["session", "stop"]:
            return {"ok": True}
        if args[:1] == ["observe"]:
            value = self.observations.pop(0) if len(self.observations) > 1 else self.observations[0]
            return value
        if args[:1] == ["tab"]:
            return {"tabs": [{"tab_id": 1, "active": True, "url": "https://example.com/", "title": "Example"}]}
        return {"ok": True}


def test_observation_parser_supports_real_ref_first_and_legacy_ref_last():
    value = {
        "text": '@e1 button "Search"\n  @e2 textbox "Search field" ="hello"\nbutton "Compose" @e3'
    }
    elements = BrowserSkillAdapter._extract_elements(value)
    assert [(e.ref, e.role, e.name, e.value) for e in elements] == [
        ("@e1", "button", "Search", ""),
        ("@e2", "textbox", "Search field", "hello"),
        ("@e3", "button", "Compose", ""),
    ]


def test_resolver_is_role_aware_and_deterministic():
    cli = FakeCLI([{"text": '@e1 link "Settings"\n@e2 button "Settings"'}])
    browser = BrowserSkillAdapter(cli)
    target = browser.resolve_target("Settings", preferred_roles=("button",))
    assert target.ref == "@e2"
    assert target.generation == browser._generation


def test_resolver_can_reject_close_ambiguity():
    cli = FakeCLI([{"text": '@e1 link "Download"\n@e2 link "Download"'}])
    browser = BrowserSkillAdapter(cli)
    with pytest.raises(BrowserSkillError, match="Ambiguous"):
        browser.resolve_target("Download", preferred_roles=("link",), reject_ambiguous=True)


def test_browser_target_from_old_observation_is_rejected_after_mutation():
    cli = FakeCLI([{"text": '@e1 button "Go"'}, {"text": '@e9 button "Other"'}])
    browser = BrowserSkillAdapter(cli)
    browser.observe()
    target = browser.resolve_target("Go", observation=browser._last_observation)
    browser.click(target)
    with pytest.raises(BrowserSkillError, match="stale"):
        browser.click(target)


def test_wait_until_observation_contains_retries_until_target_appears():
    cli = FakeCLI([
        {"text": ""},
        {"text": '@e9 link "Killshot - Official Video"'},
    ])
    browser = BrowserSkillAdapter(cli)
    target = browser.wait_until_observation_contains("Killshot", preferred_roles=("link",), timeout_s=1, poll_ms=1)
    assert target.ref == "@e9"
    assert len([c for c, _ in cli.calls if c[:1] == ["observe"]]) == 2


def test_find_ref_supports_alternative_labels():
    cli = FakeCLI([{"text": '@e1 button "Compose"'}])
    browser = BrowserSkillAdapter(cli)
    assert browser.find_ref(("New message", "Compose"), role="button") == "@e1"


def test_utf8_cli_output_is_decoded_explicitly(monkeypatch):
    class Completed:
        returncode = 0
        stdout = json.dumps({"text": "Café — हिन्दी"}, ensure_ascii=False)
        stderr = "diagnostic π"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Completed())
    monkeypatch.setattr(shutil, "which", lambda _: "bsk")
    result = BrowserSkillCLI().run(["observe"])
    assert result["text"] == "Café — हिन्दी"
    assert result["_stderr"] == "diagnostic π"


def test_status_is_not_given_a_session():
    cli = FakeCLI()
    browser = BrowserSkillAdapter(cli)
    browser.status()
    assert cli.calls[-1][0] == ["status"]


def test_playback_verifier_does_not_use_executor_result_as_evidence():
    class Backend:
        def current_url(self):
            return "https://www.youtube.com/watch?v=x"
        def page_title(self):
            return "Killshot"
        def page_text(self):
            return "Pause"
        def list_tabs(self, scope="all"):
            return []
        def observe(self):
            return {"text": 'button "Pause"'}
        def wait_for_playback(self, *, timeout_s=5):
            return None

    verifier = BrowserVerifier(Backend())
    action = BrowserAction(BrowserActionKind.PLAY_SONG, {"query": "Killshot"})
    result = verifier.verify(action, type("R", (), {"ok": False})())
    assert result.ok is True
    assert result.status == "PASS"


def test_each_fresh_observation_invalidates_previous_refs():
    cli = FakeCLI([
        {"text": '@e1 button "Go"'},
        {"text": '@e7 button "Go"'},
    ])
    browser = BrowserSkillAdapter(cli)
    browser.observe()
    old = browser.resolve_target("Go", observation=browser._last_observation)
    browser.observe()
    with pytest.raises(BrowserSkillError, match="stale"):
        browser.click(old)


def test_play_song_uses_fresh_semantic_result_and_never_hard_codes_ref():
    class SongCLI(FakeCLI):
        def __init__(self):
            super().__init__([
                {"text": '@e57 link "Killshot - Official Video"\n@e58 link "Killshot lyrics"'},
                {"text": '@e12 button "Pause"'},
            ])
            self.watch = False
            self.observation_count = 0

        def run(self, args, *, session=None, timeout_s=None):
            args = list(args)
            self.calls.append((args, session))
            if args[:2] == ["session", "start"]:
                return {"session_id": "s1"}
            if args[:1] == ["navigate"]:
                return {"ok": True, "final_url": args[3] if len(args) > 3 else ""}
            if args[:1] == ["observe"]:
                self.observation_count += 1
                if self.watch:
                    return {"text": '@e12 button "Pause"'}
                if self.observation_count == 1:
                    return {"text": ""}
                return {"text": '@e57 link "Killshot - Official Video"\n@e58 link "Killshot lyrics"'}
            if args[:1] == ["click"]:
                self.watch = True
                return {"ok": True, "used_ref": "e57"}
            if args[:1] == ["tab"]:
                return {"tabs": [{"tab_id": 1, "active": True, "url": "https://www.youtube.com/watch?v=test" if self.watch else "https://www.youtube.com/results", "title": "Killshot"}]}
            if args[:1] == ["wait-ms"]:
                return {"waited_ms": 1}
            return {"ok": True}

    cli = SongCLI()
    browser = BrowserSkillAdapter(cli)
    result = browser.play_song("Killshot", timeout_s=2)
    assert result["ok"] is True
    assert result["selected_ref"] == "@e58"
    assert result["playback"] == "verified"
    click_calls = [c for c, _ in cli.calls if c[:1] == ["click"]]
    assert click_calls and "@e58" in click_calls[0]


def test_tab_list_uses_session_scoped_tab_command_not_status():
    cli = FakeCLI()
    browser = BrowserSkillAdapter(cli)
    tabs = browser.list_tabs("agent")
    assert tabs[0]["tab_id"] == 1
    assert any(c[:2] == ["tab", "list"] and "--session" in c for c, _ in cli.calls)
