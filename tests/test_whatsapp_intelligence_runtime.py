from __future__ import annotations

import threading
import time

from agent_control.response import Narrator
from agent_control.session import Session
from agent_control.whatsapp_control import WhatsAppOwner
from agent_control.whatsapp_intelligence import WhatsAppIntelligence, WhatsAppIntelligenceStore


def _msg(message_id="new", text="confirmed meeting tomorrow at 6 PM", ts=2_000.0):
    return {
        "message_id": message_id,
        "conversation_id": "chat-1",
        "timestamp": ts,
        "sender": "alice",
        "text": text,
    }


class FakeBrowser:
    def __init__(self, observation=None, error=None):
        self.observation = observation or {"messages": []}
        self.error = error
        self.observe_count = 0
        self._last_observation = None

    def observe(self):
        if self.error:
            raise self.error
        self.observe_count += 1
        self._last_observation = self.observation
        return self.observation


def _session(tmp_path, monkeypatch, browser):
    monkeypatch.setenv("DEIMOS_WHATSAPP_INTELLIGENCE", str(tmp_path / "wa.sqlite3"))
    session = Session.build(speech=False, show_status=False, debug=True)
    monkeypatch.setattr(session, "_whatsapp_intelligence_browser_for_observation", lambda: browser)
    return session


def test_enable_starts_runtime_observer(monkeypatch, tmp_path):
    browser = FakeBrowser()
    session = _session(tmp_path, monkeypatch, browser)
    try:
        turn = session._enable_whatsapp_intelligence()
        assert "ENABLED" in turn.reply
        service = session._whatsapp_intelligence
        assert service is not None
        assert service.observer_running
        deadline = time.time() + 2
        while time.time() < deadline and browser.observe_count == 0:
            time.sleep(0.02)
        assert browser.observe_count > 0
    finally:
        session.close()


def test_observer_feeds_existing_engine(monkeypatch, tmp_path):
    browser = FakeBrowser({"messages": [_msg()]})
    session = _session(tmp_path, monkeypatch, browser)
    try:
        service = session._whatsapp_intelligence_service()
        service.enable(now=1_000.0)
        service.start_observing(browser, interval_s=0.5)
        deadline = time.time() + 2
        while time.time() < deadline and not service.store.list_events():
            time.sleep(0.02)
        assert service.store.list_events()
        assert browser.observe_count > 0
    finally:
        session.close()


def test_human_ownership_pauses_observer(monkeypatch, tmp_path):
    browser = FakeBrowser()
    session = _session(tmp_path, monkeypatch, browser)
    try:
        service = session._whatsapp_intelligence_service()
        service.enable(now=1_000.0)
        service.start_observing(browser, interval_s=0.5)
        deadline = time.time() + 1
        while time.time() < deadline and browser.observe_count == 0:
            time.sleep(0.02)
        before = browser.observe_count
        session._whatsapp_control.release_to_human()
        time.sleep(0.7)
        assert browser.observe_count == before
    finally:
        session.close()


def test_takeover_resumes_persisted_observer(monkeypatch, tmp_path):
    browser = FakeBrowser()
    session = _session(tmp_path, monkeypatch, browser)
    try:
        service = session._whatsapp_intelligence_service()
        service.enable(now=1_000.0)
        session._whatsapp_control.release_to_human()
        session._whatsapp_control.browser = browser

        browser.open_whatsapp = lambda: {"ok": True, "state": "READY"}
        turn = session._whatsapp_control_task(
            __import__("agent_control.runtime_control", fromlist=["RuntimeControlCommand", "RuntimeControlKind"]).RuntimeControlCommand(
                __import__("agent_control.runtime_control", fromlist=["RuntimeControlKind"]).RuntimeControlKind.WHATSAPP_TAKEOVER
            ),
            raw="take control of whatsapp",
            source="text",
        )
        assert "DEIMOS_CONTROL" in turn.reply
        assert session._whatsapp_control.owner is WhatsAppOwner.DEIMOS
        assert service.observer_running
    finally:
        session.close()


def test_active_whatsapp_task_prevents_observation(monkeypatch, tmp_path):
    browser = FakeBrowser()
    session = _session(tmp_path, monkeypatch, browser)
    try:
        service = session._whatsapp_intelligence_service()
        service.enable(now=1_000.0)
        session._whatsapp_control.active_task_id = "task-1"
        service.start_observing(browser, interval_s=0.5)
        time.sleep(0.7)
        assert browser.observe_count == 0
    finally:
        session.close()


def test_observer_shutdown_is_clean(monkeypatch, tmp_path):
    browser = FakeBrowser()
    session = _session(tmp_path, monkeypatch, browser)
    service = session._whatsapp_intelligence_service()
    service.enable(now=1_000.0)
    service.start_observing(browser, interval_s=0.5)
    assert service.observer_running
    session.close()
    assert not service.observer_running


def test_observation_exception_is_contained(monkeypatch, tmp_path):
    browser = FakeBrowser(error=RuntimeError("temporary browser failure"))
    session = _session(tmp_path, monkeypatch, browser)
    try:
        service = session._whatsapp_intelligence_service()
        service.enable(now=1_000.0)
        service.start_observing(browser, interval_s=0.5)
        time.sleep(0.7)
        assert service.observer_running
        # The Session remains usable after an observer-side exception.
        assert not session._closing
    finally:
        session.close()


def test_cutoff_survives_takeover_and_observation_resume(monkeypatch, tmp_path):
    browser = FakeBrowser({"messages": [_msg(ts=1_500.0)]})
    session = _session(tmp_path, monkeypatch, browser)
    try:
        service = session._whatsapp_intelligence_service()
        service.enable(now=1_000.0)
        cutoff = service.store.state().enabled_at
        session._whatsapp_control.release_to_human()
        session._whatsapp_control.browser = browser
        browser.open_whatsapp = lambda: {"ok": True, "state": "READY"}
        from agent_control.runtime_control import RuntimeControlCommand, RuntimeControlKind
        session._whatsapp_control_task(
            RuntimeControlCommand(RuntimeControlKind.WHATSAPP_TAKEOVER),
            raw="take control of whatsapp", source="text",
        )
        assert service.store.state().enabled_at == cutoff
        deadline = time.time() + 2
        while time.time() < deadline and not service.store.list_events():
            time.sleep(0.02)
        assert service.store.state().enabled_at == cutoff
        assert service.store.list_events()
    finally:
        session.close()


def test_persisted_enabled_state_restores_observer_on_session_start(monkeypatch, tmp_path):
    path = tmp_path / "wa.sqlite3"
    store = WhatsAppIntelligenceStore(path)
    store.enable(now=1_000.0)
    browser = FakeBrowser()
    monkeypatch.setenv("DEIMOS_WHATSAPP_INTELLIGENCE", str(path))
    monkeypatch.setattr(Session, "_whatsapp_intelligence_browser_for_observation", lambda self: browser)
    session = Session.build(speech=False, show_status=False, debug=True)
    try:
        service = session._whatsapp_intelligence
        assert service is not None
        assert service.store.state().enabled
        assert service.observer_running
        deadline = time.time() + 2
        while time.time() < deadline and browser.observe_count == 0:
            time.sleep(0.02)
        assert browser.observe_count > 0
    finally:
        session.close()


def _home_obs(*elements):
    return type("Obs", (), {
        "generation": 1,
        "elements": tuple(elements),
        "raw": {"elements": []},
    })()


def _row(ref, name, role="button"):
    return type("E", (), {"ref": ref, "name": name, "role": role, "value": "", "raw": None, "attributes": {}})()


def _target_browser(observations, *, header_name="Mummy"):
    class TargetBrowser:
        def __init__(self):
            self.observations = list(observations)
            self.index = 0
            self.observe_count = 0
            self.opened = []
            self.scrolled = []
            self.searched = []
            self._last_observation = None

        def observe(self):
            self.observe_count += 1
            obs = self.observations[min(self.index, len(self.observations) - 1)]
            self._last_observation = obs
            return obs

        def open_whatsapp_chat_row(self, target, timeout_s=8.0):
            self.opened.append(target)
            self.index = min(self.index + 1, len(self.observations) - 1)
            return {"ok": True}

        def open_whatsapp_chat(self, target, timeout_s=8.0):
            self.searched.append(target)
            raise AssertionError("Home discovery must never use WhatsApp search")

        def go_back(self):
            return {"ok": True}

        def scroll_to(self, target):
            self.scrolled.append(target)
            self.index = min(self.index + 1, len(self.observations) - 1)
            return {"ok": True}

    return TargetBrowser()


def test_authorized_unread_home_row_is_opened_without_search(monkeypatch, tmp_path):
    home = _home_obs(
        _row("@row1", "6 unread messages Mummy 12:28 AM Meeting at 6"),
        _row("@mummy", "Mummy"),
        _row("@random", "9 unread messages Random Group"),
    )
    browser = _target_browser([home])
    monkeypatch.setenv("DEIMOS_WHATSAPP_INTELLIGENCE", str(tmp_path / "wa.sqlite3"))
    session = _session(tmp_path, monkeypatch, browser)
    try:
        service = session._whatsapp_intelligence_service()
        service.store.authorize_target("Mummy")
        service.enable(now=1_000.0)
        service.start_observing(browser, interval_s=0.5)
        deadline = time.time() + 2
        while time.time() < deadline and not browser.opened:
            time.sleep(0.02)
        assert len(browser.opened) == 1
        assert browser.opened[0].ref == "@row1"
        assert browser.searched == []
    finally:
        session.close()


def test_authorized_visible_but_not_unread_is_not_opened(monkeypatch, tmp_path):
    browser = _target_browser([_home_obs(_row("@mummy", "Mummy 12:28 AM Old conversation"))])
    monkeypatch.setenv("DEIMOS_WHATSAPP_INTELLIGENCE", str(tmp_path / "wa.sqlite3"))
    session = _session(tmp_path, monkeypatch, browser)
    try:
        service = session._whatsapp_intelligence_service()
        service.store.authorize_target("Mummy")
        service.enable(now=1_000.0)
        service.start_observing(browser, interval_s=0.5)
        time.sleep(0.7)
        assert browser.opened == []
        assert browser.searched == []
    finally:
        session.close()


def test_unauthorized_unread_home_row_is_ignored(monkeypatch, tmp_path):
    browser = _target_browser([_home_obs(_row("@random", "6 unread messages Random Group"))])
    monkeypatch.setenv("DEIMOS_WHATSAPP_INTELLIGENCE", str(tmp_path / "wa.sqlite3"))
    session = _session(tmp_path, monkeypatch, browser)
    try:
        service = session._whatsapp_intelligence_service()
        service.store.authorize_target("Mummy")
        service.enable(now=1_000.0)
        service.start_observing(browser, interval_s=0.5)
        time.sleep(0.7)
        assert browser.opened == []
        assert browser.searched == []
    finally:
        session.close()


def test_multiple_authorized_unread_rows_are_processed_one_at_a_time(monkeypatch, tmp_path):
    home = _home_obs(
        _row("@mummy", "2 unread messages Mummy 12:28 AM Meeting"),
        _row("@dad", "1 unread message Dad 12:21 AM Documents"),
    )
    browser = _target_browser([home])
    monkeypatch.setenv("DEIMOS_WHATSAPP_INTELLIGENCE", str(tmp_path / "wa.sqlite3"))
    session = _session(tmp_path, monkeypatch, browser)
    try:
        service = session._whatsapp_intelligence_service()
        service.store.authorize_target("Mummy")
        service.store.authorize_target("Dad")
        service.enable(now=1_000.0)
        service.start_observing(browser, interval_s=0.5)
        deadline = time.time() + 2
        while time.time() < deadline and not browser.opened:
            time.sleep(0.02)
        assert len(browser.opened) == 1
        assert browser.opened[0].ref in {"@mummy", "@dad"}
    finally:
        session.close()


def test_authorized_target_can_be_found_by_bounded_home_scrolling(monkeypatch, tmp_path):
    first = _home_obs(_row("@row1", "Dad 11:00 PM Old"))
    second = _home_obs(_row("@row2", "3 unread messages Mummy 12:28 AM Meeting"))
    browser = _target_browser([first, second])
    monkeypatch.setenv("DEIMOS_WHATSAPP_INTELLIGENCE", str(tmp_path / "wa.sqlite3"))
    session = _session(tmp_path, monkeypatch, browser)
    try:
        service = session._whatsapp_intelligence_service()
        service.store.authorize_target("Mummy")
        service.enable(now=1_000.0)
        service.start_observing(browser, interval_s=0.5)
        deadline = time.time() + 2
        while time.time() < deadline and not browser.opened:
            time.sleep(0.02)
        assert len(browser.scrolled) == 1
        assert len(browser.opened) == 1
        assert browser.opened[0].ref == "@row2"
        assert browser.searched == []
    finally:
        session.close()


def test_missing_target_stops_after_bounded_scroll_budget(monkeypatch, tmp_path):
    observations = [_home_obs(_row(f"@row{i}", f"Other {i}")) for i in range(5)]
    browser = _target_browser(observations)
    monkeypatch.setenv("DEIMOS_WHATSAPP_INTELLIGENCE", str(tmp_path / "wa.sqlite3"))
    session = _session(tmp_path, monkeypatch, browser)
    try:
        service = session._whatsapp_intelligence_service()
        service.store.authorize_target("Mummy")
        service.enable(now=1_000.0)
        service.start_observing(browser, interval_s=0.5)
        deadline = time.time() + 2
        while time.time() < deadline and len(browser.scrolled) < service._home_scroll_budget:
            time.sleep(0.02)
        assert len(browser.opened) == 0
        assert len(browser.scrolled) <= service._home_scroll_budget
        assert browser.searched == []
    finally:
        session.close()


def test_opening_home_row_uses_fresh_conversation_observation(monkeypatch, tmp_path):
    home = _home_obs(_row("@row1", "1 unread message Mummy 12:28 AM Meeting"))
    conversation = type("Obs", (), {
        "generation": 2,
        "elements": (_row("@header", "Mummy", "heading"),),
        "raw": {"messages": []},
    })()
    browser = _target_browser([home, conversation])
    monkeypatch.setenv("DEIMOS_WHATSAPP_INTELLIGENCE", str(tmp_path / "wa.sqlite3"))
    session = _session(tmp_path, monkeypatch, browser)
    try:
        service = session._whatsapp_intelligence_service()
        service.store.authorize_target("Mummy")
        service.enable(now=1_000.0)
        service.start_observing(browser, interval_s=0.5)
        deadline = time.time() + 2
        while time.time() < deadline and browser.observe_count < 2:
            time.sleep(0.02)
        assert browser.opened
        assert browser.observe_count >= 2
    finally:
        session.close()


def test_human_ownership_prevents_home_observation(monkeypatch, tmp_path):
    browser = _target_browser([_home_obs(_row("@row1", "1 unread message Mummy"))])
    monkeypatch.setenv("DEIMOS_WHATSAPP_INTELLIGENCE", str(tmp_path / "wa.sqlite3"))
    session = _session(tmp_path, monkeypatch, browser)
    try:
        service = session._whatsapp_intelligence_service()
        service.store.authorize_target("Mummy")
        service.enable(now=1_000.0)
        session._whatsapp_control.release_to_human()
        service.start_observing(browser, interval_s=0.5)
        time.sleep(0.7)
        assert browser.observe_count == 0
        assert browser.opened == []
    finally:
        session.close()


def test_active_whatsapp_task_prevents_home_observation(monkeypatch, tmp_path):
    browser = _target_browser([_home_obs(_row("@row1", "1 unread message Mummy"))])
    monkeypatch.setenv("DEIMOS_WHATSAPP_INTELLIGENCE", str(tmp_path / "wa.sqlite3"))
    session = _session(tmp_path, monkeypatch, browser)
    try:
        service = session._whatsapp_intelligence_service()
        service.store.authorize_target("Mummy")
        service.enable(now=1_000.0)
        session._whatsapp_control.active_task_id = "task-1"
        service.start_observing(browser, interval_s=0.5)
        time.sleep(0.7)
        assert browser.observe_count == 0
        assert browser.opened == []
    finally:
        session.close()


def test_authorized_target_persistence_is_unchanged(monkeypatch, tmp_path):
    browser = _target_browser([_home_obs()])
    monkeypatch.setenv("DEIMOS_WHATSAPP_INTELLIGENCE", str(tmp_path / "wa.sqlite3"))
    session = _session(tmp_path, monkeypatch, browser)
    try:
        service = session._whatsapp_intelligence_service()
        service.store.authorize_target("Mummy", now=123.0)
        service.store.authorize_target("TechTalks & Collabs", now=124.0)
        assert service.store.authorized_targets() == ("Mummy", "TechTalks & Collabs")
        assert service.store.revoke_target("Mummy")
        assert service.store.authorized_targets() == ("TechTalks & Collabs",)
    finally:
        session.close()


def test_cutoff_and_deduplication_remain_authoritative(monkeypatch, tmp_path):
    browser = FakeBrowser({"messages": [_msg(message_id="old", ts=900.0), _msg(message_id="new", ts=1_500.0)]})
    session = _session(tmp_path, monkeypatch, browser)
    try:
        service = session._whatsapp_intelligence_service()
        service.enable(now=1_000.0)
        service.process_observation(browser.observation)
        first = service.store.state()
        service.process_observation(browser.observation)
        second = service.store.state()
        assert first.last_processed_message_id == "new"
        assert second.last_processed_message_id == "new"
        assert len(service.store.list_events()) >= 1
    finally:
        session.close()



def test_browser_home_row_open_path_uses_click_not_search(monkeypatch):
    from agent_control.skills.browser.backend import BrowserSkillAdapter, BrowserTarget

    backend = BrowserSkillAdapter.__new__(BrowserSkillAdapter)
    clicked = []
    from agent_control.skills.browser.backend import BrowserSkillResult
    backend.click = lambda target: clicked.append(target.ref) or BrowserSkillResult({"ok": True})
    backend._whatsapp_chat_header_matches = lambda contact, timeout_s=3.0: contact == "Mummy"

    target = BrowserTarget("@home-row", role="button", name="Mummy", generation=7)
    result = backend.open_whatsapp_chat_row(target)

    assert result.ok
    assert clicked == ["@home-row"]
    assert result["telemetry"]["semantic_target_category"] == "home_chat_row"
