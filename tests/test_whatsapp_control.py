from __future__ import annotations

import pytest

from agent_control.response import Narrator
from agent_control.runtime_control import (
    RuntimeControlKind,
    classify_runtime_control,
    classify_runtime_controls,
)
from agent_control.session import Session
from agent_control.whatsapp_control import WhatsAppControlLease, WhatsAppOwner


class FakeWhatsAppBrowser:
    def __init__(self, *, state: str = "READY"):
        self.state = state
        self.open_calls = 0
        self.close_calls = 0
        self.observation_calls = 0

    def open_whatsapp(self):
        self.open_calls += 1
        self.observation_calls += 1
        if self.state != "READY":
            raise RuntimeError("WhatsApp state unavailable")
        return {"ok": True, "provider": "whatsapp", "state": "READY"}

    def close_session(self):
        self.close_calls += 1


def _session() -> Session:
    return Session(narrator=Narrator(write=lambda _text: None), planner="mock")


def test_initial_whatsapp_owner_is_deimos():
    assert WhatsAppControlLease().owner is WhatsAppOwner.DEIMOS


def test_handoff_command_is_control_plane_intent():
    command = classify_runtime_control("give me control of WhatsApp")
    assert command is not None
    assert command.kind is RuntimeControlKind.WHATSAPP_HANDOFF


def test_takeover_command_is_control_plane_intent():
    command = classify_runtime_control("DEIMOS, you can take over WhatsApp")
    assert command is not None
    assert command.kind is RuntimeControlKind.WHATSAPP_TAKEOVER


def test_handoff_transfers_owner_to_human():
    session = _session()
    browser = FakeWhatsAppBrowser()
    session._whatsapp_control.bind_browser(browser)
    turn = session.submit("give me control of WhatsApp")
    assert session._whatsapp_control.owner is WhatsAppOwner.HUMAN
    assert "HUMAN_CONTROL" in turn.reply


def test_human_ownership_blocks_new_whatsapp_resource_acquisition():
    session = _session()
    browser = FakeWhatsAppBrowser()
    session._whatsapp_control.bind_browser(browser)
    session._whatsapp_control.release_to_human()
    with pytest.raises(RuntimeError, match="whatsapp_resource_human_owned"):
        session._browser_for_task("new-whatsapp-task", resource_key="whatsapp")


def test_human_ownership_blocks_whatsapp_automation_submission():
    session = _session()
    session._whatsapp_control.release_to_human()
    turn = session.submit("send hello to papa on whatsapp")
    assert turn.result is None
    assert "under your control" in turn.reply


def test_handoff_does_not_call_browser_close_or_logout():
    session = _session()
    browser = FakeWhatsAppBrowser()
    session._whatsapp_control.bind_browser(browser)
    session.submit("give me control of WhatsApp")
    assert browser.close_calls == 0
    assert session._whatsapp_control.browser is browser


def test_handoff_keeps_existing_browser_resource_identity():
    session = _session()
    browser = FakeWhatsAppBrowser()
    session._whatsapp_control.bind_browser(browser)
    session.submit("give me control of WhatsApp")
    assert session._whatsapp_control.browser is browser


def test_handoff_cancels_only_pending_whatsapp_runtime_tasks():
    session = _session()
    session._whatsapp_control.bind_browser(FakeWhatsAppBrowser())
    whatsapp = session._runtime.create_task("send hello on whatsapp", task_type="messaging")
    other = session._runtime.create_task("play a song", task_type="browser")
    session._runtime.attach_resource(whatsapp.task_id, "whatsapp", site="whatsapp", state="ATTACHED")
    session._runtime.attach_resource(other.task_id, "youtube", site="youtube", state="ATTACHED")
    session.submit("give me control of WhatsApp")
    assert session._runtime.get_task(whatsapp.task_id).state == "CANCELLED"
    assert session._runtime.get_task(other.task_id).state == "CREATED"


def test_active_whatsapp_action_blocks_unsafe_handoff():
    session = _session()
    session._whatsapp_control.active_task_id = "whatsapp-running"
    turn = session.submit("give me control of WhatsApp")
    assert "TRANSFER_BLOCKED" in turn.reply
    assert session._whatsapp_control.owner is WhatsAppOwner.DEIMOS



def test_takeover_crosses_human_guard_when_existing_resource_has_not_been_bound():
    session = _session()
    session._whatsapp_control.release_to_human()

    browser = FakeWhatsAppBrowser()

    class Factory:
        def new_task_session(self):
            return browser

    session._browser_backend = Factory()
    turn = session.submit("you can take over WhatsApp")

    assert "DEIMOS_CONTROL" in turn.reply
    assert session._whatsapp_control.owner is WhatsAppOwner.DEIMOS
    assert session._whatsapp_control.browser is browser
    assert browser.open_calls == 1


def test_takeover_reacquires_existing_browser_resource():
    session = _session()
    browser = FakeWhatsAppBrowser()
    session._whatsapp_control.bind_browser(browser)
    session._whatsapp_control.release_to_human()
    session.submit("you can take over WhatsApp")
    assert session._whatsapp_control.owner is WhatsAppOwner.DEIMOS
    assert session._whatsapp_control.browser is browser


def test_takeover_performs_fresh_observation():
    session = _session()
    browser = FakeWhatsAppBrowser()
    session._whatsapp_control.bind_browser(browser)
    session._whatsapp_control.release_to_human()
    session.submit("take over WhatsApp")
    assert browser.open_calls == 1
    assert browser.observation_calls == 1


def test_takeover_fails_safely_when_state_is_unusable():
    session = _session()
    browser = FakeWhatsAppBrowser(state="UNKNOWN")
    session._whatsapp_control.bind_browser(browser)
    session._whatsapp_control.release_to_human()
    turn = session.submit("take over WhatsApp")
    assert "TAKEOVER_FAILED" in turn.reply
    assert session._whatsapp_control.owner is WhatsAppOwner.HUMAN


def test_takeover_does_not_create_a_new_browser_when_existing_resource_exists():
    session = _session()
    browser = FakeWhatsAppBrowser()
    session._whatsapp_control.bind_browser(browser)
    session._whatsapp_control.release_to_human()
    session.submit("DEIMOS can take over WhatsApp")
    assert session._whatsapp_control.browser is browser
    assert len(session._browser_tasks) == 0


def test_unrelated_browser_resource_is_not_reused_for_whatsapp_control():
    session = _session()
    youtube = FakeWhatsAppBrowser()
    session._browser_tasks["youtube-task"] = youtube
    session._whatsapp_control.release_to_human()
    with pytest.raises(RuntimeError, match="whatsapp_resource_human_owned"):
        session._browser_for_task("whatsapp-task", resource_key="whatsapp")
    assert session._browser_tasks["youtube-task"] is youtube


def test_failed_handoff_preserves_control_and_task_context():
    session = _session()
    session._whatsapp_control.active_task_id = "active-whatsapp"
    turn = session.submit("give me control of WhatsApp")
    assert session._whatsapp_control.owner is WhatsAppOwner.DEIMOS
    assert "active-whatsapp" not in turn.task.text
    assert session.history[-1] is turn


def test_runtime_control_classifier_does_not_capture_normal_whatsapp_messages():
    assert classify_runtime_controls("send hello to papa on whatsapp") == []


def test_whatsapp_intelligence_authorization_command_extracts_target():
    command = classify_runtime_control("allow WhatsApp intelligence to observe Mummy")
    assert command is not None
    assert command.kind is RuntimeControlKind.WHATSAPP_INTELLIGENCE_AUTHORIZE
    assert command.scope == "mummy"


def test_whatsapp_intelligence_revoke_command_extracts_target():
    command = classify_runtime_control("revoke WhatsApp intelligence from observing Mummy")
    assert command is not None
    assert command.kind is RuntimeControlKind.WHATSAPP_INTELLIGENCE_REVOKE
    assert command.scope == "mummy"
