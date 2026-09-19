from __future__ import annotations

from types import SimpleNamespace

from agent_control.whatsapp_intelligence import (
    WhatsAppIntelligence,
    WhatsAppIntelligenceStore,
    _semantic_self_chat_header_present,
    extract_messages,
)


class E:
    def __init__(self, ref, role, name, value=""):
        self.ref = ref
        self.role = role
        self.name = name
        self.value = value
        self.raw = None
        self.attributes = {}


class Obs:
    def __init__(self, elements, text=""):
        self.elements = tuple(elements)
        self.text = text
        self.raw = {"elements": [
            {"ref": e.ref, "role": e.role, "name": e.name, "value": e.value}
            for e in elements
        ]}


def test_self_chat_semantic_header_marks_message_outgoing():
    raw = {
        "elements": [
            {"ref": "@e1", "role": "heading", "name": "Saksham"},
            {"ref": "@e2", "role": "text", "name": "You"},
            {"ref": "@e3", "role": "textbox", "name": "Type a message"},
            {"ref": "@e4", "role": "text", "name": "Meeting at 6pm"},
            {"ref": "@e5", "role": "text", "name": "11:00 PM"},
        ]
    }
    assert _semantic_self_chat_header_present(raw, "Saksham")
    messages = extract_messages(
        raw,
        conversation_hint="Saksham",
        observed_at=1_789_000_000.0,
    )
    meeting = next(m for m in messages if m.text == "Meeting at 6pm")
    assert meeting.metadata["direction"] == "outgoing"
    assert meeting.metadata["is_outgoing"] is True


def test_opened_chat_observation_is_detected_as_conversation_even_with_sidebar():
    store = WhatsAppIntelligenceStore(":memory:")
    service = WhatsAppIntelligence(store)
    store.authorize_target("Saksham")
    observation = Obs([
        E("@e1", "button", "Chats"),
        E("@e2", "listitem", "Saksham 11:00 PM Meeting at 6pm"),
        E("@e3", "heading", "Saksham"),
        E("@e4", "textbox", "Type a message"),
    ])
    assert service._active_authorized_target(observation) == "Saksham"


def test_same_home_row_is_not_retriggered_when_unread_marker_disappears(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    service = WhatsAppIntelligence(store)
    store.authorize_target("Mummy")
    unread = Obs([E("@e1", "listitem", "Mummy 1 unread messages 11:00 PM Meeting at 6pm")])
    read = Obs([E("@e1", "listitem", "Mummy 11:00 PM Meeting at 6pm")])
    candidates1, _ = service._home_candidates(unread)
    candidates2, _ = service._home_candidates(read)
    assert candidates1 and candidates2
    assert candidates1[0][2] == candidates2[0][2]


def test_active_conversation_path_does_not_scroll():
    class Browser:
        def __init__(self):
            self.opened = []
            self.scrolled = 0
        def open_whatsapp_chat_row(self, target, timeout_s=8.0):
            self.opened.append(target.name)
            return {"ok": True}
        def scroll(self, amount):
            self.scrolled += 1
            raise AssertionError("scroll must not be used inside an active conversation")

    browser = Browser()
    store = WhatsAppIntelligenceStore(":memory:")
    service = WhatsAppIntelligence(store)
    store.authorize_target("Saksham")
    obs = Obs([
        E("@e1", "heading", "Saksham"),
        E("@e2", "textbox", "Type a message"),
        E("@e3", "listitem", "Saksham 11:00 PM Meeting at 6pm"),
    ])
    # No candidate requiring scroll is allowed to enter the exploration loop
    # while the conversation state is active.
    assert service._active_authorized_target(obs) == "Saksham"
    service._observe_authorized_targets(browser, obs, allow_scroll=False)
    assert browser.scrolled == 0
