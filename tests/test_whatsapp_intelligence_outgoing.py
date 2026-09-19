from __future__ import annotations

from types import SimpleNamespace

from agent_control.whatsapp_intelligence import (
    IntelligenceEventType,
    WhatsAppIntelligence,
    WhatsAppIntelligenceStore,
    extract_messages,
)


def semantic_obs(*elements):
    return SimpleNamespace(elements=tuple(elements), text="", raw={"elements": list(elements)})


def elem(ref, role, name, value="", **attrs):
    return SimpleNamespace(ref=ref, role=role, name=name, value=value, raw=attrs, attributes=attrs)


def test_semantic_outgoing_message_is_extracted_from_you_marker():
    obs = semantic_obs(
        elem("@e10", "heading", "Saksham"),
        elem("@e11", "text", "You: Meeting at 6pm"),
        elem("@e12", "text", "11:00 PM"),
    )
    messages = extract_messages(obs, conversation_hint="saksham", observed_at=1779120000.0)
    message = next(m for m in messages if "meeting" in m.text.casefold())
    assert message.text == "Meeting at 6pm"
    assert message.metadata["direction"] == "outgoing"
    assert message.metadata["is_outgoing"] is True


def test_semantic_outgoing_message_is_extracted_from_separate_sent_marker():
    obs = semantic_obs(
        elem("@e10", "heading", "Saksham"),
        elem("@e11", "text", "Meeting at 6pm"),
        elem("@e12", "text", "11:00 PM"),
        elem("@e13", "text", "Sent", **{"aria-label": "Sent"}),
    )
    messages = extract_messages(obs, conversation_hint="saksham", observed_at=1779120000.0)
    message = next(m for m in messages if "meeting" in m.text.casefold())
    assert message.metadata["direction"] == "outgoing"
    assert message.metadata["is_outgoing"] is True


def test_structured_outgoing_message_creates_meeting_event(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    service = WhatsAppIntelligence(store)
    now = 2_000_000.0
    service.enable(now=now)
    observation = {
        "messages": [{
            "message_id": "self-1",
            "conversation_id": "saksham",
            "timestamp": now,
            "text": "I am organizing the meeting at 6pm",
            "sender": "You",
            "from_me": True,
        }]
    }
    events = service.process_observation(observation, observed_at=now)
    assert any(event.type is IntelligenceEventType.MEETING for event in events)
    event = next(event for event in events if event.type is IntelligenceEventType.MEETING)
    assert event.evidence[0]["direction"] == "outgoing"
    assert event.evidence[0]["is_outgoing"] is True


def test_home_row_prefers_clickable_child_for_special_rows(tmp_path):
    class Browser:
        def __init__(self):
            self.opened_ref = None
            self._last_observation = None
            self.home = semantic_obs(
                elem("@e1", "listitem", ""),
                elem("@e2", "button", "Saksham"),
                elem("@e3", "text", "Meeting at 6pm"),
                elem("@e4", "text", "11:00 PM"),
            )
            self.chat = semantic_obs(
                elem("@e5", "heading", "Saksham"),
                elem("@e6", "textbox", "Type a message"),
                elem("@e7", "text", "You: Meeting at 6pm"),
                elem("@e8", "text", "11:00 PM"),
            )
            self.observations = [self.home, self.chat]
            self.index = 0

        def observe(self):
            value = self.observations[min(self.index, len(self.observations) - 1)]
            self.index += 1
            self._last_observation = value
            return value

        def open_whatsapp_chat_row(self, target, timeout_s=8.0):
            self.opened_ref = target.ref
            return {"ok": True}

    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    store.authorize_target("Saksham")
    service = WhatsAppIntelligence(store)
    service.enable(now=2_000_000.0)
    browser = Browser()
    observation = browser.observe()
    service._observe_authorized_targets(browser, observation)
    assert browser.opened_ref == "@e2"
