from types import SimpleNamespace

from agent_control.whatsapp_intelligence import WhatsAppIntelligence, WhatsAppIntelligenceStore, extract_messages
from agent_control.session import Session


def elem(ref, role, name, value="", **attrs):
    return SimpleNamespace(ref=ref, role=role, name=name, value=value, raw=attrs, attributes=attrs)


def obs(*elements):
    return SimpleNamespace(elements=tuple(elements), text="", raw={"elements": list(elements)})


def test_user_sent_semantic_message_is_outgoing():
    messages = extract_messages(obs(
        elem("@h", "heading", "Mummy"),
        elem("@m", "text", "You: Meeting tomorrow at 6 PM"),
        elem("@t", "text", "10:00 PM"),
    ), conversation_hint="Mummy", observed_at=1_800_000_000)
    message = next(m for m in messages if "meeting" in m.text.casefold())
    assert message.metadata["direction"] == "outgoing"
    assert message.metadata["is_outgoing"] is True
    assert "You" in message.metadata["sender_evidence"]


def test_received_semantic_message_is_incoming():
    messages = extract_messages(obs(
        elem("@h", "heading", "Mummy"),
        elem("@m", "text", "Mummy: Meeting tomorrow at 6 PM"),
        elem("@t", "text", "10:00 PM"),
    ), conversation_hint="Mummy", observed_at=1_800_000_000)
    message = next(m for m in messages if "meeting" in m.text.casefold())
    assert message.metadata["direction"] == "incoming"
    assert message.metadata["is_outgoing"] is False
    assert "Mummy" in message.metadata["sender_evidence"]


def test_missing_sender_evidence_remains_unknown():
    messages = extract_messages({"messages": [{
        "message_id": "m1", "conversation_id": "Mummy", "timestamp": 1_800_000_000,
        "text": "meeting tomorrow", "sender": "", "unread": True,
    }]})
    assert messages[0].metadata["direction"] == "unknown"
    assert messages[0].metadata["is_outgoing"] is None


def test_read_unread_does_not_change_direction():
    base = {"message_id": "m1", "conversation_id": "Mummy", "timestamp": 1_800_000_000, "text": "meeting tomorrow", "from_me": True}
    read = extract_messages({"messages": [{**base, "read": True}]})[0]
    unread = extract_messages({"messages": [{**base, "read": False}]})[0]
    assert read.metadata["direction"] == unread.metadata["direction"] == "outgoing"
    assert read.metadata["is_outgoing"] is unread.metadata["is_outgoing"] is True


def test_group_current_user_is_outgoing_and_participant_is_incoming():
    outgoing = extract_messages({"messages": [{"message_id": "o", "conversation_id": "Group", "timestamp": 1_800_000_000, "text": "meeting tomorrow", "sender": "You"}]})[0]
    incoming = extract_messages({"messages": [{"message_id": "i", "conversation_id": "Group", "timestamp": 1_800_000_001, "text": "meeting tomorrow", "sender": "Saksham"}]})[0]
    assert outgoing.metadata["direction"] == "outgoing"
    assert incoming.metadata["direction"] == "incoming"


def test_self_chat_does_not_force_outgoing_without_evidence():
    message = extract_messages({"messages": [{"message_id": "s", "conversation_id": "You", "timestamp": 1_800_000_000, "text": "meeting tomorrow"}]})[0]
    assert message.metadata["direction"] == "unknown"
    assert message.metadata["is_outgoing"] is None


def test_event_persistence_retains_direction_evidence(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    service = WhatsAppIntelligence(store)
    now = 1_800_000_000.0
    service.enable(now=now)
    events = service.process_observation({"messages": [{
        "message_id": "m1", "conversation_id": "Mummy", "timestamp": now,
        "text": "meeting tomorrow at 6pm", "sender": "Mummy", "from_me": False,
    }]}, observed_at=now)
    event = next(e for e in events if e.type.value == "MEETING")
    evidence = store.event_evidence(event.event_id)
    assert evidence[0]["evidence_id"]
    assert event.evidence[0]["direction"] == "incoming"


def test_presentation_callback_is_wired_and_does_not_need_raw_message(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    seen = []
    service = WhatsAppIntelligence(store, on_intelligence=seen.append)
    service.enable(now=1_800_000_000.0)
    events = service.process_observation({"messages": [{
        "message_id": "m1", "conversation_id": "Mummy", "timestamp": 1_800_000_000,
        "text": "meeting tomorrow at 6pm", "sender": "Mummy", "from_me": False,
    }]}, observed_at=1_800_000_000.0)
    assert events and seen
    assert seen[0].event_id == events[0].event_id


def test_session_has_structured_whatsapp_presenter():
    assert callable(getattr(Session, "_present_whatsapp_intelligence", None))
