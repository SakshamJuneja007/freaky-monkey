from datetime import datetime, timezone
import os
import tempfile

import pytest

from agent_control.whatsapp_intelligence import (
    IntelligenceEventType,
    WhatsAppIntelligence,
    WhatsAppIntelligenceStore,
    WhatsAppMessage,
    _source_timezone_label,
    extract_messages,
)


UTC = timezone.utc


def _service():
    fd, path = tempfile.mkstemp(suffix=".sqlite3")
    os.close(fd)
    store = WhatsAppIntelligenceStore(path)
    service = WhatsAppIntelligence(store, on_debug=lambda _line: None)
    return store, service, path


def _message(text: str, *, metadata=None) -> WhatsAppMessage:
    return WhatsAppMessage(
        message_id="semantic-time-test",
        conversation_id="mummy",
        timestamp=datetime(2026, 9, 18, 19, 30, tzinfo=UTC).timestamp(),
        text=text,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# EVENT DETECTION MUST PRECEDE TIME EXTRACTION
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "1",
        "01 AM",
        "3:17 AM",
        "5",
        "5:01 AM",
        "6 PM",
        "Happy Birthday Anu 5",
        "See you at 6 PM",
        "I am free at 6 PM and busy at 8 PM",
    ],
)
def test_arbitrary_or_unassociated_clock_does_not_create_meeting(text):
    store, service, path = _service()
    try:
        message = _message(text)
        proposal = service._event_proposal(message)
        assert proposal.event_type is None
        assert proposal.action == "NO_EVENT"
        assert service._analyze_message(message, [message]) == []
        assert store.list_events() == []
    finally:
        store.close()
        os.unlink(path)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Meeting at 6 PM", "18:00"),
        ("Meeting at 10 PM", "22:00"),
        ("Meeting at 6:30 PM", "18:30"),
        ("Meeting at 12 AM", "00:00"),
        ("Meeting at 12 PM", "12:00"),
        ("Meeting at 1 PM", "13:00"),
        ("01:00 Meeting at 6 PM", "18:00"),
        ("05:00 Meeting at 10 PM", "22:00"),
        ("Meeting at 6 PM 01:00", "18:00"),
    ],
)
def test_event_clock_is_contextually_associated(text, expected):
    store, service, _path = _service()
    try:
        proposal = service._event_proposal(_message(text))
        assert proposal.event_type is IntelligenceEventType.MEETING
        assert proposal.resolved_time == expected
        assert proposal.time_expression.strip() == expected.replace("18:00", "6 PM").replace("22:00", "10 PM").replace("18:30", "6:30 PM").replace("00:00", "12 AM").replace("13:00", "1 PM").replace("12:00", "12 PM")
    finally:
        store.close()
        os.unlink(_path)


def test_multiple_event_times_selects_clock_associated_with_meeting():
    store, service, path = _service()
    try:
        message = _message("Meeting at 6 PM, finish work at 8 PM")
        proposal = service._event_proposal(message)
        assert proposal.event_type is IntelligenceEventType.MEETING
        assert proposal.time_expression.strip() == "6 PM"
        assert proposal.resolved_time == "18:00"
    finally:
        store.close()
        os.unlink(path)


def test_tomorrow_at_6_pm_has_event_context_and_local_wall_clock():
    store, service, path = _service()
    try:
        proposal = service._event_proposal(_message("Meeting tomorrow at 6 PM"))
        assert proposal.event_type is IntelligenceEventType.MEETING
        assert proposal.resolved_date == "2026-09-20"
        assert proposal.resolved_time == "18:00"
        assert proposal.timezone == "Asia/Kolkata"
    finally:
        store.close()
        os.unlink(path)


# ---------------------------------------------------------------------------
# BROWSERSKILL OBSERVATION METADATA MUST NOT BECOME MESSAGE CONTENT
# ---------------------------------------------------------------------------


def test_combined_semantic_label_separates_ui_clock_from_message_text():
    raw = {
        "elements": [
            {"ref": "e0", "role": "heading", "name": "mummy"},
            {"ref": "e1", "role": "text", "value": "01:00 Meeting at 6 PM"},
        ]
    }
    messages = extract_messages(raw, conversation_hint="mummy", observed_at=datetime(2026, 9, 18, 20, tzinfo=UTC).timestamp())
    assert len(messages) == 1
    assert messages[0].text == "Meeting at 6 PM"

    store, service, path = _service()
    try:
        proposal = service._event_proposal(messages[0])
        assert proposal.event_type is IntelligenceEventType.MEETING
        assert proposal.resolved_time == "18:00"
    finally:
        store.close()
        os.unlink(path)


def test_structured_message_timestamp_stays_outside_text():
    raw = {
        "message_id": "m1",
        "conversation_id": "mummy",
        "timestamp": "2026-09-18T19:30:00+00:00",
        "text": "Meeting at 6 PM",
    }
    message = extract_messages(raw)[0]
    assert message.text == "Meeting at 6 PM"
    assert datetime.fromtimestamp(message.timestamp, tz=UTC).isoformat() == "2026-09-18T19:30:00+00:00"


def test_explicit_utc_message_timestamp_reports_utc_source_timezone():
    message = _message(
        "Meeting at 6 PM",
        metadata={"timestamp": "2026-09-18T19:30:00+00:00"},
    )
    assert _source_timezone_label(message) == "UTC"


def test_event_debug_does_not_run_for_clock_only_message():
    debug = []
    store = WhatsAppIntelligenceStore(":memory:")
    service = WhatsAppIntelligence(store, on_debug=debug.append)
    service._event_proposal(_message("3:17 AM"))
    assert not any(line.startswith("WHATSAPP_TIME_DEBUG:") for line in debug)
    store.close()
