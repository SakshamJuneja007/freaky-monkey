import os
import tempfile
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from agent_control.whatsapp_intelligence import (
    IntelligenceEventType,
    WhatsAppIntelligence,
    WhatsAppIntelligenceStore,
    WhatsAppMessage,
)


IST = ZoneInfo("Asia/Kolkata")


def _message(text: str, *, timestamp: str = "2026-09-18T19:18:00+00:00", timezone_name: str = "Asia/Kolkata") -> WhatsAppMessage:
    return WhatsAppMessage(
        message_id="m-test",
        conversation_id="mummy",
        timestamp=datetime.fromisoformat(timestamp).timestamp(),
        text=text,
        sender="Mummy",
        metadata={"source_timezone": "UTC", "event_timezone": timezone_name},
    )


def _service(debug=None):
    fd, path = tempfile.mkstemp(suffix=".sqlite3")
    os.close(fd)
    store = WhatsAppIntelligenceStore(path)
    service = WhatsAppIntelligence(store, on_debug=debug or (lambda _line: None))
    return store, service, path


def test_1_explicit_6_pm_is_18_00():
    store, service, path = _service()
    try:
        proposal = service._event_proposal(_message("Meeting at 6 PM"))
        assert proposal.resolved_time == "18:00"
        assert proposal.timezone == "Asia/Kolkata"
    finally:
        store.close()
        os.unlink(path)


def test_2_explicit_10_pm_is_22_00():
    store, service, path = _service()
    try:
        proposal = service._event_proposal(_message("Meeting at 10 PM"))
        assert proposal.resolved_time == "22:00"
    finally:
        store.close()
        os.unlink(path)


def test_3_explicit_6_30_pm_is_18_30():
    store, service, path = _service()
    try:
        proposal = service._event_proposal(_message("Meeting at 6:30 PM"))
        assert proposal.resolved_time == "18:30"
    finally:
        store.close()
        os.unlink(path)


def test_4_tomorrow_uses_event_timezone_date_and_18_00():
    store, service, path = _service()
    try:
        message = _message("Meeting tomorrow at 6 PM")
        proposal = service._event_proposal(message)
        source_local = datetime.fromtimestamp(message.timestamp, tz=IST)
        assert proposal.resolved_date == (source_local.date() + timedelta(days=1)).isoformat()
        assert proposal.resolved_time == "18:00"
    finally:
        store.close()
        os.unlink(path)


def test_5_12_hour_conversion_boundaries():
    store, service, path = _service()
    try:
        expected = {
            "Meeting at 12 AM": "00:00",
            "Meeting at 12 PM": "12:00",
            "Meeting at 1 PM": "13:00",
            "Meeting at 6 PM": "18:00",
        }
        for text, value in expected.items():
            proposal = service._event_proposal(_message(text))
            assert proposal.resolved_time == value, (text, proposal.resolved_time)
    finally:
        store.close()
        os.unlink(path)


def test_6_asia_kolkata_round_trip_through_utc_persistence():
    store, service, path = _service()
    try:
        message = _message("Meeting today at 6 PM")
        proposal = service._event_proposal(message)
        event = service._apply_proposal(message, proposal)
        assert event is not None
        assert event.time == "18:00"
        assert event.timezone == "Asia/Kolkata"
        persisted = store.get_event(event.event_id)
        assert persisted is not None
        round_trip = datetime.fromtimestamp(persisted.event_time, tz=IST)
        assert round_trip.hour == 18
        assert round_trip.minute == 0
    finally:
        store.close()
        os.unlink(path)


def test_7_source_message_timestamp_does_not_change_explicit_event_wall_clock():
    store, service, path = _service()
    try:
        message = _message("Meeting at 6 PM", timestamp="2026-09-18T19:18:00+00:00")
        proposal = service._event_proposal(message)
        assert proposal.resolved_time == "18:00"
    finally:
        store.close()
        os.unlink(path)


def test_8_no_constant_plus_7_hour_shift():
    store, service, path = _service()
    try:
        p6 = service._event_proposal(_message("Meeting at 6 PM"))
        p10 = service._event_proposal(_message("Meeting at 10 PM"))
        assert p6.resolved_time == "18:00"
        assert p10.resolved_time == "22:00"
        assert p10.resolved_time != "05:00"
    finally:
        store.close()
        os.unlink(path)


def test_9_relative_expression_uses_local_event_timezone():
    store, service, path = _service()
    try:
        message = _message("Meeting tomorrow at 6 PM", timestamp="2026-09-18T19:18:00+00:00")
        proposal = service._event_proposal(message)
        event = service._apply_proposal(message, proposal)
        assert event is not None
        assert event.date == "2026-09-20"
        assert event.time == "18:00"
        displayed = datetime.fromtimestamp(event.event_time, tz=IST)
        assert displayed.date().isoformat() == "2026-09-20"
        assert displayed.strftime("%H:%M") == "18:00"
    finally:
        store.close()
        os.unlink(path)


def test_10_event_correction_repairs_existing_buggy_event_without_duplicate():
    store, service, path = _service()
    try:
        message = _message("Meeting at 6 PM", timestamp="2026-09-18T19:18:00+00:00")
        buggy_dt = datetime(2026, 9, 20, 1, 0, tzinfo=IST)
        buggy = service._event(
            message,
            IntelligenceEventType.MEETING,
            "Meeting",
            message.text,
            "PROPOSED",
            0.86,
            event_time=buggy_dt.timestamp(),
            date="2026-09-20",
            time_of_day="01:00",
            timezone_name="Asia/Kolkata",
            scheduler_candidate=True,
        )
        store.upsert_event(buggy)
        store.add_event_evidence(buggy.event_id, message, "MEETING_CREATED", "01:00 Meeting at 6 PM", 0.86)

        repaired = store.repair_meeting_event_times()
        assert repaired == 1
        events = store.list_events()
        assert len(events) == 1
        assert events[0].event_id == buggy.event_id
        assert events[0].time == "18:00"
        assert datetime.fromtimestamp(events[0].event_time, tz=IST).strftime("%H:%M") == "18:00"
    finally:
        store.close()
        os.unlink(path)


def test_11_debug_instrumentation_exposes_extraction_boundary_without_message_body():
    lines = []
    store, service, path = _service(lines.append)
    try:
        proposal = service._event_proposal(_message("01:00 Meeting at 6 PM"))
        assert proposal.resolved_time == "18:00"
        assert any(
            line.startswith("WHATSAPP_TIME_DEBUG:")
            and "raw_time_expression='6 PM'" in line
            and "parsed_hour=18" in line
            and "parsed_minute=0" in line
            and "parsed_meridiem=PM" in line
            and "source_message_timezone=UTC" in line
            and "event_timezone=Asia/Kolkata" in line
            and "normalized_event_time=18:00" in line
            and "Meeting at" not in line
            for line in lines
        )
    finally:
        store.close()
        os.unlink(path)
