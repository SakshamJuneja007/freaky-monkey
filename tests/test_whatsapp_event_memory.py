import time
from datetime import datetime, timezone

from agent_control.whatsapp_intelligence import (
    IntelligenceEventType,
    WhatsAppIntelligence,
    WhatsAppIntelligenceStore,
)


def service(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    svc = WhatsAppIntelligence(store)
    return svc, store


def msg(mid, text, ts, chat="mummy", sender="Mummy"):
    return {"message_id": mid, "conversation_id": chat, "timestamp": ts, "text": text, "sender": sender}


def feed(svc, message):
    ts = message["timestamp"]
    return svc.process_observation({"messages": [message]}, observed_at=ts)


def test_partial_meeting_creation_and_unknown_date(tmp_path):
    svc, store = service(tmp_path)
    now = datetime(2026, 9, 14, 10, 0, tzinfo=timezone.utc).timestamp()
    svc.enable(now=now)
    feed(svc, msg("m1", "There will be a meeting at 6 PM.", now))
    events = store.list_events()
    assert len(events) == 1
    assert events[0].type is IntelligenceEventType.MEETING
    assert events[0].event_date is None
    assert events[0].event_time_of_day == "18:00"
    assert events[0].status == "PROPOSED"


def test_followups_update_one_event(tmp_path):
    svc, store = service(tmp_path)
    base = datetime(2026, 9, 14, 10, 0, tzinfo=timezone.utc).timestamp()
    svc.enable(now=base)
    feed(svc, msg("m1", "There will be a meeting at 6 PM.", base))
    feed(svc, msg("m2", "Okay Wednesday at 6 then.", base + 3600))
    feed(svc, msg("m3", "It's on Discord.", base + 7200))
    feed(svc, msg("m4", "Actually 7 baje kar di.", base + 10800))
    feed(svc, msg("m5", "Confirmed.", base + 14400))
    feed(svc, msg("m6", "Meeting cancel hai.", base + 18000))
    events = store.list_events()
    assert len(events) == 1
    event = events[0]
    assert event.event_date == "2026-09-16"
    assert event.event_time_of_day == "19:00"
    assert event.platform == "Discord"
    assert event.status == "CANCELLED"
    assert event.source_message_ids == ("m1", "m2", "m3", "m4", "m5", "m6")


def test_relative_tomorrow_uses_message_timestamp(tmp_path):
    svc, store = service(tmp_path)
    ts = datetime(2026, 9, 14, 15, 0, tzinfo=timezone.utc).timestamp()
    svc.enable(now=ts)
    feed(svc, msg("m1", "Tomorrow at 6", ts))
    event = store.list_events()[0]
    assert event.event_date == "2026-09-15"
    assert event.event_time_of_day == "06:00"
    assert event.event_time == datetime(2026, 9, 15, 6, 0, tzinfo=timezone.utc).timestamp()


def test_five_day_window_uses_message_timestamp(tmp_path):
    svc, store = service(tmp_path)
    now = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc).timestamp()
    svc.enable(now=now)
    old = now - 5 * 86400 - 1
    fresh = now - 5 * 86400 + 1
    feed(svc, msg("old", "There will be a meeting at 6 PM.", old))
    feed(svc, msg("new", "There will be a meeting at 6 PM.", fresh))
    assert len(store.list_events()) == 1
    assert store.list_events()[0].source_message_ids == ("new",)


def test_read_state_does_not_affect_eligibility(tmp_path):
    svc, store = service(tmp_path)
    now = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc).timestamp()
    svc.enable(now=now)
    result = feed(svc, {**msg("m1", "There will be a meeting at 6 PM.", now - 60), "metadata": {"unread": False}})
    assert result


def test_repeated_message_does_not_duplicate_event_or_evidence(tmp_path):
    svc, store = service(tmp_path)
    now = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc).timestamp()
    svc.enable(now=now)
    m = msg("m1", "There will be a meeting at 6 PM.", now)
    feed(svc, m)
    feed(svc, m)
    assert len(store.list_events()) == 1
    row = store._conn().execute("SELECT COUNT(*) FROM event_evidence").fetchone()
    assert row[0] == 1


def test_evidence_and_mutation_history_are_retained(tmp_path):
    svc, store = service(tmp_path)
    now = datetime(2026, 9, 14, 10, 0, tzinfo=timezone.utc).timestamp()
    svc.enable(now=now)
    feed(svc, msg("m1", "There will be a meeting at 6 PM.", now))
    feed(svc, msg("m2", "Okay Wednesday at 6 then.", now + 3600))
    feed(svc, msg("m3", "Actually 7 baje kar di.", now + 7200))
    rows = store._conn().execute("SELECT field_name, old_value, new_value FROM event_mutations ORDER BY created_at").fetchall()
    assert [(r[0], r[1], r[2]) for r in rows if r[0] == "event_time_of_day"] == [("event_time_of_day", "18:00", "19:00")]
    assert store._conn().execute("SELECT COUNT(*) FROM event_evidence").fetchone()[0] == 3


def test_ambiguous_update_does_not_mutate(tmp_path):
    svc, store = service(tmp_path)
    now = datetime(2026, 9, 14, 10, 0, tzinfo=timezone.utc).timestamp()
    svc.enable(now=now)
    feed(svc, msg("a1", "Meeting Wednesday at 6 PM", now))
    feed(svc, msg("b1", "Meeting Wednesday at 8 PM", now + 60))
    before = [(e.event_id, e.event_time_of_day, e.event_date) for e in store.list_events()]
    feed(svc, msg("u1", "Actually move it to 7", now + 120))
    after = [(e.event_id, e.event_time_of_day, e.event_date) for e in store.list_events()]
    assert before == after


def test_event_survives_beyond_five_day_source_window(tmp_path):
    svc, store = service(tmp_path)
    day1 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc).timestamp()
    svc.enable(now=day1)
    feed(svc, msg("m1", "There will be a meeting at 6 PM.", day1))
    assert len(store.list_events()) == 1
    day7 = day1 + 6 * 86400
    feed(svc, msg("m2", "It's on Discord.", day7))
    event = store.list_events()[0]
    assert event.platform == "Discord"


def test_status_confirmation_then_cancellation(tmp_path):
    svc, store = service(tmp_path)
    now = datetime(2026, 9, 18, 10, 0, tzinfo=timezone.utc).timestamp()
    svc.enable(now=now)
    feed(svc, msg("m1", "Meeting at 6 PM", now))
    feed(svc, msg("m2", "Confirmed", now + 60))
    assert store.list_events()[0].status == "CONFIRMED"
    feed(svc, msg("m3", "Meeting cancel hai", now + 120))
    assert store.list_events()[0].status == "CANCELLED"


def test_missing_timestamp_is_not_ingested(tmp_path):
    svc, store = service(tmp_path)
    now = datetime(2026, 9, 18, 10, 0, tzinfo=timezone.utc).timestamp()
    svc.enable(now=now)
    events = svc.process_observation({"messages": [{"message_id": "m1", "conversation_id": "mummy", "text": "Meeting at 6 PM"}]}, observed_at=now)
    assert events == []
    assert store.list_events() == []
