from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
import sqlite3
import pytest

from agent_control.whatsapp_intelligence import (
    IntelligenceEvent,
    IntelligenceEventType,
    WHATSAPP_EVENT_LOOKBACK_DAYS,
    WhatsAppIntelligence,
    WhatsAppIntelligenceStore,
    WhatsAppMessage,
    extract_messages,
)


def msg(mid, text, ts, chat="mummy", sender="Alice", **extra):
    return {"message_id": mid, "conversation_id": chat, "timestamp": ts, "sender": sender, "text": text, **extra}


def service(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    svc = WhatsAppIntelligence(store)
    return store, svc


def test_same_message_has_stable_fallback_identity_across_observations():
    raw1 = {"messages": [msg(None, "meeting at 6 PM", 1_700_000.0)]}
    raw2 = {"messages": [msg(None, "meeting at 6 PM", 1_700_000.0, sender="alice")]}
    a = extract_messages(raw1)[0].message_id
    b = extract_messages(raw2)[0].message_id
    assert a == b


def test_different_chats_and_timestamps_do_not_collide():
    a = extract_messages({"messages": [msg(None, "same", 1_700_000.0, chat="a")]})[0].message_id
    b = extract_messages({"messages": [msg(None, "same", 1_700_000.0, chat="b")]})[0].message_id
    c = extract_messages({"messages": [msg(None, "same", 1_700_001.0, chat="a")]})[0].message_id
    assert len({a, b, c}) == 3


def test_repeated_message_is_processed_once(tmp_path):
    store, svc = service(tmp_path)
    now = 2_000_000.0
    svc.enable(now)
    observation = {"messages": [msg("m1", "There will be a meeting at 6 PM.", now)]}
    assert len(svc.process_observation(observation, observed_at=now)) == 1
    assert svc.process_observation(observation, observed_at=now + 1) == []
    assert len(store.list_events()) == 1


def test_five_day_lookback_and_persistent_event():
    store, svc = service(Path(__import__("tempfile").mkdtemp()))
    now = datetime(2026, 9, 18, 12, tzinfo=timezone.utc).timestamp()
    svc.enable(now)
    inside = now - (WHATSAPP_EVENT_LOOKBACK_DAYS * 86400) + 1
    old = now - (WHATSAPP_EVENT_LOOKBACK_DAYS * 86400) - 1
    assert svc.process_observation({"messages": [msg("in", "meeting at 6 PM", inside)]}, observed_at=now)
    assert svc.process_observation({"messages": [msg("old", "meeting at 7 PM", old)]}, observed_at=now) == []
    assert len(store.list_events()) == 1


def test_read_state_does_not_affect_eligibility(tmp_path):
    store, svc = service(tmp_path)
    now = 2_000_000.0
    svc.enable(now)
    event = svc.process_observation({"messages": [msg("m1", "meeting at 6 PM", now, read=True, unread=False)]}, observed_at=now)
    assert event


def test_direction_is_preserved_for_incoming_outgoing_and_self_chat():
    incoming = extract_messages({"messages": [msg("i", "hello", 1_700_000, sender="Alice", from_me=False)]})[0]
    outgoing = extract_messages({"messages": [msg("o", "hello", 1_700_001, sender="You", from_me=True)]})[0]
    self_chat = extract_messages({"messages": [msg("s", "hello", 1_700_002, chat="Saksham", sender="You", from_me=True)]})[0]
    assert incoming.metadata["direction"] == "incoming"
    assert outgoing.metadata["direction"] == "outgoing"
    assert self_chat.metadata["direction"] == "outgoing"


def test_meeting_followups_mutate_one_event_with_auditable_evidence(tmp_path):
    store, svc = service(tmp_path)
    base = datetime(2026, 9, 14, 12, tzinfo=timezone.utc).timestamp()
    svc.enable(base)
    texts = [
        "There will be a DEIMOS test meeting at 6 PM.",
        "Okay Wednesday at 6 then.",
        "It's on Discord.",
        "Actually 7 baje kar di.",
        "Confirmed.",
        "Meeting cancelled.",
    ]
    for i, text in enumerate(texts):
        ts = base + i * 3600
        svc.process_observation({"messages": [msg(str(i), text, ts)]}, observed_at=ts)
    events = store.list_events()
    assert len(events) == 1
    event = events[0]
    assert event.type is IntelligenceEventType.MEETING
    assert event.date == "2026-09-16"
    assert event.time == "19:00"
    assert event.platform == "Discord"
    assert event.status == "CANCELLED"
    assert event.source_count == 6
    evidence = store.event_evidence(event.event_id)
    assert [row["message_id"] for row in evidence] == [str(i) for i in range(6)]
    assert {row["evidence_type"] for row in evidence} >= {"MEETING_CREATED", "MEETING_DATE_ADDED", "MEETING_PLATFORM_ADDED", "MEETING_TIME_CHANGED", "MEETING_CONFIRMED", "MEETING_CANCELLED"}


def test_ambiguous_generic_time_correction_does_not_pick_arbitrarily(tmp_path):
    store, svc = service(tmp_path)
    now = 2_000_000.0
    svc.enable(now)
    svc.process_observation({"messages": [msg("a", "meeting at 6 PM", now)]}, observed_at=now)
    svc.process_observation({"messages": [msg("b", "meeting at 8 PM", now + 10)]}, observed_at=now + 10)
    svc.process_observation({"messages": [msg("c", "Actually move it to 7", now + 20)]}, observed_at=now + 20)
    events = sorted(store.list_events(), key=lambda e: e.time or "")
    assert len(events) == 2
    assert {e.time for e in events} == {"18:00", "20:00"}


def test_event_survives_restart_and_later_message_updates_it(tmp_path):
    path = tmp_path / "wa.sqlite3"
    base = datetime(2026, 9, 14, 12, tzinfo=timezone.utc).timestamp()
    first = WhatsAppIntelligence(WhatsAppIntelligenceStore(path))
    first.enable(base)
    first.process_observation({"messages": [msg("m1", "meeting at 6 PM", base)]}, observed_at=base)
    first.store.close()

    second_store = WhatsAppIntelligenceStore(path)
    second = WhatsAppIntelligence(second_store)
    second.process_observation({"messages": [msg("m2", "Okay Wednesday at 6 then.", base + 3600)]}, observed_at=base + 3600)
    events = second_store.list_events()
    assert len(events) == 1
    assert events[0].date == "2026-09-16"


def _meeting_message(message_id="msg-db-1", timestamp=2_000_000.0):
    return {
        "message_id": message_id,
        "conversation_id": "mummy",
        "timestamp": timestamp,
        "text": "Meeting at 6pm",
        "sender": "Mummy",
        "from_me": False,
    }


def test_atomic_event_persistence_marks_processed_only_after_commit(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    service = WhatsAppIntelligence(store)
    now = 2_000_000.0
    service.enable(now=now)
    msg = _meeting_message()
    logs = []
    service.on_debug = logs.append
    store.set_debug_callback(logs.append)

    def fail_always(event):
        raise sqlite3.OperationalError("database is locked")
    store.upsert_event = fail_always

    service.process_observation({"messages": [msg]}, observed_at=now)
    assert store._conn().execute("select count(*) from processed_messages").fetchone()[0] == 0
    assert store.list_events() == []
    assert any("WHATSAPP_DB:" in line and "database is locked" in line for line in logs)
    assert not any("WHATSAPP_EVENT: PERSISTED" in line for line in logs)


def test_atomic_retry_after_database_failure_persists_event_and_processed_message(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    service = WhatsAppIntelligence(store)
    now = 2_000_000.0
    service.enable(now=now)
    msg = _meeting_message()
    original = store.upsert_event
    state = {"failed": False}
    def fail_once(event):
        if not state["failed"]:
            state["failed"] = True
            raise sqlite3.OperationalError("database is locked")
        return original(event)
    store.upsert_event = fail_once
    service.process_observation({"messages": [msg]}, observed_at=now)
    # A transient busy failure is retried inside the same atomic operation.
    assert store._conn().execute("select count(*) from processed_messages").fetchone()[0] == 1
    assert len(store.list_events()) == 1
    service.process_observation({"messages": [msg]}, observed_at=now)
    assert store._conn().execute("select count(*) from processed_messages").fetchone()[0] == 1
    assert len(store.list_events()) == 1
    service.process_observation({"messages": [msg]}, observed_at=now)
    assert store._conn().execute("select count(*) from processed_messages").fetchone()[0] == 1


def test_duplicate_processed_message_is_rejected_by_primary_key(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    message = WhatsAppMessage("msg-dup", "mummy", 2_000_000.0, "hello")
    assert store.claim_message(message, True) is True
    assert store.claim_message(message, True) is False
    with pytest.raises(sqlite3.IntegrityError):
        with store._write_transaction("duplicate_processed_message"):
            store._conn().execute(
                "insert into processed_messages(message_id, conversation_id, message_timestamp, processed_at, eligible) values(?,?,?,?,?)",
                (message.message_id, message.conversation_id, message.timestamp, 2_000_001.0, 1),
            )


def test_duplicate_event_evidence_is_rejected_idempotently(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    msg = WhatsAppMessage("msg-evi", "mummy", 2_000_000.0, "Meeting at 6pm")
    event = IntelligenceEvent(
        event_id="wae-evidence", conversation_id="mummy", type=IntelligenceEventType.MEETING,
        title="Meeting", description=msg.text, status="PROPOSED", confidence=.86,
        importance=.8, urgency=.4, created_at=2_000_000.0, updated_at=2_000_000.0,
    )
    store.upsert_event(event)
    assert store.add_event_evidence(event.event_id, msg, "MEETING_CREATED", msg.text, .86) is True
    assert store.add_event_evidence(event.event_id, msg, "MEETING_CREATED", msg.text, .86) is False
    assert len(store.event_evidence(event.event_id)) == 1


def test_identical_timestamps_have_distinct_stable_ids_and_home_chat_same_id():
    timestamp = 2_000_000.0
    home = {"messages": [_meeting_message("", timestamp)]}
    chat = {"messages": [_meeting_message("", timestamp)]}
    home_id = extract_messages(home)[0].message_id
    chat_id = extract_messages(chat, conversation_hint="mummy")[0].message_id
    assert home_id == chat_id
    second = extract_messages(home)[0].message_id
    assert second == home_id
    other = _meeting_message("", timestamp) | {"text": "Meeting at 7pm"}
    other_id = extract_messages({"messages": [other]})[0].message_id
    assert other_id != home_id


def test_direction_incoming_outgoing_and_unknown_are_preserved():
    incoming = extract_messages({"messages": [{**_meeting_message(), "from_me": False, "sender": "Mummy"}]})[0]
    outgoing = extract_messages({"messages": [{**_meeting_message("out", 2_000_001.0), "from_me": True, "sender": "You"}]})[0]
    unknown = extract_messages({"messages": [{"message_id": "unknown", "conversation_id": "mummy", "timestamp": 2_000_002.0, "text": "hello"}]})[0]
    assert incoming.metadata["direction"] == "incoming"
    assert outgoing.metadata["direction"] == "outgoing"
    assert unknown.metadata["direction"] == "unknown"


def test_self_chat_requires_message_level_outgoing_evidence():
    no_evidence = extract_messages({"elements": [
        {"role": "heading", "name": "You"},
        {"role": "textbox", "name": "Type a message"},
        {"role": "text", "name": "Meeting at 6pm"},
        {"role": "text", "name": "11:00 PM"},
        {"role": "button", "name": "Saksham"},
    ]}, conversation_hint="Saksham", observed_at=2_000_000.0)
    assert no_evidence
    assert no_evidence[0].metadata["direction"] == "unknown"
    explicit = extract_messages({"messages": [{**_meeting_message("self", 2_000_001.0), "conversation_id": "Saksham", "sender": "You", "from_me": True}]})
    assert explicit[0].metadata["direction"] == "outgoing"


def test_database_failure_does_not_break_home_row_processing(tmp_path):
    class Browser:
        def __init__(self):
            self.opened = False
        def open_whatsapp_chat_row(self, target, timeout_s=8.0):
            self.opened = True
            return type("R", (), {"ok": True})()
        def observe(self):
            return {"messages": [_meeting_message()]}
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    store.authorize_target("mummy")
    service = WhatsAppIntelligence(store)
    service.enable(now=2_000_000.0)
    original = service.process_observation
    service.process_observation = lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked"))
    browser = Browser()
    home = {"elements": [{"role": "button", "name": "mummy Meeting at 6pm 11:00 PM"}]}
    # The navigation path itself remains successful; persistence errors are
    # handled by the observer layer rather than being coupled to row opening.
    service._observe_authorized_targets(browser, home, allow_scroll=False)
    assert browser.opened is True
    service.process_observation = original


def test_observer_continues_after_one_message_database_failure(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    service = WhatsAppIntelligence(store)
    now = 2_000_000.0
    service.enable(now=now)
    first = _meeting_message("first", now)
    second = {**_meeting_message("second", now + 1), "text": "Another meeting at 7pm"}
    original = store.upsert_event
    failed = {"done": False}
    def fail_first(event):
        if "first" in event.source_message_ids:
            failed["done"] = True
            raise sqlite3.OperationalError("database is locked")
        return original(event)
    store.upsert_event = fail_first
    service.process_observation({"messages": [first, second]}, observed_at=now + 1)
    rows = store._conn().execute("select message_id from processed_messages order by message_id").fetchall()
    assert {r[0] for r in rows} == {"second"}
    # The failed message is still eligible for a later cycle. Restore the
    # normal writer before the retry.
    store.upsert_event = original
    service.process_observation({"messages": [first]}, observed_at=now + 2)
    assert store._conn().execute("select count(*) from processed_messages where message_id='first'").fetchone()[0] == 1


def test_five_day_cutoff_is_unchanged(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    service = WhatsAppIntelligence(store)
    now = 2_000_000.0
    service.enable(now=now)
    inside = _meeting_message("inside", now - 4 * 24 * 3600)
    outside = _meeting_message("outside", now - 6 * 24 * 3600)
    service.process_observation({"messages": [inside, outside]}, observed_at=now)
    ids = {r[0] for r in store._conn().execute("select message_id from processed_messages")}
    assert ids == {"inside", "outside"}
    inside_row = store._conn().execute("select eligible from processed_messages where message_id='inside'").fetchone()
    outside_row = store._conn().execute("select eligible from processed_messages where message_id='outside'").fetchone()
    assert inside_row[0] == 1
    assert outside_row[0] == 0


def test_event_persisted_debug_is_emitted_only_after_commit(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    logs = []
    service = WhatsAppIntelligence(store, on_debug=logs.append)
    now = 2_000_000.0
    service.enable(now=now)
    service.process_observation({"messages": [_meeting_message()]}, observed_at=now)
    persisted = [line for line in logs if line.startswith("WHATSAPP_EVENT: PERSISTED")]
    assert len(persisted) == 1
    event_id = persisted[0].split("event=", 1)[1].split()[0]
    assert store.get_event(event_id) is not None
    assert store._conn().execute("select count(*) from event_evidence where event_id=?", (event_id,)).fetchone()[0] == 1


def test_real_sqlite_busy_contention_is_bounded_and_retried(tmp_path):
    import threading
    import time as time_module
    db = tmp_path / "wa.sqlite3"
    store = WhatsAppIntelligenceStore(db)
    service = WhatsAppIntelligence(store)
    now = 2_000_000.0
    service.enable(now=now)
    blocker = sqlite3.connect(db, timeout=0.1, check_same_thread=False)
    blocker.execute("BEGIN IMMEDIATE")
    release = threading.Event()

    def unlock():
        release.wait(0.15)
        blocker.rollback()
        blocker.close()

    thread = threading.Thread(target=unlock)
    thread.start()
    started = time_module.monotonic()
    service.process_observation({"messages": [_meeting_message()]}, observed_at=now)
    elapsed = time_module.monotonic() - started
    thread.join(timeout=2)
    assert elapsed < 2.0
    assert store._conn().execute("select count(*) from processed_messages").fetchone()[0] == 1
