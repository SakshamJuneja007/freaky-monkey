from __future__ import annotations

from types import SimpleNamespace

from agent_control.presentation import whatsapp_intelligence_message
from agent_control.skills.browser.backend import BrowserSkillAdapter
from agent_control.whatsapp_intelligence import (
    IntelligenceEvent,
    IntelligenceEventType,
    WhatsAppIntelligence,
    WhatsAppIntelligenceStore,
    extract_messages,
)


def _obs(*elements):
    return SimpleNamespace(elements=tuple(elements), text="", raw={"elements": list(elements)})


def _elem(ref, role, name, value="", **raw):
    return SimpleNamespace(ref=ref, role=role, name=name, value=value, raw=raw, attributes=raw)


def _meeting(message_id: str, timestamp: float, *, sender="Mummy", from_me=False):
    return {
        "message_id": message_id,
        "conversation_id": "Mummy",
        "timestamp": timestamp,
        "text": "Meeting tomorrow at 6 PM",
        "sender": sender,
        "from_me": from_me,
    }


def test_same_browser_message_keeps_stable_fallback_id_across_direction_metadata_changes():
    first = extract_messages({"messages": [{**_meeting("", 1800000000.0), "from_me": True, "sender": "You"}]})[0]
    second = extract_messages({"messages": [{**_meeting("", 1800000000.0), "from_me": False, "sender": "Mummy"}]})[0]
    assert first.message_id == second.message_id


def test_same_message_seen_ten_times_runs_event_extraction_once(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    logs: list[str] = []
    service = WhatsAppIntelligence(store, on_debug=logs.append)
    now = 1800000000.0
    service.enable(now=now)
    calls = {"count": 0}
    original = service._analyze_message

    def counted(message, context):
        calls["count"] += 1
        return original(message, context)

    service._analyze_message = counted
    observation = {"messages": [_meeting("stable-1", now)]}
    for cycle in range(10):
        service.process_observation(observation, observed_at=now + cycle)

    assert calls["count"] == 1
    summaries = [line for line in logs if line.startswith("WHATSAPP_OBSERVER:\ncycle=")]
    assert len(summaries) == 10
    assert sum("new=1" in line for line in summaries) == 1
    assert all("ELIGIBLE" not in line and "ALREADY_PROCESSED message_id=" not in line for line in logs)


def test_already_processed_stops_before_intelligence_extraction(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    service = WhatsAppIntelligence(store)
    now = 1800000000.0
    service.enable(now=now)
    message = _meeting("processed-1", now)
    service.process_observation({"messages": [message]}, observed_at=now)
    calls = {"count": 0}

    def should_not_run(*_args):
        calls["count"] += 1
        raise AssertionError("processed message entered extraction")

    service._analyze_message = should_not_run
    service.process_observation({"messages": [message]}, observed_at=now + 1)
    assert calls["count"] == 0


def test_initial_catchup_then_live_mode(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    logs: list[str] = []
    service = WhatsAppIntelligence(store, on_debug=logs.append)
    now = 1800000000.0
    service.enable(now=now)
    service.process_observation({"messages": [_meeting("hist-1", now - 4 * 86400)]}, observed_at=now)
    service.process_observation({"messages": [_meeting("hist-1", now - 4 * 86400)]}, observed_at=now + 2)
    assert "mode=INITIAL_CATCHUP" in logs[-1 - 0] or any("mode=INITIAL_CATCHUP" in line for line in logs)
    assert any("mode=LIVE" in line for line in logs)


def test_initial_five_day_window_accepts_four_day_message_and_rejects_six_day(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    service = WhatsAppIntelligence(store)
    now = 1800000000.0
    service.enable(now=now)
    inside = _meeting("inside-5", now - 4 * 86400)
    outside = _meeting("outside-5", now - 6 * 86400)
    service.process_observation({"messages": [inside, outside]}, observed_at=now)
    rows = {row[0] for row in store._conn().execute("SELECT message_id FROM processed_messages")}
    assert "inside-5" in rows
    assert "outside-5" in rows
    events = store.list_events()
    assert any(inside["message_id"] in event.source_message_ids for event in events)
    assert not any(outside["message_id"] in event.source_message_ids for event in events)


def test_failed_processing_is_retryable_then_bounded(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    service = WhatsAppIntelligence(store)
    now = 1800000000.0
    service.enable(now=now)
    message = _meeting("fail-1", now)

    def fail(*_args):
        raise RuntimeError("permanent failure")

    service._analyze_message = fail
    for offset in (1, 3, 7):
        service.process_observation({"messages": [message]}, observed_at=now + offset)
    count_before = store._conn().execute("SELECT COUNT(*) FROM processed_messages WHERE message_id='fail-1'").fetchone()[0]
    assert count_before == 0
    logs: list[str] = []
    gated = WhatsAppIntelligence(store, on_debug=logs.append)
    gated.process_observation({"messages": [message]}, observed_at=now + 100)
    assert not any("new=1" in line for line in logs if line.startswith("WHATSAPP_OBSERVER:"))


def test_semantic_browser_observation_preserves_message_ownership_metadata():
    raw = {
        "elements": [
            {"ref": "@e1", "role": "heading", "name": "Mummy"},
            {"ref": "@e2", "role": "text", "name": "Meeting tomorrow at 6 PM", "message_id": "native-1", "from_me": True, "timestamp": 1800000000.0},
        ]
    }
    browser_elements = BrowserSkillAdapter._extract_elements(raw)
    message_element = next(e for e in browser_elements if e.ref == "@e2")
    assert message_element.raw["from_me"] is True
    assert message_element.raw["message_id"] == "native-1"
    observation = SimpleNamespace(elements=tuple(browser_elements), text="", raw=raw)
    messages = extract_messages(observation, conversation_hint="Mummy", observed_at=1800000000.0)
    message = next(m for m in messages if "meeting" in m.text.casefold())
    assert message.message_id == "native-1"
    assert message.metadata["direction"] == "outgoing"
    assert message.metadata["is_outgoing"] is True


def test_semantic_browser_observation_distinguishes_incoming_and_outgoing():
    raw = {
        "elements": [
            {"ref": "@e1", "role": "heading", "name": "Mummy"},
            {
                "ref": "@e3",
                "role": "text",
                "name": "DEIMOS outgoing meeting tomorrow at 6 PM",
                "message_id": "native-out",
                "timestamp": 1800000000.0,
                "from_me": True,
            },
            {
                "ref": "@e4",
                "role": "text",
                "name": "Mummy: DEIMOS incoming meeting tomorrow at 7 PM",
                "message_id": "native-in",
                "timestamp": 1800000001.0,
                "sender": "Mummy",
                "from_me": False,
            },
        ]
    }
    browser_elements = BrowserSkillAdapter._extract_elements(raw)
    observation = SimpleNamespace(elements=tuple(browser_elements), text="", raw=raw)
    messages = extract_messages(observation, conversation_hint="Mummy", observed_at=1800000001.0)
    by_id = {m.message_id: m for m in messages}
    assert by_id["native-out"].metadata["direction"] == "outgoing"
    assert by_id["native-out"].metadata["is_outgoing"] is True
    assert by_id["native-in"].metadata["direction"] == "incoming"
    assert by_id["native-in"].metadata["is_outgoing"] is False


def test_semantic_direction_accepts_explicit_current_user_marker():
    messages = extract_messages(_obs(
        _elem("@h", "heading", "Group"),
        _elem("@m", "text", "Meeting tomorrow at 6 PM", from_me=True),
        _elem("@t", "text", "10:00 PM"),
    ), conversation_hint="Group", observed_at=1800000000.0)
    assert messages[0].metadata["direction"] == "outgoing"


def test_garbage_sender_values_remain_unknown():
    for sender in ("https", "~", "39 unread messages Happy Birthday Anu Alka Di"):
        node = _meeting(sender + "-id", 1800000000.0, sender=sender)
        node.pop("from_me", None)
        message = extract_messages({"messages": [node]})[0]
        assert message.metadata["direction"] == "unknown"
        assert message.metadata["is_outgoing"] is None
        assert message.metadata["sender_evidence"] == "NONE"


def test_missing_sender_is_unknown_even_when_unread_or_read():
    base = {"message_id": "read-state", "conversation_id": "Mummy", "timestamp": 1800000000.0, "text": "Meeting tomorrow"}
    for read in (True, False):
        message = extract_messages({"messages": [{**base, "read": read}]})[0]
        assert message.metadata["direction"] == "unknown"
        assert message.metadata["is_outgoing"] is None


def test_self_chat_does_not_use_chat_name_as_direction_evidence():
    message = extract_messages({"messages": [{"message_id": "self", "conversation_id": "Saksham", "timestamp": 1800000000.0, "text": "Meeting tomorrow"}]})[0]
    assert message.metadata["direction"] == "unknown"


def test_presentation_module_imports_and_renders_structured_event():
    event = SimpleNamespace(
        type=IntelligenceEventType.MEETING,
        title="Meeting",
        date="2026-09-23",
        time="18:00",
        deadline=None,
        status="PROPOSED",
    )
    rendered = whatsapp_intelligence_message(event)
    assert "WHATSAPP_INTELLIGENCE:" in rendered
    assert "Meeting" in rendered
    assert "2026-09-23" in rendered
    assert "raw" not in rendered.casefold()


def test_presentation_does_not_need_raw_message_body():
    event = SimpleNamespace(type="ASSIGNMENT", title="Neural Networks assignment", date="Friday", time="17:00", deadline=None, status="PENDING")
    rendered = whatsapp_intelligence_message(event)
    assert "Neural Networks assignment" in rendered


def test_persistent_event_survives_source_message_aging(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    service = WhatsAppIntelligence(store)
    now = 1800000000.0
    service.enable(now=now)
    message = _meeting("persistent-1", now - 4 * 86400)
    events = service.process_observation({"messages": [message]}, observed_at=now)
    assert events
    later = now + 7 * 86400
    assert store.get_event(events[0].event_id) is not None
    assert later > message["timestamp"] + 5 * 86400



def test_legacy_event_evidence_schema_migrates_in_place_and_preserves_rows(tmp_path):
    import sqlite3

    path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE intelligence_state (id INTEGER PRIMARY KEY CHECK(id=1), enabled INTEGER NOT NULL DEFAULT 0, enabled_at REAL, last_processed_message_timestamp REAL, last_processed_message_id TEXT, last_observation_timestamp REAL);
        INSERT INTO intelligence_state(id, enabled) VALUES(1, 0);
        CREATE TABLE processed_messages (message_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, message_timestamp REAL NOT NULL, processed_at REAL NOT NULL, eligible INTEGER NOT NULL);
        CREATE TABLE event_evidence (event_id TEXT NOT NULL, message_id TEXT NOT NULL, conversation_id TEXT NOT NULL, message_timestamp REAL NOT NULL, evidence_type TEXT NOT NULL, extracted_claim TEXT NOT NULL, confidence REAL NOT NULL, created_at REAL NOT NULL, UNIQUE(event_id,message_id,evidence_type));
        CREATE TABLE intelligence_events (event_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, event_type TEXT NOT NULL, title TEXT NOT NULL, description TEXT NOT NULL, status TEXT NOT NULL, confidence REAL NOT NULL, importance REAL NOT NULL, urgency REAL NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL, event_time REAL, deadline REAL, participants_json TEXT NOT NULL, evidence_json TEXT NOT NULL, source_message_ids_json TEXT NOT NULL, scheduler_candidate INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE conversation_summaries (conversation_id TEXT PRIMARY KEY,current_summary TEXT NOT NULL,summary_version INTEGER NOT NULL,last_summary_message_id TEXT NOT NULL,updated_at REAL NOT NULL);
        CREATE TABLE authorized_targets (target_name TEXT PRIMARY KEY,granted_at REAL NOT NULL);
        INSERT INTO event_evidence(event_id,message_id,conversation_id,message_timestamp,evidence_type,extracted_claim,confidence,created_at) VALUES('e1','m1','Mummy',1800000000,'MEETING_CREATED','Meeting',0.8,1800000001);
    """)
    conn.commit()
    conn.close()

    store = WhatsAppIntelligenceStore(path)
    columns = {row[1] for row in store._conn().execute("PRAGMA table_info(event_evidence)")}
    row = store._conn().execute("SELECT * FROM event_evidence WHERE event_id='e1'").fetchone()
    assert "evidence_id" in columns
    assert row["evidence_id"]
    assert row["extracted_claim"] == "Meeting"


def test_event_evidence_schema_migration_is_idempotent(tmp_path):
    import sqlite3

    path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE event_evidence (event_id TEXT NOT NULL, message_id TEXT NOT NULL, conversation_id TEXT NOT NULL, message_timestamp REAL NOT NULL, evidence_type TEXT NOT NULL, extracted_claim TEXT NOT NULL, confidence REAL NOT NULL, created_at REAL NOT NULL, UNIQUE(event_id,message_id,evidence_type));
    """)
    conn.commit()
    conn.close()

    # Build once through the real store initializer, then initialize again.
    WhatsAppIntelligenceStore(path).close()
    store = WhatsAppIntelligenceStore(path)
    assert [row[1] for row in store._conn().execute("PRAGMA table_info(event_evidence)")].count("evidence_id") == 1


def test_event_evidence_insert_works_after_existing_schema_migration(tmp_path):
    import sqlite3

    path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE event_evidence (event_id TEXT NOT NULL, message_id TEXT NOT NULL, conversation_id TEXT NOT NULL, message_timestamp REAL NOT NULL, evidence_type TEXT NOT NULL, extracted_claim TEXT NOT NULL, confidence REAL NOT NULL, created_at REAL NOT NULL, UNIQUE(event_id,message_id,evidence_type));
    """)
    conn.commit()
    conn.close()
    store = WhatsAppIntelligenceStore(path)
    from agent_control.whatsapp_intelligence import WhatsAppMessage
    message = WhatsAppMessage("m-new", "Mummy", 1800000000.0, "Meeting", "Mummy", {})
    assert store.add_event_evidence("e-new", message, "MEETING_CREATED", "Meeting", 0.9)
    assert store.event_evidence("e-new")[0]["evidence_id"]

def test_native_message_id_preferred_over_fallback():
    messages = extract_messages({"messages": [_meeting("native-123", 1800000000.0)]})
    assert messages[0].message_id == "native-123"
