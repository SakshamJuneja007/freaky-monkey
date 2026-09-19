from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone

import pytest

from agent_control.whatsapp_intelligence import (
    IntelligenceEventType,
    SemanticMemoryProposal,
    WhatsAppIntelligence,
    WhatsAppIntelligenceStore,
    WhatsAppMessage,
    extract_messages,
)


def stamp(days_ago: int = 0, *, seconds_ago: int = 60) -> float:
    return time.time() - days_ago * 86400 - seconds_ago


def msg(mid: str, text: str, ts: float | None = None, *, chat: str = "mummy", sender: str = "Mummy", from_me: bool = False) -> WhatsAppMessage:
    ts = stamp() if ts is None else ts
    direction = "outgoing" if from_me else "incoming"
    return WhatsAppMessage(
        mid,
        chat,
        ts,
        text,
        "You" if from_me else sender,
        {
            "direction": direction,
            "is_outgoing": from_me,
            "sender_evidence": "from_me=True" if from_me else f"sender={sender}",
        },
    )


def raw_message(message: WhatsAppMessage) -> dict:
    return {
        "message_id": message.message_id,
        "conversation_id": message.conversation_id,
        "timestamp": message.timestamp,
        "body": message.text,
        "sender": message.sender,
        "from_me": message.metadata.get("is_outgoing"),
    }


def engine(tmp_path, *, can_observe=lambda: True, logs=None):
    logs = logs if logs is not None else []
    store = WhatsAppIntelligenceStore(tmp_path / "whatsapp.sqlite3")
    e = WhatsAppIntelligence(store, can_observe=can_observe, on_debug=logs.append)
    store.enable()
    return store, e, logs


def process(e, message: WhatsAppMessage, *, observed_at: float | None = None):
    return e.process_observation({"messages": [raw_message(message)]}, observed_at=observed_at or time.time(), conversation_hint=message.conversation_id)


def test_initial_enable_has_no_sync_checkpoint(tmp_path):
    store, e, _ = engine(tmp_path)
    assert store.sync_checkpoint("mummy") is None
    process(e, msg("m1", "Meeting tomorrow at 6 PM"))
    assert store.sync_checkpoint("mummy") is not None
    e.stop_observing()
    store.close()


def test_initial_catchup_is_bounded_to_five_days(tmp_path):
    store, e, logs = engine(tmp_path)
    old = msg("old", "Meeting tomorrow at 6 PM", stamp(6))
    recent = msg("recent", "Meeting tomorrow at 6 PM", stamp(4))
    events = e.process_observation({"messages": [raw_message(old), raw_message(recent)]}, observed_at=time.time(), conversation_hint="mummy")
    assert len(events) == 1
    assert store.get_event(events[0].event_id) is not None
    assert any("out_of_window=1" in line for line in logs if line.startswith("WHATSAPP_SYNC:"))
    store.close()


def test_restart_sync_processes_gap_not_full_history(tmp_path):
    path = tmp_path / "wa.sqlite3"
    first = WhatsAppIntelligenceStore(path)
    first.enable()
    e1 = WhatsAppIntelligence(first)
    base = time.time() - 3 * 86400
    process(e1, msg("mon", "Meeting tomorrow at 6 PM", base))
    cp = first.sync_checkpoint("mummy")
    first.close()

    second = WhatsAppIntelligenceStore(path)
    e2 = WhatsAppIntelligence(second)
    e2._observer_started_at = time.time()
    logs: list[str] = []
    e2.on_debug = logs.append
    second.enable()  # existing enabled state is preserved; checkpoint remains authoritative.
    historic = msg("mon", "Meeting tomorrow at 6 PM", base)
    new = msg("tue", "Neural Networks assignment due Friday at 5 PM", base + 86400)
    e2.process_observation({"messages": [raw_message(historic), raw_message(new)]}, observed_at=time.time(), conversation_hint="mummy")
    assert second.sync_checkpoint("mummy").last_source_message_id == "tue"
    assert any("mode=GAP" in line for line in logs if line.startswith("WHATSAPP_SYNC:"))
    assert sum(1 for e in second.list_events() if e.type is IntelligenceEventType.ASSIGNMENT) == 1
    second.close()


def test_restart_without_missing_data_does_not_reextract_history(tmp_path):
    path = tmp_path / "wa.sqlite3"
    s = WhatsAppIntelligenceStore(path)
    s.enable()
    e = WhatsAppIntelligence(s)
    m = msg("m1", "Meeting tomorrow at 6 PM")
    process(e, m)
    s.close()

    s2 = WhatsAppIntelligenceStore(path)
    e2 = WhatsAppIntelligence(s2)
    e2._observer_started_at = time.time()
    logs: list[str] = []
    e2.on_debug = logs.append
    process(e2, m)
    assert any("mode=GAP" in line for line in logs if line.startswith("WHATSAPP_SYNC:"))
    assert any("new=0" in line for line in logs if line.startswith("WHATSAPP_SYNC:"))
    assert sum(1 for line in logs if "WHATSAPP_EVENT: EXTRACTED" in line) == 0
    s2.close()


def test_live_mode_advances_only_for_new_source_information(tmp_path):
    store, e, logs = engine(tmp_path)
    first = msg("m1", "Meeting tomorrow at 6 PM")
    second = msg("m2", "Meeting tomorrow at 7 PM")
    process(e, first)
    process(e, first, observed_at=time.time() + 2)
    process(e, second, observed_at=time.time() + 4)
    sync_lines = [line for line in logs if line.startswith("WHATSAPP_SYNC:")]
    assert any("new=0" in line and "already_processed=1" in line for line in sync_lines)
    assert any("new=1" in line for line in sync_lines)
    assert len(store.list_events()) == 1
    store.close()


def test_same_source_observed_ten_times_creates_one_event(tmp_path):
    store, e, logs = engine(tmp_path)
    m = msg("stable", "Meeting tomorrow at 6 PM")
    for i in range(10):
        process(e, m, observed_at=time.time() + i)
    assert len(store.list_events()) == 1
    assert sum(1 for line in logs if "WHATSAPP_EVENT: EXTRACTED" in line) == 1
    assert sum(1 for line in logs if "new=1" in line) == 1
    store.close()


def test_processed_message_never_reaches_processor_again(tmp_path):
    store, e, _ = engine(tmp_path)
    m = msg("once", "Meeting tomorrow at 6 PM")
    calls = []
    original = e._analyze_message
    e._analyze_message = lambda *a, **k: (calls.append(1) or original(*a, **k))
    process(e, m)
    process(e, m, observed_at=time.time() + 5)
    assert len(calls) == 1
    store.close()


def test_native_message_id_is_stable_and_authoritative(tmp_path):
    t = stamp()
    a = extract_messages({"message_id": "native-42", "conversation_id": "mummy", "timestamp": t, "body": "hi"})[0]
    b = extract_messages({"message_id": "native-42", "conversation_id": "mummy", "timestamp": t, "body": "hi", "from_me": True})[0]
    assert a.message_id == b.message_id == "native-42"


def test_fallback_identity_is_stable_when_ownership_metadata_changes(tmp_path):
    t = stamp()
    a = extract_messages({"conversation_id": "mummy", "timestamp": t, "body": "meeting tomorrow", "sender": "Mummy"})[0]
    b = extract_messages({"conversation_id": "mummy", "timestamp": t, "body": "meeting tomorrow", "from_me": False})[0]
    assert a.message_id == b.message_id


def test_checkpoint_does_not_advance_when_processing_crashes(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    store.enable()
    m = msg("crash", "Meeting tomorrow at 6 PM")
    with pytest.raises(RuntimeError):
        store.process_message_atomically(m, True, lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert not store.is_message_processed(m.message_id)
    assert store.sync_checkpoint(m.conversation_id) is None
    store.close()


def test_failed_processing_remains_retryable(tmp_path):
    store, e, _ = engine(tmp_path)
    m = msg("retry", "Meeting tomorrow at 6 PM")
    state = {"fail": True}
    original = e._analyze_message
    def proc(message, context, **kwargs):
        if state["fail"]:
            raise RuntimeError("temporary")
        return original(message, context, **kwargs)
    e._analyze_message = proc
    e.process_observation({"messages": [raw_message(m)]}, observed_at=time.time(), conversation_hint="mummy")
    assert not store.is_message_processed("retry")
    failure = store._conn().execute("SELECT failure_count,retry_after FROM message_failures WHERE message_id='retry'").fetchone()
    assert failure is not None and failure[0] == 1
    store.close()


def test_failed_retry_is_bounded(tmp_path):
    store, e, logs = engine(tmp_path)
    m = msg("perma", "Meeting tomorrow at 6 PM")
    e._analyze_message = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("permanent"))
    base = time.time()
    for i in range(6):
        e.process_observation({"messages": [raw_message(m)]}, observed_at=base + i, conversation_hint="mummy")
    row = store._conn().execute("SELECT failure_count,retry_after FROM message_failures WHERE message_id='perma'").fetchone()
    assert row is not None and row[0] == 3
    assert not store.is_message_processed("perma")
    assert any("retry_deferred=true" in line for line in logs)
    store.close()


def test_successful_retry_clears_failure_and_advances_checkpoint(tmp_path):
    store, e, _ = engine(tmp_path)
    m = msg("retry-ok", "Meeting tomorrow at 6 PM")
    attempts = {"n": 0}
    original = e._analyze_message
    def proc(message, context, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("temporary")
        return original(message, context, **kwargs)
    e._analyze_message = proc
    base = time.time()
    e.process_observation({"messages": [raw_message(m)]}, observed_at=base, conversation_hint="mummy")
    e.process_observation({"messages": [raw_message(m)]}, observed_at=base + 2.0, conversation_hint="mummy")
    assert store.is_message_processed(m.message_id) is False or attempts["n"] >= 1
    e.process_observation({"messages": [raw_message(m)]}, observed_at=base + 3.0, conversation_hint="mummy")
    assert store.is_message_processed(m.message_id)
    assert store._conn().execute("SELECT 1 FROM message_failures WHERE message_id=?", (m.message_id,)).fetchone() is None
    store.close()


def test_meaningless_message_creates_no_semantic_memory(tmp_path):
    store, e, _ = engine(tmp_path)
    process(e, msg("noise", "haha"))
    assert store.list_events() == []
    assert store._conn().execute("SELECT COUNT(*) FROM semantic_memories").fetchone()[0] == 0
    store.close()


def test_meeting_creates_one_semantic_event(tmp_path):
    store, e, _ = engine(tmp_path)
    events = process(e, msg("mtg", "Meeting tomorrow at 6 PM"))
    assert len(events) == 1
    assert events[0].type is IntelligenceEventType.MEETING
    assert events[0].time == "18:00"
    assert events[0].date is not None
    assert events[0].status == "PROPOSED"
    store.close()


def test_partial_meeting_does_not_invent_date(tmp_path):
    store, e, _ = engine(tmp_path)
    events = process(e, msg("partial", "Meeting at 6 PM"))
    assert len(events) == 1
    assert events[0].time == "18:00"
    assert events[0].date is None
    store.close()


def test_related_meeting_update_merges_into_same_event(tmp_path):
    store, e, _ = engine(tmp_path)
    base = time.time()
    first = msg("m1", "Meeting tomorrow at 6 PM", base)
    second = msg("m2", "Actually Wednesday at 7 PM", base + 60)
    e.process_observation({"messages": [raw_message(first), raw_message(second)]}, observed_at=base + 60, conversation_hint="mummy")
    events = store.list_events()
    assert len(events) == 1
    assert events[0].time == "19:00"
    assert events[0].date is not None
    store.close()


def test_meeting_cancellation_updates_same_event(tmp_path):
    store, e, _ = engine(tmp_path)
    base = time.time()
    first = msg("m1", "Meeting tomorrow at 6 PM", base)
    cancel = msg("m2", "Meeting cancelled", base + 60)
    e.process_observation({"messages": [raw_message(first)]}, observed_at=base, conversation_hint="mummy")
    e.process_observation({"messages": [raw_message(cancel), raw_message(first)]}, observed_at=base + 60, conversation_hint="mummy")
    events = store.list_events()
    assert len(events) == 1
    assert events[0].status == "CANCELLED"
    store.close()


def test_assignment_extracts_structured_event(tmp_path):
    store, e, _ = engine(tmp_path)
    events = process(e, msg("a1", "Neural Networks assignment due Friday at 5 PM"))
    assert events and events[0].type is IntelligenceEventType.ASSIGNMENT
    assert "Neural Networks" in events[0].title
    assert events[0].time == "17:00"
    store.close()


def test_exam_month_date_is_parsed(tmp_path):
    store, e, _ = engine(tmp_path)
    future = msg("x1", "Mid-sem exams start October 14. Timetable will be shared soon.")
    events = process(e, future)
    assert events and events[0].type is IntelligenceEventType.EXAM
    assert events[0].date is not None and events[0].date.endswith("-10-14")
    store.close()


def test_timetable_change_is_supported(tmp_path):
    store, e, _ = engine(tmp_path)
    events = process(e, msg("t1", "Room changed tomorrow at 10 AM"))
    assert events and events[0].type is IntelligenceEventType.TIMETABLE_CHANGE
    store.close()


def test_college_plan_is_supported(tmp_path):
    store, e, _ = engine(tmp_path)
    events = process(e, msg("c1", "College tomorrow at 10 AM"))
    assert events and events[0].type is IntelligenceEventType.COLLEGE_PLAN
    assert events[0].time == "10:00"
    store.close()


def test_general_summary_today_is_allowed(tmp_path):
    store, e, _ = engine(tmp_path)
    now = time.time()
    m = msg("p1", "Let's use YOLO then.", now)
    process(e, m)
    assert store.summary("mummy") is not None
    store.close()


def test_general_summary_yesterday_is_allowed(tmp_path):
    store, e, _ = engine(tmp_path)
    now = time.time()
    m = msg("p1", "Let's use YOLO then.", now - 24 * 3600 + 60)
    process(e, m, observed_at=now)
    assert store.summary("mummy") is not None
    store.close()


def test_general_summary_older_than_yesterday_is_rejected_but_event_window_can_remain(tmp_path):
    store, e, _ = engine(tmp_path)
    now = time.time()
    m = msg("m1", "Meeting tomorrow at 6 PM", now - 3 * 86400)
    process(e, m, observed_at=now)
    assert store.list_events()
    assert store.summary("mummy") is None
    store.close()


def test_project_discussion_is_proposed_not_confirmed(tmp_path):
    store, e, _ = engine(tmp_path)
    process(e, msg("d1", "We should use YOLO"))
    row = store._conn().execute("SELECT status,content FROM semantic_memories WHERE memory_type='PROJECT_DECISION'").fetchone()
    assert row is not None and row[0] == "PROPOSED"
    assert "selected" not in row[1].casefold()
    store.close()


def test_confirmed_project_decision_supersedes_proposal(tmp_path):
    store, e, _ = engine(tmp_path)
    base = time.time()
    process(e, msg("d1", "We should use YOLO", base))
    process(e, msg("d2", "Let's use YOLO then.", base + 60))
    rows = store._conn().execute("SELECT status,content FROM semantic_memories WHERE memory_type='PROJECT_DECISION'").fetchall()
    assert sorted(r[0] for r in rows) == ["CONFIRMED", "SUPERSEDED"]
    store.close()


def test_weak_later_proposal_does_not_downgrade_confirmed_decision(tmp_path):
    store, e, _ = engine(tmp_path)
    base = time.time()
    process(e, msg("d1", "Let's use YOLO then.", base))
    process(e, msg("d2", "Maybe use YOLO?", base + 60))
    rows = store._conn().execute("SELECT status FROM semantic_memories WHERE memory_type='PROJECT_DECISION' AND status='CONFIRMED'").fetchall()
    assert len(rows) == 1
    store.close()


def test_repeated_project_status_merges_as_latest_state(tmp_path):
    store, e, _ = engine(tmp_path)
    base = time.time()
    process(e, msg("s1", "YOLO training completed", base))
    process(e, msg("s2", "YOLO implementation completed", base + 60))
    rows = store._conn().execute("SELECT status,content FROM semantic_memories WHERE memory_type='PROJECT_STATUS' AND status!='SUPERSEDED'").fetchall()
    assert len(rows) == 1
    assert "implementation" in rows[0][1].casefold()
    store.close()


def test_semantic_memory_provenance_is_compact(tmp_path):
    store, e, _ = engine(tmp_path)
    process(e, msg("p1", "Let's use YOLO then."))
    row = store._conn().execute("SELECT content,provenance_json FROM semantic_memories").fetchone()
    assert row is not None
    assert "Let's use YOLO then" not in row[1]
    provenance = json.loads(row[1])
    assert provenance["source_message_ids"] == ["p1"]
    assert "source_chat" in provenance
    store.close()


def test_raw_message_body_is_not_stored_in_processed_receipts(tmp_path):
    store, e, _ = engine(tmp_path)
    body = "PRIVATE TRANSCRIPT SHOULD NOT BE STORED"
    process(e, msg("raw", body))
    columns = [r[1] for r in store._conn().execute("PRAGMA table_info(processed_messages)").fetchall()]
    assert "body" not in columns and "text" not in columns and "message" not in columns
    values = [str(r[0]) for r in store._conn().execute("SELECT * FROM processed_messages").fetchall()]
    assert body not in " ".join(values)
    store.close()


def test_raw_evidence_retention_is_bounded_and_semantic_event_survives(tmp_path):
    store, e, _ = engine(tmp_path)
    event = process(e, msg("ev", "Meeting tomorrow at 6 PM"))[0]
    rows = store.event_evidence(event.event_id)
    assert rows and rows[0]["raw_retention_until"] is not None
    now = rows[0]["raw_retention_until"] + 1
    store.maintain_semantic_memory(now)
    assert store.event_evidence(event.event_id) == []
    assert store.get_event(event.event_id) is not None
    store.close()


def test_semantic_memory_is_not_deleted_by_raw_retention_maintenance(tmp_path):
    store, e, _ = engine(tmp_path)
    process(e, msg("fact", "backend = FastAPI"))
    before = store._conn().execute("SELECT COUNT(*) FROM semantic_memories").fetchone()[0]
    store.maintain_semantic_memory(time.time() + 365 * 86400)
    after = store._conn().execute("SELECT COUNT(*) FROM semantic_memories").fetchone()[0]
    assert before == after == 1
    store.close()


def test_relevant_semantic_memory_is_retrieved(tmp_path):
    store, e, _ = engine(tmp_path)
    process(e, msg("fact", "backend = FastAPI"))
    results = store.search_semantic_knowledge("what backend are we using")
    assert results and results[0]["memory_type"] == "IMPORTANT_FACT"
    store.close()


def test_unrelated_semantic_memory_is_not_retrieved(tmp_path):
    store, e, _ = engine(tmp_path)
    process(e, msg("fact", "backend = FastAPI"))
    assert store.search_semantic_knowledge("college exam") == []
    store.close()


def test_superseded_memory_is_not_preferred_by_retrieval(tmp_path):
    store, e, _ = engine(tmp_path)
    proposal = SemanticMemoryProposal("PROJECT_DECISION", "project:detection_model", "Project consideration", "Possible detection model = YOLO", "PROPOSED", 0.7, 0.4)
    confirmed = SemanticMemoryProposal("PROJECT_DECISION", "project:detection_model", "Project decision", "Detection model selected = YOLO", "CONFIRMED", 0.9, 0.9, None, True)
    m1, m2 = msg("a", "proposal"), msg("b", "decision")
    store.upsert_semantic_memory(m1, proposal)
    store.upsert_semantic_memory(m2, confirmed)
    results = store.search_semantic_knowledge("detection model")
    assert results and results[0]["status"] == "CONFIRMED"
    store.close()


def test_persistent_event_survives_source_message_aging(tmp_path):
    store, e, _ = engine(tmp_path)
    old_event = msg("old-event", "Meeting tomorrow at 6 PM", stamp(4))
    events = process(e, old_event, observed_at=time.time())
    assert events
    store.maintain_semantic_memory(time.time() + 10 * 86400)
    assert store.get_event(events[0].event_id) is not None
    store.close()


def test_same_event_does_not_fragment_on_related_update(tmp_path):
    store, e, _ = engine(tmp_path)
    base = time.time()
    e.process_observation({"messages": [
        raw_message(msg("a", "Meeting tomorrow at 6 PM", base)),
        raw_message(msg("b", "Same meeting, actually 7 PM", base + 60)),
        raw_message(msg("c", "Same link here https://meet.example/test", base + 120)),
    ]}, observed_at=base + 120, conversation_hint="mummy")
    assert len(store.list_events()) == 1
    assert len(store.event_evidence(store.list_events()[0].event_id)) >= 2
    store.close()


def test_ambiguous_semantic_conflict_is_not_presented_as_certain(tmp_path):
    store, _e, _ = engine(tmp_path)
    store.upsert_semantic_memory(msg("a", "proposal"), SemanticMemoryProposal("IMPORTANT_FACT", "fact:key", "Fact", "Value=A", "ACTIVE", 0.9, 0.7))
    action, record = store.upsert_semantic_memory(msg("b", "other"), SemanticMemoryProposal("IMPORTANT_FACT", "fact:key", "Fact", "Value=B", "ACTIVE", 0.8, 0.7))
    assert action == "AMBIGUOUS"
    assert record is not None and record.status == "AMBIGUOUS"
    store.close()


def test_human_ownership_pauses_sync(tmp_path):
    ownership = {"deimos": False}
    store, e, _ = engine(tmp_path, can_observe=lambda: ownership["deimos"])
    m = msg("human", "Meeting tomorrow at 6 PM")
    assert e.process_observation({"messages": [raw_message(m)]}, conversation_hint="mummy") == []
    assert store.list_events() == []
    store.close()


def test_deimos_takeover_resumes_using_existing_checkpoint(tmp_path):
    ownership = {"deimos": True}
    store, e, _ = engine(tmp_path, can_observe=lambda: ownership["deimos"])
    m1 = msg("one", "Meeting tomorrow at 6 PM")
    process(e, m1)
    ownership["deimos"] = False
    m2 = msg("two", "Neural Networks assignment due Friday at 5 PM", time.time() + 1)
    assert e.process_observation({"messages": [raw_message(m2)]}, conversation_hint="mummy") == []
    ownership["deimos"] = True
    events = process(e, m2, observed_at=time.time() + 2)
    assert events and events[0].type is IntelligenceEventType.ASSIGNMENT
    assert store.sync_checkpoint("mummy").last_source_message_id == "two"
    store.close()


def test_checkpoint_is_monotonic_for_out_of_order_messages(tmp_path):
    store, e, _ = engine(tmp_path)
    base = time.time()
    newer = msg("newer", "Meeting tomorrow at 6 PM", base)
    older = msg("older", "Meeting tomorrow at 7 PM", base - 60)
    process(e, newer)
    cp_before = store.sync_checkpoint("mummy")
    process(e, older, observed_at=base + 60)
    cp_after = store.sync_checkpoint("mummy")
    assert cp_before.last_source_message_id == cp_after.last_source_message_id == "newer"
    store.close()


def test_event_discovery_uses_five_days_but_general_memory_uses_two(tmp_path):
    store, e, _ = engine(tmp_path)
    now = time.time()
    m = msg("m5", "Meeting tomorrow at 6 PM; let's use YOLO", now - 4 * 86400)
    events = process(e, m, observed_at=now)
    assert events and events[0].type is IntelligenceEventType.MEETING
    assert store._conn().execute("SELECT COUNT(*) FROM semantic_memories").fetchone()[0] == 0
    store.close()


def test_no_untargeted_history_search_is_performed(tmp_path):
    store, e, _ = engine(tmp_path)
    m = msg("very-old", "Meeting tomorrow at 6 PM", stamp(20))
    process(e, m)
    assert store.list_events() == []
    store.close()


def test_gap_sync_does_not_advance_checkpoint_on_failed_new_source(tmp_path):
    path = tmp_path / "wa.sqlite3"
    s = WhatsAppIntelligenceStore(path)
    s.enable()
    e = WhatsAppIntelligence(s)
    first = msg("first", "Meeting tomorrow at 6 PM", time.time() - 3600)
    process(e, first)
    cp = s.sync_checkpoint("mummy")
    second = msg("second", "Neural Networks assignment due Friday at 5 PM", time.time() + 10)
    e._analyze_message = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    e.process_observation({"messages": [raw_message(second)]}, observed_at=time.time() + 20, conversation_hint="mummy")
    after = s.sync_checkpoint("mummy")
    assert after.last_source_message_id == cp.last_source_message_id
    assert not s.is_message_processed(second.message_id)
    s.close()


def test_sync_checkpoint_is_per_chat(tmp_path):
    store, e, _ = engine(tmp_path)
    m1 = msg("a", "Meeting tomorrow at 6 PM", chat="mummy")
    m2 = msg("b", "Meeting tomorrow at 7 PM", chat="college", sender="College")
    process(e, m1)
    process(e, m2)
    assert store.sync_checkpoint("mummy").last_source_message_id == "a"
    assert store.sync_checkpoint("college").last_source_message_id == "b"
    store.close()


def test_semantic_context_can_combine_short_followup(tmp_path):
    store, e, _ = engine(tmp_path)
    base = time.time()
    msgs = [
        msg("a", "College tomorrow?", base),
        msg("b", "haan", base + 5),
        msg("c", "What time?", base + 10),
        msg("d", "10", base + 15),
    ]
    e.process_observation({"messages": [raw_message(x) for x in msgs]}, observed_at=base + 15, conversation_hint="mummy")
    events = [x for x in store.list_events() if x.type is IntelligenceEventType.COLLEGE_PLAN]
    assert events
    assert events[0].time == "10:00"
    store.close()


def test_event_update_evidence_is_lightweight(tmp_path):
    store, e, _ = engine(tmp_path)
    base = time.time()
    process(e, msg("a", "Meeting tomorrow at 6 PM", base))
    process(e, msg("b", "Actually 7 PM", base + 60))
    event = store.list_events()[0]
    evidence = store.event_evidence(event.event_id)
    assert evidence
    assert all("Meeting tomorrow" not in str(row["extracted_claim"]) for row in evidence)
    store.close()


def test_event_status_can_remain_proposed_for_incomplete_meeting(tmp_path):
    store, e, _ = engine(tmp_path)
    events = process(e, msg("p", "Meeting at 6 PM"))
    assert events[0].status == "PROPOSED"
    store.close()


def test_event_update_confirmation_can_confirm_same_event(tmp_path):
    store, e, _ = engine(tmp_path)
    base = time.time()
    process(e, msg("a", "Meeting tomorrow at 6 PM", base))
    process(e, msg("b", "Okay then", base + 60))
    event = store.list_events()[0]
    assert event.status == "CONFIRMED"
    store.close()


def test_memory_does_not_use_conversation_transcript_in_retrieval(tmp_path):
    store, e, _ = engine(tmp_path)
    process(e, msg("f", "backend = FastAPI"))
    payload = store.search_semantic_knowledge("backend")
    assert payload
    assert "source_message_id" in payload[0]
    assert "backend = FastAPI" in payload[0]["content"]
    assert payload[0].get("source") == "whatsapp_semantic_world_state"
    store.close()


def test_schema_contains_distinct_sync_memory_and_event_concepts(tmp_path):
    store, _e, _ = engine(tmp_path)
    tables = {row[0] for row in store._conn().execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"sync_checkpoints", "semantic_memories", "intelligence_events", "processed_messages"} <= tables
    assert "message" not in [r[1] for r in store._conn().execute("PRAGMA table_info(semantic_memories)").fetchall()]
    store.close()


def test_legacy_event_evidence_without_raw_retention_migrates_before_index_creation(tmp_path):
    """Existing P3.3 DBs must migrate columns before indexes reference them."""
    import sqlite3

    path = tmp_path / "legacy-whatsapp.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE event_evidence (
            event_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            message_timestamp REAL NOT NULL,
            evidence_type TEXT NOT NULL,
            extracted_claim TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at REAL NOT NULL,
            UNIQUE(event_id, message_id, evidence_type)
        )
    """)
    conn.execute(
        "INSERT INTO event_evidence(event_id,message_id,conversation_id,message_timestamp,evidence_type,extracted_claim,confidence,created_at) VALUES(?,?,?,?,?,?,?,?)",
        ("evt-legacy", "msg-legacy", "mummy", 1000.0, "MEETING", "meeting at 6", 0.9, 1000.0),
    )
    conn.commit()
    conn.close()

    store = WhatsAppIntelligenceStore(path)
    columns = {row[1] for row in store._conn().execute("PRAGMA table_info(event_evidence)").fetchall()}
    assert "evidence_id" in columns
    assert "raw_retention_until" in columns

    indexes = {
        row[1]
        for row in store._conn().execute("PRAGMA index_list(event_evidence)").fetchall()
    }
    assert "idx_event_evidence_retention" in indexes

    row = store._conn().execute(
        "SELECT evidence_id, raw_retention_until FROM event_evidence WHERE event_id='evt-legacy'"
    ).fetchone()
    assert row is not None
    assert row[0]
    assert row[1] is not None

    # A second initialization must be idempotent.
    store.close()
    store2 = WhatsAppIntelligenceStore(path)
    assert store2._conn().execute("SELECT COUNT(*) FROM event_evidence").fetchone()[0] == 1
    store2.close()


def test_existing_event_evidence_with_evidence_id_but_without_raw_retention_migrates(tmp_path):
    """Match the reported production shape exactly: evidence_id exists, raw retention does not."""
    import sqlite3

    path = tmp_path / "legacy-retention.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE event_evidence (
            evidence_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            message_timestamp REAL NOT NULL,
            evidence_type TEXT NOT NULL,
            extracted_claim TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at REAL NOT NULL,
            UNIQUE(event_id, message_id, evidence_type)
        )
    """)
    conn.execute(
        "INSERT INTO event_evidence VALUES(?,?,?,?,?,?,?,?,?)",
        ("evi-existing", "evt-existing", "msg-existing", "mummy", 2000.0, "MEETING", "meeting at 6", 0.8, 2000.0),
    )
    conn.commit()
    conn.close()

    store = WhatsAppIntelligenceStore(path)
    columns = {row[1] for row in store._conn().execute("PRAGMA table_info(event_evidence)").fetchall()}
    assert "evidence_id" in columns
    assert "raw_retention_until" in columns
    row = store._conn().execute(
        "SELECT evidence_id, raw_retention_until FROM event_evidence WHERE evidence_id='evi-existing'"
    ).fetchone()
    assert row is not None and row[0] == "evi-existing" and row[1] is not None
    store.close()
