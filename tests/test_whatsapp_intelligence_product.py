from datetime import datetime, timezone

from agent_control.whatsapp_intelligence import (
    IntelligenceEventType,
    WhatsAppIntelligence,
    WhatsAppIntelligenceStore,
    extract_messages,
    relevance,
    Relevance,
)

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc).timestamp()


def service(tmp_path, callback=None):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    svc = WhatsAppIntelligence(store, on_intelligence=callback)
    svc.enable(now=NOW)
    return svc


def msg(i, text, age_days=0, chat="chat", hour=12, metadata=None):
    ts = NOW - age_days * 86400 - (12 - hour) * 60
    value = {"message_id": i, "conversation_id": chat, "timestamp": ts, "sender": "Alice", "text": text}
    if metadata:
        value.update(metadata)
    return value


def test_general_summary_accepts_today(tmp_path):
    svc = service(tmp_path)
    svc.process_observation({"messages": [msg("1", "We should use YOLO")]}, observed_at=NOW)
    assert any(e.type is IntelligenceEventType.DECISION for e in svc.store.list_events())


def test_general_summary_accepts_yesterday(tmp_path):
    svc = service(tmp_path)
    svc.process_observation({"messages": [msg("1", "We should use YOLO", age_days=1)]}, observed_at=NOW)
    assert any(e.type is IntelligenceEventType.DECISION for e in svc.store.list_events())


def test_general_summary_rejects_messages_older_than_yesterday(tmp_path):
    svc = service(tmp_path)
    svc.process_observation({"messages": [msg("1", "We should use YOLO", age_days=2)]}, observed_at=NOW)
    assert svc.store.list_events() == []


def test_event_intelligence_accepts_messages_up_to_five_days(tmp_path):
    svc = service(tmp_path)
    svc.process_observation({"messages": [msg("1", "Meeting tomorrow at 6 PM", age_days=5)]}, observed_at=NOW)
    assert any(e.type is IntelligenceEventType.MEETING for e in svc.store.list_events())


def test_event_intelligence_rejects_messages_older_than_five_days(tmp_path):
    svc = service(tmp_path)
    svc.process_observation({"messages": [msg("1", "Meeting tomorrow at 6 PM", age_days=6)]}, observed_at=NOW)
    assert svc.store.list_events() == []


def test_persistent_event_survives_source_lookback_expiration(tmp_path):
    svc = service(tmp_path)
    svc.process_observation({"messages": [msg("1", "Meeting tomorrow at 6 PM", age_days=1)]}, observed_at=NOW)
    event_id = svc.store.list_events()[0].event_id
    svc.process_observation({"messages": [msg("2", "hi", age_days=0)]}, observed_at=NOW)
    assert svc.store.get_event(event_id) is not None


def test_read_messages_are_eligible(tmp_path):
    svc = service(tmp_path)
    svc.process_observation({"messages": [msg("1", "Assignment due Friday", metadata={"read": True, "unread": False})]}, observed_at=NOW)
    assert svc.store.list_events()


def test_unread_messages_are_eligible(tmp_path):
    svc = service(tmp_path)
    svc.process_observation({"messages": [msg("1", "Assignment due Friday", metadata={"read": False, "unread": True})]}, observed_at=NOW)
    assert svc.store.list_events()


def test_useless_chat_produces_no_event(tmp_path):
    svc = service(tmp_path)
    messages = [msg("1", "bro"), msg("2", "haan", hour=11), msg("3", "😂", hour=10)]
    svc.process_observation({"messages": messages}, observed_at=NOW)
    assert svc.store.list_events() == []


def test_college_plan_is_extracted_from_bounded_context(tmp_path):
    svc = service(tmp_path)
    messages = [msg("1", "College tomorrow?"), msg("2", "Yeah", hour=11), msg("3", "What time?", hour=10), msg("4", "10", hour=9)]
    svc.process_observation({"messages": messages}, observed_at=NOW)
    events = svc.store.list_events()
    assert len(events) == 1
    assert events[0].type is IntelligenceEventType.COLLEGE_PLAN
    assert events[0].time == "10:00"


def test_meeting_missing_date_remains_partial(tmp_path):
    svc = service(tmp_path)
    svc.process_observation({"messages": [msg("1", "There will be a meeting at 6 PM")]}, observed_at=NOW)
    event = svc.store.list_events()[0]
    assert event.type is IntelligenceEventType.MEETING
    assert event.date is None
    assert event.time == "18:00"


def test_later_message_updates_same_meeting(tmp_path):
    svc = service(tmp_path)
    svc.process_observation({"messages": [msg("1", "Meeting at 6 PM")]}, observed_at=NOW)
    first = svc.store.list_events()[0]
    svc.process_observation({"messages": [msg("2", "Actually Wednesday at 7 PM", hour=11)]}, observed_at=NOW)
    events = svc.store.list_events()
    assert len(events) == 1
    assert events[0].event_id == first.event_id
    assert events[0].time == "19:00"
    assert events[0].date == "2026-09-23"


def test_meeting_cancellation_updates_existing_event(tmp_path):
    svc = service(tmp_path)
    svc.process_observation({"messages": [msg("1", "Meeting tomorrow at 6 PM")]}, observed_at=NOW)
    event_id = svc.store.list_events()[0].event_id
    svc.process_observation({"messages": [msg("2", "Meeting cancel hai", hour=11)]}, observed_at=NOW)
    events = svc.store.list_events()
    assert len(events) == 1
    assert events[0].event_id == event_id
    assert events[0].status == "CANCELLED"


def test_assignment_is_extracted(tmp_path):
    svc = service(tmp_path)
    svc.process_observation({"messages": [msg("1", "Neural Networks assignment due Friday 5 PM")]}, observed_at=NOW)
    event = svc.store.list_events()[0]
    assert event.type is IntelligenceEventType.ASSIGNMENT
    assert "Neural Networks" in event.title
    assert event.date == "2026-09-25"
    assert event.time == "17:00"


def test_academic_announcement_is_extracted(tmp_path):
    svc = service(tmp_path)
    svc.process_observation({"messages": [msg("1", "Mid-sem exams start from October 14. Timetable will be shared soon.")]}, observed_at=NOW)
    event = svc.store.list_events()[0]
    assert event.type is IntelligenceEventType.EXAM
    assert event.date == "2026-10-14"


def test_deadline_update_modifies_existing_assignment(tmp_path):
    svc = service(tmp_path)
    svc.process_observation({"messages": [msg("1", "Neural Networks assignment due Friday 5 PM")]}, observed_at=NOW)
    event_id = svc.store.list_events()[0].event_id
    svc.process_observation({"messages": [msg("2", "Deadline extended to Monday", hour=11)]}, observed_at=NOW)
    events = svc.store.list_events()
    assert len(events) == 1
    assert events[0].event_id == event_id
    assert events[0].date == "2026-09-21"


def test_project_decision_is_extracted(tmp_path):
    svc = service(tmp_path)
    svc.process_observation({"messages": [msg("1", "We should use YOLO"), msg("2", "Okay final", hour=11)]}, observed_at=NOW)
    events = svc.store.list_events()
    assert any(e.type is IntelligenceEventType.DECISION for e in events)


def test_duplicate_messages_do_not_create_duplicate_events(tmp_path):
    svc = service(tmp_path)
    value = msg("same", "Meeting tomorrow at 6 PM")
    svc.process_observation({"messages": [value]}, observed_at=NOW)
    svc.process_observation({"messages": [value]}, observed_at=NOW)
    assert len(svc.store.list_events()) == 1


def test_raw_messages_are_not_in_normal_user_facing_output(tmp_path):
    from agent_control.session import Session

    output = []
    session = Session.__new__(Session)
    session.narrator = type("Narrator", (), {"reply": output.append})()
    event = service(tmp_path).store
    svc = WhatsAppIntelligence(event)
    svc.process_observation({"messages": [msg("1", "Neural Networks assignment due Friday 5 PM")]}, observed_at=NOW)
    stored = svc.store.list_events()[0]
    session._present_whatsapp_intelligence(stored)
    assert output
    assert "Neural Networks assignment due Friday 5 PM" not in output[0]
    assert "assignment" in output[0].casefold()


def test_debug_can_expose_diagnostic_information(tmp_path):
    debug = []
    svc = WhatsAppIntelligence(WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3"), on_debug=debug.append)
    svc.enable(now=NOW)
    svc.process_observation({"messages": [msg("1", "Meeting tomorrow at 6 PM")]}, observed_at=NOW)
    assert any("message_id=1" in line for line in debug)
    assert any("PERSISTED" in line for line in debug)


def test_official_college_group_can_produce_academic_intelligence(tmp_path):
    svc = service(tmp_path)
    svc.process_observation({"messages": [msg("1", "Neural Networks assignment due Friday", chat="Official College Group")]}, observed_at=NOW)
    assert svc.store.list_events()[0].type is IntelligenceEventType.ASSIGNMENT


def test_ordinary_chatter_in_official_group_is_ignored(tmp_path):
    svc = service(tmp_path)
    svc.process_observation({"messages": [msg("1", "hi everyone", chat="Official College Group"), msg("2", "lol", chat="Official College Group", hour=11)]}, observed_at=NOW)
    assert svc.store.list_events() == []


def test_five_day_lookback_does_not_search_unlimited_history(tmp_path):
    svc = service(tmp_path)
    svc.process_observation({"messages": [msg("1", "Meeting tomorrow at 6 PM", age_days=30)]}, observed_at=NOW)
    assert svc.store.list_events() == []


def test_general_summary_never_uses_five_day_event_window_for_decisions(tmp_path):
    svc = service(tmp_path)
    svc.process_observation({"messages": [msg("1", "We should use YOLO", age_days=4)]}, observed_at=NOW)
    assert svc.store.list_events() == []


def test_persistent_event_survives_general_summary_expiration(tmp_path):
    svc = service(tmp_path)
    svc.process_observation({"messages": [msg("1", "Meeting October 10 at 6 PM", age_days=1)]}, observed_at=NOW)
    event_id = svc.store.list_events()[0].event_id
    svc.store.update_summary("chat", "old general summary", "1")
    assert svc.store.get_event(event_id) is not None


def test_extract_messages_preserves_read_unread_independent_of_identity(tmp_path):
    a = extract_messages({"messages": [msg("1", "Meeting tomorrow", metadata={"read": True})]})[0]
    b = extract_messages({"messages": [msg("1", "Meeting tomorrow", metadata={"read": False})]})[0]
    assert a.message_id == b.message_id == "1"


def test_relevance_marks_obvious_noise_low_value():
    assert relevance("hi") is Relevance.LOW_VALUE
    assert relevance("😂") is Relevance.LOW_VALUE
    assert relevance("ok bro") is Relevance.LOW_VALUE
