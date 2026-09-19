from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from agent_control.persistent_memory import PersistentMemory
from agent_control.response import Narrator
from agent_control.session import Session
from agent_control.skills.browser.backend import BrowserElement, BrowserObservation, BrowserSkillAdapter
from agent_control.whatsapp_control import WhatsAppControlLease
from agent_control.whatsapp_intelligence import (
    IntelligenceEventType,
    WhatsAppIntelligence,
    WhatsAppIntelligenceStore,
    WhatsAppMessage,
    _message_identity,
    extract_messages,
)


def _ts(day: str, hour: int = 10, minute: int = 0) -> float:
    return datetime.fromisoformat(f"{day}T{hour:02d}:{minute:02d}:00+00:00").timestamp()


def _structured_observation(*, chat: str = "mummy", outgoing: bool = False, native_id: str = "wa-1", ts: float | None = None, body: str = "Meeting tomorrow at 6 PM"):
    ts = ts or _ts("2026-09-19", 10)
    raw_message = {
        "role": "article",
        "ref": "@e42",
        "message_id": native_id,
        "conversation_id": chat,
        "body": body,
        "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
        "from_me": outgoing,
        "sender": "You" if outgoing else "Mummy",
    }
    result = {
        "url": "https://web.whatsapp.com/",
        "elements": [raw_message],
        "text": f'@e42 article "{body}"',
    }
    return result


def test_native_source_id_is_preserved_and_stable_across_observations():
    first = extract_messages(_structured_observation(native_id="native-42"), conversation_hint="mummy", observed_at=_ts("2026-09-19", 11))
    second = extract_messages(_structured_observation(native_id="native-42"), conversation_hint="mummy", observed_at=_ts("2026-09-19", 11, 1))
    assert first and second
    assert first[0].message_id == "native-42"
    assert second[0].message_id == "native-42"


def test_same_fallback_source_id_is_stable_when_direction_metadata_changes():
    body = "Meeting tomorrow at 6 PM"
    ts = _ts("2026-09-19", 10)
    a = _message_identity("mummy", ts, "Mummy", body, {"sender_id": "p-1", "type": "text"})
    b = _message_identity("mummy", ts, "", body, {"sender_id": "p-1", "type": "text"})
    assert a == b


def test_browser_skill_normalization_preserves_structured_message_ownership():
    elements = BrowserSkillAdapter._extract_elements(_structured_observation(outgoing=True, native_id="native-out"))
    msg = next(e for e in elements if e.ref == "@e42")
    assert isinstance(msg.raw, dict)
    assert msg.raw["from_me"] is True
    assert msg.raw["message_id"] == "native-out"


def test_incoming_direction_comes_from_message_sender():
    messages = extract_messages(_structured_observation(outgoing=False, native_id="in-1"), conversation_hint="mummy", observed_at=_ts("2026-09-19", 10))
    assert messages[0].metadata["direction"] == "incoming"
    assert messages[0].metadata["is_outgoing"] is False
    assert messages[0].metadata["sender_evidence"] == "from_me=False" or messages[0].metadata["sender_evidence"].startswith("sender=")


def test_outgoing_direction_comes_from_from_me():
    messages = extract_messages(_structured_observation(outgoing=True, native_id="out-1"), conversation_hint="mummy", observed_at=_ts("2026-09-19", 10))
    assert messages[0].metadata["direction"] == "outgoing"
    assert messages[0].metadata["is_outgoing"] is True
    assert messages[0].metadata["sender_evidence"] == "from_me=True"


def test_self_chat_does_not_infer_outgoing_from_chat_name_alone():
    payload = _structured_observation(chat="saksham", outgoing=False, native_id="self-unknown")
    payload["elements"][0].pop("from_me")
    payload["elements"][0].pop("sender")
    messages = extract_messages(payload, conversation_hint="saksham", observed_at=_ts("2026-09-19", 10))
    assert messages[0].metadata["direction"] == "unknown"
    assert messages[0].metadata["is_outgoing"] is None


def test_garbage_sender_is_rejected():
    for garbage in ("https", "~", "39 unread messages Happy Birthday Anu Alka Di"):
        payload = _structured_observation(outgoing=False, native_id=f"bad-{garbage[:2]}")
        payload["elements"][0].pop("from_me")
        payload["elements"][0]["sender"] = garbage
        messages = extract_messages(payload, conversation_hint="mummy", observed_at=_ts("2026-09-19", 10))
        assert messages[0].metadata["direction"] == "unknown"
        assert messages[0].metadata["is_outgoing"] is None
        assert messages[0].metadata["sender_evidence"] == "NONE"



def test_nearby_unrelated_semantic_text_cannot_assign_message_direction():
    payload = _structured_observation(outgoing=False, native_id="nearby-1")
    payload["elements"] = [
        {"role": "text", "ref": "@e40", "name": "You"},
        payload["elements"][0],
        {"role": "text", "ref": "@e41", "name": "Sent by system"},
    ]
    messages = extract_messages(payload, conversation_hint="mummy", observed_at=_ts("2026-09-19", 10))
    assert messages
    assert messages[0].metadata["direction"] == "incoming"
    assert messages[0].metadata["sender_evidence"] == "from_me=False"



def _observation_from_messages(messages):
    return {
        "url": "https://web.whatsapp.com/",
        "elements": [
            {
                "role": "article",
                "ref": f"@e{i}",
                "message_id": mid,
                "conversation_id": chat,
                "body": body,
                "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                **meta,
            }
            for i, (mid, chat, ts, body, meta) in enumerate(messages)
        ],
    }


def _service(tmp_path: Path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    svc = WhatsAppIntelligence(store, can_observe=lambda: True)
    svc.enable(now=_ts("2026-09-19", 12))
    return svc, store


def test_initial_catchup_then_live_uses_zero_semantic_processing_for_visible_old_window(tmp_path, monkeypatch):
    svc, store = _service(tmp_path)
    calls = []
    monkeypatch.setattr(svc, "_analyze_message", lambda message, context, **kwargs: calls.append(message.message_id) or [])
    obs = _structured_observation(native_id="stable-1", ts=_ts("2026-09-19", 10), body="meeting tomorrow at 6")
    svc.process_observation(obs, observed_at=_ts("2026-09-19", 12), conversation_hint="mummy")
    assert calls == ["stable-1"]
    svc.process_observation(obs, observed_at=_ts("2026-09-19", 12, 1), conversation_hint="mummy")
    svc.process_observation(obs, observed_at=_ts("2026-09-19", 12, 2), conversation_hint="mummy")
    assert calls == ["stable-1"]
    checkpoint = store.sync_checkpoint("mummy")
    assert checkpoint is not None
    assert checkpoint.last_source_message_id == "stable-1"



def test_live_observer_honors_poll_interval_and_semantically_processes_only_new_source(tmp_path, monkeypatch):
    import time as _time
    from agent_control.skills.browser.backend import BrowserObservation

    class FakeBrowser:
        def __init__(self, payload):
            self.payload = payload
            self._last_observation = None
            self.calls = 0

        def observe(self):
            self.calls += 1
            self._last_observation = BrowserObservation(
                generation=self.calls,
                session="fake",
                tab_id="tab",
                url="https://web.whatsapp.com/",
                text=self.payload.get("text", ""),
                elements=tuple(),
                raw=self.payload,
            )
            return self._last_observation

    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    logs = []
    svc = WhatsAppIntelligence(store, can_observe=lambda: True, on_debug=logs.append)
    now = _ts("2026-09-19", 12)
    svc.enable(now=now)
    svc._active_authorized_target = lambda _obs: "mummy"
    analyze_calls = []
    monkeypatch.setattr(
        svc,
        "_analyze_message",
        lambda message, context, **kwargs: analyze_calls.append(message.message_id) or [],
    )
    payload = _structured_observation(native_id="observer-runtime-1", ts=_ts("2026-09-19", 10))
    browser = FakeBrowser(payload)
    svc.start_observing(browser, interval_s=0.5)
    _time.sleep(1.35)
    svc.stop_observing()

    # A 0.5s minimum poll interval means this must be a handful of observations,
    # not an uncontrolled tight loop.  One source message is semantically processed once.
    assert 2 <= browser.calls <= 5, browser.calls
    assert analyze_calls == ["observer-runtime-1"]
    sync_lines = [line for line in logs if line.startswith("WHATSAPP_SYNC:")]
    assert sync_lines
    assert sum("source_delta=1" in line for line in sync_lines) == 1
    assert sum("source_delta=0" in line for line in sync_lines) >= 1
    assert not any("ELIGIBLE" in line or "ALREADY_PROCESSED" in line for line in logs)

def test_new_source_after_checkpoint_is_the_only_semantic_delta(tmp_path, monkeypatch):
    svc, store = _service(tmp_path)
    calls = []
    monkeypatch.setattr(svc, "_analyze_message", lambda message, context, **kwargs: calls.append(message.message_id) or [])
    first = _structured_observation(native_id="stable-1", ts=_ts("2026-09-19", 10), body="meeting tomorrow at 6")
    svc.process_observation(first, observed_at=_ts("2026-09-19", 12), conversation_hint="mummy")
    second = {
        "url": "https://web.whatsapp.com/",
        "elements": [
            first["elements"][0],
            {
                "role": "article", "ref": "@e43", "message_id": "stable-2", "conversation_id": "mummy",
                "body": "Actually make it 7 PM", "timestamp": datetime.fromtimestamp(_ts("2026-09-19", 11), tz=timezone.utc).isoformat(),
                "from_me": False, "sender": "Mummy",
            },
        ],
        "text": '@e42 article "meeting tomorrow at 6"\n@e43 article "Actually make it 7 PM"',
    }
    svc.process_observation(second, observed_at=_ts("2026-09-19", 12, 1), conversation_hint="mummy")
    assert calls == ["stable-1", "stable-2"]
    checkpoint = store.sync_checkpoint("mummy")
    assert checkpoint and checkpoint.last_source_message_id == "stable-2"


def test_per_chat_checkpoints_are_independent(tmp_path):
    _, store = _service(tmp_path)
    m1 = WhatsAppMessage("m1", "mummy", _ts("2026-09-19", 10), "hello")
    m2 = WhatsAppMessage("m2", "saksham", _ts("2026-09-19", 10), "hello")
    store.advance_sync_checkpoint(m1, synced_at=_ts("2026-09-19", 12))
    assert store.sync_checkpoint("mummy").last_source_message_id == "m1"
    assert store.sync_checkpoint("saksham") is None
    store.advance_sync_checkpoint(m2, synced_at=_ts("2026-09-19", 12))
    assert store.sync_checkpoint("saksham").last_source_message_id == "m2"


def test_checkpoint_does_not_advance_when_semantic_processing_fails(tmp_path, monkeypatch):
    svc, store = _service(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("semantic-failure")
    monkeypatch.setattr(svc, "_analyze_message", fail)
    obs = _structured_observation(native_id="fail-1", ts=_ts("2026-09-19", 10), body="meeting tomorrow at 6")
    svc.process_observation(obs, observed_at=_ts("2026-09-19", 12), conversation_hint="mummy")
    assert store.sync_checkpoint("mummy") is None
    assert not store.is_message_processed("fail-1")


def test_crash_between_message_commit_and_checkpoint_recovers_without_reextract(tmp_path, monkeypatch):
    svc, store = _service(tmp_path)
    calls = []
    monkeypatch.setattr(svc, "_analyze_message", lambda message, context, **kwargs: calls.append(message.message_id) or [])
    obs = _structured_observation(native_id="crash-1", ts=_ts("2026-09-19", 10), body="meeting tomorrow at 6")
    original = store.advance_sync_checkpoint
    def fail_once(message, *, synced_at=None):
        monkeypatch.setattr(store, "advance_sync_checkpoint", original)
        raise RuntimeError("checkpoint-crash")
    monkeypatch.setattr(store, "advance_sync_checkpoint", fail_once)
    try:
        svc.process_observation(obs, observed_at=_ts("2026-09-19", 12), conversation_hint="mummy")
    except RuntimeError:
        pass
    assert calls == ["crash-1"]
    assert store.is_message_processed("crash-1")
    svc.process_observation(obs, observed_at=_ts("2026-09-19", 12, 1), conversation_hint="mummy")
    assert calls == ["crash-1"]


def test_raw_evidence_is_redacted_after_retention(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    msg = WhatsAppMessage("e1", "mummy", _ts("2026-09-01", 10), "meeting secret detail")
    now = _ts("2026-10-01", 10)
    event = {
        "event_id": "evt-1", "conversation_id": "mummy", "event_type": "MEETING", "title": "Meeting",
        "description": "Meeting", "status": "CONFIRMED", "confidence": 0.9, "importance": 0.8,
        "urgency": 0.4, "created_at": _ts("2026-09-01", 10), "updated_at": _ts("2026-09-01", 10),
        "event_time": None, "deadline": None, "date": None, "time": None, "timezone": None,
        "platform": None, "location": None, "meeting_link": None, "participants_json": "[]",
        "evidence_json": "[]", "source_message_ids_json": '["e1"]', "scheduler_candidate": 0,
    }
    conn = store._conn()
    conn.execute(
        "INSERT INTO intelligence_events("
        "event_id,conversation_id,event_type,title,description,status,confidence,importance,urgency,created_at,updated_at,"
        "event_time,deadline,date,time,timezone,platform,location,meeting_link,participants_json,evidence_json,source_message_ids_json,scheduler_candidate"
        ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        tuple(event.values()),
    )
    conn.execute("INSERT INTO event_evidence(evidence_id,event_id,message_id,conversation_id,message_timestamp,evidence_type,extracted_claim,confidence,created_at,raw_retention_until) VALUES(?,?,?,?,?,?,?,?,?,?)", ("ev-1","evt-1","e1","mummy",msg.timestamp,"CLAIM",msg.text,0.9,msg.timestamp,msg.timestamp+86400))
    conn.commit()
    result = store.compact_source_state(now=now)
    row = conn.execute("SELECT extracted_claim FROM event_evidence WHERE evidence_id='ev-1'").fetchone()
    assert result["evidence_redacted"] == 1
    assert row[0] == ""


def test_persistent_memory_purges_expired_transient_semantics(tmp_path):
    mem = PersistentMemory(tmp_path / "memory.sqlite3")
    rec = mem.put(memory_type="WHATSAPP_IDEA", content="Possible idea", source="test", confidence=0.7, expires_at=1, key="idea-1")
    assert rec is not None
    assert mem.get(rec.id) is not None
    assert mem.purge_expired(now=2) == 1
    assert mem.get(rec.id) is None


def test_persistent_memory_does_not_store_raw_whatsapp_body(tmp_path, monkeypatch):
    svc, _ = _service(tmp_path)
    monkeypatch.setenv("DEIMOS_AGENT_MEMORY", str(tmp_path / "memory.sqlite3"))
    memory = svc._semantic_memory_store()
    from agent_control.whatsapp_intelligence import IntelligenceEvent
    event = IntelligenceEvent(
        event_id="evt-compact", conversation_id="mummy", type=IntelligenceEventType.MEETING,
        title="Meeting", description="Meeting scheduled", status="PROPOSED", confidence=0.9,
        importance=0.8, urgency=0.4, created_at=1, updated_at=1, date="2026-09-20", time="18:00",
        source_message_ids=("src-1",),
    )
    message = WhatsAppMessage("src-1", "mummy", _ts("2026-09-19", 10), "PRIVATE RAW BODY SHOULD NOT BE STORED", metadata={"direction":"incoming"})
    svc._persist_semantic_memory(event, message)
    rows = memory.recent(limit=10)
    assert rows
    assert all("PRIVATE RAW BODY SHOULD NOT BE STORED" not in r.content for r in rows)
    assert all("PRIVATE RAW BODY SHOULD NOT BE STORED" not in json.dumps(r.evidence) for r in rows)


def test_normal_session_does_not_print_background_debug(tmp_path):
    writes = []
    narrator = Narrator.build(enabled=False, write=writes.append)
    session = Session(narrator=narrator, workspace=str(tmp_path), debug=False)
    service = session._whatsapp_intelligence_service()
    service.on_debug("WHATSAPP_SYNC: should be debug only")
    assert writes == []


def test_human_ownership_callback_pauses_processing(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    store.enable(now=_ts("2026-09-19", 12))
    allowed = {"value": False}
    svc = WhatsAppIntelligence(store, can_observe=lambda: allowed["value"])
    obs = _structured_observation(native_id="human-1", ts=_ts("2026-09-19", 10), body="meeting tomorrow at 6")
    assert svc.process_observation(obs, observed_at=_ts("2026-09-19", 12), conversation_hint="mummy") == []
    assert store.sync_checkpoint("mummy") is None


def test_direction_and_sync_diagnostics_do_not_include_message_body(tmp_path):
    logs = []
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    svc = WhatsAppIntelligence(store, can_observe=lambda: True, on_debug=logs.append)
    svc.enable(now=_ts("2026-09-19", 12))
    body = "MEGA PRIVATE BODY 123"
    obs = _structured_observation(native_id="diag-1", ts=_ts("2026-09-19", 10), body=body)
    svc.process_observation(obs, observed_at=_ts("2026-09-19", 12), conversation_hint="mummy")
    joined = "\n".join(logs)
    assert "MEGA PRIVATE BODY 123" not in joined


def test_same_fallback_identity_differs_for_different_messages():
    ts = _ts("2026-09-19", 10)
    a = _message_identity("mummy", ts, "Mummy", "meeting at 6", {"sender_id": "p-1", "type": "text"})
    b = _message_identity("mummy", ts, "Mummy", "meeting at 7", {"sender_id": "p-1", "type": "text"})
    assert a != b


def test_read_unread_flags_do_not_affect_direction():
    for read in (True, False):
        payload = _structured_observation(outgoing=False, native_id=f"read-{read}")
        payload["elements"][0]["unread"] = not read
        payload["elements"][0]["read"] = read
        messages = extract_messages(payload, conversation_hint="mummy", observed_at=_ts("2026-09-19", 10))
        assert messages[0].metadata["direction"] == "incoming"
        assert messages[0].metadata["is_outgoing"] is False


def test_self_chat_with_explicit_user_author_is_outgoing():
    payload = _structured_observation(chat="saksham", outgoing=True, native_id="self-out", body="Meeting tomorrow at 6 PM")
    payload["elements"][0]["sender"] = "You"
    messages = extract_messages(payload, conversation_hint="saksham", observed_at=_ts("2026-09-19", 10))
    assert messages[0].metadata["direction"] == "outgoing"
    assert messages[0].metadata["is_outgoing"] is True


def test_initial_catchup_is_bounded_to_current_five_days(tmp_path, monkeypatch):
    svc, store = _service(tmp_path)
    calls = []
    monkeypatch.setattr(svc, "_analyze_message", lambda message, context, **kwargs: calls.append(message.message_id) or [])
    now = _ts("2026-09-19", 12)
    old = _structured_observation(native_id="too-old", ts=_ts("2026-09-13", 12), body="meeting tomorrow at 6")
    recent = _structured_observation(native_id="recent", ts=_ts("2026-09-18", 12), body="meeting tomorrow at 6")
    svc.process_observation({"elements": old["elements"] + recent["elements"], "url": old["url"]}, observed_at=now, conversation_hint="mummy")
    assert calls == ["recent"]
    checkpoint = store.sync_checkpoint("mummy")
    assert checkpoint and checkpoint.last_source_message_id == "recent"


def test_restart_gap_processes_only_messages_after_persisted_checkpoint(tmp_path, monkeypatch):
    svc, store = _service(tmp_path)
    calls = []
    monkeypatch.setattr(svc, "_analyze_message", lambda message, context, **kwargs: calls.append(message.message_id) or [])
    t0 = _ts("2026-09-18", 10)
    first = _observation_from_messages([("m1", "mummy", t0, "meeting tomorrow at 6", {"from_me": False, "sender": "Mummy"})])
    svc.process_observation(first, observed_at=_ts("2026-09-18", 12), conversation_hint="mummy")
    restarted = WhatsAppIntelligence(WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3"), can_observe=lambda: True)
    later = _observation_from_messages([
        ("m1", "mummy", t0, "meeting tomorrow at 6", {"from_me": False, "sender": "Mummy"}),
        ("m2", "mummy", _ts("2026-09-19", 9), "assignment due Friday", {"from_me": False, "sender": "Mummy"}),
    ])
    restarted.store.enable(now=_ts("2026-09-19", 12))
    more_calls = []
    monkeypatch.setattr(restarted, "_analyze_message", lambda message, context, **kwargs: more_calls.append(message.message_id) or [])
    restarted.process_observation(later, observed_at=_ts("2026-09-19", 12), conversation_hint="mummy")
    assert more_calls == ["m2"]
    assert restarted.store.sync_checkpoint("mummy").last_source_message_id == "m2"


def test_up_to_date_live_cycle_does_not_invoke_semantic_processor(tmp_path, monkeypatch):
    svc, _ = _service(tmp_path)
    calls = []
    monkeypatch.setattr(svc, "_analyze_message", lambda message, context, **kwargs: calls.append(message.message_id) or [])
    obs = _structured_observation(native_id="once", ts=_ts("2026-09-19", 10), body="meeting tomorrow at 6")
    for minute in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
        svc.process_observation(obs, observed_at=_ts("2026-09-19", 12, minute), conversation_hint="mummy")
    assert calls == ["once"]


def test_semantic_persistence_creates_one_event_across_repeated_observations(tmp_path):
    svc, store = _service(tmp_path)
    obs = _structured_observation(native_id="meet-1", ts=_ts("2026-09-19", 10), body="Meeting tomorrow at 6 PM")
    first = svc.process_observation(obs, observed_at=_ts("2026-09-19", 12), conversation_hint="mummy")
    for minute in (1, 2, 3):
        svc.process_observation(obs, observed_at=_ts("2026-09-19", 12, minute), conversation_hint="mummy")
    events = store.list_events()
    assert len([e for e in events if e.type is IntelligenceEventType.MEETING]) == 1
    assert first and first[0].source_message_ids == ("meet-1",)


def test_failed_message_blocks_checkpoint_from_passing_it(tmp_path, monkeypatch):
    svc, store = _service(tmp_path)
    attempts = []
    def fail_first(message, context, **kwargs):
        attempts.append(message.message_id)
        if message.message_id == "bad":
            raise RuntimeError("persistent semantic failure")
        return []
    monkeypatch.setattr(svc, "_analyze_message", fail_first)
    payload = _observation_from_messages([
        ("bad", "mummy", _ts("2026-09-19", 10), "meeting tomorrow at 6", {"from_me": False, "sender": "Mummy"}),
        ("later", "mummy", _ts("2026-09-19", 11), "assignment due Friday", {"from_me": False, "sender": "Mummy"}),
    ])
    svc.process_observation(payload, observed_at=_ts("2026-09-19", 12), conversation_hint="mummy")
    assert attempts == ["bad"]
    checkpoint = store.sync_checkpoint("mummy")
    assert checkpoint is None
    svc.process_observation(payload, observed_at=_ts("2026-09-19", 12) + 0.5, conversation_hint="mummy")
    assert attempts == ["bad"]
    assert store.sync_checkpoint("mummy") is None


def test_failed_message_bounded_retry_and_other_chat_can_progress(tmp_path, monkeypatch):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    svc = WhatsAppIntelligence(store, can_observe=lambda: True)
    svc.enable(now=_ts("2026-09-19", 12))
    attempts = []
    def analyze(message, context, **kwargs):
        attempts.append(message.message_id)
        if message.message_id == "bad":
            raise RuntimeError("bad")
        return []
    monkeypatch.setattr(svc, "_analyze_message", analyze)
    payload = _observation_from_messages([
        ("bad", "mummy", _ts("2026-09-19", 10), "meeting tomorrow at 6", {"from_me": False, "sender": "Mummy"}),
        ("good", "team", _ts("2026-09-19", 11), "assignment due Friday", {"from_me": False, "sender": "Team"}),
    ])
    svc.process_observation(payload, observed_at=_ts("2026-09-19", 12), conversation_hint="")
    assert attempts == ["bad", "good"]
    assert store.sync_checkpoint("mummy") is None
    assert store.sync_checkpoint("team") is not None


def test_event_update_reuses_existing_meeting_event(tmp_path):
    svc, store = _service(tmp_path)
    first = _structured_observation(native_id="meet-a", ts=_ts("2026-09-19", 10), body="Meeting tomorrow at 6 PM")
    second = _structured_observation(native_id="meet-b", ts=_ts("2026-09-19", 11), body="Actually make it 7 PM")
    # Ensure the second source is available in context by observing both.
    svc.process_observation(first, observed_at=_ts("2026-09-19", 12), conversation_hint="mummy")
    svc.process_observation({"url": second["url"], "elements": first["elements"] + second["elements"]}, observed_at=_ts("2026-09-19", 12, 1), conversation_hint="mummy")
    meetings = [e for e in store.list_events() if e.type is IntelligenceEventType.MEETING]
    assert len(meetings) == 1
    assert meetings[0].event_id
    assert meetings[0].time == "19:00"


def test_semantic_world_state_is_retrievable_without_source_transcript(tmp_path, monkeypatch):
    svc, _ = _service(tmp_path)
    memory_path = tmp_path / "memory.sqlite3"
    monkeypatch.setenv("DEIMOS_AGENT_MEMORY", str(memory_path))
    obs = _structured_observation(native_id="mem-1", ts=_ts("2026-09-19", 10), body="Meeting tomorrow at 6 PM")
    svc.process_observation(obs, observed_at=_ts("2026-09-19", 12), conversation_hint="mummy")
    mem = svc._semantic_memory_store()
    results = mem.search("meeting tomorrow", limit=5)
    assert results
    assert any("Meeting" in r.content for r in results)
    assert all("Meeting tomorrow at 6 PM" not in r.content for r in results)


def test_general_intelligence_window_is_today_and_yesterday_only(tmp_path):
    svc, _ = _service(tmp_path)
    now = _ts("2026-09-19", 12)
    today = WhatsAppMessage("today", "mummy", _ts("2026-09-19", 9), "we decided on YOLO")
    yesterday = WhatsAppMessage("yesterday", "mummy", _ts("2026-09-18", 9), "we decided on YOLO")
    old = WhatsAppMessage("old", "mummy", _ts("2026-09-17", 23), "we decided on YOLO")
    assert svc._general_eligible(today, now)
    assert svc._general_eligible(yesterday, now)
    assert not svc._general_eligible(old, now)


def test_active_future_semantic_memory_survives_source_retention_window(tmp_path, monkeypatch):
    svc, _ = _service(tmp_path)
    memory_path = tmp_path / "memory.sqlite3"
    monkeypatch.setenv("DEIMOS_AGENT_MEMORY", str(memory_path))
    event_time = _ts("2026-11-19", 18)
    from agent_control.whatsapp_intelligence import IntelligenceEvent
    event = IntelligenceEvent(
        event_id="evt-future", conversation_id="mummy", type=IntelligenceEventType.MEETING,
        title="Principal meeting", description="Principal meeting", status="CONFIRMED", confidence=0.94,
        importance=0.9, urgency=0.7, created_at=_ts("2026-09-19", 10), updated_at=_ts("2026-09-19", 10),
        event_time=event_time, date="2026-11-19", time="18:00", source_message_ids=("src-future",),
    )
    message = WhatsAppMessage("src-future", "mummy", _ts("2026-09-19", 10), "Meeting on November 19 at 6 PM")
    svc._persist_semantic_memory(event, message)
    mem = svc._semantic_memory_store()
    record = mem.search("principal meeting", limit=5)[0]
    assert record.expires_at is not None and record.expires_at > event_time
