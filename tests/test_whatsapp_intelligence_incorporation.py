from __future__ import annotations

import time

import pytest

from agent_control.whatsapp_intelligence import (
    IntelligenceEventType,
    WhatsAppIntelligence,
    WhatsAppIntelligenceStore,
    extract_messages,
)


def _engine(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "whatsapp.sqlite3")
    now = time.time()
    store.enable(now)
    debug = []
    engine = WhatsAppIntelligence(store, on_debug=debug.append)
    return engine, store, debug, now


def _raw(chat, text, timestamp, **extra):
    value = {"conversation_id": chat, "message": text, "timestamp": timestamp}
    value.update(extra)
    return value


def test_unknown_direction_reaches_semantic_processor(tmp_path):
    engine, store, debug, now = _engine(tmp_path)
    seen = []

    def processor(message, context):
        seen.append(message)
        return []

    engine._analyze_message = processor
    engine.process_observation(_raw("mummy", "ok", now), conversation_hint="mummy")

    assert len(seen) == 1
    assert seen[0].metadata["direction"] == "unknown"
    assert any("direction_evidence_missing" in line for line in debug)


def test_unknown_ignore_is_incorporated_and_advances_checkpoint(tmp_path):
    engine, store, debug, now = _engine(tmp_path)
    before = store.state()
    engine._analyze_message = lambda message, context: []

    events = engine.process_observation(_raw("mummy", "ok", now + 1), conversation_hint="mummy")

    after = store.state()
    assert events == []
    assert after.last_processed_message_timestamp == now + 1
    assert after.last_processed_message_timestamp != before.last_processed_message_timestamp
    assert "incorporated=1" in "\n".join(debug)
    assert "semantic_changes=0" in "\n".join(debug)
    assert "checkpoint_advanced=true" in "\n".join(debug)


def test_unknown_update_is_incorporated_and_advances_checkpoint(tmp_path):
    engine, store, debug, now = _engine(tmp_path)

    def update(message, context):
        return [engine._event(
            message,
            IntelligenceEventType.TASK,
            "Task",
            message.text,
            "CONFIRMED",
            0.84,
        )]

    engine._analyze_message = update
    events = engine.process_observation(
        _raw("mummy", "task: finish report", now + 2),
        conversation_hint="mummy",
    )

    assert len(events) == 1
    assert store.state().last_processed_message_timestamp == now + 2
    assert "incorporated=1" in "\n".join(debug)


def test_multiple_unknown_messages_advance_boundary(tmp_path):
    engine, store, debug, now = _engine(tmp_path)
    engine._analyze_message = lambda message, context: []

    raw = {
        "messages": [
            _raw("mummy", "ok", now + 1),
            _raw("mummy", "thanks", now + 2),
            _raw("mummy", "noted", now + 3),
        ]
    }
    engine.process_observation(raw, conversation_hint="mummy")

    state = store.state()
    assert state.last_processed_message_timestamp == now + 3
    assert "source_delta=3" in "\n".join(debug)
    assert "incorporated=3" in "\n".join(debug)
    assert "semantic_changes=0" in "\n".join(debug)
    assert "checkpoint_advanced=true" in "\n".join(debug)


def test_semantic_failure_does_not_advance_checkpoint(tmp_path):
    engine, store, debug, now = _engine(tmp_path)
    before = store.state()

    def fail(message, context):
        raise RuntimeError("semantic processor failed")

    engine._analyze_message = fail
    engine.process_observation(_raw("mummy", "needs processing", now + 4), conversation_hint="mummy")

    after = store.state()
    assert after.last_processed_message_timestamp == before.last_processed_message_timestamp
    assert not store.is_message_processed(extract_messages(_raw("mummy", "needs processing", now + 4))[0].message_id)
    assert "semantic_incorporation_failed" in "\n".join(debug)


def test_repeated_live_observation_has_zero_source_delta(tmp_path):
    engine, store, debug, now = _engine(tmp_path)
    engine._analyze_message = lambda message, context: []

    raw = _raw("mummy", "ok", now + 5)
    engine.process_observation(raw, conversation_hint="mummy")
    debug.clear()

    engine.process_observation(raw, conversation_hint="mummy")

    assert "source_delta=0" not in "\n".join(debug)
    assert not debug or all("source_delta" not in line for line in debug)


def test_direction_metadata_does_not_change_stable_source_identity(tmp_path):
    ts = time.time()
    incoming = _raw("mummy", "hello", ts, direction="incoming")
    unknown = _raw("mummy", "hello", ts, direction="unknown")
    outgoing = _raw("mummy", "hello", ts, direction="outgoing")

    ids = [extract_messages(item)[0].message_id for item in (incoming, unknown, outgoing)]
    assert ids[0] == ids[1] == ids[2]


def test_self_chat_without_direction_is_processable(tmp_path):
    engine, store, debug, now = _engine(tmp_path)
    seen = []
    engine._analyze_message = lambda message, context: (seen.append(message) or [])

    engine.process_observation(
        _raw("You", "ok", now + 6),
        conversation_hint="You",
    )

    assert len(seen) == 1
    assert seen[0].metadata["direction"] == "unknown"
    assert store.state().last_processed_message_timestamp == now + 6
