import os
import re
import tempfile
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import agent_control.whatsapp_intelligence as wi
from agent_control.whatsapp_intelligence import (
    IntelligenceEventType,
    WhatsAppIntelligence,
    WhatsAppIntelligenceStore,
    WhatsAppMessage,
)

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")


def _message(
    message_id: str,
    text: str,
    *,
    timestamp: float | None = None,
    conversation: str = "mummy",
    direction: str = "unknown",
    sender: str = "",
    **metadata,
) -> WhatsAppMessage:
    if timestamp is None:
        timestamp = datetime(2026, 9, 18, 19, 18, tzinfo=UTC).timestamp()
    meta = dict(metadata)
    meta.setdefault("direction", direction)
    meta.setdefault("sender_evidence", "NONE")
    meta.setdefault("is_outgoing", None)
    return WhatsAppMessage(
        message_id=message_id,
        conversation_id=conversation,
        timestamp=float(timestamp),
        text=text,
        sender=sender,
        metadata=meta,
    )


def _service():
    fd, path = tempfile.mkstemp(suffix=".sqlite3")
    os.close(fd)
    store = WhatsAppIntelligenceStore(path)
    debug: list[str] = []
    service = WhatsAppIntelligence(store, on_debug=debug.append)
    return store, service, debug, path


def _close(store, path):
    store.close()
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _event_for(service, message):
    return service._event(
        message,
        IntelligenceEventType.TASK,
        "Task",
        "semantic state",
        "CONFIRMED",
        0.90,
    )


# ---------------------------------------------------------------------------
# CHECKPOINT / INCORPORATION
# ---------------------------------------------------------------------------


def test_checkpoint_observed_failure_shape_with_already_ahead_frontier_now_reconciles(monkeypatch):
    """Exact former failure: receipts and semantic changes succeeded but an already-ahead frontier made the old comparison report false."""
    store, service, debug, path = _service()
    now = datetime(2026, 9, 18, 20, 0, tzinfo=IST).timestamp()
    messages = [
        _message("older-1", "assignment one", timestamp=now - 300),
        _message("older-2", "assignment two", timestamp=now - 240),
    ]
    # Seed the persisted frontier ahead of the newly observed messages. The old
    # claim-time UPDATE could not move the frontier backwards, and the old
    # state-diff logic consequently reported checkpoint_advanced=false.
    newer = _message("frontier", "already processed", timestamp=now - 60)
    store.claim_message(newer, True)
    store._run_write(
        "seed_checkpoint",
        lambda: store._conn().execute(
            "UPDATE intelligence_state SET last_processed_message_timestamp=?, last_processed_message_id=? WHERE id=1",
            (newer.timestamp, newer.message_id),
        ),
    )
    monkeypatch.setattr(wi, "extract_messages", lambda *_a, **_k: list(messages))
    monkeypatch.setattr(service, "_analyze_message", lambda m, _context: [_event_for(service, m)])
    service.enable(now=now)

    service.process_observation(object(), observed_at=now, conversation_hint="mummy")
    sync = "\n".join(line for line in debug if line.startswith("WHATSAPP_SYNC:"))
    assert "source_delta=2" in sync
    assert "incorporated=2" in sync
    assert "semantic_changes=2" in sync
    assert "failed=0" in sync
    assert "retry_deferred=0" in sync
    assert "checkpoint_advanced=true" in sync
    assert store.state().last_processed_message_id == "frontier"
    _close(store, path)


def test_checkpoint_observed_failure_shape_now_advances(monkeypatch):
    """Regression for source_delta=2/incorporated=2/semantic_changes=2/false."""
    store, service, debug, path = _service()
    now = datetime(2026, 9, 18, 20, 0, tzinfo=IST).timestamp()
    messages = [
        _message("m1", "assignment one", timestamp=now - 120),
        _message("m2", "assignment two", timestamp=now - 60),
    ]
    monkeypatch.setattr(wi, "extract_messages", lambda *_a, **_k: list(messages))
    monkeypatch.setattr(service, "_analyze_message", lambda m, _context: [_event_for(service, m)])
    service.enable(now=now)

    service.process_observation(object(), observed_at=now, conversation_hint="mummy")
    sync = "\n".join(line for line in debug if line.startswith("WHATSAPP_SYNC:"))
    assert "source_delta=2" in sync
    assert "incorporated=2" in sync
    assert "semantic_changes=2" in sync
    assert "failed=0" in sync
    assert "retry_deferred=0" in sync
    assert "checkpoint_advanced=true" in sync
    state = store.state()
    assert state.last_processed_message_id == "m2"
    _close(store, path)


def test_unknown_ignore_is_incorporated_and_advances_checkpoint(monkeypatch):
    store, service, debug, path = _service()
    now = datetime(2026, 9, 18, 20, 0, tzinfo=IST).timestamp()
    msg = _message("ignore-1", "okay", timestamp=now - 30, direction="unknown")
    monkeypatch.setattr(wi, "extract_messages", lambda *_a, **_k: [msg])
    monkeypatch.setattr(service, "_analyze_message", lambda *_a, **_k: [])
    service.enable(now=now)
    events = service.process_observation(object(), observed_at=now, conversation_hint="mummy")
    assert events == []
    assert store.state().last_processed_message_id == "ignore-1"
    assert any("incorporated=1" in line and "semantic_changes=0" in line and "checkpoint_advanced=true" in line for line in debug)
    _close(store, path)


def test_unknown_update_is_incorporated_and_advances_checkpoint(monkeypatch):
    store, service, debug, path = _service()
    now = datetime(2026, 9, 18, 20, 0, tzinfo=IST).timestamp()
    msg = _message("update-1", "updated plan", timestamp=now - 20, direction="unknown")
    monkeypatch.setattr(wi, "extract_messages", lambda *_a, **_k: [msg])
    monkeypatch.setattr(service, "_analyze_message", lambda m, _context: [_event_for(service, m)])
    service.enable(now=now)
    service.process_observation(object(), observed_at=now, conversation_hint="mummy")
    assert store.state().last_processed_message_id == "update-1"
    assert any("incorporated=1" in line and "semantic_changes=1" in line and "checkpoint_advanced=true" in line for line in debug)
    _close(store, path)


def test_processing_failure_preserves_retry_barrier_and_blocks_checkpoint(monkeypatch):
    store, service, debug, path = _service()
    now = datetime(2026, 9, 18, 20, 0, tzinfo=IST).timestamp()
    msg = _message("fail-1", "assignment that fails", timestamp=now - 10, direction="unknown")
    monkeypatch.setattr(wi, "extract_messages", lambda *_a, **_k: [msg])
    def fail(_m, _context):
        raise RuntimeError("semantic processor failure")
    monkeypatch.setattr(service, "_analyze_message", fail)
    service.enable(now=now)
    service.process_observation(object(), observed_at=now, conversation_hint="mummy")
    state = store.state()
    assert state.last_processed_message_id is None
    assert any("checkpoint_advanced=false" in line and "failed=1" in line for line in debug)
    assert any("WHATSAPP_SYNC_ERROR:" in line for line in debug)
    assert store.is_message_processed("fail-1") is False
    _close(store, path)


def test_successful_live_cycle_has_zero_delta_and_no_terminal_output(monkeypatch):
    store, service, debug, path = _service()
    now = datetime(2026, 9, 18, 20, 0, tzinfo=IST).timestamp()
    msg = _message("live-1", "okay", timestamp=now - 10, direction="unknown")
    monkeypatch.setattr(wi, "extract_messages", lambda *_a, **_k: [msg])
    monkeypatch.setattr(service, "_analyze_message", lambda *_a, **_k: [])
    service.enable(now=now)
    service.process_observation(object(), observed_at=now, conversation_hint="mummy")
    first_debug_count = len(debug)
    service.process_observation(object(), observed_at=now + 2, conversation_hint="mummy")
    tail = debug[first_debug_count:]
    assert not any(line.startswith("WHATSAPP_SYNC:") for line in tail)
    assert not any(line.startswith("WHATSAPP_SYNC_ERROR:") for line in tail)
    _close(store, path)


def test_multiple_successful_messages_advance_boundary_once(monkeypatch):
    store, service, debug, path = _service()
    now = datetime(2026, 9, 18, 20, 0, tzinfo=IST).timestamp()
    messages = [
        _message("multi-1", "first", timestamp=now - 180),
        _message("multi-2", "second", timestamp=now - 120),
        _message("multi-3", "third", timestamp=now - 60),
    ]
    monkeypatch.setattr(wi, "extract_messages", lambda *_a, **_k: list(messages))
    monkeypatch.setattr(service, "_analyze_message", lambda m, _context: [])
    service.enable(now=now)
    service.process_observation(object(), observed_at=now, conversation_hint="mummy")
    assert store.state().last_processed_message_id == "multi-3"
    syncs = [line for line in debug if line.startswith("WHATSAPP_SYNC:")]
    assert len(syncs) == 1
    _close(store, path)


def test_unknown_direction_reaches_semantic_processor(monkeypatch):
    store, service, _debug, path = _service()
    now = datetime(2026, 9, 18, 20, 0, tzinfo=IST).timestamp()
    msg = _message("unknown-1", "okay", timestamp=now - 5, direction="unknown")
    calls = []
    monkeypatch.setattr(wi, "extract_messages", lambda *_a, **_k: [msg])
    monkeypatch.setattr(service, "_analyze_message", lambda m, _context: calls.append(m.message_id) or [])
    service.enable(now=now)
    service.process_observation(object(), observed_at=now, conversation_hint="mummy")
    assert calls == ["unknown-1"]
    _close(store, path)


# ---------------------------------------------------------------------------
# DIRECTION
# ---------------------------------------------------------------------------


def test_message_scoped_sender_evidence_resolves_incoming():
    msg = wi.extract_messages({
        "message_id": "in-1",
        "conversation_id": "mummy",
        "timestamp": "2026-09-18T19:18:00+00:00",
        "text": "hello",
        "sender": "Jyoti",
    })[0]
    assert msg.metadata["direction"] == "incoming"
    assert msg.metadata["sender_evidence"] == "sender=Jyoti"


def test_message_scoped_from_me_resolves_outgoing():
    msg = wi.extract_messages({
        "message_id": "out-1",
        "conversation_id": "mummy",
        "timestamp": "2026-09-18T19:18:00+00:00",
        "text": "hello",
        "from_me": True,
    })[0]
    assert msg.metadata["direction"] == "outgoing"
    assert msg.metadata["sender_evidence"] == "from_me=True"


def test_self_chat_from_me_is_outgoing():
    msg = wi.extract_messages({
        "message_id": "self-1",
        "conversation_id": "You",
        "timestamp": "2026-09-18T19:18:00+00:00",
        "text": "note to self",
        "sender": "You",
        "from_me": True,
    })[0]
    assert msg.metadata["direction"] == "outgoing"
    assert msg.metadata["sender_evidence"] in {"from_me=True", "sender=You"}


def test_missing_direction_is_unknown_but_message_still_processable(monkeypatch):
    store, service, _debug, path = _service()
    now = datetime(2026, 9, 18, 20, 0, tzinfo=IST).timestamp()
    raw_message = {
        "message_id": "unk-process",
        "conversation_id": "mummy",
        "timestamp": now - 1,
        "text": "okay",
    }
    extracted = wi.extract_messages(raw_message)[0]
    assert extracted.metadata["direction"] == "unknown"
    calls = []
    monkeypatch.setattr(wi, "extract_messages", lambda *_a, **_k: [extracted])
    monkeypatch.setattr(service, "_analyze_message", lambda m, _context: calls.append(m.message_id) or [])
    service.enable(now=now)
    service.process_observation(object(), observed_at=now, conversation_hint="mummy")
    assert calls == ["unk-process"]
    assert store.state().last_processed_message_id == "unk-process"
    _close(store, path)


def test_unrelated_you_page_text_cannot_assign_direction():
    assert wi._semantic_element_ownership({"role": "heading", "name": "You"})[0] == "unknown"
    assert wi._direction_evidence({"role": "heading", "name": "You"}) == "NONE"


def test_direction_metadata_does_not_change_stable_source_identity():
    base = dict(conversation_id="mummy", timestamp=1726694280.0, sender="", text="hello")
    incoming = wi._message_identity(**base, metadata={"direction": "incoming"})
    outgoing = wi._message_identity(**base, metadata={"direction": "outgoing"})
    unknown = wi._message_identity(**base, metadata={"direction": "unknown"})
    assert incoming == outgoing == unknown


def test_repeated_unknown_direction_diagnostic_is_not_emitted_again(monkeypatch):
    store, service, debug, path = _service()
    now = datetime(2026, 9, 18, 20, 0, tzinfo=IST).timestamp()
    msg = _message("diag-1", "okay", timestamp=now - 1, direction="unknown")
    monkeypatch.setattr(wi, "extract_messages", lambda *_a, **_k: [msg])
    monkeypatch.setattr(service, "_analyze_message", lambda *_a, **_k: [])
    service.enable(now=now)
    service.process_observation(object(), observed_at=now, conversation_hint="mummy")
    service.process_observation(object(), observed_at=now + 2, conversation_hint="mummy")
    assert sum(line.startswith("WHATSAPP_DEBUG: direction_evidence_missing") for line in debug) == 1
    _close(store, path)


# ---------------------------------------------------------------------------
# LOGGING / OBSERVER ERROR CATEGORIES
# ---------------------------------------------------------------------------


def test_new_source_produces_compact_sync_output(monkeypatch):
    store, service, debug, path = _service()
    now = datetime(2026, 9, 18, 20, 0, tzinfo=IST).timestamp()
    msg = _message("new-1", "okay", timestamp=now - 1)
    monkeypatch.setattr(wi, "extract_messages", lambda *_a, **_k: [msg])
    monkeypatch.setattr(service, "_analyze_message", lambda *_a, **_k: [])
    service.enable(now=now)
    service.process_observation(object(), observed_at=now, conversation_hint="mummy")
    assert any(line.startswith("WHATSAPP_SYNC:") and "source_delta=1" in line and "status=SYNC_COMPLETE" in line for line in debug)
    _close(store, path)


def test_actual_failure_produces_sync_error(monkeypatch):
    store, service, debug, path = _service()
    now = datetime(2026, 9, 18, 20, 0, tzinfo=IST).timestamp()
    msg = _message("err-1", "bad task", timestamp=now - 1)
    monkeypatch.setattr(wi, "extract_messages", lambda *_a, **_k: [msg])
    monkeypatch.setattr(service, "_analyze_message", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    service.enable(now=now)
    service.process_observation(object(), observed_at=now, conversation_hint="mummy")
    assert any(line.startswith("WHATSAPP_SYNC_ERROR:") and "semantic_processing_failed" in line for line in debug)
    _close(store, path)


def test_observer_error_categories_are_specific():
    assert wi._observer_error_category(RuntimeError("tool.scroll_to not implemented in extension")) == "scroll_to_unsupported"
    assert wi._observer_error_category(RuntimeError("wheel fallback failed")) == "wheel_fallback_failed"
    assert wi._observer_error_category(RuntimeError("browser connection failed")) == "browser_connection_failed"
    assert wi._observer_error_category(RuntimeError("execution exception")) == "execution_exception"
    assert wi._observer_error_category(RuntimeError("BrowserSkillError")) == "browser_observation_failed"


# ---------------------------------------------------------------------------
# EVENT TIME / TIMEZONE
# ---------------------------------------------------------------------------


def _event_message(text: str, *, timestamp: str = "2026-09-18T19:18:00+00:00", metadata=None):
    return WhatsAppMessage(
        message_id="time-test",
        conversation_id="mummy",
        timestamp=datetime.fromisoformat(timestamp).timestamp(),
        text=text,
        sender="Mummy",
        metadata=metadata or {},
    )


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
def test_explicit_event_clock_is_contextually_selected(text, expected):
    store, service, _debug, path = _service()
    try:
        proposal = service._event_proposal(_event_message(text))
        assert proposal.resolved_time == expected
        assert proposal.timezone == "Asia/Kolkata"
    finally:
        _close(store, path)


def test_default_event_timezone_is_asia_kolkata(monkeypatch):
    monkeypatch.delenv("DEIMOS_EVENT_TIMEZONE", raising=False)
    store, service, _debug, path = _service()
    try:
        proposal = service._event_proposal(_event_message("Meeting at 6 PM", metadata={"source_timezone": "UTC"}))
        assert proposal.timezone == "Asia/Kolkata"
    finally:
        _close(store, path)


def test_timezone_precedence_explicit_then_metadata_then_env_then_default(monkeypatch):
    store, service, _debug, path = _service()
    try:
        monkeypatch.setenv("DEIMOS_EVENT_TIMEZONE", "UTC")
        msg = _event_message("Meeting at 6 PM", metadata={"event_timezone": "Asia/Kolkata"})
        assert service._event_proposal(msg).timezone == "Asia/Kolkata"
        explicit_utc = _event_message("Meeting at 6 PM UTC", metadata={"event_timezone": "Asia/Kolkata"})
        assert service._event_proposal(explicit_utc).timezone == "UTC"
        env_only = _event_message("Meeting at 6 PM")
        assert service._event_proposal(env_only).timezone == "UTC"
    finally:
        _close(store, path)


def test_tomorrow_at_6_pm_resolves_date_and_wall_clock_together():
    store, service, _debug, path = _service()
    try:
        message = _event_message("Meeting tomorrow at 6 PM", metadata={"source_timezone": "UTC"})
        proposal = service._event_proposal(message)
        source_local = datetime.fromtimestamp(message.timestamp, tz=IST)
        assert proposal.resolved_date == (source_local.date() + timedelta(days=1)).isoformat()
        assert proposal.resolved_time == "18:00"
        event = service._apply_proposal(message, proposal)
        assert event is not None
        displayed = datetime.fromtimestamp(event.event_time, tz=IST)
        assert displayed.date() == source_local.date() + timedelta(days=1)
        assert displayed.strftime("%H:%M") == "18:00"
    finally:
        _close(store, path)


def test_message_timestamp_does_not_override_explicit_event_time():
    store, service, _debug, path = _service()
    try:
        message = _event_message("Meeting at 6 PM", timestamp="2026-09-18T19:18:00+00:00", metadata={"source_timezone": "UTC"})
        proposal = service._event_proposal(message)
        assert proposal.resolved_time == "18:00"
    finally:
        _close(store, path)


def test_event_time_round_trip_asia_kolkata_through_utc_storage():
    store, service, _debug, path = _service()
    try:
        message = _event_message("Meeting today at 6 PM")
        proposal = service._event_proposal(message)
        event = service._apply_proposal(message, proposal)
        assert event is not None
        assert event.time == "18:00"
        assert event.timezone == "Asia/Kolkata"
        persisted = store.get_event(event.event_id)
        assert persisted is not None
        utc = datetime.fromtimestamp(persisted.event_time, tz=UTC)
        assert (utc.hour, utc.minute) == (12, 30)
        local = datetime.fromtimestamp(persisted.event_time, tz=IST)
        assert local.strftime("%H:%M") == "18:00"
    finally:
        _close(store, path)


def test_no_constant_plus_seven_shift():
    store, service, _debug, path = _service()
    try:
        assert service._event_proposal(_event_message("Meeting at 6 PM")).resolved_time == "18:00"
        assert service._event_proposal(_event_message("Meeting at 10 PM")).resolved_time == "22:00"
    finally:
        _close(store, path)


def test_legacy_bad_meeting_time_is_repaired_in_place():
    store, service, _debug, path = _service()
    try:
        message = _event_message("Meeting at 6 PM", timestamp="2026-09-18T19:18:00+00:00")
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
        assert store.repair_meeting_event_times() == 1
        events = store.list_events()
        assert len(events) == 1
        assert events[0].event_id == buggy.event_id
        assert events[0].time == "18:00"
        assert datetime.fromtimestamp(events[0].event_time, tz=IST).strftime("%H:%M") == "18:00"
    finally:
        _close(store, path)


def test_time_debug_does_not_leak_message_body():
    debug: list[str] = []
    store, service, _unused, path = _service()
    service.on_debug = debug.append
    try:
        service._event_proposal(_event_message("Private meeting details at 6 PM"))
        line = next(line for line in debug if line.startswith("WHATSAPP_TIME_DEBUG:"))
        assert "raw_time_expression='6 PM'" in line
        assert "Private meeting details" not in line
        assert "normalized_event_time=18:00" in line
    finally:
        _close(store, path)


# ---------------------------------------------------------------------------
# BROWSERSKILL SCROLL FALLBACK
# ---------------------------------------------------------------------------


def _home_obs(name: str):
    return SimpleNamespace(elements=[{"role": "listitem", "name": name}])


class _ScrollBrowser:
    def __init__(self, observations, *, scroll_error="tool.scroll_to not implemented in extension", wheel_ok=True):
        self.observations = list(observations)
        self.scroll_calls = 0
        self.wheel_calls = 0
        self.observe_calls = 0
        self.open_calls = 0
        self.scroll_error = scroll_error
        self.wheel_ok = wheel_ok

    def scroll_to(self, _target):
        self.scroll_calls += 1
        raise RuntimeError(self.scroll_error)

    def wheel(self, _delta):
        self.wheel_calls += 1
        return SimpleNamespace(ok=self.wheel_ok)

    def observe(self):
        self.observe_calls += 1
        if self.observations:
            return self.observations.pop(0)
        return _home_obs("other")

    def open_whatsapp_chat_row(self, _target, timeout_s=8.0):
        self.open_calls += 1
        return SimpleNamespace(ok=True)


def test_scroll_to_unsupported_uses_wheel_then_fresh_observation(monkeypatch):
    store, service, debug, path = _service()
    try:
        store.authorize_target("mummy")
        monkeypatch.setattr(wi, "extract_messages", lambda *_a, **_k: [])
        browser = _ScrollBrowser([_home_obs("mummy Today")])
        result = service._observe_authorized_targets(browser, _home_obs("other"))
        assert result == "mummy"
        assert browser.scroll_calls == 1
        assert browser.wheel_calls == 1
        assert browser.observe_calls >= 2
        assert browser.open_calls == 1
        assert "WHATSAPP_DEBUG: scroll_to_unsupported" in debug
        assert "WHATSAPP_DEBUG: wheel_fallback_attempted" in debug
    finally:
        _close(store, path)


def test_scroll_fallback_is_bounded(monkeypatch):
    store, service, debug, path = _service()
    try:
        store.authorize_target("mummy")
        monkeypatch.setattr(wi, "extract_messages", lambda *_a, **_k: [])
        observations = [_home_obs(f"other-{i}") for i in range(1, 4)]
        browser = _ScrollBrowser(observations)
        result = service._observe_authorized_targets(browser, _home_obs("other-0"))
        assert result is None
        assert browser.wheel_calls == wi._HOME_SCROLL_BUDGET
        assert browser.scroll_calls == 1
        assert "WHATSAPP_DEBUG: scroll_to_unsupported" in debug
        assert debug.count("WHATSAPP_DEBUG: wheel_fallback_attempted") == wi._HOME_SCROLL_BUDGET
    finally:
        _close(store, path)


def test_wheel_fallback_failure_is_exposed(monkeypatch):
    store, service, debug, path = _service()
    try:
        store.authorize_target("mummy")
        monkeypatch.setattr(wi, "extract_messages", lambda *_a, **_k: [])
        browser = _ScrollBrowser([_home_obs("mummy")], wheel_ok=False)
        result = service._observe_authorized_targets(browser, _home_obs("other"))
        assert result is None
        assert browser.wheel_calls == 1
        assert "WHATSAPP_DEBUG: wheel_fallback_failed" in debug
    finally:
        _close(store, path)


# ---------------------------------------------------------------------------
# HELPER: old malformed first-clock cases are explicitly protected
# ---------------------------------------------------------------------------


def test_event_time_legacy_helper_uses_canonical_timezone():
    base = datetime(2026, 9, 18, 19, 18, tzinfo=UTC).timestamp()
    value = wi._event_time_from_text("Meeting at 6 PM", base)
    assert value is not None
    assert datetime.fromtimestamp(value, tz=IST).strftime("%H:%M") == "18:00"
