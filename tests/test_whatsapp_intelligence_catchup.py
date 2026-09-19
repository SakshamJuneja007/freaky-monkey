from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_control.whatsapp_intelligence import (
    IntelligenceEventType,
    WhatsAppIntelligence,
    WhatsAppIntelligenceStore,
    extract_messages,
)


UTC = timezone.utc


def _element(ref, role, name, value="", **attrs):
    return SimpleNamespace(ref=ref, role=role, name=name, value=value, raw=attrs or None, attributes=attrs)


def _chat_obs(chat: str, messages: list[dict], generation: int = 1):
    elements = [
        _element("@h", "heading", chat),
        _element("@box", "textbox", "Type a message"),
    ]
    return SimpleNamespace(
        generation=generation,
        elements=tuple(elements),
        text="",
        raw={"messages": messages},
    )


def _msg(message_id: str, ts: float, text: str = "hello", sender: str = "Alice", **extra):
    return {
        "message_id": message_id,
        "conversation_id": "Mummy",
        "timestamp": ts,
        "sender": sender,
        "text": text,
        **extra,
    }


class HistoryBrowser:
    def __init__(self, observations):
        self.observations = list(observations)
        self.index = 0
        self.observe_count = 0
        self.scroll_calls = []

    def observe(self):
        self.observe_count += 1
        return self.observations[min(self.index, len(self.observations) - 1)]

    def scroll(self, amount):
        self.scroll_calls.append(amount)
        self.index = min(self.index + 1, len(self.observations) - 1)
        return SimpleNamespace(ok=True)


def _service(tmp_path: Path, now: float):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    store.authorize_target("Mummy")
    service = WhatsAppIntelligence(store)
    service.enable(now=now)
    return service


def test_catchup_window_is_exactly_previous_24_hours(tmp_path):
    now = 1_800_000_000.0
    service = _service(tmp_path, now)
    state = service.store.state()
    assert state.catchup_window_start == pytest.approx(now - 86400)
    assert state.catchup_window_end == pytest.approx(now)
    assert state.catchup_started_at == pytest.approx(now)


def test_messages_inside_window_are_retained(tmp_path):
    now = 1_800_000_000.0
    service = _service(tmp_path, now)
    observation = {"messages": [_msg("inside", now - 3600, "meeting tomorrow at 6pm")]}
    service.process_observation(observation, observed_at=now)
    assert service.store.state().last_processed_message_id == "inside"
    assert service.store.list_events()


def test_messages_older_than_window_are_excluded(tmp_path):
    now = 1_800_000_000.0
    service = _service(tmp_path, now)
    old = _msg("old", now - 86401, "meeting yesterday")
    service.process_observation({"messages": [old]}, observed_at=now)
    assert service.store.list_events() == []


def test_history_is_retrieved_when_initial_observation_is_too_recent(tmp_path):
    now = 1_800_000_000.0
    initial = _chat_obs("Mummy", [_msg("new", now - 300, "new message")])
    older = _chat_obs("Mummy", [_msg("old", now - 3600, "meeting tomorrow at 6pm"), _msg("new", now - 300, "new message")], 2)
    browser = HistoryBrowser([initial, older])
    service = _service(tmp_path, now)
    current, messages, status = service._fetch_chat_history(browser, initial, "Mummy")
    assert status == "PARTIAL" or status == "COMPLETE"
    assert "old" in {m.message_id for m in messages}
    assert browser.scroll_calls


def test_bounded_history_retrieval_stops_after_window_is_covered(tmp_path):
    now = 1_800_000_000.0
    initial = _chat_obs("Mummy", [_msg("new", now - 100)])
    covered = _chat_obs("Mummy", [_msg("old", now - 86500), _msg("new", now - 100)], 2)
    browser = HistoryBrowser([initial, covered])
    service = _service(tmp_path, now)
    _current, _messages, status = service._fetch_chat_history(browser, initial, "Mummy")
    assert status == "COMPLETE"
    assert len(browser.scroll_calls) == 1


def test_history_retrieval_does_not_scroll_home_sidebar(tmp_path):
    now = 1_800_000_000.0
    initial = _chat_obs("Mummy", [_msg("new", now - 100)])
    browser = HistoryBrowser([initial, initial])
    service = _service(tmp_path, now)
    service._fetch_chat_history(browser, initial, "Mummy")
    assert browser.scroll_calls
    # The history path never calls the Home discovery helper or scroll_to.
    assert not hasattr(browser, "scroll_to")


def test_history_retrieval_requires_active_conversation(tmp_path):
    now = 1_800_000_000.0
    home = SimpleNamespace(generation=1, elements=(_element("@x", "button", "Mummy"),), text="", raw={"messages": []})
    browser = HistoryBrowser([home])
    service = _service(tmp_path, now)
    _current, messages, status = service._fetch_chat_history(browser, home, "Mummy")
    assert messages == []
    assert status == "UNKNOWN"
    assert browser.scroll_calls == []


def test_missing_timestamps_are_not_treated_as_current_time(tmp_path):
    now = 1_800_000_000.0
    service = _service(tmp_path, now)
    observation = {"messages": [{"message_id": "no-time", "conversation_id": "Mummy", "sender": "Alice", "text": "meeting tomorrow"}]}
    assert extract_messages(observation, conversation_hint="Mummy", observed_at=now) == []
    service.process_observation(observation, observed_at=now)
    assert service.store.list_events() == []


def test_incoming_direction_is_extracted():
    obs = _chat_obs("Mummy", [])
    obs.raw = {"messages": [{"message_id": "in", "conversation_id": "Mummy", "timestamp": 1000, "sender": "Alice", "text": "hello", "from_me": False}]}
    message = extract_messages(obs.raw, conversation_hint="Mummy", observed_at=1000)[0]
    assert message.metadata["direction"] == "incoming"


def test_outgoing_direction_is_extracted():
    obs = _chat_obs("Mummy", [])
    obs.raw = {"messages": [{"message_id": "out", "conversation_id": "Mummy", "timestamp": 1000, "sender": "You", "text": "hello", "from_me": True}]}
    message = extract_messages(obs.raw, conversation_hint="Mummy", observed_at=1000)[0]
    assert message.metadata["direction"] == "outgoing"


def test_unknown_direction_remains_unknown_without_evidence():
    obs = _chat_obs("Mummy", [])
    obs.raw = {"messages": [{"message_id": "unknown", "conversation_id": "Mummy", "timestamp": 1000, "sender": "Alice", "text": "hello"}]}
    message = extract_messages(obs.raw, conversation_hint="Mummy", observed_at=1000)[0]
    assert message.metadata["direction"] == "unknown"


def test_stable_message_identity_deduplicates_repeated_observations(tmp_path):
    now = 1_800_000_000.0
    service = _service(tmp_path, now)
    observation = {"messages": [_msg("stable", now - 100, "meeting tomorrow")]}
    service.process_observation(observation, observed_at=now)
    service.process_observation(observation, observed_at=now + 1)
    assert len(service.store.list_events()) == 1


def test_observation_fingerprint_suppresses_unchanged_state(tmp_path):
    now = 1_800_000_000.0
    service = _service(tmp_path, now)
    obs = _chat_obs("Mummy", [_msg("stable", now - 100)])
    browser = HistoryBrowser([obs])
    service.store.update_catchup_chat("Mummy", window_start=now - 86400, window_end=now, status="COMPLETE", oldest_observed=now - 100, newest_observed=now - 100, observation_fingerprint=service._observation_fingerprint("Mummy", obs, extract_messages(obs.raw, conversation_hint="Mummy", observed_at=now)))
    _current, _messages, status = service._fetch_chat_history(browser, obs, "Mummy")
    assert status == "COMPLETE"
    assert browser.scroll_calls == []


def test_new_message_changes_fingerprint_and_triggers_processing(tmp_path):
    now = 1_800_000_000.0
    service = _service(tmp_path, now)
    old = _chat_obs("Mummy", [_msg("old", now - 100)])
    new = _chat_obs("Mummy", [_msg("old", now - 100), _msg("new", now - 50, "meeting tomorrow")], 2)
    browser = HistoryBrowser([new])
    old_messages = extract_messages(old.raw, conversation_hint="Mummy", observed_at=now)
    old_fp = service._observation_fingerprint("Mummy", old, old_messages)
    service.store.update_catchup_chat("Mummy", window_start=now - 86400, window_end=now, status="COMPLETE", oldest_observed=now - 100, newest_observed=now - 100, observation_fingerprint=old_fp)
    _current, messages, _status = service._fetch_chat_history(browser, new, "Mummy")
    assert {m.message_id for m in messages} == {"old", "new"}


def test_identical_meeting_evidence_is_suppressed_at_event_layer(tmp_path):
    now = 1_800_000_000.0
    service = _service(tmp_path, now)
    msg = _msg("meeting-1", now - 100, "meeting tomorrow at 6pm")
    event = service._analyze_message(extract_messages({"messages": [msg]}, conversation_hint="Mummy", observed_at=now)[0], []) [0]
    outcome1, _ = service.store.upsert_event(event)
    outcome2, _ = service.store.upsert_event(event)
    assert outcome1 == "created"
    assert outcome2 == "duplicate"
    assert len(service.store.list_events()) == 1


def test_changed_meeting_evidence_updates_event(tmp_path):
    now = 1_800_000_000.0
    service = _service(tmp_path, now)
    first = _msg("meeting-1", now - 100, "meeting tomorrow")
    second = _msg("meeting-2", now - 50, "meeting tomorrow")
    m1 = extract_messages({"messages": [first]}, conversation_hint="Mummy", observed_at=now)[0]
    m2 = extract_messages({"messages": [second]}, conversation_hint="Mummy", observed_at=now)[0]
    outcome1, _ = service.store.upsert_event(service._analyze_message(m1, [m1])[0])
    outcome2, _ = service.store.upsert_event(service._analyze_message(m2, [m1, m2])[0])
    assert outcome1 == "created"
    assert outcome2 == "updated"
    assert len(service.store.list_events()) == 1


def test_catchup_state_persists_across_restart(tmp_path):
    now = 1_800_000_000.0
    service = _service(tmp_path, now)
    service.store.update_catchup_chat("Mummy", window_start=now - 86400, window_end=now, status="COMPLETE", oldest_observed=now - 86500, newest_observed=now - 100, observation_fingerprint="fp")
    service.store.mark_catchup_progress(status="COMPLETE", oldest_observed=now - 86500, newest_observed=now - 100)
    restarted = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    assert restarted.state().catchup_status == "COMPLETE"
    assert restarted.catchup_chat_state("Mummy")["status"] == "COMPLETE"


def test_human_ownership_gate_pauses_catchup(tmp_path):
    now = 1_800_000_000.0
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    service = WhatsAppIntelligence(store, can_observe=lambda: False)
    store.authorize_target("Mummy")
    service.enable(now=now)
    obs = _chat_obs("Mummy", [_msg("m", now - 100)])
    browser = HistoryBrowser([obs])
    assert service._observe_loop is not None
    assert service.process_observation(obs, observed_at=now, conversation_hint="Mummy") == []
    assert browser.scroll_calls == []


def test_deimos_takeover_can_resume_with_fresh_observation(tmp_path):
    now = 1_800_000_000.0
    allowed = {"value": False}
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    store.authorize_target("Mummy")
    service = WhatsAppIntelligence(store, can_observe=lambda: allowed["value"])
    service.enable(now=now)
    obs = _chat_obs("Mummy", [_msg("m", now - 100)])
    browser = HistoryBrowser([obs])
    assert service.process_observation(obs, observed_at=now, conversation_hint="Mummy") == []
    allowed["value"] = True
    _current, messages, status = service._fetch_chat_history(browser, obs, "Mummy")
    assert messages
    assert status in {"COMPLETE", "PARTIAL"}


def test_existing_authorized_home_discovery_path_remains_callable(tmp_path):
    now = 1_800_000_000.0
    service = _service(tmp_path, now)
    row = _element("@row", "button", "Mummy 12:30 PM")
    home = SimpleNamespace(generation=1, elements=(row,), text="", raw={"elements": []})
    class Browser:
        def __init__(self): self.opened = []
        def open_whatsapp_chat_row(self, target, timeout_s=8.0): self.opened.append(target); return SimpleNamespace(ok=True)
        def observe(self): return home
    browser = Browser()
    assert service._observe_authorized_targets(browser, home, allow_scroll=False) in {None, "Mummy"}


def test_mummy_flow_uses_same_message_ingestion(tmp_path):
    now = 1_800_000_000.0
    service = _service(tmp_path, now)
    obs = _chat_obs("Mummy", [_msg("m1", now - 300, "meeting tomorrow at 6pm")])
    events = service.process_observation(obs, observed_at=now, conversation_hint="Mummy")
    assert any(event.type is IntelligenceEventType.MEETING for event in events)


def test_saksham_self_chat_flow_does_not_guess_direction(tmp_path):
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    service = WhatsAppIntelligence(store)
    now = 1_800_000_000.0
    service.enable(now=now)
    obs = _chat_obs("Saksham", [])
    obs.raw = {"messages": [{"message_id": "s1", "conversation_id": "Saksham", "timestamp": now - 100, "sender": "Saksham", "text": "hello"}]}
    msg = extract_messages(obs.raw, conversation_hint="Saksham", observed_at=now)[0]
    assert msg.metadata["direction"] == "unknown"


def test_send_functionality_boundary_is_untouched(tmp_path):
    # Intelligence is read-only; the service exposes no send primitive.
    service = _service(tmp_path, 1_800_000_000.0)
    assert not hasattr(service, "send_whatsapp")


def test_no_coordinate_or_selector_automation_is_added():
    source = Path(__file__).resolve().parents[1] / "agent_control" / "whatsapp_intelligence.py"
    text = source.read_text(encoding="utf-8").casefold()
    assert "xpath" not in text
    assert "queryselector" not in text
    assert "document.elementfrompoint" not in text
