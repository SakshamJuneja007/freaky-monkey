from __future__ import annotations

import time

from agent_control.skills.browser.backend import BrowserElement, BrowserObservation, BrowserSkillAdapter, BrowserSkillError
from agent_control.whatsapp_intelligence import WhatsAppIntelligence, WhatsAppIntelligenceStore, extract_messages


def _obs(elements, *, generation=1, session="s1"):
    return BrowserObservation(
        generation=generation,
        session=session,
        tab_id=1,
        url="https://web.whatsapp.com/",
        text="\n".join(f'{e.ref} {e.role} "{e.name}"' for e in elements),
        elements=tuple(elements),
        raw={"elements": []},
    )


def _message_element(ref, name, timestamp, **ownership):
    raw = {"message_id": ref, "timestamp": timestamp, **ownership}
    return BrowserElement(ref, role="text", name=name, raw=raw, attributes=ownership)


def test_live_noop_synchronization_is_silent(tmp_path):
    now = time.time()
    message = _message_element("m1", "Meeting tomorrow at 6", now - 10, from_me=False, sender="Mummy")
    observation = _obs([
        BrowserElement("h1", role="heading", name="mummy"),
        message,
    ])
    logs: list[str] = []
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    service = WhatsAppIntelligence(store, on_debug=logs.append)
    service.enable(now=now - 20)
    analyze_calls: list[str] = []
    original_analyze = service._analyze_message
    def spy(message, context):
        analyze_calls.append(message.message_id)
        return original_analyze(message, context)
    service._analyze_message = spy
    service.process_observation(observation, observed_at=now, conversation_hint="mummy")
    logs.clear()
    service.process_observation(observation, observed_at=now + 1, conversation_hint="mummy")

    assert analyze_calls == ["m1"]
    assert not any(line.startswith("WHATSAPP_SYNC:") for line in logs)


def test_new_source_emits_sync_log(tmp_path):
    now = time.time()
    message = _message_element("m1", "Meeting tomorrow at 6", now - 10, from_me=False, sender="Mummy")
    logs: list[str] = []
    store = WhatsAppIntelligenceStore(tmp_path / "wa.sqlite3")
    service = WhatsAppIntelligence(store, on_debug=logs.append)
    service.enable(now=now - 20)
    service.process_observation(_obs([
        BrowserElement("h1", role="heading", name="mummy"),
        message,
    ]), observed_at=now, conversation_hint="mummy")

    sync = "\n".join(line for line in logs if line.startswith("WHATSAPP_SYNC:"))
    assert sync
    assert "source_delta=1" in sync
    assert "checkpoint_advanced=true" in sync


def test_direction_from_message_scoped_from_me():
    now = time.time()
    msgs = extract_messages(
        _obs([
            BrowserElement("h1", role="heading", name="mummy"),
            _message_element("m1", "Meeting tomorrow at 6", now - 10, from_me=True),
        ]),
        conversation_hint="mummy",
        observed_at=now,
    )
    assert len(msgs) == 1
    assert msgs[0].metadata["direction"] == "outgoing"
    assert msgs[0].metadata["is_outgoing"] is True
    assert msgs[0].metadata["sender_evidence"] == "from_me=True"


def test_direction_from_message_scoped_sender():
    now = time.time()
    msgs = extract_messages(
        _obs([
            BrowserElement("h1", role="heading", name="mummy"),
            _message_element("m1", "Meeting tomorrow at 6", now - 10, sender="Mummy"),
        ]),
        conversation_hint="mummy",
        observed_at=now,
    )
    assert msgs[0].metadata["direction"] == "incoming"
    assert msgs[0].metadata["is_outgoing"] is False
    assert msgs[0].metadata["sender_evidence"] == "sender=Mummy"


def test_actual_browserskill_semantic_text_preserves_message_scoped_direction():
    now = time.time()
    elements = BrowserSkillAdapter._parse_observation_text(
        '@e1 heading "mummy"\n@e2 text "Meeting tomorrow at 6 PM 09:30" [message_id=m1] [from_me=true]'
    )
    observation = BrowserObservation(
        generation=1, session="s1", tab_id=1, url="https://web.whatsapp.com/",
        text="", elements=tuple(elements), raw={},
    )
    msgs = extract_messages(observation, conversation_hint="mummy", observed_at=now)
    assert msgs[0].message_id == "m1"
    assert msgs[0].metadata["direction"] == "outgoing"
    assert msgs[0].metadata["sender_evidence"] == "from_me=true"


def test_self_chat_current_user_sender_is_outgoing_without_hardcoding_chat_name():
    now = time.time()
    msgs = extract_messages(
        _obs([
            BrowserElement("h1", role="heading", name="saksham (You)"),
            _message_element("m1", "Meeting tomorrow at 6", now - 10, sender="You"),
        ]),
        conversation_hint="saksham",
        observed_at=now,
    )
    assert msgs[0].metadata["direction"] == "outgoing"
    assert msgs[0].metadata["is_outgoing"] is True


def test_unrelated_page_level_you_does_not_assign_direction():
    now = time.time()
    msgs = extract_messages(
        _obs([
            BrowserElement("h1", role="heading", name="mummy"),
            BrowserElement("u1", role="text", name="You"),
            _message_element("m1", "Meeting tomorrow at 6", now - 10),
        ]),
        conversation_hint="mummy",
        observed_at=now,
    )
    assert msgs[0].metadata["direction"] == "unknown"
    assert msgs[0].metadata["is_outgoing"] is None
    assert msgs[0].metadata["sender_evidence"] == "NONE"


def test_missing_ownership_remains_unknown():
    now = time.time()
    msgs = extract_messages(
        _obs([
            BrowserElement("h1", role="heading", name="mummy"),
            _message_element("m1", "Meeting tomorrow at 6", now - 10),
        ]),
        conversation_hint="mummy",
        observed_at=now,
    )
    assert msgs[0].metadata["direction"] == "unknown"
    assert msgs[0].metadata["sender_evidence"] == "NONE"


def test_conflicting_message_ownership_fails_closed():
    now = time.time()
    msgs = extract_messages(
        _obs([
            BrowserElement("h1", role="heading", name="mummy"),
            _message_element("m1", "Meeting tomorrow at 6", now - 10, from_me=True, sender="Mummy"),
        ]),
        conversation_hint="mummy",
        observed_at=now,
    )
    assert msgs[0].metadata["direction"] == "unknown"
    assert msgs[0].metadata["sender_evidence"].startswith("CONFLICT:")


def test_unsupported_scroll_to_falls_back_to_bounded_wheel(tmp_path):
    now = time.time()
    logs: list[str] = []
    first = _obs([BrowserElement("r1", role="listitem", name="other")], generation=1)
    second = _obs([BrowserElement("r2", role="listitem", name="mummy 10:00")], generation=2)

    class FakeStore:
        def set_debug_callback(self, _callback):
            pass

        def authorized_targets(self):
            return ("mummy",)

    store = FakeStore()
    service = WhatsAppIntelligence(store, on_debug=logs.append)
    service._home_scroll_budget = 2

    class FakeBrowser:
        def __init__(self):
            self.observations = [first, second]
            self.observe_calls = 0
            self.scroll_to_calls = 0
            self.wheel_calls: list[int] = []

        def observe(self):
            self.observe_calls += 1
            return second

        def scroll_to(self, _target):
            self.scroll_to_calls += 1
            raise BrowserSkillError("tool.scroll_to not implemented in extension", code="browser_action_failed")

        def wheel(self, delta_y):
            self.wheel_calls.append(int(delta_y))
            return {"ok": True}

        def open_whatsapp_chat_row(self, _target, timeout_s=8.0):
            return {"ok": True}

    browser = FakeBrowser()
    result = service._observe_authorized_targets(browser, first, allow_scroll=True)

    assert result == "mummy"
    assert browser.scroll_to_calls == 1
    assert browser.wheel_calls == [560]
    assert browser.observe_calls >= 1
    assert any("SCROLL_TO_UNSUPPORTED_FALLBACK=wheel" in line for line in logs)


def test_scroll_to_is_not_retried_after_capability_failure(tmp_path):
    first = _obs([BrowserElement("r1", role="listitem", name="other")], generation=1)
    second = _obs([BrowserElement("r2", role="listitem", name="other2")], generation=2)

    class FakeStore:
        def set_debug_callback(self, _callback):
            pass

        def authorized_targets(self):
            return ("mummy",)

    service = WhatsAppIntelligence(FakeStore())
    service._home_scroll_budget = 2

    class FakeBrowser:
        def __init__(self):
            self.observe_calls = 0
            self.scroll_to_calls = 0
            self.wheel_calls = 0

        def observe(self):
            self.observe_calls += 1
            return second

        def scroll_to(self, _target):
            self.scroll_to_calls += 1
            raise BrowserSkillError("tool.scroll_to not implemented in extension", code="browser_action_failed")

        def wheel(self, _delta_y):
            self.wheel_calls += 1
            return {"ok": True}

    browser = FakeBrowser()
    service._observe_authorized_targets(browser, first, allow_scroll=True)
    assert browser.scroll_to_calls == 1
    assert 1 <= browser.wheel_calls <= service._home_scroll_budget


def test_non_unsupported_scroll_to_error_does_not_use_wheel():
    first = _obs([BrowserElement("r1", role="listitem", name="other")], generation=1)

    class FakeStore:
        def set_debug_callback(self, _callback):
            pass

        def authorized_targets(self):
            return ("mummy",)

    service = WhatsAppIntelligence(FakeStore())
    service._home_scroll_budget = 1

    class FakeBrowser:
        def __init__(self):
            self.scroll_to_calls = 0
            self.wheel_calls = 0

        def scroll_to(self, _target):
            self.scroll_to_calls += 1
            raise BrowserSkillError("permission denied", code="browser_action_failed")

        def wheel(self, _delta_y):
            self.wheel_calls += 1
            return {"ok": True}

    browser = FakeBrowser()
    service._observe_authorized_targets(browser, first, allow_scroll=True)
    assert browser.scroll_to_calls == 1
    assert browser.wheel_calls == 0
