from __future__ import annotations

import time

import httpx

from agent_control.conversation import ConversationTransportError
from agent_control.response import Narrator
from agent_control.session import Session, Turn


class SequencedConversation:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def reply(self, *args, **kwargs):
        self.calls += 1
        value = self.replies.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    def action_reply(self, *args, **kwargs):
        return "ok"


def _session(tmp_path):
    return Session(
        narrator=Narrator.build(enabled=False),
        runtime_persistence_path=tmp_path / "runtime.sqlite3",
    )



import pytest


@pytest.fixture(autouse=True)
def _conversation_classification_without_benchmark(monkeypatch):
    monkeypatch.setattr("agent_control.session.api.resolve_task", lambda text: None)
    monkeypatch.setattr("agent_control.session.api.parse_request", lambda text: None)


def _wait_for_history(session, size, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if len(session.history) >= size:
            return
        time.sleep(0.01)
    assert len(session.history) >= size


def test_conversation_succeeds_normally(tmp_path):
    session = _session(tmp_path)
    fake = SequencedConversation(["hello back"])
    session._conversation = fake

    turn = session.submit("hi")

    assert turn.task.intent.value == "CONVERSATION"
    assert turn.reply == "hello back"
    assert session._runtime.list_tasks() == []


def test_conversation_timeout_is_safe_and_creates_no_task(tmp_path):
    session = _session(tmp_path)
    fake = SequencedConversation([ConversationTransportError("conversation_transport_timeout: ReadTimeout")])
    session._conversation = fake

    turn = session.submit("hi")

    assert "conversation_transport_timeout" in turn.reply
    assert turn.result is None
    assert session._runtime.list_tasks() == []


def test_conversation_timeout_resets_background_lane_and_next_request_succeeds(tmp_path):
    session = _session(tmp_path)
    fake = SequencedConversation([
        ConversationTransportError("conversation_transport_timeout: ReadTimeout"),
        "second request works",
    ])
    session._conversation = fake

    session.submit_background("hi")
    _wait_for_history(session, 1)
    assert "conversation_transport_timeout" in session.history[0].reply
    assert session._conversation_executor is None

    session.submit_background("hey")
    _wait_for_history(session, 2)
    assert session.history[1].reply == "second request works"
    assert session._runtime.list_tasks() == []


def test_multiple_conversation_requests_remain_usable_after_timeout(tmp_path):
    session = _session(tmp_path)
    fake = SequencedConversation([
        ConversationTransportError("conversation_transport_timeout: ReadTimeout"),
        "two",
        "three",
        "four",
    ])
    session._conversation = fake

    for text in ("one", "two", "three", "four"):
        session.submit_background(text)
    _wait_for_history(session, 4)

    replies = [turn.reply for turn in session.history]
    assert "conversation_transport_timeout" in replies[0]
    assert replies[1:] == ["two", "three", "four"]
    assert all(turn.result is None for turn in session.history)
    assert session._runtime.list_tasks() == []


def test_transport_failure_does_not_fall_into_task_execution(tmp_path, monkeypatch):
    session = _session(tmp_path)
    fake = SequencedConversation([ConversationTransportError("conversation_transport_timeout: ReadTimeout")])
    session._conversation = fake
    called = []
    monkeypatch.setattr(session, "_run", lambda prepared: called.append(prepared) or Turn(task=prepared.task, reply="bad", result=None))

    turn = session.submit("hi")

    assert "conversation_transport_timeout" in turn.reply
    assert called == []
    assert session._runtime.list_tasks() == []


def test_real_httpx_read_timeout_is_classified_as_conversation_timeout():
    class FakeClient:
        def chat_checked(self, *args, **kwargs):
            raise httpx.ReadTimeout("provider timed out")

    from agent_control.conversation import ConversationEngine

    engine = ConversationEngine(client=FakeClient())
    try:
        engine.reply("hi")
    except ConversationTransportError as exc:
        assert str(exc).startswith("conversation_transport_timeout:")
    else:
        raise AssertionError("ReadTimeout was not classified as a conversation transport timeout")
