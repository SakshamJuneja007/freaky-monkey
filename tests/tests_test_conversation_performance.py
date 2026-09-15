from __future__ import annotations

import time
from types import SimpleNamespace

from agent_control.response import Narrator
from agent_control.session import Session
import pytest


@pytest.fixture(autouse=True)
def _without_benchmark_registry(monkeypatch):
    monkeypatch.setattr("agent_control.session.api.resolve_task", lambda text: None)
    monkeypatch.setattr("agent_control.session.api.parse_request", lambda text: None)


class StreamingClient:
    def __init__(self):
        self.calls = 0

    def chat_stream(self, messages, **kwargs):
        self.calls += 1
        yield "Hey"
        time.sleep(0.01)
        yield ". What's up?"


class StreamingConversation:
    def __init__(self):
        self.client = StreamingClient()
        self.last_metrics = {}

    def reply_stream(self, *args, **kwargs):
        yield from self.client.chat_stream([])
        self.last_metrics = {"context_build_s": 0.001, "model_request_s": 0.011, "ttft_s": 0.001, "generation_s": 0.010, "total_s": 0.011}


def _session():
    return Session(narrator=Narrator.build(enabled=False), runtime_persistence_path=":memory:")


def test_simple_conversation_does_not_touch_runtime_or_persistent_memory(monkeypatch):
    session = _session()
    session._conversation = SimpleNamespace(
        client=object(),
        reply=lambda *a, **k: "Hey.",
    )
    monkeypatch.setattr(session, "runtime_snapshot", lambda: (_ for _ in ()).throw(AssertionError("runtime queried")))
    class Memory:
        def search(self, *args, **kwargs):
            raise AssertionError("persistent memory searched")
        def append_turn(self, turn):
            pass
    session._conversation_memory_store = Memory()

    turn = session.submit("hey")

    assert turn.task.intent.value == "CONVERSATION"
    assert turn.result is None
    assert session._runtime.list_tasks() == []


def test_streaming_conversation_renders_chunks_before_completion():
    output = []
    session = Session(narrator=Narrator.build(enabled=False, write=output.append), runtime_persistence_path=":memory:")
    session._conversation = StreamingConversation()

    turn = session.submit("hey")

    assert turn.reply == "Hey. What's up?"
    assert output[0] == "Hey"
    assert output[1] == ". What's up?"
    assert output[-1] == "\n"


def test_no_speak_does_not_invoke_tts(monkeypatch):
    session = _session()
    session._conversation = SimpleNamespace(
        client=object(),
        reply=lambda *a, **k: "Hey.",
    )
    assert session.narrator.speaker is not None
    assert session.narrator.speaker.enabled is False
    session.submit("hey")
    assert session.narrator.speaker.said == []


def test_conversation_history_still_reaches_model():
    session = _session()
    seen = {}

    class FakeConversation:
        client = object()

        def reply(self, message, history=None, recent_context=None, memories=None):
            seen["history"] = list(history or [])
            return "Blue."

    session._conversation = FakeConversation()
    session.submit("my favorite color is blue")
    session.submit("what color did I just mention?")

    assert len(seen["history"]) == 1
    assert seen["history"][0].task.text == "my favorite color is blue"


def test_action_routing_is_not_swallowed_by_conversation(monkeypatch):
    session = _session()
    monkeypatch.setattr("agent_control.session.api.resolve_task", lambda text: None)
    monkeypatch.setattr("agent_control.session.api.parse_request", lambda text: None)
    monkeypatch.setattr(session, "_general_action", lambda task, **kwargs: SimpleNamespace(task=task, reply="task", result=SimpleNamespace(ok=True)))

    turn = session.submit("send hello to papa")

    assert turn.task.intent.value == "ACTION"


def test_runtime_control_is_not_conversation(monkeypatch):
    session = _session()
    monkeypatch.setattr(session, "_handle_runtime_controls", lambda controls, **kwargs: SimpleNamespace(task=SimpleNamespace(task_id="task-1")))

    result = session.submit("approve task 7")

    assert result.task.task_id == "task-1"
