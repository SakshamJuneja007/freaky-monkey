from __future__ import annotations

import json

import pytest

from agent_control.planner.base import Usage
from agent_control.planner.openai_compat import OpenAICompatPlanner
from agent_control.types import Action, NeedUserInput
from agent_control.skills.messaging.skill import MessagingExecutor
from agent_control.skills.messaging.actions import MessagingAction, MessagingActionKind
from agent_control.skills.messaging.backend import BrowserMessagingBackend


class FakeClient:
    model = "test-model"

    def __init__(self, payload: dict):
        self.payload = payload
        self.usage = Usage()
        self.calls = []

    def chat_checked(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return json.dumps(self.payload), {}, None


def planner_with(payload: dict) -> OpenAICompatPlanner:
    return OpenAICompatPlanner(FakeClient(payload))


def test_compound_incomplete_whatsapp_never_becomes_browser_target():
    planner = planner_with({"actions": [], "done": False})
    goal = "send message to mummy first then send to papa then play song"

    with pytest.raises(NeedUserInput) as exc:
        planner.plan(goal, {}, [])

    assert "mummy" in exc.value.clarification.question.lower()
    assert "send message to mummy first then send to papa then play song" not in exc.value.clarification.question


def test_structured_compound_whatsapp_actions_are_separate():
    planner = planner_with(
        {
            "reasoning": "decomposed compound request",
            "done": False,
            "actions": [
                {
                    "kind": "whatsapp_send_message",
                    "params": {"recipient": "mummy", "message": "hello"},
                },
                {
                    "kind": "whatsapp_send_message",
                    "params": {"recipient": "papa", "message": "bye"},
                },
                {
                    "kind": "browser_play_song",
                    "params": {"query": "Do I Wanna Know?"},
                },
            ],
        }
    )

    step = planner.plan(
        "send hello to mummy then send bye to papa then play Do I Wanna Know?",
        {},
        [],
    )

    assert [a.kind for a in step.actions] == [
        "whatsapp_send_message",
        "whatsapp_send_message",
        "browser_play_song",
    ]
    assert step.actions[0].params == {"recipient": "mummy", "message": "hello"}
    assert step.actions[1].params == {"recipient": "papa", "message": "bye"}
    assert step.actions[2].params["query"] == "Do I Wanna Know?"


def test_planner_rejects_whole_goal_as_whatsapp_recipient():
    goal = "send hello to mummy then send bye to papa"
    planner = planner_with(
        {
            "done": False,
            "actions": [
                {
                    "kind": "whatsapp_send_message",
                    "params": {"recipient": goal, "message": "hello"},
                }
            ],
        }
    )

    with pytest.raises(NeedUserInput) as exc:
        planner.plan(goal, {}, [])

    assert "separate" in exc.value.clarification.question.lower()


def test_structured_whatsapp_executor_passes_only_recipient_and_message():
    class Backend:
        def __init__(self):
            self.calls = []

        def send_whatsapp(self, recipient, message):
            self.calls.append((recipient, message))
            return {"provider": "whatsapp"}

    backend = Backend()
    executor = MessagingExecutor(backend)
    action = MessagingAction(
        kind=MessagingActionKind.WHATSAPP_SEND,
        params={"recipient": "mummy", "message": "hello"},
    )

    result = executor.execute(action)

    assert result.ok is True
    assert backend.calls == [("mummy", "hello")]
    assert backend.calls[0][0] != "send hello to mummy then send bye to papa"


def test_whatsapp_missing_message_does_not_execute():
    class Backend:
        def __init__(self):
            self.called = False

        def send_whatsapp(self, recipient, message):
            self.called = True
            return {}

    planner = planner_with({"done": False, "actions": []})
    with pytest.raises(NeedUserInput):
        planner.plan("send a message to papa", {}, [])


def test_browser_messaging_backend_uses_only_structured_recipient():
    class Browser:
        def __init__(self):
            self.calls = []

        def send_whatsapp_message(self, recipient, message):
            self.calls.append((recipient, message))
            return {"ok": True}

    browser = Browser()
    backend = BrowserMessagingBackend(browser)
    backend.send_whatsapp("mummy", "hello")

    assert browser.calls == [("mummy", "hello")]


def test_browser_play_song_goal_is_not_accepted_as_search_query():
    goal = "send hello to mummy then play Do I Wanna Know?"
    planner = planner_with(
        {
            "done": False,
            "actions": [
                {
                    "kind": "browser_play_song",
                    "params": {"query": goal},
                }
            ],
        }
    )

    with pytest.raises(NeedUserInput) as exc:
        planner.plan(goal, {}, [])

    assert "song title" in exc.value.clarification.question.lower()
