from __future__ import annotations

from types import MethodType, SimpleNamespace

from agent_control.presentation import (
    approval_prompt,
    completion_message,
    is_explicit_task_id_request,
    recovery_message,
    runtime_snapshot_for_user,
    sanitize_tts_text,
)
from agent_control.session import InputEvent, InputOwner, Session


class FakeNarrator:
    def __init__(self):
        self.notes: list[str] = []
        self.replies: list[str] = []

    def note(self, text: str) -> None:
        self.notes.append(text)

    def reply(self, text: str) -> None:
        self.replies.append(text)


def bare_session() -> Session:
    session = object.__new__(Session)
    session.narrator = FakeNarrator()
    session.debug = True
    session.pending_approval = None
    session.pending = None
    session._thread_state = SimpleNamespace(background=False)
    return session


def test_01_internal_task_id_remains_in_debug_output():
    session = bare_session()
    session._debug_input(
        InputEvent("open youtube"), InputOwner.TASK, InputOwner.TASK,
        task_created=True, task_id="fast-3c5e",
    )
    assert any("fast-3c5e" in line for line in session.narrator.notes)
    assert any("STATE:" in line or "ROUTE:" in line for line in session.narrator.notes)


def test_02_internal_task_id_does_not_enter_normal_tts():
    session = bare_session()
    session._emit_reply("fast-3c5e is waiting for user", goal="send message to Mummy", state="WAITING_FOR_USER")
    assert session.narrator.replies == ["The WhatsApp message is waiting for your approval."]
    assert "fast-3c5e" not in session.narrator.replies[0]


def test_03_input_id_is_not_spoken():
    assert "input-17faf45432" not in sanitize_tts_text(
        "INPUT: id=input-17faf45432 source=text"
    )


def test_04_internal_state_names_are_not_spoken():
    spoken = sanitize_tts_text(
        "fast-43a4 is RECOVERY_REQUIRED.",
        goal="play do i wanna know",
        state="RECOVERY_REQUIRED",
    )
    assert spoken == "The music task was interrupted, so I need to recover it before continuing."
    assert "RECOVERY_REQUIRED" not in spoken
    assert "fast-43a4" not in spoken


def test_05_approval_prompt_is_natural():
    spoken = approval_prompt({
        "kind": "whatsapp_send_message",
        "params": {"recipient": "mummy", "message": "hello"},
    }, "send message to mummy saying hello")
    assert spoken == "Do you want me to send the WhatsApp message to Mummy?"
    assert "whatsapp_send_message" not in spoken


def test_06_recovery_message_is_natural():
    spoken = recovery_message("play do i wanna know")
    assert spoken == "The music task was interrupted, so I need to recover it before continuing."
    assert "RECOVERY_REQUIRED" not in spoken


def test_07_completion_message_is_natural():
    spoken = completion_message("send message to Papa")
    assert spoken == "The WhatsApp message was completed."
    assert "COMPLETED" not in spoken


def test_08_explicit_task_id_request_may_return_the_id():
    assert is_explicit_task_id_request("What is the task ID?")
    session = bare_session()
    session.runtime_snapshot = MethodType(
        lambda self: {
            "focused_task_id": "fast-3c5e",
            "active_tasks": [{"task_id": "fast-3c5e", "goal": "send message to Mummy", "state": "WAITING_FOR_APPROVAL"}],
            "failed_tasks": [],
        },
        session,
    )
    reply = session._runtime_query_reply("What is the task ID?")
    assert reply == "The task ID is fast-3c5e."


def test_09_debug_output_does_not_cause_debug_strings_to_enter_tts():
    session = bare_session()
    session._debug_input(
        InputEvent("open youtube"), InputOwner.TASK, InputOwner.TASK,
        task_created=True, task_id="fast-3c5e",
    )
    session._emit_reply("The task is running.")
    assert any("fast-3c5e" in line for line in session.narrator.notes)
    assert all("fast-3c5e" not in reply for reply in session.narrator.replies)
    assert all("ROUTE:" not in reply for reply in session.narrator.replies)


def test_10_raw_runtime_objects_cannot_become_tts_input():
    raw_runtime = {
        "task_id": "fast-3c5e",
        "state": "WAITING_FOR_USER",
        "event_id": "event-abc123",
    }
    spoken = sanitize_tts_text(raw_runtime, goal="send message to Mummy", state="WAITING_FOR_USER")
    assert spoken == "I have an internal runtime result, but there is nothing user-facing to say yet."
    assert "fast-3c5e" not in spoken


def test_11_internal_metadata_is_converted_to_clean_speech():
    spoken = sanitize_tts_text(
        "TASK: fast-43a4 STATE: RECOVERY_REQUIRED ROUTE: TASK OWNER: TASK",
        goal="play do i wanna know",
        state="RECOVERY_REQUIRED",
    )
    assert spoken == "The music task was interrupted, so I need to recover it before continuing."
    assert not any(token in spoken for token in ("fast-43a4", "TASK:", "STATE:", "ROUTE:", "OWNER:"))


def test_12_normal_conversation_remains_unaffected():
    text = "Absolutely — the quickest way is to start with the smallest example."
    assert sanitize_tts_text(text) == text


def test_user_runtime_projection_strips_identifiers_but_keeps_semantics():
    projected = runtime_snapshot_for_user({
        "active_tasks": [{
            "task_id": "fast-3c5e",
            "goal": "send message to Mummy",
            "state": "WAITING_FOR_APPROVAL",
            "verification_state": "UNKNOWN",
            "failure_state": "",
        }],
        "waiting_tasks": [],
        "completed_tasks": [],
        "failed_tasks": [],
        "cancelled_tasks": [],
    })
    assert projected["active_tasks"] == [{
        "goal": "send message to Mummy",
        "state": "waiting for approval",
        "verification": "not known",
        "failure": "",
    }]
    assert "task_id" not in projected["active_tasks"][0]
