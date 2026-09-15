from __future__ import annotations

import threading
import time

from agent_control.api import AgentResult, TaskStatus
from agent_control.response import Narrator
from agent_control.session import Session, Turn, UserTask


def _session() -> Session:
    return Session.build(speech=False, show_status=False)


def test_background_submission_returns_before_task_completion():
    session = _session()
    started = threading.Event()
    release = threading.Event()

    def fake_submit(raw, *, source="text"):
        started.set()
        release.wait(2)
        result = AgentResult(request=raw, task_id="fake", status=TaskStatus.SUCCESS)
        return Turn(UserTask(raw=raw, text=raw, source=source, task_id="fake"), "done", result)

    session.submit = fake_submit
    try:
        task_id = session.submit_background("open youtube")
        assert task_id
        assert started.wait(1)
        record = next(x for x in session.background_tasks() if x.task_id == task_id)
        assert record.state == "RUNNING"
        assert not record.future.done()
        release.set()
        deadline = time.time() + 2
        while time.time() < deadline and record.state != "COMPLETED":
            time.sleep(0.01)
        assert record.state == "COMPLETED"
        assert record.result is not None and record.result.ok
    finally:
        release.set()
        session.close()


def test_background_lane_serializes_tasks():
    session = _session()
    order = []
    lock = threading.Lock()

    def fake_submit(raw, *, source="text"):
        with lock:
            order.append(("start", raw))
        time.sleep(0.03)
        with lock:
            order.append(("finish", raw))
        result = AgentResult(request=raw, task_id=raw, status=TaskStatus.SUCCESS)
        return Turn(UserTask(raw=raw, text=raw, source=source, task_id=raw), "done", result)

    session.submit = fake_submit
    try:
        a = session.submit_background("open youtube")
        b = session.submit_background("scroll down")
        deadline = time.time() + 2
        while time.time() < deadline:
            states = {x.task_id: x.state for x in session.background_tasks()}
            if states.get(a) == "COMPLETED" and states.get(b) == "COMPLETED":
                break
            time.sleep(0.01)
        assert order == [
            ("start", "open youtube"),
            ("finish", "open youtube"),
            ("start", "scroll down"),
            ("finish", "scroll down"),
        ]
    finally:
        session.close()


def test_new_command_does_not_consume_pending_approval(monkeypatch):
    session = _session()
    from agent_control.session import PendingApproval

    session.pending_approval = PendingApproval(
        request="send hello to papa",
        action={"kind": "whatsapp_send_message", "params": {"recipient": "papa", "message": "hello"}},
        asked_at=time.time(),
    )

    fake_turn = Turn(
        UserTask(raw="open youtube", text="open youtube", task_id="open-youtube", status="accepted"),
        "queued",
        None,
    )
    monkeypatch.setattr("agent_control.api.resolve_task", lambda _text: None)
    monkeypatch.setattr("agent_control.session.classify_fast", lambda _text: type("R", (), {"route": None})())
    monkeypatch.setattr(session, "_general_action", lambda task: fake_turn)

    session.submit("open youtube")
    assert session.pending_approval is not None

    approved = Turn(
        UserTask(raw="yes", text="send hello to papa", task_id="approved", status="accepted"),
        "sent",
        AgentResult(request="send hello to papa", task_id="approved", status=TaskStatus.SUCCESS),
    )
    monkeypatch.setattr(session, "_run", lambda prepared: approved)
    session.submit("yes")
    assert session.pending_approval is None
    session.close()
