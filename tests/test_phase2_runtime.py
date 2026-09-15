from __future__ import annotations

import threading
import time

from agent_control.api import AgentResult, TaskStatus
from agent_control.response import Narrator
from agent_control.runtime import RuntimeManager
from agent_control.session import PendingApproval, Session, Turn
from dataclasses import replace


def manager() -> RuntimeManager:
    return RuntimeManager()


def test_01_task_receives_stable_id():
    rt = manager()
    task = rt.create_task("send message to papa saying hello", task_id="fast-9dc1")
    assert task.task_id == "fast-9dc1"
    assert rt.get_task("fast-9dc1").task_id == "fast-9dc1"


def test_02_task_can_be_retrieved_by_id():
    rt = manager()
    rt.create_task("play Do I Wanna Know", task_id="fast-5436")
    assert rt.get_task("fast-5436").goal == "play Do I Wanna Know"


def test_03_state_transition_updates_registry():
    rt = manager()
    rt.create_task("open youtube", task_id="fast-a")
    rt.transition_task("fast-a", "PLANNING")
    rt.transition_task("fast-a", "RUNNING")
    rt.transition_task("fast-a", "VERIFYING")
    assert rt.get_task("fast-a").state == "VERIFYING"


def test_04_approval_belongs_to_exactly_one_task():
    rt = manager()
    rt.create_task("send papa hello", task_id="fast-a")
    rt.request_approval("fast-a", {"kind": "whatsapp_send_message", "params": {"recipient": "papa", "message": "hello"}})
    assert rt.get_approval("fast-a") is not None
    assert rt.get_approval("fast-b") is None
    assert [x["task_id"] for x in rt.snapshot()["pending_approvals"]] == ["fast-a"]


def test_05_yes_does_not_create_new_task():
    rt = manager()
    rt.create_task("send papa hello", task_id="fast-a")
    rt.request_approval("fast-a", {"kind": "whatsapp_send_message"})
    before = {t.task_id for t in rt.list_tasks()}
    rt.resolve_approval("fast-a", True)
    after = {t.task_id for t in rt.list_tasks()}
    assert after == before == {"fast-a"}


def test_06_approval_continuation_retains_original_id(monkeypatch):
    s = Session(narrator=Narrator.build(enabled=False))
    s._runtime.create_task("send papa hello", task_id="whatsapp-send-1", task_type="messaging")
    s._runtime.request_approval("whatsapp-send-1", {"kind": "whatsapp_send_message"})
    s.pending_approval = PendingApproval(
        request="send papa hello",
        action={"kind": "whatsapp_send_message", "params": {"recipient": "papa", "message": "hello"}},
        asked_at=time.time(), task_id="whatsapp-send-1", goal="send papa hello", runtime_task_id="whatsapp-send-1",
    )

    class DummyBrowser:
        pass

    monkeypatch.setattr(s, "_browser_for_task", lambda *a, **k: DummyBrowser())
    seen = []

    def fake_run(prepared):
        seen.append(prepared.task.task_id)
        return Turn(prepared.task, "done", AgentResult(request=prepared.task.text, task_id=prepared.task.task_id, status=TaskStatus.SUCCESS, verified="PASS"))

    monkeypatch.setattr(s, "_run", fake_run)
    s.submit("yes")
    assert seen == ["whatsapp-send-1"]
    assert [t.task_id for t in s._runtime.list_tasks()] == ["whatsapp-send-1"]
    assert s._runtime.get_task("whatsapp-send-1").state == "RUNNING"


def test_07_two_independent_tasks_coexist():
    rt = manager()
    rt.create_task("send whatsapp", task_id="fast-a")
    rt.create_task("play youtube", task_id="fast-b")
    rt.transition_task("fast-a", "RUNNING")
    rt.transition_task("fast-b", "RUNNING")
    assert {t.task_id for t in rt.list_active_tasks()} == {"fast-a", "fast-b"}


def test_08_focus_does_not_hide_other_active_tasks():
    rt = manager()
    rt.create_task("send whatsapp", task_id="fast-a")
    rt.create_task("play youtube", task_id="fast-b")
    rt.transition_task("fast-a", "RUNNING")
    rt.transition_task("fast-b", "RUNNING")
    rt.set_focus("fast-b")
    snap = rt.snapshot()
    assert snap["focused_task_id"] == "fast-b"
    assert {x["task_id"] for x in snap["active_tasks"]} == {"fast-a", "fast-b"}


def test_09_failed_task_remains_queryable():
    rt = manager()
    rt.create_task("send papa hello", task_id="fast-a")
    rt.transition_task("fast-a", "FAILED", failure_state="planner transport timeout")
    assert rt.get_task("fast-a").state == "FAILED"
    assert rt.list_failed_tasks()[0].task_id == "fast-a"


def test_10_original_goal_survives_failure():
    rt = manager()
    rt.create_task("send the exact greeting", task_id="fast-a")
    rt.transition_task("fast-a", "FAILED", failure_state="browser unavailable")
    task = rt.get_task("fast-a")
    assert task.goal == "send the exact greeting"
    assert task.failure_state == "browser unavailable"


def test_11_unknown_verification_remains_unknown():
    rt = manager()
    rt.create_task("play a video", task_id="fast-a")
    rt.transition_task("fast-a", "RUNNING")
    rt.transition_task("fast-a", "VERIFYING", verification_state="UNKNOWN")
    task = rt.get_task("fast-a")
    assert task.verification_state == "UNKNOWN"
    assert task.state == "VERIFYING"


def test_12_conversation_input_creates_no_executable_task(monkeypatch):
    s = Session(narrator=Narrator.build(enabled=False))
    monkeypatch.setattr(s, "_converse", lambda task: Turn(task, "Hey.", None))
    monkeypatch.setattr("agent_control.session.api.resolve_task", lambda _text: None)
    s.submit("hey deimos")
    assert s.runtime_snapshot()["active_tasks"] == []


def test_13_snapshot_reports_active_tasks():
    rt = manager()
    rt.create_task("play youtube", task_id="fast-a")
    rt.transition_task("fast-a", "RUNNING")
    assert rt.snapshot()["active_tasks"][0]["task_id"] == "fast-a"


def test_14_snapshot_reports_waiting_approval():
    rt = manager()
    rt.create_task("send whatsapp", task_id="fast-a")
    rt.request_approval("fast-a", {"kind": "whatsapp_send_message"})
    assert rt.snapshot()["waiting_tasks"][0]["state"] == "WAITING_FOR_APPROVAL"
    assert rt.snapshot()["pending_approvals"][0]["task_id"] == "fast-a"


def test_15_snapshot_reports_failed_tasks():
    rt = manager()
    rt.create_task("bad task", task_id="fast-a")
    rt.transition_task("fast-a", "FAILED", failure_state="timeout")
    assert rt.snapshot()["failed_tasks"][0]["failure_state"] == "timeout"


def test_16_browser_resource_association_belongs_to_task():
    rt = manager()
    rt.create_task("play youtube", task_id="fast-a")
    rt.attach_resource("fast-a", "youtube", site="youtube", state="ATTACHED")
    assert rt.resource_for_task("fast-a")["task_id"] == "fast-a"
    assert rt.get_task("fast-a").browser_resource_key == "youtube"


def test_17_whatsapp_and_youtube_resources_are_independent():
    rt = manager()
    rt.create_task("send whatsapp", task_id="fast-wa")
    rt.create_task("play youtube", task_id="fast-yt")
    rt.attach_resource("fast-wa", "whatsapp", site="whatsapp")
    rt.attach_resource("fast-yt", "youtube", site="youtube")
    resources = {x["resource_key"]: x["task_id"] for x in rt.snapshot()["resources"]}
    assert resources == {"whatsapp": "fast-wa", "youtube": "fast-yt"}


def test_18_concurrent_state_updates_do_not_corrupt_registry():
    rt = manager()
    rt.create_task("concurrent task", task_id="fast-a")

    def worker():
        for _ in range(50):
            rt.transition_task("fast-a", "RUNNING")
            rt.transition_task("fast-a", "VERIFYING")
            rt.transition_task("fast-a", "RUNNING")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert rt.get_task("fast-a") is not None
    assert rt.get_task("fast-a").state == "RUNNING"


def test_19_tasks_command_reads_runtime_registry(monkeypatch, capsys):
    import main
    s = Session(narrator=Narrator.build(enabled=False))
    s._runtime.create_task("play youtube", task_id="fast-yt")
    s._runtime.transition_task("fast-yt", "RUNNING")
    monkeypatch.setattr(main, "cmd_tasks", lambda _args: 0)
    main._chat_command(s, "/tasks")
    out = capsys.readouterr().out
    assert "RUNTIME TASKS" in out
    assert "fast-yt" in out
    assert "play youtube" in out


def test_20_what_are_you_doing_uses_runtime_truth(monkeypatch):
    s = Session(narrator=Narrator.build(enabled=False))
    s._runtime.create_task("send WhatsApp message to Papa", task_id="fast-wa")
    s._runtime.transition_task("fast-wa", "RUNNING")
    monkeypatch.setattr("agent_control.session.api.resolve_task", lambda _text: None)
    monkeypatch.setattr(s, "_conversation_engine", lambda: (_ for _ in ()).throw(AssertionError("LLM should not answer runtime truth")))
    turn = s.submit("what are you doing?")
    assert "fast-wa" in turn.reply
    assert "RUNNING" in turn.reply
    assert "send WhatsApp message to Papa" in turn.reply


def test_21_human_escalation_stays_with_original_task():
    rt = manager()
    rt.create_task("wait for operator", task_id="fast-human")
    rt.transition_task("fast-human", "WAITING_FOR_HUMAN", recovery_context={"escalation_id": "esc-1", "channel": "operator", "reason": "needs human"})
    task = rt.get_task("fast-human")
    assert task.state == "WAITING_FOR_HUMAN"
    assert task.recovery_context["escalation_id"] == "esc-1"


def test_background_submission_keeps_one_authoritative_task_id(monkeypatch):
    s = Session(narrator=Narrator.build(enabled=False))
    monkeypatch.setattr("agent_control.session.api.resolve_task", lambda _text: None)
    monkeypatch.setattr("agent_control.session.api.parse_request", lambda _text: None)
    monkeypatch.setattr(
        "agent_control.session.api.run_agent_task",
        lambda request, **kwargs: AgentResult(request=request, task_id=kwargs["task_id"], status=TaskStatus.SUCCESS, verified="PASS"),
    )
    background_id = s.submit_background("open youtube")
    deadline = time.time() + 1
    while time.time() < deadline:
        tasks = s._runtime.list_tasks()
        if tasks and tasks[0].state == "COMPLETED":
            break
        time.sleep(0.01)
    runtime_tasks = s._runtime.list_tasks()
    assert len(runtime_tasks) == 1
    assert runtime_tasks[0].task_id == background_id
    assert runtime_tasks[0].state == "COMPLETED"
    assert [t.task_id for t in runtime_tasks] == [background_id]
    s.close()


def test_23_phase1_focused_tests_are_kept_separate():
    # This test documents the boundary: P2.0 owns runtime truth and does not
    # replace the Phase 1 task/executor implementation.
    from agent_control.fast_interaction import FastInteractionTask
    from agent_control.skills.messaging.task import ApprovedMessagingTask
    assert FastInteractionTask is not None
    assert ApprovedMessagingTask is not None
