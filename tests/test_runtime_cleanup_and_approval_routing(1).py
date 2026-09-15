from __future__ import annotations

from pathlib import Path
import time
from dataclasses import replace

import pytest

import main
import agent_control.session as session_module
from agent_control.runtime import RuntimeManager
from agent_control.session import Session, Turn, UserTask
from agent_control.response import Narrator
from agent_control.runtime_control import classify_runtime_control, classify_runtime_controls


def _session(tmp_path: Path) -> Session:
    return Session(
        narrator=Narrator.build(enabled=False),
        runtime_persistence_path=tmp_path / "runtime.sqlite3",
    )


def _pending(session: Session, task_id: str, *, owner: str | None = None) -> None:
    metadata = {"session_owner_id": owner or session._session_owner_id}
    session._runtime.create_task("send test", task_id=task_id, metadata=metadata)
    session._runtime.request_approval(task_id, {
        "request": "Approve test action?",
        "action": {"kind": "whatsapp_send_message", "params": {"recipient": "x", "message": "y"}},
        "approval_request_id": f"approval-{task_id}",
    })


def test_single_pending_approval_routes_yes_without_task_id(tmp_path, monkeypatch):
    session = _session(tmp_path)
    _pending(session, "task-one")

    monkeypatch.setattr(session, "_run", lambda prepared: Turn(
        task=prepared.task, reply="continued", result=None,
    ))
    turn = session.submit("yes")

    assert "task ID" not in (turn.reply or "")
    assert turn.task.task_id == "task-one"
    assert session._runtime.get_approval("task-one") is None
    assert session._runtime.get_task("task-one").state == "RUNNING"


def test_multiple_pending_approvals_require_identification(tmp_path):
    session = _session(tmp_path)
    _pending(session, "task-one")
    _pending(session, "task-two")

    turn = session.submit("yes")

    assert "Which one do you approve?" in (turn.reply or "")
    assert "task ID" not in (turn.reply or "")
    assert session._runtime.get_approval("task-one") is not None
    assert session._runtime.get_approval("task-two") is not None


def test_no_pending_approval_does_not_treat_yes_as_approval(tmp_path, monkeypatch):
    session = _session(tmp_path)
    called = []
    monkeypatch.setattr(session_module.api, "resolve_task", lambda text: None)
    monkeypatch.setattr(session_module.api, "parse_request", lambda text: None)
    monkeypatch.setattr(session, "_run", lambda prepared: called.append(prepared) or Turn(
        task=prepared.task, reply="conversation", result=None,
    ))

    turn = session.submit("yes")

    assert "approval" not in (turn.reply or "").casefold()
    assert not called
    assert session._runtime.list_tasks() == []


def test_single_owned_approval_wins_over_unowned_approval(tmp_path):
    session = _session(tmp_path)
    _pending(session, "task-owned")
    _pending(session, "task-other", owner="different-session")

    owner = session._approval_owner_for_input("yes")
    assert owner is not None
    assert owner.task_id == "task-owned"


def test_clear_history_removes_terminal_states_only(tmp_path):
    runtime = RuntimeManager(persistence_path=tmp_path / "runtime.sqlite3")
    completed = runtime.create_task("done", task_id="completed")
    failed = runtime.create_task("failed", task_id="failed")
    cancelled = runtime.create_task("cancelled", task_id="cancelled")
    runtime.transition_task(completed.task_id, "RUNNING")
    runtime.transition_task(completed.task_id, "COMPLETED")
    runtime.transition_task(failed.task_id, "FAILED")
    runtime.transition_task(cancelled.task_id, "CANCELLED")

    removed = runtime.clear_history()

    assert set(removed) == {"completed", "failed", "cancelled"}
    assert runtime.list_tasks() == []


def test_clear_history_preserves_all_non_terminal_states_and_approvals(tmp_path):
    runtime = RuntimeManager(persistence_path=tmp_path / "runtime.sqlite3")
    running = runtime.create_task("running", task_id="running")
    runtime.transition_task(running.task_id, "RUNNING")
    waiting_approval = runtime.create_task("approval", task_id="approval")
    runtime.request_approval(waiting_approval.task_id, {"request": "approve"})
    waiting_human = runtime.create_task("human", task_id="human")
    runtime.transition_task(waiting_human.task_id, "WAITING_FOR_HUMAN")
    recovery = runtime.create_task("recovery", task_id="recovery")
    runtime.transition_task(recovery.task_id, "RUNNING")
    runtime.transition_task(recovery.task_id, "RECOVERY_REQUIRED")

    removed = runtime.clear_history()

    assert removed == []
    for task_id in ("running", "approval", "human", "recovery"):
        assert runtime.get_task(task_id) is not None
    assert runtime.get_approval("approval") is not None


def test_clear_history_does_not_touch_checkpoint_store(tmp_path):
    runtime = RuntimeManager(persistence_path=tmp_path / "runtime.sqlite3")
    task = runtime.create_task("done", task_id="done")
    runtime.transition_task(task.task_id, "FAILED")
    checkpoint = tmp_path / "workflow.sqlite3"
    checkpoint.write_bytes(b"checkpoint-marker")

    runtime.clear_history()

    assert checkpoint.read_bytes() == b"checkpoint-marker"


def test_clear_history_with_no_inactive_history_is_empty(tmp_path):
    runtime = RuntimeManager(persistence_path=tmp_path / "runtime.sqlite3")
    task = runtime.create_task("active", task_id="active")
    runtime.transition_task(task.task_id, "RUNNING")

    assert runtime.clear_history() == []
    assert runtime.get_task("active") is not None


def test_help_lists_clear_history(capsys):
    class Dummy:
        speaking = False
        show_status = True
        planner = "mock"

    main._chat_help(Dummy())
    assert "/clear-history" in capsys.readouterr().out


def test_approval_resumes_same_workflow_and_current_step(tmp_path, monkeypatch):
    session = _session(tmp_path)
    workflow_id = "fast-workflow"
    workflow = {
        "workflow_id": workflow_id,
        "goal": "step one then step two",
        "current_step": 1,
        "steps": [
            {"step_id": "fast-workflow:step-1", "index": 0, "state": "COMPLETED", "action": {"kind": "noop", "params": {}}},
            {"step_id": "fast-workflow:step-2", "index": 1, "state": "PENDING", "action": {"kind": "noop", "params": {}}},
        ],
        "status": "WAITING_FOR_APPROVAL",
    }
    session._runtime.create_task(
        "step one then step two", task_id=workflow_id,
        task_type="workflow", metadata={"session_owner_id": session._session_owner_id, "workflow": workflow},
    )
    session._runtime.request_approval(workflow_id, {
        "request": "Approve step two?", "workflow_id": workflow_id,
        "step_id": "fast-workflow:step-2", "action": {"kind": "noop", "params": {}},
        "approval_request_id": "approval-fast-workflow-step-2",
    })
    resumed = []
    monkeypatch.setattr(session, "_resume_workflow_graph", lambda wid, value, source: resumed.append((wid, value, source)) or Turn(
        task=session.history[-1].task if session.history else __import__("agent_control.session", fromlist=["UserTask"]).UserTask(raw="", text="", task_id=wid),
        reply="resumed", result=None,
    ))

    session.submit("yes")

    assert resumed == [(workflow_id, True, "text")]
    stored = session._runtime.get_task(workflow_id)
    assert stored is not None
    assert stored.metadata["workflow"]["current_step"] == 1
    assert stored.metadata["workflow"]["steps"][0]["state"] == "COMPLETED"
    assert stored.metadata["workflow"]["steps"][1]["state"] == "PENDING"


def test_clear_history_removes_failed_history(tmp_path):
    runtime = RuntimeManager(persistence_path=tmp_path / "runtime.sqlite3")
    task = runtime.create_task("failed", task_id="failed")
    runtime.transition_task(task.task_id, "FAILED")
    assert runtime.clear_history() == ["failed"]


def test_clear_history_removes_cancelled_history(tmp_path):
    runtime = RuntimeManager(persistence_path=tmp_path / "runtime.sqlite3")
    task = runtime.create_task("cancelled", task_id="cancelled")
    runtime.transition_task(task.task_id, "CANCELLED")
    assert runtime.clear_history() == ["cancelled"]


def test_clear_history_preserves_running(tmp_path):
    runtime = RuntimeManager(persistence_path=tmp_path / "runtime.sqlite3")
    task = runtime.create_task("running", task_id="running")
    runtime.transition_task(task.task_id, "RUNNING")
    runtime.clear_history()
    assert runtime.get_task("running") is not None


def test_clear_history_preserves_waiting_for_approval(tmp_path):
    runtime = RuntimeManager(persistence_path=tmp_path / "runtime.sqlite3")
    task = runtime.create_task("approval", task_id="approval")
    runtime.request_approval(task.task_id, {"request": "approve"})
    runtime.clear_history()
    assert runtime.get_task("approval").state == "WAITING_FOR_APPROVAL"


def test_clear_history_preserves_waiting_for_human(tmp_path):
    runtime = RuntimeManager(persistence_path=tmp_path / "runtime.sqlite3")
    task = runtime.create_task("human", task_id="human")
    runtime.transition_task(task.task_id, "WAITING_FOR_HUMAN")
    runtime.clear_history()
    assert runtime.get_task("human").state == "WAITING_FOR_HUMAN"


def test_clear_history_preserves_recovery_required(tmp_path):
    runtime = RuntimeManager(persistence_path=tmp_path / "runtime.sqlite3")
    task = runtime.create_task("recovery", task_id="recovery")
    runtime.transition_task(task.task_id, "RUNNING")
    runtime.transition_task(task.task_id, "RECOVERY_REQUIRED")
    runtime.clear_history()
    assert runtime.get_task("recovery").state == "RECOVERY_REQUIRED"


def test_clear_history_is_runtime_control_not_a_task(tmp_path):
    session = _session(tmp_path)
    assert session._runtime.list_tasks() == []
    turn = session.submit("/clear-history")
    assert "No inactive workflow history" in (turn.reply or "")
    assert session._runtime.list_tasks() == []


def test_clear_history_preserves_active_checkpoint_marker_and_pending_approval(tmp_path):
    runtime = RuntimeManager(persistence_path=tmp_path / "runtime.sqlite3")
    task = runtime.create_task("approval", task_id="approval")
    runtime.request_approval(task.task_id, {"request": "approve"})
    checkpoint = tmp_path / "workflow.sqlite3"
    checkpoint.write_bytes(b"active-checkpoint")
    runtime.clear_history()
    assert checkpoint.read_bytes() == b"active-checkpoint"
    assert runtime.get_approval("approval") is not None


def test_approval_selection_multiple_is_human_facing(tmp_path):
    session = _session(tmp_path)
    _pending(session, "task-mummy")
    session._runtime.update_task("task-mummy", metadata={"session_owner_id": session._session_owner_id})
    _pending(session, "task-papa")
    session._runtime.update_task("task-papa", metadata={"session_owner_id": session._session_owner_id})

    turn = session.submit("yes")

    assert "multiple actions" in turn.reply.casefold()
    assert "task-" not in turn.reply
    assert "Which one do you approve?" in turn.reply


def test_approval_selection_by_number_resolves_selected_owner(tmp_path, monkeypatch):
    session = _session(tmp_path)
    _pending(session, "task-mummy")
    session._runtime.update_task("task-mummy", metadata={"session_owner_id": session._session_owner_id})
    _pending(session, "task-papa")
    session._runtime.update_task("task-papa", metadata={"session_owner_id": session._session_owner_id})
    monkeypatch.setattr(session, "_run", lambda prepared: Turn(
        task=prepared.task, reply="approved", result=None,
    ))

    prompt = session.submit("yes")
    assert "Which one do you approve?" in prompt.reply
    resolved = session.submit("1")
    assert resolved.reply == "approved"
    assert session._runtime.get_approval("task-mummy") is None
    assert session._runtime.get_approval("task-papa") is not None


def test_terminal_retention_is_time_based_and_configurable(tmp_path):
    runtime = RuntimeManager(persistence_path=tmp_path / "runtime.sqlite3", terminal_retention_s=0, cleanup_interval_s=0)
    task = runtime.create_task("done", task_id="done")
    runtime.transition_task(task.task_id, "RUNNING")
    runtime.transition_task(task.task_id, "COMPLETED")
    assert runtime.cleanup()["removed"] == ["done"]
    assert runtime.get_task("done") is None


def test_expired_terminal_state_is_retained_then_cleaned(tmp_path):
    runtime = RuntimeManager(persistence_path=tmp_path / "runtime.sqlite3", terminal_retention_s=3600, waiting_timeout_s=50, cleanup_interval_s=0)
    task = runtime.create_task("waiting", task_id="waiting")
    runtime.transition_task(task.task_id, "WAITING_FOR_HUMAN")
    runtime._tasks[task.task_id] = replace(task, state="WAITING_FOR_HUMAN", updated_at=time.time() - 100)
    runtime.persistence.commit(runtime._tasks[task.task_id])
    result = runtime.cleanup()
    assert result["expired"] == ["waiting"]
    assert runtime.get_task("waiting").state == "EXPIRED"


def test_expired_approval_is_removed_and_cannot_be_resolved(tmp_path):
    runtime = RuntimeManager(persistence_path=tmp_path / "runtime.sqlite3", waiting_timeout_s=0, cleanup_interval_s=0)
    task = runtime.create_task("approve", task_id="approve")
    runtime.request_approval(task.task_id, {"request": "approve", "approval_request_id": "a1"})
    runtime.cleanup()
    assert runtime.get_task("approve").state == "EXPIRED"
    assert runtime.get_approval("approve") is None
    with pytest.raises(ValueError):
        runtime.resolve_approval("approve", True, approval_request_id="a1")


def test_waiting_recent_activity_prevents_expiration(tmp_path):
    runtime = RuntimeManager(persistence_path=tmp_path / "runtime.sqlite3", waiting_timeout_s=100, cleanup_interval_s=0)
    task = runtime.create_task("waiting", task_id="waiting")
    runtime.transition_task(task.task_id, "WAITING_FOR_HUMAN")
    runtime.event(task.task_id, "USER_ACTIVITY")
    result = runtime.cleanup()
    assert result["expired"] == []
    assert runtime.get_task("waiting").state == "WAITING_FOR_HUMAN"


def test_periodic_cleanup_removes_old_terminal_history(tmp_path):
    runtime = RuntimeManager(persistence_path=tmp_path / "runtime.sqlite3", terminal_retention_s=0, cleanup_interval_s=0.01)
    task = runtime.create_task("done", task_id="done")
    runtime.transition_task(task.task_id, "FAILED", failure_state="failed")
    deadline = time.time() + 1
    while time.time() < deadline and runtime.get_task("done") is not None:
        time.sleep(0.01)
    assert runtime.get_task("done") is None
    runtime.close()


def test_startup_cleanup_removes_old_terminal_history(tmp_path):
    path = tmp_path / "runtime.sqlite3"
    runtime = RuntimeManager(persistence_path=path, terminal_retention_s=3600, cleanup_interval_s=0)
    task = runtime.create_task("done", task_id="done")
    runtime.transition_task(task.task_id, "FAILED")
    old = replace(runtime.get_task("done"), updated_at=time.time() - 7200)
    runtime.persistence.commit(old)
    runtime.close()

    restarted = RuntimeManager(persistence_path=path, terminal_retention_s=3600, cleanup_interval_s=0)
    assert restarted.get_task("done") is None
    restarted.close()


def test_clear_history_removes_expired_state(tmp_path):
    runtime = RuntimeManager(persistence_path=tmp_path / "runtime.sqlite3", cleanup_interval_s=0)
    task = runtime.create_task("expired", task_id="expired")
    runtime.transition_task(task.task_id, "WAITING_FOR_HUMAN")
    runtime.transition_task(task.task_id, "EXPIRED")
    assert runtime.clear_history() == ["expired"]


def test_clear_history_preserves_active_recovering(tmp_path):
    runtime = RuntimeManager(persistence_path=tmp_path / "runtime.sqlite3", cleanup_interval_s=0)
    task = runtime.create_task("recover", task_id="recover")
    runtime.transition_task(task.task_id, "RUNNING")
    runtime.transition_task(task.task_id, "RECOVERY_REQUIRED")
    runtime.transition_task(task.task_id, "RECOVERING")
    assert runtime.clear_history() == []
    assert runtime.get_task("recover").state == "RECOVERING"


def test_tasks_view_has_mutually_exclusive_human_sections(tmp_path, capsys):
    import main
    session = _session(tmp_path)
    done = session._runtime.create_task("Completed work", task_id="done")
    session._runtime.transition_task(done.task_id, "RUNNING")
    session._runtime.transition_task(done.task_id, "COMPLETED")
    waiting = session._runtime.create_task("Waiting work", task_id="waiting")
    session._runtime.transition_task(waiting.task_id, "WAITING_FOR_HUMAN")
    main._chat_command(session, "/tasks")
    output = capsys.readouterr().out
    assert "Completed work" in output
    assert "Waiting work" in output
    assert "[done]" not in output and "[waiting]" not in output
    assert output.count("Completed work") == 1
    assert output.count("Waiting work") == 1


def test_resume_multiple_tasks_supports_first_selection(tmp_path, monkeypatch):
    session = _session(tmp_path)
    import agent_control.session as sm
    workflows = []
    for task_id, goal in (("one", "Send hello to papa"), ("two", "Play Do I Wanna Know")):
        task = session._runtime.create_task(goal, task_id=task_id)
        session._runtime.update_task(task_id, metadata={"workflow": {"workflow_id": task_id, "goal": goal, "current_step": 0, "steps": [{"step_id": f"{task_id}:step-1", "index": 0, "state": "PENDING", "action": {"kind": "noop", "params": {}}}]}})
        workflows.append(task_id)
    monkeypatch.setattr(session, "_resume_runtime_task", lambda task, source, background: Turn(
        task=sm.UserTask(raw=task.goal, text=task.goal, source=source, task_id=task.task_id, status="accepted"), reply=f"resumed {task.goal}", result=None,
    ))
    prompt = session.submit("resume")
    assert "Send hello to papa" in prompt.reply and "Play Do I Wanna Know" in prompt.reply
    resolved = session.submit("the first one")
    assert resolved.reply == "resumed Send hello to papa"


def test_expired_task_is_not_resumable(tmp_path):
    session = _session(tmp_path)
    task = session._runtime.create_task("Expired workflow", task_id="expired")
    session._runtime.transition_task(task.task_id, "WAITING_FOR_HUMAN")
    session._runtime.transition_task(task.task_id, "EXPIRED")
    assert session._resumable_runtime_tasks() == []


def test_planner_execution_context_contains_no_runtime_history(tmp_path, monkeypatch):
    session = _session(tmp_path)
    session.recent_context.last_goal = "current goal"
    old = session._runtime.create_task("Old failed task", task_id="old")
    session._runtime.transition_task(old.task_id, "FAILED")
    captured = {}
    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return AgentResult(request="x", task_id="x", status=TaskStatus.SUCCESS)
    monkeypatch.setattr(session_module.api, "run_agent_task", fake_run)
    task = __import__("agent_control.session", fromlist=["UserTask"]).UserTask(raw="x", text="x", source="text", task_id="open_named_file", status="accepted")
    from agent_control.session import Prepared
    session._run(Prepared(task=task, goal="x", runtime_task_id="current"))
    assert "runtime" not in captured.get("recent_context", {})
    assert "Old failed task" not in str(captured.get("recent_context", {}))


def test_single_resumable_task_needs_no_identifier(tmp_path, monkeypatch):
    session = _session(tmp_path)
    task = session._runtime.create_task("Send hello to papa", task_id="resume-one")
    session._runtime.update_task(task.task_id, metadata={
        "workflow": {
            "workflow_id": task.task_id,
            "goal": task.goal,
            "current_step": 0,
            "steps": [{"step_id": "resume-one:step-1", "index": 0, "state": "PENDING", "action": {"kind": "noop", "params": {}}}],
        }
    })
    seen = []
    monkeypatch.setattr(session, "_resume_runtime_task", lambda task, source, background: seen.append(task.task_id) or Turn(
        task=__import__("agent_control.session", fromlist=["UserTask"]).UserTask(raw=task.goal, text=task.goal, source=source, task_id=task.task_id, status="accepted"),
        reply="resumed", result=None,
    ))
    turn = session.submit("resume")
    assert turn.reply == "resumed"
    assert seen == ["resume-one"]


@pytest.mark.parametrize("state", ["COMPLETED", "FAILED", "CANCELLED", "EXPIRED"])
def test_each_terminal_state_is_auto_retained_then_removed(tmp_path, state):
    runtime = RuntimeManager(persistence_path=tmp_path / f"{state}.sqlite3", terminal_retention_s=0, cleanup_interval_s=0)
    task = runtime.create_task(state.lower(), task_id=state.lower())
    if state == "EXPIRED":
        runtime.transition_task(task.task_id, "WAITING_FOR_HUMAN")
    else:
        runtime.transition_task(task.task_id, state if state != "COMPLETED" else "RUNNING")
        if state == "COMPLETED":
            runtime.transition_task(task.task_id, "COMPLETED")
    if state == "EXPIRED":
        runtime.transition_task(task.task_id, "EXPIRED")
    assert task.task_id in runtime._tasks
    runtime.cleanup()
    assert runtime.get_task(task.task_id) is None


def test_multiple_approval_accepts_natural_target_selection(tmp_path, monkeypatch):
    session = _session(tmp_path)
    for task_id, recipient in (("task-mummy", "mummy"), ("task-papa", "papa")):
        session._runtime.create_task("send test", task_id=task_id, metadata={"session_owner_id": session._session_owner_id})
        session._runtime.request_approval(task_id, {"request": f"send to {recipient}", "action": {"kind": "whatsapp_send_message", "params": {"recipient": recipient, "message": "hello"}}, "approval_request_id": f"approval-{task_id}"})
    seen = []
    monkeypatch.setattr(session, "_run", lambda prepared: seen.append(prepared.task.task_id) or Turn(
        task=prepared.task, reply="approved", result=None,
    ))
    prompt = session.submit("yes")
    assert "mummy" in prompt.reply and "papa" in prompt.reply
    result = session.submit("approve the mummy message")
    assert result.reply == "approved"
    assert seen == ["task-mummy"]


def test_clear_history_command_preserves_pending_approval(tmp_path):
    session = _session(tmp_path)
    done = session._runtime.create_task("Done", task_id="done")
    session._runtime.transition_task(done.task_id, "FAILED")
    waiting = session._runtime.create_task("Waiting for approval", task_id="waiting", metadata={"session_owner_id": session._session_owner_id})
    session._runtime.request_approval(waiting.task_id, {"request": "approve", "action": {"kind": "noop", "params": {}}})
    turn = session.submit("/clear-history")
    assert "1 inactive" in turn.reply
    assert session._runtime.get_task("done") is None
    assert session._runtime.get_task("waiting") is not None
    assert session._runtime.get_approval("waiting") is not None


def _numbered_task(session, number, *, state="RUNNING", goal=None, workflow=None):
    task_id = f"runtime-{number}"
    metadata = {"session_owner_id": session._session_owner_id}
    if workflow is not None:
        metadata["workflow"] = workflow
    session._runtime.create_task(goal or f"task goal {number}", task_id=task_id, metadata=metadata)
    if state == "FAILED":
        session._runtime.transition_task(task_id, "FAILED")
    elif state != "CREATED":
        session._runtime.transition_task(task_id, state)
    return task_id


def test_task_number_resolves_current_tasks_view_for_approval(tmp_path, monkeypatch):
    session = _session(tmp_path)
    for number in range(1, 8):
        if number == 7:
            _numbered_task(session, number, state="WAITING_FOR_APPROVAL", goal="Send the mummy message")
            session._runtime.request_approval("runtime-7", {
                "request": "Approve mummy",
                "action": {"kind": "whatsapp_send_message", "params": {"recipient": "mummy", "message": "hello"}},
                "workflow_id": "runtime-7",
                "step_id": "runtime-7:step-1",
                "approval_request_id": "approval-runtime-7",
            })
        else:
            _numbered_task(session, number, goal=f"other task {number}")
    monkeypatch.setattr(session, "_resume_workflow_graph", lambda workflow_id, value, source="text": Turn(task=session.history[-1].task if session.history else __import__("agent_control.session", fromlist=["UserTask"]).UserTask(raw="", text="", task_id=workflow_id), reply="approved", result=None))

    turn = session.submit("approve task 7")

    assert turn.reply == "approved"
    assert session._runtime.get_approval("runtime-7") is not None


def test_retry_task_number_is_runtime_control_not_conversation(tmp_path):
    session = _session(tmp_path)
    for number in range(1, 9):
        _numbered_task(session, number, state="FAILED" if number == 8 else "RUNNING")

    turn = session.submit("try task 8 again")

    assert "failed" in turn.reply.casefold()
    assert session._runtime.get_task("runtime-8").state == "FAILED"


def test_multi_operation_runtime_sentence_never_reaches_conversation(tmp_path, monkeypatch):
    session = _session(tmp_path)
    for number in range(1, 9):
        if number == 7:
            _numbered_task(session, number, state="WAITING_FOR_APPROVAL", goal="Approve mummy")
            session._runtime.request_approval("runtime-7", {
                "request": "Approve mummy",
                "action": {"kind": "whatsapp_send_message", "params": {"recipient": "mummy", "message": "hello"}},
                "workflow_id": "runtime-7",
                "step_id": "runtime-7:step-1",
                "approval_request_id": "approval-runtime-7",
            })
        elif number == 8:
            _numbered_task(session, number, state="FAILED", goal="Recover papa")
        else:
            _numbered_task(session, number, goal=f"other task {number}")
    monkeypatch.setattr(session, "_resume_workflow_graph", lambda workflow_id, value, source="text": Turn(task=__import__("agent_control.session", fromlist=["UserTask"]).UserTask(raw="", text="", task_id=workflow_id), reply="approved task 7", result=None))
    called = []
    monkeypatch.setattr(session, "_conversation_engine", lambda: called.append(True))

    turn = session.submit("for task 7 yes and for task 8 try again")

    assert called == []
    assert "approved task 7" in turn.reply
    assert "failed" in turn.reply.casefold()


def test_unresolvable_task_reference_stays_runtime_control_and_does_not_execute(tmp_path, monkeypatch):
    session = _session(tmp_path)
    _numbered_task(session, 1, goal="only task")
    called = []
    monkeypatch.setattr(session, "_conversation_engine", lambda: called.append(True))

    turn = session.submit("retry task 99")

    assert "current task list" in turn.reply.casefold()
    assert called == []
    assert len(session._runtime.list_tasks()) == 1


def test_recovery_retry_does_not_create_duplicate_workflow(tmp_path, monkeypatch):
    session = _session(tmp_path)
    workflow = {
        "workflow_id": "workflow-existing",
        "goal": "compound approval then recovery",
        "current_step": 1,
        "steps": [
            {"step_id": "workflow-existing:step-1", "index": 0, "state": "COMPLETED", "action": {"kind": "noop", "params": {}}},
            {"step_id": "workflow-existing:step-2", "index": 1, "state": "RECOVERY_REQUIRED", "action": {"kind": "noop", "params": {}}},
        ],
        "status": "RECOVERING",
    }
    task_id = _numbered_task(session, 1, state="RUNNING", goal="compound approval then recovery", workflow=workflow)
    session._runtime.transition_task(task_id, "RECOVERY_REQUIRED")
    created_before = len(session._runtime.list_tasks())
    calls = []
    monkeypatch.setattr(session, "_start_workflow", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(session, "_resume_workflow", lambda *args, **kwargs: calls.append((args, kwargs)))

    turn = session.submit("retry task 1")

    assert "not failed" in turn.reply.casefold()
    assert calls == []
    assert len(session._runtime.list_tasks()) == created_before
    assert session._runtime.get_task(task_id).metadata["workflow"]["workflow_id"] == "workflow-existing"


def _view_rows(session):
    rows = []
    for section, items in session.runtime_tasks_for_display().items():
        for item in items:
            rows.append((section, item["goal"], item["state"]))
    return rows


def test_tasks_view_has_authoritative_numbered_order(tmp_path):
    session = _session(tmp_path)
    _numbered_task(session, 1, state="RUNNING", goal="row one")
    _numbered_task(session, 2, state="WAITING_FOR_APPROVAL", goal="row two")
    _numbered_task(session, 3, state="FAILED", goal="row three")
    rows = _view_rows(session)

    assert [row[1] for row in rows] == ["row one", "row two", "row three"]
    assert session._resolve_runtime_task_ref(1).goal == "row one"
    assert session._resolve_runtime_task_ref(2).goal == "row two"
    assert session._resolve_runtime_task_ref(3).goal == "row three"


def test_task_7_resolves_to_seventh_displayed_task(tmp_path):
    session = _session(tmp_path)
    for number in range(1, 8):
        _numbered_task(session, number, goal=f"display row {number}")
    assert session._resolve_runtime_task_ref(7).task_id == "runtime-7"
    assert session._resolve_runtime_task_ref(7).goal == "display row 7"


def test_task_8_resolves_to_eighth_displayed_task(tmp_path):
    session = _session(tmp_path)
    for number in range(1, 9):
        _numbered_task(session, number, goal=f"display row {number}")
    assert session._resolve_runtime_task_ref(8).task_id == "runtime-8"
    assert session._resolve_runtime_task_ref(8).goal == "display row 8"


def test_approve_task_7_resolves_approval_for_displayed_row(tmp_path, monkeypatch):
    session = _session(tmp_path)
    for number in range(1, 8):
        if number == 7:
            _numbered_task(session, number, state="WAITING_FOR_APPROVAL", goal="displayed mummy approval")
            session._runtime.request_approval("runtime-7", {
                "request": "Approve mummy",
                "action": {"kind": "whatsapp_send_message", "params": {"recipient": "mummy", "message": "hello"}},
                "workflow_id": "runtime-7",
                "step_id": "runtime-7:step-1",
                "approval_request_id": "approval-runtime-7",
            })
        else:
            _numbered_task(session, number, goal=f"display row {number}")
    monkeypatch.setattr(session, "_resolve_approval_input", lambda raw, source, background: Turn(
        task=UserTask(raw="", text="", source=source, task_id="runtime-7"), reply="approved row 7", result=None
    ))

    turn = session.submit("approve task 7")

    assert turn.reply == "approved row 7"


def test_retry_task_8_resolves_recovery_row_not_internal_number(tmp_path):
    session = _session(tmp_path)
    for number in range(1, 9):
        _numbered_task(session, number, state="FAILED" if number == 8 else "RUNNING", goal=f"display row {number}")

    command = classify_runtime_control("retry task 8")
    resolved = session._resolve_runtime_control_command(command)

    assert resolved is not None
    assert resolved.task_ref == 8
    assert resolved.task_id == "runtime-8"
    assert resolved.task_id != "fast-8"


def test_multi_operation_resolves_each_display_number_independently(tmp_path):
    session = _session(tmp_path)
    for number in range(1, 9):
        _numbered_task(session, number, state="FAILED" if number == 8 else "RUNNING", goal=f"display row {number}")

    commands = classify_runtime_controls("for task 7 yes and for task 8 try again")
    resolved = [session._resolve_runtime_control_command(command) for command in commands]

    assert [(c.task_ref, c.kind.value, c.task_id) for c in resolved] == [
        (7, "APPROVE", "runtime-7"),
        (8, "RETRY", "runtime-8"),
    ]


def test_invalid_display_number_does_not_execute(tmp_path, monkeypatch):
    session = _session(tmp_path)
    for number in range(1, 10):
        _numbered_task(session, number, goal=f"display row {number}")
    called = []
    monkeypatch.setattr(session, "_conversation_engine", lambda: called.append(True))

    turn = session.submit("approve task 99")

    assert "1–9" in turn.reply
    assert called == []
    assert len(session._runtime.list_tasks()) == 9


def test_numeric_display_reference_never_becomes_internal_id(tmp_path):
    session = _session(tmp_path)
    _numbered_task(session, 1, goal="first")
    _numbered_task(session, 2, goal="second")
    _numbered_task(session, 3, goal="third")

    command = classify_runtime_control("resume task 2")
    resolved = session._resolve_runtime_control_command(command)

    assert command.task_id is None
    assert command.task_ref == 2
    assert resolved.task_id == "runtime-2"
    assert resolved.task_id != "fast-2"


def test_display_reference_re_resolves_after_task_state_change(tmp_path):
    session = _session(tmp_path)
    first = _numbered_task(session, 1, state="RUNNING", goal="first")
    second = _numbered_task(session, 2, state="RUNNING", goal="second")
    third = _numbered_task(session, 3, state="RUNNING", goal="third")

    assert session._resolve_runtime_task_ref(2).task_id == second
    session._runtime.transition_task(first, "COMPLETED")

    # The current /tasks view changes after the state transition, so row 2 is
    # resolved against the regenerated view rather than the stale old row.
    assert session._resolve_runtime_task_ref(2).task_id == third


def test_normal_task_view_output_contains_no_internal_ids(tmp_path):
    session = _session(tmp_path)
    _numbered_task(session, 1, goal="visible task")
    groups = session.runtime_tasks_for_display()
    rendered = "\n".join(
        f"{section}: {item['goal']} {item['state']}"
        for section, items in groups.items() for item in items
    )
    assert "runtime-1" not in rendered
    assert "fast-1" not in rendered
    assert "checkpoint" not in rendered.casefold()


def test_existing_semantic_approval_reference_still_works(tmp_path, monkeypatch):
    session = _session(tmp_path)
    _numbered_task(session, 1, state="WAITING_FOR_APPROVAL", goal="Send hello to mummy")
    session._runtime.request_approval("runtime-1", {
        "request": "Approve mummy",
        "action": {"kind": "whatsapp_send_message", "params": {"recipient": "mummy", "message": "hello"}},
        "workflow_id": "runtime-1",
        "step_id": "runtime-1:step-1",
        "approval_request_id": "approval-runtime-1",
    })
    monkeypatch.setattr(session, "_resume_workflow_graph", lambda workflow_id, value, source="text": Turn(
        task=UserTask(raw="", text="", source=source, task_id=workflow_id), reply="approved mummy", result=None
    ))

    turn = session.submit("approve the mummy message")

    assert turn.reply == "approved mummy"


def test_single_approval_yes_behavior_is_unchanged(tmp_path, monkeypatch):
    session = _session(tmp_path)
    _numbered_task(session, 1, state="WAITING_FOR_APPROVAL", goal="Send hello to mummy")
    session._runtime.request_approval("runtime-1", {
        "request": "Approve mummy",
        "action": {"kind": "whatsapp_send_message", "params": {"recipient": "mummy", "message": "hello"}},
        "workflow_id": "runtime-1",
        "step_id": "runtime-1:step-1",
        "approval_request_id": "approval-runtime-1",
    })
    monkeypatch.setattr(session, "_resume_workflow_graph", lambda workflow_id, value, source="text": Turn(
        task=UserTask(raw="", text="", source=source, task_id=workflow_id), reply="approved", result=None
    ))

    turn = session.submit("yes")

    assert turn.reply == "approved"


def test_multiple_approval_yes_remains_ambiguous(tmp_path):
    session = _session(tmp_path)
    for number in (1, 2):
        _numbered_task(session, number, state="WAITING_FOR_APPROVAL", goal=f"approval {number}")
        session._runtime.request_approval(f"runtime-{number}", {
            "request": f"Approve {number}",
            "action": {"kind": "whatsapp_send_message", "params": {"recipient": f"person-{number}", "message": "hello"}},
            "workflow_id": f"runtime-{number}",
            "step_id": f"runtime-{number}:step-1",
            "approval_request_id": f"approval-runtime-{number}",
        })

    turn = session.submit("yes")

    assert "multiple" in turn.reply.casefold()


def test_critical_multi_runtime_control_uses_display_numbers_and_never_conversation(tmp_path, monkeypatch, capsys):
    session = _session(tmp_path)
    session.debug = True
    for number in range(1, 9):
        if number == 7:
            _numbered_task(session, number, state="WAITING_FOR_APPROVAL", goal="send hello to mummy")
            session._runtime.request_approval("runtime-7", {
                "request": "Approve mummy",
                "action": {"kind": "whatsapp_send_message", "params": {"recipient": "mummy", "message": "hello"}},
                "workflow_id": "runtime-7",
                "step_id": "runtime-7:step-1",
                "approval_request_id": "approval-runtime-7",
            })
        elif number == 8:
            _numbered_task(session, number, state="FAILED", goal="try the papa recovery again")
        else:
            _numbered_task(session, number, state="RUNNING", goal=f"display row {number}")

    monkeypatch.setattr(session, "_resume_workflow_graph", lambda workflow_id, value, source="text": Turn(
        task=UserTask(raw="", text="", source=source, task_id=workflow_id), reply="approved task 7", result=None
    ))
    conversation_calls = []
    monkeypatch.setattr(session, "_conversation_engine", lambda: conversation_calls.append(True))

    turn = session.submit("for task 7 yes and for task 8 try again")

    assert conversation_calls == []
    assert "approved task 7" in turn.reply
    assert "failed" in turn.reply.casefold()
    debug = capsys.readouterr().out
    assert "display_task=7 resolved_task=runtime-7" in debug
    assert "display_task=8 resolved_task=runtime-8" in debug
    assert "display_task=7 resolved_task=fast-7" not in debug
    assert "display_task=8 resolved_task=fast-8" not in debug


def test_pending_task_status_uses_same_numbered_view_as_tasks(tmp_path):
    session = _session(tmp_path)
    for number in range(1, 6):
        state = "WAITING_FOR_APPROVAL" if number == 1 else "RUNNING"
        _numbered_task(session, number, state=state, goal=f"pending row {number}")

    reply = session._runtime_query_reply("what are your pending tasks")

    assert reply.splitlines() == [
        "I currently have these pending tasks:",
        "1 - pending row 2 — running",
        "2 - pending row 3 — running",
        "3 - pending row 4 — running",
        "4 - pending row 5 — running",
        "5 - pending row 1 — waiting for your approval",
    ]
    assert session._resolve_runtime_task_ref(1).task_id == "runtime-2"
    assert session._resolve_runtime_task_ref(5).task_id == "runtime-1"


def test_approve_task_one_and_continue_checkpoint_task_five_are_two_operations(tmp_path):
    commands = classify_runtime_controls(
        "approve task 1 and continue from checkpoint the task 5"
    )

    assert [(command.kind.value, command.task_ref) for command in commands] == [
        ("APPROVE", 1),
        ("CONTINUE", 5),
    ]


def test_task_number_approval_phrase_can_resolve_waiting_user_input(tmp_path, monkeypatch):
    session = _session(tmp_path)
    workflow = {
        "workflow_id": "runtime-7",
        "goal": "send hello to papa",
        "state": "WAITING_FOR_USER",
        "current_step": 0,
        "steps": [{
            "step_id": "runtime-7:step-1",
            "index": 0,
            "state": "WAITING_FOR_USER",
            "action": {"kind": "whatsapp_send_message", "params": {"recipient": "papa", "message": ""}},
        }],
    }
    for number in range(1, 7):
        _numbered_task(session, number, state="RUNNING", goal=f"other task {number}")
    task_id = _numbered_task(session, 7, state="WAITING_FOR_USER", goal="send hello to papa", workflow=workflow)
    session._runtime.set_pending_input(task_id, {
        "input_id": "input-runtime-7",
        "workflow_id": "runtime-7",
        "step_id": "runtime-7:step-1",
        "task_id": task_id,
        "field": "message",
        "prompt": "What should I send?",
    })
    monkeypatch.setattr(session, "_resume_workflow", lambda workflow, source="text", background=False: Turn(
        task=UserTask(raw=workflow.goal, text=workflow.goal, source=source, task_id=workflow.workflow_id, status="accepted"),
        reply="continued task 7",
        result=None,
    ))

    turn = session.submit("u have approval yes for task 7")

    assert turn.reply == "continued task 7"
    assert session._runtime.get_task(task_id).state == "RUNNING"


def test_try_again_typo_does_not_fall_back_to_status(tmp_path):
    commands = classify_runtime_controls("for task 7 yes and for task 8 try againn")
    assert [(command.kind.value, command.task_ref) for command in commands] == [
        ("APPROVE", 7),
        ("RETRY", 8),
    ]

# P2.5 runtime-control routing regressions: pending state is passive unless the
# utterance itself is an explicit runtime-control/approval response.
def test_new_compound_task_never_hijacked_by_multiple_pending_approvals(tmp_path, monkeypatch, capsys):
    session = _session(tmp_path)
    session.debug = True
    for task_id in ("fast-6270", "fast-975f", "fast-ba14"):
        _pending(session, task_id)

    started = []
    monkeypatch.setattr(session_module.api, "resolve_task", lambda text: None)
    monkeypatch.setattr(session_module.api, "parse_request", lambda text: None)
    monkeypatch.setattr(session, "_start_workflow", lambda goal, source="text", workflow_id=None: started.append((goal, workflow_id)) or Turn(
        task=UserTask(raw=goal, text=goal, source=source, task_id=workflow_id or "new-workflow", status="accepted"),
        reply="new workflow started", result=None,
    ))

    turn = session.submit("open notepad and open chrome")

    assert turn.reply == "new workflow started"
    assert started and started[0][0] == "open notepad and open chrome"
    assert [session._runtime.get_approval(t) is not None for t in ("fast-6270", "fast-975f", "fast-ba14")] == [True, True, True]
    debug = capsys.readouterr().out
    assert "ROUTE: TASK" in debug
    assert "APPROVAL: resolution=ambiguous" not in debug


def test_pending_task_query_is_runtime_control_and_hides_internal_ids(tmp_path):
    session = _session(tmp_path)
    for task_id, goal in (("fast-6270", 'Send "bye" to papa'), ("fast-975f", 'Send "hello" to mummy'), ("fast-ba14", 'Send "hello" to mummy')):
        _pending(session, task_id)
        session._runtime.update_task(task_id, metadata={"session_owner_id": session._session_owner_id})

    turn = session.submit("which tasks are pending")

    assert turn.reply.startswith("I currently have these pending tasks:")
    assert "1 -" in turn.reply and "2 -" in turn.reply and "3 -" in turn.reply
    assert "fast-6270" not in turn.reply
    assert "fast-975f" not in turn.reply
    assert "fast-ba14" not in turn.reply


def test_multi_delete_resolves_current_display_numbers_and_preserves_other_task(tmp_path):
    session = _session(tmp_path)
    first = _pending(session, "fast-6270")
    second = _pending(session, "fast-975f")
    third = _pending(session, "fast-ba14")

    turn = session.submit("delete task 1 and task 2")

    assert "deleted" in turn.reply.casefold()
    assert session._runtime.get_task("fast-6270").state == "CANCELLED"
    assert session._runtime.get_task("fast-975f").state == "CANCELLED"
    assert session._runtime.get_task("fast-ba14").state == "WAITING_FOR_APPROVAL"
    assert session._runtime.get_approval("fast-6270") is None
    assert session._runtime.get_approval("fast-975f") is None
    assert session._runtime.get_approval("fast-ba14") is not None


def test_multi_delete_resolves_all_targets_before_state_changes(tmp_path):
    session = _session(tmp_path)
    for task_id in ("A", "B", "C"):
        _pending(session, task_id)

    resolved = []
    original = session._runtime.cancel_task
    def cancel(task_id, *, reason=""):
        resolved.append(task_id)
        return original(task_id, reason=reason)
    session._runtime.cancel_task = cancel

    session.submit("delete task 1 task 2")

    assert resolved == ["A", "B"]
    assert session._runtime.get_task("C").state == "WAITING_FOR_APPROVAL"


def test_resume_task_uses_same_workflow_checkpoint_identity(tmp_path, monkeypatch):
    session = _session(tmp_path)
    workflow_id = "workflow-existing"
    workflow = {
        "workflow_id": workflow_id,
        "goal": "step one then step two",
        "current_step": 1,
        "steps": [
            {"step_id": f"{workflow_id}:step-1", "index": 0, "state": "COMPLETED", "action": {"kind": "noop", "params": {}}},
            {"step_id": f"{workflow_id}:step-2", "index": 1, "state": "PENDING", "action": {"kind": "noop", "params": {}}},
        ],
        "status": "RUNNING",
    }
    session._runtime.create_task(workflow["goal"], task_id=workflow_id, task_type="workflow", metadata={"workflow": workflow})
    resumed = []
    monkeypatch.setattr(session, "_resume_workflow_graph", lambda wid, value, source="text": resumed.append((wid, value, source)) or Turn(
        task=UserTask(raw=workflow["goal"], text=workflow["goal"], source=source, task_id=wid, status="accepted"),
        reply="resumed", result=None,
    ))

    turn = session.submit("resume task 1")

    assert turn.reply == "resumed"
    assert resumed == [(workflow_id, None, "text")]
    assert len(session._runtime.list_tasks()) == 1


def test_resume_waiting_approval_establishes_targeted_approval_context(tmp_path, monkeypatch):
    session = _session(tmp_path)
    for task_id in ("A", "B", "C"):
        session._runtime.create_task(f"send {task_id}", task_id=task_id, task_type="workflow", metadata={
            "workflow": {"workflow_id": task_id, "goal": f"send {task_id}", "current_step": 0,
                         "steps": [{"step_id": f"{task_id}:step-1", "index": 0, "state": "PENDING",
                                    "action": {"kind": "whatsapp_send_message", "params": {"recipient": task_id, "message": "hello"}}}]}
        })
        session._runtime.request_approval(task_id, {
            "request": f"Approve {task_id}",
            "action": {"kind": "whatsapp_send_message", "params": {"recipient": task_id, "message": "hello"}},
            "workflow_id": task_id, "step_id": f"{task_id}:step-1", "approval_request_id": f"approval-{task_id}",
        })

    # Current display mapping is A=1, B=2, C=3 because all three are waiting.
    resumed = []
    def fake_resume(wid, value, source="text"):
        resumed.append((wid, value))
        approval = session._runtime.get_approval(wid)
        if approval is not None:
            session._runtime.resolve_approval(wid, True, approval_request_id=approval.get("approval_request_id"))
        return Turn(
            task=UserTask(raw=wid, text=wid, source=source, task_id=wid, status="accepted"), reply="approved", result=None,
        )
    monkeypatch.setattr(session, "_resume_workflow_graph", fake_resume)
    prompt = session.submit("resume task 2")
    assert "needs approval" in prompt.reply.casefold()

    session.submit("yes")

    assert resumed == [("B", True)]
    assert session._runtime.get_approval("A") is not None
    assert session._runtime.get_approval("B") is None
    assert session._runtime.get_approval("C") is not None


def test_multiple_approvals_bare_yes_is_ambiguous_without_target(tmp_path):
    session = _session(tmp_path)
    for task_id in ("A", "B", "C"):
        _pending(session, task_id)

    turn = session.submit("yes")

    assert "multiple" in turn.reply.casefold()
    assert all(session._runtime.get_approval(task_id) is not None for task_id in ("A", "B", "C"))


def test_targeted_yes_approves_only_selected_display_task(tmp_path, monkeypatch):
    session = _session(tmp_path)
    for task_id in ("A", "B", "C"):
        _pending(session, task_id)
    monkeypatch.setattr(session, "_run", lambda prepared: Turn(
        task=prepared.task, reply="approved", result=None,
    ))

    turn = session.submit("yes for task 2")

    assert turn.reply == "approved"
    assert session._runtime.get_approval("A") is not None
    assert session._runtime.get_approval("B") is None
    assert session._runtime.get_approval("C") is not None


def test_display_numbers_rebind_after_deletion(tmp_path):
    session = _session(tmp_path)
    _pending(session, "A")
    _pending(session, "B")
    _pending(session, "C")
    assert session._resolve_runtime_task_ref(1).task_id == "A"
    session._runtime.cancel_task("A", reason="test")
    assert session._resolve_runtime_task_ref(1).task_id == "B"
    assert session._resolve_runtime_task_ref(2).task_id == "C"


def test_invalid_resume_number_never_executes_or_approves(tmp_path, monkeypatch):
    session = _session(tmp_path)
    _pending(session, "A")
    called = []
    monkeypatch.setattr(session, "_resume_workflow_graph", lambda *args, **kwargs: called.append(args))

    turn = session.submit("resume task 999")

    assert "current task list" in turn.reply.casefold()
    assert called == []
    assert session._runtime.get_approval("A") is not None


def test_natural_language_new_task_stays_task_with_pending_approvals(tmp_path, monkeypatch):
    session = _session(tmp_path)
    _pending(session, "old-1")
    _pending(session, "old-2")
    started = []
    monkeypatch.setattr(session_module.api, "resolve_task", lambda text: None)
    monkeypatch.setattr(session_module.api, "parse_request", lambda text: None)
    monkeypatch.setattr(session, "_start_workflow", lambda goal, source="text", workflow_id=None: started.append(goal) or Turn(
        task=UserTask(raw=goal, text=goal, source=source, task_id=workflow_id or "new", status="accepted"),
        reply="started", result=None,
    ))

    turn = session.submit("open notepad, type hello, and play Do I Wanna Know")

    assert turn.reply == "started"
    assert started == ["open notepad, type hello, and play Do I Wanna Know"]
    assert session._runtime.get_approval("old-1") is not None
    assert session._runtime.get_approval("old-2") is not None


def test_hey_is_conversation_even_with_pending_approvals(tmp_path, monkeypatch):
    session = _session(tmp_path)
    _pending(session, "old-1")
    monkeypatch.setattr(session_module.api, "resolve_task", lambda text: None)
    monkeypatch.setattr(session_module.api, "parse_request", lambda text: None)
    monkeypatch.setattr(session, "_conversation_engine", lambda: type("E", (), {"reply": lambda self, *a, **k: "hi", "last_metrics": {}})())

    turn = session.submit("hey")

    assert turn.task.intent.value == "CONVERSATION"
    assert session._runtime.get_approval("old-1") is not None


def test_multi_delete_keeps_valid_targets_when_one_display_number_is_invalid(tmp_path):
    session = _session(tmp_path)
    for task_id in ("A", "B", "C"):
        _pending(session, task_id)

    turn = session.submit("delete task 1 and task 999")

    assert session._runtime.get_task("A").state == "CANCELLED"
    assert session._runtime.get_task("B").state == "WAITING_FOR_APPROVAL"
    assert session._runtime.get_task("C").state == "WAITING_FOR_APPROVAL"
    assert "Task 999" in turn.reply
    assert "deleted" in turn.reply.casefold()
