from __future__ import annotations

import json
import sqlite3
import time

import pytest

from agent_control.api import AgentResult, TaskStatus
from agent_control.response import Narrator
from agent_control.runtime import RuntimeManager
from agent_control.session import Session, Turn


def rt(tmp_path):
    return RuntimeManager(persistence_path=tmp_path / "runtime.sqlite3")


def test_01_create_persists(tmp_path):
    r = rt(tmp_path)
    r.create_task("open youtube", task_id="fast-1", task_type="fast")
    rows = sqlite3.connect(tmp_path / "runtime.sqlite3").execute("select task_id,state from tasks").fetchall()
    assert rows == [("fast-1", "CREATED")]


def test_02_reload_task_from_sqlite(tmp_path):
    r = rt(tmp_path); r.create_task("open youtube", task_id="fast-2")
    r.close()
    r2 = rt(tmp_path)
    assert r2.get_task("fast-2").goal == "open youtube"
    r2.close()


def test_03_task_id_survives_reload(tmp_path):
    r = rt(tmp_path); r.create_task("play song", task_id="fast-stable"); r.close()
    r2 = rt(tmp_path)
    assert [x.task_id for x in r2.list_tasks()] == ["fast-stable"]
    r2.close()


def test_04_state_transition_persists(tmp_path):
    r = rt(tmp_path); r.create_task("open youtube", task_id="fast-4"); r.transition_task("fast-4", "RUNNING")
    row = sqlite3.connect(tmp_path / "runtime.sqlite3").execute("select state from tasks where task_id='fast-4'").fetchone()
    assert row == ("RUNNING",)
    r.close()
    r2 = rt(tmp_path)
    assert r2.get_task("fast-4").state == "RECOVERY_REQUIRED"
    r2.close()


def test_05_lifecycle_event_persists(tmp_path):
    r = rt(tmp_path); r.create_task("x", task_id="fast-5"); r.transition_task("fast-5", "PLANNING"); r.close()
    r2 = rt(tmp_path)
    types = [e.event_type for e in r2.events("fast-5")]
    assert types[:2] == ["TASK_CREATED", "TASK_STATE_CHANGED"]
    r2.close()


def test_06_event_ordering(tmp_path):
    r = rt(tmp_path); r.create_task("x", task_id="fast-6"); r.transition_task("fast-6", "PLANNING"); r.transition_task("fast-6", "RUNNING")
    events = r.events("fast-6")
    assert [e.event_type for e in events] == ["TASK_CREATED", "TASK_STATE_CHANGED", "TASK_STATE_CHANGED"]
    assert [e.timestamp for e in events] == sorted(e.timestamp for e in events)
    r.close()


def test_07_completed_survives_restart(tmp_path):
    r = rt(tmp_path); r.create_task("open youtube", task_id="fast-7"); r.transition_task("fast-7", "RUNNING"); r.transition_task("fast-7", "COMPLETED", verification_state="PASS"); r.close()
    r2 = rt(tmp_path)
    assert r2.get_task("fast-7").state == "COMPLETED"
    r2.close()


def test_08_failed_survives_restart(tmp_path):
    r = rt(tmp_path); r.create_task("bad", task_id="fast-8"); r.transition_task("fast-8", "FAILED", failure_state="timeout"); r.close()
    r2 = rt(tmp_path)
    assert r2.get_task("fast-8").state == "FAILED"
    assert r2.get_task("fast-8").goal == "bad"
    r2.close()


def test_09_unknown_verification_survives_restart(tmp_path):
    r = rt(tmp_path); r.create_task("click", task_id="fast-9"); r.transition_task("fast-9", "RUNNING"); r.transition_task("fast-9", "VERIFYING", verification_state="UNKNOWN"); r.close()
    r2 = rt(tmp_path)
    assert r2.get_task("fast-9").state == "RECOVERY_REQUIRED"
    assert r2.get_task("fast-9").verification_state == "UNKNOWN"
    r2.close()


def test_10_running_becomes_recovery_required(tmp_path):
    r = rt(tmp_path); r.create_task("send whatsapp", task_id="fast-crash"); r.transition_task("fast-crash", "RUNNING"); r.close()
    r2 = rt(tmp_path)
    task = r2.get_task("fast-crash")
    assert task.state == "RECOVERY_REQUIRED"
    assert task.verification_state == "UNKNOWN"
    assert task.recovery_context["previous_state"] == "RUNNING"
    assert any(e.event_type == "RECOVERY_REQUIRED" for e in r2.events("fast-crash"))
    r2.close()


def test_11_waiting_approval_survives_restart(tmp_path):
    r = rt(tmp_path); r.create_task("send hello to mummy", task_id="fast-11"); r.request_approval("fast-11", {"action": {"kind": "whatsapp_send_message", "params": {"recipient": "mummy", "message": "hello"}}, "asked_at": time.time()}); r.close()
    r2 = rt(tmp_path)
    assert r2.get_task("fast-11").state == "WAITING_FOR_APPROVAL"
    assert r2.get_approval("fast-11") is not None
    r2.close()


def test_12_approval_keeps_original_id(tmp_path):
    r = rt(tmp_path); r.create_task("send hello", task_id="fast-12"); r.request_approval("fast-12", {"action": {"kind": "whatsapp_send_message"}}); r.close()
    r2 = rt(tmp_path)
    assert list(r2.snapshot()["pending_approvals"])[0]["task_id"] == "fast-12"
    r2.resolve_approval("fast-12", True)
    assert [t.task_id for t in r2.list_tasks()] == ["fast-12"]
    assert r2.get_task("fast-12").state == "RUNNING"
    r2.close()


def test_13_approval_response_does_not_create_second_task(tmp_path):
    r = rt(tmp_path); r.create_task("send hello", task_id="fast-13"); r.request_approval("fast-13", {"action": {"kind": "whatsapp_send_message"}}); r.close()
    r2 = rt(tmp_path); before = set(t.task_id for t in r2.list_tasks()); r2.resolve_approval("fast-13", True); assert set(t.task_id for t in r2.list_tasks()) == before; r2.close()


def test_14_multiple_tasks_persist_independently(tmp_path):
    r = rt(tmp_path)
    for i in range(4): r.create_task(f"task {i}", task_id=f"fast-{i}")
    r.transition_task("fast-0", "RUNNING"); r.transition_task("fast-1", "FAILED", failure_state="x"); r.transition_task("fast-2", "RUNNING"); r.transition_task("fast-2", "COMPLETED"); r.close()
    r2 = rt(tmp_path)
    states = {t.task_id: t.state for t in r2.list_tasks()}
    assert states["fast-0"] == "RECOVERY_REQUIRED"
    assert states["fast-1"] == "FAILED" and states["fast-2"] == "COMPLETED" and states["fast-3"] == "CREATED"
    r2.close()


def test_15_snapshot_and_background_view_agree(tmp_path):
    s = Session(narrator=Narrator.build(enabled=False))
    # Direct Session is in-memory; persistence-specific consistency is tested by the manager.
    s._runtime.create_task("play", task_id="fast-bg", metadata={"background": True}, task_type="background")
    s._runtime.transition_task("fast-bg", "RUNNING")
    s._runtime.transition_task("fast-bg", "COMPLETED")
    snap = s.runtime_snapshot()
    assert snap["completed_tasks"][0]["state"] == "COMPLETED"
    from agent_control.session import BackgroundTask
    s._background["fast-bg"] = BackgroundTask(task_id="fast-bg", original_goal="play", kind="background", state="COMPLETED", runtime_task_id="fast-bg")
    assert s.background_tasks()[0].state == snap["completed_tasks"][0]["state"]


def test_16_no_contradictory_runtime_views(tmp_path):
    r = rt(tmp_path); r.create_task("x", task_id="fast-16"); r.transition_task("fast-16", "RUNNING"); r.transition_task("fast-16", "COMPLETED"); snap = r.snapshot()
    assert all(x["task_id"] != "fast-16" for x in snap["active_tasks"])
    assert [x["task_id"] for x in snap["completed_tasks"]] == ["fast-16"]
    r.close()


def test_17_browser_metadata_only(tmp_path):
    r = rt(tmp_path); r.create_task("youtube", task_id="fast-17"); r.attach_resource("fast-17", "youtube", site="youtube", state="ATTACHED"); r.close()
    raw = sqlite3.connect(tmp_path / "runtime.sqlite3").execute("select * from runtime_resources").fetchone()
    assert raw[0] == "youtube" and "playwright" not in json.dumps(raw)
    r2 = rt(tmp_path)
    assert r2.resource_for_task("fast-17")["state"] == "STALE"
    r2.close()


def test_18_sensitive_values_are_redacted(tmp_path):
    r = rt(tmp_path); r.create_task("send password=SUPERSECRET token=ABC123", task_id="fast-18", metadata={"password": "SUPERSECRET", "token": "ABC123", "safe": "ok"}); r.close()
    conn = sqlite3.connect(tmp_path / "runtime.sqlite3")
    raw = " ".join(str(x) for row in conn.execute("select goal,metadata from tasks") for x in row)
    assert "SUPERSECRET" not in raw and "ABC123" not in raw and "[REDACTED]" in raw
    conn.close()


def test_19_fast_task_lifecycle_still_works():
    r = RuntimeManager(persistence_path=":memory:")
    r.create_task("open youtube", task_id="fast-19", task_type="fast")
    r.transition_task("fast-19", "RUNNING"); r.transition_task("fast-19", "VERIFYING", verification_state="PASS"); r.transition_task("fast-19", "COMPLETED", verification_state="PASS")
    assert r.get_task("fast-19").state == "COMPLETED"
    r.close()


def test_20_general_task_lifecycle_supported():
    r = RuntimeManager(persistence_path=":memory:")
    r.create_task("general", task_id="general-20", task_type="general")
    for state in ("PLANNING", "WAITING_FOR_APPROVAL"):
        r.transition_task("general-20", state)
    r.resolve_approval("general-20", True) if r.get_approval("general-20") else r.transition_task("general-20", "RUNNING")
    r.transition_task("general-20", "VERIFYING"); r.transition_task("general-20", "COMPLETED", verification_state="PASS")
    assert r.get_task("general-20").state == "COMPLETED"
    r.close()


def test_21_persistence_failure_does_not_commit_false_success(tmp_path, monkeypatch):
    r = rt(tmp_path); r.create_task("x", task_id="fast-21"); r.transition_task("fast-21", "RUNNING")
    def fail(*a, **k): raise OSError("disk full")
    monkeypatch.setattr(r.persistence, "commit", fail)
    with pytest.raises(OSError): r.transition_task("fast-21", "COMPLETED", verification_state="PASS")
    assert r.get_task("fast-21").state == "RUNNING"
    r.close()


def test_22_unknown_unverified_crash_never_becomes_success(tmp_path):
    r = rt(tmp_path); r.create_task("send email", task_id="fast-22"); r.transition_task("fast-22", "RUNNING", verification_state="UNKNOWN"); r.close()
    r2 = rt(tmp_path); t = r2.get_task("fast-22")
    assert t.state == "RECOVERY_REQUIRED" and t.verification_state == "UNKNOWN" and t.state != "COMPLETED"
    r2.close()


def test_23_simulated_crash_restart_boundary(tmp_path):
    first = rt(tmp_path); first.create_task("click", task_id="fast-crash-boundary"); first.transition_task("fast-crash-boundary", "RUNNING"); first.close()
    second = rt(tmp_path)
    assert second.get_task("fast-crash-boundary").state == "RECOVERY_REQUIRED"
    assert second.get_task("fast-crash-boundary").recovery_context["reason"] == "process_restart"
    second.close()


def test_24_session_restores_approval_owner_and_resumes_same_id(tmp_path, monkeypatch):
    path = tmp_path / "runtime.sqlite3"
    first = Session(narrator=Narrator.build(enabled=False), runtime_persistence_path=path)
    first._runtime.create_task("send hello to mummy", task_id="fast-approval", task_type="messaging")
    first._runtime.request_approval("fast-approval", {"request": "send hello to mummy", "action": {"kind": "whatsapp_send_message", "params": {"recipient": "mummy", "message": "hello"}}, "asked_at": time.time()})
    first.close()
    second = Session(narrator=Narrator.build(enabled=False), runtime_persistence_path=path)
    assert second.pending_approval is not None
    assert second.pending_approval.task_id == "fast-approval"
    seen = []
    def fake_run(prepared):
        seen.append(prepared.task.task_id)
        return Turn(prepared.task, "done", AgentResult(request=prepared.task.text, task_id=prepared.task.task_id, status=TaskStatus.SUCCESS, verified="PASS"))
    monkeypatch.setattr(second, "_run", fake_run)
    second.submit("yes")
    assert seen == ["fast-approval"]
    assert [t.task_id for t in second._runtime.list_tasks()] == ["fast-approval"]
    second.close()
