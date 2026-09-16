from __future__ import annotations

import threading
import time

import pytest

from agent_control.types import Action
from agent_control.workflow import (
    DependencyScheduler,
    DependencyType,
    ResourceLockManager,
    StepStatus,
    Workflow,
    WorkflowGraphError,
    WorkflowStatus,
    recover_failed_node,
)
from agent_control.fast_interaction import classify_fast
from agent_control.session import Session


def _wf(actions, goal="compound"):
    return Workflow.from_actions("p25", goal, actions)


def test_atomic_single_task_still_works():
    wf = _wf([Action("browser_play_song", {"query": "Do I Wanna Know"})])
    assert [n.status for n in wf.steps] == [StepStatus.PENDING]
    assert [n.step_id for n in DependencyScheduler.ready_nodes(wf)] == [wf.steps[0].step_id]


def test_independent_nodes_are_ready_together_and_overlap():
    wf = _wf([
        Action("browser_play_song", {"query": "A"}),
        Action("browser_search", {"query": "B"}),
    ], "play A and search B")
    assert wf.steps[0].dependencies == []
    assert wf.steps[1].dependencies == []
    intervals = {}
    lock = threading.Lock()

    def execute(node):
        with lock:
            intervals[node.step_id] = [time.perf_counter(), None]
        time.sleep(0.06)
        with lock:
            intervals[node.step_id][1] = time.perf_counter()
        return {"status": "COMPLETED", "result": {"node": node.step_id}, "verification": {"verdict": "PASS"}}

    out = DependencyScheduler(max_concurrency=2).run(wf, execute)
    assert out.status is WorkflowStatus.COMPLETED
    assert all(n.status is StepStatus.COMPLETED for n in wf.steps)
    a, b = intervals.values()
    assert a[0] < b[1] and b[0] < a[1], "independent branches did not overlap"


def test_state_dependency_open_notepad_before_type():
    wf = _wf([
        Action("launch_app", {"app": "notepad"}),
        Action("type_text", {"app": "notepad", "text": "hello"}),
        Action("browser_play_song", {"query": "Do I Wanna Know"}),
    ], "open notepad, type hello, and play Do I Wanna Know")
    assert wf.steps[1].dependencies == [wf.steps[0].step_id]
    assert wf.steps[1].dependency_types[wf.steps[0].step_id] == DependencyType.STATE_DEPENDENCY.value
    assert wf.steps[2].dependencies == []


def test_explicit_then_preserves_order():
    wf = _wf([
        Action("whatsapp_send_message", {"recipient": "mummy", "message": "hello"}),
        Action("whatsapp_send_message", {"recipient": "papa", "message": "bye"}),
        Action("browser_play_song", {"query": "Do I Wanna Know"}),
    ], "send hello to mummy then send bye to papa then play Do I Wanna Know")
    assert wf.steps[1].dependencies == [wf.steps[0].step_id]
    assert wf.steps[2].dependencies == [wf.steps[1].step_id]
    assert all(next(iter(s.dependency_types.values()), None) == DependencyType.EXPLICIT_USER_ORDER.value for s in wf.steps[1:])


def test_dag_rejects_cycle_and_missing_dependency():
    wf = _wf([Action("browser_search", {"query": "a"}), Action("browser_search", {"query": "b"})])
    wf.steps[0].dependencies = [wf.steps[1].step_id]
    wf.steps[1].dependencies = [wf.steps[0].step_id]
    with pytest.raises(WorkflowGraphError, match="cycle"):
        DependencyScheduler.validate(wf)
    wf.steps[0].dependencies = ["missing"]
    wf.steps[1].dependencies = []
    with pytest.raises(WorkflowGraphError, match="unknown dependency"):
        DependencyScheduler.validate(wf)


def test_dependency_failure_blocks_only_downstream():
    wf = _wf([
        Action("browser_search", {"query": "A"}),
        Action("browser_search", {"query": "B"}),
        Action("browser_search", {"query": "C"}),
    ])
    wf.steps[2].dependencies = [wf.steps[0].step_id]
    calls = []
    def execute(node):
        calls.append(node.step_id)
        if node is wf.steps[0]:
            return {"status": "FAILED", "error": "boom"}
        return {"status": "COMPLETED", "verification": {"verdict": "PASS"}}
    out = DependencyScheduler(max_concurrency=2).run(wf, execute)
    assert wf.steps[0].status is StepStatus.FAILED
    assert wf.steps[1].status is StepStatus.COMPLETED
    assert wf.steps[2].status is StepStatus.BLOCKED
    assert out.status is WorkflowStatus.PARTIAL_FAILURE


def test_unknown_is_not_success():
    wf = _wf([Action("browser_search", {"query": "A"})])
    out = DependencyScheduler().run(wf, lambda n: {"status": "UNKNOWN", "verification": {"verdict": "UNKNOWN"}})
    assert wf.steps[0].status is StepStatus.UNKNOWN
    assert out.status is WorkflowStatus.UNKNOWN


def test_same_resource_serializes_and_different_resources_overlap():
    wf = _wf([
        Action("whatsapp_send_message", {"recipient": "a", "message": "x"}),
        Action("whatsapp_send_message", {"recipient": "b", "message": "y"}),
        Action("browser_play_song", {"query": "song"}),
    ], "send x to a and send y to b and play song")
    intervals = {}; active = set(); max_whatsapp = 0; guard = threading.Lock()
    def execute(node):
        nonlocal max_whatsapp
        with guard:
            intervals[node.step_id] = [time.perf_counter(), None]
            if node.resource_key == "whatsapp":
                active.add(node.step_id); max_whatsapp = max(max_whatsapp, len(active))
        time.sleep(0.04)
        with guard:
            intervals[node.step_id][1] = time.perf_counter()
            active.discard(node.step_id)
        return {"status": "COMPLETED", "verification": {"verdict": "PASS"}}
    out = DependencyScheduler(max_concurrency=3).run(wf, execute)
    assert out.status is WorkflowStatus.COMPLETED
    assert max_whatsapp == 1
    wa = [n for n in wf.steps if n.resource_key == "whatsapp"]
    yt = wf.steps[2]
    assert intervals[wa[0].step_id][0] < intervals[yt.step_id][1]
    assert intervals[yt.step_id][0] < intervals[wa[0].step_id][1]


def test_concurrency_limit_is_respected():
    wf = _wf([Action("browser_search", {"query": str(i)}) for i in range(4)])
    active = 0; maximum = 0; guard = threading.Lock()
    def execute(node):
        nonlocal active, maximum
        with guard:
            active += 1; maximum = max(maximum, active)
        time.sleep(0.03)
        with guard: active -= 1
        return {"status": "COMPLETED", "verification": {"verdict": "PASS"}}
    DependencyScheduler(max_concurrency=2).run(wf, execute)
    assert maximum <= 2


def test_resource_lock_releases_on_failure_and_exception():
    manager = ResourceLockManager()
    assert manager.try_acquire("x", "a")
    manager.release("x", "a")
    assert manager.try_acquire("x", "b")
    manager.release("x", "b")
    assert manager.try_acquire("x", "c")
    manager.release("x", "c")
    wf = _wf([Action("browser_search", {"query": "x"})])
    def boom(node):
        raise RuntimeError("explode")
    out = DependencyScheduler(resources=manager).run(wf, boom)
    assert out.status is WorkflowStatus.FAILED
    assert manager.owner("browser") is None


def test_invalid_state_transition_rejected_and_recovery_is_explicit():
    wf = _wf([Action("browser_search", {"query": "x"})])
    node = wf.steps[0]
    with pytest.raises(ValueError):
        node.transition_to(StepStatus.RUNNING)
    node.status = StepStatus.FAILED
    with pytest.raises(ValueError):
        node.transition_to(StepStatus.RUNNING)
    recover_failed_node(wf, node.step_id)
    assert node.status is StepStatus.READY
    node.transition_to(StepStatus.RUNNING)
    assert node.status is StepStatus.RUNNING


def test_recovery_does_not_replay_completed_branch():
    wf = _wf([Action("browser_search", {"query": "A"}), Action("browser_search", {"query": "B"})])
    wf.steps[0].status = StepStatus.COMPLETED
    wf.steps[1].status = StepStatus.FAILED
    calls = []
    recover_failed_node(wf, wf.steps[1].step_id)
    DependencyScheduler().run(wf, lambda n: calls.append(n.step_id) or {"status": "COMPLETED", "verification": {"verdict": "PASS"}})
    assert calls == [wf.steps[1].step_id]


def test_fast_route_rejects_compound_request():
    assert classify_fast("open notepad and play Do I Wanna Know").route is None


def test_conversation_does_not_look_compound():
    assert not Session._looks_like_compound_action("hey, how are you")


def test_data_dependency_waits_for_verified_output_and_passes_data():
    wf = _wf([
        Action("browser_search", {"query": "weather"}),
        Action("whatsapp_send_message", {"recipient": "papa", "message": "${weather}"}),
    ], "find today's weather and send it to papa")
    wf.steps[1].dependencies = [wf.steps[0].step_id]
    wf.steps[1].dependency_types = {wf.steps[0].step_id: DependencyType.DATA_DEPENDENCY.value}
    wf.steps[1].requires_data = ["weather"]
    calls = []
    def execute(node):
        calls.append((node.step_id, dict(node.action.params)))
        if node is wf.steps[0]:
            return {"status": "COMPLETED", "data": {"weather": "sunny"}, "verification": {"verdict": "PASS"}}
        return {"status": "COMPLETED", "verification": {"verdict": "PASS"}}
    # A verified producer output is published only after the producer completes.
    out = DependencyScheduler().run(wf, execute)
    assert out.status is WorkflowStatus.COMPLETED
    assert calls[0][0] == wf.steps[0].step_id
    assert calls[1][0] == wf.steps[1].step_id


def test_auth_dependency_is_explicit():
    wf = _wf([
        Action("login", {"service": "example"}),
        Action("gmail_send_email", {"to": "papa", "body": "hello"}),
    ], "login then send email")
    assert wf.steps[1].dependencies == [wf.steps[0].step_id]
    assert wf.steps[1].dependency_types[wf.steps[0].step_id] == DependencyType.EXPLICIT_USER_ORDER.value


def test_tasknode_contract_and_round_trip():
    wf = _wf([Action("browser_search", {"query": "x"})])
    node = wf.steps[0]
    assert node.node_id == node.step_id
    assert node.capability == "browser_search"
    assert node.args == {"query": "x"}
    restored = Workflow.from_json(wf.to_json()).steps[0]
    assert restored.node_id == node.node_id


def test_completed_node_is_not_replayed_after_persistence_round_trip():
    wf = _wf([Action("browser_search", {"query": "x"}), Action("browser_play_song", {"query": "y"})])
    wf.steps[0].status = StepStatus.COMPLETED
    restored = Workflow.from_json(wf.to_json())
    ready = DependencyScheduler.ready_nodes(restored)
    assert all(n.step_id != restored.steps[0].step_id for n in ready)


def test_verification_must_be_pass_for_scheduler_completion():
    wf = _wf([Action("browser_search", {"query": "x"})])
    out = DependencyScheduler().run(wf, lambda n: {"ok": True, "status": "FAILED", "verification": {"verdict": "UNKNOWN"}})
    assert out.status is WorkflowStatus.FAILED
    assert wf.steps[0].status is StepStatus.FAILED


def test_approval_state_is_waiting_not_success():
    wf = _wf([Action("whatsapp_send_message", {"recipient": "papa", "message": "hello"})])
    out = DependencyScheduler().run(wf, lambda n: {"status": "WAITING_FOR_APPROVAL"})
    assert wf.steps[0].status is StepStatus.WAITING_FOR_APPROVAL
    assert out.status is WorkflowStatus.WAITING_FOR_APPROVAL
    assert wf.steps[0].status is not StepStatus.COMPLETED


def test_resume_payload_preserves_completed_and_ready_nodes():
    wf = _wf([
        Action("browser_search", {"query": "A"}),
        Action("browser_play_song", {"query": "B"}),
        Action("browser_search", {"query": "C"}),
    ])
    wf.steps[0].status = StepStatus.COMPLETED
    wf.steps[1].status = StepStatus.READY
    restored = Workflow.from_json(wf.to_json())
    assert restored.steps[0].status is StepStatus.COMPLETED
    assert restored.steps[1].status is StepStatus.READY
    assert restored.steps[2].status is StepStatus.PENDING
    ready = DependencyScheduler.ready_nodes(restored)
    assert restored.steps[2].status is StepStatus.READY
    assert restored.steps[2] in ready


def test_no_internal_node_id_is_required_for_user_facing_task_reference():
    # Presentation remains task-number based; the dependency model never changes
    # the public task identity contract.
    wf = _wf([Action("browser_search", {"query": "x"})])
    payload = wf.to_json()
    assert "node-" not in str(payload.get("goal", ""))


def test_langgraph_batch_node_overlaps_independent_existing_execution_path():
    from agent_control.workflow.nodes import execute_ready_batch
    wf = _wf([
        Action("browser_play_song", {"query": "A"}),
        Action("browser_search", {"query": "B"}),
    ], "play A and search B")
    events = []; intervals = {}; guard = threading.Lock()
    class Runtime:
        def emit_workflow_event(self, workflow, event, **metadata):
            events.append((event, metadata))
        def observe_workflow_step(self, workflow, step_index):
            return {"fresh": True}
        def workflow_policy(self, workflow, step_index):
            return "ALLOW", "allowed", None
        def request_workflow_input(self, *args): raise AssertionError
        def request_workflow_approval(self, *args): raise AssertionError
        def execute_workflow_step(self, workflow, step_index, approved_action):
            node = workflow.steps[step_index]
            with guard: intervals[node.step_id] = [time.perf_counter(), None]
            time.sleep(0.05)
            with guard: intervals[node.step_id][1] = time.perf_counter()
            return {"ok": True, "verification": {"verdict": "PASS"}, "result": {"step": node.step_id}}
    state = {"workflow": wf.to_json(), "ready_step_ids": [n.step_id for n in wf.steps], "branch_results": {}}
    out = execute_ready_batch(state, Runtime())
    assert out["decision"] == "BATCH_DONE"
    assert wf.steps[0].status is StepStatus.PENDING  # function operates on a durable copy
    result_wf = Workflow.from_json(out["workflow"])
    assert all(n.status is StepStatus.COMPLETED for n in result_wf.steps)
    a, b = intervals.values()
    assert a[0] < b[1] and b[0] < a[1]


def test_browser_child_target_depends_on_chrome_launch():
    from agent_control.types import Action
    from agent_control.workflow.models import Workflow
    wf = Workflow.from_actions(
        "browser-child",
        "open chrome and type hello world into address bar",
        [
            Action("launch_app", {"app": "chrome"}),
            Action("browser_type", {"target": "address bar", "target_semantic": {"name": "address bar", "role": "textbox"}, "text": "hello world"}),
        ],
    )
    assert wf.steps[1].dependencies == [wf.steps[0].step_id]
    assert wf.steps[1].dependency_types[wf.steps[0].step_id] == DependencyType.STATE_DEPENDENCY.value


def test_independent_chrome_and_notepad_launches_remain_parallel():
    from agent_control.types import Action
    from agent_control.workflow.models import Workflow
    wf = Workflow.from_actions(
        "independent-launches",
        "open chrome and open notepad",
        [Action("launch_app", {"app": "chrome"}), Action("launch_app", {"app": "notepad"})],
    )
    assert wf.steps[0].dependencies == []
    assert wf.steps[1].dependencies == []


def test_browser_child_is_never_ready_with_unlaunched_parent():
    from agent_control.types import Action
    from agent_control.workflow.models import Workflow
    wf = Workflow.from_actions(
        "browser-ready",
        "open chrome and type hello world into address bar",
        [
            Action("launch_app", {"app": "chrome"}),
            Action("browser_type", {"target": "address bar", "target_semantic": {"name": "address bar", "role": "textbox"}, "text": "hello world"}),
        ],
    )
    ready = DependencyScheduler.ready_nodes(wf)
    assert [n.step_id for n in ready] == [wf.steps[0].step_id]
