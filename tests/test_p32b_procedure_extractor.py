from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from agent_control.api import AgentResult, TaskStatus
from agent_control.procedure_extractor import (
    extract_procedure_candidate,
    is_learning_eligible,
    learn_verified_workflow,
    store_procedure_candidate,
)
from agent_control.procedure_store import ProcedureSourceType, ProcedureStatus, ProcedureStore
from agent_control.types import Action, Check, Verdict, VerificationResult
from agent_control.workflow.models import StepStatus, Workflow, WorkflowStatus, WorkflowStep


def verified_workflow(*, two_steps: bool = True) -> Workflow:
    actions = [Action("browser_search", {"query": "{query}"}, rationale="search browser")]
    if two_steps:
        actions.append(Action("browser_play_song", {"query": "{query}"}, rationale="play result"))
    wf = Workflow.from_actions("wf-1", "search and play", actions)
    wf.status = WorkflowStatus.COMPLETED
    for step in wf.steps:
        step.status = StepStatus.COMPLETED
        step.verification = {"verdict": "PASS", "checks": [{"name": f"{step.step_id}_verified", "verdict": "PASS"}]}
    return wf


def result_for(wf: Workflow, *, status=TaskStatus.SUCCESS, verified="PASS") -> AgentResult:
    # The extractor may consume the authoritative workflow directly, but this
    # result exercises the public AgentResult gate as well.
    from agent_control.runner import RunOutcome
    return AgentResult(request=wf.goal, task_id=wf.workflow_id, status=status, verified=verified,
                       completed=[s.step_id for s in wf.steps], outcome=None)


def test_01_fully_verified_workflow_produces_candidate():
    candidate = extract_procedure_candidate(verified_workflow())
    assert candidate is not None
    assert candidate.status is ProcedureStatus.CANDIDATE


def test_02_execution_success_without_verification_does_not_learn():
    wf = verified_workflow()
    wf.status = WorkflowStatus.RUNNING
    assert extract_procedure_candidate(result_for(wf), workflow=wf) is None


def test_03_unknown_workflow_does_not_learn():
    wf = verified_workflow()
    wf.status = WorkflowStatus.UNKNOWN
    assert extract_procedure_candidate(wf) is None
    assert is_learning_eligible(wf).reason == "verification_unknown"


def test_04_failed_verification_does_not_learn():
    wf = verified_workflow()
    wf.steps[0].status = StepStatus.FAILED
    wf.steps[0].verification = {"verdict": "FAIL"}
    wf.status = WorkflowStatus.FAILED
    assert extract_procedure_candidate(wf) is None


def test_05_partial_workflow_does_not_learn():
    wf = verified_workflow()
    wf.steps[-1].status = StepStatus.PENDING
    wf.status = WorkflowStatus.PARTIAL_FAILURE
    assert extract_procedure_candidate(wf) is None


def test_06_cancelled_workflow_does_not_learn():
    wf = verified_workflow()
    wf.status = WorkflowStatus.CANCELLED
    wf.cancel_requested = True
    assert is_learning_eligible(wf).reason == "cancelled"
    assert extract_procedure_candidate(wf) is None


def test_07_policy_denied_workflow_does_not_learn():
    wf = verified_workflow()
    wf.steps[0].policy_state = "DENIED"
    assert extract_procedure_candidate(wf) is None
    assert is_learning_eligible(wf).reason == "policy_denied"


def test_08_recovery_exhausted_workflow_does_not_learn():
    wf = verified_workflow()
    wf.steps[0].status = StepStatus.RECOVERY_REQUIRED
    wf.status = WorkflowStatus.RECOVERING
    wf.recovery_attempts = wf.max_recovery_attempts
    assert extract_procedure_candidate(wf) is None


def test_09_semantic_browser_action_becomes_procedure_step():
    c = extract_procedure_candidate(verified_workflow(two_steps=False))
    assert c.steps[0].action_name == "browser_search"
    assert c.steps[0].arguments == {"query": "{query}"}


def test_10_no_coordinates_are_stored():
    wf = verified_workflow(two_steps=False)
    wf.steps[0].action = Action("browser_search", {"query": "{query}", "x": 712, "y": 84})
    c = extract_procedure_candidate(wf)
    assert c is not None
    assert "x" not in c.steps[0].arguments and "y" not in c.steps[0].arguments


def test_11_no_pixel_information_is_stored():
    wf = verified_workflow(two_steps=False)
    wf.steps[0].action = Action("browser_search", {"query": "{query}", "coordinates": [1, 2], "pixel": "1,2"})
    c = extract_procedure_candidate(wf)
    assert c is not None
    assert "coordinates" not in c.steps[0].arguments and "pixel" not in c.steps[0].arguments


def test_12_no_fragile_dom_selector_is_stored():
    wf = verified_workflow(two_steps=False)
    wf.steps[0].action = Action("browser_search", {"query": "{query}", "css_selector": "#search", "xpath": "//input"})
    c = extract_procedure_candidate(wf)
    assert c is not None
    raw = json.dumps(c.to_dict())
    assert "#search" not in raw and "//input" not in raw


def test_13_explicit_task_parameter_becomes_procedure_parameter():
    wf = verified_workflow(two_steps=False)
    c = extract_procedure_candidate(wf, {"parameters": {"query": {"type": "string", "required": True}}})
    assert c is not None
    assert [p.name for p in c.parameters] == ["query"]


def test_14_fixed_values_are_not_automatically_parameters():
    wf = verified_workflow(two_steps=False)
    wf.steps[0].action = Action("browser_search", {"query": "fixed search"})
    c = extract_procedure_candidate(wf)
    assert c is not None
    assert c.parameters == ()
    assert "query" not in c.steps[0].arguments


def test_15_sensitive_runtime_value_is_not_persisted():
    wf = verified_workflow(two_steps=False)
    wf.steps[0].action = Action("whatsapp_send_message", {"recipient": "alice@example.com", "message": "private text"})
    c = extract_procedure_candidate(wf)
    assert c is not None
    raw = json.dumps(c.to_dict())
    assert "alice@example.com" not in raw and "private text" not in raw
    assert all(p.sensitive for p in c.parameters)


def test_16_sensitive_value_is_not_emitted_by_extractor(monkeypatch, caplog):
    wf = verified_workflow(two_steps=False)
    wf.steps[0].action = Action("whatsapp_send_message", {"recipient": "alice@example.com", "message": "private text"})
    with caplog.at_level("DEBUG"):
        extract_procedure_candidate(wf)
    assert "alice@example.com" not in caplog.text
    assert "private text" not in caplog.text


def test_17_candidate_starts_as_candidate():
    assert extract_procedure_candidate(verified_workflow()).status is ProcedureStatus.CANDIDATE


def test_18_provenance_records_verified_workflow():
    c = extract_procedure_candidate(verified_workflow())
    assert c.provenance.source_type is ProcedureSourceType.VERIFIED_WORKFLOW
    assert c.provenance.source_task_id == "wf-1"


def test_19_procedure_name_is_deterministic():
    a = extract_procedure_candidate(verified_workflow())
    b = extract_procedure_candidate(verified_workflow())
    assert a.procedure_id == b.procedure_id and a.name == b.name


def test_20_same_workflow_does_not_silently_overwrite(tmp_path):
    wf = verified_workflow()
    with ProcedureStore(tmp_path / "procedures.sqlite3") as store:
        first = learn_verified_workflow(wf, store=store)
        second = learn_verified_workflow(wf, store=store)
        assert first.procedure_id == second.procedure_id
        assert len(store.list_procedures()) == 1


def test_21_extraction_has_no_browser_side_effects():
    browser = Mock()
    c = extract_procedure_candidate(verified_workflow())
    assert c is not None
    browser.assert_not_called()


def test_22_extraction_has_no_messaging_side_effects():
    messaging = Mock()
    c = extract_procedure_candidate(verified_workflow())
    assert c is not None
    messaging.assert_not_called()


def test_23_extraction_does_not_invoke_planner():
    planner = Mock()
    c = extract_procedure_candidate(verified_workflow())
    assert c is not None
    planner.assert_not_called()


def test_24_extraction_does_not_invoke_llm():
    llm = Mock()
    c = extract_procedure_candidate(verified_workflow())
    assert c is not None
    llm.assert_not_called()


def test_25_stored_candidate_survives_restart(tmp_path):
    path = tmp_path / "procedures.sqlite3"
    wf = verified_workflow()
    with ProcedureStore(path) as store:
        candidate = learn_verified_workflow(wf, store=store)
        pid = candidate.procedure_id
    with ProcedureStore(path) as store:
        restored = store.get_procedure(pid)
        assert restored is not None
        assert restored.status is ProcedureStatus.CANDIDATE
        assert restored.provenance.source_type is ProcedureSourceType.VERIFIED_WORKFLOW


def test_26_agent_result_success_still_requires_workflow_completion():
    wf = verified_workflow()
    wf.status = WorkflowStatus.RUNNING
    result = result_for(wf)
    assert is_learning_eligible(result, workflow=wf).eligible is False


def test_27_agent_result_unknown_never_learns_even_with_completed_workflow():
    wf = verified_workflow()
    result = result_for(wf, status=TaskStatus.UNKNOWN, verified="UNKNOWN")
    assert extract_procedure_candidate(result, workflow=wf) is None


def test_28_agent_result_policy_blocked_never_learns():
    wf = verified_workflow()
    result = result_for(wf, status=TaskStatus.POLICY_BLOCKED, verified="UNKNOWN")
    assert extract_procedure_candidate(result, workflow=wf) is None


def test_29_verified_step_without_required_verification_is_not_falsely_promoted():
    wf = verified_workflow(two_steps=False)
    wf.steps[0].requires_verification = True
    wf.steps[0].verification = {"verdict": "UNKNOWN"}
    assert extract_procedure_candidate(wf) is None


def test_30_no_raw_goal_is_persisted_as_trigger():
    wf = verified_workflow(two_steps=False)
    wf.goal = "send private message: secret phrase"
    c = extract_procedure_candidate(wf)
    assert c is not None
    assert c.trigger_pattern == c.name
    assert "secret phrase" not in json.dumps(c.to_dict())


def test_31_session_terminal_workflow_persists_candidate_without_execution(tmp_path):
    from agent_control.response import Narrator
    from agent_control.session import Session

    session = Session(Narrator(write=lambda _: None, speaker=None), planner="mock", show_status=False)
    session._procedure_store = ProcedureStore(tmp_path / "procedures.sqlite3")
    wf = verified_workflow()
    try:
        session._learn_verified_workflow(wf)
        stored = session._procedure_store.list_procedures()
        assert len(stored) == 1
        assert stored[0].status is ProcedureStatus.CANDIDATE
    finally:
        session.close()


def test_32_session_does_not_persist_failed_workflow(tmp_path):
    from agent_control.response import Narrator
    from agent_control.session import Session

    session = Session(Narrator(write=lambda _: None, speaker=None), planner="mock", show_status=False)
    session._procedure_store = ProcedureStore(tmp_path / "procedures.sqlite3")
    wf = verified_workflow()
    wf.status = WorkflowStatus.FAILED
    try:
        session._learn_verified_workflow(wf)
        assert session._procedure_store.list_procedures() == []
    finally:
        session.close()
