"""P3.2-B runtime learning-boundary regressions."""
from __future__ import annotations

from pathlib import Path

from agent_control.api import AgentResult, TaskStatus
from agent_control.procedure_store import ProcedureStatus, ProcedureStore
from agent_control.response import Narrator
from agent_control.runner import AgentState, RunOutcome
from agent_control.session import Prepared, Session, UserTask
from agent_control.types import Action, Check, VerificationResult, Verdict
from agent_control.workflow.models import StepStatus, Workflow, WorkflowStatus


def _verified_workflow(workflow_id: str = "wf-runtime") -> Workflow:
    workflow = Workflow.from_actions(
        workflow_id,
        "play a song",
        [Action("browser_play_song", {"query": "Do I Wanna Know"}, rationale="play song")],
    )
    workflow.status = WorkflowStatus.COMPLETED
    for step in workflow.steps:
        step.status = StepStatus.COMPLETED
        step.verification = {
            "verdict": "PASS",
            "checks": [{"name": "independent_checkpoint", "verdict": "PASS"}],
        }
    return workflow


def _session_with_store(tmp_path: Path) -> tuple[Session, ProcedureStore]:
    session = Session(
        Narrator(write=lambda _: None, speaker=None),
        planner="mock",
        show_status=False,
        debug=True,
        runtime_persistence_path=":memory:",
    )
    store = ProcedureStore(tmp_path / "procedures.sqlite3")
    session._procedure_store = store
    return session, store


def test_runtime_completion_path_persists_candidate(tmp_path):
    """The terminal _resume_workflow path must reach SQLite, not just extract."""
    session, store = _session_with_store(tmp_path)
    workflow = _verified_workflow()
    session._runtime_create(
        workflow.goal,
        workflow.workflow_id,
        runtime_task_id=workflow.workflow_id,
        task_type="workflow",
    )
    session._runtime.update_workflow(
        workflow.workflow_id,
        workflow.to_json(),
        event_type="WORKFLOW_VERIFIED",
    )

    try:
        session._resume_workflow(workflow, source="text")
        procedures = store.list_procedures()
        assert len(procedures) == 1
        procedure = procedures[0]
        assert procedure.status is ProcedureStatus.CANDIDATE
        assert procedure.provenance.source_type.value == "VERIFIED_WORKFLOW"
        assert procedure.provenance.source_task_id == workflow.workflow_id
        store.close()
        reopened = ProcedureStore(tmp_path / "procedures.sqlite3")
        try:
            persisted = reopened.list_procedures()
            assert len(persisted) == 1
            assert persisted[0].status is ProcedureStatus.CANDIDATE
        finally:
            reopened.close()
    finally:
        session.close()


def test_runtime_completion_path_does_not_learn_without_verification(tmp_path):
    session, store = _session_with_store(tmp_path)
    workflow = _verified_workflow("wf-unverified")
    workflow.steps[0].verification = {"verdict": "UNKNOWN"}
    session._runtime_create(
        workflow.goal,
        workflow.workflow_id,
        runtime_task_id=workflow.workflow_id,
        task_type="workflow",
    )
    try:
        session._resume_workflow(workflow, source="text")
        assert store.list_procedures() == []
    finally:
        session.close()


def test_completed_verified_workflow_survives_serialization_for_learning(tmp_path):
    session, store = _session_with_store(tmp_path)
    original = _verified_workflow("wf-roundtrip")
    reconstructed = Workflow.from_json(original.to_json())
    assert reconstructed.status is WorkflowStatus.COMPLETED
    assert reconstructed.steps[0].status is StepStatus.COMPLETED
    assert reconstructed.steps[0].verification["verdict"] == "PASS"
    try:
        session._learn_verified_workflow(reconstructed)
        procedures = store.list_procedures()
        assert len(procedures) == 1
        assert procedures[0].status is ProcedureStatus.CANDIDATE
    finally:
        session.close()


def test_unknown_verification_survives_serialization_as_non_eligible(tmp_path):
    session, store = _session_with_store(tmp_path)
    original = _verified_workflow("wf-roundtrip-unknown")
    original.steps[0].verification = {"verdict": "UNKNOWN", "reason": "inconclusive"}
    reconstructed = Workflow.from_json(original.to_json())
    try:
        session._learn_verified_workflow(reconstructed)
        assert store.list_procedures() == []
    finally:
        session.close()


def _fast_success_result(task_id: str) -> AgentResult:
    verification = VerificationResult(
        checks=[Check("fast_browser_verification", Verdict.PASS, reason="independent browser verification passed")],
        label=f"fast:{task_id}",
    )
    outcome = RunOutcome(
        task_id=task_id,
        condition="runtime",
        trial=1,
        planner_name="fast-router",
        reported_success=True,
        verified=Verdict.PASS,
        final=verification,
        state=AgentState.COMPLETED,
    )
    return AgentResult(
        request="play do i wanna know",
        task_id=task_id,
        status=TaskStatus.SUCCESS,
        verified=Verdict.PASS.value,
        completed=["step-1"],
        detail="verified fast interaction",
        outcome=outcome,
    )


class _FakeFastTask:
    bucket = "fast_interaction"
    goal = "play do i wanna know"

    def action_template(self):
        return Action(
            "browser_play_song",
            {"query": "Do I Wanna Know"},
            rationale="semantic YouTube song playback",
        )


def test_fast_interaction_runtime_completion_persists_verified_candidate(tmp_path, monkeypatch):
    """The real Session._run completion boundary must feed fast routes into P3.2-B."""
    output = []
    session = Session(
        Narrator(write=output.append, speaker=None),
        planner="mock",
        show_status=False,
        debug=True,
        runtime_persistence_path=":memory:",
    )
    store = ProcedureStore(tmp_path / "procedures.sqlite3")
    session._procedure_store = store
    task_id = "fast-runtime-learning"
    result = _fast_success_result(task_id)
    monkeypatch.setattr(session, "_browser_for_task", lambda *args, **kwargs: object())
    monkeypatch.setattr("agent_control.session.api.run_agent_task", lambda *args, **kwargs: result)
    prepared = Prepared(
        task=UserTask(raw="play do i wanna know", text="play do i wanna know", task_id=task_id, status="accepted"),
        goal="play do i wanna know",
        kind="fast interaction",
        task_obj=_FakeFastTask(),
        runtime_task_id=task_id,
    )

    try:
        turn = session._run(prepared)
        assert turn.result is result
        procedures = store.list_procedures()
        assert len(procedures) == 1
        assert procedures[0].status is ProcedureStatus.CANDIDATE
        assert procedures[0].name == "browser.play.song"
        assert any("learning hook entered" in line for line in output)
        assert any("eligibility=True" in line for line in output)
        assert any("extraction=candidate" in line for line in output)
        assert any("persistence=committed" in line for line in output)
        events = session._runtime.events()
        assert any(event.event_type == "procedure_candidate_created" for event in events)
        assert not any(event.event_type == "procedure_learning_skipped" for event in events)
    finally:
        session.close()


def test_fast_interaction_runtime_failure_does_not_learn(tmp_path, monkeypatch):
    session = Session(
        Narrator(write=lambda _: None, speaker=None),
        planner="mock",
        show_status=False,
        runtime_persistence_path=":memory:",
    )
    store = ProcedureStore(tmp_path / "procedures.sqlite3")
    session._procedure_store = store
    task_id = "fast-runtime-failed"
    verification = VerificationResult(
        checks=[Check("fast_browser_verification", Verdict.FAIL, reason="independent verification failed")],
        label=f"fast:{task_id}",
    )
    outcome = RunOutcome(
        task_id=task_id,
        condition="runtime",
        trial=1,
        planner_name="fast-router",
        reported_success=False,
        verified=Verdict.FAIL,
        final=verification,
        state=AgentState.FAILED,
    )
    result = AgentResult(
        request="play do i wanna know",
        task_id=task_id,
        status=TaskStatus.FAILED,
        verified=Verdict.FAIL.value,
        outcome=outcome,
        detail="failed",
    )
    monkeypatch.setattr(session, "_browser_for_task", lambda *args, **kwargs: object())
    monkeypatch.setattr("agent_control.session.api.run_agent_task", lambda *args, **kwargs: result)
    prepared = Prepared(
        task=UserTask(raw="play do i wanna know", text="play do i wanna know", task_id=task_id, status="accepted"),
        goal="play do i wanna know",
        kind="fast interaction",
        task_obj=_FakeFastTask(),
        runtime_task_id=task_id,
    )
    try:
        session._run(prepared)
        assert store.list_procedures() == []
    finally:
        session.close()
