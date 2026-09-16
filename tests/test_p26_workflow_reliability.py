"""Deterministic P2.6 workflow reliability/recovery tests."""
from __future__ import annotations

from agent_control.recovery import Recovery, RecoveryBudget, action_can_retry, retry_safety
from agent_control.types import Action, RetrySafety
from agent_control.workflow.models import StepStatus, Workflow, WorkflowStatus
from agent_control.workflow.nodes import execute, recover, select_next_step, verify
from agent_control.workflow.scheduler import DependencyScheduler


class Runtime:
    def __init__(self, results=None, verifications=None):
        self.results = list(results or [])
        self.verifications = list(verifications or [])
        self.events = []
        self.observations = 0
        self.executions = []

    def emit_workflow_event(self, workflow, event, **metadata):
        self.events.append((event, metadata))

    def observe_workflow_step(self, workflow, step_index):
        self.observations += 1
        return {"fresh": True, "observed_at": __import__("time").time()}

    def workflow_policy(self, workflow, step_index):
        return "ALLOW", "allowed", None

    def request_workflow_input(self, *args):
        raise AssertionError

    def request_workflow_approval(self, *args):
        raise AssertionError

    def resolve_workflow_approval(self, *args):
        return None

    def resolve_workflow_input(self, *args):
        return None

    def execute_workflow_step(self, workflow, step_index, approved_action):
        self.executions.append(workflow.steps[step_index].step_id)
        return self.results.pop(0)

    def verify_workflow_step(self, workflow, step_index):
        return self.verifications.pop(0)


def wf(actions):
    return Workflow.from_actions("p26", "original task", actions)


def test_unknown_never_becomes_pass():
    w = wf([Action("browser_search", {"query": "x"})])
    w.steps[0].status = StepStatus.RECOVERY_REQUIRED
    out = verify({"workflow": w.to_json(), "step_verification": {"verdict": "UNKNOWN"}}, Runtime())
    restored = Workflow.from_json(out["workflow"])
    assert restored.steps[0].status is StepStatus.RECOVERY_REQUIRED
    assert restored.status is WorkflowStatus.RECOVERING
    assert out["decision"] == "UNKNOWN"


def test_safe_retry_metadata_is_bounded():
    action = Action("create_dir", {"path": "x"}, idempotent=True)
    budget = RecoveryBudget(max_retries=2, max_reobserves=2, max_replans=1, max_recovery_attempts=2)
    recovery = Recovery(budget=budget)
    assert action_can_retry(action)
    assert recovery.decide(__import__("agent_control.types", fromlist=["FailureClass"]).FailureClass.ACTION_FAILED)[0].value == "RETRY"
    assert recovery.decide(__import__("agent_control.types", fromlist=["FailureClass"]).FailureClass.ACTION_FAILED)[0].value == "RETRY"
    assert recovery.decide(__import__("agent_control.types", fromlist=["FailureClass"]).FailureClass.ACTION_FAILED)[0].value == "ABORT"




def test_type_text_is_never_blindly_retried_after_unknown():
    action = Action("type_text", {"app": "notepad", "text": "hello"})
    assert retry_safety(action) is RetrySafety.REOBSERVE_FIRST
    assert not action_can_retry(action, post_action_unknown=True)

def test_non_idempotent_send_is_reobserve_first_not_blind_retry():
    action = Action("gmail_send_email", {"to": "x", "body": "hello"})
    assert retry_safety(action) is RetrySafety.REOBSERVE_FIRST
    assert not action_can_retry(action, post_action_unknown=True)


def test_unknown_recovery_reobserves_and_can_prove_pass():
    w = wf([Action("gmail_send_email", {"to": "x", "body": "hello"})])
    w.steps[0].status = StepStatus.RECOVERY_REQUIRED
    rt = Runtime(verifications=[{"verdict": "PASS"}])
    out = recover({"workflow": w.to_json(), "failure_class": "verification_unknown"}, rt)
    restored = Workflow.from_json(out["workflow"])
    assert out["decision"] == "PASS"
    assert restored.steps[0].status is StepStatus.COMPLETED
    assert rt.observations == 1
    assert not rt.executions


def test_unknown_recovery_does_not_duplicate_send_when_still_unknown():
    w = wf([Action("whatsapp_send_message", {"recipient": "mummy", "message": "hello"})])
    w.steps[0].status = StepStatus.RECOVERY_REQUIRED
    rt = Runtime(verifications=[{"verdict": "UNKNOWN"}])
    out = recover({"workflow": w.to_json(), "failure_class": "verification_unknown"}, rt)
    restored = Workflow.from_json(out["workflow"])
    assert out["decision"] == "UNKNOWN"
    assert restored.status is WorkflowStatus.UNKNOWN
    assert not rt.executions


def test_resume_uses_last_verified_step():
    w = wf([Action("browser_search", {"query": "a"}), Action("browser_search", {"query": "b"}), Action("browser_search", {"query": "c"})])
    w.steps[0].status = StepStatus.COMPLETED
    w.steps[1].status = StepStatus.COMPLETED
    w.steps[2].status = StepStatus.READY
    out = select_next_step({"workflow": w.to_json()}, Runtime())
    restored = Workflow.from_json(out["workflow"])
    assert restored.steps[0].status is StepStatus.COMPLETED
    assert restored.steps[1].status is StepStatus.COMPLETED
    assert restored.steps[2].status is StepStatus.READY


def test_partial_completion_is_not_success():
    w = wf([Action("browser_search", {"query": "a"}), Action("browser_search", {"query": "b"})])
    w.steps[0].status = StepStatus.COMPLETED
    w.steps[1].status = StepStatus.UNKNOWN
    assert DependencyScheduler.aggregate(w) is WorkflowStatus.UNKNOWN


def test_dependency_failure_blocks_dependents_but_independent_branch_runs():
    w = wf([Action("browser_search", {"query": "a"}), Action("browser_search", {"query": "b"}), Action("browser_search", {"query": "d"})])
    w.steps[1].dependencies = [w.steps[0].step_id]
    w.steps[2].dependencies = []
    calls = []
    def run(n):
        calls.append(n.step_id)
        if n is w.steps[0]:
            return {"status": "FAILED", "error": "boom"}
        return {"status": "COMPLETED", "verification": {"verdict": "PASS"}}
    out = DependencyScheduler(max_concurrency=2).run(w, run)
    assert w.steps[1].status is StepStatus.BLOCKED
    assert w.steps[2].status is StepStatus.COMPLETED
    assert out.status is WorkflowStatus.PARTIAL_FAILURE


def test_cancellation_prevents_new_steps():
    w = wf([Action("browser_search", {"query": "a"}), Action("browser_search", {"query": "b"})])
    w.cancel_requested = True
    out = DependencyScheduler().run(w, lambda n: (_ for _ in ()).throw(AssertionError("must not execute")))
    assert out.status is WorkflowStatus.CANCELLED
    assert all(s.status is StepStatus.CANCELLED for s in w.steps)


def test_original_goal_and_recovery_state_round_trip():
    w = wf([Action("gmail_send_email", {"to": "x", "body": "hello"})])
    w.steps[0].status = StepStatus.UNKNOWN
    w.steps[0].failure_category = "verification_unknown"
    w.steps[0].recovery = {"attempt": 1, "retry_safety": "REOBSERVE_FIRST"}
    w.recovery_attempts = 1
    restored = Workflow.from_json(w.to_json())
    assert restored.goal == "original task"
    assert restored.steps[0].failure_category == "verification_unknown"
    assert restored.steps[0].recovery["attempt"] == 1
    assert restored.recovery_attempts == 1


def test_approval_state_is_reused_during_recovery():
    w = wf([Action("gmail_send_email", {"to": "x", "body": "hello"})])
    w.steps[0].policy_state = "APPROVED"
    # Serialization is the checkpoint boundary; approval ownership remains attached to the step.
    restored = Workflow.from_json(w.to_json())
    assert restored.steps[0].policy_state == "APPROVED"


def test_failure_taxonomy_contains_p26_categories():
    from agent_control.types import FailureClass
    required = {
        "execution_exception", "execution_timeout", "browser_connection_failed",
        "application_not_found", "target_not_found", "stale_observation",
        "verification_failed", "verification_unknown", "policy_denied",
        "approval_required", "resource_unavailable", "dependency_failed",
        "action_may_have_succeeded", "cancellation_requested",
        "recovery_exhausted", "unrecoverable_failure",
    }
    values = {x.value for x in FailureClass} | {x.name.lower() for x in FailureClass}
    # Existing compatibility vocabulary is uppercase; P2.6 additions are explicit.
    assert required <= values


def test_action_recovery_metadata_round_trips():
    action = Action(
        "gmail_send_email", {"to": "x", "body": "hello"},
        retry_safety=RetrySafety.REOBSERVE_FIRST,
        idempotent=False,
        side_effect_level="high",
        requires_fresh_observation=True,
        verification_required=True,
        recovery_strategy="REOBSERVE",
    )
    restored = Workflow.from_json(Workflow.from_actions("w", "goal", [action]).to_json()).steps[0].action
    assert restored.retry_safety is RetrySafety.REOBSERVE_FIRST
    assert restored.idempotent is False
    assert restored.side_effect_level == "high"
    assert restored.recovery_strategy == "REOBSERVE"


def test_browser_failure_requests_fresh_observation_before_safe_retry():
    w = wf([Action("browser_play_song", {"query": "song"})])
    w.steps[0].status = StepStatus.RECOVERY_REQUIRED
    rt = Runtime(results=[{"status": "COMPLETED", "verification": {"verdict": "PASS"}}])
    out = recover({"workflow": w.to_json(), "failure_class": "browser_connection_failed"}, rt)
    restored = Workflow.from_json(out["workflow"])
    assert restored.steps[0].status is StepStatus.READY
    assert rt.observations >= 1
    assert not rt.executions


def test_step_retry_budget_is_explicit_and_bounded():
    w = wf([Action("browser_search", {"query": "x"})])
    w.steps[0].max_step_retries = 2
    w.steps[0].attempt_count = 3
    w.steps[0].status = StepStatus.RECOVERY_REQUIRED
    rt = Runtime()
    out = recover({"workflow": w.to_json(), "failure_class": "execution_exception"}, rt)
    assert out["decision"] == "UNKNOWN"
    assert Workflow.from_json(out["workflow"]).steps[0].recovery["decision"] == "STEP_RETRY_EXHAUSTED"


def test_workflow_recovery_budget_is_separate_from_step_budget():
    w = wf([Action("browser_search", {"query": "x"})])
    w.max_recovery_attempts = 1
    w.recovery_attempts = 1
    w.steps[0].status = StepStatus.RECOVERY_REQUIRED
    out = recover({"workflow": w.to_json(), "failure_class": "execution_exception"}, Runtime())
    assert out["decision"] == "UNKNOWN"
    assert Workflow.from_json(out["workflow"]).steps[0].failure_category == "recovery_exhausted"


def test_execute_preserves_known_application_not_found_as_failed():
    from agent_control.types import Action
    from agent_control.workflow.models import StepStatus, Workflow, WorkflowStatus
    from agent_control.workflow.nodes import execute

    workflow = Workflow.from_actions("env-failure", "open word", [Action("launch_app", {"app": "word"})])

    class Runtime:
        def emit_workflow_event(self, workflow, event, **metadata):
            pass

        def execute_workflow_step(self, workflow, index, approved_action):
            return {
                "ok": False,
                "error": "could not resolve installed application 'word'",
                "error_category": "application_not_found",
                "result": {},
                "verification": {"verdict": "UNKNOWN"},
            }

    out = execute({"workflow": workflow.to_json(), "current_step_id": workflow.steps[0].step_id}, Runtime())
    result = Workflow.from_json(out["workflow"])
    assert result.steps[0].status is StepStatus.FAILED
    assert result.steps[0].failure_category == "application_not_found"
    assert result.status is WorkflowStatus.FAILED
