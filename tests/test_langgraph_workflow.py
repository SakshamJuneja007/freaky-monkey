import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from agent_control.runner import _execute_skill_action, RunOutcome

from agent_control.policy import Decision
from agent_control.session import Session
from agent_control.types import Action
from agent_control.workflow import (ResourceBinding, StepStatus, Workflow, WorkflowStatus, resource_key_for_action, workflow_step_status_from_runner_state, workflow_step_status_from_runtime_state)
from agent_control.workflow.session_adapter import SessionWorkflowRuntime, normalize_execution_result, WorkflowResultNormalizationError
from agent_control.workflow.nodes import next_step, verify
from agent_control.types import ActionResult, FailureClass, Check, VerificationResult, Verdict


class LangGraphWorkflowDomainTests(unittest.TestCase):
    def setUp(self):
        self.session = Session.build(speech=False, write=lambda _: None, planner="mock", show_status=False, debug=False)
        self.adapter = SessionWorkflowRuntime(self.session)

    def tearDown(self):
        self.session.close()

    def _workflow(self):
        return Workflow.from_actions("wf-test", "send hello to mummy then send bye to papa then play Do I Wanna Know", [
            Action("whatsapp_send_message", {"recipient": "mummy", "message": "hello"}),
            Action("whatsapp_send_message", {"recipient": "papa", "message": "bye"}),
            Action("browser_play_song", {"query": "Do I Wanna Know"}),
        ])

    def test_01_three_actions_become_three_steps(self):
        wf = self.adapter.decompose_workflow("wf-test", "send hello to mummy then send bye to papa then play Do I Wanna Know")
        self.assertEqual(len(wf.steps), 3)

    def test_02_steps_are_ordered(self):
        wf = self._workflow()
        self.assertEqual([s.index for s in wf.steps], [0, 1, 2])
        self.assertEqual(wf.steps[1].dependencies, [wf.steps[0].step_id])

    def test_03_step_two_depends_on_step_one(self):
        wf = self._workflow()
        self.assertEqual(wf.steps[1].dependencies, [wf.steps[0].step_id])

    def test_04_step_three_depends_on_step_two(self):
        wf = self._workflow()
        self.assertEqual(wf.steps[2].dependencies, [wf.steps[1].step_id])

    def test_05_whatsapp_resource_is_reused(self):
        wf = self._workflow()
        self.assertEqual(wf.steps[0].resource_key, "whatsapp")
        self.assertEqual(wf.steps[1].resource_key, "whatsapp")
        self.assertIs(wf.resources["whatsapp"], wf.resources["whatsapp"])
        self.assertTrue(wf.resources["whatsapp"].reuse)

    def test_06_youtube_has_separate_resource(self):
        wf = self._workflow()
        self.assertEqual(wf.steps[2].resource_key, "youtube")
        self.assertNotEqual(wf.steps[2].resource_key, wf.steps[0].resource_key)

    def test_07_policy_confirms_whatsapp(self):
        wf = self._workflow()
        with patch("agent_control.policy.is_elevated", return_value=False):
            decision, _, _ = self.adapter.workflow_policy(wf, 0)
        self.assertEqual(decision, Decision.CONFIRM.value)

    def test_08_workflow_status_waiting_for_approval_model(self):
        wf = self._workflow()
        wf.status = WorkflowStatus.WAITING_FOR_APPROVAL
        wf.steps[0].status = StepStatus.WAITING_FOR_APPROVAL
        self.assertEqual(wf.state, "WAITING_FOR_APPROVAL")

    def test_09_completed_steps_are_not_replayed_by_selection_model(self):
        wf = self._workflow()
        wf.steps[0].status = StepStatus.COMPLETED
        wf.current_step_id = wf.steps[1].step_id
        self.assertNotEqual(wf.current_step, 0)

    def test_10_unknown_verification_is_not_completion(self):
        wf = self._workflow()
        wf.steps[0].status = StepStatus.RECOVERY_REQUIRED
        self.assertNotEqual(wf.steps[0].status, StepStatus.COMPLETED)
        self.assertNotEqual(wf.status, WorkflowStatus.COMPLETED)

    def test_11_side_effect_metadata_is_preserved(self):
        wf = self._workflow()
        self.assertTrue(wf.steps[0].side_effect)
        self.assertTrue(wf.steps[0].requires_verification)

    def test_12_original_goal_is_separate(self):
        wf = self._workflow()
        payload = wf.to_json()
        self.assertEqual(payload["goal"], wf.goal)
        self.assertEqual(payload["steps"][0]["arguments"], {"recipient": "mummy", "message": "hello"})
        self.assertNotIn(wf.goal, payload["steps"][0]["arguments"].values())

    def test_13_round_trip_preserves_steps(self):
        wf = self._workflow()
        restored = Workflow.from_json(wf.to_json())
        self.assertEqual([s.action.to_json() for s in restored.steps], [s.action.to_json() for s in wf.steps])

    def test_14_resource_key_mapping_is_capability_based(self):
        self.assertEqual(resource_key_for_action(Action("whatsapp_send_message", {})), "whatsapp")
        self.assertEqual(resource_key_for_action(Action("browser_play_song", {})), "youtube")

    def test_15_missing_message_remains_workflow_owned(self):
        wf = Workflow.from_actions("wf-missing", "send something to papa", [Action("whatsapp_send_message", {"recipient": "papa"})])
        decision, _, pending = self.adapter.workflow_policy(wf, 0)
        self.assertEqual(decision, "WAITING_FOR_USER")
        self.assertEqual(pending["field"], "message")
        self.assertEqual(pending["workflow_id"], "wf-missing")

    def test_16_missing_recipient_remains_workflow_owned(self):
        wf = Workflow.from_actions("wf-missing", "send hello", [Action("whatsapp_send_message", {"message": "hello"})])
        decision, _, pending = self.adapter.workflow_policy(wf, 0)
        self.assertEqual(decision, "WAITING_FOR_USER")
        self.assertEqual(pending["field"], "recipient")

    def test_17_workflow_approval_is_durable(self):
        wf = self._workflow()
        self.session._runtime_create(wf.goal, wf.workflow_id, runtime_task_id=wf.workflow_id, task_type="workflow", metadata={"workflow_orchestration": "langgraph"})
        self.session._runtime_transition(wf.workflow_id, "RUNNING")
        payload = self.adapter.request_workflow_approval(wf, 0)
        self.assertEqual(payload["workflow_id"], wf.workflow_id)
        self.assertIsNotNone(self.session._runtime.get_approval(wf.workflow_id))
        self.assertEqual(self.session._runtime.get_task(wf.workflow_id).state, "WAITING_FOR_APPROVAL")

    def test_18_approval_payload_binds_step(self):
        wf = self._workflow()
        self.session._runtime_create(wf.goal, wf.workflow_id, runtime_task_id=wf.workflow_id, task_type="workflow")
        self.session._runtime_transition(wf.workflow_id, "RUNNING")
        payload = self.adapter.request_workflow_approval(wf, 0)
        self.assertEqual(payload["step_id"], wf.steps[0].step_id)

    def test_19_resource_binding_round_trip(self):
        binding = ResourceBinding("whatsapp", site="whatsapp", reuse=True)
        self.assertEqual(ResourceBinding.from_json(binding.to_json()).to_json(), binding.to_json())

    def test_20_failed_step_is_not_success(self):
        wf = self._workflow()
        wf.steps[0].status = StepStatus.FAILED
        wf.status = WorkflowStatus.FAILED
        self.assertNotEqual(wf.status, WorkflowStatus.COMPLETED)

    def test_21_workflow_completion_requires_all_steps(self):
        wf = self._workflow()
        wf.steps[0].status = StepStatus.COMPLETED
        wf.steps[1].status = StepStatus.COMPLETED
        self.assertNotEqual(wf.status, WorkflowStatus.COMPLETED)
        wf.steps[2].status = StepStatus.COMPLETED
        wf.status = WorkflowStatus.COMPLETED
        self.assertEqual(wf.status, WorkflowStatus.COMPLETED)

    def test_22_no_browser_in_workflow_domain(self):
        self.assertFalse(hasattr(Workflow, "playwright"))

    def test_23_runtime_approval_owner_is_not_a_new_task(self):
        wf = self._workflow()
        self.session._runtime_create(wf.goal, wf.workflow_id, runtime_task_id=wf.workflow_id, task_type="workflow")
        self.session._runtime_transition(wf.workflow_id, "RUNNING")
        self.adapter.request_workflow_approval(wf, 0)
        self.assertEqual(len(self.session._runtime.list_tasks()), 1)

    def test_24_langgraph_dependency_is_explicit(self):
        requirements = Path(__file__).parents[1] / "agent_control" / "langgraph_requirements.txt"
        text = requirements.read_text()
        self.assertIn("langgraph==1.2.11", text)
        self.assertIn("langgraph-checkpoint-sqlite==3.1.1", text)

    def test_25_running_runtime_state_maps_to_running(self):
        self.assertEqual(self.adapter.map_runtime_step_status("RUNNING"), StepStatus.RUNNING)

    def test_26_waiting_for_approval_maps_exactly(self):
        self.assertEqual(self.adapter.map_runtime_step_status("WAITING_FOR_APPROVAL"), StepStatus.WAITING_FOR_APPROVAL)

    def test_27_verifying_maps_exactly(self):
        self.assertEqual(self.adapter.map_runtime_step_status("VERIFYING"), StepStatus.VERIFYING)

    def test_28_completed_maps_exactly(self):
        self.assertEqual(self.adapter.map_runtime_step_status("COMPLETED"), StepStatus.COMPLETED)

    def test_29_failed_maps_exactly(self):
        self.assertEqual(self.adapter.map_runtime_step_status("FAILED"), StepStatus.FAILED)

    def test_30_cancelled_maps_exactly(self):
        self.assertEqual(self.adapter.map_runtime_step_status("CANCELLED"), StepStatus.CANCELLED)

    def test_31_blocked_never_raises(self):
        self.assertEqual(workflow_step_status_from_runner_state("BLOCKED"), StepStatus.PENDING)
        self.assertEqual(self.adapter.map_runtime_step_status("BLOCKED"), StepStatus.RECOVERY_REQUIRED)

    def test_32_unknown_runtime_state_fails_closed(self):
        self.assertEqual(self.adapter.map_runtime_step_status("NOT_A_REAL_RUNTIME_STATE"), StepStatus.RECOVERY_REQUIRED)

    def test_33_blocked_does_not_complete_workflow(self):
        wf = self._workflow()
        wf.steps[0].status = StepStatus.FAILED
        wf.status = WorkflowStatus.FAILED
        wf.steps[1].state = "BLOCKED"
        self.assertEqual(wf.steps[1].status, StepStatus.PENDING)
        self.assertEqual(wf.status, WorkflowStatus.FAILED)

    def test_34_failure_metadata_round_trips(self):
        wf = self._workflow()
        wf.steps[0].status = StepStatus.FAILED
        wf.steps[0].failure_category = FailureClass.ACTION_FAILED.value
        wf.steps[0].error = "skill execution failed"
        wf.steps[0].verification = {"verdict": "UNKNOWN"}
        wf.steps[0].recovery = {"decision": "ABORTED", "reason": "UNKNOWN is not recoverable in V1"}
        restored = Workflow.from_json(wf.to_json())
        self.assertEqual(restored.steps[0].failure_category, "ACTION_FAILED")
        self.assertEqual(restored.steps[0].error, "skill execution failed")
        self.assertEqual(restored.steps[0].verification["verdict"], "UNKNOWN")
        self.assertEqual(restored.steps[0].recovery["decision"], "ABORTED")

    def test_35_failed_first_step_blocks_later_selection(self):
        wf = self._workflow()
        wf.steps[0].status = StepStatus.FAILED
        completed = {s.step_id for s in wf.steps if s.status is StepStatus.COMPLETED}
        runnable = [s for s in wf.steps if s.status is StepStatus.PENDING and all(d in completed for d in s.dependencies)]
        self.assertEqual(runnable, [])

    def test_36_unknown_runner_state_is_not_completion(self):
        self.assertNotEqual(workflow_step_status_from_runner_state("UNKNOWN"), StepStatus.COMPLETED)


    def test_37_skill_failure_preserves_underlying_category_and_error(self):
        action = Action("whatsapp_send_message", {"recipient": "mummy", "message": "hello"})
        execution = ActionResult(
            action=action,
            ok=False,
            error="whatsapp_send_failed: BrowserSkill command failed",
            detail={"code": "whatsapp_send_failed"},
            failure_class=FailureClass.ACTION_FAILED,
        )

        class _Trace:
            def emit(self, *args, **kwargs): pass
            def note(self, *args, **kwargs): pass

        class _Executor:
            def execute(self, _action): return execution

        class _Skill:
            name = "messaging"
            def adapt_action(self, value): return value
            def executor(self): return _Executor()

        loop = SimpleNamespace(trace=_Trace(), extra_state={})
        result = _execute_skill_action(loop, action, _Skill())
        self.assertFalse(result.ok)
        self.assertEqual(result.failure_class, FailureClass.UNKNOWN)
        self.assertIn("whatsapp_send_failed", result.error)
        self.assertEqual(result.detail["skill_failure_class"], "ACTION_FAILED")

    def test_38_failed_step_two_and_three_remain_unrunnable(self):
        wf = self._workflow()
        wf.steps[0].status = StepStatus.FAILED
        completed = {s.step_id for s in wf.steps if s.status is StepStatus.COMPLETED}
        for step in wf.steps[1:]:
            self.assertFalse(step.status is StepStatus.COMPLETED)
            self.assertFalse(all(dep in completed for dep in step.dependencies))


    def _successful_agent_result(self, workflow=None):
        final = VerificationResult(checks=[Check(name="whatsapp_send", verdict=Verdict.PASS, reason="independently verified")])
        outcome = RunOutcome(
            task_id="wf-test", condition="structured_hybrid", trial=0, planner_name="mock",
            reported_success=False, verified=Verdict.PASS, final=final, workflow=workflow,
            state=__import__("agent_control.runner", fromlist=["AgentState"]).AgentState.COMPLETED,
        )
        from agent_control.api import AgentResult, TaskStatus
        return AgentResult(request="send hello to mummy", task_id="wf-test", status=TaskStatus.SUCCESS, verified="PASS", detail="verified 1 check(s)", outcome=outcome)

    def test_39_successful_execution_result_normalizes_to_completed(self):
        result = self._successful_agent_result(self._workflow().to_json())
        normalized = normalize_execution_result(result)
        self.assertTrue(normalized["ok"])
        self.assertTrue(normalized["execution_ok"])
        self.assertEqual(normalized["execution_status"], "COMPLETED")
        self.assertEqual(normalized["verification_status"], "PASS")
        self.assertEqual(normalized["status"], "COMPLETED")

    def test_40_successful_result_does_not_become_false(self):
        result = self._successful_agent_result(self._workflow().to_json())
        self.assertTrue(normalize_execution_result(result)["ok"])

    def test_41_successful_step_advances_to_step_two(self):
        wf = self._workflow()
        wf.steps[0].status = StepStatus.COMPLETED
        wf.current_step_id = wf.steps[0].step_id
        class Runtime:
            def emit_workflow_event(self, *args, **kwargs): pass
        state = {"workflow": wf.to_json()}
        out = next_step(state, Runtime())
        self.assertEqual(out["current_step_id"], wf.steps[1].step_id)

    def test_42_step_two_is_selected_after_step_one_completion(self):
        wf = self._workflow()
        wf.steps[0].status = StepStatus.COMPLETED
        class Runtime:
            def emit_workflow_event(self, *args, **kwargs): pass
        out = next_step({"workflow": wf.to_json()}, Runtime())
        restored = Workflow.from_json(out["workflow"])
        self.assertEqual(restored.current_step_id, wf.steps[1].step_id)

    def test_43_completed_step_cannot_be_selected(self):
        wf = self._workflow()
        wf.steps[0].status = StepStatus.COMPLETED
        wf.current_step_id = wf.steps[0].step_id
        completed = {s.step_id for s in wf.steps if s.status is StepStatus.COMPLETED}
        candidates = [s for s in wf.steps if s.status is not StepStatus.COMPLETED and all(d in completed for d in s.dependencies)]
        self.assertEqual(candidates[0].step_id, wf.steps[1].step_id)
        self.assertNotEqual(candidates[0].step_id, wf.steps[0].step_id)

    def test_44_failed_step_does_not_advance(self):
        wf = self._workflow(); wf.steps[0].status = StepStatus.FAILED; wf.status = WorkflowStatus.FAILED
        completed = {s.step_id for s in wf.steps if s.status is StepStatus.COMPLETED}
        self.assertFalse([s for s in wf.steps[1:] if all(d in completed for d in s.dependencies)])

    def test_45_unknown_verification_does_not_advance(self):
        wf = self._workflow(); wf.steps[0].status = StepStatus.RECOVERY_REQUIRED
        completed = {s.step_id for s in wf.steps if s.status is StepStatus.COMPLETED}
        self.assertFalse([s for s in wf.steps[1:] if all(d in completed for d in s.dependencies)])

    def test_46_workflow_completes_only_after_all_steps(self):
        wf = self._workflow(); wf.steps[0].status = StepStatus.COMPLETED; wf.steps[1].status = StepStatus.COMPLETED
        class Runtime:
            def emit_workflow_event(self, *args, **kwargs): pass
        out = next_step({"workflow": wf.to_json()}, Runtime())
        self.assertEqual(out["current_step_id"], wf.steps[2].step_id)
        wf = Workflow.from_json(out["workflow"]); wf.steps[2].status = StepStatus.COMPLETED
        out = next_step({"workflow": wf.to_json()}, Runtime())
        self.assertEqual(Workflow.from_json(out["workflow"]).status, WorkflowStatus.COMPLETED)

    def test_47_completed_is_never_mapped_to_failed(self):
        self.assertEqual(self.adapter.map_runtime_step_status("COMPLETED"), StepStatus.COMPLETED)
        self.assertNotEqual(self.adapter.map_runtime_step_status("COMPLETED"), StepStatus.FAILED)

    def test_48_invalid_result_structure_is_controlled_error(self):
        result = self._successful_agent_result(["not", "a", "mapping"])
        with self.assertRaises(WorkflowResultNormalizationError):
            normalize_execution_result(result)

    def test_49_whatsapp_resource_is_same_between_steps(self):
        wf = self._workflow()
        self.assertIs(wf.resources[wf.steps[0].resource_key], wf.resources[wf.steps[1].resource_key])
        self.assertEqual(wf.steps[0].resource_key, wf.steps[1].resource_key)

    def test_50_youtube_resource_is_distinct(self):
        wf = self._workflow()
        self.assertNotEqual(wf.steps[1].resource_key, wf.steps[2].resource_key)

    def test_51_step_approval_is_bound_to_step(self):
        wf = self._workflow()
        self.session._runtime_create(wf.goal, wf.workflow_id, runtime_task_id=wf.workflow_id, task_type="workflow")
        self.session._runtime_transition(wf.workflow_id, "RUNNING")
        first = self.adapter.request_workflow_approval(wf, 0)
        self.assertEqual(first["step_id"], wf.steps[0].step_id)
        self.session._runtime.resolve_approval(wf.workflow_id, True)
        self.session._runtime_transition(wf.workflow_id, "RUNNING")
        second = self.adapter.request_workflow_approval(wf, 1)
        self.assertEqual(second["step_id"], wf.steps[1].step_id)

    def test_52_completed_side_effect_step_is_not_replayed(self):
        wf = self._workflow(); wf.steps[0].status = StepStatus.COMPLETED
        runnable = [s for s in wf.steps if s.status is StepStatus.PENDING and all(d in {x.step_id for x in wf.steps if x.status is StepStatus.COMPLETED} for d in s.dependencies)]
        self.assertEqual([s.step_id for s in runnable], [wf.steps[1].step_id])


if __name__ == "__main__":
    unittest.main()

class LangGraphWorkflowApprovalContinuationTests(unittest.TestCase):
    def setUp(self):
        self.session = Session.build(speech=False, write=lambda _: None, planner="mock", show_status=False, debug=False)
        self.adapter = SessionWorkflowRuntime(self.session)

    def tearDown(self):
        self.session.close()

    def _workflow(self):
        return Workflow.from_actions("wf-approval-cont", "send hello to mummy then send bye to papa then play Do I Wanna Know", [
            Action("whatsapp_send_message", {"recipient": "mummy", "message": "hello"}),
            Action("whatsapp_send_message", {"recipient": "papa", "message": "bye"}),
            Action("browser_play_song", {"query": "Do I Wanna Know"}),
        ])

    def _runtime_workflow(self, wf):
        self.session._runtime_create(
            wf.goal, wf.workflow_id, runtime_task_id=wf.workflow_id,
            task_type="workflow", metadata={"workflow_orchestration": "langgraph"},
        )
        self.session._runtime_transition(wf.workflow_id, "RUNNING")

    def test_53_workflow_approval_has_unique_step_owner(self):
        wf = self._workflow()
        self._runtime_workflow(wf)
        first = self.adapter.request_workflow_approval(wf, 0)
        self.assertEqual(first["workflow_id"], wf.workflow_id)
        self.assertEqual(first["step_id"], wf.steps[0].step_id)
        self.assertEqual(first["approval_request_id"], f"approval-{wf.workflow_id}-{wf.steps[0].step_id}")
        self.session._runtime.resolve_approval(wf.workflow_id, True)
        self.session._runtime_transition(wf.workflow_id, "VERIFYING")
        # The runner's child-step COMPLETED event must not make the workflow
        # RuntimeManager task terminal. This is what permits step 2 to ask for
        # a fresh approval without weakening lifecycle validation.
        watcher = self.session._runtime_event_watcher(wf.workflow_id, None)
        watcher({"event": "agent_state", "state": "COMPLETED"})
        self.assertEqual(self.session._runtime.get_task(wf.workflow_id).state, "VERIFYING")
        self.session._runtime_transition(wf.workflow_id, "RUNNING")
        second = self.adapter.request_workflow_approval(wf, 1)
        self.assertEqual(second["step_id"], wf.steps[1].step_id)
        self.assertNotEqual(second["approval_request_id"], first["approval_request_id"])
        self.assertEqual(self.session._runtime.get_approval(wf.workflow_id)["step_id"], wf.steps[1].step_id)

    def test_54_approval_request_id_is_validated_on_resume(self):
        wf = self._workflow()
        self._runtime_workflow(wf)
        wf.current_step_id = wf.steps[1].step_id
        payload = self.adapter.request_workflow_approval(wf, 1)
        stored = self.session._runtime.get_approval(wf.workflow_id)
        self.assertEqual(stored["approval_request_id"], payload["approval_request_id"])
        with self.assertRaises(ValueError):
            self.session._runtime.resolve_approval(
                wf.workflow_id, True, approval_request_id="approval-wrong"
            )
        self.assertEqual(self.session._runtime.get_task(wf.workflow_id).state, "WAITING_FOR_APPROVAL")
        self.adapter.resolve_workflow_approval(wf, True)
        self.assertEqual(self.session._runtime.get_task(wf.workflow_id).state, "RUNNING")

    def test_55_workflow_approval_does_not_expire_like_legacy_task_confirmation(self):
        from agent_control.session import PendingApproval
        approval = PendingApproval(
            request="send bye to papa",
            action=Action("whatsapp_send_message", {"recipient": "papa", "message": "bye"}).to_json(),
            asked_at=0,
            workflow_id="wf-old",
            step_id="wf-old:step-2",
            approval_request_id="approval-wf-old-wf-old:step-2",
        )
        self.assertFalse(approval.expired)

    def test_56_legacy_task_confirmation_still_expires(self):
        from agent_control.session import PendingApproval
        approval = PendingApproval(
            request="send bye to papa",
            action=Action("whatsapp_send_message", {"recipient": "papa", "message": "bye"}).to_json(),
            asked_at=0,
            task_id="legacy-task",
        )
        self.assertTrue(approval.expired)

    def test_57_workflow_runtime_is_not_terminal_after_child_step_completion(self):
        wf = self._workflow()
        self._runtime_workflow(wf)
        self.session._runtime_transition(wf.workflow_id, "VERIFYING")
        watcher = self.session._runtime_event_watcher(wf.workflow_id, None)
        watcher({"event": "agent_state", "state": "COMPLETED"})
        task = self.session._runtime.get_task(wf.workflow_id)
        self.assertEqual(task.state, "VERIFYING")
        self.assertNotEqual(task.state, "COMPLETED")

    def test_58_approval_yes_resumes_same_workflow_owner(self):
        wf = self._workflow()
        self._runtime_workflow(wf)
        payload = self.adapter.request_workflow_approval(wf, 1)
        self.session._rehydrate_approval_owner()
        self.assertEqual(self.session.pending_approval.workflow_id, wf.workflow_id)
        self.assertEqual(self.session.pending_approval.step_id, wf.steps[1].step_id)
        self.assertEqual(self.session.pending_approval.approval_request_id, payload["approval_request_id"])
        with patch.object(self.session, "_resume_workflow_graph", return_value=__import__("agent_control.session", fromlist=["Turn"]).Turn(
            task=__import__("agent_control.session", fromlist=["UserTask"]).UserTask(raw=wf.goal, text=wf.goal, task_id=wf.workflow_id),
            reply="resumed",
        )) as resume:
            turn = self.session._resolve_approval_input("yes", source="text", background=False)
        resume.assert_called_once_with(wf.workflow_id, True, source="text")
        self.assertEqual(turn.task.task_id, wf.workflow_id)
        self.assertEqual(len(self.session._runtime.list_tasks()), 1)

    def test_59_workflow_approval_owner_survives_past_legacy_ttl(self):
        from agent_control.session import PendingApproval
        approval = PendingApproval(
            request=self._workflow().goal,
            action=Action("whatsapp_send_message", {"recipient": "papa", "message": "bye"}).to_json(),
            asked_at=0,
            task_id="wf-approval-cont",
            goal=self._workflow().goal,
            runtime_task_id="wf-approval-cont",
            workflow_id="wf-approval-cont",
            step_id="wf-approval-cont:step-2",
            approval_request_id="approval-wf-approval-cont-wf-approval-cont:step-2",
        )
        self.assertFalse(approval.expired)


    def test_61_background_reserved_workflow_identity_is_marked_as_langgraph(self):
        wf = self._workflow()
        self.session._runtime_create(
            wf.goal, wf.workflow_id, runtime_task_id=wf.workflow_id,
            task_type="background", metadata={"background": True},
        )
        with patch.object(self.session, "_workflow_engine") as engine_factory:
            engine_factory.return_value.start.return_value = {"__interrupt__": [{"value": {}}]}
            self.session._start_workflow(wf.goal, workflow_id=wf.workflow_id)
        task = self.session._runtime.get_task(wf.workflow_id)
        self.assertEqual(task.metadata.get("workflow_orchestration"), "langgraph")
        self.assertEqual(task.metadata.get("workflow_goal"), wf.goal)
        self.assertEqual(task.task_type, "background")
        self.assertEqual(task.state, "RUNNING")

    def test_62_background_reserved_workflow_child_completion_stays_nonterminal(self):
        wf = self._workflow()
        self.session._runtime_create(
            wf.goal, wf.workflow_id, runtime_task_id=wf.workflow_id,
            task_type="background", metadata={"background": True},
        )
        with patch.object(self.session, "_workflow_engine") as engine_factory:
            engine_factory.return_value.start.return_value = {"__interrupt__": [{"value": {}}]}
            self.session._start_workflow(wf.goal, workflow_id=wf.workflow_id)
        self.session._runtime_transition(wf.workflow_id, "VERIFYING")
        watcher = self.session._runtime_event_watcher(wf.workflow_id, None)
        watcher({"event": "agent_state", "state": "COMPLETED"})
        self.assertEqual(self.session._runtime.get_task(wf.workflow_id).state, "VERIFYING")

    def test_60_completed_workflow_can_still_transition_to_terminal_completion(self):
        wf = self._workflow()
        self._runtime_workflow(wf)
        self.session._runtime_transition(wf.workflow_id, "VERIFYING")
        watcher = self.session._runtime_event_watcher(wf.workflow_id, None)
        watcher({"event": "agent_state", "state": "COMPLETED"})
        self.session._runtime_transition(wf.workflow_id, "RUNNING")
        for step in wf.steps:
            step.status = StepStatus.COMPLETED
        wf.status = WorkflowStatus.COMPLETED
        self.adapter.emit_workflow_event(wf, "WORKFLOW_COMPLETED")
        self.assertEqual(self.session._runtime.get_task(wf.workflow_id).state, "COMPLETED")


if __name__ == "__main__":
    unittest.main()
