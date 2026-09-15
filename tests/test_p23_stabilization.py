from pathlib import Path
import unittest
from unittest.mock import patch

from agent_control.api import AgentResult, TaskStatus
from agent_control.planner.openai_compat import OpenAICompatPlanner
from agent_control.runtime import RuntimeManager
from agent_control.session import Session
from agent_control.types import Clarification, Verdict


class P23StabilizationTests(unittest.TestCase):
    def setUp(self):
        self.session = Session.build(
            speech=False,
            write=lambda _line: None,
            planner="llm",
            show_status=False,
            debug=True,
        )

    def tearDown(self):
        self.session.close()

    def test_runtime_watcher_commits_acting_as_running_before_verifying(self):
        runtime = RuntimeManager()
        task_id = runtime.create_task("send hello to papa", task_id="fast-3452").task_id
        # The watcher is the production Session -> RuntimeManager synchronization
        # boundary, not a seeded RuntimeManager transition helper.
        self.session._runtime = runtime
        watcher = self.session._runtime_event_watcher(task_id, None)
        for state in ("PLANNING", "ACTING", "VERIFYING", "COMPLETED"):
            watcher({"event": "agent_state", "state": state})
        task = runtime.get_task(task_id)
        self.assertEqual(task.state, "COMPLETED")
        rejected = [e for e in runtime.events(task_id) if e.event_type == "RUNTIME_TRANSITION_REJECTED"]
        self.assertEqual(rejected, [])
        states = [
            e.metadata.get("to")
            for e in runtime.events(task_id)
            if e.event_type in {"TASK_PLANNING", "TASK_ACTING", "VERIFICATION_STARTED", "TASK_COMPLETED"}
        ]
        self.assertEqual(states, ["PLANNING", "RUNNING", "VERIFYING", "COMPLETED"])

    def test_compound_planner_decomposes_without_llm(self):
        planner = OpenAICompatPlanner(client=type("Client", (), {"model": "test"})())
        goal = "send hello to mummy then send bye to papa then play Do I Wanna Know"
        step = planner.plan(goal, {}, [])
        self.assertFalse(step.error)
        self.assertFalse(step.done)
        self.assertEqual(
            [(a.kind, a.params) for a in step.actions],
            [
                ("whatsapp_send_message", {"recipient": "mummy", "message": "hello"}),
                ("whatsapp_send_message", {"recipient": "papa", "message": "bye"}),
                ("browser_play_song", {"query": "Do I Wanna Know"}),
            ],
        )

    def test_session_submit_then_yes_reuses_same_runtime_task(self):
        calls = []

        def fake_run_agent_task(request, **kwargs):
            calls.append(kwargs.get("task_id"))
            on_event = kwargs.get("on_event")
            if len(calls) == 1:
                return AgentResult(
                    request=request,
                    task_id=kwargs["task_id"],
                    status=TaskStatus.NEEDS_INPUT,
                    verified=Verdict.UNKNOWN.value,
                    question=Clarification(
                        question="Approve whatsapp send?",
                        context='APPROVAL_ACTION:{"action":{"kind":"whatsapp_send_message","params":{"recipient":"papa","message":"hello"}}}',
                    ),
                )
            for state in ("PLANNING", "ACTING", "VERIFYING", "COMPLETED"):
                if on_event:
                    on_event({"event": "agent_state", "state": state})
            return AgentResult(
                request=request,
                task_id=kwargs["task_id"],
                status=TaskStatus.SUCCESS,
                verified=Verdict.PASS.value,
                detail="whatsapp message verified",
            )

        with patch("agent_control.session.api.resolve_task", return_value=None), \
             patch("agent_control.session.api.run_agent_task", side_effect=fake_run_agent_task):
            first = self.session.submit("send hello to papa")
            self.assertTrue(first.result.needs_input)
            runtime_tasks = self.session.runtime_snapshot()["waiting_tasks"]
            self.assertEqual(len(runtime_tasks), 1)
            task_id = runtime_tasks[0]["task_id"]
            self.assertEqual(runtime_tasks[0]["state"], "WAITING_FOR_APPROVAL")

            second = self.session.submit("yes")
            self.assertTrue(second.result.ok)

        snapshot = self.session.runtime_snapshot()
        completed = snapshot["completed_tasks"]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["task_id"], task_id)
        self.assertEqual(completed[0]["state"], "COMPLETED")
        self.assertEqual(calls, [task_id, task_id])
        self.assertEqual(snapshot["pending_approvals"], [])
        rejected = [e for e in self.session._runtime.events(task_id) if e.event_type == "RUNTIME_TRANSITION_REJECTED"]
        self.assertEqual(rejected, [])

    def test_zero_action_planner_is_not_success(self):
        class EmptyPlanner:
            name = "empty"
            from agent_control.planner.base import Usage
            usage = Usage()
            def plan(self, goal, state, history):
                from agent_control.planner.base import PlannerStep
                return PlannerStep(actions=[], done=False, reasoning="no executable action")

        from agent_control.general_task import GeneralTask
        from agent_control.runner import RunConfig, run_task

        class MinimalPolicy:
            workspace = Path("/tmp")
            refuse_if_elevated = False

        task = GeneralTask(request="do something")
        outcome = run_task(task, EmptyPlanner(), MinimalPolicy(), RunConfig(interactive=True, max_steps=1))
        self.assertEqual(outcome.state.value, "FAILED")
        self.assertFalse(outcome.reported_success)
        self.assertNotEqual(outcome.verified, Verdict.PASS)


    def test_workflow_preserves_original_goal_separately(self):
        from agent_control.workflow import Workflow
        from agent_control.types import Action
        goal = "send hello to mummy then play Do I Wanna Know"
        workflow = Workflow.from_actions("wf-1", goal, [
            Action("whatsapp_send_message", {"recipient": "mummy", "message": "hello"}),
            Action("browser_play_song", {"query": "Do I Wanna Know"}),
        ])
        data = workflow.to_json()
        self.assertEqual(data["goal"], goal)
        self.assertEqual(data["steps"][0]["arguments"], {"recipient": "mummy", "message": "hello"})
        self.assertEqual(data["steps"][1]["arguments"], {"query": "Do I Wanna Know"})
        self.assertNotIn(goal, data["steps"][0]["arguments"].values())

    def test_missing_workflow_input_stays_waiting_for_user(self):
        from agent_control.workflow import Workflow, WorkflowStepTask
        from agent_control.planner.mock import MockPlanner
        from agent_control.runner import RunConfig, run_task

        workflow = Workflow.from_actions(
            "wf-missing",
            "send something to papa",
            [__import__("agent_control.types", fromlist=["Action"]).Action(
                "whatsapp_send_message", {"recipient": "papa"}
            )],
        )
        task = WorkflowStepTask(workflow=workflow, step=workflow.steps[0])

        class MinimalPolicy:
            workspace = Path("/tmp")
            refuse_if_elevated = False

        outcome = run_task(
            task,
            MockPlanner(reference_plan=[workflow.steps[0].action]),
            MinimalPolicy(),
            RunConfig(interactive=True, max_steps=1),
        )
        self.assertEqual(outcome.state.value, "WAITING_FOR_USER")
        self.assertTrue(outcome.awaiting)
        self.assertIsNotNone(outcome.question)

    def test_whatsapp_step_is_policy_gated_before_execution(self):
        from agent_control.workflow import Workflow, WorkflowStepTask
        from agent_control.planner.mock import MockPlanner
        from agent_control.runner import RunConfig, run_task
        from agent_control.policy import Decision
        from agent_control.types import Action

        workflow = Workflow.from_actions(
            "wf-approval",
            "send hello to papa",
            [Action("whatsapp_send_message", {"recipient": "papa", "message": "hello"})],
        )
        task = WorkflowStepTask(workflow=workflow, step=workflow.steps[0])

        class ConfirmPolicy:
            workspace = Path("/tmp")
            refuse_if_elevated = False
            def check(self, action):
                return Decision.CONFIRM, "external message requires approval"

        outcome = run_task(
            task,
            MockPlanner(reference_plan=[workflow.steps[0].action]),
            ConfirmPolicy(),
            RunConfig(interactive=True, max_steps=1),
        )
        self.assertEqual(outcome.state.value, "WAITING_FOR_USER")
        self.assertTrue(outcome.awaiting)
        self.assertTrue(outcome.question.context.startswith("APPROVAL_ACTION:"))

    def test_failed_step_preserves_workflow_and_goal(self):
        from agent_control.workflow import Workflow, WorkflowStepTask
        from agent_control.planner.mock import MockPlanner
        from agent_control.runner import RunConfig, run_task
        from agent_control.runtime import RuntimeManager
        from agent_control.policy import Decision
        from agent_control.types import Action

        goal = "send hello to papa"
        workflow = Workflow.from_actions(
            "wf-failed",
            goal,
            [Action("whatsapp_send_message", {"recipient": "papa", "message": "hello"})],
        )
        task = WorkflowStepTask(workflow=workflow, step=workflow.steps[0])

        class DenyPolicy:
            workspace = Path("/tmp")
            refuse_if_elevated = False
            def check(self, action):
                return Decision.DENY, "test denial"

        runtime = RuntimeManager()
        runtime.create_task(goal, task_id="wf-failed")
        outcome = run_task(
            task,
            MockPlanner(reference_plan=[workflow.steps[0].action]),
            DenyPolicy(),
            RunConfig(interactive=False, max_steps=1, runtime_manager=runtime, runtime_task_id="wf-failed"),
        )
        saved = runtime.get_task("wf-failed")
        self.assertEqual(outcome.state.value, "FAILED")
        self.assertEqual(saved.metadata["workflow"]["goal"], goal)
        self.assertEqual(saved.metadata["workflow"]["steps"][0]["state"], "FAILED")

    def test_compound_steps_are_individually_policy_checked(self):
        from agent_control.workflow import Workflow, WorkflowStepTask
        from agent_control.planner.mock import MockPlanner
        from agent_control.runner import RunConfig, run_task
        from agent_control.policy import Decision
        from agent_control.types import Action

        workflow = Workflow.from_actions(
            "wf-two",
            "send hello to mummy then send bye to papa",
            [
                Action("whatsapp_send_message", {"recipient": "mummy", "message": "hello"}),
                Action("whatsapp_send_message", {"recipient": "papa", "message": "bye"}),
            ],
        )
        task = WorkflowStepTask(workflow=workflow, step=workflow.steps[0])
        checks = []

        class ConfirmPolicy:
            workspace = Path("/tmp")
            refuse_if_elevated = False
            def check(self, action):
                checks.append(action)
                return Decision.CONFIRM, "approval required"

        outcome = run_task(
            task,
            MockPlanner(reference_plan=[workflow.steps[0].action]),
            ConfirmPolicy(),
            RunConfig(interactive=True, max_steps=1),
        )
        self.assertEqual(outcome.state.value, "WAITING_FOR_USER")
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0].params, {"recipient": "mummy", "message": "hello"})


if __name__ == "__main__":
    unittest.main()
