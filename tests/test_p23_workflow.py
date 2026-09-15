from __future__ import annotations
import tempfile
from pathlib import Path
from agent_control.runtime import RuntimeManager
from agent_control.workflow import Workflow
from agent_control.types import Action


def test_workflow_is_ordered_and_structured():
    wf = Workflow.from_actions("wf-1", "send hello to mummy then send bye to papa then play Do I Wanna Know", [
        Action("whatsapp_send_message", {"recipient":"mummy", "message":"hello"}),
        Action("whatsapp_send_message", {"recipient":"papa", "message":"bye"}),
        Action("browser_play_song", {"query":"Do I Wanna Know"}),
    ])
    assert wf.steps[0].action.params["recipient"] == "mummy"
    assert wf.steps[0].action.params["message"] == "hello"
    assert wf.steps[1].action.params["recipient"] == "papa"
    assert wf.steps[2].action.params["query"] == "Do I Wanna Know"
    assert wf.goal != wf.steps[0].action.params["recipient"]


def test_workflow_persists_in_authoritative_runtime():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "runtime.sqlite3"
        r1 = RuntimeManager(persistence_path=db)
        r1.create_task("compound goal", task_id="wf-1", task_type="workflow")
        wf = Workflow.from_actions("wf-1", "compound goal", [Action("whatsapp_send_message", {"recipient":"mummy", "message":"hello"}), Action("browser_play_song", {"query":"song"})])
        wf.steps[0].state = "COMPLETED"
        wf.steps[1].state = "WAITING_FOR_APPROVAL"
        wf.current_step = 1
        wf.state = "WAITING_FOR_APPROVAL"
        r1.update_workflow("wf-1", wf.to_json())
        r1.close()
        r2 = RuntimeManager(persistence_path=db)
        got = r2.get_task("wf-1")
        assert got is not None
        persisted = got.metadata["workflow"]
        assert persisted["steps"][0]["state"] == "COMPLETED"
        assert persisted["steps"][1]["state"] == "WAITING_FOR_APPROVAL"
        assert persisted["current_step"] == 1
        r2.close()

from agent_control.general_task import GeneralTask
from agent_control.planner.mock import MockPlanner
from agent_control.policy import Policy
from agent_control.runner import RunConfig, run_task
from agent_control.skills.registry import SkillRegistry
from agent_control.skills.base import Skill, SkillAction, SkillInfo
from agent_control.skills.manifest import SkillManifest
from agent_control.skills.security import Capability
from agent_control.types import ActionResult, FailureClass, VerificationResult, Check, Verdict


class _FakeExecutor:
    def __init__(self, calls, fail_first=False): self.calls, self.fail_first = calls, fail_first
    def execute(self, action):
        self.calls.append(action.params.copy())
        if self.fail_first and len(self.calls) == 1:
            return ActionResult(action=Action(action.kind, dict(action.params)), ok=False, error="boom", failure_class=FailureClass.ACTION_FAILED)
        return ActionResult(action=Action(action.kind, dict(action.params)), ok=True)

class _FakeVerifier:
    def verify(self, action, result):
        return type("V", (), {"ok": True, "status": "PASS", "detail": "fresh verification"})()

class _FakeSkill(Skill):
    def __init__(self, calls, fail_first=False):
        self.calls = calls; self.ex = _FakeExecutor(calls, fail_first)
    @property
    def info(self):
        return SkillInfo("fake", "fake", (SkillAction("fake_action", "fake"),), SkillManifest(capabilities=frozenset({Capability.BROWSER})))
    def executor(self): return self.ex
    def verifier(self): return _FakeVerifier()
    def adapt_action(self, action): return action


def _fake_task(actions):
    t = GeneralTask("compound goal")
    t.reference_plan = lambda policy: actions
    return t


def test_verified_step_unlocks_next_step():
    calls=[]
    actions=[Action("fake_action", {"target":"one"}), Action("fake_action", {"target":"two"})]
    reg=SkillRegistry(); reg.register(_FakeSkill(calls))
    with tempfile.TemporaryDirectory() as d:
        r=RuntimeManager(persistence_path=Path(d)/"r.sqlite3")
        task=_fake_task(actions); task.task_id="wf-2"
        policy=Policy(workspace=Path(d), confirm_mode="allow", refuse_if_elevated=False)
        out=run_task(task, MockPlanner(reference_plan=actions), policy, RunConfig(max_steps=3, runtime_manager=r, runtime_task_id="wf-2"), skills=reg)
        assert calls == [{"target":"one"},{"target":"two"}]
        assert out.workflow is not None
        assert [x["state"] for x in out.workflow["steps"]] == ["COMPLETED","COMPLETED"]
        r.close()


def test_failed_step_blocks_later_step():
    calls=[]
    actions=[Action("fake_action", {"target":"one"}), Action("fake_action", {"target":"two"})]
    reg=SkillRegistry(); reg.register(_FakeSkill(calls, fail_first=True))
    with tempfile.TemporaryDirectory() as d:
        r=RuntimeManager(persistence_path=Path(d)/"r.sqlite3")
        task=_fake_task(actions); task.task_id="wf-3"
        policy=Policy(workspace=Path(d), confirm_mode="allow", refuse_if_elevated=False)
        out=run_task(task, MockPlanner(reference_plan=actions), policy, RunConfig(max_steps=3, runtime_manager=r, runtime_task_id="wf-3"), skills=reg)
        assert calls == [{"target":"one"}]
        assert out.workflow["steps"][0]["state"] in {"FAILED","UNKNOWN"}
        # BLOCKED is runner scheduling vocabulary; the workflow model keeps an
        # unexecuted dependent step PENDING and records failure on the workflow.
        assert out.workflow["steps"][1]["state"] == "PENDING"
        r.close()


def test_compound_planner_output_cannot_use_goal_as_whatsapp_target():
    from agent_control.planner.openai_compat import OpenAICompatPlanner
    class C:
        model="fake"
        usage=type("U", (), {"add":lambda *a, **k:None, "to_json":lambda self:{}})()
        def chat_checked(self, messages, json_mode=False):
            return ('{"reasoning":"decomposed","done":false,"actions":[{"kind":"whatsapp_send_message","params":{"recipient":"send hello to mummy then send bye to papa then play song","message":"hello"}}]}', {}, None)
    planner=OpenAICompatPlanner(C())
    step=planner.plan("send hello to mummy then send bye to papa then play song", {}, [])
    assert step.rejected == []
    assert [a.params.get("recipient") for a in step.actions[:2]] == ["mummy", "papa"]
    assert step.actions[2].kind == "browser_play_song"

def test_approval_payload_carries_exact_workflow_step():
    from agent_control.runner import _Loop, _permit_and_execute
    from agent_control.recovery import Recovery, RecoveryBudget
    from agent_control.trace import Trace
    from agent_control.types import DecisionKind
    class ConfirmPolicy:
        def check(self, action):
            from agent_control.policy import Decision
            return Decision.CONFIRM, "confirm"
    with tempfile.TemporaryDirectory() as d:
        r=RuntimeManager(persistence_path=Path(d)/"r.sqlite3")
        r.create_task("compound", task_id="wf-approval")
        wf=Workflow.from_actions("wf-approval", "compound", [Action("whatsapp_send_message", {"recipient":"mummy","message":"hello"}), Action("whatsapp_send_message", {"recipient":"papa","message":"bye"})])
        loop=_Loop(task=GeneralTask("compound"), policy=ConfirmPolicy(), config=__import__('agent_control.runner', fromlist=['RunConfig']).RunConfig(interactive=True, runtime_manager=r, runtime_task_id="wf-approval"), trace=Trace("wf-approval","test",0), recovery=Recovery(RecoveryBudget(), True, True), workflow=wf, workflow_step_index=1)
        try:
            _permit_and_execute(loop, wf.steps[1].action)
            assert False
        except Exception as exc:
            assert exc.clarification.context.startswith("APPROVAL_ACTION:")
            import json
            payload=json.loads(exc.clarification.context.split(":",1)[1])
            assert payload["workflow_id"] == "wf-approval"
            assert payload["step_id"] == "wf-approval:step-2"
            assert payload["action"]["params"]["recipient"] == "papa"
        r.close()
