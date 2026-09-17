from dataclasses import dataclass
from pathlib import Path

import pytest

from agent_control.policy import Policy, Decision
from agent_control.procedure_execution import (
    ProcedureExecutionStatus,
    execute_procedure,
)
from agent_control.procedure_retrieval import (
    BindingResult,
    BindingStatus,
    ParameterBinding,
    ProcedureMatch,
    ProcedureValidationResult,
    ValidationStatus,
)
from agent_control.procedure_store import (
    LearnedProcedure,
    ProcedureCondition,
    ProcedureParameter,
    ProcedureProvenance,
    ProcedureSourceType,
    ProcedureStatus,
    ProcedureStep,
)
from agent_control.skills.base import Skill, SkillAction, SkillInfo, SkillVerifier
from agent_control.skills.manifest import SkillManifest
from agent_control.types import Action, FailureClass, Verdict


class FakeBackend:
    def __init__(self, state="before"):
        self.state = state
        self.observe_count = 0
        self.observed_after_execute = False

    def observe(self):
        self.observe_count += 1
        return {"state": self.state}


class FakeExecution:
    def __init__(self, ok=True):
        self.ok = ok
        self.error = None if ok else "boom"
        self.failure_class = None if ok else FailureClass.ACTION_FAILED


class FakeVerifier(SkillVerifier):
    def __init__(self, backend):
        self.backend = backend
        self.calls = 0

    def verify(self, action, result):
        self.calls += 1
        if self.backend.observe_count < 1:
            raise AssertionError("verification was attempted without a fresh observation")
        if self.backend.state == "pass":
            return type("VR", (), {"ok": True, "status": "PASS", "detail": "observed"})()
        if self.backend.state == "unknown":
            return type("VR", (), {"ok": False, "status": "UNKNOWN", "detail": "insufficient evidence"})()
        return type("VR", (), {"ok": False, "status": "FAIL", "detail": "postcondition absent"})()


class FakeSkill(Skill):
    def __init__(self, backend, *, execution_ok=True):
        self.backend = backend
        self.execution_ok = execution_ok
        self.execute_calls = 0
        self._verifier = FakeVerifier(backend)
        self._info = SkillInfo(
            name="fake",
            description="test skill",
            actions=(SkillAction("browser.play.song", "play song"),),
            manifest=SkillManifest(),
        )

    @property
    def info(self):
        return self._info

    def supports(self, kind):
        return kind == "browser.play.song"

    def observe(self):
        return self.backend.observe()

    def adapt_action(self, action):
        return action

    def executor(self):
        skill = self

        class Executor:
            def execute(self, action):
                skill.execute_calls += 1
                return FakeExecution(skill.execution_ok)

        return Executor()

    def verifier(self):
        return self._verifier


class FakeRegistry:
    def __init__(self, skill):
        self.skill = skill

    def find_for_action(self, kind):
        return self.skill if self.skill.supports(kind) else None


@dataclass
class FakeTrace:
    events: list[tuple[str, dict]]

    def emit(self, name, **data):
        self.events.append((name, data))

    def note(self, name, **data):
        self.events.append((name, data))


def make_procedure(*, status=ProcedureStatus.ACTIVE, parameter=True, preconditions=()):
    params = (
        ProcedureParameter("song", type="string", required=True),
    ) if parameter else ()
    args = {"query": "{song}"} if parameter else {"query": "Do I Wanna Know"}
    p = LearnedProcedure(
        procedure_id="proc-test",
        name="browser.play.song",
        description="play a song",
        trigger_pattern="play {song}",
        status=status,
        version=1,
        parameters=params,
        steps=(ProcedureStep("step-1", 0, "browser.play.song", args, required_capabilities=("browser.play.song",)),),
        preconditions=tuple(preconditions),
        required_capabilities=("browser.play.song",),
        provenance=ProcedureProvenance(ProcedureSourceType.MANUAL),
    )
    p.validate()
    return p


def make_match(procedure, *, binding_status=BindingStatus.BOUND, validation=ValidationStatus.APPLICABLE, execution_allowed=True, value="do i wanna know"):
    parameter = procedure.parameters[0] if procedure.parameters else ProcedureParameter("song")
    binding = BindingResult(
        binding_status,
        (ParameterBinding(parameter, value, binding_status),),
        () if binding_status is BindingStatus.BOUND else ("binding_not_ready",),
    )
    validation_result = ProcedureValidationResult(validation, () if validation is ValidationStatus.APPLICABLE else ("not_applicable",))
    return ProcedureMatch(procedure, binding, validation_result, execution_allowed=execution_allowed)


def policy(tmp_path):
    return Policy(tmp_path / "workspace", refuse_if_elevated=False)


def test_fully_bound_applicable_active_procedure_executes(tmp_path):
    backend = FakeBackend("pass")
    skill = FakeSkill(backend)
    result = execute_procedure(
        make_match(make_procedure()),
        policy=policy(tmp_path),
        skills=FakeRegistry(skill),
    )
    assert result.execution_status is ProcedureExecutionStatus.EXECUTED
    assert result.verification_status is Verdict.PASS
    assert result.success
    assert skill.execute_calls == 1
    assert backend.observe_count == 1


def test_missing_parameter_not_executed(tmp_path):
    p = make_procedure()
    result = execute_procedure(make_match(p, binding_status=BindingStatus.MISSING), policy=policy(tmp_path), skills=FakeRegistry(FakeSkill(FakeBackend())))
    assert result.execution_status is ProcedureExecutionStatus.NOT_EXECUTED
    assert result.failure_reason.startswith("parameter_binding_missing")


def test_invalid_parameter_not_executed(tmp_path):
    p = make_procedure()
    result = execute_procedure(make_match(p, binding_status=BindingStatus.INVALID), policy=policy(tmp_path), skills=FakeRegistry(FakeSkill(FakeBackend())))
    assert result.execution_status is ProcedureExecutionStatus.NOT_EXECUTED
    assert result.failure_reason.startswith("parameter_binding_invalid")


def test_ambiguous_binding_not_executed(tmp_path):
    p = make_procedure()
    result = execute_procedure(make_match(p, binding_status=BindingStatus.AMBIGUOUS), policy=policy(tmp_path), skills=FakeRegistry(FakeSkill(FakeBackend())))
    assert result.execution_status is ProcedureExecutionStatus.NOT_EXECUTED


def test_not_applicable_not_executed(tmp_path):
    p = make_procedure()
    result = execute_procedure(make_match(p, validation=ValidationStatus.NOT_APPLICABLE), policy=policy(tmp_path), skills=FakeRegistry(FakeSkill(FakeBackend())))
    assert result.execution_status is ProcedureExecutionStatus.NOT_EXECUTED


def test_unknown_state_does_not_execute(tmp_path):
    p = make_procedure()
    result = execute_procedure(make_match(p, validation=ValidationStatus.UNKNOWN), policy=policy(tmp_path), skills=FakeRegistry(FakeSkill(FakeBackend())))
    assert result.execution_status is ProcedureExecutionStatus.NOT_EXECUTED
    assert result.failure_reason == "current_state_unknown"


def test_candidate_is_not_executable(tmp_path):
    p = make_procedure(status=ProcedureStatus.CANDIDATE)
    skill = FakeSkill(FakeBackend("pass"))
    result = execute_procedure(make_match(p), policy=policy(tmp_path), skills=FakeRegistry(skill))
    assert result.execution_status is ProcedureExecutionStatus.NOT_EXECUTED
    assert skill.execute_calls == 0
    assert p.status is ProcedureStatus.CANDIDATE


def test_execution_allowed_gate_is_required(tmp_path):
    p = make_procedure()
    skill = FakeSkill(FakeBackend("pass"))
    result = execute_procedure(make_match(p, execution_allowed=False), policy=policy(tmp_path), skills=FakeRegistry(skill))
    assert result.execution_status is ProcedureExecutionStatus.NOT_EXECUTED
    assert skill.execute_calls == 0


def test_bound_parameter_reaches_semantic_action(tmp_path):
    backend = FakeBackend("pass")
    skill = FakeSkill(backend)
    seen = {}
    original = skill.adapt_action
    def adapt(action):
        seen.update(action.params)
        return original(action)
    skill.adapt_action = adapt
    result = execute_procedure(make_match(make_procedure()), policy=policy(tmp_path), skills=FakeRegistry(skill))
    assert result.success
    assert seen["query"] == "do i wanna know"


def test_policy_denial_blocks_execution(tmp_path):
    p = make_procedure()
    skill = FakeSkill(FakeBackend("pass"))
    pol = policy(tmp_path)
    pol.check = lambda action: (Decision.DENY, "denied")
    result = execute_procedure(make_match(p), policy=pol, skills=FakeRegistry(skill))
    assert result.execution_status is ProcedureExecutionStatus.NOT_EXECUTED
    assert result.failure_reason == "policy_denied"
    assert skill.execute_calls == 0


def test_policy_confirmation_blocks_without_approval(tmp_path):
    p = make_procedure()
    skill = FakeSkill(FakeBackend("pass"))
    pol = policy(tmp_path)
    pol.check = lambda action: (Decision.CONFIRM, "confirmation required")
    result = execute_procedure(make_match(p), policy=pol, skills=FakeRegistry(skill))
    assert result.failure_reason == "approval_required"
    assert skill.execute_calls == 0


def test_executor_failure_is_not_success(tmp_path):
    backend = FakeBackend("pass")
    skill = FakeSkill(backend, execution_ok=False)
    result = execute_procedure(make_match(make_procedure()), policy=policy(tmp_path), skills=FakeRegistry(skill))
    assert result.execution_status is ProcedureExecutionStatus.FAILED
    assert result.verification_status is Verdict.UNKNOWN
    assert not result.success
    assert backend.observe_count == 0


def test_executor_success_alone_is_not_pass(tmp_path):
    backend = FakeBackend("fail")
    skill = FakeSkill(backend)
    result = execute_procedure(make_match(make_procedure()), policy=policy(tmp_path), skills=FakeRegistry(skill))
    assert result.execution_status is ProcedureExecutionStatus.FAILED
    assert result.verification_status is Verdict.FAIL
    assert not result.success


def test_unknown_verification_is_not_success(tmp_path):
    backend = FakeBackend("unknown")
    skill = FakeSkill(backend)
    result = execute_procedure(make_match(make_procedure()), policy=policy(tmp_path), skills=FakeRegistry(skill))
    assert result.execution_status is ProcedureExecutionStatus.EXECUTED
    assert result.verification_status is Verdict.UNKNOWN
    assert not result.success


def test_fresh_observation_happens_after_execution(tmp_path):
    backend = FakeBackend("pass")
    skill = FakeSkill(backend)
    original_execute = skill.executor().execute
    # The verifier itself asserts that at least one post-execution observation exists.
    result = execute_procedure(make_match(make_procedure()), policy=policy(tmp_path), skills=FakeRegistry(skill))
    assert result.fresh_observation == {"state": "pass"}
    assert skill._verifier.calls == 1


def test_fail_verification_is_structured(tmp_path):
    backend = FakeBackend("fail")
    result = execute_procedure(make_match(make_procedure()), policy=policy(tmp_path), skills=FakeRegistry(FakeSkill(backend)))
    assert result.verification_evidence is not None
    assert result.verification_evidence.verdict is Verdict.FAIL
    assert result.failure_reason == "verification_failed"


def test_no_skill_falls_back_without_execution(tmp_path):
    p = make_procedure()
    class Empty:
        def find_for_action(self, kind): return None
    result = execute_procedure(make_match(p), policy=policy(tmp_path), skills=Empty())
    assert result.execution_status is ProcedureExecutionStatus.NOT_EXECUTED
    assert result.failure_reason == "unsupported_action:browser.play.song"


def test_candidate_status_never_changes(tmp_path):
    p = make_procedure(status=ProcedureStatus.CANDIDATE)
    before = p.status
    execute_procedure(make_match(p), policy=policy(tmp_path), skills=FakeRegistry(FakeSkill(FakeBackend("pass"))))
    assert p.status is before


def test_dead_session_failure_is_bounded_and_not_replayed(tmp_path):
    backend = FakeBackend("pass")
    skill = FakeSkill(backend)
    def broken_adapt(action):
        raise RuntimeError("session not registered or already stopped")
    skill.adapt_action = broken_adapt
    result = execute_procedure(make_match(make_procedure()), policy=policy(tmp_path), skills=FakeRegistry(skill))
    assert result.execution_status is ProcedureExecutionStatus.FAILED
    assert skill.execute_calls == 0


def test_malformed_empty_steps_do_not_execute(tmp_path):
    p = LearnedProcedure(
        procedure_id="bad", name="browser.play.song", description="bad", trigger_pattern="play {song}",
        status=ProcedureStatus.ACTIVE, version=1,
        parameters=(ProcedureParameter("song"),), steps=(),
        provenance=ProcedureProvenance(ProcedureSourceType.MANUAL),
    )
    # Do not call validate: this is explicitly testing the execution boundary's
    # defense in depth against malformed persisted/in-memory definitions.
    result = execute_procedure(make_match(p), policy=policy(tmp_path), skills=FakeRegistry(FakeSkill(FakeBackend())))
    assert result.execution_status is ProcedureExecutionStatus.NOT_EXECUTED
    assert result.failure_reason == "procedure_has_no_steps"


def test_unresolved_placeholder_is_not_executed(tmp_path):
    p = make_procedure()
    p = LearnedProcedure(**{**p.__dict__, "steps": (ProcedureStep("step-1", 0, "browser.play.song", {"query": "{other}"}),)})
    result = execute_procedure(make_match(p), policy=policy(tmp_path), skills=FakeRegistry(FakeSkill(FakeBackend())))
    assert result.execution_status is ProcedureExecutionStatus.NOT_EXECUTED
    assert result.failure_reason == "unresolved_parameter:other"


def test_execution_source_contains_no_browser_replay_primitives():
    source = Path(__file__).resolve().parents[1] / "agent_control" / "procedure_execution.py"
    text = source.read_text(encoding="utf-8").casefold()
    for forbidden in ("xpath", "css selector", "dom path", "pixel coordinate", "hwnd"):
        assert forbidden not in text


def test_trace_exposes_execution_and_verification_without_values(tmp_path):
    backend = FakeBackend("pass")
    trace = FakeTrace([])
    result = execute_procedure(make_match(make_procedure()), policy=policy(tmp_path), skills=FakeRegistry(FakeSkill(backend)), trace=trace)
    assert result.success
    names = [name for name, _ in trace.events]
    assert "procedure_execution_observation" in names
    assert "procedure_execution_verification" in names
    serialized = repr(trace.events)
    assert "do i wanna know" not in serialized
