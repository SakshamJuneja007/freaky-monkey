from pathlib import Path

import pytest

from agent_control.procedure_store import (
    LearnedProcedure,
    ProcedureProvenance,
    ProcedureParameter,
    ProcedureSourceType,
    ProcedureStatus,
    ProcedureStep,
    ProcedureStore,
    ProcedureValidationEventType,
)
from agent_control.procedure_validation import (
    ProcedureActivationPolicy,
    ProcedureHealthDecision,
    ProcedureHealthPolicy,
    ProcedureValidationDecision,
    ProcedureValidationEvidence,
    assess_health,
    validate_and_promote,
    validate_procedure_definition,
)


def make_song(pid="song-1", *, status=ProcedureStatus.CANDIDATE, trigger="play {song} on youtube", params=True):
    parameters = (ProcedureParameter("song", "Song title", "string", True, constraints={"min_length": 1}),) if params else ()
    return LearnedProcedure(
        procedure_id=pid,
        name="browser.play.song",
        description="Play a requested song",
        trigger_pattern=trigger,
        status=status,
        parameters=parameters,
        steps=(ProcedureStep(
            "play", 0, "browser_play_song", {"query": "{song}"} if params else {"query": "Do I Wanna Know"},
            "Play requested song", required_capabilities=("browser_play_song",),
            expected_verification={"verdict": "PASS"},
        ),),
        required_capabilities=("browser_play_song",),
        provenance=ProcedureProvenance(ProcedureSourceType.VERIFIED_WORKFLOW, "task-1", "session-1"),
        source_task_id="task-1", source_session_id="session-1",
    )


def evidence(**overrides):
    base = dict(
        verified_evidence=1,
        independent_verification_pass=True,
        policy_compatible=True,
        recovery_clean=True,
        reusable=True,
        semantic_actions_valid=True,
        capabilities_available=True,
        human_approved=True,
        failure_evidence=False,
        unknown_evidence=False,
        source_task_id="task-1",
        source_session_id="session-1",
    )
    base.update(overrides)
    return ProcedureValidationEvidence(**base)


def test_candidate_promotes_only_after_explicit_evidence_and_human_approval(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        p = store.create_procedure(make_song())
        result = validate_and_promote(
            p.procedure_id,
            store=store,
            evidence=evidence(),
            available_capabilities=("browser_play_song",),
            semantic_actions_valid=True,
        )
        assert result.decision is ProcedureValidationDecision.PROMOTE
        assert result.transitioned_from is ProcedureStatus.CANDIDATE
        assert result.transitioned_to is ProcedureStatus.ACTIVE
        assert store.get_procedure(p.procedure_id).status is ProcedureStatus.ACTIVE
        events = store.list_validation_events(p.procedure_id)
        assert {e.event_type for e in events} >= {
            ProcedureValidationEventType.VALIDATION_STARTED,
            ProcedureValidationEventType.VALIDATION_PASSED,
            ProcedureValidationEventType.PROMOTION_GRANTED,
        }


def test_one_verified_execution_without_activation_approval_does_not_promote(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        p = store.create_procedure(make_song())
        result = validate_and_promote(
            p.procedure_id,
            store=store,
            evidence=evidence(human_approved=False),
            available_capabilities=("browser_play_song",),
            semantic_actions_valid=True,
        )
        assert result.decision is ProcedureValidationDecision.KEEP_CANDIDATE
        assert store.get_procedure(p.procedure_id).status is ProcedureStatus.CANDIDATE
        assert "human_activation_approval_required" in result.reasons


def test_unknown_or_missing_verification_fails_closed(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        p = store.create_procedure(make_song())
        result = validate_and_promote(
            p.procedure_id,
            store=store,
            evidence=evidence(independent_verification_pass=False, unknown_evidence=True),
            available_capabilities=("browser_play_song",),
            semantic_actions_valid=True,
        )
        assert result.decision is ProcedureValidationDecision.KEEP_CANDIDATE
        assert store.get_procedure(p.procedure_id).status is ProcedureStatus.CANDIDATE
        assert "independent_verification_required" in result.reasons
        assert "unknown_evidence_present" in result.reasons


def test_unparameterized_one_off_song_is_invalidated_as_not_generalizable(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        p = store.create_procedure(make_song(params=False, trigger="browser.play.song"))
        result = validate_and_promote(
            p.procedure_id,
            store=store,
            evidence=evidence(),
            available_capabilities=("browser_play_song",),
            semantic_actions_valid=True,
        )
        assert result.decision is ProcedureValidationDecision.INVALIDATE
        assert store.get_procedure(p.procedure_id).status is ProcedureStatus.INVALIDATED
        assert "procedure_not_reusable" in result.reasons


def test_unparameterized_one_off_definition_is_not_generalizable():
    p = make_song(params=False, trigger="browser.play.song")
    reasons = validate_procedure_definition(
        p,
        available_capabilities=("browser_play_song",),
        semantic_actions_valid=True,
    )
    assert "not_generalizable:no_parameters" in reasons


def test_missing_capability_fails_closed_and_keeps_candidate(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        p = store.create_procedure(make_song())
        result = validate_and_promote(
            p.procedure_id,
            store=store,
            evidence=evidence(capabilities_available=False),
            available_capabilities=(),
            semantic_actions_valid=True,
        )
        assert result.decision is ProcedureValidationDecision.INVALIDATE
        assert store.get_procedure(p.procedure_id).status is ProcedureStatus.INVALIDATED


def test_missing_provenance_fails_closed():
    p = make_song()
    object.__setattr__(p, "provenance", None)
    reasons = validate_procedure_definition(
        p, available_capabilities=("browser_play_song",), semantic_actions_valid=True
    )
    assert any(reason.startswith("malformed_procedure:") for reason in reasons)


def test_threshold_is_configurable(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        p = store.create_procedure(make_song())
        result = validate_and_promote(
            p.procedure_id,
            store=store,
            policy=ProcedureActivationPolicy(min_verified_evidence=2),
            evidence=evidence(verified_evidence=1),
            available_capabilities=("browser_play_song",),
            semantic_actions_valid=True,
        )
        assert result.decision is ProcedureValidationDecision.KEEP_CANDIDATE
        assert "insufficient_verified_evidence" in result.reasons
        assert store.get_procedure(p.procedure_id).status is ProcedureStatus.CANDIDATE


def test_active_degrades_from_existing_failure_counter(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        p = store.create_procedure(make_song())
        store.set_status(p.procedure_id, ProcedureStatus.VALIDATING)
        store.set_status(p.procedure_id, ProcedureStatus.ACTIVE)
        store.record_failure(p.procedure_id)
        result = assess_health(p.procedure_id, store=store)
        assert result is ProcedureHealthDecision.DEGRADE
        assert store.get_procedure(p.procedure_id).status is ProcedureStatus.DEGRADED


def test_degraded_reactivates_only_after_explicit_revalidation(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        p = store.create_procedure(make_song())
        store.set_status(p.procedure_id, ProcedureStatus.VALIDATING)
        store.set_status(p.procedure_id, ProcedureStatus.ACTIVE)
        store.set_status(p.procedure_id, ProcedureStatus.DEGRADED)
        assert assess_health(p.procedure_id, store=store) is ProcedureHealthDecision.NO_CHANGE
        assert assess_health(p.procedure_id, store=store, revalidation_passed=True) is ProcedureHealthDecision.REACTIVATE
        assert store.get_procedure(p.procedure_id).status is ProcedureStatus.ACTIVE


def test_active_invalidates_on_explicit_invalid_evidence(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        p = store.create_procedure(make_song())
        store.set_status(p.procedure_id, ProcedureStatus.VALIDATING)
        store.set_status(p.procedure_id, ProcedureStatus.ACTIVE)
        result = assess_health(p.procedure_id, store=store, invalid_evidence=True)
        assert result is ProcedureHealthDecision.INVALIDATE
        assert store.get_procedure(p.procedure_id).status is ProcedureStatus.INVALIDATED


def test_definition_validation_does_not_execute(tmp_path):
    p = make_song()
    reasons = validate_procedure_definition(
        p,
        available_capabilities=("browser_play_song",),
        semantic_actions_valid=True,
    )
    assert reasons == ()


def test_invalid_transition_still_owned_by_store(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        p = store.create_procedure(make_song())
        with pytest.raises(ValueError):
            store.set_status(p.procedure_id, ProcedureStatus.ACTIVE)


def test_required_parameter_not_used_is_invalidated(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        p = make_song()
        p = LearnedProcedure(**{**p.__dict__, "steps": (ProcedureStep(
            "play", 0, "browser_play_song", {"query": "fixed song"},
            required_capabilities=("browser_play_song",), expected_verification={"verdict": "PASS"},
        ),)})
        p = store.create_procedure(p)
        result = validate_and_promote(
            p.procedure_id, store=store, evidence=evidence(),
            available_capabilities=("browser_play_song",), semantic_actions_valid=True,
        )
        assert result.decision is ProcedureValidationDecision.INVALIDATE
        assert "parameter_not_bound_into_action:song" in result.reasons


def test_unresolved_parameter_is_invalidated(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        p = make_song()
        p = LearnedProcedure(**{**p.__dict__, "steps": (ProcedureStep(
            "play", 0, "browser_play_song", {"query": "{other}"},
            required_capabilities=("browser_play_song",), expected_verification={"verdict": "PASS"},
        ),)})
        p = store.create_procedure(p)
        result = validate_and_promote(
            p.procedure_id, store=store, evidence=evidence(),
            available_capabilities=("browser_play_song",), semantic_actions_valid=True,
        )
        assert result.decision is ProcedureValidationDecision.INVALIDATE
        assert "unresolved_parameter:other" in result.reasons


def test_semantic_action_invalid_is_invalidated(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        p = store.create_procedure(make_song())
        result = validate_and_promote(
            p.procedure_id, store=store, evidence=evidence(semantic_actions_valid=False),
            available_capabilities=("browser_play_song",), semantic_actions_valid=False,
        )
        assert result.decision is ProcedureValidationDecision.INVALIDATE
        assert "semantic_actions_invalid" in result.reasons
        assert "semantic_actions_not_verified" in result.reasons


def test_independent_verification_absent_is_not_promotable(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        p = make_song()
        p = LearnedProcedure(**{**p.__dict__, "steps": (ProcedureStep(
            "play", 0, "browser_play_song", {"query": "{song}"},
            required_capabilities=("browser_play_song",), expected_verification={},
        ),)})
        p = store.create_procedure(p)
        result = validate_and_promote(
            p.procedure_id, store=store, evidence=evidence(),
            available_capabilities=("browser_play_song",), semantic_actions_valid=True,
        )
        assert result.decision is ProcedureValidationDecision.INVALIDATE
        assert any(reason.startswith("independent_verification_missing:") for reason in result.reasons)


def test_policy_approval_is_not_the_same_as_action_policy(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        p = store.create_procedure(make_song())
        result = validate_and_promote(
            p.procedure_id, store=store, evidence=evidence(policy_compatible=False),
            available_capabilities=("browser_play_song",), semantic_actions_valid=True,
        )
        assert result.decision is ProcedureValidationDecision.KEEP_CANDIDATE
        assert "policy_compatibility_not_verified" in result.reasons



def test_active_procedure_flows_into_existing_p32c_p32d_components(tmp_path):
    from agent_control.policy import Policy
    from agent_control.procedure_execution import execute_procedure, ProcedureExecutionStatus
    from agent_control.procedure_retrieval import find_executable_procedures
    from agent_control.types import Verdict

    class Backend:
        def __init__(self):
            self.observations = 0
        def observe(self):
            self.observations += 1
            return {"state": "pass"}

    class Skill:
        def __init__(self, backend):
            self._backend = backend
            self.calls = 0
            self.verified = 0
        def adapt_action(self, action):
            return action
        def executor(self):
            owner = self
            class Executor:
                def execute(self, action):
                    owner.calls += 1
                    return type("Execution", (), {"ok": True, "error": None})()
            return Executor()
        def verifier(self):
            owner = self
            class Verifier:
                def verify(self, action, result):
                    owner.verified += 1
                    return type("Verification", (), {"status": "PASS", "ok": True, "detail": "verified"})()
            return Verifier()

    class Registry:
        def __init__(self, skill):
            self.skill = skill
        def find_for_action(self, kind):
            return self.skill if kind == "browser_play_song" else None
        def capabilities(self):
            return ("browser_play_song",)

    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        p = store.create_procedure(make_song())
        activated = validate_and_promote(
            p.procedure_id, store=store, evidence=evidence(),
            available_capabilities=("browser_play_song",), semantic_actions_valid=True,
        )
        assert activated.transitioned_to is ProcedureStatus.ACTIVE

        matches = find_executable_procedures("play do i wanna know", store, {"capabilities": {"browser_play_song"}})
        assert len(matches) == 1
        assert matches[0].procedure.status is ProcedureStatus.ACTIVE
        assert matches[0].binding.values == {"song": "do i wanna know"}

        backend = Backend()
        skill = Skill(backend)
        result = execute_procedure(
            matches[0],
            policy=Policy(workspace=tmp_path / "workspace", confirm_mode="deny", refuse_if_elevated=False),
            skills=Registry(skill),
        )
        assert result.execution_status is ProcedureExecutionStatus.EXECUTED
        assert result.verification_status is Verdict.PASS
        assert result.verified_success
        assert skill.calls == 1
        assert skill.verified == 1
        assert backend.observations == 1
