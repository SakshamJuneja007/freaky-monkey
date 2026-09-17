from pathlib import Path

from agent_control.api import _try_execute_procedure, TaskStatus
from agent_control.policy import Policy
from agent_control.procedure_store import (
    ProcedureParameter,
    ProcedureProvenance,
    ProcedureSourceType,
    ProcedureStatus,
    ProcedureStep,
    LearnedProcedure,
    ProcedureStore,
)
from agent_control.trace import Trace
from agent_control.types import Check, VerificationResult, Verdict


class FakeBackend:
    def __init__(self):
        self.observations = 0
        self._last_observation = None

    def ensure_ready(self):
        return None

    def observe(self):
        self.observations += 1
        self._last_observation = type("Obs", (), {
            "generation": self.observations,
            "elements": (),
            "url": "https://example.test/",
            "text": "ready",
        })()
        return {"ok": True}


class FakeSkill:
    name = "fake"

    def __init__(self, backend):
        self._backend = backend
        self.calls = 0
        self.verified = 0

    def supports(self, kind):
        return kind == "browser_open_url"

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
                assert owner._backend.observations >= 1
                return type("Verification", (), {
                    "status": "PASS",
                    "ok": True,
                    "detail": "fresh state verified",
                })()

        return Verifier()


class FakeSkills:
    def __init__(self, skill):
        self.skill = skill

    def find_for_action(self, kind):
        return self.skill if self.skill.supports(kind) else None

    def capabilities(self):
        return ("browser",)


def _active_store(tmp_path: Path) -> ProcedureStore:
    store = ProcedureStore(tmp_path / "procedures.sqlite3")
    procedure = LearnedProcedure(
        procedure_id="proc-active",
        name="browser.open",
        description="active test procedure",
        trigger_pattern="open example",
        parameters=(),
        steps=(ProcedureStep(
            step_id="step-0",
            order=0,
            action_name="browser_open_url",
            arguments={"url": "https://example.test/"},
            required_capabilities=("browser_open_url",),
        ),),
        required_capabilities=("browser_open_url",),
        provenance=ProcedureProvenance(ProcedureSourceType.MANUAL),
    )
    store.create_procedure(procedure)
    store.set_status("proc-active", ProcedureStatus.VALIDATING)
    store.set_status("proc-active", ProcedureStatus.ACTIVE)
    return store


def test_active_procedure_runs_before_planner_path(tmp_path):
    store = _active_store(tmp_path)
    backend = FakeBackend()
    skill = FakeSkill(backend)
    policy = Policy(workspace=tmp_path / "workspace", confirm_mode="deny", refuse_if_elevated=False)
    trace = Trace(task_id="integration", condition="test")

    result = _try_execute_procedure(
        "open example",
        type("Task", (), {"task_id": "integration", "goal": "open example"})(),
        {},
        policy,
        backend,
        trace,
        store,
        FakeSkills(skill),
    )

    assert result is not None
    assert result.status is TaskStatus.SUCCESS
    assert result.verified == Verdict.PASS.value
    assert skill.calls == 1
    assert skill.verified == 1
    assert backend.observations >= 2
    assert store.get_procedure("proc-active").status is ProcedureStatus.ACTIVE
    store.close()


def test_candidate_is_not_executed_by_integration_lane(tmp_path):
    store = ProcedureStore(tmp_path / "procedures.sqlite3")
    candidate = LearnedProcedure(
        procedure_id="proc-candidate",
        name="browser.open",
        description="candidate",
        trigger_pattern="open example",
        steps=(ProcedureStep("step-0", 0, "browser_open_url", {"url": "https://example.test/"}),),
        provenance=ProcedureProvenance(ProcedureSourceType.MANUAL),
    )
    store.create_procedure(candidate)
    backend = FakeBackend()
    skill = FakeSkill(backend)
    policy = Policy(workspace=tmp_path / "workspace", confirm_mode="deny", refuse_if_elevated=False)
    trace = Trace(task_id="candidate", condition="test")

    result = _try_execute_procedure(
        "open example",
        type("Task", (), {"task_id": "candidate", "goal": "open example"})(),
        {}, policy, backend, trace, store, FakeSkills(skill),
    )

    assert result is None
    assert skill.calls == 0
    assert store.get_procedure("proc-candidate").status is ProcedureStatus.CANDIDATE
    store.close()


def test_session_real_route_reaches_active_procedure_lane(tmp_path, monkeypatch):
    """Exercise Session -> _run -> api -> retrieval -> P3.2-D without an LLM/browser."""
    from agent_control.session import Session

    store = _active_store(tmp_path)
    backend = FakeBackend()
    skill = FakeSkill(backend)
    policy_seen = []

    class FakeNarrator:
        def note(self, message):
            policy_seen.append(str(message))
        def accepted(self, *args, **kwargs):
            return None
        def reply(self, *args, **kwargs):
            return None
        def close(self, *args, **kwargs):
            return None

    class FakeUserTask:
        pass

    monkeypatch.setattr(
        "agent_control.skills.builtin.build_builtin_registry",
        lambda policy, browser_backend=None: FakeSkills(skill),
    )
    monkeypatch.setattr("agent_control.api.resolve_task", lambda _text: None)
    monkeypatch.setattr("agent_control.policy.is_elevated", lambda: False)
    session = Session(
        narrator=FakeNarrator(),
        planner="mock",
        debug=True,
        show_status=True,
        runtime_persistence_path=":memory:",
    )
    session._procedure_store = store
    monkeypatch.setattr(session, "_browser_for_task", lambda *args, **kwargs: backend)
    monkeypatch.setattr(session, "_action_reply", lambda task, result: result.detail)

    turn = session.submit("open example")

    assert turn.result is not None
    assert turn.result.status is TaskStatus.SUCCESS
    assert skill.calls == 1
    assert skill.verified == 1
    assert backend.observations >= 2
    assert any("[procedure] retrieval: MATCH" in line for line in policy_seen)
    assert any("[procedure] observation: FRESH" in line for line in policy_seen)
    assert any("[procedure] verification: PASS" in line for line in policy_seen)
    assert store.get_procedure("proc-active").status is ProcedureStatus.ACTIVE
    session.close()

class FakeSongSkill(FakeSkill):
    def supports(self, kind):
        return kind == "browser.play.song"


def _active_song_store(tmp_path: Path, *, preconditions=()) -> ProcedureStore:
    store = ProcedureStore(tmp_path / "song-procedures.sqlite3")
    procedure = LearnedProcedure(
        procedure_id="proc-song-active",
        name="browser.play.song",
        description="parameterized song procedure",
        trigger_pattern="play {song}",
        parameters=(ProcedureParameter("song", type="string", required=True),),
        steps=(ProcedureStep(
            step_id="step-song",
            order=0,
            action_name="browser.play.song",
            arguments={"query": "{song}"},
            required_capabilities=("browser.play.song",),
        ),),
        preconditions=tuple(preconditions),
        required_capabilities=("browser.play.song",),
        provenance=ProcedureProvenance(ProcedureSourceType.MANUAL),
    )
    store.create_procedure(procedure)
    assert store.get_procedure("proc-song-active").status is ProcedureStatus.CANDIDATE
    store.set_status("proc-song-active", ProcedureStatus.VALIDATING)
    store.set_status("proc-song-active", ProcedureStatus.ACTIVE)
    return store


def _run_song_lane(tmp_path, store, backend, skill, policy_obj=None):
    policy_obj = policy_obj or Policy(workspace=tmp_path / "workspace", confirm_mode="deny", refuse_if_elevated=False)
    records = []
    trace = Trace(task_id="song-integration", condition="test", on_event=records.append)
    result = _try_execute_procedure(
        "play killshot",
        type("Task", (), {"task_id": "song-integration", "goal": "play killshot"})(),
        {}, policy_obj, backend, trace, store, FakeSkills(skill),
    )
    return result, records


def test_parameterized_active_song_reaches_p32d(tmp_path):
    store = _active_song_store(tmp_path)
    backend = FakeBackend()
    skill = FakeSongSkill(backend)
    result, trace = _run_song_lane(tmp_path, store, backend, skill)

    assert result is not None
    assert result.status is TaskStatus.SUCCESS
    assert result.verified == Verdict.PASS.value
    assert skill.calls == 1
    assert skill.verified == 1
    assert backend.observations >= 2
    assert store.get_procedure("proc-song-active").status is ProcedureStatus.ACTIVE

    retrieval = next(data for data in trace if data["event"] == "procedure_retrieval_result")
    assert retrieval["result"] == "MATCH"
    assert retrieval["procedure_status"] == "ACTIVE"
    assert retrieval["binding_status"] == "BOUND"
    assert retrieval["validation_status"] == "APPLICABLE"
    assert retrieval["execution_allowed"] is True
    assert any(e["event"] == "procedure_policy" and e["decision"] == "ALLOW" for e in trace)
    assert any(e["event"] == "procedure_execution_action" and e["result"] == "SUCCESS" for e in trace)
    assert any(e["event"] == "procedure_fresh_observation_start" for e in trace)
    assert any(e["event"] == "procedure_execution_observation" and e["freshness"] == "FRESH" for e in trace)
    assert any(e["event"] == "procedure_execution_verification" and e["verification"] == "PASS" for e in trace)
    store.close()


def test_active_song_missing_binding_does_not_execute(tmp_path):
    store = _active_song_store(tmp_path)
    backend = FakeBackend()
    skill = FakeSongSkill(backend)
    records = []
    trace = Trace(task_id="song-missing", condition="test", on_event=records.append)
    task = type("Task", (), {
        "task_id": "song-missing",
        "goal": "play killshot",
        "action": type("Action", (), {"kind": "browser_play_song", "params": {}})(),
    })()
    result = _try_execute_procedure(
        "play killshot",
        task,
        {}, Policy(workspace=tmp_path / "workspace", refuse_if_elevated=False),
        backend, trace, store, FakeSkills(skill),
    )
    assert result is None
    assert skill.calls == 0
    match_event = next(e for e in records if e["event"] == "procedure_retrieval_result")
    assert match_event["result"] == "MATCH"
    assert match_event["procedure_status"] == "ACTIVE"
    assert match_event["binding_status"] == "MISSING"
    store.close()


def test_active_song_not_applicable_does_not_execute(tmp_path):
    from agent_control.procedure_store import ProcedureCondition

    condition = ProcedureCondition(
        condition_id="youtube-required",
        condition="youtube page is required",
        metadata={"type": "url_contains", "value": "youtube.com"},
    )
    store = _active_song_store(tmp_path, preconditions=(condition,))
    backend = FakeBackend()
    skill = FakeSongSkill(backend)
    result, trace = _run_song_lane(tmp_path, store, backend, skill)
    assert result is None
    assert skill.calls == 0
    match_event = next(e for e in trace if e["event"] == "procedure_retrieval_result")
    assert match_event["result"] == "MATCH"
    assert match_event["binding_status"] == "BOUND"
    assert match_event["validation_status"] == "NOT_APPLICABLE"
    store.close()


def test_active_song_policy_denial_does_not_execute(tmp_path):
    from agent_control.policy import Decision

    store = _active_song_store(tmp_path)
    backend = FakeBackend()
    skill = FakeSongSkill(backend)
    policy_obj = Policy(workspace=tmp_path / "workspace", refuse_if_elevated=False)
    policy_obj.check = lambda action: (Decision.DENY, "denied")
    result, trace = _run_song_lane(tmp_path, store, backend, skill, policy_obj)
    assert result is None
    assert skill.calls == 0
    assert any(e["event"] == "procedure_policy" and e["decision"] == "DENY" for e in trace)
    store.close()
