from __future__ import annotations

import time

from agent_control.procedure_retrieval import (
    BindingStatus,
    ValidationStatus,
    bind_procedure_parameters,
    find_applicable_procedures,
    retrieve_procedures,
    validate_procedure_current_state,
)
from agent_control.procedure_store import (
    LearnedProcedure,
    ProcedureCondition,
    ProcedureParameter,
    ProcedureProvenance,
    ProcedureSourceType,
    ProcedureStatus,
    ProcedureStep,
    ProcedureStore,
)


def make_song(pid="song-1", *, status=ProcedureStatus.CANDIDATE, preconditions=()):
    return LearnedProcedure(
        procedure_id=pid,
        name="browser.play.song",
        description="Play a requested song",
        trigger_pattern="play {song} on youtube",
        status=status,
        parameters=(ProcedureParameter("song", "Song title", "string", True, constraints={"min_length": 1}),),
        steps=(ProcedureStep("play", 0, "browser_play_song", {"query": "{song}"}, "Play requested song", required_capabilities=("browser_play_song",)),),
        preconditions=tuple(preconditions),
        required_capabilities=("browser_play_song",),
        provenance=ProcedureProvenance(ProcedureSourceType.VERIFIED_WORKFLOW, "task-1", "session-1"),
        source_task_id="task-1", source_session_id="session-1",
    )


def browser_state(url="https://www.youtube.com/watch?v=1", text="YouTube"):
    return {"observations": [{"source": "browser", "url": url, "text": text, "ok": True, "observed_at": time.time()}]}


def test_01_candidate_retrieval_semantic(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        store.create_procedure(make_song())
        found = retrieve_procedures("play do i wanna know", store)
        assert [p.name for p in found] == ["browser.play.song"]


def test_02_irrelevant_procedure_rejected(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        store.create_procedure(make_song())
        assert retrieve_procedures("send hello to papa", store) == []


def test_03_parameter_binding(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        p = make_song()
        store.create_procedure(p)
        result = bind_procedure_parameters(store.get_procedure(p.procedure_id), "play do i wanna know")
        assert result.status is BindingStatus.BOUND
        assert result.values == {"song": "do i wanna know"}


def test_04_missing_parameter():
    result = bind_procedure_parameters(make_song(), "play")
    assert result.status is BindingStatus.MISSING
    assert "parameter_missing:song" in result.reasons


def test_05_invalid_parameter():
    p = make_song()
    result = bind_procedure_parameters(p, {"action_kind": "browser_play_song", "parameters": {"song": 123}})
    assert result.status is BindingStatus.INVALID


def test_06_current_state_pass():
    p = make_song(preconditions=(ProcedureCondition("browser", "browser is available"), ProcedureCondition("youtube", "YouTube page is open")))
    result = validate_procedure_current_state(p, browser_state())
    assert result.status is ValidationStatus.APPLICABLE


def test_07_current_state_fail():
    p = make_song(preconditions=(ProcedureCondition("youtube", "YouTube page is open"),))
    result = validate_procedure_current_state(p, browser_state("https://example.com", "Example"))
    assert result.status is ValidationStatus.NOT_APPLICABLE


def test_08_current_state_unknown():
    p = make_song(preconditions=(ProcedureCondition("custom", "some unsupported semantic condition"),))
    result = validate_procedure_current_state(p, browser_state())
    assert result.status is ValidationStatus.UNKNOWN


def test_09_stale_state_is_not_current():
    p = make_song(preconditions=(ProcedureCondition("youtube", "YouTube page is open"),))
    old = {"observations": [{"source": "browser", "url": "https://www.youtube.com", "text": "YouTube", "ok": True, "observed_at": time.time() - 10}]}
    result = validate_procedure_current_state(p, old, max_age_s=1.0)
    assert result.status is ValidationStatus.UNKNOWN


def test_10_no_execution():
    p = make_song()
    # Validation itself has no executor dependency or action invocation.
    result = validate_procedure_current_state(p, browser_state())
    assert result.status is ValidationStatus.APPLICABLE


def test_10b_unparameterized_legacy_song_candidate_is_not_replayed():
    p = LearnedProcedure(
        procedure_id="legacy-song", name="browser.play.song",
        description="Legacy song candidate", trigger_pattern="browser.play.song",
        parameters=(),
        steps=(ProcedureStep("play", 0, "browser_play_song", {}, "Play song", required_capabilities=("browser_play_song",)),),
        required_capabilities=("browser_play_song",),
        provenance=ProcedureProvenance(ProcedureSourceType.VERIFIED_WORKFLOW, "t", "s"),
        source_task_id="t", source_session_id="s",
    )
    result = bind_procedure_parameters(p, "play do i wanna know")
    assert result.status is BindingStatus.MISSING
    assert "parameter_missing:query" in result.reasons


def test_11_candidate_status_preserved(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        p = make_song()
        store.create_procedure(p)
        find_applicable_procedures("play do i wanna know", store, browser_state())
        assert store.get_procedure(p.procedure_id).status is ProcedureStatus.CANDIDATE


def test_12_sensitive_binding_is_redacted():
    p = make_song()
    p = LearnedProcedure(**{**p.__dict__, "parameters": p.parameters + (ProcedureParameter("token", type="string", sensitive=True),)})
    result = bind_procedure_parameters(p, {"action_kind": "browser_play_song", "parameters": {"song": "x", "token": "SECRET-VALUE"}})
    assert result.status is BindingStatus.BOUND
    assert result.debug_dict()["bindings"]["token"] == "<redacted>"
    assert "SECRET-VALUE" not in str(result.debug_dict())


def test_13_duplicate_candidates_are_deterministic(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as store:
        store.create_procedure(make_song("song-a"))
        second = make_song("song-b")
        second = LearnedProcedure(**{**second.__dict__, "version": 2, "parent_procedure_id": "song-a",
                                    "provenance": ProcedureProvenance(ProcedureSourceType.VERIFIED_WORKFLOW, "task-1", "session-1", parent_procedure_id="song-a", source_version=1),
                                    "source_task_id": "task-1", "source_session_id": "session-1"})
        store.create_new_version(second, version=2)
        first = [p.procedure_id for p in retrieve_procedures("play song", store)]
        second = [p.procedure_id for p in retrieve_procedures("play song", store)]
        assert first == second


def test_14_roundtrip_retrieve_bind_validate(tmp_path):
    path = tmp_path / "p.sqlite3"
    with ProcedureStore(path) as store:
        store.create_procedure(make_song())
    with ProcedureStore(path) as store:
        matches = find_applicable_procedures("play do i wanna know", store, browser_state())
        assert len(matches) == 1
        assert matches[0].binding.values == {"song": "do i wanna know"}
        assert matches[0].validation.status is ValidationStatus.APPLICABLE
        assert matches[0].execution_allowed is False
