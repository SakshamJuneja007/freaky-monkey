from __future__ import annotations

import sqlite3
import time

import pytest

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


def make_procedure(pid="p1", *, version=1, status=ProcedureStatus.CANDIDATE, parent=None):
    return LearnedProcedure(
        procedure_id=pid, name="play_youtube_song", description="Play a requested song",
        trigger_pattern="play {song} on youtube", status=status, version=version,
        parameters=(ProcedureParameter("song", "Song title", "string", True),
                    ProcedureParameter("recipient", "Recipient", "string", True, sensitive=True)),
        steps=(
            ProcedureStep("open", 0, "browser_open", {"url": "https://youtube.com"}, "Open YouTube", required_capabilities=("browser",)),
            ProcedureStep("search", 1, "browser_search", {"query": "{song}"}, "Search for the song", dependencies=("open",), required_capabilities=("browser",)),
        ),
        preconditions=(ProcedureCondition("browser", "browser is available"),),
        postconditions=(ProcedureCondition("playing", "requested result is playing"),),
        required_capabilities=("browser",),
        resource_requirements={"browser": "available"},
        confidence=.8, importance=.7,
        provenance=ProcedureProvenance(ProcedureSourceType.MANUAL, "task-1", "session-1", parent_procedure_id=parent, source_version=version-1 if parent else None),
        source_task_id="task-1", source_session_id="session-1", parent_procedure_id=parent,
        metadata={"test": True},
    )


def test_01_database_initializes(tmp_path):
    with ProcedureStore(tmp_path / "procedures.sqlite3") as store:
        assert store.path.exists()
        assert store._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_02_survives_restart(tmp_path):
    path = tmp_path / "procedures.sqlite3"
    with ProcedureStore(path) as s: s.create_procedure(make_procedure())
    with ProcedureStore(path) as s: assert s.get_procedure("p1").name == "play_youtube_song"


def test_03_create_and_get(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        created = s.create_procedure(make_procedure())
        assert created.procedure_id == "p1" and created.status is ProcedureStatus.CANDIDATE


def test_04_parameters_persist(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        s.create_procedure(make_procedure())
        p = s.get_procedure("p1").parameters
        by_name = {x.name: x for x in p}
        assert by_name["song"].required is True and by_name["recipient"].sensitive is True


def test_05_steps_are_semantic_and_persist(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        s.create_procedure(make_procedure())
        step = s.get_procedure("p1").steps[1]
        assert step.action_name == "browser_search" and step.arguments == {"query": "{song}"}
        assert "x=" not in str(step.arguments)


def test_06_conditions_persist(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        s.create_procedure(make_procedure())
        p = s.get_procedure("p1")
        assert p.preconditions[0].condition == "browser is available"
        assert p.postconditions[0].condition == "requested result is playing"


def test_07_version_persists(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        s.create_procedure(make_procedure(version=1))
        assert s.get_procedure_version("play_youtube_song", 1).version == 1


def test_08_duplicate_name_version_rejected(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        s.create_procedure(make_procedure())
        with pytest.raises(sqlite3.IntegrityError): s.create_procedure(make_procedure(pid="p2"))


def test_09_lifecycle_transitions_are_validated(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        s.create_procedure(make_procedure())
        s.set_status("p1", ProcedureStatus.VALIDATING)
        s.set_status("p1", ProcedureStatus.ACTIVE)
        s.set_status("p1", ProcedureStatus.DEGRADED)
        s.set_status("p1", ProcedureStatus.INVALIDATED)
        assert s.get_procedure("p1").status is ProcedureStatus.INVALIDATED


def test_10_invalidated_cannot_become_active(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        s.create_procedure(make_procedure())
        s.invalidate_procedure("p1")
        with pytest.raises(ValueError): s.set_status("p1", ProcedureStatus.ACTIVE)


def test_11_new_version_keeps_parent(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        s.create_procedure(make_procedure())
        v2 = make_procedure("p2", version=2, parent="p1")
        s.create_new_version(v2, version=2)
        assert s.get_procedure_version("play_youtube_song", 1).procedure_id == "p1"
        assert s.get_procedure_version("play_youtube_song", 2).parent_procedure_id == "p1"


def test_12_counters_and_timestamps(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        s.create_procedure(make_procedure())
        before = s.get_procedure("p1").updated_at
        time.sleep(.001)
        s.record_success("p1"); s.record_failure("p1"); s.record_unknown("p1")
        p = s.get_procedure("p1")
        assert (p.success_count, p.failure_count, p.unknown_count) == (1, 1, 1)
        assert p.last_used_at and p.last_success_at and p.updated_at >= before


def test_13_sensitive_parameter_schema_has_no_runtime_value(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        p = s.create_procedure(make_procedure())
        row = s._conn.execute("SELECT * FROM procedure_parameters WHERE procedure_id='p1' AND name='recipient'").fetchone()
        assert row["sensitive"] == 1 and row["default_value_json"] is None
        assert "someone@example.com" not in str(row)


def test_14_malformed_procedure_rejected(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        bad = make_procedure()
        bad = LearnedProcedure(**{**bad.__dict__, "steps": (ProcedureStep("x", 0, "a"), ProcedureStep("y", 0, "b"))})
        with pytest.raises(ValueError): s.create_procedure(bad)


def test_15_foreign_key_integrity(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        with pytest.raises(sqlite3.IntegrityError):
            s._conn.execute("INSERT INTO procedure_steps VALUES ('missing','x',0,'a','{}','', '[]','[]','{}','{}','{}')")


def test_16_multiple_procedures_isolated(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        s.create_procedure(make_procedure("p1"))
        q = make_procedure("p2"); q = LearnedProcedure(**{**q.__dict__, "name": "open_notepad"})
        s.create_procedure(q)
        assert len(s.get_procedure("p1").steps) == 2 and s.get_procedure("p2").name == "open_notepad"


def test_17_update_requires_explicit_same_identity_and_status(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        s.create_procedure(make_procedure())
        p = s.get_procedure("p1")
        changed = LearnedProcedure(**{**p.__dict__, "description": "Updated"})
        assert s.update_procedure(changed).description == "Updated"
        status_changed = LearnedProcedure(**{**s.get_procedure("p1").__dict__, "status": ProcedureStatus.ACTIVE})
        with pytest.raises(ValueError): s.update_procedure(status_changed)


def test_18_active_insertion_is_not_automatic(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        with pytest.raises(ValueError): s.create_procedure(make_procedure(status=ProcedureStatus.ACTIVE))


def test_19_version_cannot_overwrite(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        s.create_procedure(make_procedure())
        s.create_new_version(make_procedure("p2", version=2, parent="p1"), version=2)
        with pytest.raises(sqlite3.IntegrityError): s.create_new_version(make_procedure("p3", version=2, parent="p1"), version=2)


def test_20_provenance_persists(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        s.create_procedure(make_procedure())
        p = s.get_procedure("p1")
        assert p.provenance.source_type is ProcedureSourceType.MANUAL
        assert p.provenance.source_task_id == "task-1"


def test_21_no_execution_or_retrieval_path_is_added():
    from agent_control import procedure_store
    assert not hasattr(procedure_store.ProcedureStore, "execute")
    assert not hasattr(procedure_store.ProcedureStore, "search")


def test_22_p31_store_remains_separate(tmp_path):
    from agent_control.persistent_memory import PersistentMemory
    with PersistentMemory(tmp_path / "memory.sqlite3") as m:
        m.put(memory_type="USER_FACT", content="DEIMOS is a project", source="user_explicit", confidence=.9)
    with ProcedureStore(tmp_path / "procedures.sqlite3") as s:
        assert s.list_procedures() == []


def test_23_required_capabilities_persist(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        s.create_procedure(make_procedure())
        assert s.get_procedure("p1").required_capabilities == ("browser",)


def test_24_parent_must_exist(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        with pytest.raises(ValueError): s.create_new_version(make_procedure("p2", version=2, parent="missing"), version=2)


def test_25_retired_is_terminal(tmp_path):
    with ProcedureStore(tmp_path / "p.sqlite3") as s:
        s.create_procedure(make_procedure())
        s.set_status("p1", ProcedureStatus.RETIRED)
        with pytest.raises(ValueError): s.set_status("p1", ProcedureStatus.CANDIDATE)
