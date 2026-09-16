"""P3.2-A learned-procedure models and local persistence.

This module is intentionally dormant infrastructure.  It defines semantic,
parameterized procedure data and persists it locally, but it does not retrieve,
learn, select, execute, authorize, or verify procedures.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORE = ROOT / ".agent_memory" / "procedures.sqlite3"
SCHEMA_VERSION = 1


class ProcedureStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    VALIDATING = "VALIDATING"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    INVALIDATED = "INVALIDATED"
    RETIRED = "RETIRED"


class ProcedureSourceType(str, Enum):
    VERIFIED_WORKFLOW = "VERIFIED_WORKFLOW"
    MANUAL = "MANUAL"
    IMPORTED = "IMPORTED"
    SYSTEM = "SYSTEM"


_ALLOWED_TRANSITIONS: dict[ProcedureStatus, set[ProcedureStatus]] = {
    ProcedureStatus.CANDIDATE: {ProcedureStatus.VALIDATING, ProcedureStatus.INVALIDATED, ProcedureStatus.RETIRED},
    ProcedureStatus.VALIDATING: {ProcedureStatus.ACTIVE, ProcedureStatus.CANDIDATE, ProcedureStatus.INVALIDATED, ProcedureStatus.RETIRED},
    ProcedureStatus.ACTIVE: {ProcedureStatus.DEGRADED, ProcedureStatus.INVALIDATED, ProcedureStatus.RETIRED},
    ProcedureStatus.DEGRADED: {ProcedureStatus.ACTIVE, ProcedureStatus.INVALIDATED, ProcedureStatus.RETIRED},
    ProcedureStatus.INVALIDATED: {ProcedureStatus.RETIRED},
    ProcedureStatus.RETIRED: set(),
}

_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:api[_ -]?key|access[_ -]?token|auth[_ -]?token|password|passwd|cookie|session[_ -]?token|bearer)\b\s*[:=]\s*[^\s,;]+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~-]{12,}"),
    re.compile(r"(?i)\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_-]{12,}|xox[baprs]-[A-Za-z0-9-]{12,})\b"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_text(value: Any, field_name: str, *, required: bool = True, max_len: int = 2000) -> str:
    if value is None:
        if required:
            raise ValueError(f"{field_name} is required")
        return ""
    value = str(value).strip()
    if required and not value:
        raise ValueError(f"{field_name} is required")
    if len(value) > max_len:
        raise ValueError(f"{field_name} is too long")
    return value


def _contains_secret(value: Any) -> bool:
    if isinstance(value, str):
        return any(p.search(value) for p in _SECRET_PATTERNS)
    if isinstance(value, dict):
        return any(_contains_secret(k) or _contains_secret(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_secret(v) for v in value)
    return False


def _clean_metadata(value: dict[str, Any] | None, name: str) -> dict[str, Any]:
    data = dict(value or {})
    if _contains_secret(data):
        raise ValueError(f"{name} contains credential-like data")
    return data


@dataclass(frozen=True)
class ProcedureParameter:
    name: str
    description: str = ""
    type: str = "string"
    required: bool = True
    default: Any = None
    constraints: dict[str, Any] = field(default_factory=dict)
    sensitive: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_text(self.name, "parameter name", max_len=128))
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", self.name):
            raise ValueError(f"invalid parameter name: {self.name!r}")
        object.__setattr__(self, "description", _validate_text(self.description, "parameter description", required=False, max_len=1000))
        object.__setattr__(self, "type", _validate_text(self.type, "parameter type", max_len=64))
        if _contains_secret(self.default) or _contains_secret(self.constraints):
            raise ValueError("sensitive parameter runtime data cannot be persisted")

    def to_dict(self) -> dict[str, Any]:
        # A procedure stores schema only.  Sensitive parameters never carry a
        # runtime value here; default is schema metadata, not execution input.
        return {"name": self.name, "description": self.description, "type": self.type,
                "required": self.required, "default": self.default,
                "constraints": dict(self.constraints), "sensitive": self.sensitive}


@dataclass(frozen=True)
class ProcedureStep:
    step_id: str
    order: int
    action_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    dependencies: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    expected_observation: dict[str, Any] = field(default_factory=dict)
    expected_verification: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _validate_text(self.step_id, "step_id", max_len=128))
        if self.order < 0:
            raise ValueError("step order must be non-negative")
        object.__setattr__(self, "action_name", _validate_text(self.action_name, "action_name", max_len=128))
        if _contains_secret(self.arguments) or _contains_secret(self.metadata):
            raise ValueError("procedure step contains credential-like data")
        for dep in self.dependencies:
            _validate_text(dep, "dependency", max_len=128)

    def to_dict(self) -> dict[str, Any]:
        return {"step_id": self.step_id, "order": self.order, "action_name": self.action_name,
                "arguments": dict(self.arguments), "description": self.description,
                "dependencies": list(self.dependencies), "required_capabilities": list(self.required_capabilities),
                "expected_observation": dict(self.expected_observation),
                "expected_verification": dict(self.expected_verification), "metadata": dict(self.metadata)}


@dataclass(frozen=True)
class ProcedureCondition:
    condition_id: str
    condition: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition_id", _validate_text(self.condition_id, "condition_id", max_len=128))
        object.__setattr__(self, "condition", _validate_text(self.condition, "condition", max_len=1000))
        if _contains_secret(self.metadata):
            raise ValueError("condition metadata contains credential-like data")


@dataclass(frozen=True)
class ProcedureProvenance:
    source_type: ProcedureSourceType
    source_task_id: str | None = None
    source_session_id: str | None = None
    created_at: str = field(default_factory=_now)
    parent_procedure_id: str | None = None
    source_version: int | None = None


@dataclass(frozen=True)
class LearnedProcedure:
    procedure_id: str
    name: str
    description: str
    trigger_pattern: str
    status: ProcedureStatus = ProcedureStatus.CANDIDATE
    version: int = 1
    parameters: tuple[ProcedureParameter, ...] = ()
    steps: tuple[ProcedureStep, ...] = ()
    preconditions: tuple[ProcedureCondition, ...] = ()
    postconditions: tuple[ProcedureCondition, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    resource_requirements: dict[str, Any] = field(default_factory=dict)
    success_count: int = 0
    failure_count: int = 0
    unknown_count: int = 0
    confidence: float = 0.0
    importance: float = 0.5
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    last_used_at: str | None = None
    last_success_at: str | None = None
    provenance: ProcedureProvenance = field(default_factory=lambda: ProcedureProvenance(ProcedureSourceType.MANUAL))
    source_task_id: str | None = None
    source_session_id: str | None = None
    parent_procedure_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _validate_text(self.procedure_id, "procedure_id", max_len=128)
        _validate_text(self.name, "name", max_len=200)
        _validate_text(self.description, "description", max_len=2000)
        _validate_text(self.trigger_pattern, "trigger_pattern", max_len=1000)
        if self.version < 1:
            raise ValueError("version must be >= 1")
        for value, label in ((self.confidence, "confidence"), (self.importance, "importance")):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be between 0 and 1")
        for value, label in ((self.success_count, "success_count"), (self.failure_count, "failure_count"), (self.unknown_count, "unknown_count")):
            if value < 0:
                raise ValueError(f"{label} must be non-negative")
        names = [p.name.casefold() for p in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("duplicate parameter names")
        ids = [s.step_id.casefold() for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate step IDs")
        orders = [s.order for s in self.steps]
        if orders != list(range(len(orders))):
            raise ValueError("step orders must be contiguous and deterministic starting at 0")
        condition_ids = [c.condition_id.casefold() for c in (*self.preconditions, *self.postconditions)]
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("duplicate condition IDs")
        known = set(ids)
        for step in self.steps:
            if any(dep.casefold() not in known for dep in step.dependencies):
                raise ValueError(f"step {step.step_id} has an unknown dependency")
        _clean_metadata(self.resource_requirements, "resource_requirements")
        _clean_metadata(self.metadata, "metadata")
        if self.provenance.parent_procedure_id != self.parent_procedure_id:
            raise ValueError("parent procedure ID must agree with provenance")
        if self.provenance.source_task_id != self.source_task_id:
            raise ValueError("source task ID must agree with provenance")
        if self.provenance.source_session_id != self.source_session_id:
            raise ValueError("source session ID must agree with provenance")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"procedure_id": self.procedure_id, "name": self.name, "description": self.description,
                "trigger_pattern": self.trigger_pattern, "status": self.status.value, "version": self.version,
                "parameters": [p.to_dict() for p in self.parameters], "steps": [s.to_dict() for s in self.steps],
                "preconditions": [{"condition_id": c.condition_id, "condition": c.condition, "metadata": c.metadata} for c in self.preconditions],
                "postconditions": [{"condition_id": c.condition_id, "condition": c.condition, "metadata": c.metadata} for c in self.postconditions],
                "required_capabilities": list(self.required_capabilities), "resource_requirements": dict(self.resource_requirements),
                "success_count": self.success_count, "failure_count": self.failure_count, "unknown_count": self.unknown_count,
                "confidence": self.confidence, "importance": self.importance, "created_at": self.created_at,
                "updated_at": self.updated_at, "last_used_at": self.last_used_at, "last_success_at": self.last_success_at,
                "provenance": {"source_type": self.provenance.source_type.value, "source_task_id": self.provenance.source_task_id,
                               "source_session_id": self.provenance.source_session_id, "created_at": self.provenance.created_at,
                               "parent_procedure_id": self.provenance.parent_procedure_id, "source_version": self.provenance.source_version},
                "source_task_id": self.source_task_id, "source_session_id": self.source_session_id,
                "parent_procedure_id": self.parent_procedure_id, "metadata": dict(self.metadata)}


class ProcedureStore:
    """Deterministic CRUD/version store. No execution or retrieval semantics."""

    def __init__(self, path: str | Path = DEFAULT_STORE) -> None:
        configured = os.getenv("DEIMOS_PROCEDURE_STORE", "").strip()
        self.path = Path(configured or path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self._conn.execute("INSERT OR IGNORE INTO schema_meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
            self._conn.execute("""CREATE TABLE IF NOT EXISTS procedures (
                procedure_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
                trigger_pattern TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('CANDIDATE','VALIDATING','ACTIVE','DEGRADED','INVALIDATED','RETIRED')),
                version INTEGER NOT NULL CHECK(version >= 1), confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
                importance REAL NOT NULL CHECK(importance BETWEEN 0 AND 1), success_count INTEGER NOT NULL CHECK(success_count >= 0),
                failure_count INTEGER NOT NULL CHECK(failure_count >= 0), unknown_count INTEGER NOT NULL CHECK(unknown_count >= 0),
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_used_at TEXT, last_success_at TEXT,
                provenance_json TEXT NOT NULL, source_task_id TEXT, source_session_id TEXT, parent_procedure_id TEXT,
                required_capabilities_json TEXT NOT NULL, resource_requirements_json TEXT NOT NULL, metadata_json TEXT NOT NULL,
                UNIQUE(name, version),
                FOREIGN KEY(parent_procedure_id) REFERENCES procedures(procedure_id)
            )""")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_proc_name ON procedures(name)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_proc_status ON procedures(status)")
            self._conn.execute("""CREATE TABLE IF NOT EXISTS procedure_parameters (
                procedure_id TEXT NOT NULL REFERENCES procedures(procedure_id) ON DELETE CASCADE,
                name TEXT NOT NULL, description TEXT NOT NULL, type TEXT NOT NULL, required INTEGER NOT NULL,
                default_value_json TEXT, constraints_json TEXT NOT NULL, sensitive INTEGER NOT NULL,
                PRIMARY KEY(procedure_id,name)
            )""")
            self._conn.execute("""CREATE TABLE IF NOT EXISTS procedure_steps (
                procedure_id TEXT NOT NULL REFERENCES procedures(procedure_id) ON DELETE CASCADE,
                step_id TEXT NOT NULL, step_order INTEGER NOT NULL CHECK(step_order >= 0), action_name TEXT NOT NULL,
                arguments_json TEXT NOT NULL, description TEXT NOT NULL, dependencies_json TEXT NOT NULL,
                required_capabilities_json TEXT NOT NULL, expected_observation_json TEXT NOT NULL,
                expected_verification_json TEXT NOT NULL, metadata_json TEXT NOT NULL,
                PRIMARY KEY(procedure_id,step_id), UNIQUE(procedure_id,step_order)
            )""")
            self._conn.execute("""CREATE TABLE IF NOT EXISTS procedure_preconditions (
                procedure_id TEXT NOT NULL REFERENCES procedures(procedure_id) ON DELETE CASCADE,
                condition_id TEXT NOT NULL, condition TEXT NOT NULL, metadata_json TEXT NOT NULL,
                PRIMARY KEY(procedure_id,condition_id)
            )""")
            self._conn.execute("""CREATE TABLE IF NOT EXISTS procedure_postconditions (
                procedure_id TEXT NOT NULL REFERENCES procedures(procedure_id) ON DELETE CASCADE,
                condition_id TEXT NOT NULL, condition TEXT NOT NULL, metadata_json TEXT NOT NULL,
                PRIMARY KEY(procedure_id,condition_id)
            )""")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ProcedureStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def create_procedure(self, procedure: LearnedProcedure) -> LearnedProcedure:
        procedure.validate()
        if procedure.status is ProcedureStatus.ACTIVE:
            raise ValueError("new procedures must not be inserted ACTIVE")
        if procedure.parent_procedure_id and not self.get_procedure(procedure.parent_procedure_id):
            raise ValueError("parent procedure does not exist")
        with self._conn:
            self._insert_procedure(procedure)
        return self.get_procedure(procedure.procedure_id)  # type: ignore[return-value]

    def _insert_procedure(self, p: LearnedProcedure) -> None:
        p.validate()
        self._conn.execute("""INSERT INTO procedures VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (p.procedure_id,p.name,p.description,p.trigger_pattern,p.status.value,p.version,p.confidence,p.importance,
             p.success_count,p.failure_count,p.unknown_count,p.created_at,p.updated_at,p.last_used_at,p.last_success_at,
             _json({"source_type":p.provenance.source_type.value,"source_task_id":p.provenance.source_task_id,"source_session_id":p.provenance.source_session_id,"created_at":p.provenance.created_at,"parent_procedure_id":p.provenance.parent_procedure_id,"source_version":p.provenance.source_version}),
             p.source_task_id,p.source_session_id,p.parent_procedure_id,_json(list(p.required_capabilities)),_json(p.resource_requirements),_json(p.metadata)))
        for x in p.parameters:
            self._conn.execute("INSERT INTO procedure_parameters VALUES (?,?,?,?,?,?,?,?)",
                (p.procedure_id,x.name,x.description,x.type,int(x.required),_json(x.default) if x.default is not None else None,_json(x.constraints),int(x.sensitive)))
        for x in p.steps:
            self._conn.execute("INSERT INTO procedure_steps VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (p.procedure_id,x.step_id,x.order,x.action_name,_json(x.arguments),x.description,_json(list(x.dependencies)),_json(list(x.required_capabilities)),_json(x.expected_observation),_json(x.expected_verification),_json(x.metadata)))
        for table, xs in (("procedure_preconditions",p.preconditions),("procedure_postconditions",p.postconditions)):
            for x in xs:
                self._conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?)", (p.procedure_id,x.condition_id,x.condition,_json(x.metadata)))

    def get_procedure(self, procedure_id: str) -> LearnedProcedure | None:
        row = self._conn.execute("SELECT * FROM procedures WHERE procedure_id=?", (procedure_id,)).fetchone()
        return self._from_row(row) if row else None

    def get_procedure_version(self, name: str, version: int) -> LearnedProcedure | None:
        row = self._conn.execute("SELECT * FROM procedures WHERE name=? AND version=?", (name, version)).fetchone()
        return self._from_row(row) if row else None

    def list_procedures(self, *, status: ProcedureStatus | None = None, name: str | None = None, limit: int = 100) -> list[LearnedProcedure]:
        limit = max(1, min(int(limit), 500))
        clauses, params = [], []
        if status is not None: clauses.append("status=?"); params.append(status.value)
        if name is not None: clauses.append("name=?"); params.append(name)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute("SELECT * FROM procedures" + where + " ORDER BY name, version DESC LIMIT ?", (*params,limit)).fetchall()
        return [self._from_row(r) for r in rows]

    def update_procedure(self, procedure: LearnedProcedure) -> LearnedProcedure:
        procedure.validate()
        current = self.get_procedure(procedure.procedure_id)
        if current is None: raise KeyError(procedure.procedure_id)
        if current.version != procedure.version or current.name != procedure.name:
            raise ValueError("procedure identity/version cannot be changed; create a new version")
        if current.status != procedure.status:
            raise ValueError("use set_status for lifecycle changes")
        now = _now()
        updated = LearnedProcedure(**{**procedure.__dict__, "created_at": current.created_at, "updated_at": now})
        with self._conn:
            self._conn.execute("""UPDATE procedures SET description=?, trigger_pattern=?, confidence=?, importance=?,
                success_count=?, failure_count=?, unknown_count=?, updated_at=?, last_used_at=?, last_success_at=?,
                provenance_json=?, source_task_id=?, source_session_id=?, parent_procedure_id=?,
                required_capabilities_json=?, resource_requirements_json=?, metadata_json=? WHERE procedure_id=?""",
                (updated.description, updated.trigger_pattern, updated.confidence, updated.importance,
                 updated.success_count, updated.failure_count, updated.unknown_count, updated.updated_at,
                 updated.last_used_at, updated.last_success_at,
                 _json({"source_type":updated.provenance.source_type.value,"source_task_id":updated.provenance.source_task_id,
                        "source_session_id":updated.provenance.source_session_id,"created_at":updated.provenance.created_at,
                        "parent_procedure_id":updated.provenance.parent_procedure_id,"source_version":updated.provenance.source_version}),
                 updated.source_task_id, updated.source_session_id, updated.parent_procedure_id,
                 _json(list(updated.required_capabilities)), _json(updated.resource_requirements), _json(updated.metadata),
                 updated.procedure_id))
            self._conn.execute("DELETE FROM procedure_parameters WHERE procedure_id=?", (procedure.procedure_id,))
            self._conn.execute("DELETE FROM procedure_steps WHERE procedure_id=?", (procedure.procedure_id,))
            self._conn.execute("DELETE FROM procedure_preconditions WHERE procedure_id=?", (procedure.procedure_id,))
            self._conn.execute("DELETE FROM procedure_postconditions WHERE procedure_id=?", (procedure.procedure_id,))
            for x in updated.parameters:
                self._conn.execute("INSERT INTO procedure_parameters VALUES (?,?,?,?,?,?,?,?)",
                    (updated.procedure_id,x.name,x.description,x.type,int(x.required),_json(x.default) if x.default is not None else None,_json(x.constraints),int(x.sensitive)))
            for x in updated.steps:
                self._conn.execute("INSERT INTO procedure_steps VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (updated.procedure_id,x.step_id,x.order,x.action_name,_json(x.arguments),x.description,_json(list(x.dependencies)),_json(list(x.required_capabilities)),_json(x.expected_observation),_json(x.expected_verification),_json(x.metadata)))
            for table, xs in (("procedure_preconditions",updated.preconditions),("procedure_postconditions",updated.postconditions)):
                for x in xs:
                    self._conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?)", (updated.procedure_id,x.condition_id,x.condition,_json(x.metadata)))
        return self.get_procedure(procedure.procedure_id)  # type: ignore[return-value]

    def set_status(self, procedure_id: str, target: ProcedureStatus) -> LearnedProcedure:
        current = self.get_procedure(procedure_id)
        if current is None: raise KeyError(procedure_id)
        if target not in _ALLOWED_TRANSITIONS[current.status]:
            raise ValueError(f"invalid procedure transition {current.status.value} -> {target.value}")
        with self._conn:
            self._conn.execute("UPDATE procedures SET status=?, updated_at=? WHERE procedure_id=?", (target.value,_now(),procedure_id))
        return self.get_procedure(procedure_id)  # type: ignore[return-value]

    def _record(self, procedure_id: str, column: str, *, success: bool = False) -> LearnedProcedure:
        if column not in {"success_count","failure_count","unknown_count"}: raise ValueError("invalid counter")
        current = self.get_procedure(procedure_id)
        if current is None: raise KeyError(procedure_id)
        now = _now()
        last_success = now if success else current.last_success_at
        with self._conn:
            self._conn.execute(f"UPDATE procedures SET {column}={column}+1, updated_at=?, last_used_at=?, last_success_at=? WHERE procedure_id=?", (now,now,last_success,procedure_id))
        return self.get_procedure(procedure_id)  # type: ignore[return-value]

    def record_success(self, procedure_id: str) -> LearnedProcedure: return self._record(procedure_id,"success_count",success=True)
    def record_failure(self, procedure_id: str) -> LearnedProcedure: return self._record(procedure_id,"failure_count")
    def record_unknown(self, procedure_id: str) -> LearnedProcedure: return self._record(procedure_id,"unknown_count")

    def create_new_version(self, procedure: LearnedProcedure, *, version: int) -> LearnedProcedure:
        procedure.validate()
        if version <= 1: raise ValueError("new version must be greater than 1")
        parent = self.get_procedure(procedure.parent_procedure_id or "")
        if parent is None: raise ValueError("new version requires an existing parent_procedure_id")
        if version <= parent.version: raise ValueError("new version must be greater than parent version")
        if procedure.version != version: raise ValueError("procedure.version must equal requested version")
        if procedure.name != parent.name: raise ValueError("version name must match parent")
        if procedure.status is ProcedureStatus.ACTIVE: raise ValueError("new versions must not be inserted ACTIVE")
        with self._conn:
            self._insert_procedure(procedure)
        return self.get_procedure(procedure.procedure_id)  # type: ignore[return-value]

    def invalidate_procedure(self, procedure_id: str) -> LearnedProcedure:
        return self.set_status(procedure_id, ProcedureStatus.INVALIDATED)

    def _from_row(self, row: sqlite3.Row) -> LearnedProcedure:
        pid = row["procedure_id"]
        params = self._conn.execute("SELECT * FROM procedure_parameters WHERE procedure_id=? ORDER BY name", (pid,)).fetchall()
        steps = self._conn.execute("SELECT * FROM procedure_steps WHERE procedure_id=? ORDER BY step_order", (pid,)).fetchall()
        pre = self._conn.execute("SELECT * FROM procedure_preconditions WHERE procedure_id=? ORDER BY condition_id", (pid,)).fetchall()
        post = self._conn.execute("SELECT * FROM procedure_postconditions WHERE procedure_id=? ORDER BY condition_id", (pid,)).fetchall()
        prov = json.loads(row["provenance_json"])
        p = LearnedProcedure(
            procedure_id=pid,name=row["name"],description=row["description"],trigger_pattern=row["trigger_pattern"],status=ProcedureStatus(row["status"]),version=row["version"],
            parameters=tuple(ProcedureParameter(r["name"],r["description"],r["type"],bool(r["required"]),json.loads(r["default_value_json"]) if r["default_value_json"] else None,json.loads(r["constraints_json"]),bool(r["sensitive"])) for r in params),
            steps=tuple(ProcedureStep(r["step_id"],r["step_order"],r["action_name"],json.loads(r["arguments_json"]),r["description"],tuple(json.loads(r["dependencies_json"])),tuple(json.loads(r["required_capabilities_json"])),json.loads(r["expected_observation_json"]),json.loads(r["expected_verification_json"]),json.loads(r["metadata_json"])) for r in steps),
            preconditions=tuple(ProcedureCondition(r["condition_id"],r["condition"],json.loads(r["metadata_json"])) for r in pre),
            postconditions=tuple(ProcedureCondition(r["condition_id"],r["condition"],json.loads(r["metadata_json"])) for r in post),
            required_capabilities=tuple(json.loads(row["required_capabilities_json"])), resource_requirements=json.loads(row["resource_requirements_json"]),success_count=row["success_count"],failure_count=row["failure_count"],unknown_count=row["unknown_count"],confidence=row["confidence"],importance=row["importance"],created_at=row["created_at"],updated_at=row["updated_at"],last_used_at=row["last_used_at"],last_success_at=row["last_success_at"],provenance=ProcedureProvenance(ProcedureSourceType(prov["source_type"]),prov.get("source_task_id"),prov.get("source_session_id"),prov.get("created_at",row["created_at"]),prov.get("parent_procedure_id"),prov.get("source_version")),source_task_id=row["source_task_id"],source_session_id=row["source_session_id"],parent_procedure_id=row["parent_procedure_id"],metadata=json.loads(row["metadata_json"]),
        )
        return p


__all__ = ["ProcedureStatus", "ProcedureSourceType", "ProcedureParameter", "ProcedureStep", "ProcedureCondition", "ProcedureProvenance", "LearnedProcedure", "ProcedureStore", "DEFAULT_STORE"]
