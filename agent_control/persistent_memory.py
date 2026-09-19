"""Small, structured persistent memory for DEIMOS (P3.1).

This module is intentionally separate from :mod:`agent_control.memory`, which is
and remains the file-location index.  Persistent agent memory is a local SQLite
store of compact, provenance-bearing facts and task episodes.  It is context only:
callers must still use live observation, policy, and verification as authority.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORE = ROOT / ".agent_memory" / "agent_memory.sqlite3"
SCHEMA_VERSION = 1
DEFAULT_RECALL_LIMIT = 8
DEFAULT_MAX_EPISODES = 500

# Deliberately conservative. These patterns prevent the most common credential
# forms from becoming durable memory; they are not intended as a DLP system.
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:api[_ -]?key|access[_ -]?token|auth[_ -]?token|secret|password|passwd|cookie|session[_ -]?token)\b\s*[:=]\s*[^\s,;]+"),
    re.compile(r"\b(?:sk|ghp|github_pat|xox[baprs])-[-_A-Za-z0-9]{12,}\b", re.I),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]{12,}"),
)
_SECRET_WORDS = re.compile(r"(?i)\b(?:password|passwd|api[_ -]?key|access[_ -]?token|auth[_ -]?token|session[_ -]?token|secret|cookie)\b")

@dataclass(frozen=True)
class MemoryRecord:
    id: str
    memory_type: str
    content: str
    source: str
    created_at: float
    updated_at: float
    last_accessed_at: float | None
    confidence: float
    importance: float
    status: str
    scope: str
    session_id: str | None = None
    task_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    expires_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "memory_type": self.memory_type, "content": self.content,
            "source": self.source, "created_at": self.created_at,
            "updated_at": self.updated_at, "last_accessed_at": self.last_accessed_at,
            "confidence": self.confidence, "importance": self.importance,
            "status": self.status, "scope": self.scope,
            "session_id": self.session_id, "task_id": self.task_id,
            "provenance": dict(self.provenance), "evidence": dict(self.evidence),
            "expires_at": self.expires_at,
        }


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").casefold())


def _safe_content(text: str) -> str | None:
    text = " ".join(str(text or "").split()).strip()
    if not text or len(text) > 1000:
        return None
    if _SECRET_WORDS.search(text) and any(pattern.search(text) for pattern in _SECRET_PATTERNS):
        return None
    for pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            return None
    return text


def _memory_id(memory_type: str, content: str, *, key: str | None = None) -> str:
    basis = f"{memory_type.casefold()}\0{content.casefold().strip()}"
    return "mem-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:20]


class PersistentMemory:
    """Local SQLite store for durable facts and compact episodes."""

    def __init__(self, path: str | Path = DEFAULT_STORE, *, max_episodes: int = DEFAULT_MAX_EPISODES) -> None:
        configured = os.getenv("DEIMOS_AGENT_MEMORY", "").strip()
        self.path = Path(configured or path).expanduser().resolve()
        self.max_episodes = max(50, int(max_episodes))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("""CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_accessed_at REAL,
                confidence REAL NOT NULL,
                importance REAL NOT NULL,
                status TEXT NOT NULL,
                scope TEXT NOT NULL,
                session_id TEXT,
                task_id TEXT,
                provenance_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                expires_at REAL
            )""")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(memory_type, status)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_updated ON memories(updated_at DESC)")
            self._conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id UNINDEXED, content, memory_type, scope
            )""")
            self._conn.execute("""CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(id, content, memory_type, scope)
                VALUES (new.id, new.content, new.memory_type, new.scope);
            END""")
            self._conn.execute("""CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content, memory_type, scope ON memories BEGIN
                DELETE FROM memories_fts WHERE id = old.id;
                INSERT INTO memories_fts(id, content, memory_type, scope)
                VALUES (new.id, new.content, new.memory_type, new.scope);
            END""")
            self._conn.execute("""CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                DELETE FROM memories_fts WHERE id = old.id;
            END""")
            # Repair FTS once for stores created by older/dev builds.
            self._conn.execute("DELETE FROM memories_fts WHERE id NOT IN (SELECT id FROM memories)")
            self._conn.execute("""INSERT OR IGNORE INTO memories_fts(id, content, memory_type, scope)
                SELECT id, content, memory_type, scope FROM memories""")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PersistentMemory":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _row(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"], memory_type=row["memory_type"], content=row["content"], source=row["source"],
            created_at=float(row["created_at"]), updated_at=float(row["updated_at"]),
            last_accessed_at=float(row["last_accessed_at"]) if row["last_accessed_at"] is not None else None,
            confidence=float(row["confidence"]), importance=float(row["importance"]),
            status=row["status"], scope=row["scope"], session_id=row["session_id"], task_id=row["task_id"],
            provenance=json.loads(row["provenance_json"] or "{}"), evidence=json.loads(row["evidence_json"] or "{}"),
            expires_at=float(row["expires_at"]) if row["expires_at"] is not None else None,
        )

    def put(self, *, memory_type: str, content: str, source: str, confidence: float,
            importance: float = 0.5, scope: str = "persistent", session_id: str | None = None,
            task_id: str | None = None, provenance: dict[str, Any] | None = None,
            evidence: dict[str, Any] | None = None, key: str | None = None,
            expires_at: float | None = None) -> MemoryRecord | None:
        safe = _safe_content(content)
        if safe is None:
            return None
        now = time.time()
        memory_id = _memory_id(memory_type, safe, key=key)
        old = self.get(memory_id)
        status = "ACTIVE"
        effective_provenance = dict(provenance or {})
        if key:
            effective_provenance.setdefault("memory_key", key)
        with self._conn:
            # A stable semantic key lets explicit corrections replace an older
            # fact without erasing its provenance/history.
            if key:
                old_rows = self._conn.execute(
                    "SELECT id FROM memories WHERE status='ACTIVE' AND memory_type=? AND json_extract(provenance_json, '$.memory_key')=?",
                    (memory_type, key),
                ).fetchall()
                for row in old_rows:
                    if row["id"] != memory_id and str(self.get(row["id"]).content if self.get(row["id"]) else "") != safe:
                        self._conn.execute("UPDATE memories SET status='SUPERSEDED', updated_at=? WHERE id=?", (now, row["id"]))
            self._conn.execute("""INSERT INTO memories(
                id,memory_type,content,source,created_at,updated_at,last_accessed_at,
                confidence,importance,status,scope,session_id,task_id,provenance_json,evidence_json,expires_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                  content=excluded.content, source=excluded.source, updated_at=excluded.updated_at,
                  confidence=MAX(memories.confidence, excluded.confidence),
                  importance=MAX(memories.importance, excluded.importance), status='ACTIVE',
                  scope=excluded.scope, session_id=excluded.session_id, task_id=excluded.task_id,
                  provenance_json=excluded.provenance_json, evidence_json=excluded.evidence_json,
                  expires_at=excluded.expires_at""",
                (memory_id, memory_type, safe, source, old.created_at if old else now, now,
                 old.last_accessed_at if old else None, max(0.0, min(1.0, confidence)),
                 max(0.0, min(1.0, importance)), status, scope, session_id, task_id,
                 json.dumps(effective_provenance, ensure_ascii=False, sort_keys=True),
                 json.dumps(evidence or {}, ensure_ascii=False, sort_keys=True), expires_at))
        return self.get(memory_id)

    def supersede(self, old_id: str, *, replacement_id: str) -> bool:
        now = time.time()
        with self._conn:
            cur = self._conn.execute("UPDATE memories SET status='SUPERSEDED', updated_at=? WHERE id=? AND status='ACTIVE'", (now, old_id))
        return cur.rowcount > 0

    def invalidate(self, memory_id: str) -> bool:
        with self._conn:
            cur = self._conn.execute("UPDATE memories SET status='INVALIDATED', updated_at=? WHERE id=? AND status='ACTIVE'", (time.time(), memory_id))
        return cur.rowcount > 0

    def invalidate_matching(self, query: str) -> int:
        matches = self.search(query, limit=50, include_superseded=False)
        count = 0
        with self._conn:
            for item in matches:
                cur = self._conn.execute("UPDATE memories SET status='INVALIDATED', updated_at=? WHERE id=? AND status='ACTIVE'", (time.time(), item.id))
                count += cur.rowcount
        return count

    def get(self, memory_id: str) -> MemoryRecord | None:
        row = self._conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        return self._row(row) if row else None

    def search(self, query: str, *, limit: int = DEFAULT_RECALL_LIMIT,
               memory_types: Iterable[str] | None = None,
               include_superseded: bool = False) -> list[MemoryRecord]:
        """Bounded lexical/entity/recency/importance retrieval."""
        limit = max(1, min(int(limit), DEFAULT_RECALL_LIMIT))
        q_tokens = set(_tokens(query))
        if not q_tokens:
            return []
        now = time.time()
        clauses = ["m.status IN ('ACTIVE')"] if not include_superseded else ["m.status IN ('ACTIVE','SUPERSEDED')"]
        params: list[Any] = []
        if memory_types:
            types = tuple(str(x) for x in memory_types)
            clauses.append("m.memory_type IN (%s)" % ",".join("?" for _ in types))
            params.extend(types)
        # FTS narrows the candidate set; the final score is deterministic Python
        # scoring so exact entities and metadata remain easy to inspect.
        fts_query = " OR ".join(re.findall(r"[A-Za-z0-9_]+", query or ""))
        rows: list[sqlite3.Row]
        try:
            if fts_query:
                rows = self._conn.execute(
                    "SELECT m.* FROM memories m JOIN memories_fts f ON f.id=m.id WHERE " + " AND ".join(clauses) + " AND f MATCH ? LIMIT 64",
                    (*params, fts_query),
                ).fetchall()
            else:
                rows = []
        except sqlite3.OperationalError:
            rows = self._conn.execute("SELECT * FROM memories m WHERE " + " AND ".join(clauses) + " ORDER BY updated_at DESC LIMIT 64", params).fetchall()
        scored: list[tuple[float, MemoryRecord]] = []
        for row in rows:
            item = self._row(row)
            if item.expires_at is not None and item.expires_at < now:
                continue
            tokens = set(_tokens(item.content))
            overlap = len(q_tokens & tokens)
            if overlap == 0:
                continue
            score = overlap * 5.0
            if q_tokens <= tokens:
                score += 6.0
            score += item.importance * 3.0 + item.confidence * 2.0
            age_days = max(0.0, (now - item.updated_at) / 86400.0)
            score += max(0.0, 2.0 - age_days * 0.08)
            if item.memory_type == "CONTEXT_ENTITY" and overlap:
                score += 1.0
            if item.status == "SUPERSEDED":
                score -= 8.0
            scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], -pair[1].updated_at, pair[1].id))
        selected = [item for _, item in scored[:limit]]
        if selected:
            with self._conn:
                self._conn.executemany("UPDATE memories SET last_accessed_at=? WHERE id=?", [(now, item.id) for item in selected])
        return selected

    def recent(self, *, limit: int = 8) -> list[MemoryRecord]:
        limit = max(1, min(int(limit), 50))
        rows = self._conn.execute("SELECT * FROM memories WHERE status='ACTIVE' ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(row) for row in rows]

    def status(self) -> dict[str, Any]:
        rows = self._conn.execute("SELECT status, COUNT(*) AS n FROM memories GROUP BY status").fetchall()
        counts = {str(r["status"]): int(r["n"]) for r in rows}
        return {"store": str(self.path), "schema": SCHEMA_VERSION, "total": sum(counts.values()), "active": counts.get("ACTIVE", 0), "superseded": counts.get("SUPERSEDED", 0), "invalidated": counts.get("INVALIDATED", 0)}

    def purge_expired(self, *, now: float | None = None, limit: int = 256) -> int:
        """Physically remove semantic records whose explicit TTL has elapsed.

        TTLs are set by the producer of transient semantic knowledge. Active
        records without an expiry (for example durable project facts) are never
        removed by this maintenance operation. SQLite triggers keep FTS in sync.
        """
        cutoff = float(now if now is not None else time.time())
        limit = max(1, min(int(limit), 1000))
        rows = self._conn.execute(
            "SELECT id FROM memories WHERE status='ACTIVE' AND expires_at IS NOT NULL AND expires_at < ? ORDER BY expires_at ASC LIMIT ?",
            (cutoff, limit),
        ).fetchall()
        ids = [str(row["id"]) for row in rows]
        if not ids:
            return 0
        with self._conn:
            self._conn.executemany("DELETE FROM memories WHERE id=?", [(mid,) for mid in ids])
        return len(ids)

    def maintain(self) -> int:
        """Bound episode growth while preserving current facts/history."""
        expired = self.purge_expired()
        rows = self._conn.execute("SELECT id FROM memories WHERE memory_type IN ('TASK_EPISODE','FAILURE_EPISODE') AND status='ACTIVE' ORDER BY updated_at DESC").fetchall()
        if len(rows) <= self.max_episodes:
            return expired
        ids = [str(row["id"]) for row in rows[self.max_episodes:]]
        with self._conn:
            self._conn.executemany("UPDATE memories SET status='EXPIRED', updated_at=? WHERE id=?", [(time.time(), mid) for mid in ids])
        return expired + len(ids)


class MemoryExtractor:
    """Deterministic P3.1 extractor for explicit facts/preferences and outcomes."""
    _remember = re.compile(r"(?i)^\s*(?:please\s+)?remember(?:\s+that)?\s+(.+?)\s*[.!?]*\s*$")
    _preference = re.compile(r"(?i)^\s*(?:i|we)\s+(?:prefer|like|love|use|always use)\s+(.+?)\s*[.!?]*\s*$")
    _project = re.compile(r"(?i)^\s*(?:i|we)\s+(?:always\s+)?(?:use|keep|work on|build)\s+(.+?)\s+for\s+deimos\s*[.!?]*\s*$")
    _fact = re.compile(r"(?i)^\s*(?:i|we)\s+(?:am|are|work on|build|use)\s+(.+?)\s*[.!?]*\s*$")
    _moved_project = re.compile(r"(?i)^\s*(?:my|our)\s+deimos\s+project\s+(?:moved|is now)\s+(?:to\s+)?(.+?)\s*[.!?]*\s*$")
    _dont = re.compile(r"(?i)^\s*(?:don['’]?t|do not)\s+remember\s+(.+?)\s*[.!?]*\s*$")

    @classmethod
    def extract_user(cls, text: str, *, session_id: str | None = None) -> list[dict[str, Any]]:
        raw = " ".join(str(text or "").split()).strip()
        if not raw:
            return []
        if cls._dont.match(raw):
            return [{"op": "invalidate", "query": cls._dont.match(raw).group(1)}]
        match = cls._moved_project.match(raw)
        if match:
            return cls._fact_payload(f"DEIMOS project is at {match.group(1)}", source="user_explicit", memory_type="ENVIRONMENT_FACT", session_id=session_id, key="deimos-project-location")
        match = cls._remember.match(raw)
        if match:
            remembered = match.group(1).strip()
            if re.search(r"(?i)\b(?:i|we)\s+(?:prefer|like|love|always use)\b", remembered):
                return cls._fact_payload(remembered, source="user_explicit", memory_type="USER_PREFERENCE", session_id=session_id, key="preference")
            if re.search(r"(?i)\bdeimos\s+project\b", remembered) and re.search(r"(?i)(?:d:|[a-z]:|\\)", remembered):
                return cls._fact_payload(remembered, source="user_explicit", memory_type="ENVIRONMENT_FACT", session_id=session_id, key="deimos-project-location")
            return cls._fact_payload(remembered, source="user_explicit", session_id=session_id)
        match = cls._project.match(raw)
        if match:
            return cls._fact_payload(match.group(1) + " for DEIMOS", source="user_explicit", memory_type="ENVIRONMENT_FACT", session_id=session_id, key="deimos-project-location")
        match = cls._preference.match(raw)
        if match:
            return cls._fact_payload(match.group(1), source="user_explicit", memory_type="USER_PREFERENCE", session_id=session_id, key="preference")
        return []

    @staticmethod
    def _fact_payload(content: str, *, source: str, memory_type: str = "USER_FACT", session_id: str | None = None, key: str | None = None) -> list[dict[str, Any]]:
        return [{"op": "store", "memory_type": memory_type, "content": content, "source": source,
                 "confidence": 0.98, "importance": 0.8, "scope": "persistent", "session_id": session_id,
                 "provenance": {"kind": source, **({"memory_key": key} if key else {})}, "evidence": {"statement": content}, "key": key}]

    @staticmethod
    def outcome(*, goal: str, task_id: str, status: str, verified: str, duration: float = 0.0,
                failure: str | None = None, session_id: str | None = None) -> dict[str, Any] | None:
        if status not in {"SUCCESS", "FAILED", "UNKNOWN", "PARTIAL", "POLICY_BLOCKED"}:
            return None
        if verified == "PASS" and status == "SUCCESS":
            memory_type, label = "TASK_EPISODE", "verified"
            content = f"Task completed: {goal} (verified)."
            confidence = 0.98
        elif status in {"FAILED", "UNKNOWN", "PARTIAL"}:
            memory_type, label = "FAILURE_EPISODE", "failed"
            detail = f"; {failure}" if failure else ""
            content = f"Task did not complete: {goal} ({label}){detail}."
            confidence = 0.85 if verified in {"FAIL", "UNKNOWN"} else 0.7
        else:
            return None
        return {"op": "store", "memory_type": memory_type, "content": content,
                "source": "verified_task_outcome" if verified in {"PASS", "FAIL"} else "task_outcome",
                "confidence": confidence, "importance": 0.6, "scope": "episodic", "session_id": session_id,
                "task_id": task_id, "provenance": {"kind": label, "verified": verified},
                "evidence": {"task_id": task_id, "verified": verified, "duration_s": round(duration, 3)},
                "key": f"{task_id}:{status}"}
