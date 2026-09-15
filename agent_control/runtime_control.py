"""Semantic runtime-control routing over the authoritative RuntimeManager.

This module does not execute tasks. It only identifies user intent as runtime
control and extracts an optional explicit task id. Session remains responsible
for applying the existing workflow/recovery machinery after ownership is known.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class RuntimeControlKind(str, Enum):
    STATUS = "STATUS"
    CONTINUE = "CONTINUE"
    RESUME = "RESUME"
    CANCEL = "CANCEL"
    RETRY = "RETRY"


@dataclass(frozen=True)
class RuntimeControlCommand:
    kind: RuntimeControlKind
    task_id: str | None = None
    scope: str = "focused"


_TASK_ID_RE = re.compile(r"\b(?:fast|task)-[a-z0-9]+(?:-[a-z0-9]+)?\b", re.I)

_CONTROL_VERBS = {
    "continue": RuntimeControlKind.CONTINUE,
    "resume": RuntimeControlKind.RESUME,
    "retry": RuntimeControlKind.RETRY,
    "cancel": RuntimeControlKind.CANCEL,
    "stop": RuntimeControlKind.CANCEL,
}
_STATUS_TERMS = {
    "task", "tasks", "status", "pending", "waiting", "running",
    "failed", "failure", "recover", "recovery", "complete", "completed",
}


def classify_runtime_control(text: str) -> RuntimeControlCommand | None:
    """Classify runtime-control language without matching whole phrases.

    The classifier is deliberately a small lexical semantic layer: control verbs
    plus runtime-state nouns are enough to distinguish runtime commands from
    ordinary prose without a growing list of exact strings. Explicit task ids
    always bind the command to that exact runtime record.
    """
    normalized = re.sub(r"\s+", " ", str(text or "").strip().casefold())
    if not normalized:
        return None
    tokens = set(re.findall(r"[a-z0-9_-]+", normalized))
    task_id_match = _TASK_ID_RE.search(normalized)
    task_id = task_id_match.group(0) if task_id_match else None

    verbs = [_CONTROL_VERBS[token] for token in tokens if token in _CONTROL_VERBS]
    if verbs:
        kind = RuntimeControlKind.CANCEL if RuntimeControlKind.CANCEL in verbs else verbs[0]
        # A bare control verb is a runtime command; with runtime nouns it is even
        # more explicit. This intentionally avoids phrase-specific approvals.
        if len(tokens) <= 3 or task_id or tokens & _STATUS_TERMS:
            scope = "explicit" if task_id else ("all" if {"all", "every"} & tokens else "focused")
            return RuntimeControlCommand(kind, task_id=task_id, scope=scope)

    if tokens & _STATUS_TERMS and ("what" in tokens or "show" in tokens or "list" in tokens or "status" in tokens or "tasks" in tokens):
        return RuntimeControlCommand(RuntimeControlKind.STATUS, task_id=task_id, scope="explicit" if task_id else "all")

    return None
