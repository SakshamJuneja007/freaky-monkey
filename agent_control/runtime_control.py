"""Semantic runtime-control routing over the authoritative RuntimeManager.

This module does not execute tasks. It only identifies user intent as runtime
control and extracts optional human-facing task references. Session remains
responsible for resolving those references and applying the existing
approval/recovery machinery.
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
    APPROVE = "APPROVE"
    CLEAR_HISTORY = "CLEAR_HISTORY"


@dataclass(frozen=True)
class RuntimeControlCommand:
    kind: RuntimeControlKind
    task_id: str | None = None
    scope: str = "focused"
    task_ref: int | None = None


_TASK_ID_RE = re.compile(r"\b(?:fast|task)-[a-z0-9]+(?:-[a-z0-9]+)?\b", re.I)
_TASK_REF_RE = re.compile(r"\btask\s+(\d+)\b", re.I)

_CONTROL_VERBS = {
    "continue": RuntimeControlKind.CONTINUE,
    "resume": RuntimeControlKind.RESUME,
    "retry": RuntimeControlKind.RETRY,
    "recover": RuntimeControlKind.RETRY,
    "cancel": RuntimeControlKind.CANCEL,
    "stop": RuntimeControlKind.CANCEL,
    "delete": RuntimeControlKind.CANCEL,
    "remove": RuntimeControlKind.CANCEL,
    "approve": RuntimeControlKind.APPROVE,
}
_STATUS_TERMS = {
    "task", "tasks", "status", "pending", "waiting", "running",
    "failed", "failure", "recover", "recovery", "complete", "completed",
}


def _classify_one(normalized: str, *, task_ref: int | None = None) -> RuntimeControlCommand | None:
    tokens = set(re.findall(r"[a-z0-9_-]+", normalized))
    task_id_match = _TASK_ID_RE.search(normalized)
    task_id = task_id_match.group(0) if task_id_match else None

    verbs = [_CONTROL_VERBS[token] for token in tokens if token in _CONTROL_VERBS]
    if re.search(r"\btry\s+(?:task\s+\d+\s+)?(?:it\s+)?again+n*\b", normalized):
        verbs.append(RuntimeControlKind.RETRY)
    if re.search(r"\bcontinue\s+from\s+checkpoint\b", normalized):
        verbs.append(RuntimeControlKind.CONTINUE)
    if re.search(r"\byes\b", normalized) and (task_id or task_ref is not None):
        verbs.append(RuntimeControlKind.APPROVE)

    if verbs:
        # Explicit cancellation dominates mixed lexical matches; otherwise the
        # first semantic control verb is retained. Multi-operation sentences are
        # split by classify_runtime_controls before reaching this helper.
        if RuntimeControlKind.CANCEL in verbs:
            kind = RuntimeControlKind.CANCEL
        elif RuntimeControlKind.APPROVE in verbs:
            kind = RuntimeControlKind.APPROVE
        else:
            kind = verbs[0]
        semantic_control_phrase = (
            any(normalized.startswith(prefix) for prefix in (
                "approve ", "cancel ", "retry ", "recover ", "resume ", "continue ", "try again"
            ))
            or bool(re.search(r"\bcontinue\s+from\s+checkpoint\b", normalized))
        )
        if len(tokens) <= 3 or task_id or task_ref is not None or tokens & _STATUS_TERMS or kind in {RuntimeControlKind.RESUME, RuntimeControlKind.CONTINUE} or semantic_control_phrase:
            scope = "explicit" if (task_id or task_ref is not None) else ("all" if {"all", "every"} & tokens else "focused")
            return RuntimeControlCommand(kind, task_id=task_id, scope=scope, task_ref=task_ref)

    if tokens & _STATUS_TERMS and ("what" in tokens or "show" in tokens or "list" in tokens or "status" in tokens or "tasks" in tokens):
        return RuntimeControlCommand(RuntimeControlKind.STATUS, task_id=task_id, scope="explicit" if (task_id or task_ref is not None) else "all", task_ref=task_ref)

    return None


def classify_runtime_controls(text: str) -> list[RuntimeControlCommand]:
    """Return explicit runtime operations, splitting only on human task refs.

    A sentence with ``task N`` references is runtime-control intent even when it
    contains several operations. It is intentionally conservative: without a
    task reference, ordinary prose is left to the existing single-command
    classifier.
    """
    normalized = re.sub(r"\s+", " ", str(text or "").strip().casefold())
    if not normalized:
        return []
    if normalized == "/clear-history":
        return [RuntimeControlCommand(RuntimeControlKind.CLEAR_HISTORY, scope="all")]

    # ``delete task 1 and task 2`` (and the equivalent ``tasks 1 and 2`` /
    # ``task 1 task 2`` forms) is one explicit control intent with multiple
    # presentation references.  Expand it into independent commands now so
    # Session can resolve every number against the same authoritative snapshot
    # before mutating any task.
    delete_match = re.match(
        r"^(?:delete|remove|cancel|stop)\s+(?:tasks?\s+)?(.+?)\s*$",
        normalized,
    )
    if delete_match:
        refs = [int(value) for value in re.findall(r"\b(?:task\s*)?(\d+)\b", delete_match.group(1))]
        if len(refs) >= 2:
            return [
                RuntimeControlCommand(RuntimeControlKind.CANCEL, scope="explicit", task_ref=ref)
                for ref in refs
            ]
    if not _TASK_REF_RE.search(normalized):
        command = _classify_one(normalized)
        return [command] if command is not None else []

    # Split explicit multi-operation clauses before task-reference parsing.
    # This keeps phrases such as ``approve task 1 and continue from checkpoint
    # the task 5`` as two independent runtime operations instead of letting
    # the first clause swallow the second verb.
    clause_matches = list(re.finditer(
        r"\s+(?:and|then)\s+(?=(?:approve|yes|continue|resume|retry|recover|cancel|stop|try)\b)",
        normalized,
    ))
    if clause_matches:
        commands: list[RuntimeControlCommand] = []
        starts = [0] + [match.end() for match in clause_matches]
        ends = [match.start() for match in clause_matches] + [len(normalized)]
        for start, end in zip(starts, ends):
            clause = normalized[start:end].strip(" ,;.-")
            commands.extend(classify_runtime_controls(clause))
        if commands:
            return commands

    matches = list(_TASK_REF_RE.finditer(normalized))
    commands: list[RuntimeControlCommand] = []
    for index, match in enumerate(matches):
        if index == 0:
            start = 0
            end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        else:
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        segment = normalized[start:end].strip(" ,;.-")
        command = _classify_one(segment, task_ref=int(match.group(1)))
        if command is None:
            # The task reference itself is enough to keep this out of
            # conversation, but it is not enough to execute anything safely.
            return [RuntimeControlCommand(RuntimeControlKind.STATUS, scope="explicit", task_ref=int(match.group(1)))]
        commands.append(command)
    return commands


def classify_runtime_control(text: str) -> RuntimeControlCommand | None:
    """Classify one runtime-control command; preserve the legacy API."""
    commands = classify_runtime_controls(text)
    return commands[0] if len(commands) == 1 else None
