"""Atomic semantic workflow decomposition.

The decomposer is deliberately narrow: it normalizes only action shapes that
DEIMOS already knows how to execute. It never executes, schedules, or verifies.
If a compound request contains a recognizable clause that cannot be normalized,
it fails closed instead of silently dropping that clause.
"""
from __future__ import annotations

import re
import time
from typing import Any

from ..planner.base import Planner
from ..types import Action
from .models import Workflow
from .scheduler import DependencyScheduler


_ACTION_START = re.compile(
    r"(?i)(?<!\w)(?:open|launch|start|type|write|play|send|click|search|close|refresh|go\s+to)\b"
)
_OPEN_APP = re.compile(r"(?i)^(?:open|launch|start)\s+(?:the\s+)?(.+?)\s*$")
_TYPE_TEXT = re.compile(r"(?i)^(?:type|write)\s+(?:text\s+)?(.+?)\s+(?:in|into|inside)\s+(?:the\s+)?(.+?)\s*$")
_TYPE_TEXT_BARE = re.compile(r"(?i)^(?:type|write)\s+(.+?)\s*$")
_SEARCH = re.compile(r"(?i)^(?:search|google|look\s+up)\s+(?:for\s+)?(.+?)\s*$")
_PLAY = re.compile(r"(?i)^(?:open\s+youtube\s+and\s+)?play\s+(.+?)\s*$")


def _clean(value: str) -> str:
    return value.strip().strip('"\'.,!?;:')


def _split_clauses(goal: str) -> tuple[list[str], bool]:
    """Split only at conjunctions that introduce another known action.

    This avoids treating every occurrence of ``and`` as an edge. A song title
    such as ``Do I Wanna Know`` therefore stays intact.
    """
    text = " ".join(str(goal or "").strip().split())
    if not text:
        return [], False
    parts: list[str] = []
    explicit_order = False
    cursor = 0
    pattern = re.compile(r"(?i)\s+(?:,\s*)?(then|and)\s+")
    for match in pattern.finditer(text):
        clause = text[cursor:match.start()].strip(" ,")
        if clause:
            parts.append(clause)
        explicit_order = explicit_order or match.group(1).casefold() == "then"
        cursor = match.end()
    tail = text[cursor:].strip(" ,")
    if tail:
        parts.append(tail)

    # Split comma-separated clauses only when the following token starts a
    # known semantic action. Commas inside messages and song titles remain intact.
    expanded: list[str] = []
    comma_action = re.compile(r"(?i),\s*(?=(?:open|launch|start|type|write|play|send|click|search|close|refresh|go\s+to)\b)")
    for part in parts:
        pieces = [p.strip() for p in comma_action.split(part) if p.strip()]
        expanded.extend(pieces or [part])
    parts = expanded
    return parts, explicit_order


def _normalize_clause(clause: str) -> Action | None:
    clause = clause.strip()
    # Voice transcripts may include the wake word; it is not an action.
    clause = re.sub(r"(?i)^(?:jarvis)[,\s:]+", "", clause).strip()
    try:
        from ..planner.openai_compat import _parse_whatsapp_send_goal
        whatsapp = _parse_whatsapp_send_goal(clause)
    except Exception:
        whatsapp = None
    if whatsapp is not None:
        recipient, message = whatsapp
        return Action("whatsapp_send_message", {"recipient": recipient, "message": message}, rationale="atomic WhatsApp send clause")
    m = _OPEN_APP.fullmatch(clause)
    if m:
        app = _clean(m.group(1))
        if app and app.casefold() not in {"youtube", "youtube.com"}:
            return Action("launch_app", {"app": app}, rationale="atomic open application clause")

    m = _TYPE_TEXT.fullmatch(clause)
    if m:
        text = _clean(m.group(1))
        target = _clean(m.group(2))
        if text and target:
            # A registered application is a desktop target; anything else is
            # preserved as a semantic UI target for the existing browser/semantic
            # control layer. This is intentionally generic: no control name is
            # special-cased here.
            from ..os_tools import APP_REGISTRY, _normalize_app_name
            if _normalize_app_name(target) in APP_REGISTRY:
                return Action("type_text", {"app": target, "text": text}, rationale="atomic text-entry clause")
            return Action(
                "browser_type",
                {"target_query": target, "text": text, "target_semantic": {"name": target, "role": "textbox"}},
                rationale="atomic semantic text-entry clause",
            )

    m = _SEARCH.fullmatch(clause)
    if m:
        query = _clean(m.group(1))
        if query:
            return Action("browser_search", {"query": query}, rationale="atomic semantic browser search clause")

    # If the target app was not repeated, Workflow.from_actions can bind this
    # to the nearest compatible open-app predecessor.
    m = _TYPE_TEXT_BARE.fullmatch(clause)
    if m:
        text = _clean(m.group(1))
        if text:
            return Action("type_text", {"text": text}, rationale="atomic text-entry clause")

    m = _PLAY.fullmatch(clause)
    if m:
        query = _clean(m.group(1))
        if query:
            return Action("browser_play_song", {"query": query}, rationale="atomic YouTube playback clause")

    return None


def _deterministic_atomic(goal: str) -> tuple[list[Action], bool] | None:
    clauses, explicit_order = _split_clauses(goal)
    if len(clauses) < 2:
        return None
    actions: list[Action] = []
    for clause in clauses:
        action = _normalize_clause(clause)
        if action is None:
            return None
        actions.append(action)
    return actions, explicit_order


def _covers_required_actions(goal: str, actions: list[Action]) -> bool:
    """Fail closed when a planner loses a recognizable executable clause."""
    clauses, _ = _split_clauses(goal)
    if len(clauses) < 2:
        return True
    normalized = [_normalize_clause(c) for c in clauses]
    if len(actions) < len(clauses):
        return False
    if any(a is None for a in normalized):
        return True  # ambiguous input remains the planner's responsibility
    if len(actions) != len(normalized):
        return False
    return all(
        actual.kind == expected.kind
        for actual, expected in zip(actions, normalized)
    )


class WorkflowDecomposer:
    """Normalize a compound request into atomic actions before DAG inference."""

    def __init__(self, planner: Planner) -> None:
        self.planner = planner

    def decompose(self, workflow_id: str, goal: str, *, state: dict[str, Any] | None = None) -> Workflow:
        started = time.perf_counter()
        deterministic = _deterministic_atomic(goal)
        if deterministic is not None:
            actions, explicit_order = deterministic
            workflow = Workflow.from_actions(workflow_id, goal, actions, explicit_order=explicit_order)
            DependencyScheduler.validate(workflow)
            workflow.history.append({"decomposition_ms": round((time.perf_counter() - started) * 1000, 3), "mode": "deterministic_atomic"})
            return workflow

        planned = self.planner.plan(goal, state or {}, [])
        if planned.error:
            raise RuntimeError(f"decomposition_failed: {planned.error}")
        if not planned.actions:
            raise RuntimeError("decomposition_failed: planner produced no executable actions")
        if not _covers_required_actions(goal, list(planned.actions)):
            raise RuntimeError("decomposition_failed: planner dropped one or more recognizable compound actions")
        workflow = Workflow.from_actions(workflow_id, goal, list(planned.actions))
        DependencyScheduler.validate(workflow)
        workflow.history.append({"decomposition_ms": round((time.perf_counter() - started) * 1000, 3), "mode": "planner"})
        return workflow
