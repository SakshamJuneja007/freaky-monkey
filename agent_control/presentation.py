"""User-facing presentation boundaries for DEIMOS runtime data.

Runtime state is intentionally rich and developer-oriented. This module is the
small boundary between that state and anything that may be spoken to a user.
It provides structured response builders plus a final defensive sanitizer.

The sanitizer is deliberately a last line of defence, not the primary design:
callers should pass semantic ``UserFacingResponse`` text rather than serialised
runtime objects, traces, or state records.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


_INTERNAL_ID_RE = re.compile(r"\b(?:fast|input|planner|event|browser)[-_][A-Za-z0-9][A-Za-z0-9_-]*\b", re.I)
_INTERNAL_LABEL_RE = re.compile(
    r"\b(?:TASK|STATE|ROUTE|OWNER|CONVERSATION|INPUT|PLANNER|EVENT|DEBUG)\s*:\s*",
    re.I,
)
_INTERNAL_ASSIGNMENT_RE = re.compile(
    r"\b(?:planner_task|task_created|task_id|input_id|event_id|browser_resource_id)\s*=\s*[^\s,;]+",
    re.I,
)
_INTERNAL_STATE_RE = re.compile(
    r"\b(?:CREATED|PLANNING|WAITING_FOR_APPROVAL|WAITING_FOR_USER|WAITING_FOR_HUMAN|"
    r"RUNNING|VERIFYING|COMPLETED|FAILED|CANCELLED|BLOCKED|RECOVERY_REQUIRED|UNKNOWN)\b"
)
_INTERNAL_ACTION_RE = re.compile(r"\b(?:whatsapp_send_message|gmail_send_email|browser_[a-z0-9_]+)\b", re.I)
_TRACE_RE = re.compile(r"(?:Traceback \(most recent call last\)|File \"[^\"]+\", line \d+|\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception):\s*)", re.I)


@dataclass(frozen=True)
class UserFacingResponse:
    """Only this text field is eligible for TTS."""

    text: str
    allow_internal: bool = False

    def spoken_text(self, *, goal: str = "", state: str = "") -> str:
        return sanitize_tts_text(self.text, goal=goal, state=state, allow_internal=self.allow_internal)


def user_response(text: str, *, allow_internal: bool = False) -> UserFacingResponse:
    """Construct a response explicitly intended for the presentation layer."""
    return UserFacingResponse(text=str(text), allow_internal=allow_internal)


def runtime_snapshot_for_user(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Project RuntimeManager truth into semantic, non-identifier data.

    This is used as model context for conversational presentation. The live
    RuntimeManager still owns and retains the complete snapshot.
    """
    projected: dict[str, Any] = {}
    for bucket in ("active_tasks", "waiting_tasks", "completed_tasks", "failed_tasks", "cancelled_tasks"):
        items = []
        for item in snapshot.get(bucket, []) or []:
            items.append({
                "goal": str(item.get("goal") or "the requested task"),
                "state": _friendly_state(str(item.get("state") or "")),
                "verification": _friendly_verification(str(item.get("verification_state") or "")),
                "failure": str(item.get("failure_state") or ""),
            })
        projected[bucket] = items
    return projected


def _friendly_verification(value: str) -> str:
    return {
        "PASS": "verified",
        "FAIL": "not verified",
        "UNKNOWN": "not known",
    }.get(value.upper(), "")


def _friendly_state(state: str) -> str:
    return {
        "CREATED": "starting",
        "PLANNING": "planning",
        "WAITING_FOR_APPROVAL": "waiting for approval",
        "WAITING_FOR_USER": "waiting for user input",
        "WAITING_FOR_HUMAN": "waiting for user input",
        "RUNNING": "running",
        "VERIFYING": "verifying",
        "COMPLETED": "completed",
        "FAILED": "failed",
        "CANCELLED": "cancelled",
        "BLOCKED": "blocked",
        "RECOVERY_REQUIRED": "interrupted and needs recovery",
        "UNKNOWN": "unknown",
    }.get(state, "")


def approval_prompt(action: dict[str, Any] | None, goal: str = "") -> str:
    """Build a natural approval question from structured action data."""
    action = action or {}
    kind = str(action.get("kind") or "")
    params = action.get("params") if isinstance(action.get("params"), dict) else {}

    if kind == "whatsapp_send_message":
        recipient = str(params.get("recipient") or params.get("to") or "the recipient").strip()
        recipient = recipient[:1].upper() + recipient[1:] if recipient else "the recipient"
        return f"Do you want me to send the WhatsApp message to {recipient}?"
    if kind == "gmail_send_email":
        recipient = str(params.get("recipient") or params.get("to") or "the recipient").strip()
        recipient = recipient[:1].upper() + recipient[1:] if recipient else "the recipient"
        return f"Do you want me to send the email to {recipient}?"
    if goal:
        return f"Do you want me to proceed with {goal}?"
    return "Do you want me to proceed with this action?"


def recovery_message(goal: str = "", *, needs_input: bool = False) -> str:
    """Return semantic recovery language without exposing runtime state names."""
    subject = _goal_subject(goal)
    if needs_input:
        return f"The {subject} was interrupted. Would you like me to try it again?"
    return f"The {subject} was interrupted, so I need to recover it before continuing."


def completion_message(goal: str = "") -> str:
    subject = _goal_subject(goal)
    return f"The {subject} was completed."


def _goal_subject(goal: str) -> str:
    text = re.sub(r"\s+", " ", str(goal or "")).strip()
    lower = text.casefold()
    if "whatsapp" in lower or ("send" in lower and "message" in lower):
        return "WhatsApp message"
    if "gmail" in lower or "email" in lower or "mail" in lower:
        return "email"
    if "youtube" in lower or lower.startswith("play ") or "music" in lower or "song" in lower:
        return "music task"
    if "folder" in lower or "directory" in lower:
        return "folder task"
    return "requested task"


def _looks_internal(value: str) -> bool:
    if not value:
        return False
    return bool(
        _INTERNAL_ID_RE.search(value)
        or _INTERNAL_LABEL_RE.search(value)
        or _INTERNAL_ASSIGNMENT_RE.search(value)
        or _INTERNAL_STATE_RE.search(value)
        or _INTERNAL_ACTION_RE.search(value)
        or _TRACE_RE.search(value)
    )


def _coerce_text(value: Any) -> str | None:
    """Reject raw runtime objects instead of serialising them for speech."""
    if isinstance(value, UserFacingResponse):
        return value.text
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return None


def sanitize_tts_text(
    value: Any,
    *,
    goal: str = "",
    state: str = "",
    allow_internal: bool = False,
) -> str:
    """Final defensive TTS gate.

    Normal responses are semantic text. If an internal-looking string somehow
    crosses the boundary, the whole contaminated sentence is replaced with a
    clean semantic fallback rather than speaking a partially mangled sentence.
    Explicit technical requests can opt in to internal identifiers.
    """
    text = _coerce_text(value)
    if text is None:
        return "I have an internal runtime result, but there is nothing user-facing to say yet."
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if allow_internal:
        return text
    if not _looks_internal(text):
        return text

    state_upper = str(state or "").upper()
    if state_upper == "RECOVERY_REQUIRED" or "RECOVERY_REQUIRED" in text.upper():
        return recovery_message(goal)
    if state_upper in {"WAITING_FOR_APPROVAL", "WAITING_FOR_USER"}:
        if "message" in str(goal).casefold() or "whatsapp" in str(goal).casefold():
            return "The WhatsApp message is waiting for your approval."
        return "The task is waiting for your input."
    if state_upper == "COMPLETED" or "completed" in text.casefold():
        return completion_message(goal)
    if state_upper == "FAILED" or "failed" in text.casefold():
        return f"The {_goal_subject(goal)} could not be completed."
    return f"The {_goal_subject(goal)} is in progress."


def is_explicit_task_id_request(text: str) -> bool:
    lowered = re.sub(r"\s+", " ", str(text or "").strip().casefold())
    return any(
        phrase in lowered
        for phrase in (
            "what is the task id",
            "what's the task id",
            "give me the task id",
            "show me the task id",
            "task id please",
            "what is its task id",
        )
    )
