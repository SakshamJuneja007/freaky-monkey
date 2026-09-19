"""Incremental WhatsApp intelligence with a hard persisted cutoff.

This module is deliberately read-only. It consumes structured message evidence
from the existing BrowserSkill observation and persists only intelligence state
and derived events. It never sends messages, schedules work, or grants policy
permission.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORE = ROOT / ".agent_memory" / "whatsapp_intelligence.sqlite3"
_DB_LOCKS: dict[str, threading.RLock] = {}
_DB_LOCKS_GUARD = threading.Lock()
_DB_BUSY_RETRIES = 3
_DB_RETRY_SLEEP_S = 0.05
_FAILED_RETRY_LIMIT = 3
_FAILED_RETRY_BASE_S = 1.0
_FAILED_RETRY_MAX_S = 60.0
_HOME_WHEEL_DELTA_Y = 560
_HOME_SCROLL_BUDGET = 3
DEFAULT_EVENT_TIMEZONE = "Asia/Kolkata"
DEFAULT_EVENT_TZ = ZoneInfo(DEFAULT_EVENT_TIMEZONE)


class Relevance(str, Enum):
    LOW_VALUE = "LOW_VALUE"
    POSSIBLY_MEANINGFUL = "POSSIBLY_MEANINGFUL"
    HIGH_VALUE = "HIGH_VALUE"


WHATSAPP_EVENT_LOOKBACK_DAYS = 5


class IntelligenceEventType(str, Enum):
    SUMMARY = "SUMMARY"
    MEETING = "MEETING"
    DEADLINE = "DEADLINE"
    ASSIGNMENT = "ASSIGNMENT"
    TASK = "TASK"
    DECISION = "DECISION"
    ANNOUNCEMENT = "ANNOUNCEMENT"
    IDEA = "IDEA"
    ACTION_ITEM = "ACTION_ITEM"
    CONFLICT = "CONFLICT"
    RESOLUTION = "RESOLUTION"


@dataclass(frozen=True)
class WhatsAppMessage:
    message_id: str
    conversation_id: str
    timestamp: float
    text: str
    sender: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WhatsAppIntelligenceState:
    enabled: bool
    enabled_at: float | None
    last_processed_message_timestamp: float | None
    last_processed_message_id: str | None
    last_observation_timestamp: float | None


@dataclass(frozen=True)
class EventProposal:
    action: str
    event_type: IntelligenceEventType | None
    title: str = ""
    date_expression: str | None = None
    resolved_date: str | None = None
    time_expression: str | None = None
    resolved_time: str | None = None
    timezone: str | None = None
    platform: str | None = None
    location: str | None = None
    meeting_link: str | None = None
    participants: tuple[str, ...] = ()
    confidence: float = 0.0
    referenced_event_hint: str | None = None
    cancellation: bool = False
    confirmation: bool = False
    correction: bool = False
    evidence_message_id: str = ""
    extracted_claim: str = ""


@dataclass(frozen=True)
class IntelligenceEvent:
    event_id: str
    conversation_id: str
    type: IntelligenceEventType
    title: str
    description: str
    status: str
    confidence: float
    importance: float
    urgency: float
    created_at: float
    updated_at: float
    event_time: float | None = None
    deadline: float | None = None
    date: str | None = None
    time: str | None = None
    timezone: str | None = None
    platform: str | None = None
    location: str | None = None
    meeting_link: str | None = None
    participants: tuple[str, ...] = ()
    evidence: tuple[dict[str, Any], ...] = ()
    source_message_ids: tuple[str, ...] = ()
    scheduler_candidate: bool = False

    @property
    def source_count(self) -> int:
        return len(self.source_message_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "conversation_id": self.conversation_id,
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "confidence": self.confidence,
            "importance": self.importance,
            "urgency": self.urgency,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "event_time": self.event_time,
            "deadline": self.deadline,
            "date": self.date,
            "time": self.time,
            "timezone": self.timezone,
            "platform": self.platform,
            "location": self.location,
            "meeting_link": self.meeting_link,
            "participants": list(self.participants),
            "evidence": list(self.evidence),
            "source_message_ids": list(self.source_message_ids),
            "source_count": self.source_count,
            "scheduler_candidate": self.scheduler_candidate,
        }


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _message_identity(conversation_id: str, timestamp: float, sender: str, text: str, metadata: dict[str, Any] | None = None) -> str:
    """Stable fallback identity when BrowserSkill does not provide a native ID.

    Human-readable sender/direction fields are deliberately excluded because
    semantic observations can expose or omit them between polls. A stable
    machine identifier may be used when the provider supplies one.
    """
    stable_type = ""
    stable_sender_id = ""
    if metadata:
        for key in ("message_type", "type", "media_type"):
            value = metadata.get(key)
            if value:
                stable_type = _normalize_text(str(value)).casefold()
                break
        for key in ("sender_id", "senderId", "author_id", "authorId", "from_id", "fromId", "participant_id", "participantId"):
            value = metadata.get(key)
            if value:
                stable_sender_id = _normalize_text(str(value)).casefold()
                break
    basis = "\0".join((
        _normalize_text(conversation_id).casefold(),
        str(int(round(float(timestamp) * 1000))),
        stable_sender_id,
        _normalize_text(text),
        stable_type,
    ))
    return "msg-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]


def _coerce_timestamp(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value) if float(value) > 0 else None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            pass
        try:
            parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:
            return None
    return None


_TIME_TOKEN_RE = re.compile(
    r"(?<!\d)(?P<hour>[0-2]?\d)(?::(?P<minute>[0-5]\d))?\s*(?P<ampm>a\.?m\.?|p\.?m\.?)?(?!\d)",
    re.IGNORECASE,
)
_DATE_NUMERIC_RE = re.compile(r"\b(?P<d>\d{1,2})[/-](?P<m>\d{1,2})(?:[/-](?P<y>\d{2,4}))?\b")
_DATE_MONTH_RE = re.compile(
    r"\b(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"[ .,-]+(?P<day>\d{1,2})(?:[ .,-]+(?P<year>\d{4}))?\b",
    re.IGNORECASE,
)
_DATE_DAY_MONTH_RE = re.compile(
    r"\b(?P<day>\d{1,2})[ .,-]+(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[ .,-]+(?P<year>\d{4}))?\b",
    re.IGNORECASE,
)
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_SEMANTIC_ROLES = {"button", "listitem", "text", "generic", "group", "article", "paragraph", "heading", "textbox"}
_UI_NOISE = (
    "search or start a new chat", "type a message", "voice message", "attach", "emoji",
    "send", "menu", "starred", "archived", "status", "channels", "communities",
)


def _semantic_observation_strings(raw: Any) -> list[str]:
    """Collect semantic observation strings without querying or scraping the DOM."""
    out: list[str] = []
    seen: set[int] = set()

    def walk(node: Any) -> None:
        if isinstance(node, str):
            value = node.strip()
            if value and id(node) not in seen:
                seen.add(id(node))
                out.append(value)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(raw)
    return out


def _semantic_lines(raw: Any) -> list[tuple[str, str, str]]:
    """Parse BrowserSkill semantic ``@eN role "name"`` lines from raw text."""
    pattern = re.compile(
        r'^\s*(?P<ref>@e\d+)\s+(?P<role>[^\s"]+)\s+(?P<quoted>"(?:\\.|[^"\\])*")'
        r'(?:\s+\[[^\]\n]*\])*'
        r'(?:\s*=\s*(?P<value>"(?:\\.|[^"\\])*"|[^\s]+))?\s*$'
    )
    lines: list[tuple[str, str, str]] = []
    for blob in _semantic_observation_strings(raw):
        for line in blob.splitlines():
            match = pattern.match(line)
            if not match:
                continue
            quoted = match.group("quoted")
            try:
                name = bytes(quoted[1:-1], "utf-8").decode("unicode_escape")
            except Exception:
                name = quoted[1:-1]
            value = match.group("value") or ""
            if value.startswith('"') and value.endswith('"'):
                try:
                    value = bytes(value[1:-1], "utf-8").decode("unicode_escape")
                except Exception:
                    value = value[1:-1]
            lines.append((match.group("ref"), match.group("role").casefold(), _normalize_text(f"{name} {value}")))
    return lines


def _local_now(observed_at: float | None = None) -> datetime:
    base = datetime.fromtimestamp(observed_at if observed_at is not None else time.time()).astimezone()
    return base


def _semantic_date_context(text: str, base_date: datetime) -> datetime | None:
    lower = _normalize_text(text).casefold()
    if lower == "today" or re.search(r"\btoday\b", lower):
        return base_date.replace(hour=0, minute=0, second=0, microsecond=0)
    if lower == "yesterday" or re.search(r"\byesterday\b", lower):
        return (base_date - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    match = _DATE_NUMERIC_RE.search(text)
    if match:
        day = int(match.group("d")); month = int(match.group("m")); year = match.group("y")
        year_i = int(year) + (2000 if len(year) == 2 else 0) if year else base_date.year
        try:
            return base_date.replace(year=year_i, month=month, day=day, hour=0, minute=0, second=0, microsecond=0)
        except ValueError:
            return None

    for matcher in (_DATE_MONTH_RE, _DATE_DAY_MONTH_RE):
        match = matcher.search(text)
        if match:
            month = _MONTHS[re.sub(r"\.$", "", match.group("month")).casefold()]
            day = int(match.group("day")); year = int(match.group("year") or base_date.year)
            try:
                return base_date.replace(year=year, month=month, day=day, hour=0, minute=0, second=0, microsecond=0)
            except ValueError:
                return None
    return None


def _semantic_message_timestamp(text: str, *, date_context: datetime | None, observed_at: float | None) -> tuple[float | None, str | None]:
    matches = list(_TIME_TOKEN_RE.finditer(text))
    if not matches:
        return None, None
    # WhatsApp accessibility labels commonly place the message timestamp at
    # the end of the semantic message label, after the message body. Prefer
    # that final clock token; an earlier clock often belongs to message text
    # such as "meeting at 6pm".
    match = matches[-1]
    if len(matches) > 1:
        previous = text[max(0, match.start() - 4):match.start()].casefold()
        if re.search(r"\bat\s*$", previous):
            match = matches[0]
    hour = int(match.group("hour")); minute = int(match.group("minute") or 0)
    ampm = (match.group("ampm") or "").replace(".", "").casefold()
    if ampm:
        if hour < 1 or hour > 12:
            return None, None
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
    elif hour > 23:
        return None, None

    base = date_context or _local_now(observed_at)
    candidate = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if date_context is None:
        now_local = _local_now(observed_at)
        # A displayed clock later than the current local time belongs to the
        # configured lookback window, which covers a read message such as yesterday at 11 PM
        # when today's Home row no longer has an unread badge.
        if candidate > now_local + timedelta(minutes=2):
            candidate -= timedelta(days=1)
    return candidate.timestamp(), match.group(0)


_DIRECTION_BOOL_KEYS = {
    "from_me", "fromme", "is_from_me", "isfromme", "outgoing", "is_outgoing",
    "isoutgoing", "sent_by_me", "sentbyme", "is_current_user", "iscurrentuser",
}
_DIRECTION_KEYS = {"direction", "message_direction"}
_SENDER_KEYS = {"sender", "author", "from", "sender_name", "sendername", "author_name", "authorname"}
_CURRENT_USER_KEYS = {"current_user", "currentuser", "logged_in_user", "loggedinuser", "profile_name", "profilename", "my_name", "myname"}
_SENDER_ID_KEYS = {"sender_id", "senderid", "author_id", "authorid", "from_id", "fromid", "participant_id", "participantid"}
_CURRENT_USER_ID_KEYS = {"current_user_id", "currentuserid", "profile_id", "profileid", "logged_in_user_id", "loggedinuserid", "my_id", "myid"}
_NESTED_OWNERSHIP_KEYS = {"metadata", "attributes", "ownership", "sender_info", "senderinfo", "message_meta", "messagemeta"}
_SELF_MARKERS = {"you", "you said", "me", "self", "myself"}
_RESERVED_SENDER_TOKENS = {"http", "https", "www", "unread", "read", "delivered", "seen", "typing", "message", "status", "sent"}
_GARBAGE_SENDER_RE = re.compile(r"\b\d+\s+unread\s+messages?\b", re.I)
_URL_SENDER_RE = re.compile(r"^(?:[a-z][a-z0-9+.-]*://|www\.)", re.I)


def _valid_sender_identity(value: Any) -> str:
    text = _normalize_text(str(value or ""))
    if not text or len(text) > 80:
        return ""
    folded = text.casefold()
    if folded in _RESERVED_SENDER_TOKENS or _GARBAGE_SENDER_RE.search(text) or _URL_SENDER_RE.match(text):
        return ""
    if not re.search(r"[A-Za-z]", text):
        return ""
    if re.fullmatch(r"[^A-Za-z0-9]+", text):
        return ""
    # Sender labels should be names/identities, not a long UI sentence.
    if len(text.split()) > 8:
        return ""
    return text


def _semantic_direction_text(text: str) -> str:
    """Read an explicit semantic direction marker from one semantic node."""
    value = _normalize_text(text)
    lower = value.casefold()
    if re.fullmatch(r"you\s*:\s*.*", lower) or re.search(r"(?:^|[\[|,; ])(?:sent|message sent)(?:$|[\]|,; ])", lower):
        return "outgoing"
    if re.fullmatch(r"(?:message\s+)?(?:received|incoming)(?:\s+message)?", lower):
        return "incoming"
    return "unknown"


def _ownership_from_mapping(node: dict[str, Any]) -> tuple[str, str, str]:
    """Resolve ownership from one message-scoped mapping, failing closed on conflicts."""
    if not isinstance(node, dict):
        return "unknown", "", "NONE"
    keys = {str(k).casefold(): k for k in node}
    decisions: list[tuple[str, str]] = []

    for folded in _DIRECTION_BOOL_KEYS:
        key = keys.get(folded)
        if key is None:
            continue
        value = node.get(key)
        decision: str | None = None
        if isinstance(value, bool):
            decision = "outgoing" if value else "incoming"
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            decision = "outgoing" if bool(value) else "incoming"
        else:
            normalized = _normalize_text(str(value)).casefold()
            if normalized in {"true", "1", "yes", "y", "outgoing", "sent", "mine"}:
                decision = "outgoing"
            elif normalized in {"false", "0", "no", "n", "incoming", "received"}:
                decision = "incoming"
        if decision:
            decisions.append((decision, f"{key}={value}"))

    for folded in _DIRECTION_KEYS:
        key = keys.get(folded)
        if key is None:
            continue
        value = _normalize_text(str(node.get(key)))
        normalized = value.casefold()
        if normalized in {"outgoing", "sent", "from_me", "from me"}:
            decisions.append(("outgoing", f"{key}={value}"))
        elif normalized in {"incoming", "received", "from_other", "from other"}:
            decisions.append(("incoming", f"{key}={value}"))

    sender_key = next((keys[k] for k in _SENDER_KEYS if k in keys), None)
    sender = _valid_sender_identity(node.get(sender_key)) if sender_key is not None else ""
    current_user_key = next((keys[k] for k in _CURRENT_USER_KEYS if k in keys), None)
    current_user = _valid_sender_identity(node.get(current_user_key)) if current_user_key is not None else ""
    sender_id_key = next((keys[k] for k in _SENDER_ID_KEYS if k in keys), None)
    sender_id = _normalize_text(str(node.get(sender_id_key) or "")) if sender_id_key is not None else ""
    current_user_id_key = next((keys[k] for k in _CURRENT_USER_ID_KEYS if k in keys), None)
    current_user_id = _normalize_text(str(node.get(current_user_id_key) or "")) if current_user_id_key is not None else ""

    if sender_id and current_user_id:
        if sender_id.casefold() == current_user_id.casefold():
            decisions.append(("outgoing", f"{sender_id_key}={sender_id}"))
        else:
            decisions.append(("incoming", f"{sender_id_key}={sender_id}"))

    if sender:
        if sender.casefold() in _SELF_MARKERS:
            decisions.append(("outgoing", f"{sender_key}={sender}"))
        elif current_user and sender.casefold() == current_user.casefold():
            decisions.append(("outgoing", f"{sender_key}={sender}"))
        elif not sender_id:
            # A message-scoped sender field is ownership evidence. It is not
            # inferred from chat name, row position, or unrelated page text.
            decisions.append(("incoming", f"{sender_key}={sender}"))

    if decisions:
        unique = {decision for decision, _ in decisions}
        if len(unique) > 1:
            detail = ";".join(evidence for _, evidence in decisions[:4])
            return "unknown", sender, f"CONFLICT:{detail}"
        direction = next(iter(unique))
        evidence = next(evidence for decision, evidence in decisions if decision == direction)
        return direction, sender, evidence

    return "unknown", "", "NONE"


def _node_message_ownership(node: dict[str, Any]) -> tuple[str, str, str]:
    """Read ownership from the message node and known semantic metadata only."""
    direction, sender, evidence = _ownership_from_mapping(node)
    if direction != "unknown" or evidence != "NONE":
        return direction, sender, evidence
    if not isinstance(node, dict):
        return "unknown", "", "NONE"
    for nested_key, raw_value in node.items():
        if str(nested_key).casefold() in _NESTED_OWNERSHIP_KEYS and isinstance(raw_value, dict):
            direction, sender, evidence = _ownership_from_mapping(raw_value)
            if direction != "unknown":
                return direction, sender, evidence
    return "unknown", "", "NONE"


def _node_message_direction(node: dict[str, Any]) -> str:
    return _node_message_ownership(node)[0]


def _direction_evidence(element: Any) -> str:
    raw = getattr(element, "raw", None) if not isinstance(element, dict) else element
    attrs = getattr(element, "attributes", None) if not isinstance(element, dict) else element.get("attributes")
    if isinstance(raw, dict):
        _direction, _sender, evidence = _node_message_ownership(raw)
        if evidence != "NONE":
            return evidence
    if isinstance(attrs, dict):
        annotations = attrs.get("annotations")
        if isinstance(annotations, (list, tuple)):
            for annotation in annotations:
                m = re.match(r"\s*(from_me|fromme|is_outgoing|isoutgoing|outgoing|sent_by_me|direction|message_direction|sender|author)\s*=\s*(.*?)\s*$", str(annotation), re.I)
                if not m:
                    continue
                key, value = m.group(1), _normalize_text(m.group(2))
                if key.casefold() in _DIRECTION_BOOL_KEYS | _DIRECTION_KEYS:
                    return f"{key}={value}"
                valid = _valid_sender_identity(value)
                if valid:
                    return f"{key}={valid}"
    name = getattr(element, "name", "") if not isinstance(element, dict) else element.get("name", "")
    value = getattr(element, "value", "") if not isinstance(element, dict) else element.get("value", "")
    marker = _semantic_direction_text(" ".join(str(x) for x in (name, value) if x))
    if marker == "outgoing":
        return "semantic_marker=You/Sent"
    if marker == "incoming":
        return "semantic_marker=Received"
    return "NONE"


def _semantic_element_ownership(element: Any) -> tuple[str, str, str]:
    raw = getattr(element, "raw", None) if not isinstance(element, dict) else element
    if isinstance(raw, dict):
        result = _node_message_ownership(raw)
        if result[0] != "unknown" or result[2] != "NONE":
            return result
    attrs = getattr(element, "attributes", None) if not isinstance(element, dict) else element.get("attributes")
    if isinstance(attrs, dict):
        annotations = attrs.get("annotations")
        if isinstance(annotations, (list, tuple)):
            for annotation in annotations:
                text = _normalize_text(str(annotation))
                m = re.match(r"^(from_me|fromme|is_from_me|isfromme|is_outgoing|isoutgoing|outgoing|sent_by_me|sentbyme)\s*[=:]\s*(true|1|yes|false|0|no)$", text, re.I)
                if m:
                    outgoing = m.group(2).casefold() in {"true", "1", "yes"}
                    return ("outgoing" if outgoing else "incoming"), "", f"{m.group(1)}={m.group(2)}"
                m = re.match(r"^(direction|message_direction)\s*[=:]\s*(outgoing|incoming|sent|received)$", text, re.I)
                if m:
                    direction = "outgoing" if m.group(2).casefold() in {"outgoing", "sent"} else "incoming"
                    return direction, "", f"{m.group(1)}={m.group(2)}"
                m = re.match(r"^(sender|author)\s*[=:]\s*(.*?)$", text, re.I)
                if m:
                    sender = _valid_sender_identity(m.group(2))
                    if sender:
                        return ("outgoing" if sender.casefold() in _SELF_MARKERS else "incoming"), sender, f"{m.group(1)}={sender}"
    name = getattr(element, "name", "") if not isinstance(element, dict) else element.get("name", "")
    value = getattr(element, "value", "") if not isinstance(element, dict) else element.get("value", "")
    marker = _semantic_direction_text(" ".join(str(x) for x in (name, value) if x))
    if marker == "outgoing":
        return "outgoing", "", "semantic_marker=You/Sent"
    if marker == "incoming":
        return "incoming", "", "semantic_marker=Received"
    return "unknown", "", "NONE"


def _semantic_element_direction(element: Any) -> str:
    return _semantic_element_ownership(element)[0]


def _semantic_self_chat_header_present(raw: Any, conversation_hint: str) -> bool:
    """Return True only when fresh semantic evidence identifies the self-chat.

    BrowserSkill observations may expose the header in structured nested data
    rather than directly under ``raw["elements"]``. Walk the observed semantic
    tree and rendered text, but only accept strong markers: exact ``You`` or
    ``<target> (You)`` together with evidence of the authorized target.
    """
    wanted = _normalize_text(conversation_hint).casefold()
    if not wanted:
        return False

    candidates: list[tuple[str, str]] = []
    text_parts: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            role = str(node.get("role") or node.get("type") or node.get("element_type") or "").casefold()
            name = _normalize_text(node.get("name") or node.get("label") or node.get("aria-label") or node.get("ariaLabel") or "")
            value = _normalize_text(node.get("value") or "")
            if name:
                candidates.append((role, name))
                text_parts.append(name)
            if value:
                text_parts.append(value)
            for child in node.values():
                walk(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    elements = getattr(raw, "elements", None)
    if elements:
        for element in elements:
            role = str(getattr(element, "role", "") or "").casefold()
            name = _normalize_text(getattr(element, "name", "") or "")
            value = _normalize_text(getattr(element, "value", "") or "")
            if name:
                candidates.append((role, name))
                text_parts.append(name)
            if value:
                text_parts.append(value)
            element_raw = getattr(element, "raw", None)
            if isinstance(element_raw, (dict, list, tuple)):
                walk(element_raw)
    else:
        walk(raw)

    raw_text = getattr(raw, "text", None)
    if isinstance(raw_text, str):
        text_parts.append(raw_text)
        for line in raw_text.splitlines():
            value = _normalize_text(line)
            if value:
                # BrowserSkill text snapshots use lines like ``@e7 button
                # "Saksham (You)"``; strip the ref/role prefix where possible.
                quoted = re.search(r'"([^"]+)"', value)
                candidates.append(("text", _normalize_text(quoted.group(1) if quoted else value)))

    all_text = _normalize_text(" ".join(text_parts)).casefold()
    target_present = bool(re.search(rf"(?<!\w){re.escape(wanted)}(?!\w)", all_text))
    if not target_present:
        return False

    for role, name in candidates:
        folded = name.casefold()
        if role in {"heading", "button", "link", "text", "listitem"} and folded == "you":
            return True
        if re.fullmatch(rf"{re.escape(wanted)}\s*\(you\)", folded):
            return True
    return False


def _parse_clock_match(match: re.Match[str]) -> tuple[int, int] | None:
    """Convert a matched 12/24-hour clock token to a wall-clock hour/minute."""
    try:
        hour = int(match.group("hour"))
        minute = int(match.group("minute") or 0)
    except (TypeError, ValueError):
        return None
    meridiem = (match.group("ampm") or "").replace(".", "").casefold()
    if meridiem:
        if hour < 1 or hour > 12:
            return None
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
    elif hour > 23:
        return None
    return hour, minute


def _timezone_from_value(value: Any) -> Any | None:
    text = _normalize_text(str(value or ""))
    if not text:
        return None
    try:
        return ZoneInfo(text)
    except ZoneInfoNotFoundError:
        return None


def _timezone_label(tz: Any) -> str:
    key = getattr(tz, "key", None)
    if key:
        return str(key)
    name = datetime.now(tz=tz).tzname() if tz is not None else None
    return str(name or "UTC")


_EXPLICIT_EVENT_TZ_RE = re.compile(r"\b(?P<tz>Asia/Kolkata|Asia/Calcutta|UTC|GMT|IST)\b", re.I)
_EXPLICIT_EVENT_TZ_ALIASES = {
    "asia/calcutta": DEFAULT_EVENT_TIMEZONE,
    "ist": DEFAULT_EVENT_TIMEZONE,
    "utc": "UTC",
    "gmt": "UTC",
    "asia/kolkata": DEFAULT_EVENT_TIMEZONE,
}

def _explicit_event_timezone(text: str) -> Any | None:
    match = _EXPLICIT_EVENT_TZ_RE.search(text or "")
    if not match:
        return None
    value = _EXPLICIT_EVENT_TZ_ALIASES.get(match.group("tz").casefold(), match.group("tz"))
    return _timezone_from_value(value)


def _event_timezone_for_message(message: WhatsAppMessage, text: str | None = None) -> Any:
    """Resolve event timezone: message text, account/event config, env, canonical DEIMOS default."""
    explicit = _explicit_event_timezone(text or "")
    if explicit is not None:
        return explicit
    metadata = message.metadata or {}
    for key in (
        "event_timezone", "event_timezone_name", "account_timezone",
        "account_timezone_name", "user_timezone", "user_timezone_name", "timezone",
    ):
        resolved = _timezone_from_value(metadata.get(key))
        if resolved is not None:
            return resolved
    configured = _timezone_from_value(os.environ.get("DEIMOS_EVENT_TIMEZONE"))
    if configured is not None:
        return configured
    return DEFAULT_EVENT_TZ


def _source_timezone_label(message: WhatsAppMessage) -> str:
    metadata = message.metadata or {}
    for key in ("source_timezone", "message_timezone", "timezone"):
        value = _normalize_text(str(metadata.get(key) or ""))
        if value:
            return value
    # The persisted message timestamp is an instant; without explicit source
    # metadata there is no truthful source-zone claim to make.
    return "UNKNOWN"


def _event_time_match(text: str, *, meeting_context: bool) -> re.Match[str] | None:
    """Select the event clock, not a WhatsApp message timestamp embedded in the label."""
    matches = list(_TIME_TOKEN_RE.finditer(text))
    if not matches:
        return None

    # Explicit event cues take precedence. This handles both common semantic
    # layouts: ``Meeting at 6 PM 01:00`` and ``01:00 Meeting at 6 PM``.
    for match in matches:
        prefix = text[max(0, match.start() - 32):match.start()]
        if re.search(r"\b(?:at|around|by|from|until|till)\s*$", prefix, re.I):
            return match
        if meeting_context and re.search(r"\b(?:meeting|meet|call)\b[^0-9]{0,24}$", prefix, re.I):
            return match

    # If a single clock exists, there is no competing message timestamp.
    if len(matches) == 1:
        return matches[0]

    # AM/PM is much more likely to be the natural-language event expression
    # than the UI's message timestamp. This is only a fallback; explicit event
    # cues above remain authoritative.
    meridiem_matches = [m for m in matches if m.group("ampm")]
    if len(meridiem_matches) == 1:
        return meridiem_matches[0]

    return None


def _semantic_message_text(text: str, time_token: str | None, conversation_hint: str = "") -> tuple[str, str]:
    body = _normalize_text(text)
    if time_token:
        body = _normalize_text(re.sub(re.escape(time_token), " ", body, count=1, flags=re.IGNORECASE))
    body = re.sub(r"\[[^\]]*(?:read|delivered|sent|unread)[^\]]*\]", " ", body, flags=re.IGNORECASE)
    body = re.sub(r"\b(?:read|delivered|sent)\b$", " ", body, flags=re.IGNORECASE)
    sender = ""
    sender_match = re.match(r"^(?P<sender>[^:]{1,80}):\s*(?P<body>.+)$", body)
    if sender_match:
        candidate = _valid_sender_identity(sender_match.group("sender"))
        if candidate:
            sender = candidate
            body = _normalize_text(sender_match.group("body"))
    if body.casefold() == conversation_hint.casefold():
        body = ""
    return body, sender


def _semantic_element_message_id(element: Any) -> str:
    raw = getattr(element, "raw", None) if not isinstance(element, dict) else element
    attrs = getattr(element, "attributes", None) if not isinstance(element, dict) else element.get("attributes")
    sources = [raw]
    if isinstance(attrs, dict):
        sources.append(attrs)
    for source in sources:
        if isinstance(source, dict):
            keys = {str(k).casefold(): k for k in source}
            for key in ("message_id", "messageid", "native_message_id", "nativemessageid", "wa_message_id", "wamessageid"):
                actual = keys.get(key)
                if actual is not None:
                    value = _normalize_text(str(source.get(actual) or ""))
                    if value:
                        return value
        elif isinstance(source, str):
            match = re.search(r"\[(?:message[_ -]?id|native[_ -]?message[_ -]?id)\s*[=:]\s*([^\]]+)\]", source, re.I)
            if match:
                value = _normalize_text(match.group(1))
                if value:
                    return value
    if isinstance(attrs, dict):
        annotations = attrs.get("annotations")
        if isinstance(annotations, (list, tuple)):
            for annotation in annotations:
                match = re.match(r"(?:message[_ -]?id|native[_ -]?message[_ -]?id)\s*[=:]\s*(.*?)$", str(annotation), re.I)
                if match:
                    value = _normalize_text(match.group(1))
                    if value:
                        return value
    return ""


def _extract_semantic_messages(raw: Any, *, conversation_hint: str = "", observed_at: float | None = None) -> list[WhatsAppMessage]:
    """Recover message records from the existing BrowserSkill semantic observation."""
    semantic: list[tuple[str, str, str]] = []
    direction_hints: dict[str, tuple[str, str]] = {}
    direction_diagnostics: dict[str, str] = {}
    sender_hints: dict[str, str] = {}
    message_id_hints: dict[str, str] = {}
    timestamp_hints: dict[str, float] = {}
    seen_nodes: set[tuple[str, str, str]] = set()

    def add_semantic(ref: str, role: str, value: str, element: Any) -> None:
        if not value:
            return
        item = (ref, role, value)
        if item not in seen_nodes:
            seen_nodes.add(item)
            semantic.append(item)
        direction, sender, evidence = _semantic_element_ownership(element)
        if ref and evidence != "NONE":
            direction_diagnostics[ref] = evidence
        if ref and direction != "unknown":
            direction_hints[ref] = (direction, evidence)
        if ref and sender:
            sender_hints[ref] = sender
        native_id = _semantic_element_message_id(element)
        if ref and native_id:
            message_id_hints[ref] = native_id
        raw = getattr(element, "raw", None) if not isinstance(element, dict) else element
        if isinstance(raw, dict):
            keys = {str(k).casefold(): k for k in raw}
            for key in ("timestamp", "message_timestamp", "message_time", "datetime"):
                actual = keys.get(key)
                if actual is not None:
                    parsed_timestamp = _coerce_timestamp(raw.get(actual))
                    if parsed_timestamp is not None and ref:
                        timestamp_hints[ref] = parsed_timestamp
                    break

    elements = getattr(raw, "elements", None)
    if elements:
        for element in elements:
            role = str(getattr(element, "role", "") or "").casefold()
            name = _normalize_text(getattr(element, "name", "") or "")
            value = _normalize_text(getattr(element, "value", "") or "")
            if role in _SEMANTIC_ROLES and (name or value):
                ref = str(getattr(element, "ref", "") or "")
                add_semantic(ref, role, _normalize_text(f"{name} {value}"), element)
    else:
        def walk_structured(node: Any) -> None:
            if isinstance(node, dict):
                role = str(node.get("role") or node.get("type") or node.get("element_type") or "").casefold()
                name_value = next((node.get(key) for key in ("name", "label", "aria-label", "ariaLabel", "description") if isinstance(node.get(key), str) and node.get(key).strip()), "")
                value = next((node.get(key) for key in ("value", "text", "content", "innerText") if isinstance(node.get(key), str) and node.get(key).strip()), "")
                ref = str(node.get("ref") or node.get("id") or "")
                if role in _SEMANTIC_ROLES and (name_value or value):
                    add_semantic(ref, role, _normalize_text(f"{name_value} {value}"), node)
                for child in node.values():
                    walk_structured(child)
            elif isinstance(node, (list, tuple)):
                for child in node:
                    walk_structured(child)

        walk_structured(raw)
        for ref, role, name in _semantic_lines(getattr(raw, "text", raw)):
            item = (ref, role, name)
            if item not in seen_nodes:
                seen_nodes.add(item)
                semantic.append(item)
            # Text-line renderers do not carry structured ownership apart from
            # their own visible marker, which is processed below.

    if not semantic:
        return []

    all_text = "\n".join(item[2] for item in semantic)
    if conversation_hint and not re.search(rf"(?<!\w){re.escape(conversation_hint.casefold())}(?!\w)", all_text.casefold()):
        return []

    base = _local_now(observed_at)
    date_context: datetime | None = None
    found: list[WhatsAppMessage] = []
    seen: set[str] = set()

    for index, (ref, role, value) in enumerate(semantic):
        if not value:
            continue
        maybe_date = _semantic_date_context(value, base)
        if maybe_date is not None and not _TIME_TOKEN_RE.search(value):
            date_context = maybe_date
            continue

        combined = value
        combined_refs = [ref]
        current_hint = direction_hints.get(ref, ("unknown", direction_diagnostics.get(ref, "NONE")))
        current_direction, current_evidence = current_hint
        current_sender = sender_hints.get(ref, "")
        if current_direction == "unknown" and ref in direction_diagnostics:
            current_evidence = direction_diagnostics[ref]

        # Do not combine ownership from adjacent semantic nodes. Adjacent
        # context may be used below only for reconstructing a message body when
        # the timestamp is presented as a separate semantic node.

        timestamp_match = list(_TIME_TOKEN_RE.finditer(value))
        stripped_current = _TIME_TOKEN_RE.sub(" ", value).strip() if timestamp_match else value
        if timestamp_match and ref not in timestamp_hints and role == "text" and re.search(r"\bat\s*$", value[:timestamp_match[-1].start()].casefold()):
            continue
        if timestamp_match and not stripped_current:
            parts = []
            for previous_index in range(index - 1, max(-1, index - 5), -1):
                prev_ref, prev_role, prev_value = semantic[previous_index]
                prev_lower = prev_value.casefold()
                if _semantic_date_context(prev_value, base) is not None:
                    break
                if _TIME_TOKEN_RE.search(prev_value) and not _TIME_TOKEN_RE.sub(" ", prev_value).strip():
                    break
                if prev_role in {"heading", "listitem"}:
                    break
                if any(noise in prev_lower for noise in _UI_NOISE):
                    break
                if prev_value:
                    parts.append(prev_value)
                combined_refs.append(prev_ref)
            if parts:
                combined = _normalize_text(" ".join(reversed(parts)) + " " + value)

        lower = combined.casefold()
        if any(noise in lower for noise in _UI_NOISE) and not re.search(r"\b(?:meeting|meet|deadline|task|assignment|tomorrow|today)\b", lower):
            continue
        timestamp, time_token = _semantic_message_timestamp(combined, date_context=date_context, observed_at=observed_at)
        if ref in timestamp_hints:
            timestamp = timestamp_hints[ref]
            time_token = None
        if timestamp is None:
            continue
        body, parsed_sender = _semantic_message_text(combined, time_token, conversation_hint)
        sender = current_sender or parsed_sender
        if not body or not re.search(r"[A-Za-z0-9]", body):
            continue

        direction = current_direction
        evidence = current_evidence
        if sender.casefold() in _SELF_MARKERS:
            direction, evidence = "outgoing", (f"sender={sender}" if sender else "semantic_marker=You/Sent")
        elif direction == "unknown" and parsed_sender:
            direction, evidence = "incoming", f"sender={sender}"

        # Direction evidence must stay scoped to the semantic message node.
        # Never inherit ownership from an unrelated nearby node.

        native_id = next((message_id_hints.get(r) for r in combined_refs if message_id_hints.get(r)), "")
        message_id = native_id or _message_identity(conversation_hint or "whatsapp", timestamp, sender, body)
        if message_id in seen:
            continue
        seen.add(message_id)
        found.append(WhatsAppMessage(
            message_id=message_id,
            conversation_id=conversation_hint or "whatsapp",
            timestamp=timestamp,
            text=body,
            sender=sender,
            metadata={
                "source": "browserskill_semantic_observation",
                "role": role,
                "observed_at": observed_at,
                "direction": direction,
                "is_outgoing": True if direction == "outgoing" else (False if direction == "incoming" else None),
                "sender_evidence": evidence if evidence != "NONE" else "NONE",
            },
        ))
    found.sort(key=lambda item: (item.timestamp, item.message_id))
    return found


def extract_messages(raw: Any, *, conversation_hint: str = "", observed_at: float | None = None) -> list[WhatsAppMessage]:
    """Extract explicit message records only; never invent timestamps from observation time.

    BrowserSkill observation payloads vary by extension version. A message is
    accepted only when a conversation/message identity and an actual message
    timestamp are present. This prevents the observer from treating the time it
    happened to see a DOM node as the WhatsApp message timestamp.
    """
    found: list[WhatsAppMessage] = []
    seen: set[str] = set()

    def walk(node: Any, inherited_conversation: str = "") -> None:
        if isinstance(node, dict):
            keys = {str(k).casefold(): k for k in node}
            text_key = next((keys[k] for k in ("message", "body", "text", "content") if k in keys), None)
            time_key = next((keys[k] for k in ("timestamp", "message_timestamp", "message_time", "time", "datetime") if k in keys), None)
            id_key = next((keys[k] for k in ("message_id", "messageid", "native_message_id", "nativeMessageId") if k in keys), None)
            conv_key = next((keys[k] for k in ("conversation_id", "conversationid", "chat_id", "chatid", "conversation") if k in keys), None)
            sender_key = next((keys[k] for k in ("sender", "author", "from") if k in keys), None)
            timestamp = _coerce_timestamp(node.get(time_key)) if time_key else None
            text = _normalize_text(node.get(text_key, "")) if text_key else ""
            conversation = str(node.get(conv_key) or inherited_conversation or "whatsapp").strip()
            sender = _valid_sender_identity(node.get(sender_key)) if sender_key else ""
            explicit_id = str(node.get(id_key) or "").strip() if id_key else ""
            if text and timestamp is not None:
                direction, evidence_sender, sender_evidence = _node_message_ownership(node)
                sender = sender or evidence_sender
                if sender.casefold() in _SELF_MARKERS:
                    direction, sender_evidence = "outgoing", f"sender={sender}"
                message_id = explicit_id or _message_identity(conversation, timestamp, sender, text, node)
                if message_id not in seen:
                    metadata = dict(node)
                    metadata["direction"] = direction
                    metadata["is_outgoing"] = True if direction == "outgoing" else (False if direction == "incoming" else None)
                    metadata["sender_evidence"] = sender_evidence if sender_evidence != "NONE" else "NONE"
                    seen.add(message_id)
                    found.append(WhatsAppMessage(message_id, conversation, timestamp, text, sender, metadata))
            next_conversation = conversation
            for key, value in node.items():
                if key != text_key:
                    walk(value, next_conversation)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item, inherited_conversation)

    walk(raw)
    if conversation_hint:
        for message in _extract_semantic_messages(raw, conversation_hint=conversation_hint, observed_at=observed_at):
            if message.message_id not in seen:
                seen.add(message.message_id)
                found.append(message)
    found.sort(key=lambda m: (m.timestamp, m.message_id))
    return found


class WhatsAppIntelligenceStore:
    def __init__(self, path: str | Path = DEFAULT_STORE) -> None:
        self.path = Path(path) if str(path) == ":memory:" else Path(path).expanduser().resolve()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._schema_lock = threading.Lock()
        lock_key = ":memory:" if str(self.path) == ":memory:" else str(self.path)
        with _DB_LOCKS_GUARD:
            self._db_lock = _DB_LOCKS.setdefault(lock_key, threading.RLock())
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.path),
                timeout=5.0,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    def _in_transaction(self) -> bool:
        return bool(getattr(self._local, "transaction_depth", 0))

    @contextmanager
    def _write_transaction(self, operation: str):
        if self._in_transaction():
            yield self._conn()
            return
        with self._db_lock:
            conn = self._conn()
            try:
                conn.execute("BEGIN IMMEDIATE")
                self._local.transaction_depth = 1
                yield conn
                conn.commit()
            except sqlite3.Error as exc:
                try:
                    conn.rollback()
                finally:
                    self._local.transaction_depth = 0
                self._db_debug(operation, exc)
                raise
            except Exception:
                try:
                    conn.rollback()
                finally:
                    self._local.transaction_depth = 0
                raise
            finally:
                self._local.transaction_depth = 0

    @staticmethod
    def _is_retryable_busy(exc: sqlite3.Error) -> bool:
        text = str(exc).casefold()
        return isinstance(exc, sqlite3.OperationalError) and (
            "database is locked" in text or "database is busy" in text
        )

    def _run_write(self, operation: str, callback: Callable[[], Any]) -> Any:
        """Run one short standalone write with bounded busy/lock retry."""
        for attempt in range(_DB_BUSY_RETRIES + 1):
            try:
                with self._write_transaction(operation):
                    return callback()
            except sqlite3.Error as exc:
                if self._is_retryable_busy(exc) and attempt < _DB_BUSY_RETRIES:
                    time.sleep(_DB_RETRY_SLEEP_S * (attempt + 1))
                    continue
                raise
        raise RuntimeError("database write retry loop exhausted")

    def _db_debug(self, operation: str, exc: BaseException) -> None:
        callback = getattr(self, "_debug_callback", None)
        if callback is not None:
            callback(
                f"WHATSAPP_DB: operation={operation} "
                f"error={type(exc).__name__}:{str(exc).strip()}"
            )

    def set_debug_callback(self, callback: Callable[[str], None] | None) -> None:
        self._debug_callback = callback

    @staticmethod
    def _migrate_event_evidence_schema(conn: sqlite3.Connection) -> None:
        """Migrate the pre-evidence-id table in place without losing evidence.

        Older P3.3 databases were created with event_evidence lacking the
        deterministic evidence_id column. CREATE TABLE IF NOT EXISTS cannot
        change that existing table, so the runtime INSERT could fail forever.
        SQLite cannot ADD a primary-key constraint to an existing table; the
        compatibility migration therefore adds the column, backfills stable
        IDs from the existing natural evidence key, and enforces uniqueness with
        a normal unique index. No evidence rows are deleted.
        """
        rows = conn.execute("PRAGMA table_info(event_evidence)").fetchall()
        if not rows:
            return
        columns = {row[1] for row in rows}
        if "evidence_id" not in columns:
            conn.execute("ALTER TABLE event_evidence ADD COLUMN evidence_id TEXT")
            columns.add("evidence_id")
        required = {
            "event_id", "message_id", "conversation_id", "message_timestamp",
            "evidence_type", "extracted_claim", "confidence", "created_at",
        }
        missing = sorted(required - columns)
        if missing:
            raise sqlite3.DatabaseError(
                "event_evidence_schema_incompatible:" + ",".join(missing)
            )
        existing = conn.execute(
            "SELECT rowid,event_id,message_id,evidence_type FROM event_evidence "
            "WHERE evidence_id IS NULL OR evidence_id=''"
        ).fetchall()
        for row in existing:
            evidence_id = "evi-" + hashlib.sha256(
                f"{row['event_id']}\0{row['message_id']}\0{row['evidence_type']}".encode()
            ).hexdigest()[:24]
            conn.execute(
                "UPDATE event_evidence SET evidence_id=? WHERE rowid=?",
                (evidence_id, row["rowid"]),
            )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_event_evidence_id "
            "ON event_evidence(evidence_id)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_event_evidence_natural "
            "ON event_evidence(event_id,message_id,evidence_type)"
        )

    @staticmethod
    def _migrate_failure_schema(conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS message_failures (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                failure_count INTEGER NOT NULL DEFAULT 0,
                last_failure_at REAL NOT NULL,
                retry_after REAL NOT NULL,
                error_category TEXT NOT NULL,
                error_detail TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_message_failures_retry "
            "ON message_failures(retry_after)"
        )

    def _init(self) -> None:
        with self._schema_lock, self._db_lock:
            conn = self._conn()
            # Configure WAL only once, before the observer thread can create its
            # own connection. Per-thread connections must not renegotiate the
            # journal mode.
            conn.execute("PRAGMA journal_mode=WAL")
            with conn:
                conn.executescript("""
                CREATE TABLE IF NOT EXISTS intelligence_state (
                    id INTEGER PRIMARY KEY CHECK(id=1),
                    enabled INTEGER NOT NULL DEFAULT 0,
                    enabled_at REAL,
                    last_processed_message_timestamp REAL,
                    last_processed_message_id TEXT,
                    last_observation_timestamp REAL
                );
                INSERT OR IGNORE INTO intelligence_state(id, enabled) VALUES(1, 0);
                CREATE TABLE IF NOT EXISTS processed_messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    message_timestamp REAL NOT NULL,
                    processed_at REAL NOT NULL,
                    eligible INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_processed_ts ON processed_messages(message_timestamp);
                CREATE TABLE IF NOT EXISTS event_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    message_timestamp REAL NOT NULL,
                    evidence_type TEXT NOT NULL,
                    extracted_claim TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(event_id, message_id, evidence_type)
                );
                CREATE INDEX IF NOT EXISTS idx_event_evidence_event ON event_evidence(event_id, message_timestamp);

                CREATE TABLE IF NOT EXISTS intelligence_events (
                    event_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    importance REAL NOT NULL,
                    urgency REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    event_time REAL,
                    deadline REAL,
                    date TEXT,
                    time TEXT,
                    timezone TEXT,
                    platform TEXT,
                    location TEXT,
                    meeting_link TEXT,
                    participants_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    source_message_ids_json TEXT NOT NULL,
                    scheduler_candidate INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    conversation_id TEXT PRIMARY KEY,
                    current_summary TEXT NOT NULL,
                    summary_version INTEGER NOT NULL,
                    last_summary_message_id TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS authorized_targets (
                    target_name TEXT PRIMARY KEY,
                    granted_at REAL NOT NULL
                );
                """)
                # Migrate existing P3.3 databases whose event_evidence table was
                # created before evidence_id was introduced.
                self._migrate_event_evidence_schema(conn)
                self._migrate_failure_schema(conn)

                # Legacy title/status uniqueness prevented two legitimate active
                # meetings in the same chat. Event coreference, not SQL uniqueness,
                # decides whether two events are the same.
                conn.execute("DROP INDEX IF EXISTS idx_event_merge")

                # P3.3 schema migration: an earlier development snapshot could
                # already have an ``authorized_targets`` table with a different
                # column name. ``CREATE TABLE IF NOT EXISTS`` does not migrate
                # that table, so the observer would fail on every poll with
                # ``no column named target_name``. Migrate the known legacy shape
                # in-place without touching any other intelligence state.
                columns = {row[1] for row in conn.execute("PRAGMA table_info(authorized_targets)").fetchall()}
                if "target_name" not in columns:
                    legacy_target = next(
                        (name for name in ("target", "contact_name", "name", "contact") if name in columns),
                        None,
                    )
                    if legacy_target is not None:
                        conn.execute("ALTER TABLE authorized_targets RENAME TO authorized_targets_legacy")
                        conn.execute("""
                            CREATE TABLE authorized_targets (
                                target_name TEXT PRIMARY KEY,
                                granted_at REAL NOT NULL
                            )
                        """)
                        legacy_granted = next(
                            (name for name in ("granted_at", "authorized_at", "created_at") if name in columns),
                            None,
                        )
                        if legacy_granted is not None:
                            conn.execute(
                                f"INSERT OR IGNORE INTO authorized_targets(target_name, granted_at) "
                                f"SELECT {legacy_target}, COALESCE({legacy_granted}, ?) FROM authorized_targets_legacy "
                                f"WHERE {legacy_target} IS NOT NULL",
                                (time.time(),),
                            )
                        else:
                            conn.execute(
                                f"INSERT OR IGNORE INTO authorized_targets(target_name, granted_at) "
                                f"SELECT {legacy_target}, ? FROM authorized_targets_legacy "
                                f"WHERE {legacy_target} IS NOT NULL",
                                (time.time(),),
                            )
                        conn.execute("DROP TABLE authorized_targets_legacy")
                    else:
                        # Unknown legacy shape: fail closed by replacing only this
                        # auxiliary authorization table. No message/event/state
                        # data is discarded.
                        conn.execute("DROP TABLE authorized_targets")
                        conn.execute("""
                            CREATE TABLE authorized_targets (
                                target_name TEXT PRIMARY KEY,
                                granted_at REAL NOT NULL
                            )
                        """)

            # Event-memory schema migration for structured meeting fields.
            event_columns = {row[1] for row in conn.execute("PRAGMA table_info(intelligence_events)").fetchall()}
            for column in ("date", "time", "timezone", "platform", "location", "meeting_link"):
                if column not in event_columns:
                    conn.execute(f"ALTER TABLE intelligence_events ADD COLUMN {column} TEXT")

    def repair_meeting_event_times(self, *, on_debug: Callable[[str], None] | None = None) -> int:
        """Correct legacy meeting wall-clock values from persisted source evidence.

        This is a narrow data repair for the known event-time extraction bug. It
        never creates a new event and never changes synchronization/checkpoint
        state. Only an existing MEETING is updated when its stored evidence
        contains a deterministic event clock.
        """
        repaired = 0
        rows = self._conn().execute(
            "SELECT * FROM intelligence_events WHERE event_type=? ORDER BY updated_at DESC",
            (IntelligenceEventType.MEETING.value,),
        ).fetchall()
        for row in rows:
            event = _event_from_row(row)
            evidence_rows = self._conn().execute(
                "SELECT * FROM event_evidence WHERE event_id=? ORDER BY message_timestamp, evidence_id",
                (event.event_id,),
            ).fetchall()
            for evidence in evidence_rows:
                claim = _normalize_text(evidence["extracted_claim"] or "")
                if not claim:
                    continue
                match = _event_time_match(claim, meeting_context=True)
                parsed = _parse_clock_match(match) if match else None
                if parsed is None:
                    continue
                hour, minute = parsed
                corrected_time = f"{hour:02d}:{minute:02d}"
                if event.time == corrected_time:
                    continue
                event_tz = (
                    _timezone_from_value(event.timezone)
                    if event.timezone and ("/" in str(event.timezone) or str(event.timezone) in {"UTC", DEFAULT_EVENT_TIMEZONE})
                    else None
                ) or DEFAULT_EVENT_TZ
                effective_date = event.date or datetime.fromtimestamp(float(evidence["message_timestamp"]), tz=event_tz).date().isoformat()
                corrected_event_time = datetime.combine(
                    datetime.fromisoformat(effective_date).date(),
                    datetime.strptime(corrected_time, "%H:%M").time(),
                    tzinfo=event_tz,
                ).timestamp()
                def write(event_id: str = event.event_id, new_time: str = corrected_time, new_timestamp: float = corrected_event_time) -> None:
                    self._conn().execute(
                        "UPDATE intelligence_events SET time=?, event_time=?, timezone=COALESCE(timezone,?), updated_at=? WHERE event_id=?",
                        (new_time, new_timestamp, _timezone_label(event_tz), time.time(), event_id),
                    )
                self._run_write("meeting_event_time_repair", write)
                repaired += 1
                if on_debug is not None:
                    on_debug(
                        f"WHATSAPP_EVENT: TIME_REPAIRED event={event.event_id} "
                        f"previous={event.time or 'NONE'} corrected={corrected_time} "
                        f"timezone={_timezone_label(event_tz)}"
                    )
                break
        return repaired

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                if conn.in_transaction:
                    conn.rollback()
            finally:
                conn.close()
                self._local.conn = None
                self._local.transaction_depth = 0

    def state(self) -> WhatsAppIntelligenceState:
        row = self._conn().execute("SELECT * FROM intelligence_state WHERE id=1").fetchone()
        return WhatsAppIntelligenceState(
            enabled=bool(row["enabled"]), enabled_at=row["enabled_at"],
            last_processed_message_timestamp=row["last_processed_message_timestamp"],
            last_processed_message_id=row["last_processed_message_id"],
            last_observation_timestamp=row["last_observation_timestamp"],
        )

    def enable(self, now: float | None = None) -> WhatsAppIntelligenceState:
        current = self.state()
        if current.enabled and current.enabled_at is not None:
            return current
        ts = float(now if now is not None else time.time())
        # Intelligence starts with the configured five-day historical window.
        # This lets the first observation recover relevant messages from that
        # bounded source window instead of only messages sent after ENABLE.
        cutoff = ts - WHATSAPP_EVENT_LOOKBACK_DAYS * 24 * 60 * 60
        self._run_write("intelligence_state_enable", lambda: self._conn().execute(
            "UPDATE intelligence_state SET enabled=1, enabled_at=?, last_observation_timestamp=NULL WHERE id=1", (cutoff,)
        ))
        return self.state()

    def disable(self) -> WhatsAppIntelligenceState:
        self._run_write("intelligence_state_disable", lambda: self._conn().execute("UPDATE intelligence_state SET enabled=0 WHERE id=1"))
        return self.state()

    def observation_seen(self, now: float) -> None:
        self._run_write("observation_state", lambda: self._conn().execute(
            "UPDATE intelligence_state SET last_observation_timestamp=? WHERE id=1", (float(now),)
        ))

    def authorize_target(self, target_name: str, now: float | None = None) -> str:
        target = _normalize_text(target_name)
        if not target:
            raise ValueError("target_name must be non-empty")
        self._run_write("authorize_target", lambda: self._conn().execute(
            "INSERT OR REPLACE INTO authorized_targets(target_name, granted_at) VALUES(?, ?)",
            (target, float(now if now is not None else time.time())),
        ))
        return target

    def revoke_target(self, target_name: str) -> bool:
        target = _normalize_text(target_name)
        cur = self._run_write("revoke_target", lambda: self._conn().execute(
            "DELETE FROM authorized_targets WHERE target_name=?", (target,)
        ))
        return cur.rowcount > 0

    def authorized_targets(self) -> tuple[str, ...]:
        rows = self._conn().execute(
            "SELECT target_name FROM authorized_targets ORDER BY target_name COLLATE NOCASE"
        ).fetchall()
        return tuple(str(row["target_name"]) for row in rows)

    def is_message_processed(self, message_id: str) -> bool:
        """Read-only processed-message lookup used before the intelligence pipeline."""
        row = self._conn().execute(
            "SELECT 1 FROM processed_messages WHERE message_id=? LIMIT 1",
            (str(message_id),),
        ).fetchone()
        return row is not None

    def _claim_message_in_transaction(self, message: WhatsAppMessage, eligible: bool) -> bool:
        existing = self._conn().execute(
            "SELECT eligible FROM processed_messages WHERE message_id=?",
            (message.message_id,),
        ).fetchone()
        if existing is not None:
            if eligible and not bool(existing["eligible"]):
                cur = self._conn().execute(
                    "UPDATE processed_messages SET eligible=1, processed_at=? WHERE message_id=? AND eligible=0",
                    (time.time(), message.message_id),
                )
                if cur.rowcount == 1:
                    return True
            return False

        self._conn().execute(
            "INSERT INTO processed_messages(message_id, conversation_id, message_timestamp, processed_at, eligible) VALUES(?,?,?,?,?)",
            (message.message_id, message.conversation_id, message.timestamp, time.time(), int(bool(eligible))),
        )
        return True

    def advance_checkpoint(self, observed_messages: Iterable[WhatsAppMessage], *, eligible_after: float | None = None) -> tuple[bool, WhatsAppMessage | None]:
        """Advance the source boundary only after every observed item through it is incorporated.

        Per-message receipts remain the crash-safe source of truth. This method is
        only boundary reconciliation: an unprocessed eligible item blocks the
        frontier, while IGNORE/UPDATE/CREATE and every other committed receipt
        are equally incorporated for checkpoint purposes.
        """
        ordered = sorted(
            (m for m in observed_messages if eligible_after is None or m.timestamp >= eligible_after),
            key=lambda m: (m.timestamp, m.message_id),
        )
        if not ordered:
            return False, None
        with self._write_transaction("checkpoint_advance"):
            for message in ordered:
                row = self._conn().execute(
                    "SELECT eligible FROM processed_messages WHERE message_id=? LIMIT 1",
                    (message.message_id,),
                ).fetchone()
                if row is None or not bool(row["eligible"]):
                    return False, None
            boundary = ordered[-1]
            row = self._conn().execute(
                "SELECT last_processed_message_timestamp,last_processed_message_id FROM intelligence_state WHERE id=1"
            ).fetchone()
            current_ts = row["last_processed_message_timestamp"]
            current_id = row["last_processed_message_id"]
            current_key = None if current_ts is None else (float(current_ts), str(current_id or ""))
            boundary_key = (boundary.timestamp, boundary.message_id)
            if current_key is None or boundary_key > current_key:
                self._conn().execute(
                    "UPDATE intelligence_state SET last_processed_message_timestamp=?, last_processed_message_id=? WHERE id=1",
                    (boundary.timestamp, boundary.message_id),
                )
                return True, boundary
            # A newer persisted boundary already covers the candidate. Every
            # observed item through the candidate is incorporated, so this batch
            # is considered checkpoint-reconciled rather than blocked. The stored
            # frontier remains monotonic and is never moved backward.
            return True, boundary

    def claim_message(self, message: WhatsAppMessage, eligible: bool) -> bool:
        """Compatibility wrapper for callers that only need a standalone claim."""
        return bool(self._run_write(
            "processed_message_claim",
            lambda: self._claim_message_in_transaction(message, eligible),
        ))

    def _failure_retry_state(self, message_id: str, now: float | None = None) -> tuple[bool, int, float]:
        now = float(now if now is not None else time.time())
        row = self._conn().execute(
            "SELECT failure_count,retry_after FROM message_failures WHERE message_id=?",
            (message_id,),
        ).fetchone()
        if row is None:
            return True, 0, 0.0
        failure_count = int(row["failure_count"])
        retry_after = float(row["retry_after"])
        if failure_count >= _FAILED_RETRY_LIMIT:
            return False, failure_count, retry_after
        return now >= retry_after, failure_count, retry_after

    def record_message_failure(self, message: WhatsAppMessage, exc: BaseException, *, now: float | None = None) -> dict[str, Any]:
        now = float(now if now is not None else time.time())
        category = "event_evidence_schema_mismatch" if "event_evidence" in str(exc) and "column" in str(exc) else type(exc).__name__
        detail = str(exc).strip()[:500]
        def write() -> dict[str, Any]:
            row = self._conn().execute(
                "SELECT failure_count FROM message_failures WHERE message_id=?", (message.message_id,)
            ).fetchone()
            count = int(row["failure_count"]) + 1 if row else 1
            delay = min(_FAILED_RETRY_MAX_S, _FAILED_RETRY_BASE_S * (2 ** (count - 1)))
            retry_after = now + delay
            self._conn().execute(
                "INSERT INTO message_failures(message_id,conversation_id,failure_count,last_failure_at,retry_after,error_category,error_detail) "
                "VALUES(?,?,?,?,?,?,?) ON CONFLICT(message_id) DO UPDATE SET conversation_id=excluded.conversation_id, "
                "failure_count=excluded.failure_count,last_failure_at=excluded.last_failure_at,retry_after=excluded.retry_after, "
                "error_category=excluded.error_category,error_detail=excluded.error_detail",
                (message.message_id, message.conversation_id, count, now, retry_after, category, detail),
            )
            return {"count": count, "retry_after": retry_after, "category": category, "quarantined": count >= _FAILED_RETRY_LIMIT}
        return self._run_write("message_failure", write)

    def clear_message_failure(self, message_id: str) -> None:
        self._run_write("message_failure_clear", lambda: self._conn().execute(
            "DELETE FROM message_failures WHERE message_id=?", (message_id,)
        ))

    def process_message_atomically(
        self,
        message: WhatsAppMessage,
        eligible: bool,
        processor: Callable[[], list[IntelligenceEvent]],
        *,
        operation: str = "message_process",
    ) -> tuple[str, list[IntelligenceEvent]]:
        """Process one message and commit all derived state atomically.

        BrowserSkill/LLM work is never performed here. The callback is limited
        to deterministic event matching/mutation and summary persistence. A
        busy/locked database causes the whole short transaction to retry; a
        non-busy SQLite error is surfaced immediately.
        """
        for attempt in range(_DB_BUSY_RETRIES + 1):
            try:
                with self._write_transaction(operation):
                    if not self._claim_message_in_transaction(message, eligible):
                        return "ALREADY_PROCESSED", []
                    if not eligible:
                        return "IGNORED", []
                    events = processor()
                    self._conn().execute("DELETE FROM message_failures WHERE message_id=?", (message.message_id,))
                    return "PERSISTED", events
            except sqlite3.Error as exc:
                if self._is_retryable_busy(exc) and attempt < _DB_BUSY_RETRIES:
                    time.sleep(_DB_RETRY_SLEEP_S * (attempt + 1))
                    continue
                raise
        raise RuntimeError("database transaction retry loop exhausted")

    def get_event(self, event_id: str) -> IntelligenceEvent | None:
        row = self._conn().execute("SELECT * FROM intelligence_events WHERE event_id=?", (event_id,)).fetchone()
        return _event_from_row(row) if row else None

    def find_mergeable(self, conversation_id: str, event_type: IntelligenceEventType, title: str = "", statuses: Iterable[str] = ("PROPOSED", "LIKELY", "CONFIRMED", "MODIFIED", "CANCELLED")) -> IntelligenceEvent | None:
        candidates = self.find_event_candidates(conversation_id, event_type, statuses=statuses)
        if not title:
            return candidates[0] if candidates else None
        wanted = set(re.findall(r"[a-z0-9]+", title.casefold()))
        scored = []
        for event in candidates:
            tokens = set(re.findall(r"[a-z0-9]+", event.title.casefold()))
            score = len(wanted & tokens)
            scored.append((score, event.updated_at, event))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2] if scored and scored[0][0] > 0 else None

    def find_event_candidates(self, conversation_id: str, event_type: IntelligenceEventType, *, statuses: Iterable[str] = ("PROPOSED", "LIKELY", "CONFIRMED", "MODIFIED", "CANCELLED"), limit: int = 20) -> list[IntelligenceEvent]:
        statuses = tuple(statuses)
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        rows = self._conn().execute(
            f"SELECT * FROM intelligence_events WHERE conversation_id=? AND event_type=? AND status IN ({placeholders}) ORDER BY updated_at DESC LIMIT ?",
            (conversation_id, event_type.value, *statuses, int(limit)),
        ).fetchall()
        return [_event_from_row(row) for row in rows]

    def add_event_evidence(self, event_id: str, message: WhatsAppMessage, evidence_type: str, extracted_claim: str, confidence: float) -> bool:
        def write() -> bool:
            cur = self._conn().execute(
                "INSERT OR IGNORE INTO event_evidence(evidence_id,event_id,message_id,conversation_id,message_timestamp,evidence_type,extracted_claim,confidence,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                ("evi-" + hashlib.sha256(f"{event_id}\0{message.message_id}\0{evidence_type}".encode()).hexdigest()[:24], event_id, message.message_id, message.conversation_id, message.timestamp, evidence_type, _normalize_text(extracted_claim), float(confidence), time.time()),
            )
            return cur.rowcount == 1
        return bool(self._run_write("event_evidence_insert", write))

    def event_evidence(self, event_id: str) -> list[dict[str, Any]]:
        rows = self._conn().execute("SELECT * FROM event_evidence WHERE event_id=? ORDER BY message_timestamp, evidence_id", (event_id,)).fetchall()
        return [dict(row) for row in rows]

    def upsert_event(self, event: IntelligenceEvent) -> None:
        def write() -> None:
            self._conn().execute("""INSERT INTO intelligence_events(
                event_id,conversation_id,event_type,title,description,status,confidence,importance,urgency,
                created_at,updated_at,event_time,deadline,date,time,timezone,platform,location,meeting_link,
                participants_json,evidence_json,source_message_ids_json,scheduler_candidate)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(event_id) DO UPDATE SET
                description=excluded.description,status=excluded.status,confidence=excluded.confidence,
                importance=excluded.importance,urgency=excluded.urgency,updated_at=excluded.updated_at,
                event_time=COALESCE(excluded.event_time,intelligence_events.event_time),
                deadline=COALESCE(excluded.deadline,intelligence_events.deadline),
                date=COALESCE(excluded.date,intelligence_events.date),
                time=COALESCE(excluded.time,intelligence_events.time),
                timezone=COALESCE(excluded.timezone,intelligence_events.timezone),
                platform=COALESCE(excluded.platform,intelligence_events.platform),
                location=COALESCE(excluded.location,intelligence_events.location),
                meeting_link=COALESCE(excluded.meeting_link,intelligence_events.meeting_link),
                participants_json=excluded.participants_json,evidence_json=excluded.evidence_json,
                source_message_ids_json=excluded.source_message_ids_json,scheduler_candidate=excluded.scheduler_candidate""",
                (event.event_id,event.conversation_id,event.type.value,event.title,event.description,event.status,event.confidence,
                 event.importance,event.urgency,event.created_at,event.updated_at,event.event_time,event.deadline,event.date,event.time,
                 event.timezone,event.platform,event.location,event.meeting_link,json.dumps(event.participants),json.dumps(event.evidence),
                 json.dumps(event.source_message_ids),int(event.scheduler_candidate)))
        self._run_write("event_upsert", write)

    def list_events(self) -> list[IntelligenceEvent]:
        return [_event_from_row(r) for r in self._conn().execute("SELECT * FROM intelligence_events ORDER BY created_at, event_id").fetchall()]

    def summary(self, conversation_id: str) -> dict[str, Any] | None:
        row = self._conn().execute("SELECT * FROM conversation_summaries WHERE conversation_id=?", (conversation_id,)).fetchone()
        return dict(row) if row else None

    def update_summary(self, conversation_id: str, summary: str, message_id: str) -> None:
        now = time.time()
        old = self.summary(conversation_id)
        version = int(old["summary_version"] + 1) if old else 1
        self._run_write("summary_upsert", lambda: self._conn().execute("""INSERT INTO conversation_summaries(conversation_id,current_summary,summary_version,last_summary_message_id,updated_at)
                VALUES(?,?,?,?,?) ON CONFLICT(conversation_id) DO UPDATE SET current_summary=excluded.current_summary,
                summary_version=excluded.summary_version,last_summary_message_id=excluded.last_summary_message_id,updated_at=excluded.updated_at""",
                (conversation_id, summary, version, message_id, now)))


def _merge_evidence(a: Iterable[dict[str, Any]], b: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in (*tuple(a), *tuple(b)):
        key = str(item.get("message_id") or json.dumps(item, sort_keys=True, default=str))
        if key not in seen:
            seen.add(key); out.append(dict(item))
    return out


def _event_from_row(row: sqlite3.Row) -> IntelligenceEvent:
    return IntelligenceEvent(
        event_id=row["event_id"], conversation_id=row["conversation_id"], type=IntelligenceEventType(row["event_type"]),
        title=row["title"], description=row["description"], status=row["status"], confidence=float(row["confidence"]),
        importance=float(row["importance"]), urgency=float(row["urgency"]), created_at=float(row["created_at"]), updated_at=float(row["updated_at"]),
        event_time=row["event_time"], deadline=row["deadline"],
        date=row["date"] if "date" in row.keys() else None, time=row["time"] if "time" in row.keys() else None,
        timezone=row["timezone"] if "timezone" in row.keys() else None, platform=row["platform"] if "platform" in row.keys() else None,
        location=row["location"] if "location" in row.keys() else None, meeting_link=row["meeting_link"] if "meeting_link" in row.keys() else None,
        participants=tuple(json.loads(row["participants_json"] or "[]")),
        evidence=tuple(json.loads(row["evidence_json"] or "[]")), source_message_ids=tuple(json.loads(row["source_message_ids_json"] or "[]")),
        scheduler_candidate=bool(row["scheduler_candidate"]),
    )


_LOW = re.compile(r"^(?:ok(?:ay)?|okay|lol|lmao|haan|hmm+|yes|no|fine|sure|thanks|thank you|hi|hello|hey|bye|😂+|👍+|😄+|🙂+)[!. ]*$", re.I)
_MEANINGFUL = re.compile(r"\b(meet|meeting|tomorrow|today|deadline|due|assignment|assign|task|todo|decide|decision|confirm|confirmed|project|submit|announce|announcement|important|idea|suggest|resolve|resolved|conflict|issue|action item|action)\b", re.I)
_HIGH = re.compile(r"\b(confirmed|deadline|due|assignment|assigned|must|important|announcement|decision|decided|action item|resolve|resolved|meeting at|meeting on)\b", re.I)


def relevance(text: str) -> Relevance:
    value = _normalize_text(text)
    if not value or _LOW.fullmatch(value):
        return Relevance.LOW_VALUE
    if _HIGH.search(value):
        return Relevance.HIGH_VALUE
    if re.search(r"\b(?:wednesday|thursday|friday|saturday|sunday|monday|tuesday|discord|zoom|teams|google meet|meet|meeting|cancel(?:led|l)?|confirmed|actually|same link|move it|kar di|kar diya)\b", value, re.I):
        return Relevance.POSSIBLY_MEANINGFUL
    if _MEANINGFUL.search(value):
        return Relevance.POSSIBLY_MEANINGFUL
    return Relevance.LOW_VALUE


def _event_time_from_text(text: str, base: float) -> float | None:
    """Legacy helper using the canonical DEIMOS event-time semantics."""
    lower = _normalize_text(text).casefold()
    meeting_context = bool(re.search(r"\b(?:meeting|meet|call)\b", lower))
    time_match = _event_time_match(text, meeting_context=meeting_context)
    if time_match is None:
        return None
    parsed = _parse_clock_match(time_match)
    if parsed is None:
        return None
    hour, minute = parsed
    event_tz = DEFAULT_EVENT_TZ
    base_local = datetime.fromtimestamp(base, tz=event_tz)
    if re.search(r"\btomorrow\b", lower):
        event_date = (base_local + timedelta(days=1)).date()
    elif re.search(r"\btoday\b", lower):
        event_date = base_local.date()
    else:
        event_date = base_local.date()
    return datetime.combine(
        event_date,
        datetime.strptime(f"{hour:02d}:{minute:02d}", "%H:%M").time(),
        tzinfo=event_tz,
    ).timestamp()


def _deadline_from_text(text: str, base: float) -> float | None:
    lower = text.casefold()
    m = re.search(r"\bdue\s+(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lower)
    if not m:
        return None
    token = m.group(1); base_dt = datetime.fromtimestamp(base, tz=timezone.utc)
    if token == "today": return base_dt.replace(hour=23, minute=59, second=59, microsecond=0).timestamp()
    if token == "tomorrow": return (base_dt + timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0).timestamp()
    target = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"].index(token)
    delta = (target - base_dt.weekday()) % 7
    if delta == 0: delta = 7
    return (base_dt + timedelta(days=delta)).replace(hour=23, minute=59, second=59, microsecond=0).timestamp()


def _observer_error_category(exc: BaseException) -> str:
    text = str(exc or "").casefold()
    if "scroll_to" in text or "scroll-to" in text:
        if "not implemented" in text or "unsupported" in text or "unrecognized" in text:
            return "scroll_to_unsupported"
        return "scroll_to_failed"
    if "wheel" in text and ("failed" in text or "not implemented" in text or "unrecognized" in text or "unsupported" in text):
        return "wheel_fallback_failed"
    if any(token in text for token in ("connection", "connect", "disconnected", "closed", "websocket", "browser unavailable")):
        return "browser_connection_failed"
    if any(token in text for token in ("execute", "execution", "session command", "command failed")):
        return "execution_exception"
    return "browser_observation_failed"


class WhatsAppIntelligence:
    """Pure incremental intelligence engine; browser ownership is supplied by caller."""
    def __init__(self, store: WhatsAppIntelligenceStore, *, can_observe: Callable[[], bool] | None = None, on_debug: Callable[[str], None] | None = None, on_intelligence: Callable[[IntelligenceEvent], None] | None = None) -> None:
        self.store = store
        self.can_observe = can_observe or (lambda: True)
        self.on_debug = on_debug or (lambda _line: None)
        self.on_intelligence = on_intelligence or (lambda _event: None)
        self.store.set_debug_callback(self.on_debug)
        self._observer_lock = threading.RLock()
        self._observer_thread: threading.Thread | None = None
        self._observer_stop = threading.Event()
        self._observer_browser: Any | None = None
        self._observer_interval = 2.0
        self._last_gate_state: str | None = None
        self._last_observer_error: tuple[str, float] | None = None
        self._direction_diagnostics_emitted: set[tuple[str, str]] = set()
        self._home_scroll_budget = _HOME_SCROLL_BUDGET
        self._scroll_to_unsupported = False
        self._home_opened_fingerprints: set[str] = set()
        self._observer_cycle = 0
        self._event_time_repair_done = False

    def start_observing(self, browser: Any, *, interval_s: float = 2.0) -> None:
        """Start bounded polling using the existing BrowserSkill resource.

        The worker is session-scoped and read-only.  Ownership is checked before
        every observation; HUMAN ownership therefore pauses the worker without
        touching the browser.  The worker never performs browser mutations.
        """
        with self._observer_lock:
            # A control takeover can hand the same WhatsApp resource back to the
            # service after the worker has already been started during ENABLE.
            # Refresh the browser reference rather than leaving the worker bound
            # to an older adapter instance.
            if self._observer_thread is not None and self._observer_thread.is_alive():
                self._observer_browser = browser
                self._observer_interval = max(0.5, float(interval_s))
                return
            self._observer_stop = threading.Event()
            self._observer_browser = browser
            self._observer_interval = max(0.5, float(interval_s))
            self._last_gate_state = None
            self._observer_thread = threading.Thread(
                target=self._observe_loop,
                name="deimos-whatsapp-intelligence",
                daemon=True,
            )
            self._observer_thread.start()
        self.on_debug("WHATSAPP_INTELLIGENCE: OBSERVER_STARTED")

    @property
    def observer_running(self) -> bool:
        thread = self._observer_thread
        return thread is not None and thread.is_alive()

    def stop_observing(self) -> None:
        with self._observer_lock:
            event = self._observer_stop
            event.set()
            thread = self._observer_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._observer_lock:
            if self._observer_thread is thread:
                self._observer_thread = None
                self._observer_browser = None
                self._last_gate_state = None
                self._scroll_to_unsupported = False
    
    def _observe_loop(self) -> None:
        while not self._observer_stop.is_set():
            try:
                enabled = self.store.state().enabled
            except sqlite3.Error as exc:
                self.store._db_debug("observer_state_read", exc)
                self.on_debug(
                    f"WHATSAPP_OBSERVER: processing=FAILED reason=DATABASE_ERROR "
                    f"retryable={self.store._is_retryable_busy(exc)}"
                )
                self._observer_stop.wait(self._observer_interval)
                continue
            allowed = enabled and self.can_observe()
            gate = "RUNNING" if allowed else ("DISABLED" if not enabled else "PAUSED")
            if gate != self._last_gate_state:
                if gate == "PAUSED":
                    self.on_debug("WHATSAPP_INTELLIGENCE: OBSERVER_PAUSED owner=HUMAN_OR_ACTIVE_TASK")
                elif gate == "RUNNING" and self._last_gate_state == "PAUSED":
                    self.on_debug("WHATSAPP_INTELLIGENCE: OBSERVER_RESUMED")
                self._last_gate_state = gate
            if not allowed:
                self._observer_stop.wait(self._observer_interval)
                continue
            try:
                # BrowserSkill.observe() is the existing fresh semantic
                # observation boundary.  Do not substitute DOM/screenshot
                # scraping here.
                browser = self._observer_browser
                if browser is None:
                    raise RuntimeError("whatsapp_intelligence_browser_unavailable")
                observation = browser.observe()
                raw = getattr(browser, "_last_observation", None)
                current = raw if raw is not None else observation
                active_target = self._active_authorized_target(current)
                if active_target:
                    # A fresh conversation observation contains the chat list
                    # sidebar as well as the message pane. Process the current
                    # conversation first, then allow ONE additional visible
                    # authorized Home-row candidate to be opened without
                    # semantic scrolling. Global scroll is unsafe here because
                    # it may scroll the message pane instead of the chat list.
                    extracted = extract_messages(current, conversation_hint=active_target, observed_at=time.time())
                    if extracted:
                        self.process_observation(current, conversation_hint=active_target)
                    else:
                        snapshotter = getattr(browser, "snapshot", None)
                        if callable(snapshotter):
                            snapshot = snapshotter()
                            if extract_messages(snapshot, conversation_hint=active_target, observed_at=time.time()):
                                self.on_debug("WHATSAPP_HOME: CHAT_MESSAGE_FALLBACK=SNAPSHOT")
                                self.process_observation(snapshot, conversation_hint=active_target)
                    opened_target = self._observe_authorized_targets(
                        browser, current, allow_scroll=False
                    )
                    if opened_target:
                        continue
                    continue

                opened_target = self._observe_authorized_targets(
                    browser, current, allow_scroll=True
                )
                if opened_target:
                    continue
            except Exception as exc:
                # Intelligence failures are contained to this worker; the main
                # conversational runtime must remain alive. Do not flood the
                # interactive chat with the same background error every poll.
                error_name = _observer_error_category(exc)
                now = time.monotonic()
                previous = self._last_observer_error
                if previous is None or previous[0] != error_name or now - previous[1] >= 10.0:
                    self.on_debug(f"WHATSAPP_OBSERVER: observation_error={error_name}")
                    self._last_observer_error = (error_name, now)
            self._observer_stop.wait(self._observer_interval)

    def _semantic_elements(self, observation: Any) -> list[Any]:
        value = getattr(observation, "elements", None)
        if value is not None:
            return list(value)
        raw = getattr(observation, "raw", observation)
        if isinstance(raw, dict):
            elements = raw.get("elements")
            if isinstance(elements, (list, tuple)):
                return list(elements)
        return []

    @staticmethod
    def _element_name(element: Any) -> str:
        if isinstance(element, dict):
            return _normalize_text(element.get("name", ""))
        return _normalize_text(getattr(element, "name", ""))

    @staticmethod
    def _element_role(element: Any) -> str:
        if isinstance(element, dict):
            return str(element.get("role", "") or "").casefold()
        return str(getattr(element, "role", "") or "").casefold()

    def _active_authorized_target(self, observation: Any) -> str | None:
        """Return the authorized target when a WhatsApp conversation is open.

        The desktop layout keeps the Home chat-list sidebar visible beside the
        active conversation. A composer is therefore the strong state signal;
        an authorized target merely needs to occur in the same fresh semantic
        observation. This avoids depending on one particular header label
        (self-chat may say ``You``) and prevents the observer from scrolling the
        message pane as if it were the Home list.
        """
        targets = self.store.authorized_targets()
        if not targets:
            return None
        elements = self._semantic_elements(observation)
        names = [self._element_name(e) for e in elements if self._element_name(e)]
        raw = getattr(observation, "raw", observation)
        raw_text = getattr(observation, "text", "")
        if isinstance(raw_text, str):
            names.append(raw_text)

        composer_visible = any(
            self._element_role(e) in {"textbox", "combobox"}
            or "type a message" in name.casefold()
            or "message input" in name.casefold()
            for e, name in ((e, self._element_name(e)) for e in elements)
        )
        if not composer_visible:
            return None

        haystack = "\n".join(names).casefold()
        for target in targets:
            if re.search(rf"(?<!\w){re.escape(target.casefold())}(?!\w)", haystack):
                return target
        return None

    def _home_row_target(
        self,
        element: Any,
        authorized_target: str,
        observation: Any,
    ) -> Any | None:
        """Resolve one authorized Home row from the CURRENT observation.

        WhatsApp's semantic tree may expose a row as a single element whose
        accessible name contains the chat name, unread count, time and preview.
        We intentionally require the authorized name to occur in that same
        row and never resolve the WhatsApp search control or a search result.
        """
        role = self._element_role(element)
        if role not in {"button", "link", "listitem", "option"}:
            return None
        if self._element_role(element) in {"textbox", "combobox", "searchbox"}:
            return None
        name = self._element_name(element)
        if not name:
            return None
        target_folded = authorized_target.casefold()
        name_folded = name.casefold()
        if name_folded == target_folded:
            return element
        # Row labels commonly include metadata around the contact/group name.
        # Match the authorized name as a complete phrase, not as a substring
        # of another contact/group name.
        if re.search(rf"(?<!\w){re.escape(target_folded)}(?!\w)", name_folded):
            return element
        return None

    def _row_activity_reason(self, element: Any) -> str | None:
        """Return a safe Home-row activity signal for historical discovery.

        The first intelligence observation has a 24-hour cutoff, so an
        authorized row can be opened even when its messages are already read
        if the Home UI exposes an explicit recent-time/date marker. Unread is
        preferred, but it is not required for the historical bootstrap.
        """
        values = [self._element_name(element)]
        if isinstance(element, dict):
            values.append(str(element.get("value", "") or ""))
            raw = element.get("raw")
            attrs = element.get("attributes")
        else:
            values.append(str(getattr(element, "value", "") or ""))
            raw = getattr(element, "raw", None)
            attrs = getattr(element, "attributes", None)
        if isinstance(attrs, dict):
            values.extend(str(v) for v in attrs.values())
        if isinstance(raw, dict):
            values.extend(str(v) for v in raw.values())
        haystack = " ".join(values).casefold()
        if re.search(r"\b(?:\d+\s+)?unread(?:\s+messages?)?\b", haystack):
            return "unread"

        # WhatsApp Home rows expose a displayed clock for recent messages and
        # labels such as Today/Yesterday for older recent rows. These are UI
        # evidence only: the actual 24-hour cutoff is still enforced after
        # the conversation is opened by process_observation().
        if re.search(r"\b(?:today|yesterday)\b", haystack, re.I):
            return "recent"
        if re.search(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b", haystack):
            return "recent"
        if re.search(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\s*(?:am|pm)\b", haystack, re.I):
            return "recent"
        return None

    def _home_candidates(self, observation: Any) -> tuple[list[tuple[str, Any, str]], set[str]]:
        targets = self.store.authorized_targets()
        elements = self._semantic_elements(observation)
        candidates: list[tuple[str, Any, str]] = []
        visible_authorized: set[str] = set()

        # WhatsApp may expose the clickable row as a listitem containing only
        # the contact name, with preview/time/unread metadata as sibling
        # semantic elements. Build a bounded row segment around each listitem
        # instead of requiring all metadata to live in one element.
        row_starts = [i for i, e in enumerate(elements) if self._element_role(e) in {"listitem"}]
        for target in targets:
            target_folded = target.casefold()
            matching_rows: list[tuple[Any, list[Any]]] = []
            for pos, start in enumerate(row_starts):
                end = row_starts[pos + 1] if pos + 1 < len(row_starts) else len(elements)
                segment = elements[start:end]
                joined = " ".join(self._element_name(e) for e in segment if self._element_name(e))
                if not re.search(rf"(?<!\w){re.escape(target_folded)}(?!\w)", joined.casefold()):
                    continue
                # Prefer the live clickable child inside the matched Home row.
                # WhatsApp can expose the container as ``listitem`` while the
                # actual actionable target is a nested button/link. Clicking
                # the container happens to work for some chats (e.g. Mummy)
                # but can fail for self-chat/other special rows.
                clickable_roles = {"button", "link", "option"}
                # Prefer the live actionable element whose own semantic name
                # contains the authorized target. This avoids accidentally
                # selecting a sibling menu/avatar control in special rows such
                # as self-chat. WhatsApp can also expose a clickable row as a
                # listitem, so keep the listitem only as a final fallback.
                row = next(
                    (
                        item
                        for item in segment
                        if self._element_role(item) in clickable_roles
                        and re.search(
                            rf"(?<!\w){re.escape(target_folded)}(?!\w)",
                            self._element_name(item).casefold(),
                        )
                    ),
                    None,
                )
                if row is None:
                    row = next(
                        (
                            item
                            for item in segment
                            if self._element_role(item) in clickable_roles
                        ),
                        elements[start],
                    )
                matching_rows.append((row, segment))

            if not matching_rows:
                # Fallback for WhatsApp builds that expose each conversation as
                # a standalone button/link rather than a listitem. Keep nearby
                # non-clickable semantic siblings with the matched row so time,
                # preview, and unread markers can still form activity evidence.
                row_roles = {"button", "link", "option"}
                for index, element in enumerate(elements):
                    role = self._element_role(element)
                    name = self._element_name(element)
                    if role not in row_roles or not re.search(
                        rf"(?<!\w){re.escape(target_folded)}(?!\w)", name.casefold()
                    ):
                        continue
                    segment = [element]
                    end = index + 1
                    while end < len(elements):
                        next_role = self._element_role(elements[end])
                        next_name = self._element_name(elements[end])
                        if next_role in row_roles and next_name:
                            break
                        segment.append(elements[end])
                        end += 1
                    matching_rows.append((element, segment))
                    break

            if not matching_rows:
                continue
            row, segment = matching_rows[0]
            visible_authorized.add(target_folded)
            combined = " ".join(
                " ".join(filter(None, (self._element_name(e), self._element_value(e))))
                for e in segment
            ).strip()
            reason = self._row_activity_reason_text(combined)
            # Ignore transient unread/time presentation state when deciding
            # whether this exact Home-row activity was already opened. A read
            # transition must not cause the same conversation to be reopened.
            fingerprint_source = re.sub(
                r"\b(?:\d+\s+)?unread(?:\s+messages?)?\b",
                " ",
                _normalize_text(combined),
                flags=re.IGNORECASE,
            )
            fingerprint_source = re.sub(
                r"\b(?:today|yesterday)\b", " ", fingerprint_source, flags=re.IGNORECASE
            )
            fingerprint_source = re.sub(
                r"\s+(?:[01]?\d|2[0-3]):[0-5]\d\s*(?:am|pm)?\s*$",
                " ",
                fingerprint_source,
                flags=re.IGNORECASE,
            )
            fingerprint = hashlib.sha256(
                (target_folded + "\0" + _normalize_text(fingerprint_source).casefold()).encode("utf-8")
            ).hexdigest()
            if reason is not None and fingerprint not in self._home_opened_fingerprints:
                candidates.append((target, row, fingerprint))
                self.on_debug(f"WHATSAPP_HOME: CANDIDATE target={target!r} reason={reason}")
        return candidates, visible_authorized

    @staticmethod
    def _element_value(element: Any) -> str:
        if isinstance(element, dict):
            return _normalize_text(element.get("value", ""))
        return _normalize_text(getattr(element, "value", ""))

    @staticmethod
    def _row_activity_reason_text(text: str) -> str | None:
        haystack = str(text or "").casefold()
        if re.search(r"\b(?:\d+\s+)?unread(?:\s+messages?)?\b", haystack):
            return "unread"
        if re.search(r"\b(?:today|yesterday)\b", haystack):
            return "recent"
        if re.search(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\s*(?:am|pm)?\b", haystack, re.I):
            return "recent"
        return None

    def _observe_authorized_targets(self, browser: Any, observation: Any, *, allow_scroll: bool = True) -> str | None:
        targets = self.store.authorized_targets()
        if not targets:
            return None

        # Discovery is deliberately Home-row-only. If the current observation
        # has no authorized candidate, bounded semantic scrolling may expose
        # more rows. There is no WhatsApp search fallback.
        seen_pages: set[tuple[str, ...]] = set()
        for scroll_index in range(self._home_scroll_budget + 1):
            candidates, _visible_authorized = self._home_candidates(observation)
            page_signature = tuple(
                self._element_name(e)
                for e in self._semantic_elements(observation)
                if self._element_role(e) in {"button", "link", "listitem", "option"}
            )
            if page_signature in seen_pages:
                break
            seen_pages.add(page_signature)

            if candidates:
                target, row, fingerprint = candidates[0]
                opener = getattr(browser, "open_whatsapp_chat_row", None)
                if not callable(opener):
                    self.on_debug("WHATSAPP_HOME: OPEN_ROW_SKIPPED reason=home_row_open_unavailable")
                    return None
                try:
                    self.on_debug(f"WHATSAPP_HOME: OPEN_ROW target={target!r}")
                    result = opener(
                        self._make_browser_target(row, observation, name=target),
                        timeout_s=8.0,
                    )
                    if not getattr(result, "ok", True):
                        return None
                    self._home_opened_fingerprints.add(fingerprint)
                    self.on_debug(f"WHATSAPP_HOME: CHAT_OPENED target={target!r}")
                    opened = browser.observe()
                    opened_raw = getattr(browser, "_last_observation", None)
                    current = opened_raw if opened_raw is not None else opened
                    self.on_debug("WHATSAPP_HOME: CHAT_OBSERVATION_FRESH")
                    extracted = extract_messages(current, conversation_hint=target, observed_at=time.time())
                    if extracted:
                        self.process_observation(current, conversation_hint=target)
                    else:
                        # BrowserSkill documents ``snapshot`` as the stricter
                        # static accessibility fallback when VOM observation is
                        # insufficient. Keep observe as the primary boundary;
                        # use snapshot only when no message evidence was found.
                        snapshotter = getattr(browser, "snapshot", None)
                        if callable(snapshotter):
                            snapshot = snapshotter()
                            snapshot_messages = extract_messages(snapshot, conversation_hint=target, observed_at=time.time())
                            if snapshot_messages:
                                self.on_debug("WHATSAPP_HOME: CHAT_MESSAGE_FALLBACK=SNAPSHOT")
                                self.process_observation(snapshot, conversation_hint=target)
                    # WhatsApp Web keeps the chat list visible beside the
                    # conversation. Do not use browser history ``go_back`` here:
                    # opening a chat is an in-app state change and browser back
                    # can leave the resource in an invalid/unknown state.
                    return target
                except Exception as exc:
                    code = getattr(exc, "code", "") or ""
                    suffix = f" code={code!r}" if code else ""
                    self.on_debug(
                        f"WHATSAPP_HOME: OPEN_ROW_ERROR target={target!r} error={type(exc).__name__}{suffix}"
                    )
                    return None

            if not allow_scroll or scroll_index >= self._home_scroll_budget:
                break

            elements = [
                e for e in self._semantic_elements(observation)
                if self._element_role(e) in {"button", "link", "listitem", "option"}
                and self._element_role(e) not in {"textbox", "combobox", "searchbox"}
            ]
            if not elements:
                break
            anchor = elements[-1]
            if scroll_index == 0:
                self.on_debug("WHATSAPP_HOME: SCROLLING_FOR_AUTHORIZED_TARGET")

            scrolled = False
            scroll_to = getattr(browser, "scroll_to", None)
            use_wheel_fallback = not callable(scroll_to) or getattr(self, "_scroll_to_unsupported", False)
            if callable(scroll_to) and not getattr(self, "_scroll_to_unsupported", False):
                try:
                    result = scroll_to(self._make_browser_target(anchor, observation))
                    if getattr(result, "ok", True):
                        scrolled = True
                    else:
                        error_text = str(result.get("error") or result.get("message") or "") if isinstance(result, dict) else str(result)
                        if "scroll_to" in error_text.casefold() and "not implemented" in error_text.casefold():
                            self._scroll_to_unsupported = True
                            use_wheel_fallback = True
                            self.on_debug("WHATSAPP_DEBUG: scroll_to_unsupported")
                        else:
                            self.on_debug("WHATSAPP_DEBUG: scroll_to_failed")
                            break
                except Exception as exc:
                    error_text = str(exc).casefold()
                    if ("scroll_to" in error_text or "scroll-to" in error_text) and any(token in error_text for token in ("not implemented", "unsupported", "unrecognized")):
                        self._scroll_to_unsupported = True
                        use_wheel_fallback = True
                        self.on_debug("WHATSAPP_DEBUG: scroll_to_unsupported")
                    else:
                        self.on_debug("WHATSAPP_DEBUG: scroll_to_failed")
                        break

            if not scrolled and use_wheel_fallback:
                wheel = getattr(browser, "wheel", None)
                if not callable(wheel):
                    self.on_debug("WHATSAPP_DEBUG: wheel_fallback_failed reason=unavailable")
                    break
                self.on_debug("WHATSAPP_DEBUG: wheel_fallback_attempted")
                try:
                    wheel_result = wheel(_HOME_WHEEL_DELTA_Y)
                except Exception:
                    self.on_debug("WHATSAPP_DEBUG: wheel_fallback_failed")
                    break
                if not getattr(wheel_result, "ok", True):
                    self.on_debug("WHATSAPP_DEBUG: wheel_fallback_failed")
                    break

            if not scrolled and not use_wheel_fallback:
                break

            fresh = browser.observe()
            observation = getattr(browser, "_last_observation", None) or fresh

        return None

    @staticmethod
    def _make_browser_target(element: Any, observation: Any, *, name: str | None = None) -> Any:
        """Build the existing BrowserTarget capability from the live row ref."""
        from agent_control.skills.browser.backend import BrowserTarget
        return BrowserTarget(
            getattr(element, "ref", ""),
            role=getattr(element, "role", ""),
            name=name if name is not None else getattr(element, "name", ""),
            raw=getattr(element, "raw", None),
            generation=getattr(observation, "generation", None),
        )

    def enable(self, now: float | None = None) -> WhatsAppIntelligenceState:
        state = self.store.enable(now)
        self.on_debug("WHATSAPP_INTELLIGENCE: ENABLED")
        self.on_debug(f"LOOKBACK_DAYS: {WHATSAPP_EVENT_LOOKBACK_DAYS}")
        if state.enabled_at is not None:
            start = datetime.fromtimestamp(state.enabled_at, tz=timezone.utc).isoformat()
            end = datetime.fromtimestamp(state.enabled_at + WHATSAPP_EVENT_LOOKBACK_DAYS * 24 * 60 * 60, tz=timezone.utc).isoformat()
            self.on_debug(f"CATCHUP_WINDOW: start={start} end={end}")
            self.on_debug(f"CUTOFF: {start}")
        self.on_debug("WHATSAPP_INTELLIGENCE: Message intelligence uses the previous 5 days; persistent event memory is retained beyond that window.")
        return state

    def disable(self) -> WhatsAppIntelligenceState:
        return self.store.disable()

    def process_observation(self, observation: Any, *, observed_at: float | None = None, conversation_hint: str = "") -> list[IntelligenceEvent]:
        try:
            state = self.store.state()
            if not state.enabled or not self.can_observe():
                return []
            now = float(observed_at if observed_at is not None else time.time())
            self.store.observation_seen(now)
        except sqlite3.Error as exc:
            self.store._db_debug("observation_state", exc)
            self.on_debug(
                f"WHATSAPP_OBSERVER: processing=FAILED reason=DATABASE_ERROR "
                f"retryable={self.store._is_retryable_busy(exc)}"
            )
            return []

        self._observer_cycle += 1
        cycle = self._observer_cycle
        mode = "INITIAL_CATCHUP" if state.last_observation_timestamp is None else "LIVE"
        if not self._event_time_repair_done:
            self._event_time_repair_done = True
            try:
                self.store.repair_meeting_event_times(on_debug=self.on_debug)
            except Exception as exc:
                self.on_debug(f"WHATSAPP_EVENT: TIME_REPAIR_FAILED error={type(exc).__name__}")
        messages = extract_messages(observation, conversation_hint=conversation_hint, observed_at=now)

        new_messages: list[WhatsAppMessage] = []
        already = 0
        out_of_window = 0
        retry_deferred = 0
        failed = 0
        incorporated = 0
        sync_error_reason: str | None = None

        # Deduplication is deliberately before relevance or extraction. A message
        # already committed to processed_messages is not an intelligence candidate
        # for this or any later polling cycle. Out-of-window messages are also
        # claimed as non-intelligence work once, so repeated visible history does
        # not look new on every poll.
        for message in messages:
            # Apply the temporal boundary before the processed lookup. The event
            # source window is a rolling maximum of five days, additionally
            # bounded by the persisted enable/catch-up cutoff. This prevents a
            # stale visible message from being rediscovered merely because the
            # observer has remained running for longer than five days.
            rolling_cutoff = now - WHATSAPP_EVENT_LOOKBACK_DAYS * 86400
            is_eligible = state.enabled_at is not None and message.timestamp >= max(state.enabled_at, rolling_cutoff)
            if not is_eligible:
                if self.store.is_message_processed(message.message_id):
                    already += 1
                    continue
                try:
                    outcome, _ = self.store.process_message_atomically(
                        message, False, lambda: [], operation="message_process_window"
                    )
                except sqlite3.Error as exc:
                    failed += 1
                    try:
                        failure = self.store.record_message_failure(message, exc, now=now)
                        self.on_debug(
                            f"WHATSAPP_INTELLIGENCE: failed=1 retry_scheduled={not failure['quarantined']} "
                            f"error={failure['category']} attempts={failure['count']}"
                        )
                    except sqlite3.Error as failure_exc:
                        self.store._db_debug("message_failure", failure_exc)
                        self.on_debug(
                            f"WHATSAPP_INTELLIGENCE: failed=1 retry_scheduled=false "
                            f"error={type(exc).__name__} failure_recording=FAILED"
                        )
                    continue
                if outcome == "ALREADY_PROCESSED":
                    already += 1
                else:
                    out_of_window += 1
                continue
            if self.store.is_message_processed(message.message_id):
                already += 1
                continue
            retry_allowed, _failure_count, _retry_after = self.store._failure_retry_state(message.message_id, now=now)
            if not retry_allowed:
                retry_deferred += 1
                continue
            new_messages.append(message)

        events: list[IntelligenceEvent] = []

        for message in new_messages:
            direction_label = str(message.metadata.get("direction") or "unknown").upper()
            sender_evidence = str(message.metadata.get("sender_evidence", "NONE"))
            diagnostic_key = (message.message_id, f"{direction_label}:{sender_evidence}")
            if diagnostic_key not in self._direction_diagnostics_emitted:
                if direction_label in {"INCOMING", "OUTGOING"}:
                    self.on_debug(
                        f"WHATSAPP_MESSAGE: chat={message.conversation_id} source_id={message.message_id} "
                        f"direction={direction_label} direction_evidence={sender_evidence}"
                    )
                    self._direction_diagnostics_emitted.add(diagnostic_key)
                elif sender_evidence == "NONE":
                    self.on_debug(
                        f"WHATSAPP_DEBUG: direction_evidence_missing chat={message.conversation_id} "
                        f"source_id={message.message_id}"
                    )
                    self._direction_diagnostics_emitted.add(diagnostic_key)

            def process_one() -> list[IntelligenceEvent]:
                # Low-value messages are still semantically incorporated; their
                # terminal result is simply an empty semantic change set.
                produced = self._analyze_message(message, messages)
                self._update_summary(message, produced)
                return produced

            try:
                outcome, produced = self.store.process_message_atomically(
                    message,
                    True,
                    process_one,
                    operation="message_process",
                )
            except sqlite3.Error as exc:
                self.store._db_debug("message_process", exc)
                failed += 1
                sync_error_reason = "database_error"
                try:
                    failure = self.store.record_message_failure(message, exc, now=now)
                except sqlite3.Error as failure_exc:
                    # The original message remains unprocessed. Keep the worker
                    # alive and surface the failure without pretending recovery
                    # state was persisted when its own write failed.
                    self.store._db_debug("message_failure", failure_exc)
                    self.on_debug(
                        f"WHATSAPP_INTELLIGENCE: failed=1 retry_scheduled=false "
                        f"error={type(exc).__name__} failure_recording=FAILED"
                    )
                    continue
                self.on_debug(
                    f"WHATSAPP_INTELLIGENCE: failed=1 retry_scheduled={not failure['quarantined']} "
                    f"error={failure['category']} attempts={failure['count']}"
                )
                continue
            except Exception as exc:
                failed += 1
                sync_error_reason = "semantic_processing_failed"
                try:
                    failure = self.store.record_message_failure(message, exc, now=now)
                except sqlite3.Error as failure_exc:
                    self.store._db_debug("message_failure", failure_exc)
                    self.on_debug(
                        f"WHATSAPP_INTELLIGENCE: failed=1 retry_scheduled=false "
                        f"error={type(exc).__name__} failure_recording=FAILED"
                    )
                    continue
                self.on_debug(
                    f"WHATSAPP_INTELLIGENCE: failed=1 retry_scheduled={not failure['quarantined']} "
                    f"error={failure['category']} attempts={failure['count']}"
                )
                continue

            if outcome == "ALREADY_PROCESSED":
                # A concurrent worker won the transactional claim after our
                # read-only lookup. It still never enters relevance/extraction.
                already += 1
                continue
            if outcome == "IGNORED":
                continue

            incorporated += 1
            events.extend(produced)
            for event in produced:
                self.on_debug(
                    f"WHATSAPP_EVENT: PERSISTED event={event.event_id} status={event.status} "
                    f"source_count={event.source_count}"
                )
                try:
                    self.on_intelligence(event)
                except Exception as exc:
                    # Persistence is already committed. Presentation failure must
                    # not roll the message back into the retry path.
                    self.on_debug(
                        f"WHATSAPP_INTELLIGENCE: presentation_failed "
                        f"error={type(exc).__name__}:{str(exc).strip()[:160]}"
                    )

        eligible_messages = [
            message for message in messages
            if state.enabled_at is not None
            and message.timestamp >= max(state.enabled_at, now - WHATSAPP_EVENT_LOOKBACK_DAYS * 86400)
        ]
        # A completely quiet LIVE poll must remain a true no-op. Boundary
        # reconciliation is only necessary when this cycle discovered source
        # work, deferred a retry, or encountered a failure. Otherwise there is
        # no new candidate boundary to report as advanced.
        checkpoint_advanced = False
        checkpoint_boundary = None
        if new_messages or incorporated or failed or retry_deferred:
            try:
                checkpoint_advanced, checkpoint_boundary = self.store.advance_checkpoint(eligible_messages)
            except sqlite3.Error as exc:
                failed += 1
                sync_error_reason = "checkpoint_persistence_failed"
                checkpoint_boundary = None
                self.store._db_debug("checkpoint_advance", exc)
        meaningful_sync = (
            mode != "LIVE" and (
                incorporated > 0 or failed > 0 or retry_deferred > 0 or out_of_window > 0
            )
            or incorporated > 0
            or failed > 0
            or retry_deferred > 0
            or checkpoint_advanced
        )
        if meaningful_sync:
            status = "SYNC_COMPLETE" if failed == 0 and retry_deferred == 0 else "SYNC_PARTIAL"
            self.on_debug(
                "WHATSAPP_SYNC:\n"
                f"chat={conversation_hint or 'whatsapp'}\n"
                f"source_delta={len(new_messages)}\n"
                f"incorporated={incorporated}\n"
                f"semantic_changes={len(events)}\n"
                f"already_synchronized={already}\n"
                f"out_of_window={out_of_window}\n"
                f"retry_deferred={retry_deferred}\n"
                f"failed={failed}\n"
                f"checkpoint_advanced={str(checkpoint_advanced).lower()}\n"
                f"status={status}"
            )
            if failed > 0:
                self.on_debug(
                    f"WHATSAPP_SYNC_ERROR: chat={conversation_hint or 'whatsapp'} "
                    f"reason={sync_error_reason or 'synchronization_failed'}"
                )
        return events

    def _evidence(self, message: WhatsAppMessage) -> tuple[dict[str, Any], ...]:
        return ({
            "message_id": message.message_id,
            "timestamp": message.timestamp,
            "conversation_id": message.conversation_id,
            "sender": message.sender,
            "direction": message.metadata.get("direction", "unknown"),
            "is_outgoing": message.metadata.get("is_outgoing"),
        },)

    def _event(self, message: WhatsAppMessage, kind: IntelligenceEventType, title: str, description: str, status: str, confidence: float, *, event_time: float | None = None, deadline: float | None = None, date: str | None = None, time_of_day: str | None = None, timezone_name: str | None = None, platform: str | None = None, location: str | None = None, meeting_link: str | None = None, scheduler_candidate: bool = False, event_id: str | None = None, existing: IntelligenceEvent | None = None) -> IntelligenceEvent:
        now = time.time()
        return IntelligenceEvent(
            event_id=event_id or ("wae-" + uuid.uuid4().hex[:20]),
            conversation_id=message.conversation_id, type=kind, title=title or (existing.title if existing else kind.value.title()),
            description=description, status=status, confidence=max(confidence, existing.confidence if existing else 0.0),
            importance=max(0.8 if kind is not IntelligenceEventType.SUMMARY else 0.55, existing.importance if existing else 0.0),
            urgency=max(0.8 if kind in {IntelligenceEventType.DEADLINE, IntelligenceEventType.ACTION_ITEM} else 0.4, existing.urgency if existing else 0.0),
            created_at=existing.created_at if existing else now, updated_at=now, event_time=event_time if event_time is not None else (existing.event_time if existing else None), deadline=deadline if deadline is not None else (existing.deadline if existing else None),
            date=date if date is not None else (existing.date if existing else None),
            time=time_of_day if time_of_day is not None else (existing.time if existing else None),
            timezone=timezone_name if timezone_name is not None else (existing.timezone if existing else None),
            platform=platform if platform is not None else (existing.platform if existing else None),
            location=location if location is not None else (existing.location if existing else None),
            meeting_link=meeting_link if meeting_link is not None else (existing.meeting_link if existing else None),
            participants=tuple(dict.fromkeys((*(existing.participants if existing else ()), *([message.sender] if message.sender else [])))),
            evidence=tuple(_merge_evidence(existing.evidence if existing else (), self._evidence(message))),
            source_message_ids=tuple(dict.fromkeys((*(existing.source_message_ids if existing else ()), message.message_id))),
            scheduler_candidate=scheduler_candidate and status in {"CONFIRMED", "LIKELY"},
        )

    def _event_proposal(self, message: WhatsAppMessage) -> EventProposal:
        text = message.text
        lower = text.casefold()
        meeting_context = bool(re.search(r"\b(?:meeting|meet|call)\b", lower))
        date_match = re.search(r"\b(today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next week)\b", lower)
        time_match = _event_time_match(text, meeting_context=meeting_context)
        cancellation = bool(re.search(r"\b(?:cancelled|canceled|cancel|cancel hai|cancelled hai|cancel hai)\b", lower))
        confirmation = bool(re.search(r"\b(?:confirmed|confirm|okay|ok|haan|yes|sure|done)\b", lower))
        platform_match = re.search(r"\b(discord|zoom|google meet|teams|microsoft teams)\b", lower)
        link_match = re.search(r"https?://[^\s]+", text)
        correction = bool(re.search(r"\b(?:actually|move it|changed it|kar di|kar diya|shifted|rescheduled)\b", lower))
        reference = bool(re.search(r"\b(?:that meeting|the meeting|same meeting|as discussed|same link)\b", lower))
        eventish = meeting_context or platform_match or link_match or cancellation or confirmation or correction or reference or bool(date_match) or bool(time_match)
        if not eventish:
            return EventProposal("NO_EVENT", None, evidence_message_id=message.message_id)

        event_tz = _event_timezone_for_message(message, text)
        base = datetime.fromtimestamp(message.timestamp, tz=event_tz)
        resolved_date: str | None = None
        resolved_time: str | None = None
        resolved_dt: float | None = None
        if date_match:
            token = date_match.group(1).casefold()
            if token == "today" or token == "tonight":
                target = base.date()
            elif token == "tomorrow":
                target = (base + timedelta(days=1)).date()
            elif token == "next week":
                target = (base + timedelta(days=(7 - base.weekday()))).date()
            else:
                weekdays = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
                idx = weekdays.index(token)
                delta = (idx - base.weekday()) % 7
                if delta == 0:
                    delta = 7
                target = (base + timedelta(days=delta)).date()
            resolved_date = target.isoformat()
        if time_match:
            parsed_clock = _parse_clock_match(time_match)
            if parsed_clock is not None:
                hour, minute = parsed_clock
                resolved_time = f"{hour:02d}:{minute:02d}"
                parsed_meridiem = (time_match.group("ampm") or "NONE").replace(".", "").upper()
                self.on_debug(
                    "WHATSAPP_TIME_DEBUG: "
                    f"raw_time_expression={_normalize_text(time_match.group(0))!r} "
                    f"parsed_hour={hour} parsed_minute={minute} "
                    f"parsed_meridiem={parsed_meridiem} " 
                    f"source_message_timestamp={datetime.fromtimestamp(message.timestamp, tz=timezone.utc).isoformat()} "
                    f"source_message_timezone={_source_timezone_label(message)} "
                    f"event_timezone={_timezone_label(event_tz)} "
                    f"normalized_event_time={resolved_time}"
                )
        if resolved_date and resolved_time:
            event_dt = datetime.combine(
                datetime.fromisoformat(resolved_date).date(),
                datetime.strptime(resolved_time, "%H:%M").time(),
                tzinfo=event_tz,
            )
            resolved_dt = event_dt.timestamp()

        if platform_match:
            platform = platform_match.group(1).title()
            if platform.casefold() == "Google Meet".casefold(): platform = "Google Meet"
        else:
            platform = None
        explicit_reference = bool(re.search(r"\b(?:that meeting|same meeting|as discussed|same link)\b", lower))
        action = "CANCEL_EVENT" if cancellation else ("CONFIRM_EVENT" if confirmation and not (meeting_context or date_match or time_match or platform_match or link_match or correction) else ("UPDATE_EVENT" if (explicit_reference or correction or platform_match or link_match or (date_match and not meeting_context)) else "CREATE_EVENT"))
        if meeting_context and not explicit_reference and not correction and action not in {"CANCEL_EVENT", "CONFIRM_EVENT"}:
            action = "CREATE_EVENT"
        if correction or platform_match or link_match or (date_match and not meeting_context):
            action = "UPDATE_EVENT"
        if cancellation:
            action = "CANCEL_EVENT"
        elif confirmation and not (meeting_context or date_match or time_match or platform_match or link_match or correction):
            action = "CONFIRM_EVENT"

        title = "Meeting"
        return EventProposal(
            action=action, event_type=IntelligenceEventType.MEETING, title=title,
            date_expression=date_match.group(1) if date_match else None, resolved_date=resolved_date,
            time_expression=time_match.group(0) if time_match else None, resolved_time=resolved_time,
            timezone=_timezone_label(event_tz), platform=platform, meeting_link=link_match.group(0) if link_match else None,
            confidence=0.90 if cancellation or confirmation else 0.86,
            referenced_event_hint=("explicit" if explicit_reference else ("implicit" if correction or (platform_match is not None and not meeting_context) or (date_match and not meeting_context) else None)),
            cancellation=cancellation, confirmation=confirmation, correction=correction, evidence_message_id=message.message_id, extracted_claim=text,
        )

    @staticmethod
    def _match_score(message: WhatsAppMessage, proposal: EventProposal, event: IntelligenceEvent) -> float:
        if event.conversation_id != message.conversation_id or event.type is not proposal.event_type:
            return 0.0
        score = 0.45
        age = max(0.0, message.timestamp - event.updated_at)
        if age <= 7 * 24 * 3600: score += 0.15
        if proposal.referenced_event_hint == "explicit": score += 0.35
        elif proposal.referenced_event_hint == "implicit": score += 0.20
        if proposal.resolved_time and event.time == proposal.resolved_time: score += 0.15
        if proposal.resolved_date and event.date == proposal.resolved_date: score += 0.15
        if proposal.platform and event.platform and proposal.platform.casefold() == event.platform.casefold(): score += 0.10
        return min(score, 1.0)

    def _apply_proposal(self, message: WhatsAppMessage, proposal: EventProposal) -> IntelligenceEvent | None:
        if proposal.action == "NO_EVENT" or proposal.event_type is None:
            return None
        candidates = self.store.find_event_candidates(message.conversation_id, proposal.event_type)
        scored = sorted(((self._match_score(message, proposal, e), e) for e in candidates), key=lambda x: x[0], reverse=True)
        existing: IntelligenceEvent | None = None
        if proposal.action in {"UPDATE_EVENT", "CONFIRM_EVENT", "CANCEL_EVENT"}:
            if proposal.referenced_event_hint == "implicit" and len(scored) > 1 and not (proposal.resolved_date or proposal.platform or proposal.meeting_link):
                self.on_debug(f"WHATSAPP_EVENT: AMBIGUOUS candidates={len(scored)} action=NEEDS_CLARIFICATION")
                return None
            if scored:
                if len(scored) == 1 and scored[0][0] >= 0.45:
                    existing = scored[0][1]
                elif scored[0][0] >= 0.65 and scored[0][0] > scored[1][0] + 0.10:
                    existing = scored[0][1]
            if existing is None:
                if scored:
                    self.on_debug(f"WHATSAPP_EVENT: AMBIGUOUS candidates={len(scored)} action=NEEDS_CLARIFICATION")
                return None
        elif proposal.action == "CREATE_EVENT" and candidates:
            # A genuine creation should still be idempotent if its evidence was
            # re-evaluated after a crash/restart. Same conversation/type with
            # compatible recent evidence is enough to reuse the event.
            compatible = [item for item in scored if item[0] >= 0.70]
            if len(compatible) == 1:
                existing = compatible[0][1]

        status = "CANCELLED" if proposal.cancellation else ("CONFIRMED" if proposal.confirmation and proposal.action == "CONFIRM_EVENT" else ("MODIFIED" if existing and proposal.action == "UPDATE_EVENT" else "PROPOSED"))
        if existing and existing.status == "CANCELLED" and proposal.action != "CANCEL_EVENT":
            # Old cancellation evidence is stronger than a repeated historical
            # proposal. A genuinely new explicit confirmation can reopen only
            # through a later, valid update.
            if not proposal.confirmation:
                return existing
        effective_time = proposal.resolved_time
        if existing is not None and effective_time and proposal.time_expression and not re.search(r"(?:am|pm|a\.?m\.?|p\.?m\.?)", proposal.time_expression, re.I) and existing.time:
            if not proposal.correction:
                # Follow-up "at 6" normally refers to the already-established
                # 18:00 meeting rather than silently changing it to 06:00.
                effective_time = existing.time
            elif int(effective_time[:2]) < 12 and int(existing.time[:2]) >= 12:
                # A correction such as "Actually 7 baje kar di" inherits the
                # established meeting's evening half-day when no AM/PM marker
                # is supplied.
                effective_time = f"{int(effective_time[:2]) + 12:02d}:{effective_time[3:]}"
        effective_date = proposal.resolved_date or (existing.date if existing else None)
        event_time = None
        if effective_date and effective_time:
            event_tz = _timezone_from_value(proposal.timezone) or _event_timezone_for_message(message, message.text)
            event_time = datetime.combine(
                datetime.fromisoformat(effective_date).date(),
                datetime.strptime(effective_time, "%H:%M").time(),
                tzinfo=event_tz,
            ).timestamp()
        event = self._event(
            message, IntelligenceEventType.MEETING, existing.title if existing else proposal.title, message.text, status, proposal.confidence,
            event_time=event_time,
            date=effective_date, time_of_day=effective_time,
            timezone_name=proposal.timezone or (existing.timezone if existing else None), platform=proposal.platform or (existing.platform if existing else None),
            meeting_link=proposal.meeting_link or (existing.meeting_link if existing else None),
            event_id=existing.event_id if existing else None, existing=existing, scheduler_candidate=True,
        )
        self.store.upsert_event(event)
        changed = []
        if existing is None:
            changed = ["MEETING_CREATED"]
        else:
            if event.date != existing.date: changed.append("MEETING_DATE_ADDED" if existing.date is None else "MEETING_DATE_CHANGED")
            if event.time != existing.time: changed.append("MEETING_TIME_ADDED" if existing.time is None else "MEETING_TIME_CHANGED")
            if event.platform != existing.platform: changed.append("MEETING_PLATFORM_ADDED" if existing.platform is None else "MEETING_PLATFORM_CHANGED")
            if event.meeting_link != existing.meeting_link: changed.append("MEETING_LINK_ADDED" if existing.meeting_link is None else "MEETING_LINK_CHANGED")
            if proposal.confirmation and event.status == "CONFIRMED" and event.status != existing.status: changed.append("MEETING_CONFIRMED")
            if proposal.cancellation and event.status != existing.status: changed.append("MEETING_CANCELLED")
            if not changed: changed.append("MEETING_UPDATED")
        for evidence_type in changed:
            added = self.store.add_event_evidence(event.event_id, message, evidence_type, proposal.extracted_claim, proposal.confidence)
            # Evidence is part of the enclosing message transaction; the
            # caller reports PERSISTED only after commit.
        return self.store.get_event(event.event_id) or event

    def _analyze_message(self, message: WhatsAppMessage, context: list[WhatsAppMessage]) -> list[IntelligenceEvent]:
        proposal = self._event_proposal(message)
        self.on_debug(f"WHATSAPP_EVENT: EXTRACTED type={proposal.event_type.value if proposal.event_type else 'NONE'} action={proposal.action} confidence={proposal.confidence:.2f}") if proposal.event_type else None
        event = self._apply_proposal(message, proposal)
        out = [event] if event is not None else []
        lower = message.text.casefold()
        # Preserve existing non-meeting intelligence behavior. Meeting event
        # mutations are handled structurally above.
        if re.search(r"\bassignment\b|\bassigned\b", lower):
            out.append(self._event(message, IntelligenceEventType.ASSIGNMENT, "Assignment", message.text, "CONFIRMED", 0.88))
        if re.search(r"\b(todo|to-do|task|action item|need to|must)\b", lower):
            out.append(self._event(message, IntelligenceEventType.ACTION_ITEM if "action item" in lower else IntelligenceEventType.TASK, "Action item" if "action item" in lower else "Task", message.text, "CONFIRMED", 0.84))
        return out

    def _update_summary(self, message: WhatsAppMessage, events: list[IntelligenceEvent]) -> None:
        if not events:
            return
        labels = []
        for event in events:
            if event.type is IntelligenceEventType.MEETING:
                labels.append(f"meeting ({event.status.lower()})")
            else:
                labels.append(event.title.lower())
        summary = "Meaningful activity: " + ", ".join(dict.fromkeys(labels)) + "."
        old = self.store.summary(message.conversation_id)
        if old and old.get("current_summary"):
            summary = old["current_summary"].rstrip(".") + "; " + ", ".join(dict.fromkeys(labels)) + "."
        self.store.update_summary(message.conversation_id, summary, message.message_id)
        direction = str(message.metadata.get("direction") or "unknown").upper()
        self.on_debug(f"WHATSAPP_INTELLIGENCE: relevance={relevance(message.text).value} direction={direction} event={events[0].type.value} status={events[0].status} confidence={events[0].confidence:.2f} evidence_count={len(events[0].source_message_ids)}")
