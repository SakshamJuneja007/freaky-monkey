"""One conversation, two input methods, one execution path.

This module is the "Normalize User Task" box: the join point where typing and
speaking become the same thing. Its whole reason to exist is a negative one --
there must be no second agent. :meth:`Session.submit` is the only call site of
:func:`agent_control.api.run_agent_task` outside the ``run`` subcommand and the
benchmark harness, and *both* interfaces reach it:

    typed text ----+
                   +--> normalize() --> Session.submit() --> run_agent_task()
    audio -> STT --+

Note what is deliberately *not* here. No planning: the planner is chosen inside
``run_agent_task``. No policy: policy is constructed there too, from the task id.
No verification: the verdicts in :class:`~agent_control.api.AgentResult` were
produced by the task's own verifiers. No memory: recall happens inside the
pipeline, keyed on the request this module hands it. A ``Session`` decides
exactly two things -- whether a string is a runnable request, and what to say
about the result -- and it answers the second by asking ``response.py``, which
answers from verified fields only.

Speech is one layer thinner than it looks: :meth:`Session.listen` returns *text*
and never submits anything. The chat loop submits it, by the same method a typed
line goes through, which is what makes "continue typing after a voice command"
true by construction rather than by a code path that happens to exist.
"""

from __future__ import annotations
from .conversation import ConversationEngine, ConversationTransportError
from .conversation_memory import ConversationMemory
from .persistent_memory import MemoryExtractor, PersistentMemory
from .planner.openai_compat import LLMUnavailable

import re
import os
import json
import sqlite3
import time
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from . import api
from .api import AgentResult, TaskStatus
from .types import Action, Clarification, Verdict
from .fast_interaction import classify_fast, FastInteractionTask, FastMessagingTask
from .task import Task
from .workflow import Workflow, WorkflowStepTask
from .workflow.graph import WorkflowOrchestrator
from .workflow.session_adapter import SessionWorkflowRuntime
from .skills.browser import BrowserSkillAdapter
from .runtime import RuntimeManager
from .runtime_control import RuntimeControlKind, RuntimeControlCommand, classify_runtime_control, classify_runtime_controls
from .presentation import (
    completion_message,
    is_explicit_task_id_request,
    recovery_message,
    sanitize_tts_text,
    approval_prompt,
)
from .response import (
    DeimosPresentation,
    Narrator,
    phrase_choice,
    phrase_dropped,
    phrase_empty,
    phrase_no_location,
    phrase_not_heard,
    phrase_result,
)
from .response import guard_conversation_runtime_claim

#: Trailing characters a person types or an STT model appends that are never part
#: of a task id. Kept to punctuation: stripping words would be intent inference,
#: which this layer does not do.
_TRAILING = " \t\r\n.!?,;:'\"`"

_WHITESPACE = re.compile(r"\s+")


class IntentCategory(str, Enum):
    """The one routing decision made before conversation or execution."""

    CONVERSATION = "CONVERSATION"
    ACTION = "ACTION"
    FOLLOW_UP_ACTION = "FOLLOW_UP_ACTION"
    CORRECTION = "CORRECTION"
    CLARIFICATION_RESPONSE = "CLARIFICATION_RESPONSE"
    CANCEL = "CANCEL"
    LOCAL_COMMAND = "LOCAL_COMMAND"


@dataclass(frozen=True)
class InputEvent:
    """One user utterance entering the session's authoritative dispatcher."""

    text: str
    source: str = "text"
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: f"input-{uuid.uuid4().hex[:10]}")


class InputOwner(str, Enum):
    APPROVAL = "APPROVAL"
    HUMAN = "HUMAN"
    CONVERSATION = "CONVERSATION"
    TASK = "TASK"
    NONE = "NONE"

#: Status lines are printed for these trace events and no others. An allowlist,
#: not a filter, so a new event kind is silent until somebody decides what it
#: should say -- the failure mode of a denylist here is inventing narration for
#: an event nobody designed a sentence for.
STATUS_EVENTS = (
    "agent_state",
    "policy_decision",
)

DEBUG_STATUS_EVENTS = (
    "agent_state",
    "planner_call",
    "policy_decision",
    "action",
    "verification",
    "recovery",
    "recovery_resolved",
    "failure",
    "injection",
)


@dataclass(frozen=True)
class UserTask:
    """A request after normalization, with the verdict on whether it can run.

    ``raw`` is kept beside ``text`` because they answer different questions: what
    the user actually said, and what the agent acted on. When a transcript is
    wrong, that difference is the evidence.
    """

    raw: str
    text: str
    source: str = "text"
    task_id: str = ""
    #: Arguments for a parameterized task -- currently only ``{"path": ...}`` for
    #: ``open_named_file``. Empty for every fixed task, which is why nothing in the
    #: existing flow has to know about it.
    params: dict[str, Any] = field(default_factory=dict)
    #: "accepted" | "empty" | "unregistered" | "conversation" | "dropped".
    #: Distinct causes stay distinct; an empty line is not a failed request, and
    #: "dropped" -- a line that answered a pending question with nothing
    #: selectable -- is neither.
    #:
    #: "unregistered" is the honest name for what this field can actually know:
    #: that no registered workflow answers to this sentence. It is a statement
    #: about the task registry, not about the assistant's reach -- the request
    #: may still be a general computer action (Route 2) or conversation (Route
    #: 3), and both are decided after normalization, not here. Whether something
    #: is genuinely beyond the assistant is a property of a *run*, reported as
    #: ``TaskStatus.UNSUPPORTED`` on an :class:`~agent_control.api.AgentResult`;
    #: it is deliberately not spellable as a ``UserTask`` status, because at this
    #: point nothing has looked.
    #:
    #: Deliberately not "cancelled": that word is reserved for a *run* that was
    #: interrupted, which is a different event with a different status.
    status: str = "accepted"
    intent: IntentCategory | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "accepted" and bool(self.task_id)

    def to_json(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "text": self.text,
            "source": self.source,
            "task_id": self.task_id,
            "params": dict(self.params),
            "status": self.status,
            "intent": self.intent.value if self.intent is not None else None,
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class Capture:
    """The result of listening once: text, or an honest account of why not.

    Shaped like ``Recording`` and ``Transcript`` on purpose -- ``ok`` plus a
    ``reason`` slug -- because it is those two collapsed into one value, and
    flattening a slug into a bare empty string is how a caller ends up submitting
    silence as a task.
    """

    text: str = ""
    ok: bool = False
    reason: str = ""
    detail: str = ""
    seconds: float = 0.0
    latency_seconds: float = 0.0
    confidence: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "ok": self.ok,
            "reason": self.reason,
            "detail": self.detail,
            "seconds": round(self.seconds, 3),
            "latency_seconds": round(self.latency_seconds, 3),
            "confidence": self.confidence,
        }


@dataclass
class Turn:
    """One exchange: what was asked, what happened, what was said back.

    ``result`` is ``None`` exactly when nothing was executed -- an empty line, an
    unregistered request, or a failed capture. That is the difference between "the
    agent did not run" and "the agent ran and did not succeed", and the two must
    never read alike.
    """

    task: UserTask
    reply: str
    result: AgentResult | None = None
    status_lines: list[str] = field(default_factory=list)

    @property
    def executed(self) -> bool:
        return self.result is not None

    @property
    def ok(self) -> bool:
        """Verified success, and nothing weaker. ``None`` is not success."""
        return self.result is not None and self.result.ok

    def to_json(self) -> dict[str, Any]:
        return {
            "task": self.task.to_json(),
            "reply": self.reply,
            "executed": self.executed,
            "ok": self.ok,
            "status_lines": list(self.status_lines),
            "result": self.result.to_json() if self.result else None,
        }


@dataclass
class RecentContext:
    """Small, typed context derived only from verified successful effects.

    These paths help the next planner resolve phrases such as ``that folder``
    and ``there``.  They are context, not authority: every resulting action is
    still resolved and checked by the next run's :class:`Policy`.
    """

    last_verified_directory: Path | None = None
    last_verified_file: Path | None = None
    last_verified_app: str | None = None
    last_goal: str | None = None
    recent_action_summary: str | None = None

    def planner_state(self) -> dict[str, str]:
        state: dict[str, str] = {}
        if self.last_verified_directory is not None:
            state["last_verified_directory"] = str(self.last_verified_directory)
        if self.last_verified_file is not None:
            state["last_verified_file"] = str(self.last_verified_file)
        if self.last_verified_app:
            state["last_verified_app"] = self.last_verified_app
        if self.last_goal:
            state["last_goal"] = self.last_goal
        if self.recent_action_summary:
            state["recent_action_summary"] = self.recent_action_summary
        return state


def normalize(raw: str, *, source: str = "text") -> UserTask:
    """Turn one line of input -- typed or transcribed -- into a runnable request.

    Three outcomes, and the third is the one that keeps the project honest:

    * nothing but whitespace -> ``empty``, and no execution is attempted;
    * resolves to a registered task -> ``accepted``;
    * anything else -> ``unregistered``.

    The last branch says only what was checked: this sentence is not the name of
    a registered workflow. It is not a verdict that the assistant cannot help,
    and it is not a stub. :meth:`Session.submit` takes it further -- a structured
    request shape, a general computer action, or conversation -- and refuses by
    name only once every one of those has been tried. What normalization must
    never do is coerce a request into the *nearest* registered task: guessing
    would make the interface look general while making the agent act on
    something the user did not ask for.

    Normalization stops at whitespace and trailing punctuation. Everything else
    -- case, ``-``/``_``/space equivalence -- is already handled by
    :func:`api.resolve_task`, and duplicating it here would give two places to
    disagree about what "open last day pdf" means.
    """
    text = _WHITESPACE.sub(" ", (raw or "").strip()).strip(_TRAILING)

    if not text:
        return UserTask(raw=raw or "", text="", source=source, status="empty")

    task_id = api.resolve_task(text)

    if task_id is None:
        return UserTask(raw=raw or "", text=text, source=source,
                        status="unregistered")

    return UserTask(raw=raw or "", text=text, source=source, task_id=task_id)


# ----------------------------------------------------------------------
# One unanswered question, and the narrow rules for answering it
# ----------------------------------------------------------------------

#: Returned by :func:`choose` for "never mind". Distinct from ``None``, which
#: means the words named none of the options: one drops the question, the other
#: puts it again.
CANCELLED = "\x00cancelled"

#: How many times one question may be put, including the first. A person who has
#: answered unintelligibly twice is not helped by a third identical question.
MAX_ASKS = 2

#: Spoken numbers. Required, not a nicety: STT returns "two", never "2", so
#: without these the voice path -- the point of the feature -- cannot answer.
#: Ordinals are here because "the second one" is how people actually answer.
_NUMBER_WORDS = {
    "first": 1, "1st": 1,
    "two": 2, "second": 2, "2nd": 2,
    "three": 3, "third": 3, "3rd": 3,
    "four": 4, "fourth": 4, "4th": 4,
    "five": 5, "fifth": 5, "5th": 5,
}

#: Numbers that are also ordinary English. "one" is a pronoun ("that one"), and
#: "to"/"too"/"for" are what STT writes when it does not hear a numeral. They are
#: read as an index only by the last branch of :func:`choose`, once nothing else
#: in the line has named anything.
_WEAK_NUMBERS = {"one": 1, "won": 1, "to": 2, "too": 2, "for": 4}

#: Words that point without naming. Their presence disables the weak-number
#: reading, which is what keeps "that one" from selecting option 1.
_VAGUE = frozenset({
    "that", "this", "these", "those", "it", "the", "which", "them", "same",
    "either", "whichever", "any",
})

#: Words that withdraw the question. "no" counts: in answer position it means
#: open nothing, and cancelling is the safe reading of an ambiguous refusal.
_CANCEL_WORDS = frozenset({
    "cancel", "cancelled", "canceled", "stop", "quit", "exit", "abort",
    "nevermind", "never", "mind", "forget", "nothing", "none", "no", "neither",
})

_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class PendingApproval:
    request: str
    action: dict[str, Any]
    asked_at: float
    task_id: str = ""
    goal: str = ""
    background_task_id: str | None = None
    runtime_task_id: str | None = None
    workflow_id: str | None = None
    step_id: str | None = None
    approval_request_id: str | None = None
    ttl_s: float = 180.0

    @property
    def expired(self) -> bool:
        # A LangGraph workflow approval is a durable human-in-the-loop
        # interruption. Its lifetime is the workflow checkpoint lifetime, not
        # the legacy single-task confirmation TTL. Never turn an unanswered
        # workflow approval into a failure merely because a synchronous input
        # turn took longer than the legacy 180s window.
        if self.workflow_id:
            return False
        return (time.time() - self.asked_at) > self.ttl_s


@dataclass(frozen=True)
class Pending:
    """One unanswered question, bound to the exact options it offered.

    This is the whole of the conversational state this project allows, and the
    binding is what makes it safe. A later answer can only ever select from
    ``choices``, so a stray "yes" or "two" arriving with no question in flight has
    nothing to authorize -- the same rule Directive C states for confirmations,
    applied to a choice between files.
    """

    query: str
    task_id: str
    choices: tuple[api.Choice, ...]
    asked_at: float
    ttl_s: float = 180.0
    #: How many times this question has been put. Carried across a re-ask so the
    #: count, like ``asked_at``, is not reset by re-asking.
    asks: int = 1
    runtime_task_id: str | None = None

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.asked_at)

    @property
    def expired(self) -> bool:
        return self.age_s > self.ttl_s

    @property
    def exhausted(self) -> bool:
        return self.asks >= MAX_ASKS

    def again(self) -> "Pending":
        """The same question, asked once more. ``asked_at`` deliberately unchanged."""
        return replace(self, asks=self.asks + 1)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(choice.label for choice in self.choices)

    def to_json(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "task_id": self.task_id,
            "choices": [choice.to_json() for choice in self.choices],
            "age_s": round(self.age_s, 1),
            "ttl_s": self.ttl_s,
            "asks": self.asks,
            "runtime_task_id": self.runtime_task_id,
        }


def _words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def parse_approval_response(raw: str) -> str:
    """Parse only explicit approval/rejection language; ambiguity is safe."""
    words = re.sub(r"[^a-z0-9']+", " ", (raw or "").casefold()).strip()
    yes = {"yes", "yeah", "yep", "yup", "ok", "okay", "sure", "go ahead", "do it", "confirm", "approved", "send it"}
    no = {"no", "nope", "cancel", "stop", "don't", "dont", "do not", "reject", "don't send", "dont send"}
    if words in yes:
        return "APPROVE"
    if words in no:
        return "REJECT"
    return "AMBIGUOUS"


def choose(pending: Pending, raw: str) -> str | None:
    """The path the answer selected, :data:`CANCELLED`, or ``None``.

    ``None`` is the important return. It is what a bare "yes" or "that one" gets,
    because neither names an option: with two candidates there is nothing in those
    words to select on. Returning the first choice for them would be inventing
    consent, so the caller asks again instead.

    Four things are accepted, all checked against ``pending.choices`` only:

    * a digit in range, ``1..N``;
    * a spelled-out number or ordinal, for the voice path -- "two", "the second
      one";
    * a word that appears in exactly one candidate path -- ``downloads``,
      ``assets``. A word shared by several (the filename, the drive) selects
      nothing, which is correct: it does not distinguish them;
    * a bare "one", or an STT homophone of a number, but only when the line points
      at nothing else at all.
    """
    words = _words(raw)

    if not words:
        return None

    if any(word in _CANCEL_WORDS for word in words):
        return CANCELLED

    total = len(pending.choices)

    def offered(index: int) -> str | None:
        """Index ``1..N`` as a path, or ``None`` if they named a number not offered.

        Out of range is a refusal rather than a clamp: they meant a number, just
        not one of these, and clamping would open a file nobody named.
        """
        return pending.choices[index - 1].path if 1 <= index <= total else None

    numbers = {
        int(word) if word.isdigit() else _NUMBER_WORDS[word]
        for word in words
        if word.isdigit() or word in _NUMBER_WORDS
    }

    if numbers:
        return offered(numbers.pop()) if len(numbers) == 1 else None

    vocab = [
        set(_words(choice.path)) | set(_words(choice.label))
        for choice in pending.choices
    ]

    picked = {
        index
        for word in words
        for index, own in enumerate(vocab)
        if word in own and sum(word in other for other in vocab) == 1
    }

    if len(picked) == 1:
        return pending.choices[picked.pop()].path

    # Last resort, and last for a reason: "one" is also a pronoun and "to"/"for"
    # are homophones of numbers that ordinary sentences are full of. Reading them
    # as an index is only safe once nothing else in the line named anything, and
    # never after a word like "that", which is what makes "that one" -- the
    # obvious spoken answer -- select nothing instead of guessing.
    if not any(word in _VAGUE for word in words):
        weak = {_WEAK_NUMBERS[word] for word in words if word in _WEAK_NUMBERS}
        if len(weak) == 1:
            return offered(weak.pop())

    return None


#: Verbs whose presence means the request wants the machine touched, not just
#: discussed. Matched against the whole lowered sentence rather than only
#: ``_words`` so multi-word entries ("set up") work the same way single-word
#: ones do.
_ACTION_VERBS = (
    "open", "launch", "start", "run", "execute",
    "close", "quit", "kill",
    "create", "make", "build", "generate", "set up", "setup",
    "write", "edit", "update", "append", "add", "put", "save",
    "delete", "remove", "move", "copy", "rename",
    "install", "download", "fetch",
    "find", "search", "look", "browse", "list", "show", "scroll",
    "pick", "choose", "select", "play",
    # Messaging / communication actions. These must be routed to the
    # computer-action pipeline so the planner/policy/approval/skill layers
    # can execute side effects instead of the conversation model merely
    # echoing an action-shaped JSON object as text.
    "send", "message",
    # Read-only investigation verbs. Without these, "analyze this project" has
    # no action verb at all, _requires_computer_action returns False, and the
    # request is classified CONVERSATION -- routed to the LLM chit-chat engine
    # instead of GeneralTask, which is a different and worse failure than
    # picking the wrong target: it never reaches policy, readable_roots, or
    # the filesystem, so it cannot even ask a grounded clarifying question.
    "analyze", "analyse", "inspect", "review", "investigate", "examine",
)

#: Phrasing that marks a request as a question about what to do, not an order
#: to do it. Checked first and wins over an action verb found elsewhere in the
#: sentence: "what project should I open" is still a question, even though it
#: contains "open".
_DELIBERATIVE_MARKERS = (
    "what should i", "what do you think", "what would you",
    "should i", "do you think", "any suggestions", "any ideas",
    "what project", "what's the best", "what is the best",
    "how do i feel", "i have no idea", "not sure what",
    "what did we", "what did you", "what have we", "what are we doing",
)


def _requires_computer_action(text: str) -> bool:
    """Route 2 vs Route 3: does this sentence ask for something to be *done*,
    or is it asking to talk about something?

    Conservative and lexical, in the same spirit as :func:`api.parse_request`:
    a wrong guess in the "call it conversation" direction costs the user one
    follow-up line. A wrong guess the other way hands a real request to the
    planner -- still gated by policy, still required to earn a PASS from
    verification -- but is the worse failure mode for a genuine question,
    since a question answered by launching an application is a surprise, not
    a shortcut. Ties therefore go to conversation.

    This is a heuristic, not a natural-language front end: it will misjudge
    some phrasing, and the asymmetry above is what keeps that failure mode
    cheap rather than dangerous.
    """
    lowered = (text or "").lower()

    if any(marker in lowered for marker in _DELIBERATIVE_MARKERS):
        return False

    words = set(_words(lowered))
    return any(
        (verb in words if " " not in verb else verb in lowered)
        for verb in _ACTION_VERBS
    )


def _corrects_previous_action(text: str) -> bool:
    """Whether ``text`` is phrased as a correction to an action just run.

    This is deliberately narrower than general intent inference.  These forms
    have no standalone object (``it`` / ``instead``), so they are only useful
    when :class:`Session` can also prove that the immediately preceding turn
    executed.  That keeps ordinary conversational uses of "call it" out of the
    action route while ensuring an execution correction never reaches the
    conversational model.
    """
    lowered = _WHITESPACE.sub(" ", (text or "").strip().lower())
    return (
        "instead" in lowered
        and (
            lowered.startswith("no ")
            or lowered.startswith("no,")
            or lowered.startswith("actually ")
            or "call it " in lowered
            or "rename it " in lowered
        )
    )


_LOCAL_COMMANDS = frozenset({"cls", "clear", "/help", "/tasks", "/clear-history", "/quit"})
_CANCEL_WORDS = frozenset({"cancel", "drop", "forget it", "never mind", "nevermind"})
_CONTEXT_REFERENCES = (
    "that folder", "that directory", "that file", "there", "put it there",
    "the one we just created", "the one you just created", "the one we created",
)


def _references_recent_context(text: str, context: RecentContext) -> bool:
    lowered = _WHITESPACE.sub(" ", (text or "").strip().lower())
    if any(marker in lowered for marker in _CONTEXT_REFERENCES):
        return bool(context.last_verified_directory or context.last_verified_file)

    for target in (context.last_verified_directory, context.last_verified_file):
        if target is not None and target.name.lower() in lowered:
            return True
    return False


def classify_intent(
    text: str,
    *,
    accepted: bool = False,
    structured: bool = False,
    pending: bool = False,
    previous_executed: bool = False,
    recent_context: RecentContext | None = None,
) -> IntentCategory:
    """Classify a turn deterministically without granting any authority."""
    lowered = _WHITESPACE.sub(" ", (text or "").strip().lower())
    if lowered in _LOCAL_COMMANDS:
        return IntentCategory.LOCAL_COMMAND
    if lowered in _CANCEL_WORDS:
        return IntentCategory.CANCEL
    if previous_executed and _corrects_previous_action(text):
        return IntentCategory.CORRECTION

    action = accepted or structured or _requires_computer_action(text)
    if action and recent_context is not None and _references_recent_context(
        text, recent_context,
    ):
        return IntentCategory.FOLLOW_UP_ACTION
    if pending and not action:
        return IntentCategory.CLARIFICATION_RESPONSE
    if action:
        return IntentCategory.ACTION
    return IntentCategory.CONVERSATION


def _target_name(event: dict[str, Any]) -> str:
    params = event.get("params") or {}
    raw = (
        params.get("path")
        or params.get("dest")
        or params.get("venv")
        or params.get("open_path")
        or params.get("app")
        or ""
    )
    if not raw:
        return ""
    try:
        return Path(str(raw)).name or str(raw)
    except (OSError, ValueError):
        return str(raw)


def _status_line(event: dict[str, Any]) -> str:
    """One short line describing a trace event, or ``""`` to stay quiet.

    Every line here is a rendering of an event that was actually emitted by the
    control loop. Nothing is timed, predicted, or interpolated between events:
    the display can only lag reality, never invent it.
    """
    return DeimosPresentation().progress(event)


def _debug_status_line(event: dict[str, Any]) -> str:
    """Detailed developer rendering, including internal mechanics."""
    kind = event.get("event", "")

    if kind == "agent_state":
        return f"  state: {event.get('state', '?')}"

    if kind == "planner_call":
        if event.get("error"):
            return f"  planner error: {event['error']}"
        tokens = int(event.get("completion_tokens") or 0)
        seen = " (looked at the screen)" if event.get("vision") else ""
        return f"  planning{seen} ... {tokens} tokens"

    if kind == "policy_decision":
        decision = str(event.get("decision", "")).upper()
        if decision == "ALLOW":
            return ""
        action = (event.get("action") or {}).get("kind", "action")
        return f"  policy {decision.lower()} on {action}: {event.get('reason', '')}"

    if kind == "action":
        result = event.get("result") or {}
        action = (result.get("action") or {}).get("kind", "action")
        if result.get("ok"):
            return f"  {action}: ok"
        return f"  {action}: failed -- {result.get('error') or 'no detail'}"

    if kind == "verification":
        report = event.get("verification") or {}
        stage = "checkpoint" if event.get("checkpoint") else "final"
        return f"  {stage} verification: {report.get('verdict', 'unknown')}"

    if kind == "recovery":
        return (f"  recovering from {event.get('failure_class', '?')}: "
                f"{event.get('decision', '?')} (attempt {event.get('attempt', '?')})")

    if kind == "recovery_resolved":
        return f"  recovered from {event.get('failure_class', '?')}"

    if kind == "failure":
        return f"  failure: {event.get('failure_class', '?')}"

    if kind == "injection":
        return f"  external state change: {event.get('kind', '?')}"

    return ""


@dataclass(frozen=True)
class Prepared:
    """Everything the one execution call needs, whoever assembled it.

    The two routes that execute -- a registered workflow resolved by name or by
    request shape, and a general computer action -- differ only in what they
    have to hand ``run_agent_task``. Naming that difference as a value lets both
    of them converge on a single call expression in :meth:`Session._run` instead
    of each keeping its own copy of the call and its result handling, which is
    the property ``test_there_is_exactly_one_execution_call_site_per_interface``
    exists to hold.

    ``task_obj`` and ``readable_roots`` are the general-action half and default
    to the registered route's values, so passing them is unconditional at the
    call site: ``api.run_agent_task`` reads ``readable_roots`` only in its
    ``task_obj`` branch and overwrites ``task_id`` from the task there, so
    neither route can be affected by the other's fields being present.
    """

    task: UserTask
    #: The one-line goal to announce. Read off whatever knows it -- the task
    #: registry for a registered id, the ``GeneralTask`` for a general action.
    goal: str
    #: Names the route in the "running ..." line, empty for the registered one.
    #: Narration only; nothing branches on it.
    kind: str = ""
    #: Optional stable browser resource class. Same-site follow-up browser
    #: turns may intentionally reuse it; different sites/apps do not.
    browser_resource_key: str | None = None
    task_obj: Task | None = None
    readable_roots: tuple[Path, ...] = ()
    workspace: Path | None = None
    approved_action: dict[str, Any] | None = None
    # P2.5 atomic children are internal execution units. Their result is
    # aggregated into one user-facing workflow reply instead of speaking once
    # per child. Approval/input prompts remain user-facing.
    suppress_presentation: bool = False
    fast_route: Any | None = None
    fast_latency_seconds: float = 0.0
    runtime_task_id: str | None = None
    #: Bounded structured persistent memories recalled at task/conversation entry.
    #: They are context only and are explicitly fenced again by the API.
    persistent_memories: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class PendingInput:
    """The single durable owner for a workflow's next user-provided value."""

    input_id: str
    workflow_id: str
    step_id: str
    task_id: str
    parameter_name: str
    prompt: str
    expected_input_type: str = "text"
    created_at: float = field(default_factory=time.time)
    state: str = "PENDING"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PendingInput":
        required = ("input_id", "workflow_id", "step_id", "task_id", "parameter_name", "prompt")
        if any(not payload.get(k) for k in required):
            raise ValueError("pending_input_not_found: incomplete pending input owner")
        return cls(
            input_id=str(payload["input_id"]), workflow_id=str(payload["workflow_id"]),
            step_id=str(payload["step_id"]), task_id=str(payload["task_id"]),
            parameter_name=str(payload["parameter_name"]), prompt=str(payload["prompt"]),
            expected_input_type=str(payload.get("expected_input_type", "text")),
            created_at=float(payload.get("created_at", time.time())), state=str(payload.get("state", "PENDING")),
        )


@dataclass
class BackgroundTask:
    """Live record for work submitted by the non-blocking chat interface."""

    task_id: str
    original_goal: str
    kind: str = ""
    state: str = "QUEUED"
    result: AgentResult | None = None
    failure: str = ""
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    runtime_task_id: str | None = None
    future: Future | None = field(default=None, repr=False, compare=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "original_goal": self.original_goal,
            "kind": self.kind,
            "state": self.state,
            "result": self.result.to_json() if self.result else None,
            "failure": self.failure,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "runtime_task_id": self.runtime_task_id,
        }


def _friendly_runtime_state(state: str) -> str:
    return {
        "CREATED": "starting",
        "PLANNING": "planning",
        "WAITING_FOR_USER": "waiting for your input",
        "WAITING_FOR_HUMAN": "waiting for your input",
        "RUNNING": "running",
        "VERIFYING": "being verified",
        "COMPLETED": "completed",
        "FAILED": "failed",
        "CANCELLED": "cancelled",
        "BLOCKED": "blocked",
    }.get(state, "in progress")


@dataclass
class Session:
    """A live conversation over the one execution pipeline.

    One ``Session`` per interface instance, not per turn: it holds the narrator
    (so speech can be toggled without restarting), the execution settings, and
    the history. What it does *not* hold is anything the agent needs in order to
    work -- no accumulated context is fed back into planning. That is deliberate.
    The transcript of a conversation is not evidence about the state of the
    machine, and letting turn three inherit turn one's beliefs is how an agent
    starts acting on a world that has since moved.

    :attr:`pending` is the one exception, and it is bounded so tightly that it is
    not really an exception: a single question, holding only the paths it offered,
    taken and cleared by the next :meth:`submit` whatever that turn contains, with
    a time limit on top. It reaches the planner never -- only the path it selects
    does, as an ordinary task parameter.
    """

    narrator: Narrator
    planner: str = "llm"
    max_steps: int = 10
    keep_workspace: bool = False
    use_memory: bool = True
    #: Print live status while a task runs. Printed, never spoken: a running
    #: commentary of every action is the narration Directive C rules out.
    show_status: bool = True
    #: Developer-only presentation. JSON/traces always retain internal ids;
    #: this flag permits them in the live chat display too.
    debug: bool = False
    workspace: Path | None = None
    #: Durable runtime database path. Tests/embedders may use ":memory:"; the CLI
    #: supplies the durable default under .agent_state.
    runtime_persistence_path: str | Path | None = ":memory:"
    _session_owner_id: str = field(default_factory=lambda: f"session-{uuid.uuid4().hex[:12]}", repr=False, compare=False)
    history: list[Turn] = field(default_factory=list)
    #: The one question awaiting an answer, or ``None``. This is not
    #: conversational memory: :meth:`submit` takes and clears it on its first
    #: line, so it cannot outlive the turn immediately after the one that set it,
    #: and :func:`choose` can only ever return a path it already contains.
    pending: Pending | None = None
    pending_approval: PendingApproval | None = None
    #: When ``resume task N`` reaches an approval gate, bare approval responses
    #: belong to that explicitly resumed workflow even if other approvals exist.
    _approval_context_task_id: str | None = field(default=None, repr=False, compare=False)
    pending_workflow_input: dict[str, Any] | None = None
    pending_input: PendingInput | None = field(default=None, repr=False, compare=False)
    _runtime_resume_choices: tuple[str, ...] = field(default_factory=tuple, repr=False, compare=False)
    recent_context: RecentContext = field(default_factory=RecentContext)
    #: Browser resources are task-scoped. Each independent browser task gets
    #: its own BrowserSkill session so selecting/navigating one task cannot
    #: mutate another task's active tab. Resources persist for the Session
    #: lifetime so a completed YouTube task remains available while a later
    #: WhatsApp task runs in its own session.
    _browser_backend: BrowserSkillAdapter | None = field(
        default=None, repr=False, compare=False,
    )
    _browser_tasks: dict[str, BrowserSkillAdapter] = field(
        default_factory=dict, repr=False, compare=False,
    )
    #: General actions in one conversation share one policy write root.  This is
    #: the authority that makes a verified folder writable on the following
    #: turn; the remembered path itself grants nothing.
    _general_workspace: Path | None = field(
        default=None, repr=False, compare=False,
    )
    #: Built lazily on the first Route 3 turn, not eagerly at construction: most
    #: sessions may never say anything purely conversational, and constructing
    #: an ``LLMClient`` has a real connection cost that a session which never
    #: needs it should not pay.
    _conversation_memory_store: ConversationMemory | None = field(
        default=None, repr=False, compare=False,
    )
    _persistent_memory_store: PersistentMemory | None = field(
        default=None, repr=False, compare=False,
    )
    _conversation: ConversationEngine | None = field(
        default=None, repr=False, compare=False,
    )
    #: One serialized background lane.  Keeping a single worker is deliberate:
    #: browser mutations, planner tasks that may reach the browser, approvals,
    #: and verification all pass through the existing Session pipeline without
    #: allowing two turns to mutate the shared browser session concurrently.
    _executor: ThreadPoolExecutor | None = field(
        default=None, repr=False, compare=False,
    )
    _conversation_executor: ThreadPoolExecutor | None = field(
        default=None, repr=False, compare=False,
    )
    _conversation_executor_retired: ThreadPoolExecutor | None = field(
        default=None, repr=False, compare=False,
    )
    _background: dict[str, BackgroundTask] = field(
        default_factory=dict, repr=False, compare=False,
    )
    _background_lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False,
    )
    _closing: bool = field(default=False, repr=False, compare=False)
    _thread_state: threading.local = field(
        default_factory=threading.local, repr=False, compare=False,
    )
    _runtime: RuntimeManager = field(default_factory=RuntimeManager, repr=False, compare=False)
    _workflow_orchestrator: WorkflowOrchestrator | None = field(default=None, repr=False, compare=False)
    _workflow_runtime_adapter: SessionWorkflowRuntime | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Keep direct/unit Session construction lightweight and isolated. The
        # real CLI passes a durable path, which replaces the in-memory manager.
        if self.runtime_persistence_path not in (None, ":memory:"):
            self._runtime.close()
            self._runtime = RuntimeManager(persistence_path=self.runtime_persistence_path)
        self._restore_runtime_views()

    def _owned_pending_approval_records(self) -> list[dict[str, Any]]:
        """Return pending approvals owned by this Session when ownership is known."""
        records = self._pending_approval_records() if hasattr(self, "_pending_approval_records") else list(self._runtime.snapshot().get("pending_approvals", []) or [])
        owned = []
        for item in records:
            task_id = item.get("task_id")
            task = self._runtime.get_task(task_id) if task_id else None
            metadata = task.metadata if task is not None and isinstance(task.metadata, dict) else {}
            if metadata.get("session_owner_id") == self._session_owner_id:
                owned.append(item)
        return owned

    def _rehydrate_approval_owner(self) -> PendingApproval | None:
        """Rebuild the durable approval owner before dispatching user input."""
        if self.pending_approval is not None:
            return self.pending_approval
        approvals = self._pending_approval_records()
        owned = self._owned_pending_approval_records()
        if len(owned) == 1:
            item = owned[0]
        elif len(owned) > 1 or len(approvals) != 1:
            return None
        else:
            # Restart compatibility: an older persisted task may predate the
            # session-owner marker. With exactly one global approval, it is safe
            # to rehydrate it; ambiguity still fails closed.
            item = approvals[0]
        task_id = item.get("task_id")
        task = self._runtime.get_task(task_id) if task_id else None
        payload = item.get("payload") or {}
        if task is None or task.state != "WAITING_FOR_APPROVAL":
            return None
        approval = PendingApproval(
            request=str(payload.get("request") or task.goal),
            action=dict(payload.get("action") or {}),
            asked_at=float(payload.get("asked_at") or time.time()),
            task_id=task.task_id, goal=task.goal,
            background_task_id=payload.get("background_task_id"),
            runtime_task_id=task.task_id,
            workflow_id=payload.get("workflow_id"),
            step_id=payload.get("step_id"),
            approval_request_id=payload.get("approval_request_id"),
        )
        if payload.get("explicit_resume_context"):
            self._approval_context_task_id = task.task_id
        self.pending_approval = approval
        return approval

    def _rehydrate_pending_input_owner(self) -> PendingApproval | None:
        """Rebuild the input owner from RuntimeManager before routing any input.

        ``pending_approval`` is only a session-local presentation/execution
        handle.  RuntimeManager owns the durable owner.  Reading it here on every
        input boundary closes the restart race where a fresh Session would route
        ``yes`` to conversation before its local view had been rebuilt.
        """
        if self.pending_approval is not None:
            return self.pending_approval
        approvals = self._pending_approval_records()
        if not approvals:
            return None
        owned = self._owned_pending_approval_records()
        if len(owned) == 1:
            item = owned[0]
        elif len(owned) > 1 or len(approvals) != 1:
            return None
        else:
            # Restart compatibility for a single legacy approval without an
            # owner marker. Multiple approvals remain ambiguous.
            item = approvals[0]
        task_id = item.get("task_id")
        task = self._runtime.get_task(task_id) if task_id else None
        payload = item.get("payload") or {}
        if task is None or task.state != "WAITING_FOR_APPROVAL":
            return None
        approval = PendingApproval(
            request=str(payload.get("request") or task.goal),
            action=dict(payload.get("action") or {}),
            asked_at=float(payload.get("asked_at") or time.time()),
            task_id=task.task_id, goal=task.goal,
            background_task_id=payload.get("background_task_id"),
            runtime_task_id=task.task_id,
            workflow_id=payload.get("workflow_id"),
            step_id=payload.get("step_id"),
            approval_request_id=payload.get("approval_request_id"),
        )
        self.pending_approval = approval
        return approval

    def _restore_runtime_views(self) -> None:
        """Rebuild non-authoritative presentation/approval views from runtime truth."""
        self._rehydrate_pending_input_owner()
        # BackgroundTask is a presentation/execution bookkeeping view only.
        # Recreate it from the authoritative runtime registry after restart.
        with self._background_lock:
            for task in self._runtime.list_tasks():
                if not task.metadata.get("background"):
                    continue
                if task.task_id in self._background:
                    continue
                state = task.state
                self._background[task.task_id] = BackgroundTask(
                    task_id=task.task_id, original_goal=task.goal,
                    kind=task.task_type or "background", state=state,
                    submitted_at=task.created_at, started_at=task.started_at,
                    finished_at=task.completed_at, runtime_task_id=task.task_id,
                )

    @classmethod
    def build(cls, *, speech: bool | None = None,
              write: Callable[[str], None] = print, **settings: Any) -> "Session":
        return cls(narrator=Narrator.build(enabled=speech, write=write), **settings)

    # -- output ------------------------------------------------------------
    @property
    def speaking(self) -> bool:
        return self.narrator.speaking

    def set_speaking(self, enabled: bool) -> bool:
        return self.narrator.set_speaking(enabled)

    def _browser_for_task(self, task_id: str, *, resource_key: str | None = None) -> BrowserSkillAdapter:
        """Return the BrowserSkill session owned by one task.

        A task resource is a full BrowserSkill session, not merely a selected
        tab. This is the smallest isolation boundary supported by the current
        BrowserSkill runtime: every session has its own current-tab context.
        The legacy ``_browser_backend`` remains accepted for tests/embedding,
        but production task execution never shares it directly.
        """
        key = str(resource_key or task_id or "")
        if not key:
            raise ValueError("browser task id is required")
        if key in {"browser", "youtube", "whatsapp", "gmail"}:
            key = "browser"
        existing = self._browser_tasks.get(key)
        if existing is not None:
            return existing

        base = self._browser_backend
        if base is not None and hasattr(base, "new_task_session"):
            backend = base.new_task_session()
        else:
            backend = BrowserSkillAdapter()
        try:
            backend.debug = self.debug
        except Exception:
            pass
        self._browser_tasks[key] = backend
        return backend

    def close(self, timeout_s: float = 15.0) -> None:
        """Close interface-owned resources and drain the background lane."""
        self._closing = True
        executor = self._executor
        self._executor = None
        if executor is not None:
            # ``shutdown(wait=True)`` is the safe default for BrowserSkill: do not
            # abandon a live browser mutation halfway through an operation.
            executor.shutdown(wait=True, cancel_futures=True)
        conversation_executor = self._conversation_executor
        retired_conversation_executor = self._conversation_executor_retired
        self._conversation_executor = None
        self._conversation_executor_retired = None
        for executor in (conversation_executor, retired_conversation_executor):
            if executor is not None and executor is not conversation_executor:
                executor.shutdown(wait=True, cancel_futures=True)
        if conversation_executor is not None:
            conversation_executor.shutdown(wait=True, cancel_futures=True)
        try:
            self.narrator.close(timeout_s)
        finally:
            try:
                client = getattr(self._conversation, "client", None) if self._conversation is not None else None
                close = getattr(client, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            backends = list(self._browser_tasks.values())
            self._browser_tasks.clear()
            legacy = self._browser_backend
            self._browser_backend = None
            if legacy is not None and legacy not in backends:
                backends.append(legacy)
            for backend in backends:
                try:
                    backend.close_session()
                except Exception:
                    pass
            try:
                if self._workflow_orchestrator is not None:
                    self._workflow_orchestrator.close()
                    self._workflow_orchestrator = None
            except Exception:
                pass
            try:
                self._runtime.close()
            except Exception:
                pass
            try:
                if self._persistent_memory_store is not None:
                    self._persistent_memory_store.close()
                    self._persistent_memory_store = None
            except Exception:
                pass

    def _conversation_background_executor(self) -> ThreadPoolExecutor:
        if self._closing:
            raise RuntimeError("session is shutting down")
        if self._conversation_executor is None:
            retired = self._conversation_executor_retired
            if retired is not None:
                # A transport failure retires the public lane, but work already
                # queued on that single-worker lane must finish in order. Do not
                # create a second conversation lane while it is still draining.
                return retired
            self._conversation_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="deimos-conversation"
            )
        return self._conversation_executor

    def _background_executor(self) -> ThreadPoolExecutor:
        if self._closing:
            raise RuntimeError("session is shutting down")
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=max(2, min(4, int(os.getenv("DEIMOS_TASK_CONCURRENCY", "4")))),
                thread_name_prefix="deimos-task",
            )
        return self._executor

    def _runtime_create(
        self, goal: str, workflow_task_id: str = "", *,
        parent_task_id: str | None = None, runtime_task_id: str | None = None,
        metadata: dict[str, Any] | None = None, task_type: str = "",
        source_input_id: str = "",
    ) -> str:
        """Create exactly one authoritative task record.

        The workflow task id is the runtime task id for executable work.
        Internal/background callers may provide an explicit runtime id; no
        continuation is allowed to mint a second identity.
        """
        task_id = runtime_task_id or workflow_task_id or f"task-{uuid.uuid4().hex[:12]}"
        task_metadata = dict(metadata or {})
        task_metadata.setdefault("session_owner_id", self._session_owner_id)
        self._runtime.create_task(
            goal, task_id=task_id, task_type=task_type,
            source_input_id=source_input_id, parent_task_id=parent_task_id,
            metadata=task_metadata,
        )
        return task_id

    def _runtime_transition(self, runtime_task_id: str | None, status: str, *, event_type: str | None = None, **kwargs: Any) -> None:
        if not runtime_task_id:
            return
        try:
            self._runtime.transition_task(runtime_task_id, status, event_type=event_type, **kwargs)
        except ValueError as exc:
            # Never silently discard a lifecycle mismatch. Keep RuntimeManager
            # authoritative (the state is not mutated), and journal the rejected
            # transition so diagnostics and regression tests can see exactly what
            # was attempted. The caller may continue only where its surrounding
            # pipeline can safely do so.
            try:
                self._runtime.event(runtime_task_id, "RUNTIME_TRANSITION_REJECTED", {
                    "requested_state": status, "error": str(exc),
                })
            except KeyError:
                pass
            self.narrator.note(f"RUNTIME: rejected transition {runtime_task_id} -> {status}: {exc}")
        except KeyError as exc:
            self.narrator.note(f"RUNTIME: task missing for transition {runtime_task_id} -> {status}: {exc}")

    def _runtime_event(self, runtime_task_id: str | None, event_type: str, metadata: dict[str, Any] | None = None) -> None:
        if not runtime_task_id:
            return
        try:
            self._runtime.event(runtime_task_id, event_type, metadata)
        except KeyError:
            return

    def runtime_snapshot(self) -> dict[str, Any]:
        """Return the safe authoritative runtime snapshot for CLI/conversation."""
        return self._runtime.snapshot()

    def _runtime_event_watcher(self, runtime_task_id: str | None, callback: Callable[[dict[str, Any]], None] | None) -> Callable[[dict[str, Any]], None] | None:
        if callback is None and not runtime_task_id:
            return None
        def watch(event: dict[str, Any]) -> None:
            kind = str(event.get("event", ""))
            state = str(event.get("state", ""))
            if runtime_task_id:
                runtime_task = self._runtime.get_task(runtime_task_id)
                is_workflow_task = bool(
                    runtime_task is not None
                    and (
                        runtime_task.task_type == "workflow"
                        or runtime_task.metadata.get("workflow_orchestration") == "langgraph"
                    )
                )
                mapping = {
                    "RUNNING": ("RUNNING", "TASK_RUNNING"),
                    "PLANNING": ("PLANNING", "TASK_PLANNING"),
                    # AgentState.ACTING is the execution phase represented by
                    # RuntimeManager's existing RUNNING state.  It must be
                    # synchronized explicitly; otherwise the PLANNING event
                    # overwrites RUNNING and later VERIFYING/COMPLETED
                    # transitions are rejected as impossible.  Do not add a
                    # parallel ACTING runtime state.
                    "ACTING": ("RUNNING", "TASK_ACTING"),
                    "VERIFYING": ("VERIFYING", "VERIFICATION_STARTED"),
                    "WAITING_FOR_USER": ("WAITING_FOR_USER", "TASK_WAITING"),
                    # A WorkflowStepTask is one child execution inside a
                    # durable workflow. Its runner COMPLETED event means
                    # "this step finished", not "the workflow task is
                    # terminal". Keep the RuntimeManager task non-terminal so
                    # the next step can legitimately request its own approval.
                    # The workflow node emits the real workflow COMPLETED
                    # transition only after every step has independently
                    # verified.
                    "COMPLETED": ("VERIFYING", "WORKFLOW_STEP_EXECUTION_COMPLETED") if is_workflow_task else ("COMPLETED", "TASK_COMPLETED"),
                    # Likewise, a failed child execution must not force the
                    # durable workflow identity terminal before its workflow
                    # recovery/failed node records the final outcome.
                    "FAILED": ("RECOVERY_REQUIRED", "WORKFLOW_STEP_EXECUTION_FAILED") if is_workflow_task else ("FAILED", "TASK_FAILED"),
                    "CANCELLED": ("CANCELLED", "TASK_CANCELLED"),
                }
                mapped = mapping.get(state)
                if mapped:
                    status, event_type = mapped
                    self._runtime_transition(runtime_task_id, status, event_type=event_type, current_phase=kind or state.casefold())
                elif kind == "verification":
                    verdict = str((event.get("verification") or {}).get("verdict", "UNKNOWN"))
                    self._runtime.update_task(runtime_task_id, verification_state=verdict, verification_summary=verdict, current_phase="verification")
                    self._runtime_event(runtime_task_id, "VERIFICATION_PASSED" if verdict == "PASS" else "TASK_VERIFICATION_UNKNOWN", {"verdict": verdict})
                elif kind == "recovery":
                    self._runtime_event(runtime_task_id, "TASK_RECOVERY", {"state": state})
                elif kind == "failure":
                    self._runtime_event(runtime_task_id, "TASK_FAILED", {"state": state})
            if callback is not None:
                callback(event)
        return watch

    def _approval_turn(self, raw: str, source: str, message: str, approval: PendingApproval | None = None) -> Turn:
        approval = approval or self.pending_approval
        task_id = approval.task_id if approval is not None else ""
        task = UserTask(
            raw=raw,
            text=approval.request if approval is not None else raw,
            source=source,
            task_id=task_id,
            status="accepted",
        )
        turn = Turn(task=task, reply=message, result=None)
        self._emit_reply(message, goal=task.text)
        return turn

    def _rehydrate_workflow_input(self) -> dict[str, Any] | None:
        """Rehydrate the authoritative pending-input owner from RuntimeManager.

        The local object is only a dispatcher view. The durable owner lives in
        the runtime task metadata, so restart cannot turn a workflow answer into
        an unrelated conversation turn.
        """
        if self.pending_input is not None and self.pending_input.state == "PENDING":
            return {
                "input_id": self.pending_input.input_id, "workflow_id": self.pending_input.workflow_id,
                "step_id": self.pending_input.step_id, "task_id": self.pending_input.task_id,
                "field": self.pending_input.parameter_name,
            }
        if self.pending_workflow_input is not None:
            try:
                self.pending_input = PendingInput.from_dict({
                    "input_id": self.pending_workflow_input.get("input_id", "legacy"),
                    "workflow_id": self.pending_workflow_input["workflow_id"],
                    "step_id": self.pending_workflow_input["step_id"],
                    "task_id": self.pending_workflow_input.get("task_id", self.pending_workflow_input["workflow_id"]),
                    "parameter_name": self.pending_workflow_input["field"],
                    "prompt": self.pending_workflow_input.get("prompt", "Please provide the missing value."),
                })
                return dict(self.pending_workflow_input)
            except (KeyError, ValueError, TypeError):
                self.pending_workflow_input = None
                self.pending_input = None

        # Focused task first; this is still RuntimeManager's one authoritative
        # registry, not a second input-routing registry.
        tasks = self._runtime.list_waiting_tasks()
        focused = self._runtime.snapshot().get("focused_task_id")
        ordered = sorted(tasks, key=lambda t: 0 if t.task_id == focused else 1)

        # Do not infer ownership from focus or insertion order when more than one
        # durable workflow-input owner exists. A generic answer must not guess.
        valid_owner_count = 0
        for candidate in ordered:
            metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
            owner = metadata.get("pending_input")
            workflow_data = metadata.get("workflow")
            if not isinstance(owner, dict) or not isinstance(workflow_data, dict):
                continue
            try:
                wf = Workflow.from_json(workflow_data)
            except Exception:
                continue
            step = next((x for x in wf.steps if x.step_id == owner.get("step_id")), None)
            if step is not None and step.state == "WAITING_FOR_USER":
                valid_owner_count += 1
        if valid_owner_count > 1:
            return None

        for task in ordered:
            metadata = task.metadata if isinstance(task.metadata, dict) else {}
            owner = metadata.get("pending_input")
            if isinstance(owner, dict) and str(owner.get("state", "PENDING")) == "PENDING":
                try:
                    pending = PendingInput.from_dict(owner)
                    workflow_data = metadata.get("workflow")
                    if not isinstance(workflow_data, dict):
                        raise ValueError("workflow_state_invalid: pending input has no workflow")
                    workflow = Workflow.from_json(workflow_data)
                    step = next((x for x in workflow.steps if x.step_id == pending.step_id), None)
                    if step is None or step.state != "WAITING_FOR_USER":
                        raise ValueError("pending_input_not_found: owner does not match waiting workflow step")
                    self.pending_input = pending
                    self.pending_workflow_input = {
                        "input_id": pending.input_id, "workflow_id": pending.workflow_id,
                        "step_id": pending.step_id, "task_id": pending.task_id,
                        "field": pending.parameter_name, "prompt": pending.prompt,
                    }
                    return dict(self.pending_workflow_input)
                except (ValueError, TypeError, KeyError):
                    continue

        # Backward-compatible recovery for P2.3 databases written before the
        # explicit owner record existed: derive it once from the waiting step,
        # then persist the durable owner so subsequent turns are explicit.
        for task in ordered:
            workflow_data = task.metadata.get("workflow") if isinstance(task.metadata, dict) else None
            if not isinstance(workflow_data, dict):
                continue
            try:
                wf = Workflow.from_json(workflow_data)
            except Exception:
                continue
            for step in wf.steps:
                if step.state != "WAITING_FOR_USER":
                    continue
                field_name = ("message" if step.action.kind == "whatsapp_send_message" and not step.action.params.get("message")
                              else "query" if not step.action.params.get("query") else None)
                if not field_name:
                    continue
                pending = PendingInput(
                    input_id=f"input-{uuid.uuid4().hex[:10]}", workflow_id=wf.workflow_id,
                    step_id=step.step_id, task_id=task.task_id, parameter_name=field_name,
                    prompt=(f"What should I send to {step.action.params.get('recipient', 'the contact')}?" if field_name == "message" else "What should I search for?"),
                )
                self.pending_input = pending
                self.pending_workflow_input = {
                    "input_id": pending.input_id, "workflow_id": pending.workflow_id,
                    "step_id": pending.step_id, "task_id": pending.task_id, "field": pending.parameter_name,
                    "prompt": pending.prompt,
                }
                try:
                    self._runtime.set_pending_input(task.task_id, self.pending_workflow_input)
                except Exception:
                    pass
                return dict(self.pending_workflow_input)
        return None

    def _workflow_engine(self) -> WorkflowOrchestrator:
        if self._workflow_orchestrator is None:
            adapter = self._workflow_runtime_adapter or SessionWorkflowRuntime(self)
            self._workflow_runtime_adapter = adapter
            configured = self.runtime_persistence_path
            if configured in (None, ":memory:"):
                checkpoint_path = ":memory:"
            else:
                path = Path(configured)
                path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path = str(path.with_name("workflow.sqlite3"))
            self._workflow_orchestrator = WorkflowOrchestrator(adapter, checkpoint_path)
        return self._workflow_orchestrator

    @staticmethod
    def _workflow_id_from_goal(prefix: str = "fast") -> str:
        return f"{prefix}-{uuid.uuid4().hex[:4]}"

    def _workflow_turn(self, workflow_id: str, goal: str, graph_result: dict[str, Any], *, source: str) -> Turn:
        task = UserTask(raw=goal, text=goal, source=source, task_id=workflow_id, status="accepted")
        interrupts = graph_result.get("__interrupt__") if isinstance(graph_result, dict) else None
        if interrupts:
            self._rehydrate_approval_owner()
            self._rehydrate_workflow_input()
            question = None
            if self.pending_approval is not None:
                question = Clarification(question=approval_prompt(self.pending_approval.action, goal), context="APPROVAL_ACTION:" + json.dumps({"workflow_id": workflow_id, "step_id": self.pending_approval.step_id, "approval_request_id": self.pending_approval.approval_request_id, "action": self.pending_approval.action}, sort_keys=True))
            elif self.pending_workflow_input is not None:
                question = Clarification(question=self.pending_workflow_input.get("prompt", "Please provide the missing value."), context="WORKFLOW_INPUT:" + json.dumps(self.pending_workflow_input, sort_keys=True))
            result = AgentResult(request=goal, task_id=workflow_id, status=TaskStatus.NEEDS_INPUT, verified=Verdict.UNKNOWN.value, question=question)
            reply = question.question if question is not None else "The workflow is waiting for input."
            if self.pending_approval is not None:
                reply = approval_prompt(self.pending_approval.action, goal)
            self._emit_reply(reply, goal=goal, state="WAITING_FOR_APPROVAL" if self.pending_approval else "WAITING_FOR_USER")
            return Turn(task=task, reply=reply, result=result)

        runtime_task = self._runtime.get_task(workflow_id)
        workflow_data = runtime_task.metadata.get("workflow") if runtime_task else None
        workflow = Workflow.from_json(workflow_data) if isinstance(workflow_data, dict) else None
        if workflow is not None and workflow.state == "COMPLETED":
            result = AgentResult(request=goal, task_id=workflow_id, status=TaskStatus.SUCCESS, verified=Verdict.PASS.value, completed=[s.step_id for s in workflow.steps], detail="workflow completed after independent verification")
            reply = self._workflow_completion_reply(workflow)
            self._emit_reply(reply, goal=goal, state="COMPLETED")
            return Turn(task=task, reply=reply, result=result)
        detail = workflow.status.value if workflow is not None else "workflow failed"
        result = AgentResult(request=goal, task_id=workflow_id, status=TaskStatus.UNKNOWN, verified=Verdict.UNKNOWN.value, detail=detail)
        reply = "The workflow did not complete."
        self._emit_reply(reply, goal=goal, state=detail)
        return Turn(task=task, reply=reply, result=result)

    @staticmethod
    def _looks_like_compound_action(text: str) -> bool:
        """Detect likely multi-action requests without turning conversation into workflows."""
        normalized = " ".join((text or "").casefold().split())
        if not normalized:
            return False
        if re.search(r"\bthen\b|\bafter that\b", normalized):
            return True
        if not re.search(r"\band\b|,", normalized):
            return False
        verbs = (
            "open ", "launch ", "play ", "send ", "type ", "click ", "search ",
            "write ", "read ", "close ", "refresh ", "go to ", "create ", "run ",
        )
        # Count action occurrences, not distinct verb spellings: ``open A and
        # open B`` has one verb kind but two executable clauses.
        return sum(len(re.findall(re.escape(verb), normalized)) for verb in verbs) >= 2

    def _start_workflow(self, goal: str, *, source: str = "text", workflow_id: str | None = None) -> Turn:
        workflow_id = workflow_id or self._workflow_id_from_goal()
        self._runtime_create(goal, workflow_id, runtime_task_id=workflow_id, task_type="workflow", metadata={"workflow_orchestration": "langgraph"})
        # A CLI workflow may already have a RuntimeManager row because the
        # conversational/background lane reserves the task identity before
        # dispatching ``submit``. ``create_task`` is intentionally idempotent,
        # so that reservation keeps its original task_type/metadata. Mark the
        # existing row as a LangGraph workflow projection explicitly; the
        # runner event watcher uses this durable marker to distinguish a child
        # workflow-step COMPLETED event from terminal completion of the whole
        # workflow. This changes no lifecycle state and never weakens the
        # RuntimeManager transition validator.
        runtime_task = self._runtime.get_task(workflow_id)
        if runtime_task is not None:
            metadata = dict(runtime_task.metadata)
            metadata["workflow_orchestration"] = "langgraph"
            metadata["workflow_goal"] = goal
            self._runtime.update_task(workflow_id, metadata=metadata)
        self._runtime_transition(workflow_id, "RUNNING", event_type="WORKFLOW_STARTED", current_phase="workflow", execution_state="RUNNING")
        try:
            result = self._workflow_engine().start(workflow_id, goal)
        except RuntimeError as exc:
            self._runtime_transition(workflow_id, "FAILED", event_type="WORKFLOW_FAILED", failure_state=str(exc), failure_category="DEPENDENCY", execution_state="FAILED")
            task = UserTask(raw=goal, text=goal, source=source, task_id=workflow_id, status="accepted")
            agent_result = AgentResult(request=goal, task_id=workflow_id, status=TaskStatus.UNAVAILABLE, verified=Verdict.UNKNOWN.value, detail=str(exc))
            reply = f"The workflow engine is unavailable: {exc}"
            self._emit_reply(reply, goal=goal, state="FAILED")
            return Turn(task=task, reply=reply, result=agent_result)
        return self._workflow_turn(workflow_id, goal, result, source=source)

    def _resume_workflow_graph(self, workflow_id: str, value: Any, *, source: str = "text") -> Turn:
        """Resume a durable workflow checkpoint, with a safe state-based fallback.

        Some LangGraph/checkpointer versions raise ``KeyError`` while resuming an
        interrupt whose checkpoint was created by an earlier graph schema.  That
        must not turn an already-approved action into a submission error.  The
        persisted RuntimeManager workflow is the authoritative recovery source:
        consume the exact approval, mark only the waiting step runnable, and send
        it back through the normal workflow-step runner.
        """
        task = self._runtime.get_task(workflow_id)
        if task is None:
            return self._refused(UserTask(raw=str(value), text=str(value), source=source, status="dropped"), "That workflow is no longer available.")
        try:
            result = self._workflow_engine().resume(workflow_id, value)
            return self._workflow_turn(workflow_id, task.goal, result, source=source)
        except KeyError as exc:
            metadata = task.metadata if isinstance(task.metadata, dict) else {}
            workflow_data = metadata.get("workflow")
            if not isinstance(workflow_data, dict):
                raise
            workflow = Workflow.from_json(workflow_data)
            step = next((item for item in workflow.steps if item.step_id == workflow.current_step_id), None)
            if step is None:
                step = next((item for item in workflow.steps if item.status.value == "WAITING_FOR_APPROVAL" or str(item.status) == "WAITING_FOR_APPROVAL"), None)
            if step is None:
                raise
            approval = self._runtime.get_approval(workflow_id)
            approved = bool(value) if not isinstance(value, dict) else bool(value.get("approved"))
            if approval is not None:
                self._runtime.resolve_approval(workflow_id, approved, approval_request_id=approval.get("approval_request_id"))
            if not approved:
                step.status = __import__("agent_control.workflow.models", fromlist=["StepStatus"]).StepStatus.CANCELLED
                workflow.status = __import__("agent_control.workflow.models", fromlist=["WorkflowStatus"]).WorkflowStatus.CANCELLED
                self._runtime.update_workflow(workflow_id, workflow.to_json(), event_type="APPROVAL_DENIED")
                return self._workflow_turn(workflow_id, task.goal, {"result": {"status": "CANCELLED"}}, source=source)
            step.status = __import__("agent_control.workflow.models", fromlist=["StepStatus"]).StepStatus.PENDING
            workflow.status = __import__("agent_control.workflow.models", fromlist=["WorkflowStatus"]).WorkflowStatus.RUNNING
            workflow.current_step_id = step.step_id
            updated = dict(task.metadata)
            updated["workflow_approval_granted"] = True
            updated["workflow_approval_granted_step"] = step.step_id
            updated["workflow"] = workflow.to_json()
            self._runtime.update_task(workflow_id, metadata=updated)
            self._runtime.update_workflow(workflow_id, workflow.to_json(), event_type="APPROVAL_GRANTED_FALLBACK")
            return self._resume_workflow(workflow, source=source)

    @staticmethod
    def _workflow_completion_reply(workflow: Workflow) -> str:
        """One deterministic spoken summary after every child is verified."""
        parts: list[str] = []
        for step in workflow.steps:
            action = step.action
            params = action.params
            if action.kind == "launch_app":
                app = str(params.get("app", "the application")).strip()
                parts.append(f"{app.title()} is opened")
            elif action.kind == "type_text":
                text = str(params.get("text", "")).strip()
                app = str(params.get("app", "the application")).strip()
                parts.append(f"typed {text!r} in {app.title()}")
            elif action.kind == "browser_play_song":
                query = str(params.get("query", "the requested song")).strip()
                parts.append(f"started playing {query}")
            elif action.kind == "whatsapp_send_message":
                message = str(params.get("message", "")).strip()
                recipient = str(params.get("recipient", "the contact")).strip()
                parts.append(f"sent {message!r} to {recipient}")
            else:
                parts.append(action.kind.replace("_", " "))
        if not parts:
            return "Done, sir. Anything else, sir?"
        if len(parts) == 1:
            summary = parts[0]
        elif len(parts) == 2:
            summary = f"{parts[0]} and {parts[1]}"
        else:
            summary = ", ".join(parts[:-1]) + f", and {parts[-1]}"
        return f"Done, sir. {summary}. Anything else, sir?"

    def _resume_workflow(self, workflow: Workflow, *, source: str = "text", background: bool = False) -> Turn:
        """Run the next durable sequential step through the existing execution pipeline."""
        for step in workflow.steps:
            if step.state in {"COMPLETED"}:
                continue
            if step.state in {"BLOCKED", "FAILED", "UNKNOWN", "RECOVERY_REQUIRED"}:
                reply = step.failure_reason or "The workflow needs recovery before it can continue."
                self._emit_reply(reply, goal=workflow.goal)
                return Turn(task=UserTask(raw=workflow.goal, text=workflow.goal, source=source, task_id=workflow.workflow_id, status="accepted"), reply=reply)
            if step.state == "WAITING_FOR_APPROVAL":
                return Turn(task=UserTask(raw=workflow.goal, text=workflow.goal, source=source, task_id=workflow.workflow_id, status="accepted"), reply=approval_prompt(step.action.to_json(), workflow.goal))
            step.state = "PENDING"
            prepared_task = WorkflowStepTask(workflow=workflow, step=step)
            task = UserTask(raw=workflow.goal, text=workflow.goal, source=source, task_id=workflow.workflow_id, status="accepted")
            turn = self._run(Prepared(
                task=task,
                goal=workflow.goal,
                kind="workflow step",
                task_obj=prepared_task,
                workspace=self.workspace,
                runtime_task_id=workflow.workflow_id,
                suppress_presentation=True,
            ))
            result = turn.result
            if result is None:
                return turn
            if result.needs_input or not result.ok:
                return turn
            continue
        workflow.state = "COMPLETED"
        self._runtime.update_workflow(workflow.workflow_id, workflow.to_json(), event_type="WORKFLOW_COMPLETED")
        reply = self._workflow_completion_reply(workflow)
        turn = Turn(task=UserTask(raw=workflow.goal, text=workflow.goal, source=source, task_id=workflow.workflow_id, status="accepted"), reply=reply, result=None)
        self._emit_reply(reply, goal=workflow.goal)
        return turn

    def _has_multiple_pending_workflow_inputs(self) -> bool:
        candidates = []
        for task in self._runtime.list_waiting_tasks():
            metadata = task.metadata if isinstance(task.metadata, dict) else {}
            owner = metadata.get("pending_input")
            if not isinstance(owner, dict) or str(owner.get("state", "PENDING")) != "PENDING":
                continue
            workflow_data = metadata.get("workflow")
            if not isinstance(workflow_data, dict):
                continue
            try:
                workflow = Workflow.from_json(workflow_data)
            except Exception:
                continue
            step_id = owner.get("step_id")
            if any(step.step_id == step_id and step.state == "WAITING_FOR_USER" for step in workflow.steps):
                candidates.append(task.task_id)
        return len(candidates) > 1

    def _resolve_workflow_input(self, raw: str, *, source: str, task_id: str | None = None) -> Turn:
        """Resolve the next value against the exact durable workflow/step owner."""
        if task_id is not None:
            runtime_task_for_input = self._runtime.get_task(task_id)
            metadata = runtime_task_for_input.metadata if runtime_task_for_input is not None and isinstance(runtime_task_for_input.metadata, dict) else {}
            pending_data = metadata.get("pending_input") if isinstance(metadata.get("pending_input"), dict) else None
            if pending_data is None:
                return self._refused(
                    UserTask(raw=raw, text=raw, source=source, task_id=task_id, status="dropped"),
                    "That task is not waiting for your input.",
                )
        else:
            if self._has_multiple_pending_workflow_inputs():
                return self._refused(UserTask(raw=raw, text=raw, source=source, status="dropped"), "I need the task number because more than one workflow is waiting for input.")
            pending_data = self._rehydrate_workflow_input()
            if pending_data is None:
                return self._route_without_owned_input(raw, source=source)

        try:
            pending = PendingInput.from_dict({
                "input_id": pending_data.get("input_id", ""),
                "workflow_id": pending_data["workflow_id"], "step_id": pending_data["step_id"],
                "task_id": pending_data.get("task_id", pending_data["workflow_id"]),
                "parameter_name": pending_data["field"],
                "prompt": pending_data.get("prompt", "Please provide the missing value."),
            })
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError(f"pending_input_not_found: {exc}") from exc

        runtime_task = self._runtime.get_task(pending.task_id)
        if runtime_task is None:
            self._clear_pending_input_local()
            return self._refused(UserTask(raw=raw, text=raw, source=source, status="dropped"), "That workflow is no longer available.")
        try:
            workflow = Workflow.from_json(runtime_task.metadata.get("workflow", {}))
            step = next((x for x in workflow.steps if x.step_id == pending.step_id), None)
            if step is None:
                raise RuntimeError(f"pending_input_not_found: step {pending.step_id!r} is not in workflow")
            if step.state != "WAITING_FOR_USER":
                raise RuntimeError(f"pending_input_not_found: step {pending.step_id!r} is not waiting for user input")
            if workflow.current_step != step.index:
                raise RuntimeError(f"invalid_current_step: workflow current_step={workflow.current_step}, owner step={step.index}")

            value = self._normalize_workflow_input_value(raw, pending.parameter_name)
            if not value:
                return self._refused(UserTask(raw=raw, text=raw, source=source, status="dropped"), pending.prompt)

            if runtime_task.metadata.get("workflow_orchestration") == "langgraph":
                self._clear_pending_input_local()
                self._runtime.clear_pending_input(pending.task_id)
                return self._resume_workflow_graph(pending.workflow_id, value, source=source)

            # Inject only the named parameter; all other structured parameters and
            # the original natural-language workflow goal remain untouched.
            step.action.params[pending.parameter_name] = value
            step.state = "PENDING"
            step.failure_reason = ""
            workflow.state = "RUNNING"
            workflow.current_step = step.index
            self._runtime.clear_pending_input(pending.task_id)
            self._clear_pending_input_local()
            self._runtime_transition(pending.task_id, "RUNNING", event_type="WORKFLOW_INPUT_RESOLVED", execution_state="RUNNING")
            self._runtime.update_workflow(workflow.workflow_id, workflow.to_json(), event_type="WORKFLOW_INPUT_RECEIVED")
            return self._resume_workflow(workflow, source=source)
        except RuntimeError:
            raise

    @staticmethod
    def _normalize_workflow_input_value(raw: str, field: str) -> str:
        value = str(raw or "").strip()
        if field == "message":
            match = re.match(r"^(?:say|tell)\s+(?:her|him|them)\s+(.+)$", value, flags=re.I)
            if match:
                value = match.group(1).strip()
        return value

    def _clear_pending_input_local(self) -> None:
        self.pending_input = None
        self.pending_workflow_input = None

    def _route_without_owned_input(self, raw: str, *, source: str) -> Turn:
        """Continue through the normal dispatcher only when no owner exists."""
        return self.submit(raw, source=source)

    def _pending_approval_records(self) -> list[dict[str, Any]]:
        return list(self._runtime.snapshot().get("pending_approvals", []) or [])

    def _has_multiple_pending_approvals(self) -> bool:
        records = self._pending_approval_records()
        owned = self._owned_pending_approval_records()
        return len(owned) > 1 or (not owned and len(records) > 1)

    @staticmethod
    def _approval_label(task: Any, payload: dict[str, Any]) -> str:
        action = payload.get("action") if isinstance(payload.get("action"), dict) else {}
        kind = str(action.get("kind") or "")
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        if kind == "whatsapp_send_message":
            recipient = str(params.get("recipient") or params.get("to") or "the recipient").strip()
            message = str(params.get("message") or "").strip()
            return f'Send "{message}" to {recipient} on WhatsApp' if message else f"Send a WhatsApp message to {recipient}"
        if kind == "gmail_send_email":
            recipient = str(params.get("recipient") or params.get("to") or "the recipient").strip()
            subject = str(params.get("subject") or "").strip()
            return f'Send an email to {recipient}' + (f' about "{subject}"' if subject else "")
        if kind in {"browser_play_song", "youtube_play", "browser_play_video"}:
            query = str(params.get("query") or params.get("song") or "the video").strip()
            return f"Play {query} on YouTube"
        return str(getattr(task, "goal", "") or payload.get("request") or "the requested action").strip()

    def _approval_candidates(self) -> list[dict[str, Any]]:
        records = self._pending_approval_records()
        owned = self._owned_pending_approval_records()
        return owned if owned else records

    def _approval_owner_for_input(self, raw: str) -> PendingApproval | None:
        candidates = self._approval_candidates()
        if not candidates:
            return None
        records = self._pending_approval_records()
        match = re.search(r"\b(?:fast|task)-[a-z0-9]+(?:-[a-z0-9]+)?\b", str(raw or ""), re.I)
        selected = None
        if match:
            selected = next((x for x in records if str(x.get("task_id", "")).casefold() == match.group(0).casefold()), None)
        elif len(candidates) == 1:
            selected = candidates[0]
        else:
            choices = tuple(api.Choice(path=str(item.get("task_id", "")), label=self._approval_label(self._runtime.get_task(item.get("task_id")), item.get("payload") or {}), detail="") for item in candidates)
            picked = choose(Pending(query="Which one do you approve?", task_id="approval-selection", choices=choices, asked_at=time.time()), raw)
            if picked:
                selected = next((x for x in candidates if str(x.get("task_id", "")) == picked), None)
        if selected is None:
            return None
        task_id = selected.get("task_id")
        task = self._runtime.get_task(task_id) if task_id else None
        payload = selected.get("payload") or {}
        if task is None or task.state != "WAITING_FOR_APPROVAL":
            return None
        approval = PendingApproval(
            request=str(payload.get("request") or task.goal), action=dict(payload.get("action") or {}),
            asked_at=float(payload.get("asked_at") or time.time()), task_id=task.task_id, goal=task.goal,
            background_task_id=payload.get("background_task_id"), runtime_task_id=task.task_id,
            workflow_id=payload.get("workflow_id"), step_id=payload.get("step_id"),
            approval_request_id=payload.get("approval_request_id"),
        )
        self.pending_approval = approval
        return approval

    def _multiple_approval_prompt(self) -> str:
        lines = ["There are multiple actions waiting for approval:"]
        for index, item in enumerate(self._approval_candidates(), start=1):
            task = self._runtime.get_task(item.get("task_id"))
            lines.append(f"{index}. {self._approval_label(task, item.get('payload') or {})}")
        lines.append("Which one do you approve?")
        return "\n".join(lines)

    def _resolve_approval_input(self, raw: str, *, source: str, background: bool) -> Turn:
        with self._background_lock:
            approval = self.pending_approval
        targeted_context = bool(approval is not None and self._approval_context_task_id == approval.task_id)
        multiple = len(self._approval_candidates()) > 1 and not targeted_context
        resolution = "resumed_task_owner" if targeted_context else ("single_pending_owner" if approval is not None and not multiple else ("session_pending_owner" if approval is not None else None))
        if approval is None:
            approval = self._approval_owner_for_input(raw)
            if approval is not None:
                resolution = "single_pending_owner" if not multiple else "human_selection"
        if approval is None:
            if multiple:
                if self.debug:
                    workflows = [str((x.get("payload") or {}).get("workflow_id") or x.get("task_id") or "unknown") for x in self._approval_candidates()]
                    self.narrator.note(f"APPROVAL: resolution=ambiguous pending_workflows={workflows} result=NEEDS_TASK_ID")
                return self._refused(UserTask(raw=raw, text=raw, source=source, status="dropped"), self._multiple_approval_prompt())
            return self.submit(raw, source=source)
        if approval.expired:
            with self._background_lock:
                self.pending_approval = None
            return self._approval_turn(raw, source, "That approval expired. Nothing was sent.", approval)
        decision = parse_approval_response(raw)
        if decision == "AMBIGUOUS" and multiple and resolution == "human_selection":
            decision = "APPROVE"
        decision = "APPROVE" if decision == "APPROVE" else "REJECT" if decision == "REJECT" else "AMBIGUOUS"
        if decision == "AMBIGUOUS":
            if self.debug:
                self.narrator.note(f"APPROVAL: workflow={approval.workflow_id or 'none'} step={approval.step_id or 'none'} approval={approval.approval_request_id or 'unknown'} response=AMBIGUOUS result=UNCHANGED")
            return self._approval_turn(raw, source, "Please answer yes or no. I will keep the pending action unchanged.", approval)

        with self._background_lock:
            self.pending_approval = None
        self._approval_context_task_id = None
        action = Action(kind=str(approval.action["kind"]), params=dict(approval.action.get("params", {})))
        task_id = approval.task_id or f"approved-messaging-{uuid.uuid4().hex[:12]}"
        runtime_task_id = approval.runtime_task_id
        if self.debug:
            self.narrator.note(f"APPROVAL: workflow={approval.workflow_id or 'none'} step={approval.step_id or 'none'} resolution={resolution or 'explicit_task_id'} response={'YES' if decision == 'APPROVE' else 'NO'} result={'APPROVED' if decision == 'APPROVE' else 'REJECTED'} resumed_workflow={approval.workflow_id or 'none'}")
        resource_key = ("browser" if action.kind.startswith("whatsapp_") or action.kind.startswith("gmail_") else None)
        if decision == "REJECT":
            if approval.workflow_id:
                try:
                    self._runtime.resolve_approval(approval.workflow_id, False, approval_request_id=approval.approval_request_id)
                except (KeyError, ValueError):
                    self._runtime_transition(approval.workflow_id, "CANCELLED", event_type="APPROVAL_DENIED", approval_state="DENIED")
                return self._approval_turn(raw, source, "Cancelled. Nothing was sent.", approval)
            if runtime_task_id:
                try:
                    self._runtime.resolve_approval(runtime_task_id, False, approval_request_id=approval.approval_request_id)
                except (KeyError, ValueError):
                    self._runtime_transition(runtime_task_id, "CANCELLED", event_type="APPROVAL_DENIED", approval_state="DENIED")
            if approval.background_task_id:
                with self._background_lock:
                    record = self._background.get(approval.background_task_id)
                    if record is not None:
                        record.state = "CANCELLED"
                        record.finished_at = time.time()
            return self._approval_turn(raw, source, "Cancelled. Nothing was sent.", approval)

        if approval.workflow_id:
            if decision == "APPROVE":
                # The workflow graph owns consumption of the durable approval and
                # resumes its existing checkpoint; no new task is created here.
                return self._resume_workflow_graph(approval.workflow_id, True, source=source)
            self._runtime.resolve_approval(approval.workflow_id, False, approval_request_id=approval.approval_request_id)
            return self._resume_workflow_graph(approval.workflow_id, False, source=source)
        else:
            from .skills.messaging.task import ApprovedMessagingTask
            from .skills.messaging import BrowserMessagingBackend
            approved_task = ApprovedMessagingTask(
                action, BrowserMessagingBackend(self._browser_for_task(task_id, resource_key=resource_key)), task_id=task_id,
            )
            prepared = Prepared(
                task=UserTask(raw=raw, text=approval.request, source=source, task_id=task_id, status="accepted"),
                goal=approval.goal or approved_task.goal, kind="approved messaging action", workspace=self.workspace,
                approved_action=approval.action, browser_resource_key=resource_key, runtime_task_id=runtime_task_id,
                task_obj=approved_task,
            )
        if runtime_task_id:
            try:
                self._runtime.resolve_approval(runtime_task_id, True, approval_request_id=approval.approval_request_id)
            except (KeyError, ValueError):
                self._runtime_transition(runtime_task_id, "RUNNING", event_type="APPROVAL_GRANTED", approval_state="GRANTED", execution_state="RUNNING")
        if background and approval.background_task_id:
            bg_id = approval.background_task_id
            with self._background_lock:
                record = self._background.get(bg_id)
                if record is not None:
                    record.state = "QUEUED"; record.result = None; record.failure = ""
            future = self._background_executor().submit(self._resume_approved_background, bg_id, prepared)
            with self._background_lock:
                record = self._background.get(bg_id)
                if record is not None:
                    record.future = future
            return self._approval_turn(raw, source, "Approved. I’m continuing the original task.", approval)
        turn = self._run(prepared)
        return turn

    def _resume_approved_background(self, background_task_id: str, prepared: Prepared) -> None:
        """Resume the exact approved task on the existing serialized worker lane."""
        with self._background_lock:
            record = self._background.get(background_task_id)
            if record is None or record.state == "CANCELLED":
                return
            record.state = "RUNNING"
            record.started_at = time.time()
        self._thread_state.background = True
        self._thread_state.background_task_id = background_task_id
        try:
            turn = self._run(prepared)
            result = turn.result
            if prepared.kind == "workflow step" and result is not None and result.ok:
                runtime_task = self._runtime.get_task(prepared.runtime_task_id or background_task_id)
                workflow_data = runtime_task.metadata.get("workflow") if runtime_task else None
                if isinstance(workflow_data, dict):
                    workflow_turn = self._resume_workflow(Workflow.from_json(workflow_data), source="text", background=True)
                    if workflow_turn.result is not None:
                        result = workflow_turn.result
                    turn = workflow_turn
            with self._background_lock:
                record = self._background.get(background_task_id)
                if record is None:
                    return
                record.kind = "approved messaging action"
                record.result = result
                record.finished_at = time.time()
                runtime = self._runtime.get_task(record.runtime_task_id or background_task_id)
                record.state = runtime.state if runtime is not None else (
                    "COMPLETED" if result is not None and result.ok else "FAILED"
                )
                if result is not None and result.status is TaskStatus.UNKNOWN:
                    record.failure = result.detail
            self._background_report(background_task_id, prepared.task.text, turn)
        except Exception as exc:  # noqa: BLE001
            with self._background_lock:
                record = self._background.get(background_task_id)
                if record is not None:
                    record.state = "UNKNOWN"
                    record.failure = f"{type(exc).__name__}: {exc}"
                    record.finished_at = time.time()
            self._safe_note(f"⚠ [{background_task_id}] background task failed: {type(exc).__name__}: {exc}")
        finally:
            self._thread_state.background = False
            self._thread_state.background_task_id = None
            self._thread_state.runtime_task_id = None

    def _debug_input(self, event: InputEvent, intent: IntentCategory | None, owner: InputOwner, *, task_created: bool, task_id: str | None = None) -> None:
        if not self.debug:
            return
        # Keep the transcript out of logs; event id/source and routing ownership
        # are sufficient to diagnose input races without retaining private speech.
        self.narrator.note(f"INPUT: id={event.event_id} source={event.source}")
        state = ("WAITING_FOR_APPROVAL" if self.pending_approval is not None else
                 "WAITING_FOR_HUMAN" if self.pending is not None else "NORMAL")
        self.narrator.note(f"STATE: {state}")
        self.narrator.note(f"ROUTE: {owner.value}")
        self.narrator.note(f"OWNER: {owner.value}")
        self.narrator.note(f"TASK: created={task_id if task_created else 'none'}")
        if owner is InputOwner.CONVERSATION:
            self.narrator.note("CONVERSATION: task_created=false planner_task=false")

    def _conversation_background_run(self, event: InputEvent, route_s: float = 0.0) -> None:
        try:
            self.submit(event.text, source=event.source)
            if self.debug:
                metrics = getattr(self._conversation_engine(), "last_metrics", {}) or {}
                self.narrator.note(
                    "CONVERSATION_LATENCY "
                    f"route={route_s:.3f}s "
                    f"context_build={float(metrics.get('context_build_s', 0.0)):.3f}s "
                    f"model_request={float(metrics.get('model_request_s', 0.0)):.3f}s "
                    f"ttft={float(metrics.get('ttft_s', 0.0)):.3f}s "
                    f"generation={float(metrics.get('generation_s', 0.0)):.3f}s "
                    "response_processing=0.000s "
                    "tts=0.000s "
                    f"total={float(metrics.get('total_s', 0.0)) + route_s:.3f}s"
                )
        except ConversationTransportError as exc:
            self._reset_conversation_executor()
            self._safe_note(f"⚠ [{event.event_id}] conversation failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            self._safe_note(f"⚠ [{event.event_id}] conversation failed: {type(exc).__name__}: {exc}")

    @staticmethod
    def _looks_like_owned_response(text: str) -> bool:
        lowered = _WHITESPACE.sub(" ", (text or "").strip().casefold())
        if re.search(r"\b(?:fast|task)-[a-z0-9]+(?:-[a-z0-9]+)?\b", lowered):
            return True
        if re.search(r"\btask\s+\d+\b", lowered):
            return True
        return lowered in {
            "yes", "no", "y", "n", "okay", "ok", "do it", "continue",
            "resume", "retry", "try again", "recover", "cancel", "stop",
            "go ahead", "approve", "reject", "that one", "the first one",
        }

    def _approval_response_is_explicit(self, raw: str) -> bool:
        """Return whether this utterance is actually an approval response.

        Pending approval state is intentionally *not* part of new-task routing.
        This lexical gate is the only reason approval state is consulted before
        normal task/conversation classification.  An arbitrary action such as
        ``open notepad and open chrome`` is therefore never inspected against
        the pending-approval registry.
        """
        normalized = _WHITESPACE.sub(" ", str(raw or "").strip().casefold())
        if not normalized:
            return False
        return parse_approval_response(normalized) != "AMBIGUOUS" or normalized in {
            "that one", "the first one", "the second one", "the third one",
        } or bool(re.fullmatch(r"\d+", normalized))

    def _debug_route(self, route: str, *, control: str | None = None,
                     approval: str | None = None, display_task: int | None = None,
                     resolved_workflow: str | None = None, checkpoint_resume: bool = False) -> None:
        if not self.debug:
            return
        self.narrator.note(f"ROUTE: {route}")
        if control is not None:
            self.narrator.note(f"CONTROL: {control}")
        if approval is not None:
            self.narrator.note(f"APPROVAL: resolution={approval}")
        if display_task is not None:
            self.narrator.note(f"DISPLAY_TASK={display_task}")
        if resolved_workflow is not None:
            self.narrator.note(f"RESOLVED_WORKFLOW={resolved_workflow}")
        if checkpoint_resume:
            self.narrator.note("CHECKPOINT_RESUME=true")

    def submit_background(self, raw: str, *, source: str = "text") -> str:
        """Dispatch one input without making conversation wait behind task work.

        Approval ownership is checked first.  Obvious conversation is deliberately
        handled outside the serialized task worker: the worker is a resource for
        computer mutations, not a queue for unrelated chat turns.  Computer/task
        input still uses the exact existing ``submit`` pipeline on the background
        lane.
        """
        dispatch_started = time.perf_counter()
        event = InputEvent(text=raw or "", source=source)
        # Rehydrate durable ownership at the dispatcher boundary.  This is the
        # critical restart invariant: pending approval outranks conversation and
        # new task routing even in the first turn of a fresh process.
        # Approval is a distinct owned interaction and must be rehydrated first.
        # Workflow-input rehydration is deliberately separate; calling only that
        # method here allowed a durable approval to fall through to conversation.
        # Avoid durable runtime scans for ordinary conversation. Session-local
        # pending owners remain authoritative; ambiguous approval/workflow
        # answers still trigger the existing durable rehydration path.
        if self.pending_approval is not None or self.pending_input is not None or self.pending_workflow_input is not None or self._looks_like_owned_response(event.text):
            self._rehydrate_pending_input_owner()
            self._rehydrate_approval_owner()
        controls = classify_runtime_controls(event.text)
        if controls:
            resolved = self._handle_runtime_controls(controls, raw=event.text, source=event.source, background=True)
            return resolved.task.task_id if resolved.task.task_id else event.event_id

        if self._approval_response_is_explicit(event.text):
            with self._background_lock:
                approval_pending = self.pending_approval is not None
            if approval_pending or self._has_multiple_pending_approvals():
                if self.debug:
                    self._debug_input(event, IntentCategory.CLARIFICATION_RESPONSE, InputOwner.APPROVAL, task_created=False, task_id=self.pending_approval.task_id if self.pending_approval is not None else None)
                resolved = self._resolve_approval_input(event.text, source=event.source, background=True)
                return resolved.task.task_id

        # Pending workflow input is passive here as well.  The normal submit
        # path will classify this utterance and resolve an owned input only when
        # it is not a fresh task.

        text = _WHITESPACE.sub(" ", event.text.strip()).strip(_TRAILING)
        # Runtime-status questions are answered from the authoritative registry
        # before normal conversation/task classification. This keeps /chat usable
        # for runtime inspection even when the optional benchmark task catalog is
        # unavailable in a checkout.
        runtime_reply = self._runtime_query_reply(text) if self._is_runtime_query(text) else None
        if runtime_reply is not None:
            turn = Turn(
                task=UserTask(raw=event.text, text=text, source=event.source, status="conversation"),
                reply=runtime_reply,
                result=None,
            )
            self._emit_reply(runtime_reply)
            self.history.append(turn)
            self._remember_turn(turn)
            return event.event_id

        # This is the authoritative lightweight ownership decision for background
        # input. It runs before a BackgroundTask record exists, so conversation can
        # never masquerade as a task merely because the CLI is non-blocking.
        intent = classify_intent(text, recent_context=self.recent_context)
        if intent is IntentCategory.CONVERSATION:
            # Conversation has its own lightweight response lane. It therefore
            # cannot sit behind a long-running browser/planner task, while still
            # reusing the existing Session._converse response mechanism.
            route_s = time.perf_counter() - dispatch_started
            self._conversation_background_executor().submit(
                self._conversation_background_run, event, route_s
            )
            return event.event_id

        base_id = f"fast-{uuid.uuid4().hex[:4]}"
        with self._background_lock:
            task_id = base_id
            suffix = 2
            while task_id in self._background and self._background[task_id].state in {"QUEUED", "RUNNING", "WAITING_FOR_USER"}:
                task_id = f"{base_id}-{suffix}"
                suffix += 1
            runtime_task_id = self._runtime_create(text or raw or "", task_id, runtime_task_id=task_id, metadata={"background": True}, task_type="background")
            record = BackgroundTask(task_id=task_id, original_goal=text or raw or "", kind="background", runtime_task_id=runtime_task_id)
            self._background[task_id] = record
            self._runtime_transition(runtime_task_id, "CREATED", execution_state="QUEUED")
        self._debug_input(event, intent, InputOwner.TASK, task_created=True, task_id=task_id)
        executor = self._background_executor()
        future = executor.submit(self._background_run, task_id, raw, source)
        record.future = future
        return task_id

    def _background_run(self, task_id: str, raw: str, source: str) -> None:
        with self._background_lock:
            record = self._background.get(task_id)
            if record is None:
                return
            if record.state == "CANCELLED":
                return
            record.state = "RUNNING"
            record.started_at = time.time()
        try:
            self._thread_state.background = True
            self._thread_state.background_task_id = task_id
            record = self._background.get(task_id)
            self._thread_state.runtime_task_id = record.runtime_task_id if record is not None else None
            turn = self.submit(raw, source=source)
            result = turn.result
            if turn.task.intent is IntentCategory.CONVERSATION and turn.reply:
                self._emit_reply(turn.reply, goal=turn.task.text, allow_background=True)
            with self._background_lock:
                record = self._background.get(task_id)
                if record is None:
                    return
                record.kind = turn.task.intent.value if turn.task.intent is not None else record.kind
                record.result = result
                record.finished_at = time.time()
                runtime = self._runtime.get_task(record.runtime_task_id or task_id)
                # Some embedders/tests replace ``submit`` with a stub. If that
                # bypasses _run entirely, reconcile the stub result into the
                # authoritative registry first; the background record never
                # invents a competing final state.
                if runtime is not None and runtime.state == "CREATED":
                    try:
                        self._runtime_transition(record.runtime_task_id or task_id, "RUNNING", current_phase="execution")
                        if result is not None and result.needs_input:
                            self._runtime_transition(record.runtime_task_id or task_id, "WAITING_FOR_USER")
                        elif result is not None and result.status is TaskStatus.CANCELLED:
                            self._runtime_transition(record.runtime_task_id or task_id, "CANCELLED")
                        elif result is not None and result.ok:
                            self._runtime_transition(record.runtime_task_id or task_id, "COMPLETED", verification_state=result.verified, result_summary=result.detail)
                        elif result is not None:
                            self._runtime_transition(record.runtime_task_id or task_id, "FAILED", failure_state=result.detail, failure_category="UNKNOWN" if result.status is TaskStatus.UNKNOWN else "EXECUTION", verification_state=result.verified)
                    except Exception:
                        pass
                    runtime = self._runtime.get_task(record.runtime_task_id or task_id)
                # BackgroundTask is presentation/execution bookkeeping only.
                # Never let it invent a state that disagrees with RuntimeManager.
                record.state = runtime.state if runtime is not None else (
                    "COMPLETED" if result is None or result.ok else "FAILED"
                )
                if result is not None and result.status is TaskStatus.UNKNOWN:
                    record.failure = result.detail
            self._background_report(task_id, raw, turn)
        except Exception as exc:  # noqa: BLE001
            runtime_id = None
            with self._background_lock:
                record = self._background.get(task_id)
                if record is not None:
                    runtime_id = record.runtime_task_id
                    record.failure = f"{type(exc).__name__}: {exc}"
                    record.finished_at = time.time()
            # A background exception is a runtime failure, not an alternate
            # UNKNOWN state owned by the presentation record.
            try:
                runtime = self._runtime.get_task(runtime_id or task_id)
                if runtime is not None and runtime.state not in {"COMPLETED", "FAILED", "CANCELLED"}:
                    self._runtime_transition(runtime.task_id, "FAILED", failure_state=f"{type(exc).__name__}: {exc}", failure_category="BACKGROUND", current_phase="failure")
                    runtime = self._runtime.get_task(runtime.task_id)
                with self._background_lock:
                    record = self._background.get(task_id)
                    if record is not None:
                        record.state = runtime.state if runtime is not None else "FAILED"
            except Exception:
                with self._background_lock:
                    record = self._background.get(task_id)
                    if record is not None:
                        record.state = "FAILED"
            # A background exception must never kill the CLI. Keep the exception
            # attached to the task record and surface only a concise diagnostic.
            self._safe_note(f"⚠ [{task_id}] background task failed: {type(exc).__name__}: {exc}")
        finally:
            self._thread_state.background = False
            self._thread_state.background_task_id = None
            self._thread_state.runtime_task_id = None

    def _background_report(self, task_id: str, raw: str, turn: Turn) -> None:
        if turn.result is None:
            return
        if turn.result.ok:
            self._safe_note(f"✓ [{task_id}] {raw} — verified")
        elif turn.result.needs_input:
            self._safe_note(f"? [{task_id}] {raw} — waiting for user input")
        elif turn.result.status is TaskStatus.UNKNOWN:
            self._safe_note(f"⚠ [{task_id}] {raw} — verification unknown")
        else:
            self._safe_note(f"✗ [{task_id}] {raw} — {turn.result.status.value.lower()}")

    def _safe_note(self, message: str) -> None:
        with self._background_lock:
            try:
                self.narrator.note(message)
            except Exception:
                pass

    def _emit_reply(
        self,
        message: Any,
        *,
        goal: str = "",
        state: str = "",
        allow_internal: bool = False,
        allow_background: bool = False,
    ) -> None:
        """Send only user-facing presentation text to the narrator/TTS layer.

        Runtime/debug output deliberately continues to use ``narrator.note``.
        This boundary is the final defence against accidental metadata leakage.
        """
        if getattr(self._thread_state, "background", False) and not allow_background:
            return
        spoken = sanitize_tts_text(
            message, goal=goal, state=state, allow_internal=allow_internal
        )
        self.narrator.reply(spoken)

    def background_tasks(self) -> list[BackgroundTask]:
        # BackgroundTask is a presentation view. Refresh lifecycle fields from
        # RuntimeManager on read so it can never become a second source of truth.
        with self._background_lock:
            for record in self._background.values():
                runtime = self._runtime.get_task(record.runtime_task_id or record.task_id)
                if runtime is None:
                    continue
                record.state = runtime.state
                record.started_at = runtime.started_at
                record.finished_at = runtime.completed_at
            return list(self._background.values())

    def cancel_background(self, task_id: str) -> bool:
        with self._background_lock:
            record = self._background.get(task_id)
            if record is None:
                return False
            future = record.future
            if record.state != "QUEUED" or future is None:
                return False
            if not future.cancel():
                return False
            record.state = "CANCELLED"
            record.finished_at = time.time()
            self._runtime_transition(record.runtime_task_id, "CANCELLED", event_type="TASK_CANCELLED", cancellation_state="CANCELLED")
            return True

    # -- input -------------------------------------------------------------
    def listen(self, *, max_seconds: float = 30.0,
               on_status: Callable[[str], None] | None = None) -> Capture:
        """Record one utterance and transcribe it. Returns text; runs nothing.

        This is the entire speech-input responsibility: audio in, words out. The
        words then go through :meth:`submit` exactly as a typed line would, which
        is why there is no voice-specific execution path to keep in step.

        A capture that failed, or a transcript that is empty or errored, comes
        back with ``ok=False`` and a reason. It must not be submitted -- an empty
        transcript handed to the agent as a task would be a fabricated request.
        """
        from .speech import record_utterance, transcribe

        wait_idle = getattr(self.narrator.speaker, "wait_until_idle", None)
        if callable(wait_idle):
            wait_idle(timeout_s=max(1.0, max_seconds))
        recording = record_utterance(max_seconds=max_seconds, on_status=on_status)

        if not recording.ok:
            return Capture(
                reason=recording.stopped_by or "error",
                detail=recording.error or "",
                seconds=recording.duration_seconds,
            )

        transcript = transcribe(recording.wav)

        if not transcript.ok:
            # A capture that heard nothing but the noise floor is reported as
            # silence rather than as a recogniser problem: the distinction decides
            # whether the user should check their microphone or their API key.
            silent = (transcript.reason == "empty_transcript"
                      and recording.peak_level < 0.01)
            return Capture(
                reason="silent" if silent else (transcript.reason or "error"),
                detail=transcript.error or "",
                seconds=recording.duration_seconds,
                latency_seconds=transcript.latency_seconds,
            )

        return Capture(
            text=transcript.text,
            ok=True,
            seconds=recording.duration_seconds,
            latency_seconds=transcript.latency_seconds,
            confidence=transcript.confidence,
        )

    # -- execution ---------------------------------------------------------
    def submit(self, raw: str, *, source: str = "text") -> Turn:
        """Normalize one request and run it. The only entry point to the agent.

        Typed lines and voice transcripts arrive here identically; ``source`` is
        recorded in the turn and changes nothing about execution. That is the
        single-pipeline guarantee, and it is a property of there being one method
        rather than of two methods being kept in agreement.

        An answer to a pending question takes the same road: it becomes an
        ordinary accepted :class:`UserTask` and is handed to :meth:`_run` like
        any other. Nothing about answering a question executes anything by
        itself.

        Every route that executes ends in ``self._run(Prepared(...))``. This
        method never calls ``run_agent_task`` itself, which is what keeps
        "one execution path" a fact about the code rather than a convention.
        """
        # Take and clear, on the first line and unconditionally. A question
        # therefore cannot survive the turn that follows it, whatever that turn
        # turns out to be -- which is what keeps this from becoming context the
        # agent accumulates.
        pending, self.pending = self.pending, None
        approval = None
        workflow_input_owner = None

        # Explicit slash controls and deterministic natural-language controls
        # always win.  They are intent, not state, so pending approvals are not
        # consulted to decide what the user meant.
        explicit_control = classify_runtime_control(raw) if str(raw or "").lstrip().startswith("/") else None
        if explicit_control is not None:
            return self._handle_runtime_control(explicit_control, raw=raw, source=source, background=False)

        controls = classify_runtime_controls(raw)
        if controls:
            return self._handle_runtime_controls(controls, raw=raw, source=source, background=False)

        # A pending multi-resume selection is also an explicit interaction.  It
        # is consulted only for selection-like input; a fresh task still wins.
        if self._runtime_resume_choices and self._approval_response_is_explicit(raw):
            return self._resolve_runtime_resume_selection(raw, source=source)

        # A bare yes/no (or another explicit approval phrase) is the only case
        # where pending approval state participates in routing.  A new task is
        # otherwise classified first, so old approvals remain passive.
        approval_response = self._approval_response_is_explicit(raw)
        if approval_response:
            approval = self.pending_approval or self._rehydrate_approval_owner()
            if approval is not None and approval.expired:
                self.pending_approval = None
                approval = None
            if approval is not None or self._has_multiple_pending_approvals():
                if self.debug:
                    self._debug_input(InputEvent(text=raw or "", source=source), IntentCategory.CLARIFICATION_RESPONSE, InputOwner.APPROVAL, task_created=False, task_id=approval.task_id if approval is not None else None)
                return self._resolve_approval_input(raw, source=source, background=False)

        if pending is not None and pending.expired:
            self.narrator.note(
                f"  (dropping an unanswered question after "
                f"{pending.age_s:.0f}s)"
            )
            pending = None

        task = normalize(raw, source=source)
        self._extract_user_memory(task.text)
        persistent_memories = self._persistent_memory_records(task.text)

        # A fresh request always beats an unanswered question, in both
        # directions: a new task is never hijacked by a stale choice, and a
        # choice is never read out of a line that was plainly a new request.
        # ``parse_request`` is lexical and touches no memory, so this guard costs
        # nothing. It must be the *same* parser the resolution step below uses,
        # or a sentence naming a project would be read as an answer to a pending
        # question about a file.
        parsed = api.parse_request(task.text)
        previous_executed = bool(self.history) and self.history[-1].executed
        intent = classify_intent(
            task.text,
            accepted=task.accepted,
            structured=parsed is not None,
            pending=pending is not None,
            previous_executed=previous_executed,
            recent_context=self.recent_context,
        )
        task = replace(task, intent=intent)
        if self.debug:
            owner = InputOwner.CONVERSATION if intent is IntentCategory.CONVERSATION else InputOwner.TASK
            self._debug_input(InputEvent(text=raw or "", source=source), intent, owner, task_created=False)
        fresh = intent in {
            IntentCategory.ACTION,
            IntentCategory.FOLLOW_UP_ACTION,
            IntentCategory.CORRECTION,
            IntentCategory.LOCAL_COMMAND,
        }

        if fresh:
            self._debug_route("TASK", approval="not_considered")

        if pending is not None and intent in {
            IntentCategory.CLARIFICATION_RESPONSE,
            IntentCategory.CANCEL,
        }:
            if intent is IntentCategory.CANCEL:
                return self._refused(
                    replace(task, status="dropped"),
                    phrase_dropped(),
                )
            answered = self._answer(pending, task, source=source)

            if isinstance(answered, Turn):
                return answered

            task = answered

        if intent is IntentCategory.LOCAL_COMMAND:
            return self._refused(
                task,
                "That local command is handled by the chat terminal and was not sent to the planner.",
            )
        if intent is IntentCategory.CANCEL and pending is None:
            return self._refused(task, "There is no pending action to cancel.")

        if task.status == "empty":
            return self._refused(task, phrase_empty())

        if task.status == "unregistered":
            # Contextual follow-ups and corrections already have a verified,
            # policy-rechecked reference.  Let GeneralTask interpret them before
            # the location index can rewrite or reject the target.
            if intent in {
                IntentCategory.FOLLOW_UP_ACTION,
                IntentCategory.CORRECTION,
            }:
                
                runtime_id = getattr(self._thread_state, "runtime_task_id", None)
                return self._general_action(task, runtime_task_id=runtime_id, persistent_memories=persistent_memories) if runtime_id else self._general_action(task, persistent_memories=persistent_memories)

            # ``resolve_request`` covers every request shape the assistant knows
            # -- a named file, a named project folder, and a project to create --
            # so "open my hermes project in vs code", "open main1.mp4" and "set up
            # a python project called test_project" travel the same road to the
            # same ``run_agent_task``. The order the shapes are tried in is
            # deliberate; see ``api.resolve_request``.
            if re.search(r"\bthen\b", task.text, flags=re.IGNORECASE):
                workflow_id = getattr(self._thread_state, "runtime_task_id", None) or self._workflow_id_from_goal()
                return self._start_workflow(task.text, source=source, workflow_id=workflow_id)

            # P2.5: never let a single-intent fast route consume the first
            # actionable clause of a compound request. Preserve the entire
            # utterance for workflow decomposition.
            if self._looks_like_compound_action(task.text):
                workflow_id = getattr(self._thread_state, "runtime_task_id", None) or self._workflow_id_from_goal(prefix="workflow")
                return self._start_workflow(task.text, source=source, workflow_id=workflow_id)

            fast = classify_fast(task.text)
            if fast.route is not None:
                fast_task_id = getattr(self._thread_state, "runtime_task_id", None) or f"fast-{uuid.uuid4().hex[:12]}"
                if fast.route.action_kind == "whatsapp_send_message":
                    from .fast_interaction import FastMessagingTask
                    action = Action(
                        kind="whatsapp_send_message",
                        params=dict(fast.route.params),
                        consequential=True,
                        rationale=fast.route.reason,
                    )
                    fast_task = FastMessagingTask(task.text, action)
                    fast_task.task_id = fast_task_id
                    resource_key = "browser"
                    browser = self._browser_for_task(fast_task_id, resource_key=resource_key)
                    fast_task.browser = browser
                    self.narrator.note(f"  fast route: whatsapp_send_message ({fast.latency_seconds * 1000:.2f}ms classification)") if self.debug else None
                    runtime_task_id = self._runtime_create(fast_task.goal, fast_task.task_id, runtime_task_id=getattr(task, "runtime_task_id", None), task_type="messaging")
                    return self._run(Prepared(
                        task=replace(task, task_id=fast_task.task_id, status="accepted"),
                        goal=fast_task.goal,
                        kind="fast messaging",
                        task_obj=fast_task,
                        workspace=self.workspace,
                        fast_route=fast.route,
                        fast_latency_seconds=fast.latency_seconds,
                        browser_resource_key=resource_key,
                        runtime_task_id=runtime_task_id,
                        persistent_memories=persistent_memories,
                    ))
                resource_key = "browser" if fast.route.action_kind.startswith("browser_") else None
                browser = self._browser_for_task(fast_task_id, resource_key=resource_key)
                fast_task = FastInteractionTask(task.text, browser, fast.route)
                fast_task.task_id = fast_task_id
                # Do not resolve the semantic target here. The runner owns the
                # freshness gate and resolves only after its final precondition
                # observation, immediately before execution.
                self.narrator.note(f"  fast route: {fast.route.action_kind} ({fast.latency_seconds * 1000:.2f}ms classification)") if self.debug else None
                runtime_task_id = self._runtime_create(fast_task.goal, fast_task.task_id, runtime_task_id=getattr(task, "runtime_task_id", None))
                return self._run(Prepared(
                    task=replace(task, task_id=fast_task.task_id, status="accepted"),
                    goal=fast_task.goal,
                    kind="fast interaction",
                    task_obj=fast_task,
                    workspace=self.workspace,
                    fast_route=fast.route,
                    fast_latency_seconds=fast.latency_seconds,
                    browser_resource_key=resource_key,
                    runtime_task_id=runtime_task_id,
                ))

            resolved = api.resolve_request(
                task.text,
                use_memory=self.use_memory,
            )

            if resolved is None:
                # Neither a registered task id nor one of the structured request
                # shapes matched. That no longer means refusal by itself: Route 2
                # (general computer action) and Route 3 (pure conversation) both
                # live here, and which applies is decided by one lexical question.
                if intent is IntentCategory.ACTION:
                    runtime_id = getattr(self._thread_state, "runtime_task_id", None)
                    return self._general_action(task, runtime_task_id=runtime_id, persistent_memories=persistent_memories) if runtime_id else self._general_action(task, persistent_memories=persistent_memories)

                return self._converse(task, persistent_memories=persistent_memories)

            if resolved.ambiguous:
                return self._ask(task, resolved)

            if not resolved.runnable:
                return self._refused(
                    task,
                    # The verb travels with the refusal: a project that could not
                    # be created was never a file that could not be found, and one
                    # sentence shape phrased in the wrong verb is how a report
                    # starts describing a search that never happened.
                    phrase_no_location(
                        resolved.query, resolved.detail, action=resolved.action,
                    ),
                )

            task = replace(
                task,
                task_id=resolved.task_id,
                params=dict(resolved.params),
                status="accepted", runtime_task_id=pending.runtime_task_id,
            )

        runtime_task_id = self._runtime_create(api.task_goal(task.task_id, **task.params), task.task_id, runtime_task_id=getattr(task, "runtime_task_id", None))
        return self._run(Prepared(task=task, goal=api.task_goal(task.task_id, **task.params), runtime_task_id=runtime_task_id, persistent_memories=persistent_memories))

    def submit_capture(self, capture: Capture, *, background: bool = False) -> Turn | str:
        """Route one transcript exactly once, preserving approval ownership.

        Normal voice input can be queued without blocking microphone/CLI input.
        An approval response is different: it belongs to the input loop and is
        resolved immediately against the existing pending task.
        """
        if not capture.ok:
            task = UserTask(raw="", text="", source="voice", status="empty")
            if capture.detail:
                self.narrator.note(f"  ({capture.detail})")
            return self._refused(task, phrase_not_heard(capture.reason))

        self._rehydrate_pending_input_owner()
        if self.pending_approval is not None:
            return self._resolve_approval_input(
                capture.text, source="voice", background=background
            )
        if background:
            return self.submit_background(capture.text, source="voice")
        return self.submit(capture.text, source="voice")

    # -- internals ---------------------------------------------------------
    def _ask(self, task: UserTask, resolved: api.Resolved) -> Turn:
        """Put the question and record it. Runs nothing: ``result`` stays ``None``.

        The turn is indistinguishable from any other refusal to a caller that only
        looks at ``executed`` -- which is right. Asking is not running, and a
        transcript where the agent asked something must not read as an attempt.
        """
        for index, choice in enumerate(resolved.choices, start=1):
            self.narrator.note(
                f"  {index}. {choice.label}: {choice.detail or choice.path}"
            )

        runtime_task_id = self._runtime_create(task.text, resolved.task_id)
        self.pending = Pending(
            query=resolved.query, task_id=resolved.task_id, choices=resolved.choices,
            asked_at=time.time(), runtime_task_id=runtime_task_id,
        )
        self._runtime_transition(runtime_task_id, "WAITING_FOR_USER", event_type="TASK_WAITING", execution_state="WAITING_FOR_USER")

        return self._refused(
            task,
            phrase_choice(resolved.query, self.pending.labels),
        )

    def _answer(
        self,
        pending: Pending,
        task: UserTask,
        *,
        source: str,
    ) -> UserTask | Turn:
        """Read this line as an answer to ``pending``.

        Returns the task to run when it selected one of the offered paths, and a
        finished :class:`Turn` when it did not: either the question withdrawn, or
        put once more. The ``Turn`` return is what stops a line that answered
        nothing from falling through into execution.
        """
        picked = choose(pending, task.text)

        if picked == CANCELLED:
            self._runtime_transition(pending.runtime_task_id, "CANCELLED", event_type="TASK_CANCELLED", cancellation_state="CANCELLED")
            return self._refused(
                replace(task, status="dropped"),
                phrase_dropped(),
            )

        if picked:
            return replace(
                task,
                task_id=pending.task_id,
                params={"path": picked},
                status="accepted",
            )

        if pending.exhausted:
            self.narrator.note(
                "  (that did not name one of the options; dropping the question)"
            )
            return self._refused(replace(task, status="dropped"), phrase_dropped())

        # Ask again, carrying the original ``asked_at`` so re-asking cannot
        # extend the window in which a stale answer is accepted.
        self.pending = pending.again()

        return self._refused(
            replace(task, status="dropped"),
            "That did not name one of them. " + phrase_choice(
                pending.query, pending.labels
            ),
        )

    def _general_action(self, task: UserTask, *, runtime_task_id: str | None = None, persistent_memories: tuple[dict[str, Any], ...] = ()) -> Turn:
        """Route 2: no registered workflow, but the request needs the machine
        touched. Builds a :class:`~agent_control.general_task.GeneralTask` and
        runs it through the exact pipeline a registered task uses -- same
        planner, same policy, same recovery, same verification. Nothing here
        is a second execution path; it is one more way to reach the existing
        one.
        """
        from .general_task import GeneralTask

        roots = api.general_readable_roots(task.text)
        general_task = GeneralTask(request=task.text, readable_roots=roots)
        if runtime_task_id:
            general_task.task_id = runtime_task_id
        if self._general_workspace is None:
            self._general_workspace = (
                self.workspace.resolve()
                if self.workspace is not None
                else api.default_workspace("general").resolve()
            )

        runtime_task_id = self._runtime_create(general_task.goal, general_task.task_id, runtime_task_id=runtime_task_id)
        return self._run(Prepared(
            task=replace(task, task_id=general_task.task_id, status="accepted"),
            goal=general_task.goal,
            kind="general action",
            task_obj=general_task,
            readable_roots=roots,
            workspace=self._general_workspace,
            runtime_task_id=runtime_task_id,
            persistent_memories=persistent_memories,
        ))

    @staticmethod
    def _browser_resource_key(prepared: Prepared) -> str | None:
        """Choose the narrowest reusable browser resource class for a task."""
        if prepared.browser_resource_key:
            # BrowserSkill is one real browser session/resource. Keep scheduler
            # resource labels (youtube/whatsapp/gmail) for dependency semantics,
            # but map them to the same live BrowserSkill backend so a later
            # browser task reuses the healthy session instead of starting a new
            # one.
            if prepared.browser_resource_key in {"browser", "youtube", "whatsapp", "gmail"}:
                return "browser"
            return prepared.browser_resource_key
        text = f"{prepared.task.text} {prepared.goal}".casefold()
        if "whatsapp" in text or "youtube" in text or "gmail" in text or "mail.google.com" in text or re.search(r"\bplay\s+.+", text):
            return "browser"
        return None

    def _run(self, prepared: Prepared) -> Turn:
        """Execute one prepared request. The only call to
        :func:`api.run_agent_task` in this module, and the only place a
        :class:`Turn` with a result is built.

        A failure here is reported, not raised. ``run_agent_task`` already turns
        an in-run crash into ``UNKNOWN`` (``runner._harness_error``) and a
        failure to assemble the run into the same, so what is left for this
        clause is a defect on the path between the two -- and the cost of
        letting that reach ``main.cmd_chat``, which catches only
        ``KeyboardInterrupt``, is the whole session rather than the turn. The
        exception is converted into the result value the rest of this method
        already knows how to report, with its type and message kept in
        ``detail``: nothing is discarded, and ``UNKNOWN`` is not success, so a
        run that died here cannot read as one that worked.
        """
        task = prepared.task
        runtime_task_id = prepared.runtime_task_id or getattr(self._thread_state, "runtime_task_id", None)
        runtime_task_id = self._runtime_create(
            prepared.goal, task.task_id, runtime_task_id=runtime_task_id,
            task_type=prepared.kind, source_input_id=task.task_id,
        )
        self._runtime.set_focus(runtime_task_id)
        self._runtime_transition(runtime_task_id, "RUNNING", event_type="TASK_RUNNING", current_phase="execution", execution_state="RUNNING")
        if self.debug:
            self.narrator.accepted(task.task_id, prepared.goal, debug=True)
            aside = f" ({prepared.kind})" if prepared.kind else ""
            self.narrator.note(f"[{task.task_id}] running{aside} ...")
        lines: list[str] = []

        # Browser resources are task-scoped. Constructing the adapter does not
        # start BrowserSkill; its session is created lazily by the first browser
        # operation for this task.
        resource_key = self._browser_resource_key(prepared)
        browser_backend = self._browser_for_task(
            task.task_id,
            resource_key=resource_key,
        )
        if resource_key:
            site = resource_key
            self._runtime.attach_resource(runtime_task_id, resource_key, site=site, state="ATTACHED")

        try:
            result = api.run_agent_task(
                task.text,
                task_id=task.task_id,
                task_params=task.params,
                task_obj=prepared.task_obj,
                readable_roots=prepared.readable_roots,
                planner=("mock" if prepared.approved_action is not None else self.planner),
                max_steps=self.max_steps,
                keep_workspace=self.keep_workspace,
                workspace=(prepared.workspace
                           if prepared.workspace is not None
                           else self.workspace),
                use_memory=self.use_memory,
                interactive=True,
                recent_context=self.recent_context.planner_state(),
                persistent_memories=prepared.persistent_memories,
                approved_action=prepared.approved_action,
                fast_route=prepared.fast_route,
                fast_latency_seconds=prepared.fast_latency_seconds,
                on_event=self._runtime_event_watcher(runtime_task_id, self._watcher(lines)),
                browser_backend=browser_backend,
                runtime_manager=self._runtime,
            )

        except Exception as exc:
            result = AgentResult(
                request=task.text,
                task_id=task.task_id,
                status=TaskStatus.UNKNOWN,
                detail=f"the run raised {type(exc).__name__}: {exc}",
            )

        if result.ok:
            self._update_recent_context(prepared, result)

        runtime_after_run = self._runtime.get_task(runtime_task_id) if runtime_task_id else None
        runtime_state_after_run = runtime_after_run.state if runtime_after_run is not None else None

        if result.needs_input:
            # Approval ownership is persisted below, after the approval envelope is
            # decoded. Do not first publish WAITING_FOR_USER: that creates a real
            # observable state in which an approval exists only in a session-local
            # field and the next dispatcher turn can race/fall through to chat.
            context = result.question.context if result.question is not None else ""
            if not context.startswith("APPROVAL_ACTION:"):
                self._runtime_transition(runtime_task_id, "WAITING_FOR_USER", execution_state="WAITING_FOR_USER")
        elif result.status is TaskStatus.CANCELLED:
            self._runtime_transition(runtime_task_id, "CANCELLED", event_type="TASK_CANCELLED", cancellation_state="CANCELLED")
        elif result.ok:
            workflow_data = result.outcome.workflow if result.outcome is not None else None
            workflow_complete = isinstance(workflow_data, dict) and workflow_data.get("state") == "COMPLETED"
            if runtime_state_after_run == "COMPLETED":
                # The runner watcher already established the terminal lifecycle
                # state from AgentState.COMPLETED. RuntimeManager remains
                # authoritative; do not replay a transition from a terminal state.
                pass
            elif workflow_complete or not isinstance(workflow_data, dict):
                self._runtime_transition(runtime_task_id, "COMPLETED", event_type="TASK_COMPLETED", execution_state="COMPLETED", verification_state=result.verified, verification_summary=result.verified, result_summary=result.detail)
            else:
                self._runtime_transition(runtime_task_id, "RUNNING", event_type="WORKFLOW_STEP_COMPLETED", execution_state="RUNNING", verification_state=result.verified, verification_summary=result.verified, result_summary=result.detail)
        elif result.status is TaskStatus.UNKNOWN:
            if runtime_state_after_run != "FAILED":
                self._runtime_transition(runtime_task_id, "FAILED", event_type="TASK_FAILED", failure_state=result.detail, failure_category="UNKNOWN", execution_state="FAILED", verification_state=result.verified, verification_summary=result.verified, result_summary=result.detail)
        else:
            if runtime_state_after_run != "FAILED":
                self._runtime_transition(runtime_task_id, "FAILED", event_type="TASK_FAILED", failure_state=result.detail, failure_category="EXECUTION", execution_state="FAILED", verification_state=result.verified, verification_summary=result.verified, result_summary=result.detail)

        if self.debug:
            for line in result.report_lines():
                self.narrator.note(line)

        if result.needs_input and result.question is not None:
            context = result.question.context or ""
            if context.startswith("APPROVAL_ACTION:"):
                try:
                    envelope = json.loads(context[len("APPROVAL_ACTION:"):])
                    if not isinstance(envelope, dict):
                        raise ValueError("approval envelope is not an object")
                    action = envelope.get("action", envelope)
                    if not isinstance(action, dict) or not isinstance(action.get("kind"), str):
                        raise ValueError("approval action is missing kind")
                    workflow_id = envelope.get("workflow_id")
                    step_id = envelope.get("step_id")
                    self.pending_approval = PendingApproval(
                        request=task.text, action=action, asked_at=time.time(),
                        task_id=task.task_id, goal=prepared.goal,
                        background_task_id=getattr(self._thread_state, "background_task_id", None),
                        runtime_task_id=runtime_task_id,
                        workflow_id=workflow_id,
                        step_id=step_id,
                        approval_request_id=envelope.get("approval_request_id"),
                    )
                    if runtime_task_id:
                        self._runtime.request_approval(runtime_task_id, {
                            "request": task.text, "action": action, "asked_at": self.pending_approval.asked_at,
                            "background_task_id": self.pending_approval.background_task_id,
                            "workflow_id": self.pending_approval.workflow_id,
                            "step_id": self.pending_approval.step_id,
                            "approval_request_id": self.pending_approval.approval_request_id,
                        })
                except Exception:
                    pass
            elif context.startswith("WORKFLOW_INPUT:"):
                try:
                    payload = json.loads(context[len("WORKFLOW_INPUT:"):])
                    if isinstance(payload, dict) and payload.get("workflow_id") and payload.get("step_id") and payload.get("field"):
                        pending = PendingInput(
                            input_id=str(payload.get("input_id") or f"input-{uuid.uuid4().hex[:10]}"),
                            workflow_id=str(payload["workflow_id"]),
                            step_id=str(payload["step_id"]),
                            task_id=str(payload.get("task_id") or runtime_task_id),
                            parameter_name=str(payload["field"]),
                            prompt=str(result.question.question if result.question is not None else "Please provide the missing value."),
                        )
                        self.pending_input = pending
                        self.pending_workflow_input = {
                            "input_id": pending.input_id, "workflow_id": pending.workflow_id,
                            "step_id": pending.step_id, "task_id": pending.task_id,
                            "field": pending.parameter_name, "prompt": pending.prompt,
                        }
                        if runtime_task_id:
                            self._runtime.set_pending_input(runtime_task_id, self.pending_workflow_input)
                except Exception as exc:
                    if self.debug:
                        self.narrator.note(f"INPUT: pending owner registration failed: {type(exc).__name__}: {exc}")

        # A run suspended on a question (NEEDS_INPUT) is not a failure and
        # must not be handed to the free-form conversational reply path: that
        # path only ever describes an action as having succeeded or failed,
        # so a pending approval could be worded back to the user as "I
        # couldn't send the message" even though nothing has failed and the
        # pending action above was preserved correctly. Ask the question
        # directly instead.
        if result.needs_input and result.question is not None:
            if self.pending_approval is not None:
                reply = approval_prompt(self.pending_approval.action, prepared.goal)
            else:
                reply = result.question.question
        elif prepared.suppress_presentation:
            # Avoid a second conversation-model call for an internal atomic
            # child. The workflow owner speaks once after all children pass.
            reply = ""
        else:
            reply = self._action_reply(task, result)
        if result is not None and not result.needs_input and self._runtime is not None and runtime_task_id:
            pass
        self._remember_task_outcome(prepared, result)
        turn = Turn(task=task, reply=reply, result=result,
                    status_lines=lines)
        if result.needs_input and result.question is not None:
            self._emit_reply(
                turn.reply,
                goal=prepared.goal,
                state="WAITING_FOR_APPROVAL" if self.pending_approval else "WAITING_FOR_USER",
            )
        elif not prepared.suppress_presentation:
            self._emit_reply(
                turn.reply,
                goal=prepared.goal,
                state=result.status.value if result is not None else "",
            )
        self.history.append(turn)
        self._remember_turn(turn)
        return turn

    def _update_recent_context(
        self,
        prepared: Prepared,
        result: AgentResult,
    ) -> None:
        """Record references from effects whose final result verified PASS."""
        from .general_task import GeneralTask

        if isinstance(prepared.task_obj, GeneralTask):
            for effect in prepared.task_obj.effects():
                target = Path(effect.target)
                if effect.kind == "create_dir":
                    self.recent_context.last_verified_directory = target
                elif effect.kind in {"write_file", "fetch_file"}:
                    self.recent_context.last_verified_file = target
                    self.recent_context.last_verified_directory = target.parent
                elif effect.kind == "open_file":
                    if target.is_dir():
                        self.recent_context.last_verified_directory = target
                    else:
                        self.recent_context.last_verified_file = target
                        self.recent_context.last_verified_directory = target.parent
                elif effect.kind == "launch_app":
                    self.recent_context.last_verified_app = effect.target
                    opened = effect.params.get("open_path")
                    if isinstance(opened, str) and opened:
                        opened_path = Path(opened).resolve()
                        if opened_path.is_dir():
                            self.recent_context.last_verified_directory = opened_path
                        else:
                            self.recent_context.last_verified_file = opened_path
                            self.recent_context.last_verified_directory = opened_path.parent
            if prepared.task_obj.effects():
                effect = prepared.task_obj.effects()[-1]
                name = Path(effect.target).name or effect.target
                summaries = {
                    "create_dir": f"Created folder {name}.",
                    "write_file": f"Created or updated file {name}.",
                    "fetch_file": f"Downloaded file {name}.",
                    "open_file": f"Opened {name}.",
                    "launch_app": f"Launched {name}.",
                }
                self.recent_context.recent_action_summary = summaries.get(
                    effect.kind, f"Completed the requested action for {name}."
                )
        else:
            path = prepared.task.params.get("path")
            if isinstance(path, str) and path:
                resolved = Path(path).resolve()
                if prepared.task.task_id in {
                    "setup_python_project", "open_project_in_vscode",
                }:
                    self.recent_context.last_verified_directory = resolved
                elif prepared.task.task_id == "open_named_file":
                    self.recent_context.last_verified_file = resolved
                    self.recent_context.last_verified_directory = resolved.parent

            if result.app:
                self.recent_context.last_verified_app = result.app

            if result.target:
                verb = "Opened" if prepared.task.task_id != "setup_python_project" else "Set up"
                self.recent_context.recent_action_summary = f"{verb} {result.target}."

        self.recent_context.last_goal = prepared.goal

        if self.debug:
            for key, value in self.recent_context.planner_state().items():
                self.narrator.note(f"  recent_context.{key} = {value}")

    def _runtime_task_view_entries(self) -> list[dict[str, Any]]:
        """Build the one authoritative, ephemeral mapping behind ``/tasks``.

        The presentation number is deliberately not persisted.  Every call
        rebuilds it from the current RuntimeManager state, so ``task 7`` always
        means row 7 of the current task view rather than an internal ID derived
        from the number.  Both CLI rendering and runtime-control resolution use
        this exact list.
        """
        state_map = {
            "CREATED": "ACTIVE", "PLANNING": "ACTIVE", "RUNNING": "ACTIVE", "VERIFYING": "ACTIVE",
            "WAITING_FOR_APPROVAL": "WAITING FOR YOU", "WAITING_FOR_HUMAN": "WAITING FOR YOU", "WAITING_FOR_USER": "WAITING FOR YOU",
            "RECOVERY_REQUIRED": "RECOVERY", "RECOVERING": "RECOVERY",
            "COMPLETED": "COMPLETED", "FAILED": "FAILED", "CANCELLED": "CANCELLED", "EXPIRED": "EXPIRED",
        }
        sections = ("ACTIVE", "WAITING FOR YOU", "RECOVERY", "COMPLETED", "FAILED", "CANCELLED", "EXPIRED")
        grouped: dict[str, list[Any]] = {section: [] for section in sections}
        for task in self._runtime.list_tasks():
            section = state_map.get(task.state)
            if section is not None:
                grouped[section].append(task)

        entries: list[dict[str, Any]] = []
        display_number = 1
        for section in sections:
            for task in grouped[section]:
                entries.append({"display_number": display_number, "task": task, "section": section})
                display_number += 1
        return entries

    def _resolve_runtime_task_ref(self, task_ref: int) -> Any | None:
        """Resolve a human ``/tasks`` number through the current view mapping."""
        entries = self._runtime_task_view_entries()
        for entry in entries:
            if entry["display_number"] == int(task_ref):
                return entry["task"]
        return None

    def _runtime_task_view_bounds(self) -> tuple[int, int]:
        """Return the current human-facing task-number range."""
        count = len(self._runtime_task_view_entries())
        return (1, count)

    def _resolve_runtime_control_command(self, command: RuntimeControlCommand) -> RuntimeControlCommand | None:
        if command.task_ref is None:
            return command
        task = self._resolve_runtime_task_ref(command.task_ref)
        if task is None:
            return None
        return replace(command, task_id=task.task_id)

    def _handle_runtime_controls(self, commands: list[RuntimeControlCommand], *, raw: str, source: str, background: bool) -> Turn:
        """Resolve a deterministic runtime-control sentence before conversation."""
        resolved: list[RuntimeControlCommand] = []
        invalid: list[str] = []
        for command in commands:
            if command.task_ref is None and command.task_id is None:
                resolved.append(command)
                continue
            item = self._resolve_runtime_control_command(command)
            if item is None or not item.task_id:
                if command.task_ref is not None:
                    _first, last = self._runtime_task_view_bounds()
                    invalid.append(
                        f"Task {command.task_ref} could not be found in the current task list (1–{last})."
                        if last else f"Task {command.task_ref} could not be found in the current task list."
                    )
                else:
                    invalid.append("One referenced task could not be found.")
                continue
            if self.debug and command.task_ref is not None:
                self.narrator.note(
                    f"RUNTIME_CONTROL: operation={command.kind.value} "
                    f"display_task={command.task_ref} resolved_task={item.task_id}"
                )
            resolved.append(item)

        if invalid and len(commands) == 1:
            return self._refused(
                UserTask(raw=raw, text=raw, source=source, status="dropped"),
                invalid[0] + " Use /tasks and tell me the correct task number.",
            )
        if not resolved:
            return self._refused(
                UserTask(raw=raw, text=raw, source=source, status="dropped"),
                " ".join(invalid) or "I couldn't safely identify the referenced tasks.",
            )

        if self.debug and len(resolved) > 1:
            if all(command.kind is RuntimeControlKind.CANCEL for command in resolved):
                self.narrator.note("ROUTE: RUNTIME_CONTROL")
                self.narrator.note("CONTROL: DELETE_TASKS")
                self.narrator.note(f"DISPLAY_TASKS={[command.task_ref for command in commands]}")
                self.narrator.note(f"RESOLVED_WORKFLOWS={[command.task_id for command in resolved]}")

        if len(resolved) == 1 and len(commands) == 1:
            command = resolved[0]
            if command.kind is RuntimeControlKind.APPROVE:
                if command.task_id:
                    approval = self._approval_owner_for_input(command.task_id)
                    if approval is None:
                        target = self._runtime.get_task(command.task_id)
                        if target is not None and target.state in {"WAITING_FOR_USER", "WAITING_FOR_HUMAN"}:
                            return self._resolve_workflow_input("yes", source=source, task_id=command.task_id)
                        return self._refused(UserTask(raw=raw, text=raw, source=source, status="dropped"), "That task is not waiting for an approval.")
                    return self._resolve_approval_input("yes", source=source, background=background)
                if not self._approval_candidates():
                    return self._refused(UserTask(raw=raw, text=raw, source=source, status="dropped"), "There are no pending approvals right now.")
                approval = self._approval_owner_for_input(raw)
                if approval is None:
                    return self._refused(UserTask(raw=raw, text=raw, source=source, status="dropped"), "I couldn't safely identify the approval. Please review the pending actions and choose one.")
                return self._resolve_approval_input("yes", source=source, background=background)
            return self._handle_runtime_control(command, raw=raw, source=source, background=background)

        # Resolve every target before executing any operation, so an ambiguous
        # reference cannot cause a partially applied multi-operation sentence.
        results: list[Turn] = []
        for command in resolved:
            if command.kind is RuntimeControlKind.APPROVE:
                if command.task_id:
                    approval = self._approval_owner_for_input(command.task_id)
                    if approval is None:
                        target = self._runtime.get_task(command.task_id)
                        if target is not None and target.state in {"WAITING_FOR_USER", "WAITING_FOR_HUMAN"}:
                            results.append(self._resolve_workflow_input("yes", source=source, task_id=command.task_id))
                            continue
                        return self._refused(UserTask(raw=raw, text=raw, source=source, status="dropped"), "One of the referenced tasks is not waiting for approval.")
                    results.append(self._resolve_approval_input("yes", source=source, background=background))
                else:
                    if not self._approval_candidates():
                        return self._refused(UserTask(raw=raw, text=raw, source=source, status="dropped"), "There are no pending approvals right now.")
                    approval = self._approval_owner_for_input(raw)
                    if approval is None:
                        return self._refused(UserTask(raw=raw, text=raw, source=source, status="dropped"), "I couldn't safely identify the approval. Please review the pending actions and choose one.")
                    results.append(self._resolve_approval_input("yes", source=source, background=background))
            else:
                results.append(self._handle_runtime_control(command, raw=raw, source=source, background=background))
        replies = [result.reply for result in results if result.reply]
        replies.extend(invalid)
        reply = "\n".join(replies)
        last = results[-1]
        return replace(last, reply=reply)

    def _runtime_control_task(self, command: RuntimeControlCommand):
        snapshot = self.runtime_snapshot()
        tasks = self._runtime.list_tasks()
        if command.task_id:
            task = self._runtime.get_task(command.task_id)
            if task is None:
                return None, "That task does not exist in the current runtime."
            return task, None
        if command.scope == "all":
            return None, None
        active = [t for t in tasks if t.state not in {"COMPLETED", "FAILED", "CANCELLED"}]
        if len(active) == 1:
            return active[0], None
        if not active:
            return None, "There are no active tasks right now."
        return None, "I found multiple active tasks. Tell me which one you mean."

    @staticmethod
    def _runtime_task_reply(task) -> str:
        if task is None:
            return "There are no matching runtime tasks to continue."
        state = str(task.state)
        if state == "WAITING_FOR_APPROVAL":
            return "The task is waiting for your approval."
        if state in {"WAITING_FOR_USER", "WAITING_FOR_HUMAN"}:
            return "The task is waiting for your input."
        if state == "RECOVERY_REQUIRED":
            return "The task needs recovery before it can continue."
        if state == "RECOVERING":
            return "The task is being recovered."
        if state == "RUNNING":
            return "The task is already running."
        if state == "COMPLETED":
            return "The task has already completed; I will not replay it."
        if state == "FAILED":
            return "The task failed; I will not replay it without an explicit retry."
        if state == "CANCELLED":
            return "The task was cancelled and will not be replayed."
        if state == "EXPIRED":
            return "The task expired and will not be resumed."
        return f"The task is {state.casefold()}."

    def _resume_runtime_task(self, task, *, source: str, background: bool) -> Turn:
        if task is None:
            return Turn(task=UserTask(raw="", text="", source=source, status="dropped"), reply="There are no matching runtime tasks to continue.")
        state = str(task.state)
        if state == "WAITING_FOR_APPROVAL":
            approval = self._approval_owner_for_input(task.task_id)
            if approval is None:
                return self._refused(UserTask(raw=task.goal, text=task.goal, source=source, task_id=task.task_id, status="accepted"), "That task is waiting for approval, but its approval owner could not be recovered safely.")
            self._approval_context_task_id = task.task_id
            try:
                payload = dict(self._runtime.get_approval(task.task_id) or {})
                payload["explicit_resume_context"] = True
                self._runtime.request_approval(task.task_id, payload)
            except Exception:
                pass
            if self.debug:
                self._debug_route("RUNTIME_CONTROL", control="RESUME_TASK", resolved_workflow=task.task_id, checkpoint_resume=True)
            reply = f"Task needs approval: {task.goal}\nApprove?"
            self._emit_reply(reply, goal=task.goal, state="WAITING_FOR_APPROVAL")
            return Turn(task=UserTask(raw=task.goal, text=task.goal, source=source, task_id=task.task_id, status="accepted"), reply=reply, result=None)
        if state in {"WAITING_FOR_USER", "WAITING_FOR_HUMAN", "RECOVERY_REQUIRED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"}:
            reply = self._runtime_task_reply(task)
            return self._refused(UserTask(raw=task.goal, text=task.goal, source=source, task_id=task.task_id, status="accepted"), reply)
        workflow_data = task.metadata.get("workflow") if isinstance(task.metadata, dict) else None
        if not isinstance(workflow_data, dict):
            return self._refused(UserTask(raw=task.goal, text=task.goal, source=source, task_id=task.task_id, status="accepted"), "That task has no resumable workflow state.")
        try:
            workflow = Workflow.from_json(workflow_data)
        except Exception:
            return self._refused(UserTask(raw=task.goal, text=task.goal, source=source, task_id=task.task_id, status="accepted"), "The task's workflow state could not be recovered safely.")
        # A workflow is executable only when every dependency before the current
        # step is verified complete and the current step itself is pending.
        idx = workflow.current_step
        if idx < 0 or idx >= len(workflow.steps):
            return self._refused(UserTask(raw=task.goal, text=task.goal, source=source, task_id=task.task_id, status="accepted"), "The task's workflow state is invalid, so I will not continue it.")
        if any(step.state != "COMPLETED" for step in workflow.steps[:idx]):
            return self._refused(UserTask(raw=task.goal, text=task.goal, source=source, task_id=task.task_id, status="accepted"), "The task has an incomplete prerequisite, so I will not skip ahead.")
        if workflow.steps[idx].state != "PENDING":
            return self._refused(UserTask(raw=task.goal, text=task.goal, source=source, task_id=task.task_id, status="accepted"), self._runtime_task_reply(task))
        # Explicit resume is a continuation of the durable workflow, not a new
        # execution of its current step.  Re-enter the existing LangGraph thread
        # by canonical workflow id so completed branches/checkpoints are retained.
        if self.debug:
            self._debug_route("RUNTIME_CONTROL", control="RESUME_TASK", resolved_workflow=workflow.workflow_id, checkpoint_resume=True)
        return self._resume_workflow_graph(workflow.workflow_id, None, source=source)

    def runtime_tasks_for_display(self) -> dict[str, list[dict[str, Any]]]:
        """Return the current human-facing task view, without internal IDs.

        Ordering is inherited from ``_runtime_task_view_entries`` so the same
        numbered presentation is authoritative for both ``/tasks`` and natural
        language runtime-control references.
        """
        groups = {"ACTIVE": [], "WAITING FOR YOU": [], "RECOVERY": [], "COMPLETED": [], "FAILED": [], "CANCELLED": [], "EXPIRED": []}
        for number, entry in enumerate(self._runtime_task_view_entries(), 1):
            task = entry["task"]
            section = entry["section"]
            item = {"display_number": number, "goal": task.goal, "state": task.state}
            metadata = task.metadata if isinstance(task.metadata, dict) else {}
            workflow = metadata.get("workflow")
            if isinstance(workflow, dict):
                steps = workflow.get("steps") if isinstance(workflow.get("steps"), list) else []
                current = workflow.get("current_step")
                if isinstance(current, int) and 0 <= current < len(steps):
                    step = steps[current] if isinstance(steps[current], dict) else {}
                    action = step.get("action") if isinstance(step.get("action"), dict) else {}
                    params = action.get("params") if isinstance(action.get("params"), dict) else {}
                    kind = str(action.get("kind") or "")
                    if kind == "whatsapp_send_message":
                        recipient = str(params.get("recipient") or params.get("to") or "")
                        item["step"] = f"Step {current + 1} of {len(steps)}: Send a WhatsApp message to {recipient}" if recipient else f"Step {current + 1} of {len(steps)}"
                    elif kind in {"browser_play_song", "youtube_play", "browser_play_video"}:
                        query = str(params.get("query") or params.get("song") or "")
                        item["step"] = f"Step {current + 1} of {len(steps)}: Play {query} on YouTube" if query else f"Step {current + 1} of {len(steps)}"
                    elif steps:
                        item["step"] = f"Step {current + 1} of {len(steps)}"
            groups[section].append(item)
        return groups

    def _resumable_runtime_tasks(self) -> list[Any]:
        candidates = []
        for task in self._runtime.list_tasks():
            if task.state not in {"CREATED", "PLANNING"}:
                continue
            workflow_data = task.metadata.get("workflow") if isinstance(task.metadata, dict) else None
            if not isinstance(workflow_data, dict):
                continue
            try:
                workflow = Workflow.from_json(workflow_data)
            except Exception:
                continue
            idx = workflow.current_step
            if 0 <= idx < len(workflow.steps) and workflow.steps[idx].state == "PENDING" and all(step.state == "COMPLETED" for step in workflow.steps[:idx]):
                candidates.append(task)
        return candidates

    def _resume_selection_prompt(self, candidates: list[Any]) -> str:
        lines = ["I found multiple tasks that can be resumed:"]
        lines.extend(f"{i}. {task.goal}" for i, task in enumerate(candidates, 1))
        lines.append("Which one should I resume?")
        return "\n".join(lines)

    def _resolve_runtime_resume_selection(self, raw: str, *, source: str) -> Turn:
        candidates = [self._runtime.get_task(task_id) for task_id in self._runtime_resume_choices]
        candidates = [task for task in candidates if task is not None]
        if not candidates:
            self._runtime_resume_choices = ()
            return self._refused(UserTask(raw=raw, text=raw, source=source, status="dropped"), "There are no resumable tasks right now.")
        choices = tuple(api.Choice(path=task.task_id, label=task.goal, detail="") for task in candidates)
        picked = choose(Pending(query="Which task should I resume?", task_id="resume-selection", choices=choices, asked_at=time.time()), raw)
        if not picked:
            return self._refused(UserTask(raw=raw, text=raw, source=source, status="dropped"), self._resume_selection_prompt(candidates))
        task = next((task for task in candidates if task.task_id == picked), None)
        self._runtime_resume_choices = ()
        return self._resume_runtime_task(task, source=source, background=False)

    def _handle_runtime_control(self, command: RuntimeControlCommand, *, raw: str, source: str, background: bool) -> Turn:
        if command.kind in {RuntimeControlKind.CONTINUE, RuntimeControlKind.RESUME} and not command.task_id:
            candidates = self._resumable_runtime_tasks()
            if len(candidates) == 1:
                return self._resume_runtime_task(candidates[0], source=source, background=background)
            if len(candidates) > 1:
                choices = tuple(api.Choice(path=task.task_id, label=task.goal, detail="") for task in candidates)
                picked = choose(Pending(query="Which task should I resume?", task_id="resume-selection", choices=choices, asked_at=time.time()), raw)
                if picked:
                    selected = next(task for task in candidates if task.task_id == picked)
                    return self._resume_runtime_task(selected, source=source, background=background)
                self._runtime_resume_choices = tuple(task.task_id for task in candidates)
                return self._refused(UserTask(raw=raw, text=raw, source=source, status="dropped"), self._resume_selection_prompt(candidates))
        if self.debug:
            owner = InputOwner.TASK
            self._debug_input(InputEvent(text=raw, source=source), None, owner, task_created=False, task_id=command.task_id)
            self.narrator.note(f"RUNTIME_CONTROL: {command.kind.value} task={command.task_id or command.scope}")
        task, error = self._runtime_control_task(command)
        if error:
            return self._refused(UserTask(raw=raw, text=raw, source=source, task_id=command.task_id or "", status="accepted"), error)
        if command.kind is RuntimeControlKind.CLEAR_HISTORY:
            removed = self._runtime.clear_history()
            active = len(self._runtime.list_active_tasks())
            if removed:
                reply = f"History cleared: {len(removed)} inactive workflows/tasks removed. Active workflows preserved: {active}."
            else:
                reply = "No inactive workflow history to clear."
            return self._refused(UserTask(raw=raw, text=raw, source=source, status="accepted"), reply)

        if command.kind is RuntimeControlKind.STATUS:
            if self.debug:
                self._debug_route("RUNTIME_CONTROL", control="LIST_PENDING_TASKS")
            reply = self._runtime_query_reply("what tasks are pending") or "There are no pending tasks right now."
            return self._refused(UserTask(raw=raw, text=raw, source=source, task_id=task.task_id if task else "", status="accepted"), reply)
        if command.kind is RuntimeControlKind.CANCEL:
            if task is None:
                return self._refused(UserTask(raw=raw, text=raw, source=source, status="accepted"), "I need a task ID because multiple active tasks need attention.")
            if task.state in {"COMPLETED", "FAILED", "CANCELLED"}:
                return self._refused(UserTask(raw=raw, text=raw, source=source, task_id=task.task_id, status="accepted"), self._runtime_task_reply(task))
            self._runtime.cancel_task(task.task_id, reason="cancelled by user")
            if self.debug:
                self._debug_route("RUNTIME_CONTROL", control="DELETE_TASKS")
            return self._refused(UserTask(raw=raw, text=raw, source=source, task_id=task.task_id, status="accepted"), "The task was deleted.")
        if command.kind is RuntimeControlKind.RETRY:
            if task is None:
                return self._refused(UserTask(raw=raw, text=raw, source=source, status="accepted"), "I found multiple failed tasks. Tell me which one you mean.")
            if task.state != "FAILED":
                return self._refused(UserTask(raw=raw, text=raw, source=source, task_id=task.task_id, status="accepted"), "The task is not failed, so I will not retry it.")
            # The existing recovery pipeline owns retries; this control command
            # deliberately does not synthesize a new action or replay a terminal run.
            return self._refused(UserTask(raw=raw, text=raw, source=source, task_id=task.task_id, status="accepted"), "The task failed. Recovery must re-observe it before another execution attempt.")
        if command.kind in {RuntimeControlKind.CONTINUE, RuntimeControlKind.RESUME}:
            if command.scope == "all":
                active = self._runtime.list_active_tasks()
                resumable = []
                blocked = []
                for candidate in active:
                    if candidate.state in {"WAITING_FOR_APPROVAL", "WAITING_FOR_USER", "WAITING_FOR_HUMAN", "RECOVERY_REQUIRED", "RUNNING"}:
                        blocked.append(self._runtime_task_reply(candidate))
                        continue
                    workflow = candidate.metadata.get("workflow") if isinstance(candidate.metadata, dict) else None
                    if isinstance(workflow, dict):
                        try:
                            wf = Workflow.from_json(workflow)
                            idx = wf.current_step
                            if 0 <= idx < len(wf.steps) and wf.steps[idx].state == "PENDING" and all(s.state == "COMPLETED" for s in wf.steps[:idx]):
                                resumable.append(candidate)
                        except Exception:
                            blocked.append("A task has invalid workflow state and was not continued.")
                if not resumable:
                    return self._refused(UserTask(raw=raw, text=raw, source=source, status="accepted"), "No task is ready to continue. " + " ".join(blocked[:2]))
                if len(resumable) > 1:
                    return self._refused(UserTask(raw=raw, text=raw, source=source, status="accepted"), "Multiple tasks are ready to continue. Please provide a task ID.")
                return self._resume_runtime_task(resumable[0], source=source, background=background)
            return self._resume_runtime_task(task, source=source, background=background)
        return self._refused(UserTask(raw=raw, text=raw, source=source, status="accepted"), "I could not resolve that runtime command safely.")

    @staticmethod
    def _is_runtime_query(text: str) -> bool:
        lowered = _WHITESPACE.sub(" ", (text or "").strip().casefold())
        if is_explicit_task_id_request(text):
            return True
        return any(phrase in lowered for phrase in (
            "what are you doing", "what task is running", "what tasks are running",
            "what is running", "what are your pending tasks",
            "what are the pending tasks", "what tasks are pending",
            "which tasks are pending", "what is pending", "show my tasks",
            "what tasks do i have", "show my pending tasks", "show pending tasks",
            "list my pending tasks", "list pending tasks",
            "what went wrong", "what task went wrong", "what failed",
        ))

    def _runtime_query_reply(self, text: str) -> str | None:
        """Answer explicit runtime-status questions without touching runtime state otherwise."""
        if not self._is_runtime_query(text):
            return None
        lowered = _WHITESPACE.sub(" ", (text or "").strip().casefold())
        snapshot = self.runtime_snapshot()
        active = snapshot["active_tasks"]
        failed = snapshot["failed_tasks"]

        if is_explicit_task_id_request(text):
            return "I keep internal task identifiers private. Use /tasks to see your tasks."

        status_query = (
            "what are you doing" in lowered
            or "what task is running" in lowered
            or "what tasks are running" in lowered
            or "what is running" in lowered
            or "what are your pending tasks" in lowered
            or "what are the pending tasks" in lowered
            or "what tasks are pending" in lowered
            or "which tasks are pending" in lowered
            or "what is pending" in lowered
            or "show my tasks" in lowered
            or "what tasks do i have" in lowered
            or "show my pending tasks" in lowered
            or "show pending tasks" in lowered
            or "list my pending tasks" in lowered
            or "list pending tasks" in lowered
        )
        failure_query = (
            "what went wrong" in lowered
            or "what task went wrong" in lowered
            or "what failed" in lowered
        )
        if status_query:
            entries = [
                entry for entry in self._runtime_task_view_entries()
                if entry["section"] in {"ACTIVE", "WAITING FOR YOU", "RECOVERY"}
            ]
            if not entries:
                return "There are no pending tasks right now."
            lines = ["I currently have these pending tasks:"]
            for entry in entries:
                number = entry["display_number"]
                task = entry["task"]
                state = str(task.state or "")
                if state == "WAITING_FOR_APPROVAL":
                    detail = "waiting for your approval"
                elif state in {"WAITING_FOR_USER", "WAITING_FOR_HUMAN"}:
                    detail = "waiting for your input"
                elif state == "RECOVERY_REQUIRED":
                    detail = "interrupted and needs recovery"
                else:
                    detail = _friendly_runtime_state(state)
                lines.append(f"{number} - {task.goal} — {detail}")
            return "\n".join(lines)
        if failure_query:
            if not failed:
                return "I don't have any failed tasks in the current runtime."
            item = failed[-1]
            goal = item.get("goal") or "the requested task"
            return f"The {goal} could not be completed."
        return None


    def _converse(self, task: UserTask, *, persistent_memories: tuple[dict[str, Any], ...] = ()) -> Turn:
        """Route 3: pure conversation. ``result`` stays ``None`` unconditionally
        here, so :attr:`Turn.executed` and :attr:`Turn.ok` read false no matter
        what the reply says -- this path must never be mistaken for one that
        ran something.

        The task is restamped ``conversation`` before the turn is recorded. It
        arrived here as ``unregistered``, which was true of the registry and is
        no longer the whole story: this request was answered as conversation, on
        purpose, and a transcript that still called it unregistered would read
        as a request the assistant failed to place.
        """
        direct_runtime_reply = self._runtime_query_reply(task.text)
        if direct_runtime_reply is not None:
            turn = Turn(task=replace(task, status="conversation"), reply=direct_runtime_reply, result=None)
            self._emit_reply(direct_runtime_reply)
            self.history.append(turn)
            self._remember_turn(turn)
            return turn

        try:
            # Pure conversation gets only conversational state. Runtime state and
            # persistent memory are opt-in for turns that actually need them.
            needs_runtime_context = _references_recent_context(task.text, self.recent_context)
            context = dict(self.recent_context.planner_state()) if needs_runtime_context else None
            memories = list(persistent_memories) if persistent_memories else None
            if self._conversation_needs_persistent_memory(task.text):
                legacy = self._conversation_memory().search(task.text, limit=4)
                memories = (memories or []) + legacy

            engine = self._conversation_engine()
            # Keep the known-good NVIDIA/OpenAI-compatible request path as the
            # default. The experimental SSE transport is intentionally not
            # selected automatically because it regressed gpt-oss-20b TTFT.
            # Conversation still stays on its dedicated fast lane and uses one
            # real model request.
            streamed = False
            reply = engine.reply(
                task.text, history=self.history,
                recent_context=context, memories=memories,
            )

            # Most conversational replies need no runtime lookup. Only invoke
            # the existing safety guard when the model actually makes an
            # execution-state claim.
            if re.search(r"\b(?:i(?:'m)?|we(?:'re)?|deimos|the task|it)\s+(?:am|are|is|was|were|has|have)?\s*(?:now\s+)?(?:sending|sent|started|resumed|running|completed|finished|failed|stopped|cancelled|canceled|playing)\b", reply, re.I):
                reply = guard_conversation_runtime_claim(reply, self.runtime_snapshot())
        except (LLMUnavailable, RuntimeError) as exc:
            if isinstance(exc, ConversationTransportError):
                self._reset_conversation_executor()
            return self._refused(
                task,
                f"I could not reach the conversation model ({exc}), so I have "
                "nothing to say about that and have not run anything.",
            )

        turn = Turn(task=replace(task, status="conversation"), reply=reply,
                    result=None)
        if not streamed:
            self._emit_reply(reply)
        self.history.append(turn)
        self._remember_turn(turn)
        return turn

    @staticmethod
    def _conversation_needs_persistent_memory(text: str) -> bool:
        lowered = _WHITESPACE.sub(" ", (text or "").strip().casefold())
        # Recent in-session history is the normal memory mechanism. Persistent
        # search is reserved for requests that clearly ask for remembered/past
        # information, avoiding a full JSONL scan for "hey".
        return any(marker in lowered for marker in (
            "remember", "forgot", "previous", "earlier", "before",
            "yesterday", "last time", "do you recall", "what did i",
        ))

    @staticmethod
    def _conversation_streaming_supported(engine: Any) -> bool:
        client = getattr(engine, "client", None)
        return callable(getattr(client, "chat_stream", None))

    def _stream_conversation_reply(
        self, engine: Any, task: UserTask, context: dict[str, str] | None,
        memories: list[dict[str, Any]] | None,
    ) -> str:
        chunks: list[str] = []
        first = True
        for chunk in engine.reply_stream(
            task.text, history=self.history, recent_context=context, memories=memories,
        ):
            if not chunk:
                continue
            chunks.append(chunk)
            if first:
                self.narrator.write(chunk)
                first = False
            else:
                self.narrator.write(chunk)
        if not chunks:
            raise RuntimeError("Conversation model returned an empty response.")
        reply = "".join(chunks).strip()
        if first:
            raise RuntimeError("Conversation model returned an empty response.")
        self.narrator.write("\n")
        # TTS is deliberately queued only after the complete text is visible.
        self.narrator.say(sanitize_tts_text(reply, goal=task.text))
        return reply

    def _action_reply(self, task: UserTask, result: AgentResult) -> str:
        memory = self._conversation_memory()
        memories = memory.search(task.text, limit=8)
        memories = [record.to_dict() for record in self._persistent_memory_records(task.text, limit=6)] + memories
        context = dict(self.recent_context.planner_state())
        event = {
            "request": task.text,
            "status": result.status.value,
            "success": result.ok,
            "target": result.target,
            "app": result.app,
            "detail": result.detail,
            "duration_seconds": result.duration_seconds,
        }
        try:
            return self._conversation_engine().action_reply(
                task.text, event, self.history, context, memories
            )
        except (LLMUnavailable, RuntimeError):
            return self.narrator.presentation.result(result)

    def _persistent_memory(self) -> PersistentMemory:
        if self._persistent_memory_store is None:
            self._persistent_memory_store = PersistentMemory()
        return self._persistent_memory_store

    def _persistent_memory_records(self, text: str, *, limit: int = 6) -> tuple[dict[str, Any], ...]:
        if not self.use_memory or not text.strip() or len(text.strip()) < 3:
            return ()
        try:
            records = self._persistent_memory().search(text, limit=limit)
            return tuple(record.to_dict() for record in records)
        except (OSError, sqlite3.Error):
            if self.debug:
                self.narrator.note("[memory] persistent recall unavailable")
            return ()

    def _extract_user_memory(self, text: str) -> None:
        if not self.use_memory:
            return
        try:
            for item in MemoryExtractor.extract_user(text, session_id=self._session_owner_id):
                if item.get("op") == "invalidate":
                    self._persistent_memory().invalidate_matching(str(item.get("query", "")))
                elif item.get("op") == "store":
                    self._persistent_memory().put(**{k: v for k, v in item.items() if k != "op"})
            self._persistent_memory().maintain()
        except (OSError, sqlite3.Error):
            if self.debug:
                self.narrator.note("[memory] could not persist extracted memory")

    def _remember_task_outcome(self, prepared: Prepared, result: AgentResult) -> None:
        if not self.use_memory or result.status in {TaskStatus.CANCELLED, TaskStatus.NEEDS_INPUT, TaskStatus.POLICY_BLOCKED}:
            return
        try:
            item = MemoryExtractor.outcome(
                goal=prepared.goal, task_id=result.task_id, status=result.status.value,
                verified=result.verified, duration=result.duration_seconds,
                failure=result.detail if not result.ok else None, session_id=self._session_owner_id,
            )
            if item:
                self._persistent_memory().put(**{k: v for k, v in item.items() if k != "op"})
                self._persistent_memory().maintain()
        except (OSError, sqlite3.Error):
            if self.debug:
                self.narrator.note("[memory] could not persist task episode")

    def _conversation_memory(self) -> ConversationMemory:
        if self._conversation_memory_store is None:
            self._conversation_memory_store = ConversationMemory.from_env()

        return self._conversation_memory_store

    def _remember_turn(self, turn: Turn) -> None:
        try:
            self._conversation_memory().append_turn(turn)
        except OSError:
            if self.debug:
                self.narrator.note("[memory] could not persist conversation turn")

    def _conversation_engine(self) -> ConversationEngine:
        """Build (once) and return this session's :class:`ConversationEngine`."""
        if self._conversation is None:
            self._conversation = ConversationEngine.from_env()
        return self._conversation

    def _reset_conversation_executor(self) -> None:
        """Retire the public lane after a transport fault without reordering queued turns."""
        with self._background_lock:
            executor = self._conversation_executor
            self._conversation_executor = None
            if executor is not None:
                self._conversation_executor_retired = executor

    def _refused(self, task: UserTask, reply: str) -> Turn:
        """Record a turn that ran nothing. ``result`` stays ``None``."""
        turn = Turn(task=task, reply=reply)
        self._emit_reply(reply)
        self.history.append(turn)
        return turn

    def _watcher(self, lines: list[str]) -> Callable[[dict[str, Any]], None] | None:
        """A trace callback that prints progress, or ``None`` when status is off.

        Returning ``None`` rather than a no-op function matters: the trace skips
        the callback entirely, so the default text path is byte-for-byte what it
        was before this module existed.
        """
        if not self.show_status:
            return None

        def watch(event: dict[str, Any]) -> None:
            allowed = DEBUG_STATUS_EVENTS if self.debug else STATUS_EVENTS
            if event.get("event") not in allowed:
                return
            line = (
                _debug_status_line(event)
                if self.debug
                else self.narrator.presentation.progress(event)
            )
            if line:
                lines.append(line)
                self.narrator.note(line)

        return watch
