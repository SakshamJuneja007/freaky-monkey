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
from .conversation import ConversationEngine
from .conversation_memory import ConversationMemory
from .planner.openai_compat import LLMUnavailable

import re
import json
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
from .types import Action
from .fast_interaction import classify_fast, FastInteractionTask
from .task import Task
from .skills.browser import BrowserSkillAdapter
from .runtime import RuntimeManager
from .presentation import (
    completion_message,
    is_explicit_task_id_request,
    recovery_message,
    runtime_snapshot_for_user,
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
    ttl_s: float = 180.0

    @property
    def expired(self) -> bool:
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


_LOCAL_COMMANDS = frozenset({"cls", "clear", "/help", "/tasks", "/quit"})
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
    fast_route: Any | None = None
    fast_latency_seconds: float = 0.0
    runtime_task_id: str | None = None


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


@dataclass
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
    history: list[Turn] = field(default_factory=list)
    #: The one question awaiting an answer, or ``None``. This is not
    #: conversational memory: :meth:`submit` takes and clears it on its first
    #: line, so it cannot outlive the turn immediately after the one that set it,
    #: and :func:`choose` can only ever return a path it already contains.
    pending: Pending | None = None
    pending_approval: PendingApproval | None = None
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

    def __post_init__(self) -> None:
        # Keep direct/unit Session construction lightweight and isolated. The
        # real CLI passes a durable path, which replaces the in-memory manager.
        if self.runtime_persistence_path not in (None, ":memory:"):
            self._runtime.close()
            self._runtime = RuntimeManager(persistence_path=self.runtime_persistence_path)
        self._restore_runtime_views()

    def _rehydrate_pending_input_owner(self) -> PendingApproval | None:
        """Rebuild the input owner from RuntimeManager before routing any input.

        ``pending_approval`` is only a session-local presentation/execution
        handle.  RuntimeManager owns the durable owner.  Reading it here on every
        input boundary closes the restart race where a fresh Session would route
        ``yes`` to conversation before its local view had been rebuilt.
        """
        if self.pending_approval is not None:
            return self.pending_approval
        approvals = self._runtime.snapshot().get("pending_approvals", [])
        if not approvals:
            return None

        # Prefer the focused task when it has a pending approval; otherwise use
        # the oldest durable approval.  This chooses from RuntimeManager's one
        # authoritative registry and never creates a second owner map.
        focused = self._runtime.snapshot().get("focused_task_id")
        item = next((x for x in approvals if x.get("task_id") == focused), approvals[0])
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
        self._conversation_executor = None
        if conversation_executor is not None:
            conversation_executor.shutdown(wait=True, cancel_futures=True)
        try:
            self.narrator.close(timeout_s)
        finally:
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
                self._runtime.close()
            except Exception:
                pass

    def _conversation_background_executor(self) -> ThreadPoolExecutor:
        if self._closing:
            raise RuntimeError("session is shutting down")
        if self._conversation_executor is None:
            self._conversation_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="deimos-conversation"
            )
        return self._conversation_executor

    def _background_executor(self) -> ThreadPoolExecutor:
        if self._closing:
            raise RuntimeError("session is shutting down")
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
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
        self._runtime.create_task(
            goal, task_id=task_id, task_type=task_type,
            source_input_id=source_input_id, parent_task_id=parent_task_id,
            metadata=metadata,
        )
        return task_id

    def _runtime_transition(self, runtime_task_id: str | None, status: str, *, event_type: str | None = None, **kwargs: Any) -> None:
        if not runtime_task_id:
            return
        try:
            self._runtime.transition_task(runtime_task_id, status, event_type=event_type, **kwargs)
        except (KeyError, ValueError):
            # Runtime truth must never take down the execution loop. Invalid
            # transitions are still visible in debug output.
            if self.debug:
                self.narrator.note(f"RUNTIME: ignored transition {runtime_task_id} -> {status}")

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
                mapping = {
                    "RUNNING": ("RUNNING", "TASK_RUNNING"),
                    "PLANNING": ("PLANNING", "TASK_PLANNING"),
                    "VERIFYING": ("VERIFYING", "VERIFICATION_STARTED"),
                    "WAITING_FOR_USER": ("WAITING_FOR_USER", "TASK_WAITING"),
                    "COMPLETED": ("COMPLETED", "TASK_COMPLETED"),
                    "FAILED": ("FAILED", "TASK_FAILED"),
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

    def _resolve_approval_input(self, raw: str, *, source: str, background: bool) -> Turn:
        """Consume one approval response for the exact pending task.

        The durable RuntimeManager is consulted before falling back to normal
        routing.  Therefore a restarted Session cannot turn a valid ``yes`` into
        conversation merely because its local PendingApproval object was not yet
        populated.
        """
        with self._background_lock:
            approval = self.pending_approval
        if approval is None:
            approval = self._rehydrate_pending_input_owner()
        if approval is None:
            return self.submit(raw, source=source)
        if approval.expired:
            with self._background_lock:
                self.pending_approval = None
            return self._approval_turn(raw, source, "That approval expired. Nothing was sent.", approval)

        decision = parse_approval_response(raw)
        decision = "APPROVE" if decision == "APPROVE" else "REJECT" if decision == "REJECT" else "AMBIGUOUS"
        if decision == "AMBIGUOUS":
            if self.debug:
                self.narrator.note(f"APPROVAL: pending_task={approval.task_id or 'unknown'} response=AMBIGUOUS result=UNCHANGED")
            return self._approval_turn(raw, source, "Please answer yes or no. I will keep the pending action unchanged.", approval)

        with self._background_lock:
            self.pending_approval = None
        from .skills.messaging.task import ApprovedMessagingTask
        from .skills.messaging import BrowserMessagingBackend
        action = Action(kind=str(approval.action["kind"]), params=dict(approval.action.get("params", {})))
        task_id = approval.task_id or f"approved-messaging-{uuid.uuid4().hex[:12]}"
        runtime_task_id = approval.runtime_task_id
        if self.debug:
            self.narrator.note(f"APPROVAL: pending_task={task_id} response={'YES' if decision == 'APPROVE' else 'NO'} result={'APPROVED' if decision == 'APPROVE' else 'REJECTED'} resumed_task={task_id}")
        resource_key = (
            "whatsapp" if action.kind == "whatsapp_send_message"
            else "gmail" if action.kind == "gmail_send_email"
            else None
        )
        if decision == "REJECT":
            if runtime_task_id:
                try:
                    self._runtime.resolve_approval(runtime_task_id, False)
                except (KeyError, ValueError):
                    self._runtime_transition(runtime_task_id, "CANCELLED", event_type="APPROVAL_DENIED", approval_state="DENIED")
            if approval.background_task_id:
                with self._background_lock:
                    record = self._background.get(approval.background_task_id)
                    if record is not None:
                        record.state = "CANCELLED"
                        record.finished_at = time.time()
            return self._approval_turn(raw, source, "Cancelled. Nothing was sent.", approval)

        approved_task = ApprovedMessagingTask(
            action,
            BrowserMessagingBackend(self._browser_for_task(task_id, resource_key=resource_key)),
            task_id=task_id,
        )
        prepared = Prepared(
            task=UserTask(raw=raw, text=approval.request, source=source, task_id=task_id, status="accepted"),
            goal=approval.goal or approved_task.goal,
            kind="approved messaging action",
            task_obj=approved_task,
            workspace=self.workspace,
            approved_action=approval.action,
            browser_resource_key=resource_key,
            runtime_task_id=runtime_task_id,
        )
        if runtime_task_id:
            try:
                self._runtime.resolve_approval(runtime_task_id, True)
            except (KeyError, ValueError):
                self._runtime_transition(runtime_task_id, "RUNNING", event_type="APPROVAL_GRANTED", approval_state="GRANTED", execution_state="RUNNING")
        if background and approval.background_task_id:
            bg_id = approval.background_task_id
            with self._background_lock:
                record = self._background.get(bg_id)
                if record is not None:
                    record.state = "QUEUED"
                    record.result = None
                    record.failure = ""
            future = self._background_executor().submit(self._resume_approved_background, bg_id, prepared)
            with self._background_lock:
                record = self._background.get(bg_id)
                if record is not None:
                    record.future = future
            return self._approval_turn(raw, source, "Approved. I’m continuing the original task.", approval)
        return self._run(prepared)

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

    def _conversation_background_run(self, event: InputEvent) -> None:
        try:
            self.submit(event.text, source=event.source)
        except Exception as exc:  # noqa: BLE001
            self._safe_note(f"⚠ [{event.event_id}] conversation failed: {type(exc).__name__}: {exc}")

    def submit_background(self, raw: str, *, source: str = "text") -> str:
        """Dispatch one input without making conversation wait behind task work.

        Approval ownership is checked first.  Obvious conversation is deliberately
        handled outside the serialized task worker: the worker is a resource for
        computer mutations, not a queue for unrelated chat turns.  Computer/task
        input still uses the exact existing ``submit`` pipeline on the background
        lane.
        """
        event = InputEvent(text=raw or "", source=source)
        # Rehydrate durable ownership at the dispatcher boundary.  This is the
        # critical restart invariant: pending approval outranks conversation and
        # new task routing even in the first turn of a fresh process.
        self._rehydrate_pending_input_owner()
        with self._background_lock:
            approval_pending = self.pending_approval is not None
        if approval_pending:
            resolved = self._resolve_approval_input(event.text, source=event.source, background=True)
            return resolved.task.task_id

        text = _WHITESPACE.sub(" ", event.text.strip()).strip(_TRAILING)
        # Runtime-status questions are answered from the authoritative registry
        # before normal conversation/task classification. This keeps /chat usable
        # for runtime inspection even when the optional benchmark task catalog is
        # unavailable in a checkout.
        runtime_reply = self._runtime_query_reply(text)
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
            self._conversation_background_executor().submit(
                self._conversation_background_run, event
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
        with self._background_lock:
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
        approval = self.pending_approval or self._rehydrate_pending_input_owner()
        if approval is not None and approval.expired:
            self.pending_approval = None
            approval = None

        # Approval state is authoritative input ownership. While a consequential
        # action is awaiting confirmation, this input is never normalized, routed,
        # planned, or turned into a new task.
        if self.pending_approval is not None:
            return self._resolve_approval_input(raw, source=source, background=False)

        if pending is not None and pending.expired:
            self.narrator.note(
                f"  (dropping an unanswered question after "
                f"{pending.age_s:.0f}s)"
            )
            pending = None

        task = normalize(raw, source=source)

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
                return self._general_action(task, runtime_task_id=getattr(self._thread_state, "runtime_task_id", None))

            # ``resolve_request`` covers every request shape the assistant knows
            # -- a named file, a named project folder, and a project to create --
            # so "open my hermes project in vs code", "open main1.mp4" and "set up
            # a python project called test_project" travel the same road to the
            # same ``run_agent_task``. The order the shapes are tried in is
            # deliberate; see ``api.resolve_request``.
            fast = classify_fast(task.text)
            if fast.route is not None:
                fast_task_id = getattr(self._thread_state, "runtime_task_id", None) or f"fast-{uuid.uuid4().hex[:12]}"
                resource_key = (
                    "youtube"
                    if (
                        "video" in task.text.casefold()
                        or "youtube" in task.text.casefold()
                        or fast.route.action_kind == "browser_play_song"
                        or re.match(r"\s*play\s+", task.text.casefold())
                    )
                    else None
                )
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
                    return self._general_action(task, runtime_task_id=getattr(self._thread_state, "runtime_task_id", None))

                return self._converse(task)

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
        return self._run(Prepared(task=task, goal=api.task_goal(task.task_id, **task.params), runtime_task_id=runtime_task_id))

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

    def _general_action(self, task: UserTask, *, runtime_task_id: str | None = None) -> Turn:
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
        ))

    @staticmethod
    def _browser_resource_key(prepared: Prepared) -> str | None:
        """Choose the narrowest reusable browser resource class for a task."""
        if prepared.browser_resource_key:
            return prepared.browser_resource_key
        text = f"{prepared.task.text} {prepared.goal}".casefold()
        if "whatsapp" in text:
            return "whatsapp"
        if "youtube" in text or re.search(r"\bplay\s+.+", text):
            return "youtube"
        if "gmail" in text or "mail.google.com" in text:
            return "gmail"
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
                approved_action=prepared.approved_action,
                fast_route=prepared.fast_route,
                fast_latency_seconds=prepared.fast_latency_seconds,
                on_event=self._runtime_event_watcher(runtime_task_id, self._watcher(lines)),
                browser_backend=browser_backend,
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

        if result.needs_input:
            if self.pending_approval is not None:
                self._runtime_transition(runtime_task_id, "WAITING_FOR_APPROVAL", approval_state="WAITING", execution_state="WAITING_FOR_APPROVAL")
            else:
                self._runtime_transition(runtime_task_id, "WAITING_FOR_USER", execution_state="WAITING_FOR_USER")
        elif result.status is TaskStatus.CANCELLED:
            self._runtime_transition(runtime_task_id, "CANCELLED", event_type="TASK_CANCELLED", cancellation_state="CANCELLED")
        elif result.ok:
            self._runtime_transition(runtime_task_id, "COMPLETED", event_type="TASK_COMPLETED", execution_state="COMPLETED", verification_state=result.verified, verification_summary=result.verified, result_summary=result.detail)
        elif result.status is TaskStatus.UNKNOWN:
            self._runtime_transition(runtime_task_id, "FAILED", event_type="TASK_FAILED", failure_state=result.detail, failure_category="UNKNOWN", execution_state="FAILED", verification_state=result.verified, verification_summary=result.verified, result_summary=result.detail)
        else:
            self._runtime_transition(runtime_task_id, "FAILED", event_type="TASK_FAILED", failure_state=result.detail, failure_category="EXECUTION", execution_state="FAILED", verification_state=result.verified, verification_summary=result.verified, result_summary=result.detail)

        if self.debug:
            for line in result.report_lines():
                self.narrator.note(line)

        if result.needs_input and result.question is not None:
            context = result.question.context or ""
            if context.startswith("APPROVAL_ACTION:"):
                try:
                    action = json.loads(context[len("APPROVAL_ACTION:"):])
                    if isinstance(action, dict) and isinstance(action.get("kind"), str):
                        self.pending_approval = PendingApproval(
                            request=task.text, action=action, asked_at=time.time(),
                            task_id=task.task_id, goal=prepared.goal,
                            background_task_id=getattr(self._thread_state, "background_task_id", None),
                            runtime_task_id=runtime_task_id,
                        )
                        if runtime_task_id:
                            self._runtime.request_approval(runtime_task_id, {
                                "request": task.text, "action": action, "asked_at": self.pending_approval.asked_at,
                                "background_task_id": self.pending_approval.background_task_id,
                            })
                except Exception:
                    pass

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
        else:
            reply = self._action_reply(task, result)
        if result is not None and not result.needs_input and self._runtime is not None and runtime_task_id:
            pass
        turn = Turn(task=task, reply=reply, result=result,
                    status_lines=lines)
        if result.needs_input and result.question is not None:
            self._emit_reply(
                turn.reply,
                goal=prepared.goal,
                state="WAITING_FOR_APPROVAL" if self.pending_approval else "WAITING_FOR_USER",
            )
        else:
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

    def _runtime_query_reply(self, text: str) -> str | None:
        """Answer runtime-status questions from RuntimeManager truth.

        Technical task IDs are exposed only for an explicit task-ID request.
        Ordinary status questions receive semantic descriptions suitable for TTS.
        """
        lowered = _WHITESPACE.sub(" ", (text or "").strip().casefold())
        snapshot = self.runtime_snapshot()
        active = snapshot["active_tasks"]
        failed = snapshot["failed_tasks"]

        if is_explicit_task_id_request(text):
            focused_id = snapshot.get("focused_task_id")
            task_id = focused_id if isinstance(focused_id, str) else None
            if not task_id and active:
                task_id = active[0].get("task_id")
            if task_id:
                return sanitize_tts_text(f"The task ID is {task_id}.", allow_internal=True)
            return "There is no active task with a task ID right now."

        status_query = (
            "what are you doing" in lowered
            or "what task is running" in lowered
            or "what tasks are running" in lowered
            or "what is running" in lowered
        )
        failure_query = (
            "what went wrong" in lowered
            or "what task went wrong" in lowered
            or "what failed" in lowered
        )
        if status_query:
            if not active:
                return "There are no active tasks right now."
            parts = []
            for item in active:
                goal = item.get("goal") or "the requested task"
                state = str(item.get("state") or "")
                if state == "WAITING_FOR_APPROVAL":
                    parts.append(f"{goal} is waiting for your approval")
                elif state == "RECOVERY_REQUIRED":
                    parts.append(f"{goal} was interrupted and needs recovery")
                else:
                    parts.append(f"{goal} is {_friendly_runtime_state(state)}")
            return "I currently have " + "; ".join(parts) + "."
        if failure_query:
            if not failed:
                return "I don't have any failed tasks in the current runtime."
            item = failed[-1]
            goal = item.get("goal") or "the requested task"
            return f"The {goal} could not be completed."
        return None


    def _converse(self, task: UserTask) -> Turn:
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
            runtime = self.runtime_snapshot()
            context = dict(self.recent_context.planner_state())
            context["runtime"] = runtime
            reply = self._conversation_engine().reply(
                task.text,
                history=self.history,
                recent_context=context,
                memories=self._conversation_memory().search(task.text, limit=8),
            )
        except (LLMUnavailable, RuntimeError) as exc:
            return self._refused(
                task,
                f"I could not reach the conversation model ({exc}), so I have "
                "nothing to say about that and have not run anything.",
            )

        turn = Turn(task=replace(task, status="conversation"), reply=reply,
                    result=None)
        self._emit_reply(reply)
        self.history.append(turn)
        self._remember_turn(turn)
        return turn

    def _action_reply(self, task: UserTask, result: AgentResult) -> str:
        memory = self._conversation_memory()
        memories = memory.search(task.text, limit=8)
        context = dict(self.recent_context.planner_state())
        context["runtime"] = runtime_snapshot_for_user(self.runtime_snapshot())
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
