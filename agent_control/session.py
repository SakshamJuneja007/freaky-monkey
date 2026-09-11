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
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from . import api
from .api import AgentResult, TaskStatus
from .types import Action
from .task import Task
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
        }


def _words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


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
    "find", "search", "look", "browse", "list", "show",
    "pick", "choose", "select", "play",
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
    task_obj: Task | None = None
    readable_roots: tuple[Path, ...] = ()
    workspace: Path | None = None
    approved_action: dict[str, Any] | None = None


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
    history: list[Turn] = field(default_factory=list)
    #: The one question awaiting an answer, or ``None``. This is not
    #: conversational memory: :meth:`submit` takes and clears it on its first
    #: line, so it cannot outlive the turn immediately after the one that set it,
    #: and :func:`choose` can only ever return a path it already contains.
    pending: Pending | None = None
    pending_approval: PendingApproval | None = None
    recent_context: RecentContext = field(default_factory=RecentContext)
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

    def close(self, timeout_s: float = 15.0) -> None:
        self.narrator.close(timeout_s)

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
        approval, self.pending_approval = self.pending_approval, None

        if approval is not None and approval.expired:
            approval = None

        normalized_raw = (raw or "").strip().lower()
        if approval is not None and normalized_raw in {"yes", "y", "approve", "approved", "send it", "do it", "confirm"}:
            from .skills.messaging.task import ApprovedMessagingTask
            from .skills.builtin import _BROWSER_BACKEND
            if _BROWSER_BACKEND is None:
                return self._refused(UserTask(raw=raw, text=approval.request, source=source, status="dropped"), "The messaging browser is not available, so nothing was sent.")
            approved_task = ApprovedMessagingTask(
                Action(kind=str(approval.action["kind"]), params=dict(approval.action.get("params", {}))),
                __import__("agent_control.skills.messaging", fromlist=["BrowserMessagingBackend"]).BrowserMessagingBackend(_BROWSER_BACKEND),
            )
            return self._run(Prepared(
                task=UserTask(raw=raw, text=approval.request, source=source, task_id=approved_task.task_id, status="accepted"),
                goal=approved_task.goal, kind="approved messaging action", task_obj=approved_task,
                readable_roots=(), workspace=self.workspace, approved_action=approval.action,
            ))

        if approval is not None and normalized_raw in {"no", "n", "cancel", "stop", "never mind", "nevermind"}:
            return self._refused(UserTask(raw=raw, text=approval.request, source=source, status="dropped"), "Cancelled. Nothing was sent.")

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
                return self._general_action(task)

            # ``resolve_request`` covers every request shape the assistant knows
            # -- a named file, a named project folder, and a project to create --
            # so "open my hermes project in vs code", "open main1.mp4" and "set up
            # a python project called test_project" travel the same road to the
            # same ``run_agent_task``. The order the shapes are tried in is
            # deliberate; see ``api.resolve_request``.
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
                    return self._general_action(task)

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
                status="accepted",
            )

        return self._run(Prepared(
            task=task,
            goal=api.task_goal(task.task_id, **task.params),
        ))

    def submit_capture(self, capture: Capture) -> Turn:
        """Submit a transcript, or report honestly why there is none.

        The convenience is the point of the feature: nobody should have to retype
        what they just said. It is convenience only -- the accepted branch calls
        the same :meth:`submit` a keyboard does.
        """
        if not capture.ok:
            task = UserTask(raw="", text="", source="voice", status="empty")
            if capture.detail:
                self.narrator.note(f"  ({capture.detail})")
            return self._refused(task, phrase_not_heard(capture.reason))

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

        self.pending = Pending(
            query=resolved.query,
            task_id=resolved.task_id,
            choices=resolved.choices,
            asked_at=time.time(),
        )

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

    def _general_action(self, task: UserTask) -> Turn:
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
        if self._general_workspace is None:
            self._general_workspace = (
                self.workspace.resolve()
                if self.workspace is not None
                else api.default_workspace("general").resolve()
            )

        return self._run(Prepared(
            task=replace(task, task_id=general_task.task_id, status="accepted"),
            goal=general_task.goal,
            kind="general action",
            task_obj=general_task,
            readable_roots=roots,
            workspace=self._general_workspace,
        ))

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
        if self.debug:
            self.narrator.accepted(task.task_id, prepared.goal, debug=True)
            aside = f" ({prepared.kind})" if prepared.kind else ""
            self.narrator.note(f"[{task.task_id}] running{aside} ...")
        lines: list[str] = []

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
                on_event=self._watcher(lines),
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

        if self.debug:
            for line in result.report_lines():
                self.narrator.note(line)

        if result.needs_input and result.question is not None:
            context = result.question.context or ""
            if context.startswith("APPROVAL_ACTION:"):
                try:
                    action = json.loads(context[len("APPROVAL_ACTION:"):])
                    if isinstance(action, dict) and isinstance(action.get("kind"), str):
                        self.pending_approval = PendingApproval(task.text, action, time.time())
                except Exception:
                    pass

        reply = self._action_reply(task, result)
        turn = Turn(task=task, reply=reply, result=result,
                    status_lines=lines)
        self.narrator.reply(turn.reply)
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
        try:
            reply = self._conversation_engine().reply(
                task.text,
                history=self.history,
                recent_context=self.recent_context.planner_state(),
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
        self.narrator.reply(reply)
        self.history.append(turn)
        self._remember_turn(turn)
        return turn

    def _action_reply(self, task: UserTask, result: AgentResult) -> str:
        memory = self._conversation_memory()
        memories = memory.search(task.text, limit=8)
        context = dict(self.recent_context.planner_state())
        event = {
            "request": task.text,
            "status": result.status.value,
            "success": result.ok,
            "task_id": result.task_id,
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
        self.narrator.reply(reply)
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