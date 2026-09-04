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

import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from . import api
from .api import AgentResult
from .response import (
    Narrator,
    phrase_choice,
    phrase_dropped,
    phrase_empty,
    phrase_no_location,
    phrase_not_heard,
    phrase_result,
    phrase_unsupported,
)

#: Trailing characters a person types or an STT model appends that are never part
#: of a task id. Kept to punctuation: stripping words would be intent inference,
#: which this layer does not do.
_TRAILING = " \t\r\n.!?,;:'\"`"

_WHITESPACE = re.compile(r"\s+")

#: Status lines are printed for these trace events and no others. An allowlist,
#: not a filter, so a new event kind is silent until somebody decides what it
#: should say -- the failure mode of a denylist here is inventing narration for
#: an event nobody designed a sentence for.
STATUS_EVENTS = (
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
    #: "accepted" | "empty" | "unsupported" | "dropped". Distinct causes stay
    #: distinct; an empty line is not a failed request, and "dropped" -- a line
    #: that answered a pending question with nothing selectable -- is neither.
    #: Deliberately not "cancelled": that word is reserved for a *run* that was
    #: interrupted, which is a different event with a different status.
    status: str = "accepted"

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


def normalize(raw: str, *, source: str = "text") -> UserTask:
    """Turn one line of input -- typed or transcribed -- into a runnable request.

    Three outcomes, and the third is the one that keeps the project honest:

    * nothing but whitespace -> ``empty``, and no execution is attempted;
    * resolves to a registered task -> ``accepted``;
    * anything else -> ``unsupported``.

    The last branch is not a stub waiting for a general natural-language
    front-end. The backend runs registered workflows; a request outside that set
    is refused by name, with the set listed, rather than being coerced into the
    nearest task. Guessing would make the interface look general while making the
    agent act on something the user did not ask for.

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
                        status="unsupported")

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


def _status_line(event: dict[str, Any]) -> str:
    """One short line describing a trace event, or ``""`` to stay quiet.

    Every line here is a rendering of an event that was actually emitted by the
    control loop. Nothing is timed, predicted, or interpolated between events:
    the display can only lag reality, never invent it.
    """
    kind = event.get("event", "")

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
    workspace: Path | None = None
    history: list[Turn] = field(default_factory=list)
    #: The one question awaiting an answer, or ``None``. This is not
    #: conversational memory: :meth:`submit` takes and clears it on its first
    #: line, so it cannot outlive the turn immediately after the one that set it,
    #: and :func:`choose` can only ever return a path it already contains.
    pending: Pending | None = None

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
        ordinary accepted :class:`UserTask` and falls into the same execution
        block below. Nothing about answering a question executes anything by
        itself.
        """
        # Take and clear, on the first line and unconditionally. A question
        # therefore cannot survive the turn that follows it, whatever that turn
        # turns out to be -- which is what keeps this from becoming context the
        # agent accumulates.
        pending, self.pending = self.pending, None

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
        fresh = task.accepted or api.parse_request(task.text) is not None

        if pending is not None and not fresh:
            answered = self._answer(pending, task, source=source)

            if isinstance(answered, Turn):
                return answered

            task = answered

        if task.status == "empty":
            return self._refused(task, phrase_empty())

        if task.status == "unsupported":
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
                self.narrator.note(
                    f"No registered workflow matches {task.text!r}.\n"
                    f"Registered: {', '.join(api.registered_tasks())}"
                )
                return self._refused(task, phrase_unsupported())

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

        lines: list[str] = []
        self.narrator.accepted(
            task.task_id,
            api.task_goal(task.task_id, **task.params),
        )
        self.narrator.note(f"[{task.task_id}] running ...")

        result = api.run_agent_task(
            task.text,
            task_id=task.task_id,
            task_params=task.params,
            planner=self.planner,
            max_steps=self.max_steps,
            keep_workspace=self.keep_workspace,
            workspace=self.workspace,
            use_memory=self.use_memory,
            on_event=self._watcher(lines),
        )

        for line in result.report_lines():
            self.narrator.note(line)

        turn = Turn(task=task, reply=phrase_result(result), result=result,
                    status_lines=lines)
        self.narrator.reply(turn.reply)
        self.history.append(turn)
        return turn

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
            if event.get("event") not in STATUS_EVENTS:
                return
            line = _status_line(event)
            if line:
                lines.append(line)
                self.narrator.note(line)

        return watch
