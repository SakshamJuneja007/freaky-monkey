"""The one execution entry point. CLI, text, and voice all come through here.

There is deliberately no second way to run a task. Before this module the only
caller was `main.py`, which assembled workspace, policy, task, planner, config
and trace inline and then printed -- so a voice layer would have had to rebuild
that assembly, and two assemblies drift. `run_agent_task` is that assembly,
extracted into one place.

The important boundary is explicit:

- the disposable workspace is the only general write location;
- registered benchmark tasks may receive narrowly scoped read-only roots;
- the policy layer still decides whether every proposed action is allowed;
- success is derived from verification, never from what the planner claims.
"""

from __future__ import annotations

import itertools
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence

from .policy import Policy
from .memory import DEFAULT_STORE, FileMemory
from .recovery import RecoveryBudget
from .runner import RunConfig, RunOutcome, run_task
from .task import Task
from .trace import Trace
from .types import Check, Clarification, FailureClass, PolicyDenied, Verdict


ROOT = Path(__file__).resolve().parent.parent


class TaskStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    UNKNOWN = "UNKNOWN"
    UNSUPPORTED = "UNSUPPORTED"
    UNAVAILABLE = "UNAVAILABLE"
    #: The person stopped the run. Not a verdict about the task -- the absence of
    #: one. Distinct from UNKNOWN, which means we looked and could not tell.
    CANCELLED = "CANCELLED"
    #: The run stopped to ask the person something, and can be resumed from where
    #: it stopped. Also the absence of a verdict rather than a bad one, and
    #: deliberately not UNKNOWN: nothing went wrong and nothing was unreadable --
    #: what is missing is a decision only they can make. Waiting is not failure
    #: (PART 3 requirement 6), and because ``is_success`` stays False it can never
    #: be spoken as completion (requirement 8).
    NEEDS_INPUT = "NEEDS_INPUT"

    @property
    def is_success(self) -> bool:
        return self is TaskStatus.SUCCESS


@dataclass
class AgentResult:
    """What actually happened, derived from verification."""

    request: str
    task_id: str = ""
    status: TaskStatus = TaskStatus.UNKNOWN

    completed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)

    false_success: bool = False
    reported_success: bool = False
    verified: str = Verdict.UNKNOWN.value

    steps_used: int = 0
    duration_seconds: float = 0.0

    failure_categories: list[str] = field(default_factory=list)
    recovery_attempts: int = 0
    recovery_successes: int = 0

    aborted_reason: str | None = None
    detail: str = ""
    workspace: str = ""
    trace_file: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    #: What the task acted on, as a short display name -- ``chess-ai``, not
    #: ``D:\chess-ai``. Present so the response layer can say "Opened chess-ai in
    #: VS Code" without re-deriving the object from the request string, which is
    #: the user's wording and may not name the thing that was actually resolved.
    #: Empty when the task never got as far as being built, or does not have a
    #: single target. A short name and not a path on purpose: this is the string a
    #: sentence is built from, and an absolute path read aloud is unusable.
    target: str = ""

    #: The registered application a task acts through, when it names one
    #: (``vscode``). Empty for tasks that hand their target to whatever the shell
    #: has associated with it, because naming an application there would be a
    #: guess.
    app: str = ""

    #: What the run stopped to ask, when ``status`` is ``NEEDS_INPUT``. The
    #: :class:`~agent_control.types.Clarification` object rather than its text,
    #: because a caller holding only a sentence would have to re-derive the
    #: observed options and the record of what could *not* be observed -- and
    #: would then be free to invent both, which is the one thing PART 3 forbids.
    question: Clarification | None = None

    outcome: RunOutcome | None = None

    @property
    def ok(self) -> bool:
        return self.status is TaskStatus.SUCCESS

    @property
    def needs_input(self) -> bool:
        """Whether this run is suspended on a question rather than finished.

        Not the complement of ``ok``: a suspended run is neither a success nor a
        failure, and a caller has to tell "ask, then continue this run" apart from
        "report it and stop". Everything that reads ``ok`` keeps working unchanged,
        because a suspended run answers False there too.
        """
        return self.status is TaskStatus.NEEDS_INPUT

    def to_json(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "task_id": self.task_id,
            "status": self.status.value,
            "completed": self.completed,
            "failed": self.failed,
            "unresolved": self.unresolved,
            "checks": self.checks,
            "false_success": self.false_success,
            "reported_success": self.reported_success,
            "verified": self.verified,
            "steps_used": self.steps_used,
            "duration_seconds": round(self.duration_seconds, 3),
            "failure_categories": self.failure_categories,
            "recovery_attempts": self.recovery_attempts,
            "recovery_successes": self.recovery_successes,
            "aborted_reason": self.aborted_reason,
            "detail": self.detail,
            "workspace": self.workspace,
            "trace_file": self.trace_file,
            "usage": self.usage,
            "target": self.target,
            "app": self.app,
            "question": self.question.to_json() if self.question else None,
            "needs_input": self.needs_input,
            "ok": self.ok,
        }

    def report_lines(self) -> list[str]:
        lines = [f"{self.status.value}: {self.detail}"]

        reasons = {
            check["name"]: check.get("reason", "")
            for check in self.checks
        }

        for title, names in (
            ("Completed and verified", self.completed),
            ("Failed", self.failed),
            ("Unresolved", self.unresolved),
        ):
            if not names:
                continue

            lines.append(f"\n{title}:")

            for name in names:
                reason = reasons.get(name, "")
                lines.append(
                    f"  - {name}" + (f": {reason}" if reason else "")
                )

        if self.false_success:
            lines.append(
                "\nThe planner claimed success; verification did not agree."
            )

        return lines


def registered_tasks() -> tuple[str, ...]:
    from benchmark.tasks import TASK_IDS

    return tuple(TASK_IDS)


def resolve_task(request: str) -> str | None:
    """Resolve a registered task id.

    This deliberately performs only task-id normalization, not general natural
    language intent inference.
    """
    candidate = (request or "").strip().lower()

    if not candidate:
        return None

    known = registered_tasks()

    if candidate in known:
        return candidate

    normalised = (
        candidate
        .replace("-", "_")
        .replace(" ", "_")
        .strip("_.!?")
    )

    while "__" in normalised:
        normalised = normalised.replace("__", "_")

    return normalised if normalised in known else None


# ----------------------------------------------------------------------
# "open <filename>": the one request shape resolved from the location index
# ----------------------------------------------------------------------

#: File types this task may hand to the OS. ``open_file`` calls ``os.startfile``,
#: which runs whatever the shell associates with the extension, so an allowlist of
#: *data* is the difference between opening a document and executing a program.
#: Same shape and same reasoning as ``policy.DEFAULT_ALLOWED_EXECUTABLES``: the
#: things not named here are refused, including ones nobody thought of.
OPENABLE_SUFFIXES = frozenset({
    ".pdf",
    ".mp4", ".mkv", ".mov", ".avi", ".webm",
    ".mp3", ".wav", ".flac", ".m4a",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg",
    ".txt", ".md", ".csv", ".log", ".json", ".xml", ".yml", ".yaml",
    ".docx", ".xlsx", ".pptx", ".rtf",
    ".html", ".htm",
    ".zip",
})

#: A request to open something. Not a general verb list: each of these takes a
#: file as its object, so none of them can be satisfied by a different action.
_OPEN_VERBS = frozenset({"open", "show", "view", "launch", "start"})

#: Leading words that carry no request. "hey" is here because there is no wake
#: word: the user's "hey open main1.mp4" arrives with the "hey" still in it, and
#: dropping it is transcript cleanup rather than intent inference.
_LEADING_FILLER = frozenset({
    "hey", "hi", "hello", "ok", "okay", "yo", "so", "um", "uh", "erm",
    "please", "now", "just", "quickly",
    "can", "could", "would", "will", "you", "u",
    "jarvis", "assistant", "agent", "computer",
})

#: A name in quotes, straight or curly. Needed because real filenames contain
#: spaces -- ``open "Last Day.pdf"`` has no whitespace-delimited token to find.
_QUOTED_NAME = re.compile(r"[\"'‘“]([^\"'’”]{1,160})[\"'’”]")

_STRIP_AROUND_TOKEN = " \t\r\n.,;:!?()[]{}<>\"'`‘’“”"

#: Punctuation stripped from ordinary sentence words. Narrower than the set used on
#: the filename itself, because brackets belong to names like ``report (1).pdf``.
_STRIP_SENTENCE = " \t\r\n.,;:!?\"'`‘’“”"

#: Inflections, because spoken sentences do not arrive in the imperative. "See how
#: it opens account statement1.pdf" is the same request as "open account
#: statement1.pdf", and refusing the first while accepting the second is a quirk of
#: the parser rather than a fact about what the agent can do.
_VERB_FORMS = frozenset(
    form
    for verb in _OPEN_VERBS
    for form in (verb, f"{verb}s", f"{verb}ed", f"{verb}ing")
)

#: Words that cannot be part of a filename, used to find where a multi-word name
#: begins. Bounded by the verb on the left in any case, so this only has to stop
#: the connective tissue between the verb and the name.
_NOT_A_NAME_PART = _LEADING_FILLER | _VERB_FORMS | frozenset({
    "the", "a", "an", "my", "your", "our", "this", "that", "these", "those",
    "it", "its", "file", "files", "filename", "named", "called", "document",
    "video", "movie", "song", "audio", "image", "photo", "picture", "sheet",
    "in", "on", "at", "for", "from", "of", "with", "into", "to", "up",
    "and", "or", "then", "is", "was", "are", "be",
    "see", "how", "let", "lets", "want", "wanna", "need", "me", "i", "we",
    "dot", "also", "some", "any", "there", "here", "which", "what", "where",
})

#: ``dot`` spelled out. STT writes "main1 dot mp4" for a name the speaker
#: pronounced with an extension, so joining it back up is transcript repair -- the
#: words already say which extension, nothing is being guessed.
_SPOKEN_DOT = re.compile(
    r"\s+dot\s+(" + "|".join(
        re.escape(suffix[1:])
        for suffix in sorted(OPENABLE_SUFFIXES, key=len, reverse=True)
    ) + r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Choice:
    """One candidate location, with the shortest word that distinguishes it."""

    path: str
    label: str
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"path": self.path, "label": self.label, "detail": self.detail}


@dataclass(frozen=True)
class Resolved:
    """What a request to open a named file turned into.

    Exactly one of three things is true, and they are different answers rather
    than degrees of the same one: it can run, it needs one question answered
    first, or it cannot be run and ``detail`` says why.
    """

    query: str = ""
    task_id: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    choices: tuple[Choice, ...] = ()
    detail: str = ""
    #: The verb this request was asking for, carried only so that a refusal can
    #: be phrased in it. "I could not open test_project" is the wrong sentence
    #: for a request to *create* test_project -- it names a search that was never
    #: the point -- and the resolver is the last place that still knows which of
    #: the two was asked. ``response.phrase_no_location`` reads it.
    action: str = "open"

    @property
    def runnable(self) -> bool:
        return bool(self.task_id) and not self.choices

    @property
    def ambiguous(self) -> bool:
        return bool(self.choices)

    def to_json(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "task_id": self.task_id,
            "params": dict(self.params),
            "choices": [choice.to_json() for choice in self.choices],
            "detail": self.detail,
            "action": self.action,
            "runnable": self.runnable,
            "ambiguous": self.ambiguous,
        }


def parse_open_candidates(text: str) -> tuple[str, ...]:
    """Filenames the sentence might be naming, longest first.

    Strictly lexical, and that is the point. It requires a verb from a closed set
    and exactly one token carrying an openable extension. It does not infer what an
    unrecognised sentence probably meant -- ``normalize`` refuses unregistered
    requests by name, and a parser that guessed here would make that refusal a lie.

    Several candidates rather than one because a filename may contain spaces and
    nothing in the words says where it starts: "open the file account
    statement1.pdf" could name ``account statement1.pdf`` or ``statement1.pdf``.
    Both are returned, longest first, and the location index decides -- a guess
    checked against the disk is better than a guess about English.

    Empty for: no verb, no filename, an extension outside
    :data:`OPENABLE_SUFFIXES`, or *two* filenames -- "open a.pdf and b.pdf" names
    two things and this task opens one.
    """
    text = _SPOKEN_DOT.sub(lambda match: f".{match.group(1)}", text or "")
    words = text.split()
    if not words:
        return ()

    lowered = [word.strip(_STRIP_AROUND_TOKEN).lower() for word in words]
    verb_at = next(
        (index for index, word in enumerate(lowered) if word in _VERB_FORMS),
        None,
    )
    if verb_at is None:
        return ()

    rest = " ".join(words[verb_at + 1:])
    quoted = _QUOTED_NAME.search(rest)
    if quoted:
        name = quoted.group(1).strip()
        return (name,) if Path(name).suffix.lower() in OPENABLE_SUFFIXES else ()

    #: Interior words keep their brackets; only the name-bearing token is stripped
    #: hard, so a trailing "?" cannot end up inside the extension.
    parts = [word.strip(_STRIP_SENTENCE) for word in words]
    tails = [
        index
        for index in range(verb_at + 1, len(words))
        if Path(words[index].strip(_STRIP_AROUND_TOKEN)).suffix.lower()
        in OPENABLE_SUFFIXES
    ]
    if len(tails) != 1:
        return ()

    end = tails[0]
    parts[end] = words[end].strip(_STRIP_AROUND_TOKEN)

    start = end
    while (
        start - 1 > verb_at
        and parts[start - 1]
        and lowered[start - 1] not in _NOT_A_NAME_PART
    ):
        start -= 1

    return tuple(
        " ".join(parts[first:end + 1]) for first in range(start, end + 1)
    )


def parse_open_request(text: str) -> str | None:
    """The most specific filename in an open request, or ``None``.

    The whole shortlist is in :func:`parse_open_candidates`; this is the single
    best reading of the sentence, for callers with nothing to check it against.
    """
    candidates = parse_open_candidates(text)
    return candidates[0] if candidates else None


def _labels(paths: Sequence[str]) -> list[str]:
    """One short word per path that no other path in the list contains.

    Read aloud, two absolute paths are indistinguishable noise; "Downloads or
    assets" is a question a person can answer. The label is picked from the
    deepest folder inwards, so it is the most specific difference, not the drive.
    """
    parents = [list(Path(path).parent.parts) for path in paths]
    labels: list[str] = []

    for index, own in enumerate(parents):
        others = [set(other) for pos, other in enumerate(parents) if pos != index]
        unique = next(
            (
                part
                for part in reversed(own)
                if all(part not in other for other in others)
            ),
            "",
        )
        fallback = own[-1] if own else "disk"
        labels.append((unique or fallback).strip("\\/:") or "disk")

    return labels


def _live_locations(
    name: str,
    *,
    limit: int,
    memory: Any | None,
) -> tuple[Any, list[Any], list[Any]]:
    """``(recall, remembered_with_this_name, still_on_disk)`` for one filename.

    ``memory.recall`` is a *ranked* lookup: asked for ``main1.mp4`` on this machine
    it also returns ``mainn.mp4.mp4`` and ``classroom.mp4.mp4``, which is right for
    "find me something like this" and wrong here. A file not called what the user
    said is not the file they named, so the shortlist is cut to exact filename
    matches -- otherwise the question would claim to know five places with a name
    that exists in two.

    Then the disk is re-checked. The index is a cache written at refresh time; a
    file that has since moved is a stale row, and offering it would be presenting a
    cache as a fact.
    """
    from . import memory as memory_module

    #: A wider pool than we will offer, because the ranking mixes near-misses in
    #: among the exact matches and the cut below throws them away: asking for
    #: exactly ``limit`` rows could drop a real third location behind two of them.
    found = memory_module.recall(name, limit=max(1, limit) * 4, memory=memory)
    named = [
        hit
        for hit in found.hits
        if hit.entry.is_file and hit.entry.name.lower() == name.lower()
    ]
    live = [hit for hit in named if Path(hit.entry.path).is_file()][: max(1, limit)]
    return found, named, live


def resolve_open_request(
    text: str,
    *,
    use_memory: bool = True,
    limit: int = 5,
    memory: Any | None = None,
) -> Resolved | None:
    """Turn "open <filename>" into something runnable, or into one question.

    ``None`` means the text was never an open-by-name request, so the caller's
    existing refusal path applies unchanged. A :class:`Resolved` means it was one,
    and then every outcome -- including "I have no idea where that is" -- is
    reported as itself.

    Remembered paths are re-checked against the filesystem before being offered,
    and the shortlist is cut to exact filename matches; both live in
    :func:`_live_locations`.

    Where the spoken sentence leaves it unclear which words are the filename,
    :func:`parse_open_candidates` hands over every reading and this function tries
    them longest first, keeping the first that names something actually on disk.

    ``memory`` overrides the shared index, mirroring ``memory.recall``'s own
    parameter. Only tests pass it; the pipeline uses the one index.
    """
    candidates = parse_open_candidates(text)

    if not candidates:
        return None

    if not use_memory:
        return Resolved(
            query=candidates[0],
            detail=(
                "file memory is off for this session, so I have no way to find "
                f"{candidates[0]} without a full path."
            ),
        )

    name = candidates[0]
    first_named: list[Any] = []
    indexed = 0
    live: list[Any] = []

    for position, candidate in enumerate(candidates):
        found, named, live = _live_locations(candidate, limit=limit, memory=memory)

        if not found.available:
            return Resolved(
                query=candidate,
                detail=(
                    "there is no location index yet, so I do not know where "
                    f"{candidate} is. Build one with: python main.py memory --refresh"
                ),
            )

        if position == 0:
            first_named, indexed = named, found.indexed

        if live:
            name = candidate
            break

    if not live:
        why = (
            f"I remembered {len(first_named)} location(s) for {name}, but nothing is "
            "there any more -- the index is out of date. Refresh it with: "
            "python main.py memory --refresh"
            if first_named
            else f"nothing in the location index is named {name} "
                 f"({indexed} entries known)."
        )
        return Resolved(query=name, detail=why)

    if len(live) == 1:
        return Resolved(
            query=name,
            task_id="open_named_file",
            params={"path": live[0].entry.path},
        )

    paths = [hit.entry.path for hit in live]

    return Resolved(
        query=name,
        task_id="open_named_file",
        choices=tuple(
            Choice(path=hit.entry.path, label=label, detail=hit.describe())
            for hit, label in zip(live, _labels(paths))
        ),
    )


# ----------------------------------------------------------------------
# Projects: the same road, with a directory at the end of it
# ----------------------------------------------------------------------

#: Words that name a directory rather than a file. A superset of
#: ``memory.DIRECTORY_WORDS`` because that set decides *ranking* and this one
#: decides *intent*: "repo" is a reliable sign the user means a folder even where
#: it would be a poor search term.
_PROJECT_WORDS = frozenset({
    "project", "projects", "folder", "directory", "dir",
    "repo", "repository", "workspace", "codebase",
})

#: How a person names the editor out loud. Matched against the whole normalised
#: sentence rather than token by token, because two of the three are multi-word
#: and STT writes "vs code" as often as "vscode".
_EDITOR_PHRASES = (
    "visual studio code", "vs code", "vs-code", "vscode", "code editor",
)

#: The same names, reduced to their vocabulary. A candidate built only from these
#: words has named the editor, not a project, however it was spelled -- so this
#: catches "vscode", "vs_code", "visual studio code" and the bare "code editor"
#: with one rule instead of a list of spellings that would always be missing one.
_EDITOR_VOCAB = frozenset({
    "visual", "studio", "code", "vs", "vscode", "editor", "ide",
})


def _names_no_project(spelling: str) -> bool:
    """Whether a candidate is made only of words that never identify a project.

    Two cases, both of which arise from the same suffix enumeration: "open vscode"
    reduces to the editor's own name, and dropping leading words from "chess-ai
    project" eventually reduces to the bare word "project". Neither is something
    to go looking for on disk, and searching anyway would answer a request the
    user did not make.
    """
    words = [word for word in re.split(r"[^a-z0-9]+", spelling.lower()) if word]
    return bool(words) and all(
        word in _EDITOR_VOCAB or word in _PROJECT_WORDS for word in words
    )

#: Words that cannot continue a folder name, used to find where the name ends.
#: The name is bounded by the verb on the left, so this only has to stop the
#: connective tissue on the right -- "... in vscode", "... and then run it".
_ENDS_A_PROJECT_NAME = _PROJECT_WORDS | frozenset({
    "in", "into", "with", "using", "inside", "on", "at", "via", "under", "to",
    "and", "or", "then", "for", "please", "now", "again",
})

#: Dropped from the front of the name. Only words that are never part of a
#: folder name: everything else is left in place and handled by trying shorter
#: readings, so a project really called "new-site" is still reachable.
#:
#: "up" is here rather than in ``_ENDS_A_PROJECT_NAME`` because "open up my
#: tekken ai project" puts it immediately after the verb, where treating it as a
#: terminator ends the name before it starts.
_NOT_A_PROJECT_NAME_START = _LEADING_FILLER | _VERB_FORMS | frozenset({
    "the", "a", "an", "my", "your", "our", "its", "this", "that", "these",
    "those", "it", "me", "i", "we", "want", "wanna", "need", "let", "lets",
    "see", "show", "up",
})

#: Below this, a remembered folder name is too short to be worth searching for.
_MIN_PROJECT_NAME_CHARS = 3

#: Possessives a speech recogniser may weld onto the front of a name it does not
#: know. Observed, not imagined: reading "Open my Hermes project in VS Code" aloud
#: on this machine comes back from the provider as "Open MyHermes project in VS
#: Code", because ``My<Name>`` is a real product-naming pattern and the recogniser
#: has more of those in its vocabulary than it has Greek gods. Peeling the
#: possessive back off is transcript repair of the same kind as ``_SPOKEN_DOT`` and
#: ``_project_name_variants``: the sentence already said which folder.
#:
#: Possessives only, and articles deliberately left out. "the" is a prefix of
#: "theme", "theory" and "these", so including it would manufacture a reading for
#: every project whose name starts that way, and no recogniser has been seen to
#: weld it. The split reading is only ever *added* after the welded one, so a
#: project genuinely called ``myhermes`` is still found first either way.
_GLUED_POSSESSIVES = ("my", "our", "your")

#: Ceiling on how many readings of one sentence are tried. A spoken project name
#: is one to three words; six is generous and keeps the loop bounded.
_MAX_PROJECT_NAME_WORDS = 6


def _names_the_editor(text: str) -> bool:
    """Whether the sentence mentions VS Code by any of its spoken names."""
    flat = " ".join(text.lower().split())
    return any(phrase in flat for phrase in _EDITOR_PHRASES)


def _project_name_variants(words: Sequence[str]) -> list[str]:
    """One phrase, spelled the ways a folder on disk might spell it.

    Speech does not pronounce separators: "chess ai" and "chess-ai" are the same
    utterance, and only one of them is the folder's actual name. Generating the
    spellings is transcript repair of the same kind as ``_SPOKEN_DOT`` -- the
    words already say which folder, so nothing is being guessed about intent, and
    each spelling still has to match an indexed directory that exists on disk
    before it is offered.
    """
    phrase = " ".join(words)

    if not phrase:
        return []

    spellings = [phrase, "-".join(words), "_".join(words), "".join(words)]

    seen: set[str] = set()
    return [
        spelling
        for spelling in spellings
        if len(spelling) >= _MIN_PROJECT_NAME_CHARS
        and not (spelling.lower() in seen or seen.add(spelling.lower()))
    ]


def _unglued(words: Sequence[str]) -> list[str] | None:
    """``words`` with a welded possessive peeled off the front of its first token.

    ``None`` when there is nothing to peel, which is the ordinary case: this fires
    on ``["myhermes", "project"]`` and leaves ``["hermes", "project"]`` alone.
    Only the first token is examined -- a possessive welded to a word in the middle
    of a name would be a different phenomenon, and none has been observed.

    The remainder has to clear ``_MIN_PROJECT_NAME_CHARS`` for the same reason a
    remembered name does: "my" plus two letters is a fragment, not a folder.
    """
    if not words:
        return None

    head = words[0]

    for possessive in _GLUED_POSSESSIVES:
        if not head.startswith(possessive):
            continue
        rest = head[len(possessive):]
        if len(rest) >= _MIN_PROJECT_NAME_CHARS:
            return [rest, *words[1:]]

    return None


def parse_project_candidates(text: str) -> tuple[str, ...]:
    """Folder names the sentence might be naming, longest reading first.

    Lexical, like :func:`parse_open_candidates`, and gated the same way: a verb
    from the closed set **and** a signal that a directory is meant -- either one
    of ``_PROJECT_WORDS`` or a mention of the editor. Without that gate "open
    notes" would become a folder search, and the honest refusal that
    ``normalize`` gives unregistered requests would turn into a guess.

    The gate is why "open the folder" yields nothing: it signals a directory but
    never names one, and a request with no object is not a request this can run.
    """
    if not text or not text.strip():
        return ()

    words = [
        word
        for word in text.strip().split()
        if word.strip(_STRIP_SENTENCE)
    ]
    lowered = [word.strip(_STRIP_SENTENCE).lower() for word in words]

    verb_at = next(
        (index for index, word in enumerate(lowered) if word in _VERB_FORMS),
        None,
    )
    if verb_at is None:
        return ()

    signalled = _names_the_editor(text) or any(
        word in _PROJECT_WORDS for word in lowered
    )
    if not signalled:
        return ()

    tail = lowered[verb_at + 1:]

    # A quoted name is unambiguous and outranks any reading of the bare words:
    # ``open "rocket os" in vscode`` says exactly which two words are the name.
    quoted = _QUOTED_NAME.search(text)
    readings: list[list[str]] = []

    if quoted:
        readings.append(quoted.group(1).split())

    front = list(
        itertools.dropwhile(lambda word: word in _NOT_A_PROJECT_NAME_START, tail)
    )
    before = list(itertools.takewhile(
        lambda word: word not in _ENDS_A_PROJECT_NAME, front,
    ))

    if before:
        stopped_at = front[len(before)] if len(front) > len(before) else ""
        if stopped_at in _PROJECT_WORDS:
            # The word that signals a project can also be part of the project's
            # own name: this repository is called ``agentic-project``, so "the
            # agentic project folder" has to be tried both ways. Longest first,
            # as everywhere else here -- the cost of the wrong reading is one
            # lookup that finds nothing.
            readings.append(before + [stopped_at])
        readings.append(before)
    else:
        # "open project rocket-os", "open the folder chess-ai": the name follows
        # the word that announced it rather than preceding it.
        after = list(itertools.dropwhile(
            lambda word: word not in _PROJECT_WORDS, front,
        ))[1:]
        readings.append(list(itertools.takewhile(
            lambda word: word not in _ENDS_A_PROJECT_NAME, after,
        )))

    # Appended last, after every literal reading, so the words as transcribed are
    # always searched for first and only a sentence that found nothing pays for
    # the repair. See ``_GLUED_POSSESSIVES``.
    for reading in list(readings):
        split = _unglued(reading)
        if split is not None:
            readings.append(split)

    candidates: list[str] = []
    seen: set[str] = set()

    for reading in readings:
        reading = reading[:_MAX_PROJECT_NAME_WORDS]
        # Longest first, then progressively dropping leading words: a stray word
        # the filler list does not know about costs one failed lookup, not the
        # whole request.
        for start in range(len(reading)):
            words = reading[start:]
            # Judged on the words, not on each spelling: "visualstudiocode" is
            # one token and would slip past a per-spelling check that the same
            # three words separated by spaces would fail.
            if _names_no_project(" ".join(words)):
                continue
            for spelling in _project_name_variants(words):
                if spelling.lower() in seen:
                    continue
                seen.add(spelling.lower())
                candidates.append(spelling)

    return tuple(candidates)


def parse_project_request(text: str) -> str | None:
    """The most specific folder name in a project-open request, or ``None``."""
    candidates = parse_project_candidates(text)
    return candidates[0] if candidates else None


def _live_projects(
    name: str,
    *,
    limit: int,
    memory: Any | None,
) -> tuple[Any, list[Any], list[Any]]:
    """``(recall, remembered_with_this_name, still_on_disk)`` for one folder name.

    The word "folder" is appended to the query on purpose. ``memory.recall``
    already distinguishes a request for a directory from a request for a file --
    ``_query_terms`` sets ``wants_dir`` from ``DIRECTORY_WORDS`` and the ordering
    puts the wanted kind first -- and without that switch the ranking answers
    "chess-ai" with the *files* inside the project, because a project's files
    outnumber the project. Measured on this machine: ``recall("chess-ai")``
    returns zero directories in its first forty hits; ``recall("chess-ai
    folder")`` returns ``D:\\chess-ai`` first.

    Then the same two cuts the file path makes, for the same two reasons: an
    exact name match, because a folder not called what the user said is not the
    folder they named; and a re-check against the disk, because the index is a
    cache written at refresh time and offering a moved folder would present a
    cache as a fact.
    """
    from . import memory as memory_module

    #: Wider than the file pool. A project's own subdirectories score just below
    #: it and are all discarded by the exact-name cut, so asking for ``limit``
    #: rows would routinely return the project and none of its namesakes.
    found = memory_module.recall(
        f"{name} folder", limit=max(1, limit) * 8, memory=memory,
    )
    named = [
        hit
        for hit in found.hits
        if hit.entry.kind == "dir" and hit.entry.name.lower() == name.lower()
    ]
    live = [hit for hit in named if Path(hit.entry.path).is_dir()][: max(1, limit)]
    return found, named, live


def resolve_project_request(
    text: str,
    *,
    use_memory: bool = True,
    limit: int = 5,
    memory: Any | None = None,
) -> Resolved | None:
    """Turn "open my X project in VS Code" into something runnable, or a question.

    The directory-flavoured twin of :func:`resolve_open_request`, returning the
    same :class:`Resolved` so that everything downstream -- the ambiguity
    question, the pending-answer road, ``run_agent_task`` -- is shared rather
    than reimplemented. ``None`` means the text was never a project-open request.
    """
    candidates = parse_project_candidates(text)

    if not candidates:
        return None

    if not use_memory:
        return Resolved(
            query=candidates[0],
            detail=(
                "file memory is off for this session, so I have no way to find "
                f"the {candidates[0]} project without a full path."
            ),
        )

    name = candidates[0]
    #: Remembered folders of this name that are no longer on disk, per reading.
    #: Kept per reading rather than pooled because the refusal names one spelling
    #: and has to count only what that spelling matched: "I remembered 2
    #: location(s) for a hermes folder" must not be counting a dead ``hermes
    #: project`` as well.
    remembered: dict[str, list[Any]] = {}
    indexed = 0
    live: list[Any] = []

    for position, candidate in enumerate(candidates):
        found, named, live = _live_projects(candidate, limit=limit, memory=memory)

        if not found.available:
            return Resolved(
                query=candidate,
                detail=(
                    "there is no location index yet, so I do not know where the "
                    f"{candidate} project is. Build one with: "
                    "python main.py memory --refresh"
                ),
            )

        if position == 0:
            indexed = found.indexed

        remembered[candidate] = named

        if live:
            name = candidate
            break

    if not live:
        # Report the plainest reading rather than the longest one. Every spelling
        # was tried, so which one appears in the sentence the user hears back is
        # a presentation choice, and "no folder named hermes" is recognisable
        # where "no folder named hermes project" sounds like a different question.
        #
        # The plainest reading *that matched something*, though, when one did. A
        # spoken "hermes project" whose remembered ``hermes`` folder has since
        # moved is an out-of-date index, and the only reading with the evidence to
        # say so is not the first one tried; reporting "no folder named hermes is
        # known" there would send someone looking for a naming problem instead of
        # running the refresh that fixes it.
        evidenced = [word for word in candidates if remembered.get(word)]
        spoken = min(evidenced or candidates,
                     key=lambda word: (len(word.split()), len(word)))
        stale = remembered.get(spoken) or []
        why = (
            f"I remembered {len(stale)} location(s) for a {spoken} folder, but "
            "nothing is there any more -- the index is out of date. Refresh it "
            "with: python main.py memory --refresh"
            if stale
            else f"no folder named {spoken} is in the location index "
                 f"({indexed} entries known)."
        )
        return Resolved(query=spoken, detail=why)

    if len(live) == 1:
        return Resolved(
            query=name,
            task_id="open_project_in_vscode",
            params={"path": live[0].entry.path},
        )

    paths = [hit.entry.path for hit in live]

    return Resolved(
        query=name,
        task_id="open_project_in_vscode",
        choices=tuple(
            Choice(path=hit.entry.path, label=label, detail=hit.describe())
            for hit, label in zip(live, _labels(paths))
        ),
    )


# ----------------------------------------------------------------------
# "set up a python project called X": the one request shape that *creates*
# ----------------------------------------------------------------------

#: Environment variable naming where new projects are created, so the location is
#: configurable without editing code.
PROJECTS_ROOT_ENV = "AGENT_PROJECTS_ROOT"

#: Where they go when it is unset: one directory inside the repository.
PROJECTS_DIR = "projects"


def projects_root() -> Path:
    """The one directory new projects may be created in.

    Narrow on purpose, and never the home folder or a drive root: this path
    becomes ``Policy.workspace`` for a setup run, and the workspace is the only
    general write root the policy layer allows, so widening it here would widen
    every write the task can make. Everything a setup run produces -- the project,
    its files, its interpreter -- lands inside it, and anything a planner proposes
    outside it is refused by ``resolve_write_path`` rather than by this function.
    """
    override = os.environ.get(PROJECTS_ROOT_ENV, "").strip()

    if override:
        return Path(override).expanduser().resolve()

    return (ROOT / PROJECTS_DIR).resolve()


#: "set up" and its inflections, collapsed to the single token ``setup`` before
#: parsing. The same class of transcript repair as ``_SPOKEN_DOT``: the spellings
#: are one utterance, and a token-level verb list cannot see a two-word verb.
_SET_UP = re.compile(r"\bset(?:s|ting)?\s+up\b", re.IGNORECASE)

#: Verbs that ask for something to be *made*. "start" and "build" are also open
#: verbs -- a person starts a project they have and one they do not -- and the
#: required name marker below is what separates the two readings, so the overlap
#: costs nothing.
_MAKE_VERBS = frozenset({
    "create", "make", "setup", "scaffold", "bootstrap", "initialise",
    "initialize", "init", "generate", "start", "build",
})


def _inflections(verb: str) -> tuple[str, ...]:
    """``create`` -> ``create, creates, created, creating``.

    Present tense is the imperative a person types; the rest is what speech
    produces ("can you make me a project", "creating a project called x").
    """
    stem = verb[:-1] if verb.endswith("e") else verb
    return (verb, f"{verb}s", f"{stem}ed", f"{stem}ing")


_MAKE_VERB_FORMS = frozenset(
    form for verb in _MAKE_VERBS for form in _inflections(verb)
) | frozenset({
    # Irregular or doubled forms the rule above cannot generate.
    "made", "built", "bootstrapped", "bootstrapping", "new",
})

#: Words saying that the thing being made is a project. A creation verb alone is
#: not enough: "create a file called notes.txt" is a different request, and
#: answering it with a whole Python scaffold would be this parser inventing a job.
_SETUP_KINDS = frozenset({"project", "projects"})

#: "python", however it is said. Together with one of ``_PROJECT_WORDS`` this also
#: counts as naming a project, so "make a python repo called x" is understood.
_PYTHON_WORDS = frozenset({"python", "python3", "py"})

#: The words that *introduce* a name. Required, and that requirement is the whole
#: reason this parser cannot steal sentences from ``parse_project_candidates``:
#: "start my chess-ai project" refers to a project that exists and falls through
#: to be opened, while "start a project called chess-ai" says outright that the
#: name is being given rather than referred to.
_NAME_MARKERS = frozenset({
    "called", "named", "name", "titled", "labelled", "labeled",
})

#: The words after which a spoken dependency list may appear.
_PACKAGE_MARKERS = frozenset({"with", "using", "including", "includes"})

#: What ends the new name on the right: the connective tissue of the rest of the
#: sentence, and the start of its second half ("... and open it in VS Code").
_ENDS_A_NEW_NAME = _PROJECT_WORDS | _PACKAGE_MARKERS | frozenset({
    "in", "into", "inside", "on", "at", "via", "under", "to", "for", "from",
    "and", "then", "or", "please", "now", "again", "that", "which", "so",
    "open", "opens", "opened", "opening", "run", "runs", "launch", "launches",
})

#: Ceiling on a spoken project name, mirroring ``_MAX_PROJECT_NAME_WORDS``.
_MAX_NEW_NAME_WORDS = 4

#: A directory name this task will create. Anchored on an alphanumeric first
#: character and admitting no separator, which is what makes ``projects_root() /
#: name`` provably inside ``projects_root()``: no ``..``, no ``/``, no ``C:``. A
#: validation that merely *discouraged* those would be a different guarantee.
_SAFE_NEW_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: A dependency name this task will pass to pip. Deliberately *only* a name: no
#: version specifier, no URL, no flag. pip reads ``--index-url`` and ``-r`` out of
#: a requirements line, so a pattern admitting a leading dash would let a spoken
#: word become a pip option; anchoring on an alphanumeric first character makes
#: that impossible rather than unlikely.
_PACKAGE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: Words that appear inside a spoken dependency list without naming a package.
#: Skipped rather than terminating: "with pandas and numpy" names two.
_NOT_A_PACKAGE = frozenset({
    "and", "or", "also", "plus", "the", "a", "an", "some", "both",
    "package", "packages", "library", "libraries", "module", "modules",
    "dependency", "dependencies", "installed", "install", "just", "only",
})

#: Words that *end* a spoken dependency list, because everything after them is
#: the rest of the sentence. Without this, "with pandas and open it in vs code"
#: would install a package called "open" -- pip would fail, but the request would
#: already have been misread, and the failure would be reported against a name the
#: user never said.
_ENDS_A_PACKAGE_LIST = _SETUP_KINDS | _NAME_MARKERS | frozenset({
    "then", "in", "into", "inside", "on", "at", "to", "for", "from", "of",
    "open", "opens", "opened", "opening", "launch", "launches", "run", "runs",
    "show", "start", "create", "make", "setup", "so", "but", "please",
    "it", "them", "that", "this", "vscode", "vs", "code", "editor",
})

#: Ceiling on a spoken dependency list. Someone naming more than a handful out
#: loud is dictating a requirements file, which is a different job than this one.
MAX_SPOKEN_PACKAGES = 8


def parse_setup_candidates(text: str) -> tuple[str, ...]:
    """Names for a project the sentence asks to *create*, best reading first.

    Three things are required, and each one rules out a sentence this task must
    not answer:

    * a creation verb, so an open request is never read as a create;
    * a word saying a *project* is what is being created, so "create a file
      called notes.txt" is not answered with a Python scaffold;
    * an explicit name marker ("called", "named") or quotes, so a sentence
      *referring* to a project the user already has -- "start my chess-ai
      project" -- stays with the parser that opens one.

    What comes back is a directory *name*, never a path: it is validated as one
    safe path segment here and joined to :func:`projects_root` by
    :func:`resolve_setup_request`, so a spoken "called ../etc" cannot become a
    write outside the projects directory.
    """
    if not text or not text.strip():
        return ()

    flat = _SET_UP.sub("setup", text)
    words = [word for word in flat.split() if word.strip(_STRIP_SENTENCE)]
    lowered = [word.strip(_STRIP_SENTENCE).lower() for word in words]

    verb_at = next(
        (
            index
            for index, word in enumerate(lowered)
            if word in _MAKE_VERB_FORMS
        ),
        None,
    )
    if verb_at is None:
        return ()

    readings: list[list[str]] = []

    def names_a_project(words_in_phrase: Sequence[str]) -> bool:
        """The object of this creation phrase is a project, not later prose."""
        return any(word in _SETUP_KINDS for word in words_in_phrase) or (
            any(word in _PYTHON_WORDS for word in words_in_phrase)
            and any(word in _PROJECT_WORDS for word in words_in_phrase)
        )

    # Quotes are a name marker of their own, and the most explicit one there is:
    # ``create a python project "rocket os"`` says exactly which words the name is.
    quoted = _QUOTED_NAME.search(flat)
    quoted_prefix = [
        word.strip(_STRIP_SENTENCE).lower()
        for word in flat[:quoted.start()].split()
    ] if quoted else []
    if quoted and names_a_project(quoted_prefix[verb_at + 1:]):
        readings.append(quoted.group(1).split())

    marker_at = next(
        (
            index
            for index, word in enumerate(lowered)
            if index > verb_at and word in _NAME_MARKERS
        ),
        None,
    )
    if (
        marker_at is not None
        and names_a_project(lowered[verb_at + 1:marker_at])
    ):
        after = lowered[marker_at + 1:]
        take = len(list(itertools.takewhile(
            lambda word: word not in _ENDS_A_NEW_NAME, after,
        )))
        readings.append([
            word.strip(_STRIP_SENTENCE)
            for word in words[marker_at + 1: marker_at + 1 + take]
        ])

    names: list[str] = []
    seen: set[str] = set()

    for reading in readings:
        reading = [word for word in reading if word][:_MAX_NEW_NAME_WORDS]

        if not reading:
            continue

        # Speech does not pronounce separators, and unlike the open path there is
        # no folder on disk to check a spelling against -- so one spelling has to
        # be chosen rather than shortlisted. A hyphen is the convention that
        # survives being read back aloud.
        name = reading[0] if len(reading) == 1 else "-".join(reading)

        if not _SAFE_NEW_NAME.match(name) or name.lower() in seen:
            continue

        seen.add(name.lower())
        names.append(name)

    return tuple(names)


def parse_setup_request(text: str) -> str | None:
    """The name a create-a-project request gives, or ``None``."""
    candidates = parse_setup_candidates(text)
    return candidates[0] if candidates else None


def parse_setup_packages(text: str) -> tuple[str, ...]:
    """Dependencies the sentence *names*, and nothing it merely implies.

    A starter project that installed what a model assumed a "web project" needs
    would be making a supply-chain decision on the user's behalf, so this reads
    only the span after "with" / "using" / "including", accepts only tokens that
    are plain distribution names, and stops at the first word that is plainly the
    rest of the sentence. Nothing is inferred from the project's name or purpose,
    and the normal result -- empty -- means pip is never run at all.
    """
    flat = _SET_UP.sub("setup", text or "")
    lowered = [word.strip(_STRIP_AROUND_TOKEN).lower() for word in flat.split()]

    marker_at = next(
        (
            index
            for index, word in enumerate(lowered)
            if word in _PACKAGE_MARKERS
        ),
        None,
    )
    if marker_at is None:
        return ()

    found: list[str] = []

    for word in lowered[marker_at + 1:]:
        if not word or word in _NOT_A_PACKAGE:
            continue

        if word in _ENDS_A_PACKAGE_LIST or not _PACKAGE_NAME.match(word):
            break

        if word not in found:
            found.append(word)

        if len(found) >= MAX_SPOKEN_PACKAGES:
            break

    return tuple(found)


def resolve_setup_request(
    text: str,
    *,
    use_memory: bool = True,
    limit: int = 5,
    memory: Any | None = None,
) -> Resolved | None:
    """Turn "set up a python project called X" into something runnable.

    No memory lookup happens here, and the unused keywords are accepted only so
    that :func:`resolve_request` can call every resolver the same way. The location
    index answers "where is the thing I have?"; this request is about a thing that
    does not exist yet, so the answer is fixed by policy rather than searched for:
    :func:`projects_root`, the only directory a setup run may write in.

    Refusing an existing project *here* rather than inside the task is deliberate.
    The task's plan is a list of writes, and the moment at which "that already
    exists" is still answerable with a sentence -- instead of with a half-
    overwritten directory -- is before any of them run.
    """
    candidates = parse_setup_candidates(text)

    if not candidates:
        return None

    name = candidates[0]
    root = projects_root()
    target = (root / name).resolve()

    # Belt and braces over ``_SAFE_NEW_NAME``: that pattern already forbids
    # separators, and this re-checks the *resolved* path, so a spelling which
    # somehow slipped past it still cannot land outside the writable directory.
    if not target.is_relative_to(root) or target == root:
        return Resolved(
            query=name,
            action="set up",
            detail=(
                f"{name!r} is not a usable project folder name. Say a plain name, "
                "like: set up a python project called test_project"
            ),
        )

    try:
        occupied = target.is_dir() and any(target.iterdir())
    except OSError as exc:
        return Resolved(
            query=name,
            action="set up",
            detail=(
                f"I could not read {target} ({exc.strerror or exc}), so I have "
                "not created anything there."
            ),
        )

    if target.exists() and not target.is_dir():
        return Resolved(
            query=name,
            action="set up",
            detail=(
                f"there is already a file at {target}, so I have not created a "
                "project there. Pick another name."
            ),
        )

    if occupied:
        return Resolved(
            query=name,
            action="set up",
            detail=(
                f"{target} already exists and is not empty, so I have not touched "
                "it. Pick another name, or move that folder aside yourself."
            ),
        )

    return Resolved(
        query=name,
        task_id="setup_python_project",
        action="set up",
        params={
            "path": str(target),
            # A list rather than a tuple: params travel through the trace as JSON,
            # and the task's ``__post_init__`` normalises it back.
            "packages": list(parse_setup_packages(text)),
        },
    )


def parse_request(text: str) -> str | None:
    """The object of a request -- a filename, a folder name, or a name to give a
    project that does not exist yet -- or ``None``.

    One lexical question ("is this sentence asking for something specific?")
    asked in one place, so the session's guard against mistaking a new request
    for the answer to a pending question covers every kind.
    """
    return (
        parse_open_request(text)
        or parse_setup_request(text)
        or parse_project_request(text)
    )


#: File types as they are *said* rather than written: "pdf", not ".pdf". Derived
#: from the one allowlist ``open_file`` already enforces so the two cannot drift
#: apart, and so this adds no vocabulary of its own.
_FILE_TYPE_WORDS = frozenset(suffix[1:] for suffix in OPENABLE_SUFFIXES)

#: Words that pick a member out of a set instead of naming one. Not a general
#: adjective list: each of these is only meaningful when several candidates exist
#: and the sentence is deliberately leaving the choice to whoever looks.
_SELECTION_WORDS = frozenset({
    "last", "latest", "newest", "recent", "recently",
    "first", "oldest",
    "largest", "biggest", "smallest", "longest", "shortest",
    "most", "least",
    "any", "random",
})


def describes_a_selection(text: str) -> bool:
    """Whether the sentence asks for a file identified by a *property* rather
    than by name: "the most recently modified PDF", "the newest mp4".

    Two signals are required together, because either one alone is ordinary:

    * a selection word -- "open my latest project" has one and means a folder;
    * a **bare** file-type word -- ``pdf``, never ``report.pdf``. A token that
      carries a real extension is a name, and ``resolve_open_request`` already
      resolves those against the location index.

    The pair is what a folder request never carries: "open the downloads folder"
    names a directory and no reading of it involves a file type. A request that
    has both is naming a *kind* of file plus a rule for choosing among them,
    which is discovery work -- something the general route can do by listing and
    comparing, and something the location index structurally cannot answer,
    since it holds names and not modification times.

    This is a routing condition and not a phrase table: it decides which
    resolver may not speak, and knows nothing about any particular sentence.
    """
    words = {word.strip(_STRIP_SENTENCE).lower() for word in text.split()}

    return bool(words & _SELECTION_WORDS) and bool(words & _FILE_TYPE_WORDS)


def resolve_request(
    text: str,
    *,
    use_memory: bool = True,
    limit: int = 5,
    memory: Any | None = None,
) -> Resolved | None:
    """Resolve a request of any known shape, or ``None`` if it is none of them.

    The order is strictest first, and each step of it prevents a specific misread:

    * **files** demand a token carrying an openable *extension*, which no folder
      name has, so a sentence naming a real file is never read as a folder;
    * **setup** demands a creation verb *and* a name marker, so "set up a project
      called python-tools" is answered by creating it rather than by searching the
      location index for a folder called "python";
    * **project-open** is last because it is the widest: its gate is a verb plus a
      hint that a directory is meant, which "start a python project called foo"
      also satisfies while meaning the opposite. That width is also why it is the
      one step a selection suppresses (see ``describes_a_selection``): "open the
      most recently modified PDF in my Downloads folder" satisfies its gate on the
      word "folder" alone, and answering as a folder search reports "no folder
      named pdf", which is not a fact about the request.
    """
    resolved = (
        resolve_open_request(
            text, use_memory=use_memory, limit=limit, memory=memory,
        )
        or resolve_setup_request(
            text, use_memory=use_memory, limit=limit, memory=memory,
        )
    )

    if resolved is not None:
        return resolved

    # Neither of the shapes above claimed it, and a sentence that picks a file
    # type by a property is not a folder request either. Leaving it unresolved
    # hands it to the caller's general route, which can list and compare; a
    # resolver that only knows names cannot answer it and should not refuse it.
    if describes_a_selection(text):
        return None

    return resolve_project_request(
        text, use_memory=use_memory, limit=limit, memory=memory,
    )


def _derive(
    outcome: RunOutcome,
) -> tuple[list[str], list[str], list[str], list[dict[str, Any]]]:
    """Split final/checkpoint verification into user-facing groups."""

    verdicts: dict[str, Check] = {}

    for result in list(outcome.checkpoints) + [outcome.final]:
        for check in result.checks:
            verdicts[check.name] = check

    completed = [
        name
        for name, check in verdicts.items()
        if check.verdict is Verdict.PASS
    ]

    failed = [
        name
        for name, check in verdicts.items()
        if check.verdict is Verdict.FAIL
    ]

    unresolved = [
        name
        for name, check in verdicts.items()
        if check.verdict is Verdict.UNKNOWN
    ]

    checks = [
        {
            "name": check.name,
            "verdict": check.verdict.value,
            "reason": check.reason,
        }
        for check in verdicts.values()
    ]

    return completed, failed, unresolved, checks


def _status(
    outcome: RunOutcome,
    completed: list[str],
    failed: list[str],
    unresolved: list[str],
) -> tuple[TaskStatus, str]:
    """Determine status from verification and runner state."""

    # First, and before the PASS check, because an interrupted run must never be
    # reported as PASS or as an ordinary FAIL (plan E phase 2). Checkpoints that
    # passed before the interrupt stay in ``completed`` -- those really were
    # verified -- but the run as a whole gets no verdict, because it did not
    # finish. ``verify_final`` was never called on this path, so there is no final
    # verdict here to override.
    if outcome.cancelled:
        return (
            TaskStatus.CANCELLED,
            outcome.aborted_reason
            or "stopped by the user; the run did not finish",
        )

    # Second, and for the same structural reason: a run suspended on a question
    # stopped somewhere other than the end, so it has no verdict to report. It is
    # ahead of every branch below because each of those is an answer -- PASS, a
    # refusal, partial progress, a failure, "could not tell" -- and this run gave
    # none of them. ``runner._awaiting`` produced zero checks on purpose, which
    # would otherwise fall through to UNKNOWN and describe a pending decision as
    # something we looked at and could not determine (PART 3: these must not
    # collapse into UNKNOWN).
    if outcome.awaiting:
        return (
            TaskStatus.NEEDS_INPUT,
            outcome.question.question if outcome.question
            else outcome.aborted_reason
            or "waiting for a decision only you can make",
        )

    # Third, and before the PASS check: a run that was aborted because Policy
    # denied an action must never be reported as SUCCESS, no matter what
    # ``verify_final`` found. ``verify_final`` still runs after this kind of
    # abort (see ``runner.run_task``) and is diagnostically useful -- it can
    # say which of the *attempted* effects genuinely hold -- but its verdict
    # describes only the actions Policy allowed to execute. A denied action
    # is invisible to that verdict by construction (``GeneralTask._remember``
    # records nothing for a refused path), so an incidental PASS on an
    # unrelated effect must not be read as the run having succeeded. Checking
    # this ahead of ``outcome.verified is Verdict.PASS`` is what makes
    # POLICY_BLOCKED terminal: nothing below can promote an aborted run back
    # to SUCCESS. ``completed``/``failed``/``unresolved`` -- and therefore the
    # diagnostic detail in ``AgentResult.checks`` -- are still derived from the
    # same verification and remain visible to the caller either way.
    aborted = outcome.aborted_reason or ""

    if aborted.startswith(FailureClass.PERMISSION_DENIED.value):
        return TaskStatus.POLICY_BLOCKED, aborted

    if outcome.verified is Verdict.PASS:
        return (
            TaskStatus.SUCCESS,
            f"verified {len(completed)} check(s)",
        )

    if completed and (failed or unresolved):
        return (
            TaskStatus.PARTIAL,
            f"{len(completed)} check(s) verified, "
            f"{len(failed)} failed, "
            f"{len(unresolved)} unresolved",
        )

    if failed:
        return (
            TaskStatus.FAILED,
            outcome.aborted_reason or "verification failed",
        )

    # A concrete execution/environment category is authoritative even when the
    # final diagnostic read cannot establish a postcondition. Do not turn a known
    # launch/application/browser failure into UNKNOWN merely because final
    # verification itself is inconclusive.
    hard_failure_categories = {
        FailureClass.EXECUTION_EXCEPTION.value,
        FailureClass.EXECUTION_TIMEOUT.value,
        FailureClass.APPLICATION_NOT_FOUND.value,
        FailureClass.BROWSER_CONNECTION_FAILED.value,
        FailureClass.ENVIRONMENT.value,
        FailureClass.PERMISSION_DENIED.value,
        FailureClass.POLICY_DENIED.value,
        FailureClass.TARGET_NOT_FOUND.value,
        FailureClass.VERIFICATION_FAILED.value,
        FailureClass.ACTION_FAILED.value,
        FailureClass.RESOURCE_UNAVAILABLE.value,
    }
    concrete_failure = next(
        (str(category) for category in outcome.failure_categories if str(category) in hard_failure_categories),
        None,
    )
    if concrete_failure:
        return TaskStatus.FAILED, outcome.aborted_reason or concrete_failure

    return (
        TaskStatus.UNKNOWN,
        outcome.aborted_reason
        or "verification could not determine the result",
    )


def default_workspace(task_id: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return ROOT / ".sandbox" / f"run-{stamp}" / task_id


#: Environment variable naming the persistent root used by Route 2 general
#: computer actions. It is deliberately separate from ``AGENT_PROJECTS_ROOT``:
#: benchmark project creation and free-form computer work have different
#: lifecycles, and changing one must not silently change the other.
GENERAL_WRITE_ROOT_ENV = "AGENT_GENERAL_WRITE_ROOT"


def general_write_root() -> Path:
    """Return the persistent write root for free-form computer actions.

    Route 2 is user-facing computer work, so its successful writes must remain
    after the run finishes. The old default pointed at ``.sandbox``, which made
    created files look like scratch artifacts rather than user deliverables.

    The default is a dedicated directory under :func:`projects_root`, which is
    already the project's approved persistent creation area. An explicit
    ``AGENT_GENERAL_WRITE_ROOT`` may relocate that directory, but the policy
    still makes the returned directory the only write root for the run.
    This function does not grant the D: drive, Downloads, or any other path
    write access.
    """
    override = os.environ.get(GENERAL_WRITE_ROOT_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()

    return (projects_root() / "deimos").resolve()


def write_root_for_task(
    task_id: str,
    params: dict[str, Any] | None = None,
) -> Path:
    """Where a task may write when the caller did not name a workspace itself.

    Almost every task wants the disposable sandbox: its writes are scratch, the run
    is a measurement, and deleting them afterwards is right. A task whose
    *deliverable* is a directory on disk cannot use it -- a project created inside
    ``.sandbox/run-20260904-101500/`` is not a project the user has -- so
    ``setup_python_project`` writes inside :func:`projects_root` instead.

    It is the projects directory and **not the project itself**, and that
    difference is load-bearing. ``Policy.__post_init__`` creates the workspace
    before the first action runs, so a workspace pointed at the new project would
    make "the project directory exists" pass before the agent had done anything --
    a verified success with nothing behind it. Handing the policy layer the parent
    leaves the project genuinely absent until an action creates it, which is what
    makes the check evidence instead of decoration.
    """
    if task_id == "setup_python_project":
        chosen = (params or {}).get("path")
        return Path(chosen).resolve().parent if chosen else projects_root()

    return default_workspace(task_id)


def task_goal(task_id: str, **params: Any) -> str:
    """The goal sentence a task states for itself.

    ``params`` is forwarded to the constructor so a parameterized task describes
    the file it was actually pointed at. Called for narration before a run, so it
    must accept the same arguments the run will use.
    """
    from benchmark.tasks import build_task

    return build_task(task_id, **params).goal.strip()


def readable_roots_for_task(
    task_id: str,
    params: dict[str, Any] | None = None,
) -> tuple[Path, ...]:
    """Return explicitly approved read-only roots for a registered task.

    Public, and deliberately so: the benchmark harness builds its own ``Policy``
    for each trial and must grant *the same* roots this path grants. When it did
    not, every ``open_last_day_pdf`` trial aborted with ``PolicyDenied`` while the
    identical task succeeded through ``run_agent_task`` -- two pipelines that had
    quietly disagreed about what a task is allowed to read. One definition, called
    from both.

    The sandbox remains the only unrestricted write root.

    `open_last_day_pdf` needs to inspect the user's Downloads directory to
    locate an existing PDF. Downloads is therefore granted as read-only for
    this task only.

    ``open_named_file`` is granted **the one chosen file** and nothing else --
    not its folder, not the drive it lives on. ``resolve_read_path`` tests
    ``resolved.is_relative_to(root)`` and a path is relative to itself, so a
    single file is a legal root and is the narrowest grant the policy layer can
    express. A sibling file in the same directory stays unreadable, which is also
    what keeps ``_remembered``'s shortlist down to the path the user picked.

    ``open_project_in_vscode`` is granted **the one chosen project directory**,
    which is wider than a single file and narrower than anything else: its own
    subtree, read-only, and nothing above or beside it. That width is not a
    convenience -- the action hands the folder to an editor that will read the
    tree, ``verify_dir`` has to list it, and a grant of only the folder's own
    inode would deny both. The drive it sits on stays unreadable, so a resolution
    that returned the wrong folder cannot become a way to read the disk.

    ``setup_python_project`` is granted **nothing**, and that is not an omission.
    Its workspace is the projects directory (``write_root_for_task``), the project
    it creates is inside that workspace, and ``resolve_read_path`` already allows
    the workspace -- so every path it writes, verifies, or hands to the editor is
    readable without a grant, and anything needing one would by definition be
    outside the directory it is allowed to work in.
    """

    home = Path.home()
    chosen = (params or {}).get("path")

    if task_id in ("open_named_file", "open_project_in_vscode"):
        return (Path(chosen).resolve(),) if chosen else ()

    task_roots: dict[str, tuple[Path, ...]] = {
        "open_last_day_pdf": (
            home / "Downloads",
        ),
    }

    return task_roots.get(task_id, ())


#: Matches "this project", "this codebase", "current project", "the current
#: repo", etc. -- a demonstrative or "current" immediately governing a word
#: that names the running application's own source tree. Deliberately does
#: NOT match "a project", "my project", "another project", or "the project in
#: Downloads": none of those says *this one*, and a request naming or implying
#: a different target must fall through to the existing clarification path,
#: not be silently redirected here. "the" before "current" is optional so
#: both "current project" and "the current project" match; nothing is
#: optional in front of "this", since "this" already an unambiguous
#: demonstrative on its own.
_SELF_REFERENTIAL_PROJECT = re.compile(
    r"\b(?:this|(?:the\s+)?current)\s+(?:project|codebase|repo|repository)\b"
)


def _is_self_referential_project_request(request: str) -> bool:
    """Whether ``request`` is asking about *this* running DEIMOS repository.

    Narrow by design (see :data:`_SELF_REFERENTIAL_PROJECT`): this exists so
    "analyze this project" resolves to the repository DEIMOS is running from,
    without making every sentence containing the word "project" do the same.
    """
    return bool(_SELF_REFERENTIAL_PROJECT.search((request or "").lower()))


def general_readable_roots(request: str = "") -> tuple[Path, ...]:
    """Read-only roots granted to Route 2 (general computer-action) requests.

    Deliberately the union of grants this module already makes elsewhere to
    registered tasks, and nothing wider than any of them:

    * :func:`projects_root` -- already the write root ``setup_python_project``
      is given, and already documented there as "never the home folder or a
      drive root";
    * ``Path.home() / "Downloads"`` -- already the read-only grant
      :func:`readable_roots_for_task` gives ``open_last_day_pdf``;
    * :data:`ROOT` -- the directory containing ``main.py`` and
      ``agent_control`` -- granted **only** when ``request`` is
      self-referential (:func:`_is_self_referential_project_request`), so
      "analyze this project" has an actual source tree to look at instead of
      an empty sandbox. Granted read-only, exactly like the other two: this
      function only ever returns entries for ``Policy.readable_roots``, never
      for ``Policy.workspace``, so nothing it returns can become writable.
      ``request`` defaults to "" (no self-reference), which reproduces the
      old two-root behavior exactly for every caller that does not pass one.

    A general request needing anything outside these hits the same
    ``PolicyDenied`` a registered task would. General computer actions write
    only inside :func:`general_write_root`; the roots returned here remain
    read-only. This function grants nothing by
    itself -- ``Policy.readable_roots``, constructed from its return value, is
    the actual gate -- it exists only so the grant is named once and stays
    auditable, the same reason :func:`readable_roots_for_task` is public.
    """
    roots = [projects_root(), Path.home() / "Downloads"]

    # General computer actions may inspect the user's normal Downloads folder
    # and fixed non-system data drives.  Reuse FileMemory's OS-level discovery so
    # redirected Downloads locations and the actual D: volume are handled without
    # hardcoding a second, drifting mechanism.  These are READ grants only.
    try:
        discovered, _ = FileMemory(DEFAULT_STORE).discover_roots()
    except (OSError, ValueError, RuntimeError):
        discovered = []

    for role, path in discovered:
        if role in {"downloads", "drive"} and path.exists():
            roots.append(path)

    # Keep a deterministic fallback for a normal Windows D: volume if root
    # discovery is temporarily unavailable.  Never grant it write access.
    if os.name == "nt":
        d_drive = Path("D:\\")
        if d_drive.exists():
            roots.append(d_drive)

    if _is_self_referential_project_request(request):
        roots.append(ROOT)

    unique: list[Path] = []
    seen: set[str] = set()
    for path in roots:
        try:
            resolved = Path(path).resolve()
        except OSError:
            continue
        marker = str(resolved).casefold()
        if marker not in seen:
            seen.add(marker)
            unique.append(resolved)

    return tuple(unique)


def _remembered(
    request: str,
    policy: Policy,
    use_memory: bool,
) -> dict[str, Any]:
    """Look up remembered file locations for this request.

    Two narrow rules make this safe to hand to a planner:

    * every candidate must survive ``policy.resolve_read_path``, so a stale cache
      can never name a location this run would not be allowed to read anyway;
    * the surviving shortlist is fenced as untrusted data by ``Recall.as_state``.

    The query is the *request* and deliberately not the task goal. Adding the goal
    was tried and measured worse: ``open_last_day_pdf``'s goal says "folder" three
    times, which read as a request for a directory and pushed the PDF out of the
    shortlist entirely. The goal is a procedure description; only the request is
    what someone asked for.

    Returns ``{}`` -- meaning "nothing remembered" -- whenever memory is off, the
    index is missing, or nothing matched. Recall failure is never a task failure.
    """
    if not use_memory:
        return {}

    from . import memory

    found = memory.recall(request)

    def readable(path: str) -> bool:
        try:
            policy.resolve_read_path(path)
        except PolicyDenied:
            return False
        return True

    return found.keeping(readable).as_state()


def _permissions(policy: Policy) -> dict[str, Any]:
    """Tell the planner which paths it may name. Measured necessity, not polish.

    The prompt has always said "only inside the workspace root given to you" while
    no workspace root was ever given to it, so that clause referred to nothing. Fed
    the goal *"Open the existing file at C:\\Users\\...\\main1.mp4"*, a real planner
    (gpt-oss-120b) read the clause, saw an absolute path it could not place inside
    any workspace it had been shown, and returned ``actions: []`` with the
    reasoning *"there is no permitted way to open an external file"* -- which was
    the correct deduction from what it had been told, and wrong about the machine.

    The asymmetry it could not see is real and is enforced in exactly one place,
    :meth:`Policy.resolve_read_path` versus :meth:`Policy.resolve_write_path`:
    writes are confined to the workspace, reads are allowed from the workspace
    *and* from the roots this task was granted. ``open_file`` only reads.

    This grants nothing. The policy layer is still the only thing that decides;
    naming the grant in the state merely stops the planner from having to guess at
    it -- and a planner that guesses "denied" is as broken as one that guesses
    "allowed", it just fails more quietly.
    """
    return {
        "path_permissions": {
            "note": "Enforced by the policy layer, which overrules you. Writes "
                    "outside write_root are refused. Reads, including open_file, "
                    "are allowed inside write_root and inside readable_paths.",
            "write_root": str(policy.workspace),
            "readable_paths": [str(root) for root in policy.readable_roots],
        }
    }


def _decisions(answers: Sequence[str], approved_action: dict[str, Any] | None = None) -> dict[str, Any]:
    """Tell the planner what the person already decided. Context, never a grant.

    A resumed run is the *same* run: the question it stopped on was about which of
    several real continuations to take, so the answer belongs in the state the
    planner reads, beside the observations, exactly where ``_remembered`` puts a
    remembered path. It arrives as data and is treated as data -- the policy layer
    still decides whether the action the planner picks is allowed, and verification
    still decides whether it worked. An answer that named something out of bounds
    is refused by ``resolve_write_path`` like anything else, which is what keeps
    "the user said so" from being a permission (PART 9 invariants 1-3).

    Empty for the ordinary first attempt, so nothing about the prompt changes for a
    run nobody has been asked anything about.
    """
    if not answers and approved_action is None:
        return {}

    result: dict[str, Any] = {}
    if answers:
        result["user_decisions"] = {
            "note": "Answers this person gave to questions this run already "
                    "asked. Most recent last. They resolve a choice you could "
                    "not make; they do not permit anything the policy layer "
                    "refuses.",
            "answers": [str(answer) for answer in answers],
        }
    if approved_action is not None:
        result["approved_action"] = dict(approved_action)
    return result


def run_agent_task(
    request: str,
    *,
    task_id: str | None = None,
    task_params: dict[str, Any] | None = None,
    task_obj: Task | None = None,
    readable_roots: tuple[Path, ...] | None = None,
    planner: str = "llm",
    max_steps: int = 10,
    fresh_state: bool = True,
    recovery: bool = True,
    keep_workspace: bool = False,
    workspace: Path | None = None,
    condition: str = "structured_hybrid",
    trial: int = 0,
    trace: Trace | None = None,
    use_memory: bool = True,
    interactive: bool = False,
    answers: Sequence[str] = (),
    approved_action: dict[str, Any] | None = None,
    recent_context: dict[str, str] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    browser_backend: Any | None = None,
    fast_route: Any | None = None,
    fast_latency_seconds: float = 0.0,
    runtime_manager: Any | None = None,
) -> AgentResult:
    """Run a task through the single execution pipeline.

    ``on_event`` is forwarded to the trace so an interface can show real progress
    while the run proceeds. It is ignored when ``trace`` is supplied, because the
    caller that owns the trace already owns its callback.

    ``task_id`` and ``task_params`` let a caller that has *already* resolved a
    request say so, instead of encoding the answer back into a string for
    ``resolve_task`` to re-derive. ``request`` then stays the sentence the person
    actually said, which is what the trace and ``_remembered`` want. Passing
    ``task_id`` skips resolution; it does not skip anything else -- the same
    policy, planner, runner, and verifiers follow.

    ``task_obj`` is the general-computer-action route (plan: general-task
    routing): a caller that has already built a ``Task`` -- currently only
    ``GeneralTask``, from ``Session``'s Route 2 -- hands it to us directly
    instead of a task id. When given, resolution, the registered-task gate,
    and ``build_task`` are all skipped entirely; every existing call site,
    which passes ``task_obj=None`` implicitly, is unaffected and takes the
    exact path it always has. ``readable_roots`` is meaningful only alongside
    ``task_obj``: it is what the constructed ``Policy`` grants, in place of
    ``readable_roots_for_task``, which only knows about registered task ids.

    ``interactive`` says whether someone is present to answer a question. False --
    the default, and every benchmark trial -- means a run that finds an ambiguity
    it cannot resolve ends instead of suspending, so no measured number moves
    because this parameter exists. True lets the same run come back as
    ``NEEDS_INPUT`` carrying the question in ``AgentResult.question``.

    ``answers`` carries what the person said when a previous attempt at *this*
    request asked. It reaches the planner as read-only state and nothing else; see
    :func:`_decisions`.
    """

    from .planner import LLMUnavailable, MockPlanner
    from .planner.openai_compat import LLMClient, OpenAICompatPlanner
    from .skills.builtin import build_builtin_registry

    params = dict(task_params or {})

    #: Bound before the guard below so a failure to prepare can still name the
    #: workspace it was trying to create -- the one fact worth reporting about a
    #: run that never started.
    root: Path | None = None

    # Assembling a run touches the filesystem: resolving the workspace path and
    # constructing the Policy creates it. Both can fail on input a caller cannot
    # pre-validate -- a task id that is not spellable as a directory on this
    # platform, a missing or read-only drive -- and until now that failure landed
    # in whatever loop called us. ``main.cmd_chat`` catches only
    # KeyboardInterrupt, so a single bad request ended the session.
    #
    # A run that could not be prepared produced no verdict, which is exactly what
    # UNKNOWN already means in this module (see ``runner._harness_error``, which
    # reports an in-run crash the same way), so it is reported as UNKNOWN with the
    # exception preserved in ``detail`` rather than as a new status. The clause is
    # narrow on purpose: OSError and PolicyDenied are what this assembly is known
    # to raise, and anything else is a bug in the project that stays loud.
    try:
        if task_obj is not None:
            # Everything below this branch, down to policy construction, exists
            # to turn a *name* into a registered Task. None of it applies when
            # the Task already exists -- there is no id to resolve, no
            # registered set to check membership in, and no build_task to call.
            # Every line after this branch closes is the same pipeline a
            # registered task goes through: same Policy, same planner
            # selection, same run_task, same verification.
            task = task_obj
            task_id = task.task_id

            root = (
                Path(workspace).resolve()
                if workspace is not None
                else general_write_root()
            )

            named, through = _describe_target(task)

            approved_fingerprints = frozenset()
            if approved_action is not None:
                from .types import Action as CoreAction
                approved_fingerprints = frozenset({
                    Policy.action_fingerprint(CoreAction(
                        kind=str(approved_action.get("kind", "")),
                        params=dict(approved_action.get("params", {})),
                    ))
                })
            policy = Policy(
                workspace=root,
                readable_roots=tuple(readable_roots or ()),
                confirm_mode=("ask" if interactive else "deny"),
                approved_action_fingerprints=approved_fingerprints,
            )

        else:
            from benchmark.tasks import build_task

            if task_id is None:
                task_id = resolve_task(request)

            if task_id is None:
                return AgentResult(
                    request=request,
                    status=TaskStatus.UNSUPPORTED,
                    detail=(
                        f"no registered workflow matches {request!r}; "
                        f"known: {', '.join(registered_tasks())}"
                    ),
                )

            if task_id not in registered_tasks():
                return AgentResult(
                    request=request,
                    status=TaskStatus.UNSUPPORTED,
                    detail=(
                        f"unknown task {task_id!r}; "
                        f"known: {', '.join(registered_tasks())}"
                    ),
                )

            # A parameterized task with no parameter has nothing to do. Refusing
            # here keeps the empty-path specimen that ``main.py tasks`` builds
            # from ever reaching a planner as a runnable instruction.
            if task_id == "open_named_file" and not params.get("path"):
                return AgentResult(
                    request=request,
                    task_id=task_id,
                    status=TaskStatus.UNSUPPORTED,
                    detail=(
                        "this task needs a filename; say 'open <filename>' and "
                        "I will look the location up"
                    ),
                )

            if task_id == "open_project_in_vscode" and not params.get("path"):
                return AgentResult(
                    request=request,
                    task_id=task_id,
                    status=TaskStatus.UNSUPPORTED,
                    detail=(
                        "this task needs a project folder; say 'open <name> "
                        "project in vscode' and I will look the location up"
                    ),
                )

            if task_id == "setup_python_project" and not params.get("path"):
                return AgentResult(
                    request=request,
                    task_id=task_id,
                    status=TaskStatus.UNSUPPORTED,
                    detail=(
                        "this task needs a name for the new project; say 'set "
                        "up a python project called <name>' and I will choose "
                        "the location"
                    ),
                )

            root = (
                Path(workspace).resolve()
                if workspace is not None
                else write_root_for_task(task_id, params).resolve()
            )

            task = build_task(task_id, **params)

            # Read off the task rather than out of the request string: the task
            # holds the path that was actually resolved, where the request holds
            # what the person said, and the two differ exactly when resolution
            # did something useful.
            named, through = _describe_target(task)

            approved_fingerprints = frozenset()
            if approved_action is not None:
                from .types import Action as CoreAction
                approved_fingerprints = frozenset({
                    Policy.action_fingerprint(CoreAction(
                        kind=str(approved_action.get("kind", "")),
                        params=dict(approved_action.get("params", {})),
                    ))
                })
            policy = Policy(
                workspace=root,
                readable_roots=readable_roots_for_task(task_id, params),
                confirm_mode=("ask" if interactive else "deny"),
                approved_action_fingerprints=approved_fingerprints,
            )

    except (OSError, PolicyDenied) as exc:
        return AgentResult(
            request=request,
            task_id=task_id or "",
            status=TaskStatus.UNKNOWN,
            detail=(
                f"could not prepare the run: {type(exc).__name__}: {exc}"
            ),
            workspace=str(root) if root is not None else "",
        )

    try:
        # A resumed WorkflowStepTask already owns the exact structured action.
        # Do not re-plan from the user's parameter answer; replay the same step
        # through the normal runner so Policy, execution and verification remain
        # authoritative. MockPlanner here is only the deterministic adapter for
        # an already-resolved action, not an approval or execution bypass.
        if fast_route is not None:
            engine = None
        elif hasattr(task_obj, "workflow_step"):
            engine = MockPlanner(reference_plan=[task_obj.workflow_step.action])
        elif planner == "mock":
            engine = MockPlanner(
                reference_plan=task.reference_plan(policy)
            )
        else:
            engine = OpenAICompatPlanner(
                client=LLMClient.from_env()
            )

    except LLMUnavailable as exc:
        return AgentResult(
            request=request,
            task_id=task_id,
            status=TaskStatus.UNAVAILABLE,
            workspace=str(root),
            target=named,
            app=through,
            detail=f"planner unavailable: {exc}",
        )

    config = RunConfig(
        condition=condition,
        max_steps=max_steps,
        fresh_precondition=fresh_state,
        recovery_enabled=recovery,
        budget=RecoveryBudget(),
        interactive=interactive,
        runtime_manager=runtime_manager,
        runtime_task_id=task_id,
    )

    own_trace = trace is None

    active = trace or Trace(
        task_id=task_id,
        condition=condition,
        trial=trial,
        on_event=on_event,
    )

    outcome: RunOutcome | None = None

    # Conversational Session instances own a long-lived browser backend.
    # Direct API callers omit it and receive a temporary legacy backend that
    # this function owns and cleans up.
    skills = build_builtin_registry(
        policy,
        browser_backend=browser_backend,
    )
    owns_browser_backend = browser_backend is None

    try:
        direct_action_factory = None
        if fast_route is not None:
            def direct_action_factory(loop, _route=fast_route, _task=task):
                resolver = getattr(_task, "resolve_current_action", None)
                if callable(resolver):
                    return resolver()
                action = getattr(_task, "action", None)
                if action is None:
                    raise ValueError("fast route did not resolve a browser action")
                return action

        outcome = run_task(
            task,
            engine,
            policy,
            config,
            trace=active,
            skills=skills,
            teardown=not keep_workspace,
            direct_action_factory=direct_action_factory,
            fast_latency_seconds=fast_latency_seconds,
            extra_state={
                **_permissions(policy),
                **_decisions(answers, approved_action),
                **(
                    {
                        "recent_context": {
                            **{
                                key: str(value)
                                for key, value in (recent_context or {}).items()
                                if key in {
                                    "last_verified_directory",
                                    "last_verified_file",
                                    "last_verified_app",
                                    "last_goal",
                                }
                                and value
                            },
                            "note": (
                                "Verified references from the immediately recent "
                                "successful work. Use them to resolve phrases such "
                                "as 'that folder', 'it', and 'there'. They are not "
                                "permissions; Policy still decides every path."
                            ),
                        }
                    }
                    if recent_context
                    else {}
                ),
                **_remembered(
                    request,
                    policy,
                    use_memory,
                ),
                **({"workflow": task_obj.workflow.to_json(), "workflow_step_index": task_obj.workflow_step.index} if hasattr(task_obj, "workflow") and hasattr(task_obj, "workflow_step") else {}),
            },
        )

    except KeyboardInterrupt:
        # ``run_task`` handles an interrupt during the run and returns a cancelled
        # outcome; this covers the window before it is entered -- chiefly
        # ``_remembered``, which reads the whole location index. Nothing was
        # executed, so there is nothing to verify and nothing to claim.
        return AgentResult(
            request=request,
            task_id=task_id,
            status=TaskStatus.CANCELLED,
            workspace=str(root),
            target=named,
            app=through,
            detail="stopped by the user before the run started",
        )

    finally:
        if own_trace:
            active.close()

        # LLMClient keeps a connection pool alive for repeated planner calls.
        # Direct API runs own their planner, so close that pool at the run
        # boundary; long-lived Session/Workflow planners remain alive with
        # their owning session instead.
        if engine is not None and task_obj is None and hasattr(engine, "client"):
            close_client = getattr(engine.client, "close", None)
            if callable(close_client):
                try:
                    close_client()
                except Exception:
                    pass

        if owns_browser_backend:
            try:
                from .skills.builtin import close_builtin_browser_session

                close_builtin_browser_session()
            except Exception:
                # Browser cleanup must never replace the actual task result.
                pass

    if outcome is None:
        return AgentResult(
            request=request,
            task_id=task_id,
            status=TaskStatus.UNKNOWN,
            workspace=str(root),
            target=named,
            app=through,
            detail="runner exited without producing an outcome",
        )

    return _result_from(
        request=request,
        outcome=outcome,
        root=root,
        target=named,
        app=through,
    )


def _describe_target(task: Any) -> tuple[str, str]:
    """``(display name, application)`` for a built task, each possibly empty.

    Both are optional parts of the task duck-type, so both are read defensively:
    a task that has neither is not a broken task, it is one whose result cannot be
    phrased in terms of an object, and the response layer already has wording for
    that case. Only the final path component is kept -- see ``AgentResult.target``.
    """
    named = ""
    getter = getattr(task, "target", None)

    if callable(getter):
        try:
            named = Path(getter()).name
        except (TypeError, ValueError, OSError):
            named = ""

    app = getattr(task, "app", "")

    return named, (app if isinstance(app, str) else "")


def _result_from(
    request: str,
    outcome: RunOutcome,
    root: Path,
    target: str = "",
    app: str = "",
) -> AgentResult:
    """Build the final public result from the runner outcome."""

    completed, failed, unresolved, checks = _derive(outcome)

    status, detail = _status(
        outcome,
        completed,
        failed,
        unresolved,
    )

    counters: dict[str, Any] = (
        (outcome.trace or {}).get("counters", {}) or {}
    )

    return AgentResult(
        request=request,
        task_id=outcome.task_id,
        status=status,
        completed=sorted(completed),
        failed=sorted(failed),
        unresolved=sorted(unresolved),
        checks=checks,
        false_success=outcome.false_success,
        reported_success=outcome.reported_success,
        verified=outcome.verified.value,
        steps_used=outcome.steps_used,
        duration_seconds=outcome.wall_clock_s,
        failure_categories=list(outcome.failure_categories),
        recovery_attempts=int(
            counters.get("recovery_attempts", 0) or 0
        ),
        recovery_successes=int(
            counters.get("recovery_successes", 0) or 0
        ),
        aborted_reason=outcome.aborted_reason,
        detail=detail,
        workspace=str(root),
        target=target,
        app=app,
        #: Only from a suspended run. ``RunOutcome`` already clears it otherwise,
        #: and the guard is repeated here so a question can never outlive the wait
        #: it belongs to and be re-asked after the run finished.
        question=outcome.question if outcome.awaiting else None,
        trace_file=(outcome.trace or {}).get("trace_file"),
        usage=dict(outcome.usage or {}),
        outcome=outcome,
    )