"""What is worth saying about a run, derived from verified fields only.

This module exists so that speech and text share one set of sentences. The
alternative -- letting the TTS layer summarise, or asking the planner for a
closing remark -- would reintroduce the failure this project is built to avoid:
a confident spoken "done" over a run that did not pass verification.

So the phrasing here is a pure function of ``AgentResult``:

* ``status`` decides the sentence, and only ``TaskStatus.SUCCESS`` -- which
  ``api._status`` grants solely on ``Verdict.PASS`` -- may be phrased as
  completion. Every other status names itself.
* ``PARTIAL``, ``UNKNOWN``, ``POLICY_BLOCKED`` and ``CANCELLED`` get their own
  wording. "I could not verify it", "it failed" and "you stopped me" are three
  different claims about the world, and collapsing them into one cheerful
  sentence would be the dishonesty the CLI report already refuses.
* Nothing is invented. Counts come from the check lists; a false success is said
  out loud, because the interesting failure is the one the planner denies.

``Narrator`` is the output layer: it prints (always) and speaks (only when
enabled). It is the single place TTS is wired in, so no action, verifier, or
planner has any knowledge that speech exists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence

from .api import AgentResult, TaskStatus
from .speech.tts import Speaker, TTSConfig

#: Spoken lines stay one sentence long. The detail belongs in the text report,
#: which the user can re-read; a paragraph of synthesised speech cannot be.
MAX_GOAL_WORDS = 14


def _shorten(goal: str, limit: int = MAX_GOAL_WORDS) -> str:
    words = (goal or "").strip().split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]) + ", and so on"


#: Registry key -> the name a person says. The keys mirror
#: ``os_tools.APP_REGISTRY``, which is the closed set of applications any task may
#: launch. Written out rather than imported from there so that the response layer,
#: which every interface loads, does not pull in the action layer's dependencies.
_APP_NAMES = {
    "vscode": "VS Code",
    "chrome": "Chrome",
}

#: Task id -> the past-tense verb for a sentence about a *verified* run of it.
#: Keyed by the whole id and not by a prefix: a task added later has to be given
#: its own verb deliberately, rather than inheriting one that describes a
#: different action. An unlisted task gets the generic sentence, which is never
#: wrong -- only less specific.
_DID = {
    "open_last_day_pdf": "Opened",
    "open_named_file": "Opened",
    "open_project_in_vscode": "Opened",
    "setup_python_project": "Set up",
}

#: Tasks whose target is *created* rather than found. The distinction changes
#: every failure sentence: a directory that is not there means a stale location
#: index for a task that opens things, and a write that did not happen for a task
#: that makes them, and reporting the first for the second would send someone to
#: refresh an index that was never consulted.
_CREATES = frozenset({"setup_python_project"})

#: Name of the interpreter directory a setup run builds, mirroring
#: ``tasks.setup_project.VENV_DIR``. Written out rather than imported for the same
#: reason ``_APP_NAMES`` is: the response layer is loaded by every interface and
#: must not pull in the action layer's dependencies to name a folder.
_VENV_DIR = ".venv"


def _object(result: AgentResult) -> str:
    """``"chess-ai in VS Code"``, ``"main1.mp4"``, or ``""`` when unnameable.

    Both halves come from :class:`~agent_control.api.AgentResult` fields that the
    pipeline filled in from the *built task*, never from the request string: the
    request is what the person said, and the point of resolution is that the two
    can differ.
    """
    if not result.target:
        return ""

    app = _APP_NAMES.get(result.app, "")

    return f"{result.target} in {app}" if app else result.target


def _created_clause(result: AgentResult) -> str:
    """Why a project setup did not finish, named in dependency order.

    Innermost first, for the same reason :func:`_failure_clause` reports the file
    before the window: the directory not existing explains every later failure, so
    naming the editor that never showed it would describe a symptom and send the
    user looking in the wrong place.

    Every branch reads the *names of failed checks*, which is why the file names
    are recoverable at all -- ``tasks.setup_project`` namespaces each file's checks
    by its project-relative path, so ``tests/test_main.py/file_exists`` says both
    what failed and which file it was.
    """
    obj = result.target
    failed = set(result.failed)
    app = _APP_NAMES.get(result.app, "")

    if "dir_exists" in failed:
        return (f"I could not create the {obj} project folder, so nothing else "
                "ran. Check that the projects directory is writable.")

    files = sorted({
        name.rsplit("/", 1)[0]
        for name in failed
        if "/" in name and name.rsplit("/", 1)[1].startswith("file")
    })

    if files:
        listed = ", ".join(files[:3])
        more = f" and {len(files) - 3} more" if len(files) > 3 else ""
        return (f"I created {obj}, but its starter files are missing or empty "
                f"({listed}{more}), so I am not calling it set up.")

    if any(name.startswith(f"{_VENV_DIR}/venv") for name in failed):
        return (f"I created the {obj} files, but its virtual environment was not "
                "built, so I am not calling it set up.")

    packages = sorted({
        name.rsplit(":", 1)[1] for name in failed if "/package:" in name
    })

    if packages:
        return (f"I set up {obj}, but {', '.join(packages)} did not install into "
                "its virtual environment.")

    if "app_process" in failed and app:
        return f"I created {obj}, but {app} could not be started."

    if {"app_window", "viewer_window"} & failed and app:
        return (f"I created {obj}, but {app} never showed it, so I am not "
                "claiming it is open.")

    return ""


def _failure_clause(result: AgentResult) -> str:
    """Why it did not work, in terms of the object -- or ``""`` if that is unknown.

    Built from the *names of the checks that failed*, so a sentence here cannot
    assert a cause verification did not establish. The order is causal rather than
    alphabetical: a project that is not on disk is the reason the editor never
    showed it, so it is reported instead of the window it explains.
    """
    obj = result.target

    if not obj:
        return ""

    # A task that creates its target has an entirely different set of reasons for
    # failing, starting with the fact that a missing directory is its own doing.
    if result.task_id in _CREATES:
        return _created_clause(result)

    failed = set(result.failed)
    app = _APP_NAMES.get(result.app, "")

    if {"dir_exists", "file_exists"} & failed:
        return (f"I could not find {obj} where my location index said it was, "
                "so I did not open anything. Refresh the index and try again.")

    if "app_process" in failed and app:
        return f"I found {obj}, but {app} could not be started."

    if "app_window" in failed and app:
        return (f"I found {obj} and {app} started, but it never drew a window, "
                "so I am not claiming it opened.")

    if "viewer_window" in failed:
        return (f"I found {obj}, but nothing on screen is showing it, so I am "
                "not claiming it opened.")

    return ""


def phrase_accepted(task_id: str, goal: str, *, debug: bool = False) -> str:
    """Said once, before execution: what the agent understood."""
    if debug:
        return f"Understood. Starting {task_id.replace('_', ' ')}: {_shorten(goal)}"
    return f"DEIMOS is understanding your request: {_shorten(goal)}"


def phrase_unsupported() -> str:
    """A request with no registered workflow is a clarification, not a failure."""
    return ("I do not have a workflow for that request, so I have not run "
            "anything. Ask for one of the registered tasks.")


def phrase_empty() -> str:
    """Nothing arrived. Not an error, and certainly not a task."""
    return "I did not get a request, so there is nothing to run."


def phrase_choice(query: str, labels: Sequence[str]) -> str:
    """The one question the agent is allowed to ask: which of these files?

    Short folder labels, never absolute paths. Two paths read aloud differ only
    somewhere in the middle and a listener cannot hold either of them; "Downloads
    or assets" is a question a person can answer in one word. The full paths are
    printed alongside by the caller, so nothing is hidden -- it is just not spoken.

    The sentence names how to answer, because the obvious spoken reply ("that
    one") carries no information when there are two candidates.
    """
    numbered = ", ".join(
        f"{index} {label}" for index, label in enumerate(labels, start=1)
    )
    return (
        f"I know {len(labels)} places with {query}: {numbered}. "
        "Which one -- say the number."
    )


def phrase_no_location(query: str, detail: str = "", action: str = "open") -> str:
    """A request that could not be turned into a task. Not a failure, not a guess.

    ``detail`` comes from the resolvers in :mod:`agent_control.api` and
    distinguishes the causes that matter to the user -- no index yet, nothing
    matching, an index gone stale, a name already taken. Reporting them as one
    "not found" would send someone hunting for a file that is present and merely
    unindexed.

    ``action`` is the verb the request asked for, and it is a parameter because
    the sentence is otherwise a lie for half of them: a request to *create*
    ``test_project`` that could not proceed did not fail to "open" anything, and
    saying so would describe a search that was never run.
    """
    opening = f"I could not {action} {query}"
    return f"{opening}: {detail}" if detail else f"{opening}: I do not know where it is."


def phrase_dropped() -> str:
    """The pending question was cancelled. Nothing ran, and nothing is remembered."""
    return "Dropped that question. I have not opened anything."


#: Why no usable words came out of the microphone, in the user's terms. Keys are
#: the ``stopped_by`` slugs of :class:`~agent_control.speech.mic.Recording` and the
#: ``reason`` slugs of :class:`~agent_control.speech.stt.Transcript`; both layers
#: name their failures rather than returning an empty string, and the distinctions
#: they draw are kept here instead of being flattened into "say that again".
_NOT_HEARD = {
    "no_device": ("I have no microphone available, so I cannot listen. "
                  "Typing still works."),
    "error": "The microphone failed during capture, so I have no audio to send.",
    "user": ("You stopped the recording before I captured anything, so there is "
             "nothing to transcribe."),
    "silent": ("I recorded only background noise. Check that the microphone is "
               "not muted, then try again."),
    "too_short": "That was too short to transcribe. Hold the key a little longer.",
    "malformed_audio": ("The captured audio was not usable, so I did not send it "
                        "for transcription."),
    "unavailable": ("Speech recognition is not configured, so I could not "
                    "transcribe that. This is a setup problem, not something you "
                    "said. Typing still works."),
    "transport": ("I could not reach the speech recognition service, so I have no "
                  "transcript. Typing still works."),
    "empty_transcript": ("The recogniser returned no words for that audio, so I "
                         "am not guessing at a task."),
}


def phrase_not_heard(reason: str) -> str:
    """Why a spoken turn produced no task. Never a guess at what was said.

    An unknown slug is reported as itself rather than smoothed into an apology:
    "I could not turn that into text" plus the reason is honest, whereas "sorry,
    could you repeat that?" would blame the user for a broken API key. The
    underlying error string is deliberately not spoken -- it is printed by the
    caller, because a stack-trace fragment read aloud is noise.
    """
    sentence = _NOT_HEARD.get(reason)
    if sentence:
        return sentence
    named = f" ({reason})" if reason else ""
    return f"I could not turn that audio into text{named}, so I have not run anything."


def phrase_result(result: AgentResult) -> str:
    """One honest sentence about what actually happened.

    The ordering is the point: ``ok`` is consulted before anything else, and no
    branch below it is allowed to use the word "done".
    """
    if result.status is TaskStatus.UNSUPPORTED:
        return phrase_unsupported()

    if result.status is TaskStatus.UNAVAILABLE:
        return ("I could not reach the planner, so the task was never attempted. "
                "This is a configuration problem, not a task failure.")

    verified = f"{len(result.completed)} check{'' if len(result.completed) == 1 else 's'}"

    if result.status is TaskStatus.CANCELLED:
        # The person already knows they pressed Ctrl+C; what they do not know is
        # whether anything was left half-done. Checks named here passed at a
        # checkpoint *before* the interrupt, so they are spoken as progress and
        # never as the task having finished.
        if result.completed:
            return (f"Stopped. {verified} had passed before that, but the run did "
                    "not finish, so I am not claiming it worked.")
        return ("Stopped before anything was verified. I am not claiming anything "
                "about it.")

    if result.ok:
        # The only branch allowed to phrase a run as having worked, and the only
        # one allowed to name the object as opened: both rest on ``ok``, which
        # ``api._status`` grants on ``Verdict.PASS`` alone.
        did, obj = _DID.get(result.task_id, ""), _object(result)

        if did and obj:
            return f"{did} {obj}. Verified {verified}."

        return f"Task complete. Verified {verified}."

    if result.status is TaskStatus.PARTIAL:
        obj = _object(result)
        opening = f"Partly done with {obj}." if obj else "Partly done."
        sentence = (f"{opening} {verified} verified, "
                    f"{len(result.failed)} failed")
        if result.unresolved:
            sentence += f", {len(result.unresolved)} unverifiable"
        return sentence + "."

    if result.status is TaskStatus.POLICY_BLOCKED:
        obj = _object(result)
        creating = result.task_id in _CREATES
        where = f" to {'create' if creating else 'reach'} {obj}" if obj else ""
        return (f"I do not have permission{where}: the policy layer refused the "
                "action, so the task was stopped rather than completed.")

    if result.status is TaskStatus.UNKNOWN:
        obj = _object(result)
        did = "was set up" if result.task_id in _CREATES else "opened"
        if obj:
            return (f"I could not verify that {obj} {did}, so I am not claiming "
                    "it worked. Check the text output.")
        return ("I could not verify the result, so I am not claiming it worked. "
                "Check the text output.")

    tail = (" The planner reported success, but verification disagreed."
            if result.false_success else "")

    # Prefer the sentence that names what went wrong. It is derived from the same
    # failed-check list as the count below, so it is not a softer claim -- just a
    # more useful one.
    because = _failure_clause(result)

    if because:
        return because + tail

    failed = f"{len(result.failed)} check{'' if len(result.failed) == 1 else 's'}"
    return f"Task failed. {failed} did not pass verification.{tail}"


@dataclass(frozen=True)
class PersonaConfig:
    """Presentation-only settings for DEIMOS.

    These values affect wording and nothing else.  In particular, they are never
    passed to the planner, Policy, executor, or verifier.
    """

    name: str = "DEIMOS"
    address: str = "sir"
    dry_humor: bool = True

    @classmethod
    def from_env(cls) -> "PersonaConfig":
        humor = os.getenv("DEIMOS_DRY_HUMOR", "1").strip().lower()
        return cls(
            name=os.getenv("DEIMOS_PERSONA_NAME", "DEIMOS").strip() or "DEIMOS",
            address=os.getenv("DEIMOS_ADDRESS", "sir").strip(),
            dry_humor=humor not in {"0", "false", "no", "off"},
        )


@dataclass(frozen=True)
class DeimosPresentation:
    """Turn verified runtime facts into one consistent assistant voice."""

    config: PersonaConfig = field(default_factory=PersonaConfig.from_env)

    def acknowledgement(self, request: str, goal: str) -> str:
        """A pre-execution acknowledgement that never implies completion."""
        lowered = f"{request} {goal}".lower()
        address = f", {self.config.address}" if self.config.address else ""
        destructive = any(
            word in lowered for word in ("delete", "remove", "erase", "destroy")
        )

        if destructive:
            return f"Understood{address}. I'll check the target and policy before acting."
        if self.config.dry_humor and "python project" in lowered:
            return (
                f"Another project{address}. Ambition remains undefeated. "
                "I'll set it up and verify each required result."
            )
        if "folder" in lowered or "directory" in lowered:
            return f"Understood{address}. I'll handle the folder and verify the result."
        if "file" in lowered or ".txt" in lowered:
            return f"Understood{address}. I'll handle the file and verify the result."
        return f"Understood{address}. I'll handle it and verify the result."

    def result(self, result: AgentResult) -> str:
        """Render completion only from independently verified success."""
        if not result.ok:
            return phrase_result(result)

        address = f", {self.config.address}" if self.config.address else ""
        did, obj = _DID.get(result.task_id, ""), _object(result)
        if did and obj:
            return f"Done{address}. {did} {obj}, and verification passed."
        return f"Done{address}. The requested change is complete and verification passed."

    def progress(self, event: dict) -> str:
        """Render only a state transition that the runner actually emitted."""
        kind = event.get("event", "")
        if kind == "agent_state":
            state = str(event.get("state", ""))
            purpose = str(event.get("purpose", ""))
            action = str(event.get("action", ""))
            params = event.get("params") or {}
            raw_target = (
                params.get("path") or params.get("dest") or params.get("venv")
                or params.get("open_path") or params.get("app") or ""
            )
            try:
                target = Path(str(raw_target)).name if raw_target else ""
            except (OSError, ValueError):
                target = str(raw_target)
            named = f' "{target}"' if target else ""

            if state == "OBSERVING" and purpose in {"initial", "precondition"}:
                return f"{self.config.name} is checking the current workspace ..."
            if state == "PLANNING":
                return f"{self.config.name} is planning the next step ..."
            if state == "ACTING":
                verbs = {
                    "create_dir": "creating the folder",
                    "write_file": "updating the file",
                    "fetch_file": "downloading",
                    "open_file": "opening",
                    "launch_app": "launching",
                    "create_venv": "creating the environment",
                    "install_requirements": "installing the requested packages for",
                    "run_command": "running the requested command for",
                }
                verb = verbs.get(action, "performing the requested action on")
                return f"{self.config.name} is {verb}{named} ..."
            if state == "VERIFYING":
                target_text = named or " the result"
                return f"{self.config.name} is verifying{target_text} ..."

        if kind == "policy_decision" and str(event.get("decision", "")).upper() != "ALLOW":
            return f"{self.config.name} stopped because policy did not allow that action."
        return ""


@dataclass
class Narrator:
    """The output layer: prints everything, speaks the few things worth hearing.

    Every method is safe to call with speech disabled, which is the default, and
    a broken audio backend can only cost silence -- ``Speaker`` swallows its own
    failures and this class never inspects the result to decide control flow.
    """

    speaker: Speaker | None = None
    write: Callable[[str], None] = print
    presentation: DeimosPresentation = field(default_factory=DeimosPresentation)
    #: Sentences handed to speech, in order. Kept for the ``--json`` payload and
    #: for tests that assert what was *not* said.
    spoken: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, *, enabled: bool | None = None, blocking: bool = False,
              write: Callable[[str], None] = print) -> "Narrator":
        config = TTSConfig.from_env(enabled=enabled)
        return cls(speaker=Speaker(config, blocking=blocking), write=write)

    @property
    def speaking(self) -> bool:
        return self.speaker is not None and self.speaker.enabled

    def note(self, message: str) -> None:
        """Print without speaking. Progress detail is read, not narrated."""
        self.write(message)

    def say(self, message: str) -> None:
        """Speak without printing -- used when the text form differs."""
        if not message:
            return
        self.spoken.append(message)
        if self.speaker is not None:
            self.speaker.say(message)

    def reply(self, message: str) -> None:
        """Print *and* speak one sentence: the assistant's turn in a conversation.

        ``note`` and ``say`` exist because progress detail is read rather than
        heard and the accepted-sentence is heard rather than read. A reply is the
        one thing that belongs in both channels, and it must be the same string in
        both -- a chat transcript that disagrees with what was spoken would let
        the two output paths drift, which is the drift this module prevents.
        """
        if not message:
            return
        self.write(message)
        self.say(message)

    def set_speaking(self, enabled: bool) -> bool:
        """Turn speech on or off mid-session. Returns the state now in force.

        Rebinding the config rather than mutating it keeps ``TTSConfig`` frozen,
        and returns False honestly when there is no speaker to enable.
        """
        if self.speaker is None:
            return False
        self.speaker.config = replace(self.speaker.config, enabled=bool(enabled))
        return self.speaker.enabled

    def accepted(self, task_id: str, goal: str, *, debug: bool = False) -> None:
        self.say(phrase_accepted(task_id, goal, debug=debug))

    def finished(self, result: AgentResult) -> None:
        """The report. Text is the full verdict list; speech is one sentence."""
        for line in result.report_lines():
            self.write(line)
        self.say(phrase_result(result))

    def close(self, timeout_s: float = 15.0) -> None:
        if self.speaker is not None:
            self.speaker.close(timeout_s)
