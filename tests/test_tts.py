"""Tests for the speech output layer.

Two properties matter, and both are about restraint rather than about audio:

1. **Speech cannot claim what verification did not.** ``phrase_result`` is a pure
   function of ``AgentResult``, so the words "complete" and "done" are asserted
   to be reachable only from ``TaskStatus.SUCCESS`` -- which ``api._status``
   grants solely on ``Verdict.PASS``. A run the planner called finished and the
   verifier called FAIL must not be spoken as success.
2. **Speech cannot break a run.** The backend is a subprocess against the Windows
   audio stack. Every way that can go wrong -- missing interpreter, timeout,
   nonzero exit, an exception inside the backend itself -- has to end as a silent
   ``Utterance``, never as an exception reaching the runner.

No test here makes a sound. The backend is injected, so the assertions are about
what *would* have been said.
"""

from __future__ import annotations

import subprocess

import pytest

from agent_control.api import AgentResult, TaskStatus
from agent_control.response import Narrator, phrase_accepted, phrase_result
from agent_control.speech import tts
from agent_control.speech.tts import Speaker, TTSConfig, Utterance, speak


def result(status: TaskStatus, **kwargs) -> AgentResult:
    return AgentResult(request="demo", task_id="demo", status=status, **kwargs)


@pytest.fixture
def enabled() -> TTSConfig:
    return TTSConfig(enabled=True)


class Recorder:
    """A backend that records instead of speaking."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def __call__(self, text: str, *, config=None) -> Utterance:
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("audio device exploded")
        return Utterance(text=text, spoken=True, provider="test")


# -- the words are bound to the verdict --------------------------------------

def test_only_a_verified_pass_is_spoken_as_completion() -> None:
    """The property the whole layer exists to protect."""
    spoken = phrase_result(result(TaskStatus.SUCCESS, completed=["dir_exists"]))
    assert "complete" in spoken.lower()
    assert "1 check." in spoken


#: Every status except SUCCESS, derived from the enum rather than listed by hand:
#: a status added later (``CANCELLED`` was) must not be able to slip past the
#: completion guard because nobody remembered to extend a literal list here.
_NOT_SUCCESS = [s for s in TaskStatus if s is not TaskStatus.SUCCESS]


@pytest.mark.parametrize("status", _NOT_SUCCESS, ids=lambda s: s.value)
def test_no_other_status_may_be_spoken_as_completion(status: TaskStatus) -> None:
    spoken = phrase_result(result(status, completed=["a"], failed=["b"])).lower()
    assert "task complete" not in spoken
    assert "done" not in spoken or "partly done" in spoken


def test_a_false_success_is_said_out_loud() -> None:
    """The planner claiming success while verification disagreed is the single
    most important thing to say, so it is said rather than smoothed over."""
    spoken = phrase_result(result(TaskStatus.FAILED, failed=["dir_exists"],
                                  false_success=True, reported_success=True))
    assert "failed" in spoken.lower()
    assert "verification disagreed" in spoken


def test_unverifiable_is_not_phrased_as_failure() -> None:
    """UNKNOWN is a different claim about the world than FAIL, and speech that
    collapses them tells the user something untrue about their machine."""
    spoken = phrase_result(result(TaskStatus.UNKNOWN, unresolved=["dir_exists"]))
    assert "could not verify" in spoken.lower()
    assert "failed" not in spoken.lower()


def test_a_policy_refusal_says_it_was_refused() -> None:
    spoken = phrase_result(result(TaskStatus.POLICY_BLOCKED, failed=["dir_exists"]))
    assert "refused" in spoken.lower()


def test_partial_progress_reports_both_halves() -> None:
    spoken = phrase_result(result(TaskStatus.PARTIAL, completed=["cloned"],
                                  failed=["installed"], unresolved=["ran"]))
    assert "1 check verified" in spoken and "1 failed" in spoken
    assert "1 unverifiable" in spoken


def test_an_unsupported_request_says_nothing_ran() -> None:
    spoken = phrase_result(result(TaskStatus.UNSUPPORTED))
    assert "not run anything" in spoken


def test_a_missing_planner_is_not_reported_as_a_task_failure() -> None:
    spoken = phrase_result(result(TaskStatus.UNAVAILABLE))
    assert "never attempted" in spoken
    assert "configuration" in spoken


def test_a_long_goal_is_shortened_rather_than_read_out() -> None:
    goal = " ".join(f"word{n}" for n in range(60))
    spoken = phrase_accepted("open_last_day_pdf", goal)
    assert spoken.startswith("DEIMOS is understanding your request:")
    assert "open last day pdf" not in spoken
    assert spoken.endswith("and so on")
    assert len(spoken.split()) < 25


def test_debug_accepted_message_may_name_the_internal_route() -> None:
    spoken = phrase_accepted("open_last_day_pdf", "open it", debug=True)
    assert "open last day pdf" in spoken


# -- the layer is off unless asked -------------------------------------------

def test_speech_is_disabled_by_default() -> None:
    """Adding an output layer must not change existing behaviour for anyone who
    did not ask for sound."""
    assert TTSConfig().enabled is False
    assert speak("hello", config=TTSConfig()).reason == "disabled"


def test_a_disabled_speaker_never_reaches_the_backend() -> None:
    recorder = Recorder()
    speaker = Speaker(TTSConfig(enabled=False), blocking=True, backend=recorder)
    speaker.say("anything")
    assert recorder.calls == []


def test_an_explicit_flag_overrides_the_environment(monkeypatch) -> None:
    monkeypatch.setattr(tts, "load_env", lambda *a, **k: None)
    monkeypatch.setenv("TTS_ENABLED", "0")
    assert TTSConfig.from_env().enabled is False
    assert TTSConfig.from_env(enabled=True).enabled is True
    monkeypatch.setenv("TTS_ENABLED", "true")
    assert TTSConfig.from_env().enabled is True
    assert TTSConfig.from_env(enabled=False).enabled is False


def test_the_rate_is_clamped_to_what_sapi_accepts(monkeypatch) -> None:
    monkeypatch.setattr(tts, "load_env", lambda *a, **k: None)
    monkeypatch.setenv("TTS_RATE", "99")
    assert TTSConfig.from_env().rate == tts.MAX_RATE
    monkeypatch.setenv("TTS_RATE", "not a number")
    assert TTSConfig.from_env().rate == 0


# -- a broken backend costs silence, nothing else -----------------------------

def test_a_raising_backend_does_not_propagate(enabled) -> None:
    """The runner must never see an exception from the output layer."""
    speaker = Speaker(enabled, blocking=True, backend=Recorder(fail=True))
    speaker.say("hello")
    assert speaker.said[-1].spoken is False
    assert speaker.said[-1].reason == "backend_error"


def test_a_missing_interpreter_is_reported_not_raised(monkeypatch, enabled) -> None:
    monkeypatch.setattr(tts, "_powershell", lambda: None)
    utterance = speak("hello", config=enabled)
    assert utterance.spoken is False
    assert utterance.reason == "unavailable"


def test_a_timeout_is_its_own_reason(monkeypatch, enabled) -> None:
    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="powershell", timeout=1.0)

    monkeypatch.setattr(tts, "_powershell", lambda: "powershell")
    monkeypatch.setattr(tts, "_invoke", hang)
    assert speak("hello", config=enabled).reason == "timeout"


def test_a_nonzero_exit_carries_the_backend_message(monkeypatch, enabled) -> None:
    def failed(*args, **kwargs):
        return subprocess.CompletedProcess(args=["powershell"], returncode=1,
                                           stdout="", stderr="No audio device\n")

    monkeypatch.setattr(tts, "_powershell", lambda: "powershell")
    monkeypatch.setattr(tts, "_invoke", failed)
    utterance = speak("hello", config=enabled)
    assert utterance.reason == "backend_error"
    assert "No audio device" in (utterance.error or "")


def test_empty_text_is_not_sent_to_the_backend(monkeypatch, enabled) -> None:
    monkeypatch.setattr(tts, "_invoke", lambda *a, **k: pytest.fail("called"))
    assert speak("   ", config=enabled).reason == "empty"


# -- text is data, never PowerShell ------------------------------------------

def test_the_text_is_passed_as_data_and_never_interpolated(monkeypatch,
                                                           enabled) -> None:
    """A filename or README containing PowerShell must be pronounced, not run.
    The script is a constant; the text travels base64 in the child's env."""
    captured: dict = {}

    def capture(script, extra_env, timeout_s):
        captured["script"] = script
        captured["env"] = extra_env
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(tts, "_powershell", lambda: "powershell")
    monkeypatch.setattr(tts, "_invoke", capture)

    payload = "'; Remove-Item -Recurse D:\\; #"
    assert speak(payload, config=enabled).spoken is True
    assert "Remove-Item" not in captured["script"]
    assert payload not in captured["script"]
    assert tts._TEXT_VAR in captured["env"]


def test_overlong_text_is_truncated_with_a_spoken_marker(monkeypatch,
                                                         enabled) -> None:
    monkeypatch.setattr(tts, "_powershell", lambda: "powershell")
    monkeypatch.setattr(tts, "_invoke", lambda *a, **k: subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr=""))
    utterance = speak("x " * 2000, config=enabled)
    assert utterance.truncated is True
    assert utterance.text.endswith("Details are in the text output.")


# -- the narrator prints everything and speaks little ------------------------

def test_the_narrator_prints_the_whole_report_but_speaks_one_sentence() -> None:
    lines: list[str] = []
    recorder = Recorder()
    narrator = Narrator(speaker=Speaker(TTSConfig(enabled=True), blocking=True,
                                        backend=recorder),
                        write=lines.append)
    narrator.finished(result(TaskStatus.SUCCESS, completed=["dir_exists"],
                             detail="verified 1 check(s)",
                             checks=[{"name": "dir_exists", "verdict": "PASS",
                                      "reason": "exists"}]))
    assert any("dir_exists" in line for line in lines)
    assert len(recorder.calls) == 1


def test_notes_are_printed_and_never_spoken() -> None:
    """"Do not narrate every low-level action" is enforced structurally: the
    only method that reaches speech is ``say``."""
    lines: list[str] = []
    recorder = Recorder()
    narrator = Narrator(speaker=Speaker(TTSConfig(enabled=True), blocking=True,
                                        backend=recorder),
                        write=lines.append)
    narrator.note("steps         7")
    assert lines == ["steps         7"]
    assert recorder.calls == []


def test_a_narrator_without_a_speaker_still_prints() -> None:
    lines: list[str] = []
    narrator = Narrator(write=lines.append)
    narrator.accepted("demo", "do the thing")
    narrator.finished(result(TaskStatus.FAILED, failed=["x"], detail="failed"))
    assert lines and narrator.speaking is False
    assert len(narrator.spoken) == 2  # phrased, recorded, not voiced


def test_queued_speech_does_not_block_the_caller() -> None:
    """A two second sentence must not add two seconds to a run, so the default
    speaker hands off to a worker thread and returns."""
    import threading

    released = threading.Event()
    seen: list[str] = []

    def slow(text: str, *, config=None) -> Utterance:
        released.wait(5.0)
        seen.append(text)
        return Utterance(text=text, spoken=True)

    speaker = Speaker(TTSConfig(enabled=True), backend=slow)
    speaker.say("first")
    assert seen == []  # the caller already moved on
    released.set()
    speaker.close(timeout_s=5.0)
    assert seen == ["first"]
