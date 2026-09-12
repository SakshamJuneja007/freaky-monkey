"""Tests for the one join point where typing and speaking become the same thing.

The claim this file has to defend is a negative one: there is no second agent.
Everything else here supports it.

1. **One pipeline.** A typed line and a voice transcript reach the *same*
   ``run_agent_task`` with the same arguments. Nothing about execution varies
   with the input modality.
2. **Normalization is honest.** Empty input runs nothing. A request outside the
   registry is refused by name instead of being coerced into the nearest task.
3. **A bad capture is never submitted.** No microphone, no service, no words --
   each is reported as itself, and none becomes a task.
4. **Modality is free.** Text after voice, voice after text, within one session,
   with no restart and no mode flag.
5. **Status is real.** Progress lines are renderings of trace events that were
   actually emitted; nothing is invented between them.
6. **A question is bound to what it offered.** One clarification -- *"which
   main1.mp4?"* -- is allowed to cross a turn boundary. It is taken and cleared on
   the first line of ``submit``, it expires, it is put at most twice, and the only
   paths it can ever produce are the ones it named. A bare "yes" after it
   authorizes nothing.

``api.run_agent_task`` is patched at the ``session`` module's attribute, since
that is where the name is bound.
"""

from __future__ import annotations

import time

import pytest

from agent_control import session as sess
from agent_control.api import AgentResult, TaskStatus
from agent_control.response import Narrator
from agent_control.session import Capture, Session, UserTask, normalize

REGISTERED = "open_last_day_pdf"


@pytest.fixture
def printed() -> list[str]:
    return []


@pytest.fixture
def chat(printed: list[str]) -> Session:
    """A session that captures its output and never touches the speech backend."""
    return Session(narrator=Narrator(speaker=None, write=printed.append),
                   planner="mock")


@pytest.fixture
def calls(monkeypatch) -> list[dict]:
    """Record every call to the execution entry point, and execute nothing."""
    seen: list[dict] = []

    def fake(request, **kwargs):
        seen.append({"request": request, **kwargs})
        return AgentResult(request=request, task_id=request,
                           status=TaskStatus.SUCCESS, completed=["file_exists"],
                           verified="PASS", detail="verified 1 check(s)")

    monkeypatch.setattr(sess.api, "run_agent_task", fake)
    return seen


# ----------------------------------------------------------------------
# 1. One pipeline
# ----------------------------------------------------------------------

def test_typed_and_spoken_requests_take_the_same_path(chat: Session, calls: list[dict]):
    """The property the whole feature rests on. Same entry point, same arguments;
    only the recorded ``source`` differs, and it differs *after* execution."""
    typed = chat.submit(REGISTERED, source="text")
    spoken = chat.submit(REGISTERED, source="voice")

    assert len(calls) == 2
    assert calls[0]["request"] == calls[1]["request"] == REGISTERED

    # A fresh callback differs by construction, and verified recent context is
    # expected to differ after the first successful turn. Neither changes the
    # typed/voice execution route being asserted here.
    ignore = {"on_event", "recent_context"}
    assert ({k: v for k, v in calls[0].items() if k not in ignore}
            == {k: v for k, v in calls[1].items() if k not in ignore})

    assert typed.task.source == "text"
    assert spoken.task.source == "voice"
    assert typed.reply == spoken.reply


def test_a_transcript_is_submitted_like_a_typed_line(chat: Session, calls: list[dict]):
    """``submit_capture`` is convenience over ``submit``, not a second route."""
    chat.submit_capture(Capture(text=REGISTERED, ok=True, seconds=1.5))

    assert len(calls) == 1
    assert calls[0]["request"] == REGISTERED
    assert chat.history[-1].task.source == "voice"


def test_session_settings_are_what_the_pipeline_is_told(chat: Session,
                                                        calls: list[dict]):
    chat.planner = "mock"
    chat.max_steps = 4
    chat.use_memory = False
    chat.submit(REGISTERED)

    assert calls[0]["planner"] == "mock"
    assert calls[0]["max_steps"] == 4
    assert calls[0]["use_memory"] is False


def test_the_reply_comes_from_the_verified_result(chat: Session, calls: list[dict]):
    """The sentence is a pure function of ``AgentResult``; the session adds no
    wording of its own, so it cannot upgrade a verdict on the way out."""
    turn = chat.submit(REGISTERED)
    assert turn.reply == chat.narrator.presentation.result(turn.result)
    assert turn.ok is True


def test_a_failed_run_is_not_reported_as_done(chat: Session, monkeypatch):
    monkeypatch.setattr(
        sess.api, "run_agent_task",
        lambda request, **kw: AgentResult(request=request, task_id=request,
                                         status=TaskStatus.FAILED,
                                         failed=["file_exists"], verified="FAIL"))
    turn = chat.submit(REGISTERED)

    assert turn.ok is False
    assert "complete" not in turn.reply.lower()
    assert "failed" in turn.reply.lower()


def test_an_unverified_run_claims_nothing(chat: Session, monkeypatch):
    monkeypatch.setattr(
        sess.api, "run_agent_task",
        lambda request, **kw: AgentResult(request=request, task_id=request,
                                         status=TaskStatus.UNKNOWN,
                                         verified="UNKNOWN"))
    turn = chat.submit(REGISTERED)

    assert turn.ok is False
    assert "not claiming" in turn.reply


# ----------------------------------------------------------------------
# 2. Normalization
# ----------------------------------------------------------------------

def test_whitespace_and_trailing_punctuation_are_normalized():
    task = normalize("  open   last day pdf.  ")
    assert task.text == "open last day pdf"
    assert task.task_id == REGISTERED
    assert task.accepted is True


def test_a_spoken_sentence_resolves_like_a_typed_id():
    """STT returns capitalised prose with a full stop; the id it names is the
    same one. Case, spacing and punctuation are the only gap this layer closes."""
    assert normalize("Open Last Day PDF.").task_id == REGISTERED
    assert normalize("open-last-day-pdf").task_id == REGISTERED


@pytest.mark.parametrize("blank", ["", "   ", "\n", "...", "?!"])
def test_empty_input_runs_nothing(blank: str, chat: Session, calls: list[dict]):
    turn = chat.submit(blank)

    assert turn.task.status == "empty"
    assert turn.executed is False
    assert turn.result is None
    assert calls == []


def test_messaging_requests_are_routed_to_computer_action(chat: Session, calls: list[dict]):
    """A messaging command must reach GeneralTask, not the conversation LLM."""
    turn = chat.submit("send whatsapp to papa saying test message")

    assert len(calls) == 1
    assert calls[0]["request"] == "send whatsapp to papa saying test message"
    assert calls[0]["task_obj"] is not None
    assert calls[0]["task_id"].startswith("general-")
    assert turn.task.intent is sess.IntentCategory.ACTION


def test_a_request_with_no_capability_behind_it_claims_nothing(
    chat: Session, calls: list[dict], printed: list[str],
):
    """No faked generality. "Book me a flight" names no registered workflow and
    asks for nothing this machine can do, so it is answered as conversation --
    and a conversational turn can never read as an attempt: ``result`` stays
    ``None``, so ``executed`` and ``ok`` are both false however the sentence is
    worded. Nothing was booked and nothing claims to have been.

    The conversation engine is stubbed because what is under test is the
    *routing and the claim*, not the model's wording; a live call would make
    this test an availability check on someone else's service.
    """
    class Stub:
        def reply(self, text, history=None, recent_context=None):
            return "I cannot book flights; I have no booking tool."

    chat._conversation = Stub()

    turn = chat.submit("book me a flight to Lisbon")

    assert turn.task.status == "conversation"
    assert turn.task.task_id == "", "not steered into a task the user did not ask for"
    assert turn.executed is False
    assert turn.result is None
    assert turn.ok is False
    assert calls == [], "nothing was run"


def test_a_near_miss_is_not_snapped_to_a_registered_task(chat: Session,
                                                         calls: list[dict]):
    """"open the last pdf" is one word away from the registered
    ``open_last_day_pdf`` and must not be rounded to it. What it *is* now is a
    general computer action -- a real request about real files -- so it reaches
    the pipeline as a ``GeneralTask`` carrying the sentence itself, never as the
    registered task id.
    """
    turn = chat.submit("open the last pdf")

    assert len(calls) == 1
    assert calls[0]["task_id"] != REGISTERED, "not snapped to the near-miss task"
    assert calls[0]["task_id"].startswith("general-")
    assert calls[0]["task_obj"] is not None, "it went as a GeneralTask"
    assert calls[0]["request"] == "open the last pdf", "the sentence travels intact"
    assert turn.task.task_id == calls[0]["task_id"]


def test_clear_folder_actions_always_route_to_execution(
    chat: Session,
    calls: list[dict],
):
    """A prior turn cannot make a later supported action look conversational."""
    first = chat.submit("create a folder called AlphaTest")
    second = chat.submit("create a folder called ConversationTest")

    assert first.executed is True
    assert second.executed is True
    assert len(calls) == 2
    assert all(call["task_obj"] is not None for call in calls)
    assert all(call["request"].startswith("create a folder") for call in calls)


def test_follow_up_name_correction_routes_to_execution(
    chat: Session,
    calls: list[dict],
):
    chat.submit("create a folder called AlphaTest")
    corrected = chat.submit("no, call it BetaTest instead")

    assert corrected.executed is True
    assert len(calls) == 2
    assert calls[-1]["task_obj"] is not None
    assert calls[-1]["request"] == "no, call it BetaTest instead"


def test_the_raw_input_is_kept_beside_the_normalized_one():
    """When a transcript is wrong, the difference between the two is the evidence."""
    task = normalize("  Open Last Day PDF!  ", source="voice")
    assert task.raw == "  Open Last Day PDF!  "
    assert task.text == "Open Last Day PDF"
    assert task.source == "voice"


# ----------------------------------------------------------------------
# 3. A bad capture is never submitted
# ----------------------------------------------------------------------

@pytest.mark.parametrize("reason,expected", [
    ("no_device", "no microphone"),
    ("silent", "background noise"),
    ("too_short", "too short"),
    ("unavailable", "not configured"),
    ("transport", "could not reach"),
    ("empty_transcript", "no words"),
    ("user", "you stopped the recording"),
])
def test_each_capture_failure_is_reported_as_itself(reason: str, expected: str,
                                                    chat: Session,
                                                    calls: list[dict]):
    turn = chat.submit_capture(Capture(ok=False, reason=reason))

    assert calls == [], "a failed capture must never become a task"
    assert turn.executed is False
    assert expected in turn.reply.lower()


def test_stopping_a_recording_is_not_reported_as_a_fault(chat: Session,
                                                         monkeypatch):
    """``mic.record_utterance`` turns Ctrl+C into ``stopped_by="user"`` and keeps
    whatever it had. Interrupted before the first block, there is no audio -- a
    deliberate stop, which must not read like a broken microphone."""
    from agent_control.speech.mic import Recording
    import agent_control.speech as speech

    monkeypatch.setattr(speech, "record_utterance",
                        lambda **kw: Recording(stopped_by="user"))
    capture = chat.listen()

    assert (capture.ok, capture.reason) == (False, "user")
    reply = chat.submit_capture(capture).reply
    assert "you stopped the recording" in reply.lower()
    assert "(user)" not in reply, "a slug leaked into a sentence read to the user"


def test_an_unknown_capture_failure_still_refuses_to_guess(chat: Session,
                                                           calls: list[dict]):
    turn = chat.submit_capture(Capture(ok=False, reason="something_new"))
    assert calls == []
    assert "something_new" in turn.reply


def test_listen_returns_text_and_executes_nothing(chat: Session, calls: list[dict],
                                                  monkeypatch):
    """STT's whole job. ``listen`` may not run a task even when it succeeds.

    Patched on the ``speech`` package rather than on ``session``, because
    ``listen`` imports the two functions at call time -- which is what keeps the
    runner-free ``voice-test`` path free of a speech import and vice versa.
    """
    import agent_control.speech as speech

    from agent_control.speech.mic import Recording
    from agent_control.speech.stt import Transcript

    monkeypatch.setattr(speech, "record_utterance",
                        lambda **kw: Recording(wav=b"RIFFwave", duration_seconds=2.0,
                                               stopped_by="silence", peak_level=0.3))
    monkeypatch.setattr(speech, "transcribe",
                        lambda audio, **kw: Transcript(text="open last day pdf",
                                                       provider="test",
                                                       duration_seconds=2.0))

    capture = chat.listen()

    assert capture.ok is True
    assert capture.text == "open last day pdf"
    assert calls == [], "listening is not executing"


def test_a_dead_microphone_never_reaches_stt(chat: Session, monkeypatch):
    from agent_control.speech.mic import Recording
    import agent_control.speech as speech

    reached = []
    monkeypatch.setattr(speech, "record_utterance",
                        lambda **kw: Recording(stopped_by="no_device",
                                               error="PortAudio missing"))
    monkeypatch.setattr(speech, "transcribe",
                        lambda audio, **kw: reached.append(audio))

    capture = chat.listen()

    assert capture.ok is False
    assert capture.reason == "no_device"
    assert capture.detail == "PortAudio missing"
    assert reached == [], "no audio, nothing to transcribe"


def test_silence_is_distinguished_from_a_broken_recogniser(chat: Session,
                                                          monkeypatch):
    """Whether the user should check the microphone or the API key is a real
    difference, and collapsing it would send them to the wrong place."""
    from agent_control.speech.mic import Recording
    from agent_control.speech.stt import Transcript
    import agent_control.speech as speech

    monkeypatch.setattr(speech, "transcribe",
                        lambda audio, **kw: Transcript(reason="empty_transcript"))

    monkeypatch.setattr(speech, "record_utterance",
                        lambda **kw: Recording(wav=b"x", duration_seconds=2.0,
                                               peak_level=0.001))
    assert chat.listen().reason == "silent"

    monkeypatch.setattr(speech, "record_utterance",
                        lambda **kw: Recording(wav=b"x", duration_seconds=2.0,
                                               peak_level=0.4))
    assert chat.listen().reason == "empty_transcript"


# ----------------------------------------------------------------------
# 4. Switching modality mid-session
# ----------------------------------------------------------------------

def test_text_voice_text_in_one_session(chat: Session, calls: list[dict]):
    """The literal requirement: no restart, in either direction, and a completed
    turn never closes the session to the other modality."""
    chat.submit(REGISTERED, source="text")
    chat.submit_capture(Capture(text=REGISTERED, ok=True))
    chat.submit(REGISTERED, source="text")
    chat.submit_capture(Capture(ok=False, reason="silent"))
    chat.submit(REGISTERED, source="text")

    assert [turn.task.source for turn in chat.history] == [
        "text", "voice", "text", "voice", "text"]
    assert len(calls) == 4, "only the failed capture ran nothing"


def test_a_refused_turn_does_not_end_the_session(chat: Session, calls: list[dict]):
    chat.submit("")
    chat.submit("something unregistered")
    chat.submit(REGISTERED)

    assert len(calls) == 1
    assert chat.history[-1].ok is True


def test_history_records_what_ran_and_what_did_not(chat: Session, calls: list[dict]):
    chat.submit(REGISTERED)
    chat.submit("")

    assert [turn.executed for turn in chat.history] == [True, False]
    assert chat.history[1].result is None


def test_speech_can_be_toggled_without_restarting(chat: Session):
    """Input modality and output modality are independent: type and listen, or
    speak and read."""
    from agent_control.speech.tts import Speaker, TTSConfig

    chat.narrator.speaker = Speaker(TTSConfig(enabled=False), blocking=True,
                                    backend=lambda text, config=None: None)
    assert chat.speaking is False
    assert chat.set_speaking(True) is True
    assert chat.speaking is True
    assert chat.set_speaking(False) is False


def test_toggling_speech_without_a_speaker_says_so(chat: Session):
    assert chat.narrator.speaker is None
    assert chat.set_speaking(True) is False


# ----------------------------------------------------------------------
# 5. Status lines are real events
# ----------------------------------------------------------------------

def test_status_lines_come_from_emitted_events_only(chat: Session, monkeypatch):
    """Every line shown corresponds to an event the run actually emitted. The
    fake progress this project refuses is impossible here because the session has
    no other source of lines."""
    def emitting(request, **kwargs):
        watch = kwargs["on_event"]
        watch({"event": "agent_state", "state": "OBSERVING",
               "purpose": "initial"})
        watch({"event": "agent_state", "state": "PLANNING"})
        watch({"event": "agent_state", "state": "ACTING",
               "action": "create_dir", "params": {"path": "C:/work/Banana"}})
        watch({"event": "agent_state", "state": "VERIFYING",
               "action": "create_dir", "params": {"path": "C:/work/Banana"}})
        return AgentResult(request=request, task_id=request,
                           status=TaskStatus.SUCCESS, completed=["file_exists"])

    monkeypatch.setattr(sess.api, "run_agent_task", emitting)
    turn = chat.submit(REGISTERED)

    assert turn.status_lines == []


def test_unlisted_events_produce_no_line(chat: Session, monkeypatch):
    def emitting(request, **kwargs):
        for event in ("observation", "note", "recovery_budget", ""):
            kwargs["on_event"]({"event": event})
        kwargs["on_event"]({"event": "policy_decision", "decision": "ALLOW",
                            "action": {"kind": "open_path"}, "reason": "fine"})
        return AgentResult(request=request, task_id=request,
                           status=TaskStatus.SUCCESS)

    monkeypatch.setattr(sess.api, "run_agent_task", emitting)
    assert chat.submit(REGISTERED).status_lines == []


def test_a_refusal_and_a_failure_are_shown(chat: Session, monkeypatch):
    chat.debug = True
    def emitting(request, **kwargs):
        kwargs["on_event"]({"event": "policy_decision", "decision": "DENY",
                            "action": {"kind": "run_process"},
                            "reason": "executable not allowed"})
        kwargs["on_event"]({"event": "action", "result": {
            "ok": False, "action": {"kind": "run_process"},
            "error": "exit 127"}})
        kwargs["on_event"]({"event": "recovery", "failure_class": "TRANSIENT",
                            "decision": "RETRY", "attempt": 1})
        kwargs["on_event"]({"event": "recovery_resolved",
                            "failure_class": "TRANSIENT"})
        return AgentResult(request=request, task_id=request,
                           status=TaskStatus.POLICY_BLOCKED)

    monkeypatch.setattr(sess.api, "run_agent_task", emitting)
    lines = chat.submit(REGISTERED).status_lines

    assert lines[0] == "  policy deny on run_process: executable not allowed"
    assert lines[1] == "  run_process: failed -- exit 127"
    assert lines[2] == "  recovering from TRANSIENT: RETRY (attempt 1)"
    assert lines[3] == "  recovered from TRANSIENT"


def test_status_off_passes_no_callback_at_all(chat: Session, calls: list[dict]):
    """Not a silenced callback -- none. The default text path stays exactly what
    it was before this module existed."""
    chat.show_status = False
    chat.submit(REGISTERED)

    assert calls[0]["on_event"] is None


def test_a_broken_status_display_cannot_fail_a_run(chat: Session, monkeypatch):
    """The trace drops a callback that raises. Asserted here against the real
    ``Trace`` rather than the stub, because this is the guarantee that lets the
    session hand a display function to the control loop at all."""
    from agent_control.trace import Trace

    boom = Trace(task_id="t", condition="c", enabled=False,
                 on_event=lambda record: (_ for _ in ()).throw(RuntimeError("ui")))
    boom.emit("action", result={})
    boom.emit("action", result={})

    assert boom.on_event is None


# ----------------------------------------------------------------------
# 6. One question, bound to the options it offered
# ----------------------------------------------------------------------

DOWNLOADS = "C:\\Users\\x\\Downloads\\main1.mp4"
ASSETS = "D:\\site\\src\\assets\\main1.mp4"


@pytest.fixture
def two_places(monkeypatch) -> sess.api.Resolved:
    """Make "open main1.mp4" ambiguous without touching the real disk index.

    Resolution itself is tested in ``test_open_named.py``; what is under test here
    is what a ``Session`` does with the answer, so the resolver is the seam.
    """
    resolved = sess.api.Resolved(
        query="main1.mp4",
        task_id="open_named_file",
        choices=(
            # ``detail`` mirrors ``memory.Hit.describe()``, which begins with the
            # absolute path -- the reason ``_ask`` prints the label and the detail
            # rather than the label and the path.
            sess.api.Choice(path=DOWNLOADS, label="Downloads",
                            detail=f"{DOWNLOADS}  [15.5 MB, modified 2026-08-26]"),
            sess.api.Choice(path=ASSETS, label="assets",
                            detail=f"{ASSETS}  [27.3 MB, modified 2026-08-23]"),
        ),
    )
    monkeypatch.setattr(sess.api, "resolve_open_request",
                        lambda text, **kw: resolved
                        if sess.api.parse_open_request(text) else None)
    return resolved


def test_an_ambiguous_request_asks_and_runs_nothing(chat: Session, two_places,
                                                    calls: list[dict],
                                                    printed: list[str]):
    """Asking is not running. The turn must not read as an attempt."""
    turn = chat.submit("hey open main1.mp4")

    assert calls == []
    assert turn.executed is False
    assert turn.result is None
    assert "Downloads" in turn.reply and "assets" in turn.reply
    assert "1 " in turn.reply and "2 " in turn.reply
    assert chat.pending is not None


def test_the_spoken_question_names_folders_and_prints_paths(chat: Session,
                                                            two_places,
                                                            printed: list[str]):
    """One speakable sentence in both channels; the absolute paths are printed
    beside it, because a path read aloud is noise."""
    turn = chat.submit("open main1.mp4")

    assert DOWNLOADS not in turn.reply
    assert ASSETS not in turn.reply
    assert any(DOWNLOADS in line for line in printed)
    assert any(ASSETS in line for line in printed)
    assert turn.reply in chat.narrator.spoken


def test_answering_with_a_number_runs_the_one_pipeline(chat: Session, two_places,
                                                       calls: list[dict]):
    """The point of the feature, and the property that matters about it: the
    answer produces an ordinary accepted task on the same entry point a typed
    request uses."""
    chat.submit("open main1.mp4")
    turn = chat.submit("2")

    assert len(calls) == 1
    assert calls[0]["task_id"] == "open_named_file"
    assert calls[0]["task_params"] == {"path": ASSETS}
    assert calls[0]["request"] == "2"
    assert turn.executed is True
    assert turn.task.params == {"path": ASSETS}


def test_a_spoken_number_is_accepted(chat: Session, two_places, calls: list[dict]):
    """STT transcribes speech as words. Without this the voice path -- the whole
    point -- cannot answer its own question."""
    chat.submit("open main1.mp4")
    chat.submit("two", source="voice")

    assert calls[0]["task_params"] == {"path": ASSETS}


def test_a_distinguishing_word_is_accepted(chat: Session, two_places,
                                           calls: list[dict]):
    chat.submit("open main1.mp4")
    chat.submit("downloads")

    assert calls[0]["task_params"] == {"path": DOWNLOADS}


def test_the_question_does_not_survive_the_next_turn(chat: Session, two_places,
                                                     calls: list[dict]):
    """Take-and-clear. Whatever the following turn contains, the question is gone
    afterwards -- so a later stray answer has nothing to select from."""
    chat.submit("open main1.mp4")
    assert chat.pending is not None

    chat.submit("1")
    assert chat.pending is None

    chat.submit("2")  # a number with no question in flight
    assert len(calls) == 1
    assert chat.history[-1].executed is False


def test_a_bare_yes_authorizes_nothing(chat: Session, two_places,
                                       calls: list[dict]):
    """"yes" and "that one" name no option. With two candidates there is nothing
    in those words to select on, so returning the first would be inventing
    consent -- the confirmation rule applied to a choice."""
    chat.submit("open main1.mp4")
    turn = chat.submit("yes")

    assert calls == []
    assert turn.executed is False
    assert "did not name one of them" in turn.reply


@pytest.mark.parametrize("answer", [
    "yes", "sure", "go ahead", "obviously",       # agreement is not a selection
    "that one", "this one", "the one",            # points at nothing
    "9",                                          # a number, but not one offered
])
def test_nothing_selectable_selects_nothing(answer: str, two_places):
    """The user's own words for this were *"then i will say that one"*. With two
    candidates those words carry no information, so the question is put again
    rather than answered by picking the first."""
    pending = sess.Pending(query="main1.mp4", task_id="open_named_file",
                           choices=two_places.choices, asked_at=time.time())

    assert sess.choose(pending, answer) is None


@pytest.mark.parametrize("answer,expected", [
    ("1", DOWNLOADS), ("2", ASSETS),
    ("one", DOWNLOADS), ("two", ASSETS),
    ("second", ASSETS), ("the second one", ASSETS),
    ("the first one I said", DOWNLOADS),   # "first" does name an option
    ("number one", DOWNLOADS),
    ("to", ASSETS),                        # what STT writes when it hears "two"
    ("Downloads", DOWNLOADS), ("assets", ASSETS),
    ("the assets one", ASSETS), ("the one in assets", ASSETS),
    ("site", ASSETS),
])
def test_choose_only_ever_returns_an_offered_path(answer: str, expected: str,
                                                  two_places):
    pending = sess.Pending(query="main1.mp4", task_id="open_named_file",
                           choices=two_places.choices, asked_at=time.time())
    picked = sess.choose(pending, answer)

    assert picked == expected
    assert picked in {choice.path for choice in pending.choices}


def test_a_word_shared_by_both_candidates_selects_nothing(two_places):
    """"main1" is in both paths, so it does not distinguish them."""
    pending = sess.Pending(query="main1.mp4", task_id="open_named_file",
                           choices=two_places.choices, asked_at=time.time())

    assert sess.choose(pending, "main1") is None
    assert sess.choose(pending, "mp4") is None


def test_cancelling_drops_the_question_quietly(chat: Session, two_places,
                                               calls: list[dict]):
    chat.submit("open main1.mp4")
    turn = chat.submit("never mind")

    assert calls == []
    assert chat.pending is None
    assert turn.task.status == "dropped"
    assert "not opened anything" in turn.reply


def test_an_expired_question_is_dropped(chat: Session, two_places,
                                       calls: list[dict]):
    chat.submit("open main1.mp4")
    chat.pending = sess.replace(chat.pending, asked_at=time.time() - 10_000)

    turn = chat.submit("2")

    assert calls == []
    assert turn.executed is False
    assert chat.pending is None


def test_re_asking_does_not_extend_the_window(chat: Session, two_places):
    """The TTL is a bound on how long a stale answer can be accepted. Re-asking
    must not reset it, or an unanswerable exchange could hold the question open
    indefinitely."""
    chat.submit("open main1.mp4")
    first = chat.pending.asked_at

    chat.submit("yes")

    assert chat.pending is not None
    assert chat.pending.asked_at == first
    assert chat.pending.asks == 2


def test_a_question_is_put_at_most_twice(chat: Session, two_places,
                                         calls: list[dict]):
    chat.submit("open main1.mp4")
    chat.submit("yes")
    turn = chat.submit("yes again")

    assert chat.pending is None
    assert calls == []
    assert turn.task.status == "dropped"


def test_a_new_request_is_never_hijacked_by_a_stale_question(chat: Session,
                                                             two_places,
                                                             calls: list[dict]):
    """A pending question loses to a fresh request in both directions: the new
    task runs, and the question is gone rather than answered by it."""
    chat.submit("open main1.mp4")
    turn = chat.submit(REGISTERED)

    assert len(calls) == 1
    assert calls[0]["task_id"] == REGISTERED
    assert calls[0]["task_params"] == {}
    assert turn.executed is True
    assert chat.pending is None


def test_a_second_open_request_replaces_the_question(chat: Session, two_places,
                                                    calls: list[dict]):
    chat.submit("open main1.mp4")
    chat.submit("open main1.mp4")

    assert calls == []
    assert chat.pending is not None
    assert chat.pending.asks == 1


def test_a_single_location_runs_without_asking(chat: Session, calls: list[dict],
                                               monkeypatch):
    monkeypatch.setattr(
        sess.api, "resolve_open_request",
        lambda text, **kw: sess.api.Resolved(
            query="notes.txt", task_id="open_named_file",
            params={"path": "D:\\notes.txt"}),
    )
    turn = chat.submit("open notes.txt")

    assert chat.pending is None
    assert calls[0]["task_params"] == {"path": "D:\\notes.txt"}
    assert turn.executed is True


def test_an_unfindable_file_says_why_and_runs_nothing(chat: Session,
                                                      calls: list[dict],
                                                      monkeypatch):
    monkeypatch.setattr(
        sess.api, "resolve_open_request",
        lambda text, **kw: sess.api.Resolved(
            query="ghost.mp4", detail="the index is out of date"),
    )
    turn = chat.submit("open ghost.mp4")

    assert calls == []
    assert turn.executed is False
    assert "ghost.mp4" in turn.reply
    assert "out of date" in turn.reply


def test_the_request_recorded_is_the_sentence_that_was_said(chat: Session,
                                                            two_places,
                                                            calls: list[dict]):
    """``request`` stays human. The trace should show what the user said, with the
    resolved id and path carried separately rather than encoded back into text."""
    chat.submit("hey open main1.mp4")
    chat.submit("2")

    assert calls[0]["request"] == "2"
    assert calls[0]["task_id"] == "open_named_file"


def test_pending_json_hides_nothing(chat: Session, two_places):
    import json

    chat.submit("open main1.mp4")
    payload = chat.pending.to_json()
    json.dumps(payload)

    assert [choice["path"] for choice in payload["choices"]] == [DOWNLOADS, ASSETS]
    assert payload["asks"] == 1


# ----------------------------------------------------------------------
# The chat command surface
# ----------------------------------------------------------------------

def test_a_plain_line_is_a_task_and_a_slash_line_is_not(chat: Session,
                                                        calls: list[dict],
                                                        capsys):
    """The only rule the loop has. ``/voice`` is an input option inside the text
    interface, not a mode: nothing about the session changes when it is used."""
    import main

    assert main._chat_command(chat, "/tasks") is True
    assert calls == [], "a command is not a task"

    chat.submit(REGISTERED)
    assert len(calls) == 1

    assert main._chat_command(chat, "/quit") is False


@pytest.mark.parametrize("command", ["cls", "clear", " CLS "])
def test_clear_commands_are_handled_locally(command: str, monkeypatch):
    import main

    cleared: list[bool] = []
    monkeypatch.setattr(main, "_clear_terminal", lambda: cleared.append(True))

    assert main._chat_local_command(command) is True
    assert cleared == [True]


def test_non_clear_text_is_not_consumed_as_a_local_command(monkeypatch):
    import main

    monkeypatch.setattr(
        main,
        "_clear_terminal",
        lambda: pytest.fail("clear should not run"),
    )
    assert main._chat_local_command("create a folder called clear") is False


def test_normal_chat_hides_internal_task_ids(
    chat: Session,
    calls: list[dict],
    printed: list[str],
):
    chat.submit("create a folder called PrivacyTest")

    output = "\n".join(printed)
    assert calls and calls[0]["task_id"].startswith("general-")
    assert "general-" not in output
    assert "setup_python_project" not in output

    printed.clear()
    chat.submit("set up a Python project called PrivacySetup987")
    assert calls[-1]["task_id"] == "setup_python_project"
    assert "setup_python_project" not in "\n".join(printed)


def test_debug_chat_may_show_internal_task_ids(
    chat: Session,
    calls: list[dict],
    printed: list[str],
):
    chat.debug = True
    chat.submit("create a folder called DebugPrivacyTest")

    assert calls[0]["task_id"] in "\n".join(printed)


def test_supported_action_never_returns_conversation_capability_denial(
    chat: Session,
    calls: list[dict],
):
    class DenyingConversation:
        def reply(self, text, history=None):
            return "I don't have the ability to create folders directly."

    chat._conversation = DenyingConversation()
    turn = chat.submit("create a folder called CapabilityTest")

    assert turn.executed is True
    assert calls and calls[0]["task_obj"] is not None
    assert "don't have the ability" not in turn.reply.lower()


def test_unknown_commands_are_named_rather_than_run(chat: Session,
                                                    calls: list[dict], capsys):
    import main

    assert main._chat_command(chat, "/teleport now") is True
    assert "unknown command /teleport" in capsys.readouterr().out
    assert calls == []


def test_the_status_toggle_changes_the_session(chat: Session, capsys):
    import main

    main._chat_command(chat, "/status off")
    assert chat.show_status is False
    main._chat_command(chat, "/status on")
    assert chat.show_status is True

    main._chat_command(chat, "/status")          # bare form reports, sets nothing
    assert chat.show_status is True

def test_the_planner_toggle_refuses_a_planner_that_does_not_exist(chat: Session,
                                                                  capsys):
    import main

    main._chat_command(chat, "/planner llm")
    assert chat.planner == "llm"
    main._chat_command(chat, "/planner telepathy")
    assert chat.planner == "llm", "an unknown planner is ignored, not invented"


def test_a_voice_command_submits_what_was_heard_without_retyping(chat: Session,
                                                                 calls: list[dict],
                                                                 monkeypatch,
                                                                 capsys):
    """The point of the convenience, and the boundary of it: the transcript is
    echoed first, so a wrong one is visible before anything runs."""
    import main

    monkeypatch.setattr(Session, "listen",
                        lambda self, **kw: Capture(text=REGISTERED, ok=True,
                                                   seconds=2.0,
                                                   latency_seconds=0.5))
    main._chat_command(chat, "/voice")

    assert f'heard: "{REGISTERED}"' in capsys.readouterr().out
    assert len(calls) == 1, "submitted automatically, not left for the user to retype"
    assert chat.history[-1].task.source == "voice"


def test_a_failed_voice_command_leaves_the_session_usable(chat: Session,
                                                          calls: list[dict],
                                                          monkeypatch):
    import main

    monkeypatch.setattr(Session, "listen",
                        lambda self, **kw: Capture(ok=False, reason="no_device",
                                                   detail="PortAudio missing"))
    main._chat_command(chat, "/voice")
    assert calls == []

    chat.submit(REGISTERED)
    assert len(calls) == 1, "typing still works after the microphone did not"


# ----------------------------------------------------------------------
# The architecture requirement, asserted against the source tree
# ----------------------------------------------------------------------

def test_there_is_exactly_one_execution_call_site_per_interface():
    """"There must be exactly one canonical task execution pipeline."

    Every other test here can only show that the paths *currently* agree. This
    one shows they cannot diverge: outside tests, ``run_agent_task`` is called
    from two places, and the interfaces are not among them -- ``cmd_chat`` goes
    through ``Session.submit``, and so does every voice turn. If someone adds a
    second call site, that is the moment a "voice agent" starts to exist, and
    this test is what makes it a decision rather than an accident.
    """
    import ast
    from collections import Counter
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent

    #: Source, not artefacts. ``benchmark/results`` holds thousands of ``.py``
    #: files copied out of trial workspaces; parsing those would make this test
    #: cost a minute and would say nothing about the architecture.
    sources = [root / "main.py"]
    sources += sorted((root / "agent_control").rglob("*.py"))
    sources += sorted((root / "benchmark").glob("*.py"))
    sources += sorted((root / "benchmark" / "baselines").glob("*.py"))
    sources += sorted((root / "benchmark" / "tasks").glob("*.py"))

    def calls_in(tree: ast.AST) -> int:
        """Real call expressions only -- the module docstring draws the pipeline."""
        return sum(
            1 for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (getattr(node.func, "id", None) == "run_agent_task"
                 or getattr(node.func, "attr", None) == "run_agent_task")
        )

    sites = Counter()
    for path in sources:
        found = calls_in(ast.parse(path.read_text(encoding="utf-8")))
        if found:
            sites[path.relative_to(root).as_posix()] = found

    assert dict(sites) == {"main.py": 1, "agent_control/session.py": 1}, (
        f"the set of execution call sites changed: {dict(sites)}. One belongs to "
        "the `run` subcommand and one to Session._run; a third is a second agent."
    )


# ----------------------------------------------------------------------
# Shapes
# ----------------------------------------------------------------------

def test_turn_json_round_trips(chat: Session, calls: list[dict]):
    import json

    payload = chat.submit(REGISTERED).to_json()
    json.dumps(payload, default=str)

    assert payload["task"]["source"] == "text"
    assert payload["ok"] is True
    assert payload["executed"] is True


def test_capture_json_round_trips():
    import json

    json.dumps(Capture(text="x", ok=True, seconds=1.0).to_json())


def test_a_task_with_no_registered_workflow_is_not_accepted():
    """``accepted`` means "there is a workflow to run and we know its name".
    Neither an unregistered request nor a conversational one qualifies -- both
    reach a route that is decided later, and neither may be mistaken for a
    resolved task by anything reading this field.
    """
    assert UserTask(raw="x", text="x", status="unregistered").accepted is False
    assert UserTask(raw="x", text="x", status="conversation").accepted is False
    assert UserTask(raw="x", text="x", task_id="").accepted is False
    assert UserTask(raw="x", text="x", task_id=REGISTERED).accepted is True
