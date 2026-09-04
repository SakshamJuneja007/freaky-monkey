"""Real end-to-end voice smoke test: say a sentence, get a project open in VS Code.

Not a unit test. Nothing here is mocked. A real audio device is recorded, the real
speech-to-text provider is called over the network, the transcript goes through the
same ``Session.submit`` a typed line goes through, VS Code is really launched, and
the verified answer is really spoken aloud. Run it deliberately:

    .venv/Scripts/python.exe scripts/smoke_voice_open_project.py

**Where the voice comes from.** By default the request is spoken by this machine's
own speech synthesiser and captured back through a loopback input ("Stereo Mix"),
which mirrors whatever is being played. That keeps every claim the test makes
honest -- device opened, room calibrated, speech detected, pre-roll kept, WAV
framed, network transcription returned -- while dropping the two things a script
cannot supply: a human voice and a real room. Pass ``--mic`` to supply those
yourself, and the script records the default microphone and waits for you to read
the sentence aloud instead.

**Why the project is created in a separate run.** The scenario names a Hermes
project, and asking to open a folder that is not there can only ever demonstrate
the honest refusal. So the first run creates ``projects/hermes`` through the *text*
half of the same pipeline, refreshes the location index, and stops -- because
creating a project also opens it, and a window that is already showing Hermes
would make the voice run's window check prove nothing. Close that window and run
again for the voice leg. The script reports what the editor was already showing
either way, so the evidence can be read for what it is.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_control import api, memory  # noqa: E402
from agent_control import session as session_module  # noqa: E402
from agent_control.platform_window import get_backend  # noqa: E402
from agent_control.session import Session  # noqa: E402
from agent_control.speech import speak  # noqa: E402
from agent_control.speech.mic import CHANNELS, SAMPLE_RATE  # noqa: E402
from agent_control.speech.tts import TTSConfig  # noqa: E402

#: What a person would say. Spoken, transcribed, and never passed to the agent as
#: a string -- only whatever speech-to-text actually heard is.
SPOKEN_REQUEST = "Open my Hermes project in VS Code."

#: The text-mode request that makes the project the voice leg opens.
SETUP_REQUEST = "Set up a Python project called hermes and open it in VS Code."

PROJECT = "hermes"

#: Substrings that name an input device fed by the system's own output. Ordered:
#: Realtek's name for it first, then the two other common ones.
LOOPBACK_HINTS = ("stereo mix", "what u hear", "loopback")

#: Seconds to wait before speaking the request. Long enough for
#: ``record_utterance`` to finish calibrating (0.4 s) and reach its listening
#: state, so the first syllable arrives while something is listening for it.
SPEAK_AFTER_S = 1.2


def _loopback_device() -> tuple[int, str] | None:
    """A working input device that hears this machine's own output, if there is one.

    Every candidate is *checked*, not just named. Windows enumerates the same
    Stereo Mix under each host API it supports -- four entries on this machine --
    and the ones that are not usable at 16 kHz mono fail here rather than half way
    through a capture.
    """
    import sounddevice as sd

    for index, device in enumerate(sd.query_devices()):
        name = str(device["name"])
        if device["max_input_channels"] < CHANNELS:
            continue
        if not any(hint in name.lower() for hint in LOOPBACK_HINTS):
            continue
        try:
            sd.check_input_settings(
                device=index, channels=CHANNELS,
                samplerate=SAMPLE_RATE, dtype="int16",
            )
        except Exception:  # noqa: BLE001 - any refusal means "not this one"
            continue
        return index, name
    return None


def _editor_windows(needle: str) -> list[str]:
    """Titles of visible windows that already name ``needle``.

    Read *before* the voice leg and printed with the result. If the editor was
    already showing the project, ``viewer_window`` passing afterwards is not
    evidence that this run put it there, and the report should not be read as if
    it were.
    """
    backend = get_backend()
    if not getattr(backend, "available", False):
        return []
    return [
        window.title
        for window in backend.list_windows()
        if window.visible and needle.lower() in window.title.lower()
    ]


def _create_project(target: Path) -> int:
    """Make the project the voice leg will open, through the text pipeline.

    The same ``Session.submit`` the voice leg uses, with ``source="text"`` -- so
    this leg is the text-mode half of the vertical slice, not a shortcut around
    it. It stops the script afterwards on purpose; see the module docstring.
    """
    print(f"[smoke] {target} does not exist yet.")
    print(f"[smoke] creating it in TEXT mode: {SETUP_REQUEST!r}")
    print()

    session = Session.build(speech=False, planner="llm", max_steps=8)
    try:
        turn = session.submit(SETUP_REQUEST, source="text")
    finally:
        session.close()

    print()
    print(f"[smoke] text-mode status : "
          f"{turn.result.status.value if turn.result else 'no result'}")
    print(f"[smoke] project on disk  : {target.exists()}")

    if turn.result is None or not turn.result.ok:
        print("[smoke] the project could not be created, so there is nothing to "
              "open by voice.")
        return 1

    report = memory.refresh()
    print(f"[smoke] index refreshed  : {report.indexed} entries in "
          f"{report.elapsed_s:.1f}s{(' -- ' + report.error) if report.error else ''}")

    print()
    print("[smoke] Hermes exists and VS Code has it open, which is why this run "
          "stops here:")
    print("[smoke] a window already titled after the project would make the voice "
          "run's window")
    print("[smoke] check prove nothing. Close that VS Code window, then run this "
          "script again.")
    return 0


def _say_in_background(text: str) -> threading.Thread:
    """Speak ``text`` through the machine's speakers, a moment from now.

    The synthesiser is the repository's own ``speak`` with speech forced on, so the
    audio the recorder hears was produced by the same code path that reads answers
    back to the user. Failures are printed, never raised: the recorder is already
    running by then and the capture result says plainly enough that nothing arrived.
    """

    def talk() -> None:
        time.sleep(SPEAK_AFTER_S)
        said = speak(text, config=TTSConfig.from_env(enabled=True))
        if not said.ok:
            print(f"  [voice] the request could not be spoken: "
                  f"{said.reason} {said.error or ''}")

    thread = threading.Thread(target=talk, name="smoke-voice", daemon=True)
    thread.start()
    return thread


def _voice_leg(target: Path, *, use_mic: bool, max_seconds: float) -> int:
    """Record, transcribe, submit, verify, speak. The whole voice-mode slice."""
    if use_mic:
        source = "the default microphone -- read the sentence aloud when told to"
    else:
        found = _loopback_device()
        if found is None:
            print("[smoke] no loopback input device is available on this machine.")
            print("[smoke] enable Stereo Mix in Sound settings, or run with --mic "
                  "and speak the")
            print(f"[smoke] sentence yourself: {SPOKEN_REQUEST!r}")
            return 2
        import sounddevice as sd

        index, name = found
        sd.default.device = (index, None)
        source = f"[{index}] {name} (this machine's own output, looped back)"

    before = _editor_windows(PROJECT)

    print(f"[smoke] project on disk  : {target}")
    print(f"[smoke] capturing from   : {source}")
    print(f"[smoke] windows already naming {PROJECT!r}: {before or 'none'}")
    print(f"[smoke] sentence         : {SPOKEN_REQUEST!r}")
    print()

    session = Session.build(speech=True, planner="llm", max_steps=8)
    try:
        if use_mic:
            print(f"[smoke] say this after the prompt: {SPOKEN_REQUEST}")
            talker = None
        else:
            talker = _say_in_background(SPOKEN_REQUEST)

        capture = session.listen(
            max_seconds=max_seconds,
            on_status=lambda message: print(f"  [mic] {message}"),
        )
        if talker is not None:
            talker.join(timeout=20)

        print()
        print(f"[smoke] heard            : {capture.ok}")
        print(f"[smoke] audio seconds    : {capture.seconds:.2f}")
        print(f"[smoke] stt latency      : {capture.latency_seconds:.2f}s")
        print(f"[smoke] transcript       : {capture.text!r}")
        if not capture.ok:
            print(f"[smoke] reason           : {capture.reason} "
                  f"{capture.detail or ''}")
        print()

        # Submitted either way. An unusable capture must reach the agent as a
        # refusal rather than as a task, and watching it get refused is the point
        # of submitting it: ``submit_capture`` is where that guarantee lives.
        turn = session.submit_capture(capture)

        print()
        print(f"[smoke] source recorded  : {turn.task.source}")
        print(f"[smoke] task id          : {turn.task.task_id or 'none'}")
        print(f"[smoke] executed         : {turn.result is not None}")
        print(f"[smoke] reply            : {turn.reply!r}")
        print(f"[smoke] handed to speech : {turn.reply in session.narrator.spoken}")
        print(f"[smoke] same pipeline    : "
              f"{session_module.api.run_agent_task is api.run_agent_task}")

        if turn.result is None:
            print()
            print("[smoke] nothing was executed. That is the correct outcome for a "
                  "capture that")
            print("[smoke] could not be understood, and a failure for one that "
                  "could.")
            return 3 if not capture.ok else 1

        for line in turn.result.report_lines():
            print(f"[smoke] {line}")
        print(f"[smoke] verified         : {turn.result.verified}")
        print(f"[smoke] false success    : {turn.result.false_success}")
        print(f"[smoke] steps used       : {turn.result.steps_used}")
        print(f"[smoke] completed        : {sorted(turn.result.completed)}")
        print(f"[smoke] failed           : {sorted(turn.result.failed)}")
        print(f"[smoke] trace            : {turn.result.trace_file or 'none'}")

        after = _editor_windows(PROJECT)
        print(f"[smoke] windows now naming {PROJECT!r}: {after or 'none'}")
        if before:
            print("[smoke] note: a window already named the project before this "
                  "run, so the")
            print("[smoke] window check confirms the state, not that this run "
                  "caused it.")

        return 0 if turn.result.ok else 1
    finally:
        # Closing flushes the speech queue, which is where the reply is actually
        # pronounced. A process that exited first would print a spoken sentence
        # nobody heard.
        session.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mic", action="store_true",
        help="record the default microphone and read the sentence aloud yourself, "
             "instead of letting the machine speak it into a loopback input",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=0.0,
        help="cap on one capture (default: 15 for loopback, 30 for --mic)",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="print the input devices this machine offers, and exit",
    )
    args = parser.parse_args(argv)

    if args.list_devices:
        import sounddevice as sd

        for index, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                print(f"  [{index:>2}] {device['name']} "
                      f"({device['max_input_channels']} in)")
        return 0

    target = api.projects_root() / PROJECT

    print(f"[smoke] projects root    : {api.projects_root()}")
    if not target.is_dir():
        return _create_project(target)

    return _voice_leg(
        target,
        use_mic=args.mic,
        max_seconds=args.max_seconds or (30.0 if args.mic else 15.0),
    )


if __name__ == "__main__":
    raise SystemExit(main())
