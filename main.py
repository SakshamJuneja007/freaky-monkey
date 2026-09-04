#!/usr/bin/env python3
"""Single entry point for the AI Computer Control Plane.

The experiment itself lives in ``benchmark.harness``; the runtime lives in
``agent_control``. This file adds no logic of its own -- it is a launcher, so
that "how do I run this" has one answer:

    python main.py                        talk to the agent (chat + /voice)
    python main.py doctor                 check config, ping the provider
    python main.py models                 list models the endpoint offers
    python main.py tasks                  list the Group A tasks
    python main.py memory --refresh       index where this machine keeps files
    python main.py chat --no-speak        the same session, silent
    python main.py run create_directory   one task, one trial, live
    python main.py say "hello"            check speech output; runs nothing
    python main.py bench --planner llm    the full experiment -> results table
    python main.py test                   the unit tests

Every command that touches the model needs credentials in ``.env``; copy
``.env.example`` and fill it in. ``bench --planner mock`` and ``test`` do not.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 128 + SIGINT, the shell convention for "the user stopped it". Deliberately not
#: 1: a script that logs exit 1 as a failed task would be recording a verdict
#: nobody reached (plan E phase 2).
EXIT_INTERRUPTED = 130


def _price_defaults() -> tuple[float, float]:
    """Prices from .env, so the cost column does not depend on remembering flags."""
    def read(name: str) -> float:
        try:
            return float(os.environ.get(name, "") or 0.0)
        except ValueError:
            return 0.0
    return read("LLM_PRICE_INPUT_PER_M"), read("LLM_PRICE_OUTPUT_PER_M")


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------

def _mask(secret: str) -> str:
    """Show enough of a key to recognise it, never enough to use it."""
    if not secret:
        return "(unset)"
    return f"{secret[:7]}...{secret[-4:]} ({len(secret)} chars)" if len(secret) > 14 else "(short)"


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check the environment, then make one real call. A ping that never leaves
    the process proves nothing, so this actually spends a token or two."""
    from agent_control.planner.openai_compat import LLMClient, LLMUnavailable, load_env

    load_env()
    print("== configuration ==")
    env_path = ROOT / ".env"
    print(f".env                 {'found' if env_path.exists() else 'MISSING (copy .env.example)'}")
    for name in ("LLM_BASE_URL", "LLM_MODEL", "VLM_BASE_URL", "VLM_MODEL"):
        print(f"{name:<20} {os.environ.get(name) or '(unset)'}")
    for name in ("LLM_API_KEY", "VLM_API_KEY"):
        print(f"{name:<20} {_mask(os.environ.get(name, ''))}")

    print("\n== dependencies ==")
    for module in ("httpx", "psutil", "PIL", "dotenv", "pytest", "playwright"):
        try:
            __import__(module)
            print(f"{module:<20} ok")
        except ImportError:
            optional = module in ("playwright",)
            print(f"{module:<20} {'absent (optional, V2 browser tasks)' if optional else 'MISSING'}")

    from agent_control.policy import is_elevated
    print(f"\n== privileges ==\nelevated             {'YES -- plan S10 refuses this' if is_elevated() else 'no (correct)'}")

    print("\n== speech output ==")
    from agent_control.speech.tts import TTSConfig, voices as tts_voices
    tts = TTSConfig.from_env()
    print(f"TTS_ENABLED          {'on' if tts.enabled else 'off (default; --speak overrides)'}")
    print(f"TTS_VOICE            {tts.voice or '(system default)'}")
    installed = tts_voices()
    print(f"installed voices     {len(installed)}"
          f"{': ' + ', '.join(installed) if installed else ' (SAPI unavailable)'}")

    if args.offline:
        return 0

    print("\n== live planner call ==")
    try:
        client = LLMClient.from_env()
    except LLMUnavailable as exc:
        print(f"unavailable: {exc}")
        return 2

    started = time.time()
    try:
        text, usage, truncated = client.chat_checked(
            [{"role": "user", "content": 'Reply with exactly this JSON: {"ok": true}'}],
            json_mode=True,
        )
    except Exception as exc:  # noqa: BLE001 - doctor reports, never raises
        print(f"FAILED  {type(exc).__name__}: {exc}")
        return 1
    print(f"model                {client.model}")
    print(f"max_tokens           {client.max_tokens}")
    print(f"latency              {time.time() - started:.2f}s")
    print(f"json mode            {'accepted' if client.supports_json_mode else 'rejected (falling back to brace extraction)'}")
    print(f"tokens               in {usage.get('prompt_tokens', 0)} / out {usage.get('completion_tokens', 0)}")
    print(f"hidden reasoning     {usage.get('reasoning_chars', 0)} chars"
          f"{' (reasoning model: it bills against max_tokens)' if usage.get('reasoning_chars') else ''}")
    print(f"finish_reason        {usage.get('finish_reason') or '?'}")
    print(f"reply                {text.strip()[:200]!r}")

    if truncated:
        print(f"\nFAILED: {truncated}")
        return 1
    try:
        json.loads(text.strip())
    except ValueError:
        print("\nFAILED: reply is not valid JSON, and the planner requires JSON every turn.")
        return 1
    print("\nplanner reachable and returning parseable JSON.")

    if args.vision:
        return _check_vision()
    print("\n(vision baseline not checked; add --vision)")
    return 0


def _check_vision() -> int:
    """Prove the baseline's VLM accepts a screenshot -- transport only.

    Deliberately calls ``propose`` and never ``apply``: applying would synthesise
    real clicks and keystrokes into whatever window currently has focus, which on
    a developer's own desktop is not a check, it is an incident. Plan S10 wants
    that loop inside a disposable VM; running the arm is a separate decision.
    """
    from agent_control.planner.openai_compat import LLMClient, LLMUnavailable
    from agent_control.vision_fallback import VisionActor, screenshot

    print("\n== vision baseline (transport only, no input synthesised) ==")
    try:
        vlm = LLMClient.from_env(vision=True)
    except LLMUnavailable as exc:
        print(f"unavailable: {exc}")
        return 2

    shot = screenshot()
    if not shot.ok:
        print(f"screenshot           FAILED: {shot.error}")
        return 1
    print(f"screenshot           {shot.value['width']}x{shot.value['height']}, "
          f"{shot.value['bytes'] // 1024} KiB")
    print(f"model                {vlm.model}")

    started = time.time()
    step = VisionActor(client=vlm).propose("Describe what application is in focus.", shot, [])
    print(f"latency              {time.time() - started:.2f}s")
    if step.error:
        print(f"FAILED: {step.error}")
        return 1
    print(f"proposed action      {step.action}  (NOT applied)")
    print(f"reasoning            {step.reasoning[:160]!r}")
    print("\nvision baseline reachable. It can see; whether it can ground a click "
          "is what the benchmark measures.")
    return 0


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

def cmd_models(args: argparse.Namespace) -> int:
    """List what the endpoint actually serves. Cheaper than guessing a model id."""
    import httpx

    from agent_control.planner.openai_compat import load_env

    load_env()
    base = (os.environ.get("LLM_BASE_URL") or "").rstrip("/")
    key = os.environ.get("LLM_API_KEY") or ""
    if not base:
        print("LLM_BASE_URL unset; fill in .env", file=sys.stderr)
        return 2

    try:
        response = httpx.get(f"{base}/models",
                             headers={"Authorization": f"Bearer {key}"}, timeout=60.0)
        response.raise_for_status()
        ids = sorted(item.get("id", "") for item in response.json().get("data", []))
    except Exception as exc:  # noqa: BLE001
        print(f"failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    needle = (args.filter or "").lower()
    shown = [i for i in ids if needle in i.lower()]
    for model_id in shown:
        print(model_id)
    print(f"\n{len(shown)}/{len(ids)} models", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------

def cmd_tasks(_args: argparse.Namespace) -> int:
    from benchmark.tasks import TASK_IDS, build_task

    for task_id in TASK_IDS:
        task = build_task(task_id)
        print(f"{task.task_id:<22} [{task.bucket}]")
        print(f"  {task.goal.strip()}\n")
    return 0


# --------------------------------------------------------------------------
# memory -- the persistent file-location index
# --------------------------------------------------------------------------

def cmd_memory(args: argparse.Namespace) -> int:
    """Inspect, rebuild, or query the location index. Executes no task.

    Refresh is exposed as an explicit command rather than run on every start-up:
    it walks real disks for several seconds, and a cache that silently rebuilds
    itself is a cache whose cost is invisible. ``run`` uses whatever is on disk.
    """
    from agent_control import memory as mem

    # A one-shot CLI process gains nothing from the shared singleton, and
    # constructing directly is what lets the limit flags actually take effect.
    store = mem.FileMemory(
        mem.DEFAULT_STORE,
        max_depth=args.max_depth or mem.DEFAULT_MAX_DEPTH,
        max_entries=args.max_entries or mem.DEFAULT_MAX_ENTRIES,
        timeout_s=args.timeout or mem.DEFAULT_TIMEOUT_S,
    )

    if args.refresh:
        print(f"scanning (depth <= {store.max_depth}, "
              f"cap {store.max_entries}, timeout {store.timeout_s:.0f}s) ...")
        report = store.refresh()
        for line in report.as_lines():
            print(line)
        if args.json:
            print(json.dumps(report.to_json(), indent=2))
        if not report.ok:
            print(f"\nFAILED: {report.error or 'no entries indexed'}", file=sys.stderr)
            return 1
        if report.truncated:
            print("\nSome roots stopped early; recall still works over what was "
                  "indexed. Raise --max-entries or --timeout to reach further.")

    if args.query:
        found = store.recall(args.query, limit=args.limit)
        print(f"\nquery: {args.query!r}")
        for line in found.as_lines():
            print(f"  {line}")
        if found.stale:
            print(f"  (index is {found.age_s / 3600:.1f}h old; "
                  f"paths may have moved -- python main.py memory --refresh)")
        if args.json:
            print(json.dumps(found.to_json(), indent=2))
        return 0 if found.hits else 1

    if not args.refresh:
        status = store.status()
        print(f"store                {status['store']}")
        print(f"exists               {'yes' if status['exists'] else 'no'}")
        print(f"indexed              {status['indexed']} entries "
              f"({status['kinds'].get('file', 0)} files, "
              f"{status['kinds'].get('dir', 0)} folders)")
        if status["exists"]:
            print(f"age                  {status['age_s'] / 3600:.1f}h"
                  f"{'  STALE' if status['stale'] else ''}")
            for report in status["roots"]:
                note = f"  stopped: {report['stopped']}" if report.get("stopped") else ""
                print(f"  {report['role']:9} {report['root']:<44} "
                      f"{report['indexed']:>7}{note}")
        else:
            print("\nNothing remembered yet. Build the index with:\n"
                  "  python main.py memory --refresh")
        if args.json:
            print(json.dumps(status, indent=2))
    return 0


# --------------------------------------------------------------------------
# chat -- the permanent text interface; voice is one way to fill it in
# --------------------------------------------------------------------------

CHAT_HELP = """\
Type a task and press Enter -- or press Enter on an empty line to speak one.

  <blank line>         speak a task instead of typing it (same as /voice)
  /voice               the same thing, named
  /speak [on|off]      read replies aloud (currently: {speaking})
  /status [on|off]     show live execution status (currently: {status})
  /planner [mock|llm]  which planner to use (currently: {planner})
  /memory <query>      look up remembered file locations; runs nothing
  /tasks               list the registered workflows
  /help                this list
  /quit                leave

Anything that is not a command is treated as a task. Typing and speaking are
interchangeable at any point -- both go through the same execution pipeline."""


def _chat_help(session) -> None:
    print(CHAT_HELP.format(
        speaking="on" if session.speaking else "off",
        status="on" if session.show_status else "off",
        planner=session.planner,
    ))


def _toggle(argument: str, current: bool, apply) -> None:
    """Shared handling for /speak and /status: bare form reports, on|off sets."""
    word = argument.strip().lower()
    if word in ("on", "off"):
        current = apply(word == "on")
    print(f"  {'on' if current else 'off'}")


def _chat_voice(session) -> None:
    """One capture, echoed, then submitted through the ordinary path.

    The echo is not decoration. A wrong transcript that silently becomes a task
    is indistinguishable from the agent misbehaving, so what was heard is shown
    before anything runs -- and it is submitted automatically, because making the
    user retype it would defeat the feature.
    """
    print("  (speak now; recording stops on silence)")
    capture = session.listen(on_status=lambda message: print(f"  mic: {message}"))

    if capture.ok:
        print(f'  heard: "{capture.text}"'
              f"  [{capture.seconds:.1f}s audio, {capture.latency_seconds:.1f}s stt]")
    session.submit_capture(capture)


def _chat_memory(session, query: str) -> None:
    """Query the location index from the chat. Reads only; runs no task."""
    from agent_control import memory as mem

    if not query:
        status = mem.shared().status()
        print(f"  {status['indexed']} locations indexed"
              f"{', STALE' if status.get('stale') else ''}"
              f"{'' if status['exists'] else ' -- run: python main.py memory --refresh'}")
        return

    found = mem.recall(query)
    for line in found.as_lines():
        print(f"  {line}")
    if not found.hits:
        print("  nothing remembered matches that")


def _chat_command(session, line: str) -> bool:
    """Handle one slash command. Returns False when the session should end."""
    word, _, argument = line[1:].partition(" ")
    name = word.strip().lower()

    if name in ("quit", "exit", "q"):
        return False
    if name in ("help", "?", ""):
        _chat_help(session)
    elif name == "voice":
        _chat_voice(session)
    elif name == "speak":
        _toggle(argument, session.speaking, session.set_speaking)
    elif name == "status":
        def apply(value: bool) -> bool:
            session.show_status = value
            return value
        _toggle(argument, session.show_status, apply)
    elif name == "planner":
        choice = argument.strip().lower()
        if choice in ("mock", "llm"):
            session.planner = choice
        print(f"  {session.planner}")
    elif name == "memory":
        _chat_memory(session, argument.strip())
    elif name == "tasks":
        cmd_tasks(argparse.Namespace())
    else:
        print(f"  unknown command /{name}; try /help")
    return True


def cmd_chat(args: argparse.Namespace) -> int:
    """The chatbox: type or speak a task, get a verified answer, ask again.

    A thin loop over ``Session``, which owns the one execution entry point. This
    command deliberately contains no planning, policy, verification or memory
    logic -- if any of that were here, the voice path and the text path would
    already be two agents.

    Text is the permanent interface: ``/voice`` is an input option inside it, not
    a separate mode. Both produce a string, and the string goes through
    ``Session.submit`` either way, so switching between them mid-conversation
    needs no state machine and cannot leave the two paths in disagreement.
    """
    from agent_control.api import registered_tasks
    from agent_control.session import Session

    session = Session.build(
        speech=args.speak,
        planner=args.planner,
        max_steps=args.max_steps,
        keep_workspace=args.keep_workspace,
        use_memory=not args.no_memory,
        show_status=not args.no_status,
    )

    print("AI Computer Control Plane -- chat")
    print(f"registered tasks: {', '.join(registered_tasks())}")
    print(f"speech: {'on' if session.speaking else 'off'}   "
          f"planner: {session.planner}")
    print("Type a task, or press Enter on an empty line to speak one. "
          "/help, /quit.\n")

    try:
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not line:
                # Push-to-talk, with Enter as the button. Still one deliberate
                # gesture per utterance -- nothing listens between them -- so this
                # is the same contract as ``/voice``, minus typing the word.
                if sys.stdin.isatty():
                    _chat_voice(session)
                    print()
                continue
            if line.startswith("/"):
                if not _chat_command(session, line):
                    break
                print()
                continue

            try:
                session.submit(line)
            except KeyboardInterrupt:
                # An interrupt *during a run* no longer arrives here: ``run_task``
                # catches it and the turn comes back as CANCELLED, printed and
                # spoken like any other verdict-free result. This remains the net
                # for the rest of a turn -- resolving the request, reading the
                # location index, narrating -- where there is no run to attribute
                # it to and so nothing to report but the interruption itself.
                print("\n  interrupted. Nothing was run, so nothing is claimed.")
            print()

        if args.json:
            print(json.dumps([turn.to_json() for turn in session.history],
                             indent=2, default=str))
        return 0
    finally:
        session.close()


# --------------------------------------------------------------------------
# run -- one task, one trial
# --------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    """One trial of one task, for iterating without paying for the whole grid.

    This is now a thin printer over ``agent_control.api.run_agent_task``: the
    assembly of workspace, policy, task, planner, config and trace lives there, so
    the CLI, a text loop and the voice layer cannot drift into three different
    execution paths (and, previously, three different definitions of success).

    Speech is layered on top of the same printer rather than pushed into the
    runner: ``Narrator`` prints every line it always printed and additionally
    speaks two sentences (what was understood, what was verified). With
    ``--no-speak``, or with ``TTS_ENABLED`` unset, the two paths are identical.
    """
    from agent_control.api import (TaskStatus, default_workspace, registered_tasks,
                                  run_agent_task, task_goal)
    from agent_control.response import Narrator

    known = registered_tasks()
    if args.task not in known:
        print(f"unknown task {args.task!r}; known: {', '.join(known)}", file=sys.stderr)
        return 2

    workspace = default_workspace(args.task)
    goal = task_goal(args.task)
    print(f"goal: {goal}\nworkspace: {workspace}\n")

    speech = None if args.speak is None else args.speak
    narrator = Narrator.build(enabled=speech)
    narrator.accepted(args.task, goal)

    try:
        result = run_agent_task(
            args.task, planner=args.planner, max_steps=args.max_steps,
            fresh_state=not args.no_fresh_state, recovery=not args.no_recovery,
            keep_workspace=args.keep_workspace, workspace=workspace,
        )
        if result.status is TaskStatus.UNAVAILABLE:
            narrator.say("I could not reach the planner, so nothing was attempted.")
            print(result.detail, file=sys.stderr)
            return 2

        print()
        narrator.finished(result)
        narrator.note(f"\nverified      {result.verified}")
        narrator.note(f"reported      {'yes' if result.reported_success else 'no'}")
        narrator.note(f"false success {'YES' if result.false_success else 'no'}")
        narrator.note(f"steps         {result.steps_used}")
        narrator.note(f"seconds       {result.duration_seconds:.2f}")
        if result.aborted_reason:
            narrator.note(f"aborted       {result.aborted_reason}")
        if result.failure_categories:
            narrator.note(f"failures      {', '.join(result.failure_categories)}")
        if result.recovery_attempts:
            narrator.note(f"recovery      "
                          f"{result.recovery_successes}/{result.recovery_attempts} resolved")
        if result.trace_file:
            narrator.note(f"trace         {result.trace_file}")
        if args.json:
            payload = result.to_json()
            payload["spoken"] = list(narrator.spoken)
            if result.outcome is not None:
                payload["run"] = result.outcome.to_json()
            print(json.dumps(payload, indent=2, default=str))
        if result.status is TaskStatus.CANCELLED:
            return EXIT_INTERRUPTED
        return 0 if result.ok else 1
    finally:
        narrator.close()


# --------------------------------------------------------------------------
# say -- text -> TTS -> speaker. Nothing is planned or executed.
# --------------------------------------------------------------------------

def cmd_say(args: argparse.Namespace) -> int:
    """Check the output half of the voice layer in isolation, like ``voice-test``.

    Deliberately the mirror image of ``voice-test``: it imports no runner, builds
    no Policy, and resolves no task, so a sentence that reads like an instruction
    has nothing to act on. ``speak`` is called blocking here because the
    utterance *is* the result being checked.
    """
    from agent_control.speech.tts import TTSConfig, speak, voices

    if args.list_voices:
        installed = voices()
        for name in installed:
            print(name)
        print(f"\n{len(installed)} voice(s)", file=sys.stderr)
        return 0 if installed else 1

    # ``say`` is an explicit request for sound, so it overrides TTS_ENABLED.
    config = TTSConfig.from_env(enabled=True)
    if args.voice:
        config = replace(config, voice=args.voice)

    print(f"voice                {config.voice or '(system default)'}")
    print(f"rate                 {config.rate}")

    result = speak(args.text, config=config)
    print(f"provider             {result.provider or '(none)'}")
    print(f"latency              {result.latency_seconds:.2f}s")
    if result.truncated:
        print(f"truncated            yes (over {len(args.text)} chars)")
    if args.json:
        print(json.dumps(result.to_json(), indent=2))

    if not result.ok:
        print(f"\nFAILED [{result.reason}]  {result.error}")
        return 1
    print("\nspoken. Output layer only: no planner call, no policy check, no execution.")
    return 0


# --------------------------------------------------------------------------
# voice-test -- microphone -> STT -> stdout. Nothing is executed.
# --------------------------------------------------------------------------

def cmd_voice_test(args: argparse.Namespace) -> int:
    """Prove the input half of the voice layer, in isolation from the agent.

    This command cannot run a task even by accident: it never imports the runner
    and never constructs a Policy, so a transcript that happens to read like an
    instruction has nothing to act on. That separation is the point -- speech
    recognition is worth trusting only after it has been checked on its own.
    """
    from agent_control.speech import record_utterance, transcribe

    if args.from_wav:
        source = Path(args.from_wav)
        if not source.is_file():
            print(f"no such file: {source}", file=sys.stderr)
            return 2
        audio = source.read_bytes()
        print(f"source               {source} ({len(audio) // 1024} KiB)")
    else:
        recording = record_utterance(
            max_seconds=args.max_seconds, silence_seconds=args.silence,
            on_status=lambda message: print(f"  mic: {message}"),
        )
        print(f"\ncaptured             {recording.duration_seconds:.2f}s  "
              f"stopped_by={recording.stopped_by}  peak={recording.peak_level:.4f}")
        if not recording.ok:
            print(f"FAILED               {recording.error}")
            print("\nNothing was sent to STT. No task was executed.")
            return 1
        audio = recording.wav
        if args.keep:
            saved = ROOT / ".sandbox" / f"voice-{time.strftime('%Y%m%d-%H%M%S')}.wav"
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(audio)
            print(f"saved                {saved}")

    transcript = transcribe(audio)
    print(f"provider             {transcript.provider}")
    print(f"audio duration       {transcript.duration_seconds:.2f}s")
    print(f"stt latency          {transcript.latency_seconds:.2f}s")
    print(f"chunks               {transcript.chunks}"
          f"{' (concatenated; a word may be cut at a boundary)' if transcript.chunks > 1 else ''}")
    print(f"confidence           {transcript.confidence if transcript.confidence is not None else 'not reported by provider'}")
    if args.json:
        print(json.dumps(transcript.to_json(), indent=2))

    if not transcript.ok:
        print(f"\nFAILED [{transcript.reason}]  {transcript.error}")
        print("An unusable transcript is never forwarded to the agent as a task.")
        return 1
    print(f"\ntranscript           {transcript.text!r}")
    print("\nInput layer only: no planner call, no policy check, no execution.")
    return 0


# --------------------------------------------------------------------------
# bench / test
# --------------------------------------------------------------------------

def cmd_bench(_args: argparse.Namespace, passthrough: list[str]) -> int:
    """The experiment. Unrecognised flags go straight to ``benchmark.harness``."""
    from agent_control.planner.openai_compat import load_env
    from benchmark.harness import main as harness_main

    load_env()
    price_in, price_out = _price_defaults()
    if price_in and not any(a.startswith("--price-in") for a in passthrough):
        passthrough += ["--price-in", str(price_in)]
    if price_out and not any(a.startswith("--price-out") for a in passthrough):
        passthrough += ["--price-out", str(price_out)]
    return harness_main(passthrough)


def cmd_test(_args: argparse.Namespace, passthrough: list[str]) -> int:
    return subprocess.call([sys.executable, "-m", "pytest", *(passthrough or ["-q"])],
                           cwd=ROOT)


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="AI Computer Control Plane -- state-first computer-use agent.",
        epilog="Start with: python main.py doctor",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    doctor = sub.add_parser("doctor", help="check config and make one live model call")
    doctor.add_argument("--offline", action="store_true", help="skip the live call")
    doctor.add_argument("--vision", action="store_true",
                        help="also check the baseline VLM accepts a screenshot")
    doctor.set_defaults(func=cmd_doctor)

    models = sub.add_parser("models", help="list models the endpoint serves")
    models.add_argument("--filter", default="", help="substring filter, e.g. vision")
    models.set_defaults(func=cmd_models)

    sub.add_parser("tasks", help="list the Group A tasks").set_defaults(func=cmd_tasks)

    memory = sub.add_parser(
        "memory", help="the persistent file-location index: status, refresh, query")
    memory.add_argument("query", nargs="?", default="",
                        help='look up locations, e.g. "internship certificate"')
    memory.add_argument("--refresh", action="store_true",
                        help="rescan the disks and rewrite the index")
    memory.add_argument("--limit", type=int, default=5,
                        help="how many candidates to show (default 5)")
    memory.add_argument("--max-depth", type=int, default=None,
                        help="folder depth ceiling per root")
    memory.add_argument("--max-entries", type=int, default=None,
                        help="hard cap on indexed entries")
    memory.add_argument("--timeout", type=float, default=None,
                        help="seconds allowed per root")
    memory.add_argument("--json", action="store_true", help="dump the raw report")
    memory.set_defaults(func=cmd_memory)

    chat = sub.add_parser(
        "chat", help="type or speak tasks in one session (the main interface)")
    chat.add_argument("--planner", choices=("mock", "llm"), default="llm")
    chat.add_argument("--max-steps", type=int, default=10)
    chat.add_argument("--keep-workspace", action="store_true")
    chat.add_argument("--no-memory", action="store_true",
                      help="ignore the remembered file locations")
    chat.add_argument("--no-status", action="store_true",
                      help="hide live execution status lines")
    chat.add_argument("--json", action="store_true",
                      help="dump every turn as JSON on exit")
    chat.add_argument("--speak", dest="speak", action="store_true", default=None,
                      help="read replies aloud (overrides TTS_ENABLED)")
    chat.add_argument("--no-speak", dest="speak", action="store_false",
                      help="stay silent even if TTS_ENABLED is set")
    chat.set_defaults(func=cmd_chat)

    run = sub.add_parser("run", help="one task, one trial, structured-first")
    run.add_argument("task", help="task id (see: python main.py tasks)")
    run.add_argument("--planner", choices=("mock", "llm"), default="llm")
    run.add_argument("--max-steps", type=int, default=10)
    run.add_argument("--no-recovery", action="store_true", help="H3 ablation")
    run.add_argument("--no-fresh-state", action="store_true", help="H2 ablation")
    run.add_argument("--keep-workspace", action="store_true")
    run.add_argument("--json", action="store_true", help="dump the full outcome")
    run.add_argument("--speak", dest="speak", action="store_true", default=None,
                     help="speak the result (overrides TTS_ENABLED)")
    run.add_argument("--no-speak", dest="speak", action="store_false",
                     help="stay silent even if TTS_ENABLED is set")
    run.set_defaults(func=cmd_run)

    say = sub.add_parser("say", help="text -> TTS -> speaker; executes nothing")
    say.add_argument("text", nargs="?", default="Control plane speech output is working.")
    say.add_argument("--voice", default="", help="voice name (see --list-voices)")
    say.add_argument("--list-voices", action="store_true", help="list installed voices")
    say.add_argument("--json", action="store_true", help="dump the Utterance")
    say.set_defaults(func=cmd_say)

    voice_test = sub.add_parser(
        "voice-test", help="microphone -> STT -> stdout; executes nothing")
    voice_test.add_argument("--from-wav", default="",
                            help="transcribe this WAV instead of recording (16 kHz mono)")
    voice_test.add_argument("--max-seconds", type=float, default=30.0,
                            help="hard recording ceiling (default 30)")
    voice_test.add_argument("--silence", type=float, default=1.2,
                            help="seconds of silence that end an utterance")
    voice_test.add_argument("--keep", action="store_true", help="save the capture as a WAV")
    voice_test.add_argument("--json", action="store_true", help="dump the Transcript")
    voice_test.set_defaults(func=cmd_voice_test)

    bench = sub.add_parser("bench", help="the full experiment; flags pass to benchmark.harness",
                           add_help=False)
    bench.set_defaults(func=cmd_bench, passthrough=True)

    test = sub.add_parser("test", help="run pytest; flags pass through", add_help=False)
    test.set_defaults(func=cmd_test, passthrough=True)
    return parser


#: What a bare ``python main.py`` means. Rewriting argv rather than building a
#: Namespace by hand keeps one definition of the chat defaults -- the subparser's --
#: so this shortcut cannot drift from the explicit command. ``--speak`` is here and
#: not in the subparser default because the shortcut is the *conversational* entry
#: point; ``python main.py chat`` on its own still obeys TTS_ENABLED.
DEFAULT_COMMAND = ["chat", "--speak"]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv:
        argv = list(DEFAULT_COMMAND)

    args, extra = parser.parse_known_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        if getattr(args, "passthrough", False):
            return args.func(args, extra)
        if extra:
            parser.error(f"unrecognised arguments: {' '.join(extra)}")
        return args.func(args)
    except KeyboardInterrupt:
        # The last net. Commands that own a run report CANCELLED themselves; this
        # covers everything else (indexing, doctor, a prompt) where a traceback
        # would be the only output of an ordinary Ctrl+C.
        print("\ninterrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    raise SystemExit(main())
