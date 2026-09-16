#!/usr/bin/env python3

"""Single entry point for DEIMOS, the conversational AI computer agent.

The benchmark experiment lives in ``benchmark.harness``; the agent runtime lives
in ``agent_control``. This file is intentionally a CLI launcher and presentation
layer. It does not decide what a natural-language request means.

The default conversational interface accepts natural-language requests. Depending
on the request, ``Session`` may:

    * answer a purely conversational request without executing anything;
    * resolve and run a registered structured workflow; or
    * construct a GeneralTask for an open-ended computer action and execute it
      through the normal policy, observation, recovery, and verification pipeline.

The CLI remains intentionally thin:

    python main.py
        start the conversational DEIMOS interface

    python main.py doctor
        check configuration and ping the model provider

    python main.py models
        list models offered by the configured endpoint

    python main.py tasks
        list registered benchmark / structured workflows

    python main.py memory --refresh
        refresh the persistent file-location index

    python main.py chat --no-speak
        start the conversational interface without speech output

    python main.py run create_directory
        run one registered task directly for testing

    python main.py say "hello"
        check speech output; executes nothing

    python main.py bench --planner llm
        run the benchmark experiment

    python main.py test
        run the unit tests

Every command that touches the model needs credentials in ``.env``. Copy
``.env.example`` and fill it in. ``bench --planner mock`` and ``test`` do not
require model credentials.
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


#: 128 + SIGINT, the shell convention for "the user stopped it".
EXIT_INTERRUPTED = 130


def _price_defaults() -> tuple[float, float]:
    """Read model prices from ``.env`` for benchmark cost reporting."""

    def read(name: str) -> float:
        try:
            return float(os.environ.get(name, "") or 0.0)
        except ValueError:
            return 0.0

    return (
        read("LLM_PRICE_INPUT_PER_M"),
        read("LLM_PRICE_OUTPUT_PER_M"),
    )


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def _mask(secret: str) -> str:
    """Show enough of a key to recognise it, never enough to use it."""

    if not secret:
        return "(unset)"

    if len(secret) > 14:
        return f"{secret[:7]}...{secret[-4:]} ({len(secret)} chars)"

    return "(short)"


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check configuration, dependencies, privileges, speech, and model access."""

    from agent_control.planner.openai_compat import (
        LLMClient,
        LLMUnavailable,
        load_env,
    )

    load_env()

    print("== configuration ==")

    env_path = ROOT / ".env"

    print(
        f".env                 "
        f"{'found' if env_path.exists() else 'MISSING (copy .env.example)'}"
    )

    for name in (
        "LLM_BASE_URL",
        "LLM_MODEL",
        "VLM_BASE_URL",
        "VLM_MODEL",
    ):
        print(f"{name:<20} {os.environ.get(name) or '(unset)'}")

    for name in (
        "LLM_API_KEY",
        "VLM_API_KEY",
    ):
        print(f"{name:<20} {_mask(os.environ.get(name, ''))}")

    print("\n== dependencies ==")

    for module in (
        "httpx",
        "psutil",
        "PIL",
        "dotenv",
        "pytest",
        "playwright",
        "langgraph",
        "langgraph.checkpoint.sqlite",
    ):
        try:
            __import__(module)
            print(f"{module:<20} ok")

        except ImportError:
            optional = module in ("playwright",)

            print(
                f"{module:<20} "
                f"{'absent (optional, V2 browser tasks)' if optional else 'MISSING'}"
            )

    from agent_control.policy import is_elevated

    print(
        "\n== privileges ==\n"
        f"elevated             "
        f"{'YES -- plan S10 refuses this' if is_elevated() else 'no (correct)'}"
    )

    print("\n== speech output ==")

    from agent_control.speech.tts import (
        TTSConfig,
        voices as tts_voices,
    )

    tts = TTSConfig.from_env()

    print(
        f"TTS_ENABLED          "
        f"{'on' if tts.enabled else 'off (default; --speak overrides)'}"
    )

    print(
        f"TTS_VOICE            "
        f"{tts.voice or '(system default)'}"
    )

    installed = tts_voices()

    print(
        f"installed voices     {len(installed)}"
        f"{': ' + ', '.join(installed) if installed else ' (SAPI unavailable)'}"
    )

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
            [
                {
                    "role": "user",
                    "content": 'Reply with exactly this JSON: {"ok": true}',
                }
            ],
            json_mode=True,
        )

    except Exception as exc:  # noqa: BLE001
        print(f"FAILED  {type(exc).__name__}: {exc}")
        return 1

    print(f"model                {client.model}")
    print(f"max_tokens           {client.max_tokens}")
    print(f"latency              {time.time() - started:.2f}s")

    print(
        f"json mode            "
        f"{'accepted' if client.supports_json_mode else 'rejected (falling back to brace extraction)'}"
    )

    print(
        f"tokens               "
        f"in {usage.get('prompt_tokens', 0)} / "
        f"out {usage.get('completion_tokens', 0)}"
    )

    print(
        f"hidden reasoning     "
        f"{usage.get('reasoning_chars', 0)} chars"
        f"{' (reasoning model: it bills against max_tokens)' if usage.get('reasoning_chars') else ''}"
    )

    print(
        f"finish_reason        "
        f"{usage.get('finish_reason') or '?'}"
    )

    print(f"reply                {text.strip()[:200]!r}")

    if truncated:
        print(f"\nFAILED: {truncated}")
        return 1

    try:
        json.loads(text.strip())

    except ValueError:
        print(
            "\nFAILED: reply is not valid JSON, "
            "and the planner requires JSON every turn."
        )
        return 1

    print("\nplanner reachable and returning parseable JSON.")

    if args.vision:
        return _check_vision()

    print("\n(vision baseline not checked; add --vision)")

    return 0


def _check_vision() -> int:
    """Check whether the configured VLM can receive a screenshot.

    This is transport-only. No proposed computer action is applied.
    """

    from agent_control.planner.openai_compat import (
        LLMClient,
        LLMUnavailable,
    )
    from agent_control.vision_fallback import (
        VisionActor,
        screenshot,
    )

    print(
        "\n== vision baseline "
        "(transport only, no input synthesised) =="
    )

    try:
        vlm = LLMClient.from_env(vision=True)

    except LLMUnavailable as exc:
        print(f"unavailable: {exc}")
        return 2

    shot = screenshot()

    if not shot.ok:
        print(f"screenshot           FAILED: {shot.error}")
        return 1

    print(
        f"screenshot           "
        f"{shot.value['width']}x{shot.value['height']}, "
        f"{shot.value['bytes'] // 1024} KiB"
    )

    print(f"model                {vlm.model}")

    started = time.time()

    step = VisionActor(
        client=vlm,
    ).propose(
        "Describe what application is in focus.",
        shot,
        [],
    )

    print(f"latency              {time.time() - started:.2f}s")

    if step.error:
        print(f"FAILED: {step.error}")
        return 1

    print(f"proposed action      {step.action}  (NOT applied)")
    print(f"reasoning            {step.reasoning[:160]!r}")

    print(
        "\nvision baseline reachable. It can see; whether it can "
        "ground a click is what the benchmark measures."
    )

    return 0


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


def cmd_models(args: argparse.Namespace) -> int:
    """List models served by the configured endpoint."""

    import httpx

    from agent_control.planner.openai_compat import load_env

    load_env()

    base = (
        os.environ.get("LLM_BASE_URL") or ""
    ).rstrip("/")

    key = os.environ.get("LLM_API_KEY") or ""

    if not base:
        print(
            "LLM_BASE_URL unset; fill in .env",
            file=sys.stderr,
        )
        return 2

    try:
        response = httpx.get(
            f"{base}/models",
            headers={
                "Authorization": f"Bearer {key}",
            },
            timeout=60.0,
        )

        response.raise_for_status()

        ids = sorted(
            item.get("id", "")
            for item in response.json().get("data", [])
        )

    except Exception as exc:  # noqa: BLE001
        print(
            f"failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    needle = (args.filter or "").lower()

    shown = [
        model_id
        for model_id in ids
        if needle in model_id.lower()
    ]

    for model_id in shown:
        print(model_id)

    print(
        f"\n{len(shown)}/{len(ids)} models",
        file=sys.stderr,
    )

    return 0


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------


def cmd_tasks(_args: argparse.Namespace) -> int:
    """List registered structured / benchmark workflows."""

    from benchmark.tasks import (
        TASK_IDS,
        build_task,
    )

    for task_id in TASK_IDS:
        task = build_task(task_id)

        print(
            f"{task.task_id:<22} [{task.bucket}]"
        )

        print(
            f"  {task.goal.strip()}\n"
        )

    return 0


# --------------------------------------------------------------------------
# memory
# --------------------------------------------------------------------------


def cmd_memory(args: argparse.Namespace) -> int:
    """Inspect, rebuild, or query the persistent file-location index."""

    from agent_control import memory as mem

    store = mem.FileMemory(
        mem.DEFAULT_STORE,
        max_depth=(
            args.max_depth
            or mem.DEFAULT_MAX_DEPTH
        ),
        max_entries=(
            args.max_entries
            or mem.DEFAULT_MAX_ENTRIES
        ),
        timeout_s=(
            args.timeout
            or mem.DEFAULT_TIMEOUT_S
        ),
    )

    if args.refresh:

        print(
            f"scanning "
            f"(depth <= {store.max_depth}, "
            f"cap {store.max_entries}, "
            f"timeout {store.timeout_s:.0f}s) ..."
        )

        report = store.refresh()

        for line in report.as_lines():
            print(line)

        if args.json:
            print(
                json.dumps(
                    report.to_json(),
                    indent=2,
                )
            )

        if not report.ok:

            print(
                f"\nFAILED: "
                f"{report.error or 'no entries indexed'}",
                file=sys.stderr,
            )

            return 1

        if report.truncated:

            print(
                "\nSome roots stopped early; recall still works over what was "
                "indexed. Raise --max-entries or --timeout to reach further."
            )

    if args.query:

        found = store.recall(
            args.query,
            limit=args.limit,
        )

        print(
            f"\nquery: {args.query!r}"
        )

        for line in found.as_lines():
            print(f"  {line}")

        if found.stale:

            print(
                f"  (index is {found.age_s / 3600:.1f}h old; "
                f"paths may have moved -- "
                f"python main.py memory --refresh)"
            )

        if args.json:

            print(
                json.dumps(
                    found.to_json(),
                    indent=2,
                )
            )

        return 0 if found.hits else 1

    if not args.refresh:

        status = store.status()

        print(
            f"store                {status['store']}"
        )

        print(
            f"exists               "
            f"{'yes' if status['exists'] else 'no'}"
        )

        print(
            f"indexed              "
            f"{status['indexed']} entries "
            f"({status['kinds'].get('file', 0)} files, "
            f"{status['kinds'].get('dir', 0)} folders)"
        )

        if status["exists"]:

            print(
                f"age                  "
                f"{status['age_s'] / 3600:.1f}h"
                f"{'  STALE' if status['stale'] else ''}"
            )

            for report in status["roots"]:

                note = (
                    f"  stopped: {report['stopped']}"
                    if report.get("stopped")
                    else ""
                )

                print(
                    f"  {report['role']:9} "
                    f"{report['root']:<44} "
                    f"{report['indexed']:>7}"
                    f"{note}"
                )

        else:

            print(
                "\nNothing remembered yet. Build the index with:\n"
                "  python main.py memory --refresh"
            )

        if args.json:

            print(
                json.dumps(
                    status,
                    indent=2,
                )
            )

    return 0


# --------------------------------------------------------------------------
# chat
# --------------------------------------------------------------------------


CHAT_HELP = """\
Type a message and press Enter -- or press Enter on an empty line to speak one.

  <blank line>         speak instead of typing (same as /voice)

  /voice               capture one spoken message

  /speak [on|off]      read replies aloud (currently: {speaking})

  /status [on|off]     show live execution status (currently: {status})

  /planner [mock|llm]  select the execution planner (currently: {planner})

  /memory              show agent + file-memory status; runs nothing
  /memory recent       show recent structured agent memories
  /memory forget <q>  invalidate matching structured memory
  /memory <query>     look up remembered file locations; runs nothing

  /tasks               show current tasks and what they are waiting for
  /resume              resume a task that can safely continue
  /clear-history       remove inactive runtime history
  /cancel              cancel a task when it can be identified safely

  cls | clear          clear the terminal locally; no model call

  /help                show this help

  /quit                leave

Anything that is not a command is sent to DEIMOS as a natural-language request.

Depending on the request, DEIMOS may:

  - answer conversationally without executing anything;
  - resolve a registered structured workflow; or
  - construct a GeneralTask for an open-ended computer action and execute it
    through the normal policy, observation, recovery, and verification pipeline.

Typing and speaking are interchangeable. Both enter the same Session.submit()
pipeline, so request interpretation remains centralized.
"""


def _chat_help(session) -> None:
    """Print interactive chat help."""

    print(
        CHAT_HELP.format(
            speaking=(
                "on"
                if session.speaking
                else "off"
            ),
            status=(
                "on"
                if session.show_status
                else "off"
            ),
            planner=session.planner,
        )
    )


def _toggle(
    argument: str,
    current: bool,
    apply,
) -> None:
    """Shared handling for /speak and /status."""

    word = argument.strip().lower()

    if word in ("on", "off"):
        current = apply(word == "on")

    print(
        f"  {'on' if current else 'off'}"
    )


def _chat_voice(session) -> None:
    """Capture one utterance and submit it through the normal request path."""

    print(
        "  (speak now; recording stops on silence)"
    )

    capture = session.listen(
        on_status=lambda message: print(
            f"  mic: {message}"
        )
    )

    if capture.ok:

        print(
            f'  heard: "{capture.text}"'
            f"  [{capture.seconds:.1f}s audio, "
            f"{capture.latency_seconds:.1f}s stt]"
        )

    submitted = session.submit_capture(capture, background=True)
    if isinstance(submitted, str):
        print("  submitted")


def _chat_memory(
    session,
    query: str,
) -> None:
    """Inspect structured agent memory while preserving legacy file-location lookup."""
    from agent_control import memory as mem
    from agent_control.persistent_memory import PersistentMemory

    raw = (query or "").strip()
    parts = raw.split(None, 1)
    command = parts[0].casefold() if parts else "status"
    argument = parts[1].strip() if len(parts) > 1 else ""

    agent = PersistentMemory()
    try:
        if command in {"status", ""} and not argument:
            status = agent.status()
            file_status = mem.shared().status()
            print(f"  agent memory: {status['active']} active, {status['superseded']} superseded, {status['invalidated']} invalidated")
            print(f"  file memory:  {file_status['indexed']} locations indexed" + (", STALE" if file_status.get("stale") else ""))
            return

        if command == "recent":
            records = agent.recent(limit=8)
            if not records:
                print("  no persistent agent memories")
                return
            for item in records:
                print(f"  [{item.memory_type}] {item.content} ({item.status.lower()})")
            return

        if command == "forget":
            if not argument:
                print("  usage: /memory forget <query>")
                return
            count = agent.invalidate_matching(argument)
            print(f"  invalidated {count} matching agent memor{'y' if count == 1 else 'ies'}")
            return

        # Backward compatibility: /memory <query> remains the file-location
        # command. Structured memory can be inspected explicitly with recent/forget.
        found = mem.recall(raw)
        for line in found.as_lines():
            print(f"  {line}")
        if not found.hits:
            print("  nothing remembered matches that")
    finally:
        agent.close()


def _chat_command(
    session,
    line: str,
) -> bool:
    """Handle one slash command.

    Returns False when the chat session should end.
    """

    word, _, argument = line[1:].partition(" ")

    name = word.strip().lower()

    if name in ("quit", "exit", "q"):
        return False

    if name in ("help", "?", ""):
        _chat_help(session)

    elif name == "voice":
        _chat_voice(session)

    elif name == "speak":

        _toggle(
            argument,
            session.speaking,
            session.set_speaking,
        )

    elif name == "status":

        def apply(value: bool) -> bool:
            session.show_status = value
            return value

        _toggle(
            argument,
            session.show_status,
            apply,
        )

    elif name == "planner":

        choice = argument.strip().lower()

        if choice in ("mock", "llm"):
            session.planner = choice

        print(
            f"  {session.planner}"
        )

    elif name == "memory":

        _chat_memory(
            session,
            argument.strip(),
        )

    elif name == "clear-history":

        session.submit("/clear-history", source="text")

    elif name == "resume":

        session.submit(f"resume {argument.strip()}".strip(), source="text")

    elif name == "tasks":

        groups = session.runtime_tasks_for_display()
        shown = 0
        print("TASKS")
        for title, items in groups.items():
            if not items:
                continue
            print(f"\n{title}")
            for item in items:
                shown += 1
                number = item.get("display_number", shown)
                print(f"  {number} - {item['goal']}")
                state = item.get("state", "")
                friendly = {
                    "CREATED": "Starting", "PLANNING": "Planning", "RUNNING": "Running", "VERIFYING": "Verifying",
                    "WAITING_FOR_APPROVAL": "Waiting for approval", "WAITING_FOR_USER": "Waiting for your input",
                    "WAITING_FOR_HUMAN": "Waiting for your input", "RECOVERY_REQUIRED": "Recovery required",
                    "RECOVERING": "Recovering", "COMPLETED": "Completed", "FAILED": "Failed",
                    "CANCELLED": "Cancelled", "EXPIRED": "Expired",
                }.get(state, "In progress")
                print(f"     {friendly}")
                if item.get("step"):
                    print(f"     {item['step']}")
        if shown == 0:
            print("\nNo active or retained runtime tasks.")

    elif name == "cancel":

        session.submit(f"cancel {argument.strip()}".strip(), source="text")

    else:

        print(
            f"  unknown command /{name}; try /help"
        )

    return True


def _clear_terminal() -> None:
    """Clear the current terminal with the platform's local command."""
    os.system("cls" if os.name == "nt" else "clear")


def _chat_local_command(line: str) -> bool:
    """Handle exact non-slash terminal commands before request routing."""
    if (line or "").strip().lower() not in {"cls", "clear"}:
        return False
    _clear_terminal()
    return True


def cmd_chat(args: argparse.Namespace) -> int:
    """Run the permanent conversational DEIMOS interface.

    Every typed or spoken request enters ``Session.submit`` through the same
    pipeline. ``Session`` decides whether the request is:

    * a conversational request answered without computer execution;
    * a registered or structured request that resolves to an existing workflow;
      or
    * a general computer action represented by ``GeneralTask`` and executed
      through the normal agent pipeline.

    ``main.py`` deliberately performs none of that routing. It owns terminal
    interaction, command dispatch, and presentation only. Request
    interpretation belongs to ``Session`` so the CLI cannot become a second
    agent implementation.
    """

    from agent_control.session import Session

    session = Session.build(
        speech=args.speak,
        planner=args.planner,
        max_steps=args.max_steps,
        keep_workspace=args.keep_workspace,
        use_memory=not args.no_memory,
        show_status=bool(getattr(args, "debug", False)) and not args.no_status,
        debug=getattr(args, "debug", False),
        runtime_persistence_path=ROOT / ".agent_state" / "runtime.sqlite3",
    )

    print(
        "DEIMOS -- Conversational Computer Agent"
    )

    print(
        "Ask a question, describe something you want done, "
        "or request a computer action."
    )

    print(
        f"speech: {'on' if session.speaking else 'off'}   "
        f"planner: {session.planner}"
    )

    print(
        "Type a message, or press Enter on an empty line to speak. "
        "/help, /tasks, /quit.\n"
    )

    try:

        while True:

            try:
                line = input("you> ").strip()

            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not line:

                if sys.stdin.isatty():
                    _chat_voice(session)
                    print()

                continue

            if line.startswith("/"):

                if not _chat_command(
                    session,
                    line,
                ):
                    break

                print()
                continue

            if _chat_local_command(line):
                continue

            try:
                session.submit_background(line)
                print("  submitted")

            except KeyboardInterrupt:

                print(
                    "\n  interrupted. "
                    "Nothing is claimed without a completed result."
                )
            except Exception as exc:
                # Submission failures are presentation errors, not reasons to
                # terminate the interactive session. Background execution itself
                # records its exceptions on the task registry.
                print(f"  could not submit task: {type(exc).__name__}: {exc}")

            print()

        if args.json:

            print(
                json.dumps(
                    [
                        turn.to_json()
                        for turn in session.history
                    ],
                    indent=2,
                    default=str,
                )
            )

        return 0

    finally:
        session.close()


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    """Run one registered task directly.

    ``run`` is intentionally a developer / testing command for explicitly named
    registered tasks. Open-ended natural-language requests belong in the
    conversational ``chat`` interface, where ``Session`` can choose between
    conversation, structured execution, and ``GeneralTask`` execution.
    """

    from agent_control.api import (
        TaskStatus,
        default_workspace,
        registered_tasks,
        run_agent_task,
        task_goal,
    )

    from agent_control.response import Narrator

    known = registered_tasks()

    if args.task not in known:

        print(
            f"unknown task {args.task!r}; "
            f"known: {', '.join(known)}",
            file=sys.stderr,
        )

        return 2

    workspace = default_workspace(
        args.task
    )

    goal = task_goal(
        args.task
    )

    print(
        f"goal: {goal}\n"
        f"workspace: {workspace}\n"
    )

    speech = (
        None
        if args.speak is None
        else args.speak
    )

    narrator = Narrator.build(
        enabled=speech
    )

    narrator.accepted(
        args.task,
        goal,
    )

    try:

        result = run_agent_task(
            args.task,
            planner=args.planner,
            max_steps=args.max_steps,
            fresh_state=not args.no_fresh_state,
            recovery=not args.no_recovery,
            keep_workspace=args.keep_workspace,
            workspace=workspace,
        )

        if result.status is TaskStatus.UNAVAILABLE:

            narrator.say(
                "I could not reach the planner, "
                "so nothing was attempted."
            )

            print(
                result.detail,
                file=sys.stderr,
            )

            return 2

        print()

        narrator.finished(result)

        narrator.note(
            f"\nverified      {result.verified}"
        )

        narrator.note(
            f"reported      "
            f"{'yes' if result.reported_success else 'no'}"
        )

        narrator.note(
            f"false success "
            f"{'YES' if result.false_success else 'no'}"
        )

        narrator.note(
            f"steps         {result.steps_used}"
        )

        narrator.note(
            f"seconds       {result.duration_seconds:.2f}"
        )

        if result.aborted_reason:

            narrator.note(
                f"aborted       "
                f"{result.aborted_reason}"
            )

        if result.failure_categories:

            narrator.note(
                f"failures      "
                f"{', '.join(result.failure_categories)}"
            )

        if result.recovery_attempts:

            narrator.note(
                f"recovery      "
                f"{result.recovery_successes}/"
                f"{result.recovery_attempts} resolved"
            )

        if result.trace_file:

            narrator.note(
                f"trace         "
                f"{result.trace_file}"
            )

        if args.json:

            payload = result.to_json()

            payload["spoken"] = list(
                narrator.spoken
            )

            if result.outcome is not None:

                payload["run"] = (
                    result.outcome.to_json()
                )

            print(
                json.dumps(
                    payload,
                    indent=2,
                    default=str,
                )
            )

        if result.status is TaskStatus.CANCELLED:
            return EXIT_INTERRUPTED

        return 0 if result.ok else 1

    finally:
        narrator.close()


# --------------------------------------------------------------------------
# say
# --------------------------------------------------------------------------


def cmd_say(args: argparse.Namespace) -> int:
    """Test text-to-speech without planning or executing anything."""

    from agent_control.speech.tts import (
        TTSConfig,
        speak,
        voices,
    )

    if args.list_voices:

        installed = voices()

        for name in installed:
            print(name)

        print(
            f"\n{len(installed)} voice(s)",
            file=sys.stderr,
        )

        return 0 if installed else 1

    config = TTSConfig.from_env(
        enabled=True
    )

    if args.voice:

        config = replace(
            config,
            voice=args.voice,
        )

    print(
        f"voice                "
        f"{config.voice or '(system default)'}"
    )

    print(
        f"rate                 "
        f"{config.rate}"
    )

    result = speak(
        args.text,
        config=config,
    )

    print(
        f"provider             "
        f"{result.provider or '(none)'}"
    )

    print(
        f"latency              "
        f"{result.latency_seconds:.2f}s"
    )

    if result.truncated:

        print(
            f"truncated           "
            f"yes (over {len(args.text)} chars)"
        )

    if args.json:

        print(
            json.dumps(
                result.to_json(),
                indent=2,
            )
        )

    if not result.ok:

        print(
            f"\nFAILED [{result.reason}]  "
            f"{result.error}"
        )

        return 1

    print(
        "\nspoken. Output layer only: "
        "no planner call, no policy check, no execution."
    )

    return 0


# --------------------------------------------------------------------------
# voice-test
# --------------------------------------------------------------------------


def cmd_voice_test(
    args: argparse.Namespace,
) -> int:
    """Test microphone and STT without forwarding anything to the agent."""

    from agent_control.speech import (
        record_utterance,
        transcribe,
    )

    if args.from_wav:

        source = Path(
            args.from_wav
        )

        if not source.is_file():

            print(
                f"no such file: {source}",
                file=sys.stderr,
            )

            return 2

        audio = source.read_bytes()

        print(
            f"source               "
            f"{source} "
            f"({len(audio) // 1024} KiB)"
        )

    else:

        recording = record_utterance(
            max_seconds=args.max_seconds,
            silence_seconds=args.silence,
            on_status=lambda message: print(
                f"  mic: {message}"
            ),
        )

        print(
            f"\ncaptured             "
            f"{recording.duration_seconds:.2f}s  "
            f"stopped_by={recording.stopped_by}  "
            f"peak={recording.peak_level:.4f}"
        )

        if not recording.ok:

            print(
                f"FAILED               "
                f"{recording.error}"
            )

            print(
                "\nNothing was sent to STT. "
                "No task was executed."
            )

            return 1

        audio = recording.wav

        if args.keep:

            saved = (
                ROOT
                / ".sandbox"
                / f"voice-{time.strftime('%Y%m%d-%H%M%S')}.wav"
            )

            saved.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            saved.write_bytes(
                audio
            )

            print(
                f"saved                "
                f"{saved}"
            )

    transcript = transcribe(
        audio
    )

    print(
        f"provider             "
        f"{transcript.provider}"
    )

    print(
        f"audio duration       "
        f"{transcript.duration_seconds:.2f}s"
    )

    print(
        f"stt latency          "
        f"{transcript.latency_seconds:.2f}s"
    )

    print(
        f"chunks               "
        f"{transcript.chunks}"
        f"{' (concatenated; a word may be cut at a boundary)' if transcript.chunks > 1 else ''}"
    )

    print(
        f"confidence           "
        f"{transcript.confidence if transcript.confidence is not None else 'not reported by provider'}"
    )

    if args.json:

        print(
            json.dumps(
                transcript.to_json(),
                indent=2,
            )
        )

    if not transcript.ok:

        print(
            f"\nFAILED [{transcript.reason}]  "
            f"{transcript.error}"
        )

        print(
            "An unusable transcript is never "
            "forwarded to the agent."
        )

        return 1

    print(
        f"\ntranscript           "
        f"{transcript.text!r}"
    )

    print(
        "\nInput layer only: no planner call, "
        "no policy check, no execution."
    )

    return 0


# --------------------------------------------------------------------------
# bench / test
# --------------------------------------------------------------------------


def cmd_bench(
    _args: argparse.Namespace,
    passthrough: list[str],
) -> int:
    """Run the benchmark experiment."""

    from agent_control.planner.openai_compat import load_env
    from benchmark.harness import main as harness_main

    load_env()

    price_in, price_out = _price_defaults()

    if (
        price_in
        and not any(
            arg.startswith("--price-in")
            for arg in passthrough
        )
    ):

        passthrough += [
            "--price-in",
            str(price_in),
        ]

    if (
        price_out
        and not any(
            arg.startswith("--price-out")
            for arg in passthrough
        )
    ):

        passthrough += [
            "--price-out",
            str(price_out),
        ]

    return harness_main(
        passthrough
    )


def cmd_test(
    _args: argparse.Namespace,
    passthrough: list[str],
) -> int:
    """Run pytest."""

    return subprocess.call(
        [
            sys.executable,
            "-m",
            "pytest",
            *(
                passthrough
                or ["-q"]
            ),
        ],
        cwd=ROOT,
    )


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the DEIMOS command-line interface."""

    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "DEIMOS -- conversational state-first computer-use agent."
        ),
        epilog=(
            "Start with: python main.py doctor"
        ),
    )

    sub = parser.add_subparsers(
        dest="command",
        metavar="<command>",
    )

    # ------------------------------------------------------------------
    # doctor
    # ------------------------------------------------------------------

    doctor = sub.add_parser(
        "doctor",
        help=(
            "check configuration and make one live model call"
        ),
    )

    doctor.add_argument(
        "--offline",
        action="store_true",
        help="skip the live call",
    )

    doctor.add_argument(
        "--vision",
        action="store_true",
        help=(
            "also check the baseline VLM accepts a screenshot"
        ),
    )

    doctor.set_defaults(
        func=cmd_doctor
    )

    # ------------------------------------------------------------------
    # models
    # ------------------------------------------------------------------

    models = sub.add_parser(
        "models",
        help="list models the endpoint serves",
    )

    models.add_argument(
        "--filter",
        default="",
        help=(
            "substring filter, e.g. vision"
        ),
    )

    models.set_defaults(
        func=cmd_models
    )

    # ------------------------------------------------------------------
    # tasks
    # ------------------------------------------------------------------

    sub.add_parser(
        "tasks",
        help=(
            "list registered benchmark / structured workflows"
        ),
    ).set_defaults(
        func=cmd_tasks
    )

    # ------------------------------------------------------------------
    # memory
    # ------------------------------------------------------------------

    memory = sub.add_parser(
        "memory",
        help=(
            "persistent file-location index: "
            "status, refresh, query"
        ),
    )

    memory.add_argument(
        "query",
        nargs="?",
        default="",
        help=(
            'look up locations, '
            'e.g. "internship certificate"'
        ),
    )

    memory.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "rescan the configured roots "
            "and rewrite the index"
        ),
    )

    memory.add_argument(
        "--limit",
        type=int,
        default=5,
        help=(
            "how many candidates to show "
            "(default 5)"
        ),
    )

    memory.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help=(
            "folder depth ceiling per root"
        ),
    )

    memory.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help=(
            "hard cap on indexed entries"
        ),
    )

    memory.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "seconds allowed per root"
        ),
    )

    memory.add_argument(
        "--json",
        action="store_true",
        help=(
            "dump the raw report"
        ),
    )

    memory.set_defaults(
        func=cmd_memory
    )

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------

    chat = sub.add_parser(
        "chat",
        help=(
            "conversational interface for questions and computer actions"
        ),
    )

    chat.add_argument(
        "--planner",
        choices=(
            "mock",
            "llm",
        ),
        default="llm",
    )

    chat.add_argument(
        "--max-steps",
        type=int,
        default=10,
    )

    chat.add_argument(
        "--keep-workspace",
        action="store_true",
    )

    chat.add_argument(
        "--no-memory",
        action="store_true",
        help=(
            "ignore remembered file locations"
        ),
    )

    chat.add_argument(
        "--no-status",
        action="store_true",
        help=(
            "hide live execution status lines"
        ),
    )

    chat.add_argument(
        "--debug",
        action="store_true",
        help=(
            "show internal task ids and detailed execution events"
        ),
    )

    chat.add_argument(
        "--json",
        action="store_true",
        help=(
            "dump every turn as JSON on exit"
        ),
    )

    chat.add_argument(
        "--speak",
        dest="speak",
        action="store_true",
        default=True,
        help=(
            "read replies aloud "
            "(overrides TTS_ENABLED)"
        ),
    )

    chat.add_argument(
        "--no-speak",
        dest="speak",
        action="store_false",
        help=(
            "stay silent even if "
            "TTS_ENABLED is set"
        ),
    )

    chat.set_defaults(
        func=cmd_chat
    )

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    run = sub.add_parser(
        "run",
        help=(
            "run one registered task directly"
        ),
    )

    run.add_argument(
        "task",
        help=(
            "registered task id "
            "(see: python main.py tasks)"
        ),
    )

    run.add_argument(
        "--planner",
        choices=(
            "mock",
            "llm",
        ),
        default="llm",
    )

    run.add_argument(
        "--max-steps",
        type=int,
        default=10,
    )

    run.add_argument(
        "--no-recovery",
        action="store_true",
        help="H3 ablation",
    )

    run.add_argument(
        "--no-fresh-state",
        action="store_true",
        help="H2 ablation",
    )

    run.add_argument(
        "--keep-workspace",
        action="store_true",
    )

    run.add_argument(
        "--json",
        action="store_true",
        help="dump the full outcome",
    )

    run.add_argument(
        "--speak",
        dest="speak",
        action="store_true",
        default=None,
        help=(
            "speak the result "
            "(overrides TTS_ENABLED)"
        ),
    )

    run.add_argument(
        "--no-speak",
        dest="speak",
        action="store_false",
        help=(
            "stay silent even if "
            "TTS_ENABLED is set"
        ),
    )

    run.set_defaults(
        func=cmd_run
    )

    # ------------------------------------------------------------------
    # say
    # ------------------------------------------------------------------

    say = sub.add_parser(
        "say",
        help=(
            "text -> TTS -> speaker; executes nothing"
        ),
    )

    say.add_argument(
        "text",
        nargs="?",
        default=(
            "Control plane speech output is working."
        ),
    )

    say.add_argument(
        "--voice",
        default="",
        help=(
            "voice name "
            "(see --list-voices)"
        ),
    )

    say.add_argument(
        "--list-voices",
        action="store_true",
        help=(
            "list installed voices"
        ),
    )

    say.add_argument(
        "--json",
        action="store_true",
        help=(
            "dump the Utterance"
        ),
    )

    say.set_defaults(
        func=cmd_say
    )

    # ------------------------------------------------------------------
    # voice-test
    # ------------------------------------------------------------------

    voice_test = sub.add_parser(
        "voice-test",
        help=(
            "microphone -> STT -> stdout; executes nothing"
        ),
    )

    voice_test.add_argument(
        "--from-wav",
        default="",
        help=(
            "transcribe this WAV instead of recording "
            "(16 kHz mono)"
        ),
    )

    voice_test.add_argument(
        "--max-seconds",
        type=float,
        default=30.0,
        help=(
            "hard recording ceiling "
            "(default 30)"
        ),
    )

    voice_test.add_argument(
        "--silence",
        type=float,
        default=1.2,
        help=(
            "seconds of silence that end an utterance"
        ),
    )

    voice_test.add_argument(
        "--keep",
        action="store_true",
        help=(
            "save the capture as a WAV"
        ),
    )

    voice_test.add_argument(
        "--json",
        action="store_true",
        help=(
            "dump the Transcript"
        ),
    )

    voice_test.set_defaults(
        func=cmd_voice_test
    )

    # ------------------------------------------------------------------
    # bench
    # ------------------------------------------------------------------

    bench = sub.add_parser(
        "bench",
        help=(
            "the full experiment; "
            "flags pass to benchmark.harness"
        ),
        add_help=False,
    )

    bench.set_defaults(
        func=cmd_bench,
        passthrough=True,
    )

    # ------------------------------------------------------------------
    # test
    # ------------------------------------------------------------------

    test = sub.add_parser(
        "test",
        help=(
            "run pytest; flags pass through"
        ),
        add_help=False,
    )

    test.set_defaults(
        func=cmd_test,
        passthrough=True,
    )

    return parser


# A bare ``python main.py`` enters the conversational interface.
#
# ``--speak`` is intentionally part of the shortcut rather than the ``chat``
# subcommand default: ``python main.py`` is the primary interactive entry point,
# while ``python main.py chat`` continues to respect TTS_ENABLED unless explicitly
# overridden.
DEFAULT_COMMAND = [
    "chat",
    "--speak",
]


def main(
    argv: list[str] | None = None,
) -> int:
    """Parse CLI arguments and dispatch exactly one command."""

    argv = list(
        sys.argv[1:]
        if argv is None
        else argv
    )

    parser = build_parser()

    if not argv:
        argv = list(DEFAULT_COMMAND)

    args, extra = parser.parse_known_args(
        argv
    )

    if not getattr(
        args,
        "func",
        None,
    ):

        parser.print_help()
        return 0

    try:

        if getattr(
            args,
            "passthrough",
            False,
        ):

            return args.func(
                args,
                extra,
            )

        if extra:

            parser.error(
                f"unrecognised arguments: "
                f"{' '.join(extra)}"
            )

        return args.func(
            args
        )

    except KeyboardInterrupt:

        print(
            "\ninterrupted.",
            file=sys.stderr,
        )

        return EXIT_INTERRUPTED


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
