"""Text-to-speech behind one seam: ``speak(text) -> Utterance``.

This is the output half of the interface layer, and it is deliberately the
dumbest module in the package: it converts a string that somebody else already
decided was worth saying into sound. It does not plan, decide, summarise, or
consult a model. What is worth saying is decided in ``agent_control/response.py``
from verified fields of ``AgentResult``; this file only pronounces it.

Three properties are load-bearing:

* **It cannot break a run.** ``speak`` never raises. A missing PowerShell, a
  dead audio device, a timeout, or a nonzero exit code all come back as an
  ``Utterance`` with ``spoken=False`` and a reason. Callers print regardless.
* **Text is never code.** The spoken text and the voice name are passed to
  PowerShell through the child's environment and base64, never interpolated
  into the script. A task name, a filename, or a repository README containing
  ``"; Remove-Item -Recurse"`` is pronounced, not executed.
* **It does not block the agent.** ``Speaker`` serialises utterances on one
  daemon thread, so a two second sentence does not add two seconds to a run.

The backend is the Windows Speech API through ``System.Speech``, which ships
with the OS: no new dependency, no network call, and no audio leaves the
machine. ``TTS_ENABLED`` gates the whole layer and defaults to off, so text
behaviour is unchanged until it is explicitly turned on.
"""

from __future__ import annotations

import base64
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

from ..planner.openai_compat import load_env

#: Long enough for any status sentence this project produces, short enough that
#: a mistake cannot trap the user in a minute of narration. Overflow is cut with
#: a spoken marker rather than silently dropped.
MAX_SPOKEN_CHARS = 600

#: SAPI rate is -10..10, 0 being the voice's natural speed.
MIN_RATE, MAX_RATE = -10, 10

#: A sentence takes seconds; anything past this is a stuck subprocess.
DEFAULT_TIMEOUT_S = 60.0

_TEXT_VAR = "AGENT_TTS_TEXT_B64"
_VOICE_VAR = "AGENT_TTS_VOICE_B64"
_RATE_VAR = "AGENT_TTS_RATE"

#: Reads its inputs from the environment on purpose -- see the module docstring.
_SPEAK_SCRIPT = f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$text = [Text.Encoding]::UTF8.GetString(
    [Convert]::FromBase64String($env:{_TEXT_VAR}))
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$name = $env:{_VOICE_VAR}
if ($name) {{
    $wanted = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($name))
    try {{ $synth.SelectVoice($wanted) }} catch {{ }}
}}
$synth.Rate = [int]$env:{_RATE_VAR}
$synth.SetOutputToDefaultAudioDevice()
$synth.Speak($text)
$synth.Dispose()
"""

_VOICES_SCRIPT = """
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
foreach ($v in $synth.GetInstalledVoices()) {
    if ($v.Enabled) { Write-Output $v.VoiceInfo.Name }
}
$synth.Dispose()
"""


@dataclass(frozen=True)
class Utterance:
    """What was said, or an honest account of why nothing was.

    ``spoken`` is the fact; ``ok`` is the same fact named for callers that read
    it as a gate. Note what is deliberately absent: there is no field saying the
    utterance was *correct*, because this layer has no way to know that. Whether
    a sentence should have been said at all is decided upstream from verified
    verdicts, never here.
    """

    text: str = ""
    spoken: bool = False
    provider: str = ""
    latency_seconds: float = 0.0
    error: str | None = None
    #: Short slug for tests and traces: "disabled", "empty", "unavailable",
    #: "timeout", "backend_error". Distinct causes stay distinct.
    reason: str = ""
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.spoken

    def to_json(self) -> dict:
        return {
            "text": self.text,
            "spoken": self.spoken,
            "provider": self.provider,
            "latency_seconds": round(self.latency_seconds, 3),
            "error": self.error,
            "reason": self.reason,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class TTSConfig:
    """Speech output settings, read from the environment once.

    ``enabled`` defaults to **False**: adding an output layer must not change
    how the tool already behaves for anyone who has not asked for sound.
    """

    enabled: bool = False
    #: Empty means "whatever the system default voice is". A name that is not
    #: installed is ignored by the backend rather than failing the utterance.
    voice: str = ""
    rate: int = 0
    timeout_s: float = DEFAULT_TIMEOUT_S

    @classmethod
    def from_env(cls, *, enabled: bool | None = None) -> "TTSConfig":
        """Build from ``.env``. An explicit ``enabled`` overrides the file."""
        load_env()

        def number(name: str, fallback: float) -> float:
            try:
                return float(os.environ.get(name, "") or fallback)
            except ValueError:
                return fallback

        from_file = (os.environ.get("TTS_ENABLED", "") or "").strip().lower()
        rate = int(max(MIN_RATE, min(MAX_RATE, number("TTS_RATE", 0))))
        return cls(
            enabled=(from_file in ("1", "true", "yes", "on")
                     if enabled is None else enabled),
            voice=(os.environ.get("TTS_VOICE", "") or "").strip(),
            rate=rate,
            timeout_s=number("TTS_TIMEOUT_S", DEFAULT_TIMEOUT_S),
        )


def _powershell() -> str | None:
    """The interpreter to drive SAPI with, or None on a machine without one."""
    for candidate in ("powershell", "pwsh"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _encode(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _invoke(script: str, extra_env: dict[str, str], timeout_s: float
            ) -> subprocess.CompletedProcess[str]:
    """Run one PowerShell script with a fixed argv and no shell."""
    shell = _powershell()
    if shell is None:  # pragma: no cover - guarded by callers
        raise FileNotFoundError("powershell not found on PATH")

    return subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        env={**os.environ, **extra_env},
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def speak(text: str, *, config: TTSConfig | None = None) -> Utterance:
    """Say one sentence. Blocks until it has been said.

    Never raises: every failure mode is a returned ``Utterance``. Callers are
    output layers, and an output layer that can abort a verified run by failing
    to make a noise would be worse than silence.
    """
    settings = config or TTSConfig.from_env()
    body = (text or "").strip()

    if not settings.enabled:
        return Utterance(text=body, reason="disabled")
    if not body:
        return Utterance(reason="empty")
    if _powershell() is None:
        return Utterance(text=body, reason="unavailable",
                         error="no PowerShell interpreter on PATH")

    truncated = len(body) > MAX_SPOKEN_CHARS
    if truncated:
        body = body[:MAX_SPOKEN_CHARS].rstrip() + ". Details are in the text output."

    started = time.time()
    try:
        proc = _invoke(
            _SPEAK_SCRIPT,
            {_TEXT_VAR: _encode(body),
             _VOICE_VAR: _encode(settings.voice) if settings.voice else "",
             _RATE_VAR: str(settings.rate)},
            settings.timeout_s,
        )
    except subprocess.TimeoutExpired:
        return Utterance(text=body, reason="timeout", truncated=truncated,
                         latency_seconds=time.time() - started,
                         error=f"speech did not finish within {settings.timeout_s:.0f}s")
    except Exception as exc:  # noqa: BLE001 - an output layer reports, never raises
        return Utterance(text=body, reason="backend_error", truncated=truncated,
                         latency_seconds=time.time() - started,
                         error=f"{type(exc).__name__}: {exc}")

    elapsed = time.time() - started
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return Utterance(text=body, reason="backend_error", truncated=truncated,
                         latency_seconds=elapsed, provider="sapi",
                         error=f"powershell exited {proc.returncode}: "
                               f"{detail[0] if detail else 'no output'}")

    return Utterance(text=body, spoken=True, provider="sapi",
                     latency_seconds=elapsed, truncated=truncated)


def voices(*, timeout_s: float = 30.0) -> list[str]:
    """Installed, enabled voice names. Empty when the backend is unusable."""
    if _powershell() is None:
        return []
    try:
        proc = _invoke(_VOICES_SCRIPT, {}, timeout_s)
    except Exception:  # noqa: BLE001
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


class Speaker:
    """Serialises utterances on one background thread.

    Without this, speaking is synchronous: a two second sentence before a run
    would add two seconds to the run, which is exactly the interference the
    output layer is supposed to avoid. One worker thread (not a pool) because
    two voices talking over each other is worse than waiting.

    ``blocking=True`` runs in the calling thread instead, which is what the CLI
    ``say`` command and the tests want -- there the utterance *is* the result.
    """

    def __init__(self, config: TTSConfig | None = None, *, blocking: bool = False,
                 backend=speak) -> None:
        self.config = config or TTSConfig.from_env()
        self.blocking = blocking
        #: Injectable so tests can assert what would have been said without
        #: making a sound, and can simulate a backend that fails.
        self._backend = backend
        self.said: list[Utterance] = []
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def say(self, text: str) -> None:
        """Queue one sentence. Returns immediately unless ``blocking``."""
        if not self.config.enabled or not (text or "").strip():
            return
        if self.blocking:
            self._speak_now(text)
            return
        self._ensure_worker()
        self._queue.put(text)

    def wait_until_idle(self, timeout_s: float = 15.0) -> bool:
        """Wait until all queued speech has finished."""
        deadline = time.time() + max(0.0, float(timeout_s))
        while time.time() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return self._queue.unfinished_tasks == 0

    def close(self, timeout_s: float = 15.0) -> None:
        """Let queued speech finish, then stop the worker.

        Bounded on purpose: exiting the command must not depend on the audio
        stack behaving.
        """
        worker = self._worker
        if worker is None:
            return
        self._queue.put(None)
        worker.join(timeout=timeout_s)
        self._worker = None

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._drain, name="tts",
                                        daemon=True)
        self._worker.start()

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._speak_now(item)
            finally:
                self._queue.task_done()

    def _speak_now(self, text: str) -> None:
        try:
            result = self._backend(text, config=self.config)
        except Exception as exc:  # noqa: BLE001 - a broken backend stays silent
            result = Utterance(text=text, reason="backend_error",
                               error=f"{type(exc).__name__}: {exc}")
        with self._lock:
            self.said.append(result)
