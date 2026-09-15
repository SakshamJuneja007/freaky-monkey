"""Push-to-talk microphone capture, bounded by construction.

Push-to-talk only: there is no wake word and nothing listens in the background.
A capture starts when the user runs a command and ends one of three ways --
silence after speech, the hard ``max_seconds`` ceiling, or Ctrl+C -- and every
one of those is reported in ``Recording.stopped_by`` rather than inferred.

The default ceiling sits below ``stt.MAX_CHUNK_SECONDS``, so a single utterance
is always inside the span whose transcription was verified complete end to end.
Long audio still chunks correctly in ``stt.transcribe``; this just means the
push-to-talk path never relies on it.

Silence detection is deliberately dumb: RMS against a threshold calibrated from
the room's own noise floor in the first fraction of a second. An absolute
threshold would work on one microphone and fail on the next, and anything
cleverer would be a model to debug rather than an input device to use.

Stdlib only -- ``array`` for the RMS, ``wave`` for the container. ``sounddevice``
is imported lazily so that importing this module never requires PortAudio.
"""

from __future__ import annotations

import array
import math
import time
import threading
import wave
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from typing import Callable

#: 16 kHz mono 16-bit is what the endpoint was verified against; resampling in
#: the client would add a failure mode for no benefit.
SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2

_CAPTURE_LOCK = threading.Lock()

#: Small enough that silence is noticed promptly, large enough not to spin.
BLOCK_SECONDS = 0.05

#: Absolute floor for the speech threshold, so a pathologically quiet room
#: cannot calibrate its way into treating hiss as speech.
MIN_THRESHOLD = 0.004

#: Multiple of the measured noise floor that counts as speech.
NOISE_MULTIPLE = 3.0

#: Audio kept from *before* the threshold was crossed, so the first consonant of
#: the first word is not clipped by the detector's own reaction time. An RMS
#: threshold is crossed a fraction of a syllable late, so some pre-roll is
#: necessary; a long one would reintroduce the leading silence it exists to
#: remove. Everything earlier than this is discarded rather than transcribed.
PREROLL_SECONDS = 0.3


@dataclass(frozen=True)
class Recording:
    """Captured audio, or why there is none."""

    wav: bytes = b""
    duration_seconds: float = 0.0
    #: "silence" | "max_seconds" | "user" | "no_device" | "error". Never guessed.
    stopped_by: str = ""
    error: str | None = None
    #: Loudest block seen, 0..1. Useful when a capture came back empty: a peak at
    #: the noise floor means the microphone was muted, not that the user was.
    peak_level: float = 0.0
    #: Wall-clock seconds spent listening, including the silence before speech
    #: that ``PREROLL_SECONDS`` discarded. Kept separate from
    #: ``duration_seconds`` -- which is the length of the audio actually sent for
    #: transcription -- so that "I waited nine seconds and heard two" stays
    #: sayable. Collapsing them would make a long pause look like a long utterance.
    waited_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and self.duration_seconds > 0.0 and bool(self.wav)


def _rms(block: bytes) -> float:
    """Normalised RMS of one int16 block, 0..1."""
    samples = array.array("h")
    samples.frombytes(block[: len(block) - (len(block) % 2)])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768.0


def _to_wav(frames: bytes) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(frames)
    return buffer.getvalue()


def record_utterance(*, max_seconds: float = 30.0, silence_seconds: float = 1.2,
                     calibrate_seconds: float = 0.4,
                     on_status: Callable[[str], None] | None = None) -> Recording:
    """Record one utterance from the default input device. Never raises.

    ``on_status`` receives short human-readable strings ("calibrating", "listening",
    "recording", "stopping: silence") so the CLI can show what the microphone is
    doing without this module knowing anything about the terminal.
    """
    if not _CAPTURE_LOCK.acquire(blocking=False):
        return Recording(stopped_by="listener_busy", error="another speech listener is already capturing")
    try:

        say = on_status or (lambda _message: None)
        try:
            import sounddevice  # noqa: PLC0415 - lazy: PortAudio is optional
        except (ImportError, OSError) as exc:
            return Recording(stopped_by="no_device", error=(
                f"microphone capture unavailable ({type(exc).__name__}: {exc}); "
                "pip install sounddevice"))

        block_frames = max(1, int(SAMPLE_RATE * BLOCK_SECONDS))
        #: Rolling window of the audio before speech starts. Bounded, so a person who
        #: takes ten seconds to think does not send ten seconds of room tone to the
        #: recogniser -- which costs latency, and invites a transcript of the room.
        preroll: deque[bytes] = deque(
            maxlen=max(1, int(PREROLL_SECONDS / BLOCK_SECONDS))
        )
        captured: list[bytes] = []
        noise_floor = 0.0
        peak = 0.0
        waited = 0.0
        stopped_by = "max_seconds"

        try:
            stream = sounddevice.RawInputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16",
                blocksize=block_frames,
            )
        except Exception as exc:  # noqa: BLE001 - PortAudio raises its own types
            return Recording(stopped_by="no_device", error=(
                f"could not open the default input device: {type(exc).__name__}: {exc}"))

        try:
            with stream:
                say("calibrating (stay quiet)")
                deadline = time.time() + calibrate_seconds
                while time.time() < deadline:
                    block, _overflow = stream.read(block_frames)
                    noise_floor = max(noise_floor, _rms(bytes(block)))
                threshold = max(MIN_THRESHOLD, noise_floor * NOISE_MULTIPLE)

                say(f"listening (noise floor {noise_floor:.4f}, "
                    f"speech above {threshold:.4f}); Ctrl+C to stop")
                started = time.time()
                speech_started = False
                last_voice = started
                while True:
                    waited = time.time() - started
                    if waited >= max_seconds:
                        break
                    block, _overflow = stream.read(block_frames)
                    data = bytes(block)
                    level = _rms(data)
                    peak = max(peak, level)

                    if speech_started:
                        captured.append(data)
                    elif level >= threshold:
                        # The threshold was crossed part-way into a word, so the
                        # window that was being held back is prepended rather than
                        # dropped: without it the recogniser receives a syllable that
                        # begins mid-consonant.
                        speech_started = True
                        captured.extend(preroll)
                        captured.append(data)
                        preroll.clear()
                        say("recording")
                    else:
                        # Still waiting. The block is remembered only as pre-roll, so
                        # a long pause before speaking costs nothing downstream.
                        preroll.append(data)

                    if level >= threshold:
                        last_voice = time.time()
                    elif speech_started and time.time() - last_voice >= silence_seconds:
                        stopped_by = "silence"
                        break
        except KeyboardInterrupt:
            # A deliberate stop, not a failure: keep what was said up to this point.
            stopped_by = "user"
            say("stopping: user")
        except Exception as exc:  # noqa: BLE001 - a dead device is a datum
            return Recording(stopped_by="error", peak_level=peak, waited_seconds=waited, error=(
                f"capture failed after {len(captured) * BLOCK_SECONDS:.1f}s: "
                f"{type(exc).__name__}: {exc}"))
        else:
            say(f"stopping: {stopped_by}")

        frames = b"".join(captured)
        seconds = len(frames) / float(SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)
        if peak < max(MIN_THRESHOLD, noise_floor * NOISE_MULTIPLE):
            # Nothing above the room's own noise. Reporting this as an empty
            # transcript later would blame the model for a muted microphone. The span
            # named is the one that was listened to, not the pre-roll that survived
            # it -- "no speech in 0.3s" would look like the recorder gave up.
            return Recording(
                duration_seconds=seconds, stopped_by=stopped_by, peak_level=peak,
                waited_seconds=waited,
                error=f"no speech detected in {waited:.1f}s (peak level {peak:.4f} "
                      f"vs noise floor {noise_floor:.4f}); is the microphone muted?",
            )
        return Recording(wav=_to_wav(frames), duration_seconds=seconds,
                         stopped_by=stopped_by, peak_level=peak, waited_seconds=waited)
    finally:
        _CAPTURE_LOCK.release()
