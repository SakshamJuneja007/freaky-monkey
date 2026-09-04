"""Speech-to-text behind one seam: ``transcribe(audio) -> Transcript``.

Everything asserted here was established by calling the endpoint, never by
reading documentation:

* ``integrate.api.nvidia.com`` (the planner's base URL) has no
  ``/audio/transcriptions`` route -- it returns 404. Nor does
  ``ai.api.nvidia.com/v1/models`` answer this key.
* ``GET https://api.nvcf.nvidia.com/v2/nvcf/functions`` lists 186 functions this
  key may invoke, nine of them ACTIVE speech models. Every one reports
  ``apiBodyFormat: CUSTOM`` with ``health.protocol: gRPC``, and a plain HTTP
  ``POST /v2/nvcf/pexec/functions/{id}`` with a JSON body returns 500. The
  transport is therefore Riva gRPC, not an OpenAI-compatible HTTP route.
* ``ai-whisper-large-v3`` transcribed a 4.33 s known-text WAV word for word in
  3.31 s, and 66 s of unique speech (14 distinct sentences) with the final
  sentence intact in 5.44 s. Nothing was dropped at that length.
* The endpoint returns ``confidence == 0.0`` for every alternative. That is an
  absent signal, not a low one, so it is surfaced as ``None`` rather than as a
  number a caller might threshold on.
* 300 s of audio was accepted without error, but faithfulness was only
  *verified* to 66 s. ``MAX_CHUNK_SECONDS`` therefore sits below what was
  proven, and longer audio is split into bounded chunks -- never truncated.

Callers depend on this module only through ``Transcript``; the backend can be
replaced without anything above it changing.
"""

from __future__ import annotations

import os
import time
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from ..planner.openai_compat import load_env

#: NVCF's gRPC gateway. Overridable, because a self-hosted Riva NIM speaks the
#: same protocol on a different host.
DEFAULT_GRPC_URI = "grpc.nvcf.nvidia.com:443"

#: ``ai-whisper-large-v3``, found by enumerating the functions this key can
#: actually invoke. Not copied from a docs page, and overridable via
#: ``STT_FUNCTION_ID`` so a different discovered model needs no code change.
DEFAULT_FUNCTION_ID = "b702f636-f60c-4a3d-a6f4-f3568c13bd7d"

#: Longest span whose transcript was verified complete end to end (66 s was
#: proven; this leaves headroom). Audio longer than this is chunked.
MAX_CHUNK_SECONDS = 45.0

#: Below this there is no utterance to transcribe, only a keypress. Sending it
#: anyway would spend a call to be told nothing.
MIN_AUDIO_SECONDS = 0.25


@dataclass(frozen=True)
class Transcript:
    """What the microphone turned into, or an honest account of why it did not.

    ``ok`` is the only gate callers should use. A transcript that is not ok must
    never be handed to the agent as a task: an empty or garbled request would be
    planned against, and the run would be attributed to the user's intent rather
    than to a failed input layer.
    """

    text: str = ""
    #: ``None`` when the provider does not report one. See the module docstring:
    #: whisper-large-v3 on NVCF reports 0.0 for everything, which means "not
    #: measured", so it is not passed through as if it were measured.
    confidence: float | None = None
    provider: str = ""
    #: Wall-clock length of the audio, not of the API call.
    duration_seconds: float = 0.0
    #: Round trip to the provider, kept separate so a slow network is not
    #: mistaken for a long utterance.
    latency_seconds: float = 0.0
    error: str | None = None
    #: Short slug for the trace and for tests: "too_short", "malformed_audio",
    #: "unavailable", "transport", "empty_transcript". Distinct causes stay
    #: distinct -- collapsing them would hide which half of the stack broke.
    reason: str = ""
    #: How many bounded chunks the audio was split into. >1 means the transcript
    #: is a concatenation and a word may have been cut at a boundary.
    chunks: int = 1

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())

    def to_json(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "provider": self.provider,
            "duration_seconds": round(self.duration_seconds, 3),
            "latency_seconds": round(self.latency_seconds, 3),
            "error": self.error,
            "reason": self.reason,
            "chunks": self.chunks,
            "ok": self.ok,
        }


class STTUnavailable(RuntimeError):
    """No usable STT credentials or client library. Distinct from a failed call."""


@dataclass(frozen=True)
class STTConfig:
    """Where to send audio. Read from the environment, never hard-coded in code."""

    api_key: str
    function_id: str = DEFAULT_FUNCTION_ID
    grpc_uri: str = DEFAULT_GRPC_URI
    language: str = "en-US"
    provider: str = "nvidia-riva/whisper-large-v3"
    timeout_s: float = 120.0

    @classmethod
    def from_env(cls) -> "STTConfig":
        load_env()
        # Falls back to LLM_API_KEY on purpose: it is the same nvapi credential,
        # and requiring a second copy of one key invites them to drift apart.
        key = os.environ.get("STT_API_KEY") or os.environ.get("LLM_API_KEY") or ""
        if not key:
            raise STTUnavailable(
                "no STT credential: set STT_API_KEY (or LLM_API_KEY) in .env"
            )
        timeout = os.environ.get("STT_TIMEOUT_S") or ""
        return cls(
            api_key=key,
            function_id=os.environ.get("STT_FUNCTION_ID") or DEFAULT_FUNCTION_ID,
            grpc_uri=os.environ.get("STT_GRPC_URI") or DEFAULT_GRPC_URI,
            language=os.environ.get("STT_LANGUAGE") or "en-US",
            provider=os.environ.get("STT_PROVIDER") or "nvidia-riva/whisper-large-v3",
            timeout_s=float(timeout) if timeout.replace(".", "", 1).isdigit() else 120.0,
        )


def _wav_facts(payload: bytes) -> tuple[bytes, int, int, int, float]:
    """Return ``(frames, rate, channels, sample_width, seconds)``.

    Raises ``ValueError`` for anything that is not a readable WAV, so a corrupted
    capture is reported as corrupted rather than posted to the provider.
    """
    with wave.open(BytesIO(payload), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if not rate or not channels or not width:
        raise ValueError("WAV header declares no rate, channels, or sample width")
    seconds = len(frames) / float(rate * channels * width)
    return frames, rate, channels, width, seconds


def _rebuild_wav(frames: bytes, rate: int, channels: int, width: int) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return buffer.getvalue()


def _split(frames: bytes, rate: int, channels: int, width: int,
           chunk_seconds: float) -> list[bytes]:
    """Cut audio into bounded whole-frame chunks.

    The alternative -- sending the whole thing and hoping -- is exactly the silent
    truncation this must avoid: the provider accepted 300 s without complaint,
    and a caller cannot tell an abridged transcript from a complete one.
    """
    stride = int(chunk_seconds * rate) * channels * width
    if stride <= 0 or len(frames) <= stride:
        return [frames]
    return [frames[start:start + stride] for start in range(0, len(frames), stride)]


def _recognize(payload: bytes, rate: int, channels: int,
               config: STTConfig) -> tuple[str, float | None]:
    """One Riva gRPC offline_recognize call. Returns ``(text, confidence)``.

    Imported lazily so that importing this module -- and therefore running the
    test suite -- does not require the gRPC stack to be installed.
    """
    try:
        import riva.client  # noqa: PLC0415 - deliberate lazy import
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise STTUnavailable(
            "nvidia-riva-client is not installed; pip install nvidia-riva-client"
        ) from exc

    auth = riva.client.Auth(
        uri=config.grpc_uri,
        use_ssl=True,
        metadata_args=[
            ["function-id", config.function_id],
            ["authorization", f"Bearer {config.api_key}"],
        ],
    )
    request = riva.client.RecognitionConfig(
        language_code=config.language,
        max_alternatives=1,
        enable_automatic_punctuation=True,
        audio_channel_count=channels,
        sample_rate_hertz=rate,
    )
    response = riva.client.ASRService(auth).offline_recognize(payload, request)

    parts: list[str] = []
    scores: list[float] = []
    for result in response.results:
        for alternative in result.alternatives:
            parts.append(alternative.transcript.strip())
            scores.append(float(alternative.confidence))
    # Every measured score was exactly 0.0, i.e. unreported. Treat that as absent
    # rather than as certainty about uncertainty.
    confidence = (sum(scores) / len(scores)) if scores and any(scores) else None
    return " ".join(p for p in parts if p).strip(), confidence


def transcribe(audio: bytes | str | Path, *, config: STTConfig | None = None,
               chunk_seconds: float = MAX_CHUNK_SECONDS,
               recognize=_recognize) -> Transcript:
    """Audio in, ``Transcript`` out. Never raises.

    A raised exception here would either crash the voice command or, worse, get
    swallowed by a caller and become an empty task. Every failure path returns a
    ``Transcript`` with ``ok is False`` and a ``reason``, so the caller has to
    decide what to do about it rather than being handed a plausible-looking
    empty string.

    ``recognize`` is injected so the chunking and refusal logic can be tested
    without spending API calls; production callers never pass it.
    """
    provider = config.provider if config else "nvidia-riva/whisper-large-v3"
    try:
        payload = Path(audio).read_bytes() if isinstance(audio, (str, Path)) else audio
    except OSError as exc:
        return Transcript(provider=provider, error=f"cannot read audio: {exc}",
                          reason="malformed_audio")

    try:
        frames, rate, channels, width, seconds = _wav_facts(payload)
    except (wave.Error, ValueError, EOFError) as exc:
        return Transcript(provider=provider, reason="malformed_audio",
                          error=f"not a readable WAV: {type(exc).__name__}: {exc}")

    if seconds < MIN_AUDIO_SECONDS:
        return Transcript(
            provider=provider, duration_seconds=seconds, reason="too_short",
            error=f"only {seconds:.2f}s of audio (minimum {MIN_AUDIO_SECONDS}s); "
                  "nothing was said",
        )

    try:
        resolved = config or STTConfig.from_env()
    except STTUnavailable as exc:
        return Transcript(provider=provider, duration_seconds=seconds,
                          reason="unavailable", error=str(exc))

    pieces = _split(frames, rate, channels, width, chunk_seconds)
    texts: list[str] = []
    scores: list[float] = []
    started = time.time()
    for index, piece in enumerate(pieces):
        try:
            text, score = recognize(
                _rebuild_wav(piece, rate, channels, width), rate, channels, resolved,
            )
        except STTUnavailable as exc:
            return Transcript(provider=resolved.provider, duration_seconds=seconds,
                              reason="unavailable", error=str(exc),
                              latency_seconds=time.time() - started)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            detail = getattr(exc, "details", None)
            detail = detail() if callable(detail) else detail
            return Transcript(
                provider=resolved.provider, duration_seconds=seconds,
                reason="transport", latency_seconds=time.time() - started,
                error=f"STT call failed on chunk {index + 1}/{len(pieces)}: "
                      f"{type(exc).__name__}: {detail or exc}",
            )
        if text:
            texts.append(text)
        # A falsy score means "not reported": whisper-large-v3 returns 0.0 for
        # every alternative, and averaging that in would hand the caller a number
        # to threshold on that no model actually computed.
        if score:
            scores.append(score)

    latency = time.time() - started
    combined = " ".join(texts).strip()
    if not combined:
        return Transcript(
            provider=resolved.provider, duration_seconds=seconds,
            latency_seconds=latency, chunks=len(pieces), reason="empty_transcript",
            error=f"provider returned no words for {seconds:.2f}s of audio",
        )
    return Transcript(
        text=combined, provider=resolved.provider, duration_seconds=seconds,
        latency_seconds=latency, chunks=len(pieces),
        confidence=(sum(scores) / len(scores)) if scores else None,
    )
