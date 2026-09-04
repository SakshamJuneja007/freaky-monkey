"""Tests for the voice input layer.

None of these touch a microphone, a network, or the agent. That is deliberate:
the properties worth pinning down are the refusals -- what must *not* happen when
speech recognition goes wrong -- and those are exactly the paths a live demo
never exercises.

The provider call is injected (``transcribe(..., recognize=spy)``) so a test can
assert the endpoint was never reached, which is stronger than asserting the
return value looked right.
"""

from __future__ import annotations

import wave
from io import BytesIO

import pytest

from agent_control.speech import mic, stt


def make_wav(seconds: float, *, rate: int = 16_000, level: int = 6_000,
             channels: int = 1) -> bytes:
    """A WAV of the requested length holding a simple alternating waveform."""
    frames_count = int(seconds * rate)
    sample = level.to_bytes(2, "little", signed=True) * channels
    quiet = (0).to_bytes(2, "little", signed=True) * channels
    frames = b"".join(sample if index % 8 < 4 else quiet for index in range(frames_count))
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return buffer.getvalue()


class Spy:
    """Stands in for the provider. Records every call; returns fixed text."""

    def __init__(self, text: str = "hello", confidence: float | None = None,
                 raises: Exception | None = None) -> None:
        self.text, self.confidence, self.raises = text, confidence, raises
        self.calls: list[int] = []

    def __call__(self, payload: bytes, rate: int, channels: int, config) -> tuple:
        self.calls.append(len(payload))
        if self.raises is not None:
            raise self.raises
        return self.text, self.confidence


@pytest.fixture
def config() -> stt.STTConfig:
    return stt.STTConfig(api_key="test-key", provider="test/provider")


# -- Transcript.ok is the only gate a caller may trust -----------------------

def test_error_transcript_is_not_ok() -> None:
    assert not stt.Transcript(text="anything", error="boom").ok


def test_blank_transcript_is_not_ok() -> None:
    """A provider that returns whitespace has not produced a task."""
    assert not stt.Transcript(text="   ").ok
    assert stt.Transcript(text="do the thing").ok


# -- audio that cannot be a task never reaches the provider -----------------

def test_malformed_audio_is_refused_without_calling_the_provider(config) -> None:
    spy = Spy()
    result = stt.transcribe(b"this is not a wav file", config=config, recognize=spy)
    assert spy.calls == []
    assert not result.ok
    assert result.reason == "malformed_audio"


def test_audio_shorter_than_an_utterance_is_refused(config) -> None:
    spy = Spy()
    result = stt.transcribe(make_wav(0.1), config=config, recognize=spy)
    assert spy.calls == []
    assert not result.ok
    assert result.reason == "too_short"


def test_missing_credentials_report_unavailable_not_failure(monkeypatch) -> None:
    """A missing key is a configuration problem, and must say so by name."""
    monkeypatch.setattr(stt, "load_env", lambda *a, **k: None)
    monkeypatch.delenv("STT_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    spy = Spy()
    result = stt.transcribe(make_wav(1.0), recognize=spy)
    assert spy.calls == []
    assert result.reason == "unavailable"
    assert "STT_API_KEY" in (result.error or "")


# -- long audio is chunked, never truncated ---------------------------------

def test_long_audio_is_split_into_bounded_chunks(config) -> None:
    """Every sample must be sent. The provider accepted 300 s without complaint,
    so a caller cannot detect an abridged transcript -- the split has to be ours."""
    audio = make_wav(10.0)
    spy = Spy(text="part")
    result = stt.transcribe(audio, config=config, chunk_seconds=3.0, recognize=spy)

    assert result.ok
    assert result.chunks == 4  # 3 + 3 + 3 + 1
    with wave.open(BytesIO(audio), "rb") as handle:
        total_frames = handle.getnframes() * handle.getsampwidth() * handle.getnchannels()
    # 44 bytes of RIFF header per rebuilt chunk; the audio itself must add up.
    sent = sum(size - 44 for size in spy.calls)
    assert sent == total_frames, "chunking dropped audio"
    assert result.text == "part part part part"


def test_short_audio_is_one_chunk(config) -> None:
    spy = Spy()
    result = stt.transcribe(make_wav(2.0), config=config, recognize=spy)
    assert result.chunks == 1
    assert len(spy.calls) == 1


# -- provider failures stay failures ---------------------------------------

def test_transport_failure_is_classified_not_swallowed(config) -> None:
    spy = Spy(raises=RuntimeError("connection reset"))
    result = stt.transcribe(make_wav(1.0), config=config, recognize=spy)
    assert not result.ok
    assert result.reason == "transport"
    assert "connection reset" in (result.error or "")
    assert result.text == ""


def test_empty_provider_response_is_a_failure_not_a_blank_task(config) -> None:
    """The dangerous case: a successful call that recognised nothing. Forwarding
    '' as a task would plan against an empty goal."""
    spy = Spy(text="   ")
    result = stt.transcribe(make_wav(1.0), config=config, recognize=spy)
    assert not result.ok
    assert result.reason == "empty_transcript"


def test_absent_confidence_is_none_not_zero(config) -> None:
    """whisper-large-v3 reports 0.0 for everything; that is unmeasured, and a
    caller must not be able to threshold on it as if it were measured."""
    result = stt.transcribe(make_wav(1.0), config=config, recognize=Spy(confidence=0.0))
    assert result.ok
    assert result.confidence is None


def test_reported_confidence_is_passed_through(config) -> None:
    result = stt.transcribe(make_wav(1.0), config=config, recognize=Spy(confidence=0.82))
    assert result.confidence == pytest.approx(0.82)


# -- the recorder's silence detection, without a microphone ------------------

def test_rms_distinguishes_silence_from_speech() -> None:
    silence = (0).to_bytes(2, "little", signed=True) * 800
    loud = (12_000).to_bytes(2, "little", signed=True) * 800
    assert mic._rms(silence) == 0.0
    assert mic._rms(loud) > mic.MIN_THRESHOLD * 10


def test_rms_tolerates_a_truncated_block() -> None:
    """PortAudio hands over whole frames, but an odd-length buffer must not raise."""
    assert mic._rms(b"\x01") == 0.0


def test_recording_without_audio_is_not_ok() -> None:
    assert not mic.Recording(stopped_by="no_device", error="no device").ok
    assert not mic.Recording(duration_seconds=3.0, stopped_by="silence").ok
    assert mic.Recording(wav=make_wav(1.0), duration_seconds=1.0, stopped_by="silence",
                         peak_level=0.2).ok


def test_default_recording_ceiling_stays_inside_the_verified_span() -> None:
    """A single push-to-talk capture must never need chunking: 66 s was verified
    complete, MAX_CHUNK_SECONDS sits below that, and the ceiling below again."""
    import inspect

    ceiling = inspect.signature(mic.record_utterance).parameters["max_seconds"].default
    assert ceiling <= stt.MAX_CHUNK_SECONDS <= 66.0


# -- the CLI command cannot execute anything --------------------------------

def _voice_test_args(wav_path, **overrides):
    import argparse

    fields = {"from_wav": str(wav_path), "max_seconds": 30.0, "silence": 1.2,
              "keep": False, "json": False}
    fields.update(overrides)
    return argparse.Namespace(**fields)


@pytest.fixture
def no_execution(monkeypatch):
    """Make any attempt to run a task an outright test failure."""
    import agent_control.runner as runner

    def forbidden(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("voice-test executed a task")

    monkeypatch.setattr(runner, "run_task", forbidden)


def test_voice_test_reports_failure_without_executing(tmp_path, monkeypatch,
                                                      no_execution) -> None:
    import agent_control.speech as speech
    import main

    wav = tmp_path / "in.wav"
    wav.write_bytes(make_wav(1.0))
    monkeypatch.setattr(speech, "transcribe", lambda *a, **k: stt.Transcript(
        provider="test/provider", reason="transport", error="endpoint down"))

    assert main.cmd_voice_test(_voice_test_args(wav)) == 1


def test_voice_test_succeeds_without_executing(tmp_path, monkeypatch,
                                               no_execution) -> None:
    import agent_control.speech as speech
    import main

    wav = tmp_path / "in.wav"
    wav.write_bytes(make_wav(1.0))
    monkeypatch.setattr(speech, "transcribe", lambda *a, **k: stt.Transcript(
        text="install the dependencies", provider="test/provider"))

    assert main.cmd_voice_test(_voice_test_args(wav)) == 0
