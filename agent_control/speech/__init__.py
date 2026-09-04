"""Voice I/O for the control plane -- a thin interface layer, nothing more.

Nothing in this package plans, executes, verifies, or decides. It turns sound
into a string and a string back into sound; every consequential decision stays
where it already lives (planner, policy, runner, verifiers).

Input is ``mic.record_utterance`` plus ``stt.transcribe``: one bounded capture,
one ``Transcript``. That is the whole of speech input -- audio to text. What
happens to the text afterwards is not this package's concern: it is handed to
``agent_control.session.Session``, which treats a transcript exactly as it treats
a typed line, so there is no voice-specific execution path to keep in step.
``python main.py voice-test`` exercises this half in isolation, with no runner
imported at all.

The output half is ``tts.speak``, which pronounces a sentence somebody else
already decided was worth saying. It is off unless ``TTS_ENABLED`` says
otherwise, and it cannot fail a run -- see ``agent_control/response.py`` for
where the sentences come from.
"""

from .mic import Recording, record_utterance
from .stt import STTConfig, STTUnavailable, Transcript, transcribe
from .tts import Speaker, TTSConfig, Utterance, speak, voices

__all__ = [
    "Recording",
    "record_utterance",
    "STTConfig",
    "STTUnavailable",
    "Transcript",
    "transcribe",
    "Speaker",
    "TTSConfig",
    "Utterance",
    "speak",
    "voices",
]
