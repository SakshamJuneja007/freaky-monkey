"""Global, read-only browser text detection.

The detector answers one narrow question: what editable text is currently
exposed by the browser UI, and did that text change between two fresh reads?
It is deliberately independent from browser action results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...platform_window import get_backend
from ...text_observation import TextObservation


@dataclass(frozen=True)
class BrowserTextSnapshot:
    text: str
    observation: TextObservation
    values: tuple[str, ...]


@dataclass(frozen=True)
class BrowserTextChange:
    changed: bool | None
    typed: bool | None
    before: BrowserTextSnapshot
    after: BrowserTextSnapshot
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    reason: str = ""


class GlobalBrowserTextDetector:
    """Fresh semantic detector for browser-owned editable text."""

    def __init__(self, backend: Any | None = None) -> None:
        self._backend = backend or get_backend()

    def observe(self) -> BrowserTextSnapshot:
        reader = getattr(self._backend, "read_browser_text_observation", None)
        if not callable(reader):
            observation = TextObservation(
                text="", source="unavailable", target={"kind": "browser_global"},
                fresh=True, ok=False,
                error="platform backend does not expose global browser text observation",
            )
            return BrowserTextSnapshot("", observation, ())
        try:
            observation = reader()
        except Exception as exc:
            observation = TextObservation(
                text="", source="unavailable", target={"kind": "browser_global"},
                fresh=False, ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        values = self._values(observation)
        return BrowserTextSnapshot(observation.text if observation.ok else "", observation, values)

    @staticmethod
    def _values(observation: TextObservation) -> tuple[str, ...]:
        controls = observation.metadata.get("controls", [])
        values: list[str] = []
        if isinstance(controls, list):
            for control in controls:
                if isinstance(control, dict):
                    value = str(control.get("text") or "").strip()
                    if value and value not in values:
                        values.append(value)
        if not values and observation.ok and observation.text.strip():
            values.extend(x.strip() for x in observation.text.splitlines() if x.strip())
        return tuple(values)

    @staticmethod
    def compare(before: BrowserTextSnapshot, after: BrowserTextSnapshot) -> BrowserTextChange:
        if not before.observation.ok or not after.observation.ok:
            return BrowserTextChange(
                changed=None, typed=None, before=before, after=after,
                reason="browser editable text was not independently observable",
            )
        b, a = set(before.values), set(after.values)
        added = tuple(sorted(a - b))
        removed = tuple(sorted(b - a))
        changed = before.values != after.values
        return BrowserTextChange(
            changed=changed,
            typed=bool(added),
            before=before,
            after=after,
            added=added,
            removed=removed,
            reason=("editable browser text changed" if changed else "editable browser text did not change"),
        )

    def detect_change(self) -> BrowserTextChange:
        """Capture two fresh observations and compare editable browser text."""
        before = self.observe()
        after = self.observe()
        return self.compare(before, after)
