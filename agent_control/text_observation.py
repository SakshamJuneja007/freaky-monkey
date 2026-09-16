"""Application-independent semantic text observation.

This module is deliberately a thin extension of DEIMOS' existing observation
layer.  It normalizes text exposed by native Windows UI Automation and browser
semantic observation into one immutable evidence object; it does not execute
actions and it never receives the requested text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .types import DEFAULT_MAX_STALENESS_S


@dataclass(frozen=True)
class TextObservation:
    """One independently acquired piece of application text evidence."""

    text: str
    source: str
    target: dict[str, Any] = field(default_factory=dict)
    fresh: bool = True
    observed_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str | None = None

    def age(self, now: float | None = None) -> float:
        return (time.time() if now is None else now) - self.observed_at

    def is_fresh(self, max_age_s: float = DEFAULT_MAX_STALENESS_S) -> bool:
        return self.ok and self.fresh and self.age() <= max_age_s

    def to_json(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "target": self.target,
            "fresh": self.fresh,
            "observed_at": self.observed_at,
            "age_s": round(self.age(), 4),
            "metadata": self.metadata,
            "ok": self.ok,
            "error": self.error,
        }


class TextReader(Protocol):
    def read(self, target: Any) -> TextObservation | None: ...


class UniversalTextReader:
    """Select the strongest semantic text reader for a target.

    Browser targets stay on BrowserSkill's existing semantic observation path.
    Native application/window targets use the platform backend, whose Windows
    implementation is UI Automation first.  Optional providers are injected so
    application adapters can be added without changing the verifier.
    """

    def __init__(
        self,
        *,
        native_reader: TextReader | None = None,
        browser_reader: TextReader | None = None,
        accessibility_reader: TextReader | None = None,
        application_readers: tuple[TextReader, ...] = (),
    ) -> None:
        self.native_reader = native_reader
        self.browser_reader = browser_reader
        self.accessibility_reader = accessibility_reader
        self.application_readers = application_readers

    def read(self, target: Any) -> TextObservation | None:
        """Freshly read semantic text for one target; never accepts requested text."""
        if self._is_browser_target(target):
            readers = (self.browser_reader, *self.application_readers)
        else:
            readers = (
                self.native_reader,
                self.accessibility_reader,
                *self.application_readers,
            )

        for reader in readers:
            if reader is None:
                continue
            try:
                observation = reader.read(target)
            except Exception:
                observation = None
            if observation is not None and observation.ok:
                return observation
        return None

    @staticmethod
    def _is_browser_target(target: Any) -> bool:
        if isinstance(target, dict):
            kind = str(target.get("kind", "")).casefold()
            return kind == "browser" or "browser" in kind
        return str(getattr(target, "kind", "")).casefold() == "browser"


class CallableTextReader:
    """Small adapter used by tests and future providers."""

    def __init__(self, fn: Callable[[Any], TextObservation | None]) -> None:
        self._fn = fn

    def read(self, target: Any) -> TextObservation | None:
        return self._fn(target)


class BrowserTextReader:
    """Normalize an existing BrowserSkill semantic page observation."""

    def __init__(self, browser: Any) -> None:
        self.browser = browser

    def read(self, target: Any) -> TextObservation | None:
        try:
            text = self.browser.page_text()
        except Exception:
            return None
        if not isinstance(text, str):
            return None
        return TextObservation(
            text=text,
            source="browser_dom",
            target=target if isinstance(target, dict) else {"kind": "browser"},
            fresh=True,
            metadata={"browser": type(self.browser).__name__},
        )
