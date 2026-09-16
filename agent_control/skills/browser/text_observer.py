"""Fresh, read-only browser text/state observation through BrowserSkill.

This module deliberately does not execute browser actions.  It consumes one
fresh BrowserSkill observation and normalizes only the compact, readable state
needed by independent verification.
"""

from __future__ import annotations

from typing import Any

from ...text_observation import TextObservation
from .backend import BrowserObservation


class BrowserTextObserver:
    """Turn one fresh BrowserSkill observation into generic text evidence."""

    _TEXT_KEYS = frozenset({"text", "name", "label", "description", "value", "title"})
    _URL_KEYS = frozenset({"url", "final_url", "finalUrl"})
    _TITLE_KEYS = frozenset({"title", "page_title", "pageTitle"})

    def __init__(self, browser: Any) -> None:
        if browser is None:
            raise ValueError("BrowserTextObserver requires a BrowserSkill backend")
        self.browser = browser

    def observe(self, target: Any | None = None) -> TextObservation:
        """Acquire exactly one fresh BrowserSkill observation."""
        try:
            observe = getattr(self.browser, "observe", None)
            if callable(observe):
                result = observe()
            else:
                # Compatibility for lightweight browser backends that predate
                # the explicit observe primitive. These are still fresh reads
                # from the backend; executor results are never consulted.
                result = {}
                current_url = getattr(self.browser, "current_url", None)
                page_title = getattr(self.browser, "page_title", None)
                page_text = getattr(self.browser, "page_text", None)
                if callable(current_url):
                    result["url"] = current_url()
                if callable(page_title):
                    result["title"] = page_title()
                if callable(page_text):
                    result["text"] = page_text()
            browser_obs = getattr(self.browser, "_last_observation", None)
            if isinstance(browser_obs, BrowserObservation):
                url = browser_obs.url
                text = browser_obs.text
                raw = browser_obs.raw
                generation = browser_obs.generation
            else:
                raw = result
                url = self._first_string(raw, self._URL_KEYS)
                text = self._readable_text(raw)
                generation = getattr(self.browser, "_generation", None)

            title = self._first_string(raw, self._TITLE_KEYS)
            if not title and isinstance(browser_obs, BrowserObservation):
                title = self._first_string(browser_obs.raw, self._TITLE_KEYS)

            # BrowserObservation.text already contains normalized semantic
            # element names/values.  For generic/fake backends, build the same
            # compact representation from readable fields only.
            if not text:
                text = self._readable_text(raw)

            metadata = {
                "url": url,
                "title": title,
                "text_available": bool(text.strip()),
                "text_length": len(text),
                "source": "browserskill_observation",
            }
            if generation is not None:
                metadata["generation"] = generation

            return TextObservation(
                text=text,
                source="browserskill_observation",
                target=target if isinstance(target, dict) else {"kind": "browser"},
                fresh=True,
                metadata=metadata,
                ok=True,
            )
        except Exception as exc:
            return TextObservation(
                text="",
                source="browserskill_observation",
                target=target if isinstance(target, dict) else {"kind": "browser"},
                fresh=False,
                metadata={"text_available": False},
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def read(self, target: Any | None = None) -> TextObservation:
        """UniversalTextReader-compatible alias."""
        return self.observe(target)

    @classmethod
    def _readable_text(cls, value: Any) -> str:
        values: list[str] = []
        seen: set[str] = set()

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                for key, item in node.items():
                    key_s = str(key)
                    if key_s in cls._TEXT_KEYS and isinstance(item, str):
                        item = item.strip()
                        if item and item not in seen:
                            seen.add(item)
                            values.append(item)
                    if isinstance(item, (dict, list, tuple)):
                        walk(item)
            elif isinstance(node, (list, tuple)):
                for item in node:
                    walk(item)
            elif isinstance(node, str):
                # Raw semantic observation text can contain @eN refs; retain
                # it as readable evidence rather than inventing a DOM model.
                item = node.strip()
                if item and item not in seen:
                    seen.add(item)
                    values.append(item)

        walk(value)
        return "\n".join(values)

    @staticmethod
    def _first_string(value: Any, keys: frozenset[str]) -> str:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key) in keys and isinstance(item, str) and item.strip():
                    return item.strip()
            for item in value.values():
                found = BrowserTextObserver._first_string(item, keys)
                if found:
                    return found
        elif isinstance(value, (list, tuple)):
            for item in value:
                found = BrowserTextObserver._first_string(item, keys)
                if found:
                    return found
        return ""
