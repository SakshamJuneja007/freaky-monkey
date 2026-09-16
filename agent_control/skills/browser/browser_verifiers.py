from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .backend import canonical_url
from .text_observer import BrowserTextObserver


class BrowserVerifierBackend(Protocol):
    def current_url(self) -> str: ...
    def page_title(self) -> str: ...
    def page_text(self) -> str: ...
    def list_tabs(self, scope: str = "all") -> Any: ...
    def observe(self) -> Any: ...
    def wait_for_playback(self, *, timeout_s: float = 5.0) -> None: ...


@dataclass(frozen=True)
class BrowserVerificationResult:
    ok: bool
    detail: str = ""
    status: str = "PASS"


class BrowserVerifier:
    """Independent browser-state verifier.

    The executor result is deliberately not treated as evidence.  Verification
    always re-observes the browser.  ``UNKNOWN`` is a first-class outcome and
    is never converted into PASS.
    """

    def __init__(self, backend: BrowserVerifierBackend) -> None:
        if backend is None:
            raise ValueError("BrowserVerifier requires a backend")
        self._backend = backend
        self._text_observer = BrowserTextObserver(backend)

    def verify(self, action: Any, result: Any) -> BrowserVerificationResult:
        from .actions import BrowserAction, BrowserActionKind

        if not isinstance(action, BrowserAction):
            raise TypeError("BrowserVerifier expects BrowserAction")
        p = action.params

        # Explicit postconditions are authoritative and are checked from the
        # browser, regardless of whether the executor claimed success.
        if "expected_url" in p:
            return self.verify_url(str(p["expected_url"]))
        if "expected_url_contains" in p:
            return self.verify_url_contains(str(p["expected_url_contains"]))
        if "expected_title" in p:
            return self.verify_title(str(p["expected_title"]))
        if "expected_text" in p:
            return self.verify_text(str(p["expected_text"]))

        if action.kind is BrowserActionKind.PLAY_SONG:
            return self.verify_playback(str(p.get("query", "")))

        if p.get("verify_youtube_playback") is True:
            target = p.get("target")
            if target is not None:
                name = str(getattr(target, "name", ""))
                raw = getattr(target, "raw", None)
                hrefs = []

                def collect(value: Any) -> None:
                    if isinstance(value, dict):
                        for key, item in value.items():
                            if str(key).casefold() in {"href", "url", "link", "target_url", "targeturl"} and isinstance(item, str):
                                hrefs.append(item)
                            elif isinstance(item, (dict, list, tuple)):
                                collect(item)
                    elif isinstance(value, (list, tuple)):
                        for item in value:
                            collect(item)

                collect(raw)
                if "short" in name.casefold() or any("/shorts/" in h.casefold() or "youtube.com/shorts" in h.casefold() for h in hrefs):
                    return BrowserVerificationResult(False, "selected YouTube result was a Short", "FAIL")
            return self.verify_playback("")

        if action.kind is BrowserActionKind.OPEN_URL:
            expected = str(p.get("url", ""))
            if expected:
                return self.verify_url(expected)

        if action.kind is BrowserActionKind.GET_CURRENT_PAGE:
            try:
                url = self._backend.current_url()
                title = self._backend.page_title()
            except Exception as exc:
                return BrowserVerificationResult(False, f"independent observation failed: {exc}", "FAIL")
            if url and url != "about:blank":
                return BrowserVerificationResult(True, f"active page observed: {url!r} title={title!r}")
            return BrowserVerificationResult(False, "browser is blank", "FAIL")

        # Primitive actions without a declared postcondition cannot be
        # meaningfully proven from a generic read-only observation.
        return BrowserVerificationResult(
            False,
            "no independent postcondition supplied for this browser action",
            "UNKNOWN",
        )

    def verify_playback(self, query: str = "") -> BrowserVerificationResult:
        """Independently prove the requested media is actively playing."""
        try:
            url = self._backend.current_url()
            if "youtube.com/watch" not in url.casefold():
                return BrowserVerificationResult(False, f"not on a YouTube watch page: {url!r}", "FAIL")
            if query.strip():
                title = self._backend.page_title()
                text = self._backend.page_text()
                haystack = f"{title}\n{text}".casefold()
                if query.casefold() not in haystack:
                    return BrowserVerificationResult(False, f"watch page does not independently contain requested query {query!r}", "FAIL")
            self._backend.wait_for_playback(timeout_s=2.0)
            return BrowserVerificationResult(True, "fresh browser observation confirmed the requested video is actively playing", "PASS")
        except Exception as exc:
            # A failed proof is not necessarily a browser-action failure: the
            # environment may simply not expose sufficient playback state.
            try:
                url = self._backend.current_url()
                if "youtube.com/watch" not in url.casefold():
                    return BrowserVerificationResult(False, f"not on a YouTube watch page: {url!r}", "FAIL")
            except Exception as url_exc:
                return BrowserVerificationResult(False, f"could not independently inspect playback: {url_exc}", "FAIL")
            return BrowserVerificationResult(False, f"playback state is unknown: {exc}", "UNKNOWN")

    def _fresh_observation(self) -> Any:
        """Acquire one fresh browser observation for an independent check."""
        return self._text_observer.observe({"kind": "browser", "purpose": "verification"})

    def verify_url(self, expected_url: str) -> BrowserVerificationResult:
        observation = self._fresh_observation()
        if not observation.ok:
            return BrowserVerificationResult(False, f"browser observation unavailable: {observation.error}", "UNKNOWN")
        actual = canonical_url(str(observation.metadata.get("url") or ""))
        expected = canonical_url(expected_url)
        if actual == expected:
            return BrowserVerificationResult(True, f"URL matched {actual!r}", "PASS")
        from urllib.parse import urlsplit
        ep, ap = urlsplit(expected), urlsplit(actual)
        if ep.netloc == ap.netloc and ep.path == "/" and not ep.query:
            return BrowserVerificationResult(True, f"requested host reached: {ap.netloc!r}", "PASS")
        return BrowserVerificationResult(False, f"URL mismatch: expected {expected!r}, observed {actual!r}", "FAIL")

    def verify_url_contains(self, fragment: str) -> BrowserVerificationResult:
        observation = self._fresh_observation()
        if not observation.ok:
            return BrowserVerificationResult(False, f"browser observation unavailable: {observation.error}", "UNKNOWN")
        actual = str(observation.metadata.get("url") or "")
        if fragment.casefold() in actual.casefold():
            return BrowserVerificationResult(True, f"URL contains {fragment!r}", "PASS")
        return BrowserVerificationResult(False, f"URL does not contain {fragment!r}; observed {actual!r}", "FAIL")

    def verify_title(self, expected: str) -> BrowserVerificationResult:
        observation = self._fresh_observation()
        if not observation.ok:
            return BrowserVerificationResult(False, f"browser observation unavailable: {observation.error}", "UNKNOWN")
        actual = str(observation.metadata.get("title") or "")
        if expected.casefold() in actual.casefold():
            return BrowserVerificationResult(True, f"title contains {expected!r}", "PASS")
        return BrowserVerificationResult(False, f"title {actual!r} does not contain {expected!r}", "FAIL")

    def verify_text(self, expected: str) -> BrowserVerificationResult:
        observation = self._fresh_observation()
        if not observation.ok:
            return BrowserVerificationResult(False, f"browser observation unavailable: {observation.error}", "UNKNOWN")
        actual = observation.text
        if expected.casefold() in actual.casefold():
            return BrowserVerificationResult(True, f"page contains {expected!r}", "PASS")
        return BrowserVerificationResult(False, f"page does not contain {expected!r}", "FAIL")
